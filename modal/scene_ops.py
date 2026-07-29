"""dc-scene-ops — the GLB half of the dataset pipeline, running next to the volumes.

Deliberately its own app rather than part of the Trellis farm: the farm is a shared,
GPU-scheduled service whose campaigns cannot overlap, and this is a small always-idle CPU
service that several pipeline runs may call at once. Coupling them would mean redeploying
the farm to change a voxel setting, and a crash here would take the farm's endpoint with it.

    modal deploy modal/scene_ops.py

Everything it does is volume-to-volume, which is the point. A sample's meshes come off the
farm's output volume, get voxelized where they already live, and land posed on the scene
volume — roughly 13 MB a sample that no longer has to travel to a laptop and back.

The real work is `pipeline/ops.mjs`, invoked per request. It is Node because `pipeline/glb.mjs`
and `pipeline/voxelize.mjs` are, and the voxel grids decide what the placement model sees:
a Python reimplementation could shift them in ways nothing would catch.

`dc-pipeline` runs those same commands in-process, since it has the volumes mounted itself.
This app is what a pipeline driven from a laptop talks to, and what serves the viewer.
"""

from __future__ import annotations  # the local CLI may be older than the container's 3.12

import json
import subprocess
import threading
import time
from pathlib import Path

import modal

REPO = Path(__file__).parent.parent

app = modal.App("dc-scene-ops")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "ca-certificates")
    .run_commands(
        "curl -fsSL https://deb.nodesource.com/setup_22.x | bash -",
        "apt-get install -y --no-install-recommends nodejs",
        "mkdir -p /app && cd /app && npm init -y && npm install three@0.185.1 sharp@0.35.3 fzstd@0.1.1",
    )
    .pip_install("fastapi[standard]==0.115.6")
    # Added at runtime rather than copied, so editing the pipeline does not rebuild the image.
    .add_local_dir(REPO / "pipeline", "/app/pipeline")
    .add_local_dir(REPO / "modal", "/app/modal")
)

farm_out = modal.Volume.from_name("t2farm-output-v2")
farm_in = modal.Volume.from_name("t2farm-input-v2")
scene = modal.Volume.from_name("trellis-scene-vol-v2")

VOLUMES = {"/farm-out": farm_out, "/farm-in": farm_in, "/scene": scene}

# Kept in step with the defaults in ops.mjs; only the file route needs it on this side.
PUBLISH_PREFIX = "datasets/raw/stage1"

# Exit code `ops.mjs` uses for a file that is not on the volume. Kept in step with the
# constant there.
MISSING_INPUT = 17

# What each command touches, which is what decides when to reload and when to commit.
READS_FARM = {"assets", "collect"}
# These read the scene volume once a run, so syncing up front costs nothing at that rate.
READS_SCENE = {"stage", "published"}
# These read it once a sample. The meshes they read are written by `collect` and stop changing
# when stage 3 ends, so reloading before each of tens of thousands of calls re-syncs a snapshot
# that is already right — a third of a second every time, growing with the file count of the
# whole volume. They reload on a missing input instead, which is the only symptom a snapshot
# older than the write has.
RELOADS_ON_MISS = {"voxelize", "bake"}
WRITES_SCENE = {"collect", "bake", "publish"}
# Which of those has someone waiting on the write. Collect hands its meshes to the placement
# stage, and publish is what marks a sample finished, so both push before replying. Nothing
# reads a posed mesh until stage 7, so bake rides a timer along with every bake since the last
# one — a commit costs three quarters of a second before it moves a byte, and at four files a
# sample that fixed cost is most of what baking spends.
COMMITS_ON_REPLY = {"collect", "publish"}
# Staging is the one thing that writes the farm's side. It has to be committed before the
# campaign starts, or `/run` walks the input volume and finds nothing there.
WRITES_FARM_IN = {"stage"}

COMMANDS = READS_FARM | READS_SCENE | RELOADS_ON_MISS | WRITES_SCENE | WRITES_FARM_IN

# How long a coalesced write may go unpushed. Bounded rather than left to the container's
# lifetime so an ungraceful death costs a handful of posed meshes, which the next pass rebakes
# from the raw ones it never touches.
COMMIT_EVERY_S = 15

# Volume reload discards nothing only while no sibling request has written without
# committing, so a container runs its operations one at a time. The work is a fifth of a
# second per sample; throughput comes from more containers, not from interleaving here.
_volumes = threading.Lock()
# Whether this container is holding scene writes back, and when it last let go. Both are only
# touched under `_volumes`.
_scene_dirty = False
_scene_pushed = time.monotonic()


def _commit_scene() -> None:
    global _scene_dirty, _scene_pushed
    scene.commit()
    _scene_dirty = False
    _scene_pushed = time.monotonic()


def _reload_scene() -> None:
    """Push before pulling, always: a reload can drop writes this container has not committed
    yet, and holding bakes back means there are usually some waiting."""
    if _scene_dirty:
        _commit_scene()
    scene.reload()


def _node(command: str, payload: dict) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["node", "/app/pipeline/ops.mjs", command],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
    )


def run_op(command: str, payload: dict) -> dict:
    global _scene_dirty

    with _volumes:
        if command in READS_FARM:
            farm_out.reload()
            farm_in.reload()
        if command in READS_SCENE:
            _reload_scene()

        # Marked before the run, not after it: a command that failed halfway has still left
        # whatever it wrote before that, and the reload below must not pull over the top of it.
        if command in WRITES_SCENE:
            _scene_dirty = True

        done = _node(command, payload)
        # One reload and one retry, so a container warmer than the write pays for the gap once
        # instead of on every call. Both commands re-run cleanly: voxelize only reads, and bake
        # rewrites the same posed copies from raw meshes it leaves alone.
        if done.returncode == MISSING_INPUT and command in RELOADS_ON_MISS:
            _reload_scene()
            done = _node(command, payload)

        if done.returncode != 0:
            raise RuntimeError(f"{command} failed: {done.stderr[-2000:]}")

        # A commit pushes everything this container has written, so one made for its own sake
        # carries any held-back bakes with it — as does the next request of any kind once the
        # timer is up, which is what keeps the window closing without a thread to close it.
        if command in COMMITS_ON_REPLY or (_scene_dirty and time.monotonic() - _scene_pushed >= COMMIT_EVERY_S):
            _commit_scene()
        if command in WRITES_FARM_IN:
            farm_in.commit()
        return json.loads(done.stdout)


@app.function(image=image, volumes=VOLUMES, timeout=3600)
@modal.concurrent(max_inputs=8)
@modal.asgi_app()
def web():
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse

    api = FastAPI(title="dc-scene-ops")

    from fastapi.responses import FileResponse

    @api.get("/health")
    def health():
        return {"ok": True, "commands": sorted(COMMANDS)}

    @api.get("/file/{sample}/{name}")
    def file(sample: str, name: str):
        """One published file, so the viewer can show a sample without anyone downloading
        the dataset. `.name` on both parts keeps a crafted path inside the prefix."""
        with _volumes:
            _reload_scene()
            target = Path("/scene") / PUBLISH_PREFIX / Path(sample).name / Path(name).name
            if not target.is_file():
                raise HTTPException(status_code=404, detail=f"no {name} for {sample}")
            return FileResponse(target)

    @api.post("/{command}")
    async def dispatch(command: str, payload: dict | None = None):
        if command not in COMMANDS:
            raise HTTPException(status_code=404, detail=f"unknown command {command}")
        try:
            return JSONResponse(run_op(command, payload or {}))
        except RuntimeError as err:
            raise HTTPException(status_code=500, detail=str(err)) from err

    return api


@app.function(image=image, volumes=VOLUMES, timeout=600)
def op(command: str, payload: dict | None = None) -> dict:
    """The same commands as the endpoint, for calling from Python or `modal run`."""
    return run_op(command, payload or {})


@app.local_entrypoint()
def smoke(run_id: str = ""):
    """`modal run modal/scene_ops.py --run-id dc-xxxx` — check the volumes are readable."""
    print("published:", op.remote("published", {}))
    if run_id:
        print("assets:", op.remote("assets", {"run_id": run_id}))
