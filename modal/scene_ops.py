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

# What each command touches, which is what decides when to reload and when to commit.
READS_FARM = {"assets", "collect"}
READS_SCENE = {"stage", "fetch", "voxelize", "refine", "pose", "bake", "drape", "published"}
WRITES_SCENE = {"collect", "fetch", "bake", "drape", "publish"}
# Staging is the one thing that writes the farm's side. It has to be committed before the
# campaign starts, or `/run` walks the input volume and finds nothing there.
WRITES_FARM_IN = {"stage"}

COMMANDS = READS_FARM | READS_SCENE | WRITES_SCENE | WRITES_FARM_IN

# Volume reload discards nothing only while no sibling request has written without
# committing, so a container runs its operations one at a time. The work is a fifth of a
# second per sample; throughput comes from more containers, not from interleaving here.
_volumes = threading.Lock()


def run_op(command: str, payload: dict) -> dict:
    with _volumes:
        if command in READS_FARM:
            farm_out.reload()
            farm_in.reload()
        if command in READS_SCENE:
            scene.reload()

        done = subprocess.run(
            ["node", "/app/pipeline/ops.mjs", command],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
        )
        if done.returncode != 0:
            raise RuntimeError(f"{command} failed: {done.stderr[-2000:]}")

        if command in WRITES_SCENE:
            scene.commit()
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
            scene.reload()
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
