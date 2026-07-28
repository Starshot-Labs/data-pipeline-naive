"""dc-pipeline — the whole dataset pipeline, run on Modal.

    modal run modal/pipeline.py --pairs 500            watch it
    modal run --detach modal/pipeline.py --pairs 500    let it outlive your terminal

One container runs all seven stages, with `generated/` living on the scene volume instead of
a laptop. Nothing crosses a home connection: the images are made here and staged onto the
farm's input volume from here, the meshes never leave Modal at all, and the only traffic is
the model calls each stage makes.

Stages run as separate `node` invocations on purpose. Modal volumes are snapshots — a
container has to `commit()` for its writes to be visible elsewhere and `reload()` to see
anyone else's, and only Python can call either. Stage boundaries are exactly where ownership
of the volume changes hands, so that is where the syncing goes. Each stage is independently
resumable, so re-invoking after a failure costs only the gaps.
"""

from __future__ import annotations  # the local CLI may be older than the container's 3.12

import contextlib
import os
import subprocess
import threading
from pathlib import Path

import modal

REPO = Path(__file__).parent.parent

app = modal.App("dc-pipeline")

# Deliberately identical to the image in scene_ops.py: Modal keys its build cache on the
# definition, so two matching specs share one built image rather than paying twice.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "ca-certificates")
    .run_commands(
        "curl -fsSL https://deb.nodesource.com/setup_22.x | bash -",
        "apt-get install -y --no-install-recommends nodejs",
        "mkdir -p /app && cd /app && npm init -y && npm install three@0.185.1 sharp@0.35.3 fzstd@0.1.1",
    )
    .add_local_dir(REPO / "pipeline", "/app/pipeline")
)

scene = modal.Volume.from_name("trellis-scene-vol-v2")

# Where `generated/` lives — the small half of every sample, alongside the work and published
# prefixes the rest of the pipeline writes.
STAGING_PREFIX = "datasets/raw/staging"
# Long enough that a stage never dies mid-flight; a stalled campaign has its own limits.
TIMEOUT_S = 24 * 60 * 60
# A stage can run for half an hour, and a container that died before its boundary would lose
# all of it. Safe mid-stage because every file the pipeline writes lands by atomic rename.
COMMIT_EVERY_S = 60

_volume = threading.Lock()


def sync(reload: bool = False) -> None:
    """Push this container's work, then optionally pick up anyone else's.

    Always in that order. `dc-scene-ops` writes the work prefix during stage 3, so the
    pipeline has to reload to see it — but reloading before committing would discard whatever
    it had written and not yet pushed.
    """
    with _volume:
        scene.commit()
        if reload:
            scene.reload()


@contextlib.contextmanager
def committing_every(seconds: int):
    stop = threading.Event()

    def loop() -> None:
        while not stop.wait(seconds):
            sync()

    worker = threading.Thread(target=loop, daemon=True)
    worker.start()
    try:
        yield
    finally:
        stop.set()
        worker.join(timeout=10)


def node_env() -> dict[str, str]:
    generated = f"/scene/{STAGING_PREFIX}"
    return {
        **os.environ,  # carries OPENROUTER_API_KEY and GOOGLE_API_KEY in from the secret
        "GENERATED_DIR": generated,
        "SCENE_DIR": "/scene",
        # `/scene` is mounted here, so the operations that only touch it run in-process.
        # Anything reading a farm volume still goes to dc-scene-ops, which can reload.
        "SCENE_OPS_DIRECT": "1",
        # On the volume rather than in the container, so an interrupted campaign is still
        # attachable after the container that started it is gone.
        "TRELLIS_STATE": f"{generated}/.trellis-campaign.json",
    }


def run_stage(label: str, script: str, *args: str) -> int:
    print(f"\n=== {label}", flush=True)
    done = subprocess.run(["node", f"/app/pipeline/{script}", *args], env=node_env())
    sync(reload=True)
    return done.returncode


@app.function(
    image=image,
    volumes={"/scene": scene},
    secrets=[modal.Secret.from_name("dc-pipeline-env")],
    timeout=TIMEOUT_S,
)
def run(pairs: int = 0) -> dict:
    Path(f"/scene/{STAGING_PREFIX}").mkdir(parents=True, exist_ok=True)
    sync()

    stages = [
        ("1-2 · invent and render", "build-images.mjs", [f"--pairs={pairs}"]),
        ("3 · meshes", "build-meshes.mjs", []),
        ("4-6 · place", "run.mjs", []),
        ("7 · publish", "upload.mjs", []),
    ]

    # A non-zero stage is logged rather than fatal: every one of them skips work already done,
    # so the stages after it still have something useful to do with what did land.
    failed = []
    with committing_every(COMMIT_EVERY_S):
        for label, script, args in stages:
            if run_stage(label, script, *args) != 0:
                failed.append(label)
                print(f"   ✗ {label} reported a failure", flush=True)

    return {"pairs": pairs, "failed_stages": failed}


@app.local_entrypoint()
def main(pairs: int = 0):
    print(run.remote(pairs))
