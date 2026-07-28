"""dc-selftest — proves the Modal pipeline works without letting it near the farm.

    modal run modal/selftest.py --pairs 2

Runs the real stages in the real image against throwaway prefixes: OpenRouter invents,
Google renders, images are staged onto the farm's input volume, dummy meshes stand in for a
campaign, then voxelize, place, bake and publish all run for real. Roughly $0.60 for two
pairs, almost all of it the image and placement calls.

**It cannot start a campaign.** `selftest.mjs` never imports `generateMeshes`, this file
points `TRELLIS_BASE_URL` at a discard port so no farm request could succeed anyway, and the
one thing it does touch on the farm's side is writing images to the input volume — which is
inert until `POST /run`, and nothing here calls that.

Steps are separate `node` invocations with a volume sync between, exactly as in pipeline.py,
because that boundary is half of what needs testing: `dc-scene-ops` can only read what the
pipeline container has committed.
"""

from __future__ import annotations  # the local CLI may be older than the container's 3.12

import os
import subprocess
import threading
from pathlib import Path

import modal

REPO = Path(__file__).parent.parent

app = modal.App("dc-selftest")

# The production image, plus this directory so the test script rides along. The layers below
# are shared with dc-pipeline, so what runs here is the environment that runs for real.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "ca-certificates")
    .run_commands(
        "curl -fsSL https://deb.nodesource.com/setup_22.x | bash -",
        "apt-get install -y --no-install-recommends nodejs",
        "mkdir -p /app && cd /app && npm init -y && npm install three@0.185.1 sharp@0.35.3 fzstd@0.1.1",
    )
    .add_local_dir(REPO / "pipeline", "/app/pipeline")
    .add_local_dir(REPO / "modal", "/app/modal")
)

scene = modal.Volume.from_name("trellis-scene-vol-v2")

# Its own prefixes, so nothing the test makes can land in the dataset. The meshes it uses do
# not match the images beside them, which would be corrupt data anywhere real.
STAGING_PREFIX = "datasets/raw/selftest-staging"
WORK_PREFIX = "datasets/raw/selftest-work"
PUBLISH_PREFIX = "datasets/raw/selftest"

# Discard port: connections are refused immediately, so a farm call fails instead of running.
UNREACHABLE_FARM = "http://127.0.0.1:9"

_volume = threading.Lock()


def sync() -> None:
    with _volume:
        scene.commit()
        scene.reload()


def node_env() -> dict[str, str]:
    return {
        **os.environ,  # OPENROUTER_API_KEY and GOOGLE_API_KEY arrive from the secret
        "GENERATED_DIR": f"/scene/{STAGING_PREFIX}",
        "SCENE_DIR": "/scene",
        "SCENE_WORK_PREFIX": WORK_PREFIX,
        "SCENE_PUBLISH_PREFIX": PUBLISH_PREFIX,
        "SCENE_OPS_DIRECT": "1",
        "TRELLIS_BASE_URL": UNREACHABLE_FARM,
        "TRELLIS_STATE": f"/scene/{STAGING_PREFIX}/.trellis-campaign.json",
    }


def step(name: str, *args: str) -> int:
    done = subprocess.run(["node", "/app/modal/selftest.mjs", name, *args], env=node_env())
    sync()
    return done.returncode


@app.function(
    image=image,
    volumes={"/scene": scene},
    secrets=[modal.Secret.from_name("dc-pipeline-env")],
    timeout=3600,
)
def selftest(pairs: int = 2, keep: bool = False) -> dict:
    # Start from nothing, so leftovers from an earlier failed run cannot skew the counts.
    step("clean")
    Path(f"/scene/{STAGING_PREFIX}").mkdir(parents=True, exist_ok=True)
    sync()

    failed = []
    for name, args in [
        ("images", [str(pairs)]),
        ("stage", []),
        ("inject", []),
        ("place", []),
        ("publish", []),
        ("verify", []),
    ]:
        if step(name, *args) != 0:
            failed.append(name)
            print(f"  !! {name} failed — stopping so the state can be inspected", flush=True)
            break

    if failed and keep:
        print("  leaving the selftest prefixes in place for inspection", flush=True)
    else:
        step("clean")

    return {"pairs": pairs, "failed_steps": failed, "passed": not failed}


@app.local_entrypoint()
def main(pairs: int = 2, keep: bool = False):
    result = selftest.remote(pairs, keep)
    print(result)
    if not result["passed"]:
        raise SystemExit(1)
