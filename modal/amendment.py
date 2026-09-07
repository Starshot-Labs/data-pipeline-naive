"""dc-amendment — the placement phrase in four progressively shortened forms.

    modal run modal/amendment.py --limit 30 --dry     look before committing to the corpus
    modal run --detach modal/amendment.py             every staged sample

Both generation paths now run this themselves, so it exists for corpora written before that
was true. It writes the forms into each staged sample — `placement` becomes the four,
`placement_original` keeps the first, placement.txt mirrors them — and keeps a copy in the
amendment prefix. Samples that already carry their forms are skipped, and an amendment file
already on disk is adopted rather than re-derived, so re-running is cheap and idempotent.
See pipeline/placement-variants.mjs for what the four lines are and why the variants are
checked to be deletions rather than rewrites.

One container is enough: there are no meshes here, no downloads and no physics — just a
concurrent read of the corpus, a few hundred batched model calls, and small text writes.
"""

from __future__ import annotations  # the local CLI may be older than the container's 3.12

import os
import subprocess
from pathlib import Path

import modal

REPO = Path(__file__).parent.parent

app = modal.App("dc-amendment")

# Deliberately identical to the image in pipeline.py, so the built layers are shared.
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

STAGING_PREFIX = "datasets/raw/staging"
AMENDMENT_PREFIX = "datasets/raw/amendment"
TIMEOUT_S = 6 * 60 * 60


@app.function(
    image=image,
    volumes={"/scene": scene},
    secrets=[modal.Secret.from_name("dc-pipeline-env")],
    timeout=TIMEOUT_S,
)
def amend(limit: int = 0, dry: bool = False, force: bool = False) -> dict:
    scene.reload()
    if not dry:
        Path(f"/scene/{AMENDMENT_PREFIX}").mkdir(parents=True, exist_ok=True)

    argv = ["node", "/app/pipeline/placement-variants.mjs"]
    if limit:
        argv.append(f"--limit={limit}")
    if dry:
        argv.append("--dry")
    if force:
        argv.append("--force")

    done = subprocess.run(
        argv,
        env={
            **os.environ,  # carries OPENROUTER_API_KEY in from the secret
            "SCENE_DIR": "/scene",
            # Staging, because that is what publishing copies from — writing the forms into
            # the published folder alone is what left them stranded the first time.
            "GENERATED_DIR": f"/scene/{STAGING_PREFIX}",
            "SCENE_AMENDMENT_PREFIX": AMENDMENT_PREFIX,
            # Node runs `fs.promises` on libuv's pool, four threads by default, so asking for
            # sixty-four concurrent reads of a network volume otherwise buys sixteen-deep
            # queueing rather than concurrency.
            "UV_THREADPOOL_SIZE": "64",
        },
    )
    if not dry:
        scene.commit()
    return {"dry": dry, "limit": limit, "failed": done.returncode != 0}


@app.local_entrypoint()
def main(limit: int = 0, dry: bool = False, force: bool = False):
    print(amend.remote(limit, dry, force))
