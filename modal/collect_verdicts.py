"""Collect review decisions into one flat JSON array on the scene volume.

    modal run modal/collect_verdicts.py

Writes `datasets/raw/review/verdicts.json` in `trellis-scene-vol-v2`.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal

app = modal.App("dc-collect-review-verdicts")
image = modal.Image.debian_slim(python_version="3.12")
scene = modal.Volume.from_name("trellis-scene-vol-v2")

REVIEW = Path("/scene/datasets/raw/review")
OUTPUT = REVIEW / "verdicts.json"
WORKERS = 64


def read_verdict(folder: Path) -> dict | None:
    try:
        decision = json.loads((folder / "decision.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None

    sample_id = decision.get("id")
    verdict = decision.get("verdict")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError(f"{folder / 'decision.json'} has no valid id")
    if sample_id != folder.name:
        raise ValueError(f"{folder / 'decision.json'} has id {sample_id!r}")
    if not isinstance(verdict, bool):
        raise ValueError(f"{folder / 'decision.json'} has no boolean verdict")
    return {"id": sample_id, "verdict": verdict}


@app.function(image=image, volumes={"/scene": scene}, timeout=3600)
def collect() -> dict:
    scene.reload()
    if not REVIEW.is_dir():
        raise FileNotFoundError(f"{REVIEW} does not exist")

    folders = sorted((entry for entry in REVIEW.iterdir() if entry.is_dir()), key=lambda entry: entry.name)
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        decisions = list(pool.map(read_verdict, folders))

    missing = [folder.name for folder, decision in zip(folders, decisions) if decision is None]
    if missing:
        preview = ", ".join(missing[:5])
        raise RuntimeError(f"{len(missing)} review folder(s) have no decision.json: {preview}")

    rows = [decision for decision in decisions if decision is not None]
    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_text(f"{json.dumps(rows, indent=2)}\n", encoding="utf-8")
    temporary.replace(OUTPUT)
    scene.commit()

    print(f"wrote {len(rows)} verdict(s) to {OUTPUT}", flush=True)
    return {"output": str(OUTPUT), "verdicts": len(rows)}


@app.local_entrypoint()
def main():
    print(collect.remote())
