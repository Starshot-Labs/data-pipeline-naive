"""dc-archive — move a finished generation aside so the next one starts from nothing.

    modal run modal/archive.py --label pre-render-fix
    modal run modal/archive.py                        labels it with today's date

Everything one campaign produced — the staged samples, the raw and published meshes, and the
placement logs — moves under `datasets/archive/<label>/`. The moves happen on the volume
itself, so hundreds of gigabytes cost a rename rather than a copy, and nothing is deleted:
an archived run stays readable and can be moved back.

Two things stay behind on purpose. The Objaverse seed pool is an input rather than output,
and rebuilding it means paying for a tagging run again. The cached GLBs are byte-identical
downloads from Hugging Face that no change to this repo can invalidate — only the renders
sitting beside them go stale, so those are the one part of the cache that gets archived out.
"""

from __future__ import annotations  # the local CLI may be older than the container's 3.12

import json
import shutil
from datetime import date
from pathlib import Path

import modal

app = modal.App("dc-archive")

image = modal.Image.debian_slim(python_version="3.12")

scene = modal.Volume.from_name("trellis-scene-vol-v2")

RAW = Path("/scene/datasets/raw")
ARCHIVE = Path("/scene/datasets/archive")

# Whole prefixes: everything a campaign writes.
PREFIXES = ("staging", "stage1", "stage1-work", "placement-logs")
# Kept in place apart from its renders, which a renderer change is exactly what invalidates.
CACHE = "objaverse-cache"

ROLES = ("anchor", "placed")


@app.function(image=image, volumes={"/scene": scene}, timeout=3600)
def archive(label: str = "") -> dict:
    scene.reload()
    target = ARCHIVE / (label or date.today().isoformat())
    if target.exists():
        raise FileExistsError(f"{target} already exists — pick another --label")
    target.mkdir(parents=True)

    moved: dict[str, int] = {}
    for name in PREFIXES:
        source = RAW / name
        if not source.is_dir():
            continue
        moved[name] = sum(1 for _ in source.iterdir())
        shutil.move(str(source), str(target / name))

    cache = RAW / CACHE
    if cache.is_dir():
        renders = target / f"{CACHE}-renders"
        renders.mkdir()
        pngs = list(cache.glob("*.png"))
        for png in pngs:
            shutil.move(str(png), str(renders / png.name))
        moved[f"{CACHE}/*.png"] = len(pngs)
        moved[f"{CACHE}/*.glb (kept)"] = sum(1 for _ in cache.glob("*.glb"))

    scene.commit()

    for name, count in moved.items():
        print(f"  {name:<26} {count}", flush=True)
    print(f"\narchived into {target}", flush=True)
    print(f"still in place: {RAW / 'pool'}, {cache} minus its renders", flush=True)
    return {"archive": str(target), "moved": moved}


@app.function(image=image, volumes={"/scene": scene}, timeout=3600)
def restore(label: str, prefixes: str = "staging") -> dict:
    """Put archived prefixes back under `datasets/raw`, merging into whatever is there.

        modal run modal/archive.py::restore --label pre-render-fix

    Symmetric with `archive`, and needed for the case that motivates it: scenes are expensive
    to invent and say nothing about how their assets get rendered, so a corpus archived
    because its *images* went stale is worth putting back rather than paying to invent again.
    A sample that already exists under `datasets/raw` is left in the archive rather than
    overwritten, so restoring can never clobber newer work.
    """
    scene.reload()
    source_root = ARCHIVE / label
    if not source_root.is_dir():
        raise FileNotFoundError(f"{source_root} does not exist")

    restored: dict[str, int] = {}
    for name in (part.strip() for part in prefixes.split(",") if part.strip()):
        source = source_root / name
        if not source.is_dir():
            continue
        target = RAW / name
        target.mkdir(parents=True, exist_ok=True)

        count = 0
        kept = 0
        for entry in source.iterdir():
            if (target / entry.name).exists():
                kept += 1
                continue
            shutil.move(str(entry), str(target / entry.name))
            count += 1
        restored[name] = count
        if kept:
            print(f"  {name}: left {kept} already present under datasets/raw", flush=True)
        if not any(source.iterdir()):
            source.rmdir()

    scene.commit()
    for name, count in restored.items():
        print(f"  {name:<26} {count} restored", flush=True)
    return {"restored": restored}


@app.function(image=image, volumes={"/scene": scene}, timeout=3600)
def prune(label: str) -> dict:
    """Move samples out of staging that a fresh campaign cannot use.

        modal run modal/archive.py::prune --label pre-render-fix

    Restoring scenes without their meshes leaves two kinds of sample behind. One records a
    `mesh` whose file went to the archive with `stage1-work`: fetch skips it as done and
    placement then fails on a folder that is not there. The other was seeded before uids were
    mandatory and can never be fetched at all. Both go to the archive beside the meshes they
    belong with, leaving staging holding only scenes stage 2 can actually act on.
    """
    scene.reload()
    staging = RAW / "staging"
    target = ARCHIVE / label / "staging-processed"
    target.mkdir(parents=True, exist_ok=True)

    moved = {"already processed": 0, "no objaverse uid": 0}
    kept = 0
    for folder in sorted(staging.iterdir()):
        try:
            data = json.loads((folder / "metadata.json").read_text())
        except (OSError, ValueError):
            continue
        roles = [data.get(role) or {} for role in ROLES]

        if any(role.get("mesh") for role in roles):
            reason = "already processed"
        elif not all((role.get("objaverse") or {}).get("uid") for role in roles):
            reason = "no objaverse uid"
        else:
            kept += 1
            continue

        shutil.move(str(folder), str(target / folder.name))
        moved[reason] += 1

    scene.commit()
    for reason, count in moved.items():
        print(f"  moved {count:>5}  ({reason})", flush=True)
    print(f"\n{kept} fresh scene(s) left in {staging}", flush=True)
    return {"moved": moved, "kept": kept}


@app.local_entrypoint()
def main(label: str = ""):
    print(archive.remote(label))
