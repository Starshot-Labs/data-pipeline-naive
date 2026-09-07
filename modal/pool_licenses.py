"""dc-pool-licenses — join Objaverse's licenses into a pool built without them.

    modal run modal/pool_licenses.py

`objaverse-pool.mjs --no-licenses` skips the join against Objaverse's 160 metadata
shards, and a pool built that way carries no `license` on any row: `seedOf()` then
records no attribution on the samples it seeds, and `POOL_LICENSES` filters against a
field that is not there. The shards are already cached beside the pool on the scene
volume, so repairing an existing `pool.jsonl` costs a read rather than the ~580 MB
download the original join paid for.

Only the uids the pool actually holds are kept while the shards are scanned, so the
join is bounded by the pool's size rather than by Objaverse's. Rows are rewritten
through a temporary name and renamed into place, the same way every other write to
this volume lands.
"""

from __future__ import annotations  # the local CLI may be older than the container's 3.12

import gzip
import json
from collections import Counter
from pathlib import Path

import modal

app = modal.App("dc-pool-licenses")

image = modal.Image.debian_slim(python_version="3.12")

scene = modal.Volume.from_name("trellis-scene-vol-v2")

POOL = Path("/scene/datasets/raw/pool")


@app.function(image=image, volumes={"/scene": scene}, timeout=3600, memory=4096)
def backfill(pool: str = "pool.jsonl") -> dict:
    scene.reload()
    source = POOL / pool
    if not source.is_file():
        raise FileNotFoundError(f"{source} — nothing to repair")

    wanted = set()
    with source.open(encoding="utf-8") as rows:
        for line in rows:
            if line.strip():
                wanted.add(json.loads(line)["uid"])
    print(f"{len(wanted)} uid(s) in {pool}", flush=True)

    annotations: dict[str, dict] = {}
    shards = sorted((POOL / "cache").glob("metadata-000-*.json.gz"))
    if not shards:
        raise FileNotFoundError(f"{POOL / 'cache'} holds no metadata shards")
    for n, shard in enumerate(shards, 1):
        with gzip.open(shard, "rt", encoding="utf-8") as handle:
            for uid, entry in json.load(handle).items():
                if uid in wanted:
                    annotations[uid] = {"name": entry.get("name"), "license": entry.get("license")}
        if n % 40 == 0:
            print(f"  … {n}/{len(shards)} shard(s), {len(annotations)} matched", flush=True)
    print(f"{len(annotations)}/{len(wanted)} uid(s) matched across {len(shards)} shard(s)", flush=True)

    licenses: Counter[str] = Counter()
    temporary = source.with_name(f"{source.name}.tmp")
    with source.open(encoding="utf-8") as rows, temporary.open("w", encoding="utf-8") as out:
        for line in rows:
            if not line.strip():
                continue
            row = json.loads(line)
            # Whatever the shards know wins; a row that was already licensed keeps its own.
            found = annotations.get(row["uid"], {})
            licenses[found.get("license") or row.get("license") or "unknown"] += 1
            out.write(json.dumps({**row, **found}) + "\n")

    temporary.replace(source)
    scene.commit()

    total = sum(licenses.values())
    print("licenses: " + "  ".join(f"{name}={count}" for name, count in licenses.most_common()), flush=True)
    return {"rows": total, "licensed": total - licenses["unknown"]}


@app.local_entrypoint()
def main(pool: str = "pool.jsonl"):
    print(backfill.remote(pool))
