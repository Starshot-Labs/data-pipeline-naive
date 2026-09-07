"""dc-judge — a quality verdict on every published placement.

    modal run modal/judge.py --limit 50                        rerun exactly 50 sampled pairs
    modal run --detach modal/judge.py --shards 24              the whole unjudged corpus
    modal run modal/judge.py --limit 4 --dry                   render exactly four, call nothing
    modal run modal/judge.py::collect                          rebuild decisions.json alone

Renders four cardinal views from 15° above each sample's posed pair, top and bottom, and an
anchor cutaway that reveals containment, then asks a vision model whether the arrangement
honours the phrase. Verdicts, and the frames they
were made from, land under `datasets/raw/review/<id>/`. Nothing here touches the dataset:
acting on a `false` stays a separate, deliberate step.

Rendering is synchronous JavaScript on one core, so a container's throughput is bounded by it
however wide the model calls run — which is what `--shards` is for. Shards are dealt explicit
id lists rather than each picking "whatever has no verdict", because several containers doing
that against one volume would race for the same samples and render them twice.
"""

from __future__ import annotations  # the local CLI may be older than the container's 3.12

import json
import os
import subprocess
import time
from pathlib import Path

import modal

REPO = Path(__file__).parent.parent

app = modal.App("dc-judge")

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

PUBLISH_PREFIX = "datasets/raw/stage1"
REVIEW_PREFIX = "datasets/raw/review"
SECRETS = [modal.Secret.from_name("dc-pipeline-env")]

TIMEOUT_S = 12 * 60 * 60
# Seven renders a sample is real CPU work, and it is what actually limits a container here.
SHARD_CPU = 4


def node_env(**extra: str) -> dict[str, str]:
    return {
        **os.environ,  # carries OPENROUTER_API_KEY in from the secret
        "SCENE_DIR": "/scene",
        "SCENE_PUBLISH_PREFIX": PUBLISH_PREFIX,
        "SCENE_REVIEW_PREFIX": REVIEW_PREFIX,
        "UV_THREADPOOL_SIZE": "64",
        "CORPUS_CONCURRENCY": "64",
        **extra,
    }


def run_node(label: str, *args: str) -> int:
    print(f"\n=== {label}", flush=True)
    started = time.monotonic()
    done = subprocess.run(["node", "/app/pipeline/judge.mjs", *args], env=node_env())
    print(f"=== {label} finished in {time.monotonic() - started:.0f}s (exit {done.returncode})", flush=True)
    return done.returncode


@app.function(image=image, volumes={"/scene": scene}, timeout=TIMEOUT_S)
def select(limit: int = 0, seed: int = 0) -> list[str]:
    """Pick the exact batch before any judge shard starts.

    A positive limit uses bounded-memory reservoir sampling, so asking for 100 visits the
    published directory once and never materializes or sorts the corpus. An unlimited
    production run selects only gaps. Shards receive explicit ids and cannot expand them.
    """
    import random

    scene.reload()
    started = time.monotonic()
    published = Path(f"/scene/{PUBLISH_PREFIX}")
    review = Path(f"/scene/{REVIEW_PREFIX}")
    rng = random.Random(seed or None)

    if limit > 0:
        picked: list[str] = []
        seen = 0
        for entry in published.iterdir():
            if not entry.is_dir():
                continue
            seen += 1
            if len(picked) < limit:
                picked.append(entry.name)
                continue
            replacement = rng.randrange(seen)
            if replacement < limit:
                picked[replacement] = entry.name
        scope = f"{seen} published"
    else:
        # Nothing has a verdict when the review prefix is gone, so eighty thousand stat calls
        # against a network volume would all miss — a fresh pass starts immediately instead.
        judged = review.is_dir()
        picked = [
            entry.name
            for entry in published.iterdir()
            if entry.is_dir() and not (judged and (review / entry.name / "decision.json").is_file())
        ]
        scope = f"{len(picked)} unjudged"

    picked.sort()
    print(
        f"  ✓ {len(picked)} sample(s) picked from {scope}"
        f" in {time.monotonic() - started:.0f}s",
        flush=True,
    )
    return picked


@app.function(image=image, volumes={"/scene": scene}, secrets=SECRETS, timeout=TIMEOUT_S, cpu=SHARD_CPU)
def judge(ids: list[str], dry: bool = False, force: bool = False) -> dict:
    """One shard: render and judge only the ids it was dealt.

    The slice goes to the script as a file rather than as arguments — eighty thousand ids is
    about four megabytes of argv against an exec limit nearer two, so a run with few shards
    would die on `Argument list too long` before rendering anything."""
    scene.reload()
    listing = Path("/tmp/shard-ids.txt")
    listing.write_text("\n".join(ids))
    args = [f"--ids-file={listing}"] + (["--dry"] if dry else []) + (["--force"] if force else [])
    failed = run_node(f"judge · {len(ids)} sample(s)", *args) != 0
    if not dry:
        scene.commit()
    return {"dealt": len(ids), "failed": failed}


@app.function(image=image, volumes={"/scene": scene}, timeout=TIMEOUT_S, cpu=8)
def breakdown(field: str = "category") -> dict:
    """What the filter kept, grouped by a metadata field — `category` unless told otherwise.

    The verdicts know nothing about how a sample was dealt, so this joins decisions.json back
    onto each sample's metadata. Category quotas are exact by construction, which is what makes
    the answer worth having: an even filter leaves the mix alone, and an uneven one silently
    re-weights a corpus that was balanced on purpose.
    """
    from concurrent.futures import ThreadPoolExecutor

    scene.reload()
    started = time.monotonic()
    decisions = json.loads(Path(f"/scene/{REVIEW_PREFIX}/decisions.json").read_text())["decisions"]
    published = Path(f"/scene/{PUBLISH_PREFIX}")

    def group_of(sample_id: str) -> str:
        try:
            return json.loads((published / sample_id / "metadata.json").read_text()).get(field) or "unknown"
        except (OSError, ValueError):
            return "unreadable"

    # Reads on a network volume are latency, not CPU, so the pool is wide.
    with ThreadPoolExecutor(max_workers=128) as pool:
        groups = pool.map(group_of, decisions.keys(), chunksize=64)

    tally: dict[str, dict[str, int]] = {}
    for group, row in zip(groups, decisions.values()):
        counts = tally.setdefault(group, {"kept": 0, "dropped": 0})
        counts["kept" if row["verdict"] else "dropped"] += 1

    kept_total = sum(c["kept"] for c in tally.values()) or 1
    before_total = sum(c["kept"] + c["dropped"] for c in tally.values()) or 1
    print(f"\n  {before_total} judged, joined in {time.monotonic() - started:.0f}s\n", flush=True)
    print(f"  {field:<14} {'before':>16} {'after':>16}   keep rate", flush=True)
    for group, counts in sorted(tally.items(), key=lambda item: -item[1]["kept"]):
        total = counts["kept"] + counts["dropped"]
        print(
            f"  {group:<14} {total:>7} {total / before_total * 100:>6.1f}%"
            f" {counts['kept']:>7} {counts['kept'] / kept_total * 100:>6.1f}%"
            f"   {counts['kept'] / total * 100:>5.1f}%",
            flush=True,
        )
    return tally


@app.function(image=image, volumes={"/scene": scene}, timeout=TIMEOUT_S)
def collect() -> dict:
    """Rolls every per-sample verdict up into one decisions.json."""
    scene.reload()
    run_node("collect verdicts", "--collect")
    scene.commit()
    summary = Path(f"/scene/{REVIEW_PREFIX}/decisions.json")
    if not summary.is_file():
        return {"total": 0}
    data = json.loads(summary.read_text())
    return {"total": data["total"], "kept": data["kept"], "dropped": data["dropped"]}


@app.local_entrypoint()
def main(limit: int = 0, shards: int = 1, dry: bool = False, seed: int = 0, ids_file: str = ""):
    if limit < 0:
        raise ValueError("--limit must be zero or greater")
    if shards < 1:
        raise ValueError("--shards must be at least one")

    # An explicit id list is both the tightest bound there is and the quickest start: walking
    # eighty thousand directories on the volume costs twenty minutes before a single render.
    if ids_file:
        ids = [line.strip() for line in Path(ids_file).read_text().splitlines() if line.strip()]
        bounded = True
        print(f"  ✓ {len(ids)} sample(s) named in {ids_file}", flush=True)
    else:
        ids = select.remote(limit, seed)
        bounded = limit > 0
    if not ids:
        print("nothing to judge")
        return

    slices = [ids[i::shards] for i in range(shards)]
    slices = [s for s in slices if s]
    print(f"\n{len(ids)} sample(s) across {len(slices)} shard(s)", flush=True)

    # `return_exceptions` lets the other bounded shards finish when one container dies.
    for result in judge.starmap(
        ((sample_ids, dry, bounded) for sample_ids in slices),
        return_exceptions=True,
        wrap_returned_exceptions=False,
    ):
        if isinstance(result, Exception):
            print(f"   ✗ a shard crashed: {result}", flush=True)
        elif result["failed"]:
            print(f"   ✗ shard of {result['dealt']} reported failures", flush=True)

    # A sample request must stay a sample request. Corpus-wide compilation remains an explicit
    # operation and runs automatically only after an intentionally unlimited production pass.
    if not dry and not bounded:
        print(collect.remote())
