"""dc-pipeline — the whole retrieval dataset pipeline, run on Modal end to end.

    modal run modal/pipeline.py --samples 500                    watch it
    modal run --detach modal/pipeline.py --samples 20000         let it outlive your terminal
    modal run --detach modal/pipeline.py --samples 20000 --shards 24
    modal run --detach modal/pipeline.py --relations 60000       grow the existing pairs
    modal run modal/pipeline.py --relations 1500 --pairs 500     ...or a slice of them first

`--relations` is the total number of backfilled relations the corpus should end up with, not
how many to add, so re-issuing the same command after a crash resumes instead of overshooting
and a later larger run counts a finished smoke test as progress.

Three phases, because only the middle one parallelizes. Scene generation deals exact
per-category quotas against everything already on disk and dedups globally, so it has to see
one whole corpus and runs alone; publishing is a handful of batched writes and does too.
Between them sits all the actual work — fetching each sample's Objaverse meshes, rendering
its reference images, voxelizing, placing, refining and baking — and every bit of that is
per-sample and independent.

It is also, apart from the physics pass, synchronous JavaScript. One container runs it on one
core no matter how wide `PLACE_CONCURRENCY` goes, because that knob overlaps HTTP and not
CPU. So the middle phase fans out instead. `fetch-assets.mjs` and `run.mjs` both already take
explicit sample ids, which makes a shard nothing more than those same commands over a slice,
and Modal volumes permit concurrent modification as long as containers do not write the same
files — which disjoint sample folders satisfy. Twelve shards turn ~15 hours of 20k samples
into ~1.3, for the same core-seconds and the same model spend.

Volume discipline is the one thing to respect. A shard reloads once on entry and commits once
on exit and never in between: a reload makes the volume look empty to the container that
issued it, and Modal's own background commits already persist work as it lands. Every file
the pipeline writes goes down by atomic rename, so a snapshot taken mid-flight is always
whole. Each stage skips what is already on the volume, so re-invoking after a failure — or
after a shard hits its timeout — costs only the gaps.
"""

from __future__ import annotations  # the local CLI may be older than the container's 3.12

import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
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
    # Mounted rather than copied, so it does not invalidate the shared image build.
    .add_local_dir(REPO / "scripts", "/app/scripts")
)

scene = modal.Volume.from_name("trellis-scene-vol-v2")

# Where `generated/` lives — the small half of every sample, alongside the work and published
# prefixes the rest of the pipeline writes.
STAGING_PREFIX = "datasets/raw/staging"
# The Objaverse seed pool — captions, tags and download caches — shared across runs.
POOL_PREFIX = "datasets/raw/pool"
# Every placement's prompt, answer and physics report. On the volume rather than in the
# container, whose filesystem goes away with it — taking the run's whole audit trail.
LOG_PREFIX = "datasets/raw/placement-logs"
# Verification reports. On the volume rather than in the container, for the same reason the
# logs are: the container's filesystem goes away with it.
REPORT_PREFIX = "datasets/raw/reports"

SECRETS = [modal.Secret.from_name("dc-pipeline-env")]

# Generating and publishing are minutes even at 50k. A shard carries its slice of the corpus
# and wants room for the tail of it.
PHASE_TIMEOUT_S = 6 * 60 * 60
SHARD_TIMEOUT_S = 12 * 60 * 60

# Fleet-wide, then divided across however many shards actually run. Every concurrency knob in
# the pipeline is per-process, so a fan-out that kept the single-container defaults would
# multiply them by the shard count — which is how twelve containers become a rate limit
# rather than a speedup. Dividing here keeps `--shards` a pure speed dial.
FLEET_WIDTHS = {"PLACE_CONCURRENCY": 240, "FETCH_CONCURRENCY": 72}

# The placement shard is the only phase that does real CPU work — the physics solvers, and
# a cloth drape most of all. Without an explicit request a container gets 0.125 cores and
# bursts on whatever the worker happens to have spare, while the physics pool sizes itself
# from the *host's* core count and cheerfully starts sixteen threads on that fraction.
# Asking for the cores and pinning the pool to match trades a few dollars for solves that
# run at a predictable speed instead of a contended one.
SHARD_CPU = 4

ROLES = ("anchor", "placed")


def shard_widths(shards: int) -> dict[str, str]:
    return {name: str(max(1, total // shards)) for name, total in FLEET_WIDTHS.items()}


def node_env(**extra: str) -> dict[str, str]:
    return {
        **os.environ,  # carries OPENROUTER_API_KEY in from the secret
        "GENERATED_DIR": f"/scene/{STAGING_PREFIX}",
        "POOL_DIR": f"/scene/{POOL_PREFIX}",
        "PLACEMENT_LOG_DIR": f"/scene/{LOG_PREFIX}",
        "SCENE_DIR": "/scene",
        # `/scene` is mounted here, so every operation — fetch, voxelize, refine, bake —
        # runs in-process against it rather than round-tripping through dc-scene-ops.
        "SCENE_OPS_DIRECT": "1",
        # Node runs `fs.promises` on libuv's pool, four threads by default. Publishing reads
        # twenty thousand metadata files off a network volume, so leaving it at four turns a
        # concurrent walk back into a serial one.
        # `fs.promises` runs on libuv's threadpool, four threads by default, so every
        # concurrent corpus read in the pipeline is capped by this rather than by its own
        # width knob. It has to match `CORPUS_CONCURRENCY` or the width is a lie.
        "UV_THREADPOOL_SIZE": "64",
        "CORPUS_CONCURRENCY": "64",
        **extra,
    }


def run_stage(label: str, script: str, *args: str, env: dict[str, str] | None = None) -> int:
    """`script` is a path under /app — "pipeline/run.mjs", "scripts/backfill-report.mjs"."""
    print(f"\n=== {label}", flush=True)
    started = time.monotonic()
    target = script if "/" in script else f"pipeline/{script}"
    done = subprocess.run(["node", f"/app/{target}", *args], env=env or node_env())
    print(f"=== {label} finished in {time.monotonic() - started:.0f}s (exit {done.returncode})", flush=True)
    return done.returncode


# Reading tens of thousands of small files off a network volume is latency, not CPU, so the
# GIL is irrelevant and threads are the whole win — serially this is a quarter of an hour of
# silence before the shards are even dealt.
READ_THREADS = 64


def sample_ids() -> list[str]:
    started = time.monotonic()
    staging = Path(f"/scene/{STAGING_PREFIX}")
    # One directory listing, then a concurrent existence check — `is_file` per entry is a
    # round trip, and serially that is the same stall as reading them all.
    names = sorted(p.name for p in staging.iterdir() if p.is_dir())
    with ThreadPoolExecutor(max_workers=READ_THREADS) as pool:
        found = pool.map(lambda n: n if (staging / n / "metadata.json").is_file() else None, names)
    ids = [name for name in found if name]
    print(f"  ✓ listed {len(ids)} sample(s) in {time.monotonic() - started:.0f}s", flush=True)
    return ids


def read_metadata(ids: list[str], label: str) -> list[tuple[str, dict]]:
    """Every id's metadata, read concurrently, with a progress line every few seconds."""
    started = time.monotonic()
    total = len(ids)
    done = 0
    out: list[tuple[str, dict]] = []
    last = started

    def load(sample: str):
        try:
            return sample, json.loads(Path(f"/scene/{STAGING_PREFIX}/{sample}/metadata.json").read_text())
        except (OSError, ValueError):
            return None

    with ThreadPoolExecutor(max_workers=READ_THREADS) as pool:
        for result in pool.map(load, ids):
            done += 1
            if result:
                out.append(result)
            now = time.monotonic()
            if now - last >= 3 or done == total:
                rate = done / max(now - started, 1e-6)
                eta = (total - done) / rate if rate else 0
                print(f"  … {label} {done}/{total} · {rate:.0f}/s · eta {eta:.0f}s", flush=True)
                last = now

    print(f"  ✓ {label} {len(out)}/{total} in {time.monotonic() - started:.0f}s", flush=True)
    return out


def meshed(ids: list[str]) -> list[str]:
    """The shard's ids that actually have both meshes. Placement takes named samples at face
    value and reports one without a mesh as a failure, so a fetch that did not land would
    otherwise read as a broken shard rather than a gap for the next run to fill."""
    return [
        sample
        for sample, data in read_metadata(ids, "checking meshes")
        if all(data.get(role, {}).get("mesh") for role in ROLES)
    ]


def default_report_name() -> str:
    from datetime import datetime, timezone

    return f"backfill-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.md"


def unplaced_ids() -> list[str]:
    """Staging samples not yet placed — what a backfill (or an interrupted run) leaves.

    A backfill grows an already-finished corpus, so dealing the shards every sample id would
    hand them tens of thousands of placed folders to skip one metadata read at a time."""
    return [
        sample
        for sample, data in read_metadata(sample_ids(), "scanning for unplaced")
        if not isinstance(data.get("combined_size"), list)
    ]


@app.function(image=image, volumes={"/scene": scene}, secrets=SECRETS, timeout=PHASE_TIMEOUT_S)
def prepare(samples: int) -> list[str]:
    """Stages 0 and 1, then the corpus the shards divide between them."""
    scene.reload()
    for prefix in (STAGING_PREFIX, POOL_PREFIX, LOG_PREFIX):
        Path(f"/scene/{prefix}").mkdir(parents=True, exist_ok=True)

    for label, script, args in [
        # The pool build skips everything already on the volume, so after its first run this
        # stage is a few seconds of checking.
        ("0 · objaverse pool", "objaverse-pool.mjs", []),
        ("1 · scenes", "generate-scenes.mjs", [f"--samples={samples}"]),
        # Part of generation rather than an afterthought: a sample without its shortened
        # phrase forms is half-written, and running this here is what stops the corpus
        # needing a separate amendment pass ever again. It skips whatever already has them.
        ("1c · placement variants", "placement-variants.mjs", []),
    ]:
        if run_stage(label, script, *args) != 0:
            print(f"   ✗ {label} reported a failure", flush=True)

    # Committed before any shard starts, or their containers mount a volume without the
    # scenes they were dealt.
    scene.commit()
    return sample_ids()


@app.function(image=image, volumes={"/scene": scene}, secrets=SECRETS, timeout=PHASE_TIMEOUT_S)
def backfill_relations(relations: int, pairs: int = 0) -> list[str]:
    """Stage 1b: new relationships dealt across the corpus's existing pairs.

    Like `prepare`, it runs alone — quotas and per-pair novelty need one whole view of the
    corpus. `pairs` restricts the deal to the first N pairs by sorted key, which is what
    makes a smoke test a genuine prefix of the full run rather than a throwaway: the same N
    are chosen every time, so re-running finishes them, and a later unrestricted run counts
    them as progress instead of redoing them.

    Returns only the samples that still need meshes or a placement, which after a backfill
    is the new folders plus whatever an earlier run left behind."""
    scene.reload()
    args = [f"--relations={relations}"] + ([f"--pairs={pairs}"] if pairs else [])
    if run_stage("1b · backfill relations", "backfill-relations.mjs", *args) != 0:
        print("   ✗ 1b · backfill relations reported a failure", flush=True)
    # Same reason it follows generation: the new relations are not finished samples until
    # they carry their shortened forms too.
    if run_stage("1c · placement variants", "placement-variants.mjs") != 0:
        print("   ✗ 1c · placement variants reported a failure", flush=True)
    scene.commit()
    return unplaced_ids()


@app.function(image=image, volumes={"/scene": scene}, timeout=PHASE_TIMEOUT_S)
def report(name: str) -> str:
    """The verification report, written to the volume so it outlives the container."""
    scene.reload()
    Path(f"/scene/{REPORT_PREFIX}").mkdir(parents=True, exist_ok=True)
    target = f"/scene/{REPORT_PREFIX}/{name}"
    # A non-zero exit here means the report found phrase-check failures, which is a finding
    # to read rather than a reason to lose the file.
    run_stage("report", "scripts/backfill-report.mjs", f"--out={target}", "--limit-examples=0")
    scene.commit()
    return target


@app.function(
    image=image, volumes={"/scene": scene}, secrets=SECRETS, timeout=SHARD_TIMEOUT_S, cpu=SHARD_CPU
)
def process(ids: list[str], shards: int, force: bool = False, overrides: dict | None = None) -> dict:
    """One shard: fetch and render its slice, then voxelize, place, refine and bake it.

    `force` re-places samples that are already placed, which is how a slice gets re-run
    after a solver change rather than being skipped as finished. `overrides` sets env vars
    for this shard only — the caller's shell does not reach the container, so tuning a
    solver knob against a real sample has to travel as an argument."""
    scene.reload()
    env = node_env(PHYSICS_WORKERS=str(SHARD_CPU), **shard_widths(shards), **(overrides or {}))

    failed = []
    if run_stage(f"2 · fetch + render · {len(ids)} sample(s)", "fetch-assets.mjs", *ids, env=env) != 0:
        failed.append("2 · fetch + render")

    ready = meshed(ids)
    place_args = (["--force"] if force else []) + ready
    if ready and run_stage(f"3-5 · place · {len(ready)} sample(s)", "run.mjs", *place_args, env=env) != 0:
        failed.append("3-5 · place")

    scene.commit()
    return {"dealt": len(ids), "placeable": len(ready), "failed_stages": failed}


@app.function(image=image, volumes={"/scene": scene}, secrets=SECRETS, timeout=PHASE_TIMEOUT_S)
def publish(force: bool = False) -> dict:
    """Stage 6, once every shard has committed what it placed.

    `--force` skips the walk over already-published folders. On a corpus this size that walk
    is tens of thousands of serial stats to build a skip-set that is nearly empty, so when
    little is published yet it costs far more than the re-writes it saves.
    """
    scene.reload()
    failed = run_stage("6 · publish", "upload.mjs", *(["--force"] if force else [])) != 0
    scene.commit()
    return {"failed_stages": ["6 · publish"] if failed else []}


#
# `main` is deliberately the only local entrypoint: with more than one, plain
# `modal run modal/pipeline.py` stops resolving and every documented command has to grow a
# `::main`. The operational extras below are ordinary functions, reached with `::name`.
#


@app.function(image=image, volumes={"/scene": scene}, timeout=PHASE_TIMEOUT_S)
def verify_variants() -> None:
    """Audits the four placement forms across staging and the published dataset.

        modal run modal/pipeline.py::verify_variants
    """
    scene.reload()
    run_stage("verify placement variants", "scripts/verify-variants.mjs")


@app.function(image=image, timeout=PHASE_TIMEOUT_S)
def replace(ids: str, shards: int = 1, force: bool = True, env: str = "") -> dict:
    """Re-place a named slice — how a solver change is checked against the samples that
    motivated it, instead of inferring from a unit test that the corpus agrees.

        modal run modal/pipeline.py::replace --ids id1,id2,id3
        modal run modal/pipeline.py::replace --ids id1 --env DRAPE_BUDGET_S=600
    """
    wanted = [i.strip() for i in ids.split(",") if i.strip()]
    if not wanted:
        raise ValueError("--ids takes a comma-separated list of sample ids")
    overrides = dict(pair.split("=", 1) for pair in env.split(",") if "=" in pair)
    print(f"re-placing {len(wanted)} sample(s), force={force}, overrides={overrides}", flush=True)
    return process.remote(wanted, shards, force, overrides)


@app.local_entrypoint()
def main(samples: int = 0, shards: int = 12, relations: int = 0, pairs: int = 0, report_name: str = ""):
    if samples and relations:
        raise ValueError("--samples generates new pairs, --relations grows the existing ones — pick one")
    if pairs and not relations:
        raise ValueError("--pairs only restricts a backfill")

    if relations:
        ids = backfill_relations.remote(relations, pairs)
        print(f"\n{len(ids)} sample(s) still need meshes or a placement", flush=True)
    else:
        ids = prepare.remote(samples)
        print(f"\n{len(ids)} sample(s) in the corpus", flush=True)
    if not ids:
        # Nothing to fetch or place, but a backfill still owes its report — this is the path
        # a re-run after a fully-finished attempt takes.
        if relations:
            print(f"report → {report.remote(report_name or default_report_name())}")
        return

    # Dealt round-robin rather than sliced, so a corpus whose heavy assets cluster
    # alphabetically does not land all of them on one shard.
    slices = [ids[i::shards] for i in range(shards)]
    slices = [slice_ for slice_ in slices if slice_]
    widths = shard_widths(len(slices))
    print(f"{len(slices)} shard(s) of ~{len(slices[0])} sample(s), {widths} each", flush=True)

    # `return_exceptions` so a container that dies — a timeout, an OOM, a preemption —
    # reports itself and leaves the rest to finish. Without it the whole fan-out unwinds on
    # the first crash and publishing never runs, stranding work that had already committed.
    # `wrap_returned_exceptions=False` hands back the real exception rather than Modal's
    # internal wrapper, which is the behaviour it is migrating to anyway.
    crashed = 0
    dealt = ((slice_, len(slices)) for slice_ in slices)
    for result in process.starmap(dealt, return_exceptions=True, wrap_returned_exceptions=False):
        if isinstance(result, Exception):
            crashed += 1
            print(f"   ✗ a shard crashed: {result}", flush=True)
        elif result["failed_stages"]:
            print(f"   ✗ shard of {result['dealt']}: {', '.join(result['failed_stages'])}", flush=True)

    # Publishing runs either way: whatever the surviving shards placed is committed and
    # deserves to land, and re-invoking the same command fills whatever did not.
    print(publish.remote())
    if relations:
        print(f"\nreport → {report.remote(report_name or default_report_name())}")
    if crashed:
        print(f"\n{crashed} shard(s) crashed — re-run to fill their gaps", flush=True)
