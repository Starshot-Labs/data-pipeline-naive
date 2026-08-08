"""dc-partfield — NVIDIA PartField (github.com/nv-tlabs/PartField) as a segmentation service.

    modal deploy modal/partfield_app.py
    modal run modal/partfield_app.py::prefetch
    modal run modal/partfield_app.py --path chair.glb --parts 8

Named `_app` for the same reason `voxhammer_app.py` is: Modal mounts this module flat at
/root, and a `partfield.py` sitting there is one sys.path away from shadowing the checkout's
own `partfield` package.

PartField predicts a continuous feature field over a shape and takes the mean feature of each
triangle; parts fall out of clustering those features under the mesh's own face adjacency.
Two scripts, driven exactly as the README drives them:

    partfield_inference.py   per-face features    exp_results/<name>/part_feat_<uid>_0_batch.npy
    run_part_clustering.py   agglomerative        <dump>/cluster_out/<uid>_0_<NN>.npy

Agglomerative clustering does not return one segmentation but a hierarchy, and the second
script writes every level of it: `<uid>_0_08.npy` is the cut with exactly eight parts. A job
is therefore the hierarchy rather than any one segmentation of it, and how many parts to take
is a question asked of a finished job instead of a reason to run another. Building the tree
costs about ninety seconds of A10G; reading a level out of it costs under a second, which is
why /cut is its own endpoint on a container with no GPU attached. What the right number of
parts is depends on the shape — eight is generous for a mug and mean for a chair — so it is
worth being cheap to change your mind about.

Both scripts run as subprocesses with the job directory as their working directory, because
`partfield_inference.py` has no flag for where its output goes: it always writes
`exp_results/<result_name>` relative to cwd. Running it from the job's own folder on the
volume is what puts the features somewhere a later container can find them, and is why
re-running a job resumes rather than recomputes.

Four details of the upstream scripts are load-bearing here:

  * Both scripts declare their flags as `type=bool`, so any non-empty value parses as True.
    A flag is turned off by leaving it out, or by passing the empty string.
  * `predict_step` silently skips any shape whose uid is "car", and the clustering script
    splits its output filenames on "_" to recover uids. Every upload is called `mesh`.
  * `n_point_per_face` is a request, not a setting — see `_sampling`. The demo's value is
    sized for the shapes it ships with and asks a 500k-face mesh for 55 GB of VRAM.
  * `export_colored_mesh_ply` colours its PLYs through `plt.cm.get_cmap`, which matplotlib
    removed in 3.9. Nothing here reads those PLYs, but the pin below keeps them an option.

The labels index the faces of `trimesh.load(path, force="mesh", process=False)`, the load
PartField's own dataloader performs before it centres and rescales the copy it clusters. So
`parts.glb` is rebuilt from the file as it was uploaded and comes back in the caller's
coordinate frame, one named and coloured node per part.

Long runs, so the shape is the one the rest of this project already talks to: POST /segment
spawns and returns a job id, /jobs/{id} reports the step it is on, and the artifacts come off
the volume one file at a time.
"""

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import modal

app = modal.App("dc-partfield")

REPO = "/root/PartField"
CKPT = "/ckpt"
JOBS = "/jobs"

# torch 2.4.0 / cu124 / python 3.10 is not a preference: torch-scatter publishes one wheel per
# (torch, cuda) pair, and the README's is the pair PartField was trained and released against.
PYTHON = "3.10"
GPU = "A10G"

CKPT_REPO = "mikaelaangel/partfield-ckpt"
CKPT_FILE = "model_objaverse.ckpt"

# `result_name` is joined onto a hardcoded `exp_results/`, so this is a name, not a path.
FEATURES = "features"
UID = "mesh"

# What a face-sample costs on the card. `sample_points` holds five float64 weight tensors of
# (faces x samples) and, while it sums them, up to three more three times that wide: ~112
# bytes each at peak, in the dtype trimesh hands over, which is always float64. The demo's
# 1000 samples a face is sized for the ~20k-face shapes it ships with; a 500k-face mesh asks
# for 55 GB of it. The budget is what an A10G can spare once the triplanes are resident.
SAMPLE_BYTES = 112
SAMPLE_BUDGET = 14 * 1024**3
# Points per triplane-sampling chunk. The demo's 10000 is ten faces at its own sample count.
CHUNK_POINTS = 65536

image = (
    modal.Image.debian_slim(python_version=PYTHON)
    .apt_install(
        "git",
        # open3d, vtk and pymeshlab each drag in GL/X11/Qt at import, and `dataloader.py`
        # imports all three at module scope whether or not their code paths ever run.
        "libgl1",
        "libglu1-mesa",
        "libglib2.0-0",
        "libgomp1",
        "libx11-6",
        "libxrender1",
        "libxext6",
        "libxt6",
        "libsm6",
        "libxkbcommon0",
        "libegl1",
        "libdbus-1-3",
        "libusb-1.0-0",
    )
    # matplotlib is imported at module scope by the clustering script, on a box with no display.
    .env({"MPLBACKEND": "Agg"})
    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "lightning==2.2",
        "h5py",
        "yacs",
        "trimesh",
        "scikit-image",
        "scikit-learn",
        "networkx",
        "loguru",
        # `dataloader.py` imports boto3 at module scope for a training path that never runs.
        "boto3",
        "mesh2sdf",
        "tetgen",
        "vtk",
        "pymeshlab",
        "plyfile",
        "einops",
        "libigl",
        "polyscope",
        "potpourri3d",
        "simple_parsing",
        "arrgh",
        "open3d",
        "psutil",
        "numpy<2",
        # partfield/config stamps its output directory with a Los Angeles timestamp, and
        # nothing else in the tree brings pytz in.
        "pytz",
        "matplotlib<3.9",
        "huggingface_hub",
        "fastapi[standard]",
        "python-multipart",
    )
    .pip_install("torch-scatter", find_links="https://data.pyg.org/whl/torch-2.4.0+cu124.html")
    # Last, so iterating on the checkout does not rebuild the stack above it.
    .run_commands(f"git clone --depth 1 https://github.com/nv-tlabs/PartField {REPO}")
)

ckpt_volume = modal.Volume.from_name("partfield-ckpt", create_if_missing=True)
jobs_volume = modal.Volume.from_name("partfield-jobs", create_if_missing=True)
jobs = modal.Dict.from_name("dc-partfield-jobs", create_if_missing=True)

VOLUMES = {CKPT: ckpt_volume, JOBS: jobs_volume}


def _ensure_ckpt() -> str:
    """The one weight file, on the checkpoint volume. 1.4 GB, public, so no secret."""
    from huggingface_hub import hf_hub_download

    target = Path(CKPT) / CKPT_FILE
    if not target.is_file():
        ckpt_volume.reload()
    if not target.is_file():
        hf_hub_download(repo_id=CKPT_REPO, filename=CKPT_FILE, local_dir=CKPT)
        ckpt_volume.commit()
    return str(target)


def _run(command: list[str], cwd: Path, stage: str) -> None:
    """stdout streams to the Modal log; stderr is kept, since that is where the traceback is.

    Unbuffered, because both scripts report progress by printing and a job that looks hung for
    ten minutes and a job that is hung are otherwise the same thing from outside.
    """
    done = subprocess.run(
        command,
        cwd=cwd,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        stderr=subprocess.PIPE,
        text=True,
    )
    if done.returncode != 0:
        raise RuntimeError(f"{stage} failed:\n{done.stderr[-3000:]}")


def _artifacts(num_parts: int) -> list[str]:
    """Named for the level they are, so the cuts of one job accumulate instead of overwrite."""
    return [f"parts_{num_parts:02d}.glb", f"labels_{num_parts:02d}.bin", f"labels_{num_parts:02d}.json"]


def _source(work: Path) -> Path:
    found = next(iter(sorted((work / "data").glob(f"{UID}.*"))), None)
    if found is None:
        raise RuntimeError(f"no mesh in {work / 'data'} — /segment writes it before spawning")
    return found


def _load(path: Path):
    """`partfield/utils.py:load_mesh_util` — the load its labels are indexed against."""
    import trimesh

    return trimesh.load(str(path), force="mesh", process=False)


def _colours(count: int):
    """One RGBA per part off a perceptually ordered ramp, reported so a caller can draw the
    same segmentation somewhere else and have the parts come out the same colour."""
    import numpy as np
    from matplotlib import colormaps

    turbo = colormaps["turbo"]
    return (np.array([turbo((part + 0.5) / count) for part in range(count)]) * 255).astype(np.uint8)


def _parts_glb(mesh, labels, colours) -> bytes:
    """One named, coloured node per part, over the faces of the mesh as it was uploaded."""
    import numpy as np
    import trimesh

    scene = trimesh.Scene()
    for part, colour in enumerate(colours):
        used, faces = np.unique(mesh.faces[labels == part], return_inverse=True)
        piece = trimesh.Trimesh(
            vertices=mesh.vertices[used],
            faces=faces.reshape(-1, 3),
            vertex_colors=np.tile(colour, (len(used), 1)),
            process=False,
        )
        name = f"part_{part:02d}"
        scene.add_geometry(piece, node_name=name, geom_name=name)
    return scene.export(file_type="glb")


def _cut(work: Path, mesh, num_parts: int, context: dict) -> dict:
    """One level of the hierarchy, as a GLB of named parts and a per-face label array.

    A submesh and an export, against a tree the clustering already wrote — which is the whole
    reason a granularity can be changed without the model, and why this is shared with `cut`.
    """
    import numpy as np

    level = work / "clustering" / "cluster_out" / f"{UID}_0_{num_parts:02d}.npy"
    if not level.is_file():
        raise RuntimeError(f"no {num_parts}-part cut in this job's hierarchy — {level.name} was never written")

    # Agglomerative labels are union-find roots, not 0..k-1, so compact them first.
    _, labels = np.unique(np.squeeze(np.load(level)), return_inverse=True)
    if len(mesh.faces) != len(labels):
        raise RuntimeError(f"the mesh has {len(mesh.faces)} faces, {level.name} labels {len(labels)}")

    sizes = np.bincount(labels).tolist()
    colours = _colours(len(sizes))
    parts_glb, labels_bin, labels_json = _artifacts(num_parts)
    (work / parts_glb).write_bytes(_parts_glb(mesh, labels, colours))
    (work / labels_bin).write_bytes(labels.astype("<i4").tobytes())

    summary = {
        "faces": len(labels),
        "vertices": len(mesh.vertices),
        "num_parts": len(sizes),
        "part_faces": sizes,
        "part_colors": colours[:, :3].tolist(),
        # Every other cut this job can answer, none of which needs the GPU again.
        "levels": sorted(int(path.stem.rsplit("_", 1)[1]) for path in level.parent.glob(f"{UID}_0_*.npy")),
        **context,
    }
    (work / labels_json).write_text(json.dumps(summary, indent=2))
    return summary


def _sampling(faces: int, ceiling: int) -> dict:
    """How densely each triangle can be sampled before the card runs out.

    The requested count is a ceiling rather than a setting: PartField's demo value is sized
    for the shapes it ships with, and a mesh an order of magnitude denser has to sample an
    order of magnitude more sparsely to fit. Fewer samples is only a coarser mean over a
    triangle that is itself proportionally smaller, so the feature barely moves.

    `n_sample_each` follows from it. `predict_step` reshapes each chunk it samples to
    (-1, n_point_per_face, C) to average within a face, and asserts the chunk divides evenly,
    so a chunk has to be whole faces — which is all the demo's 10000-with-1000 pairing is.
    """
    per_face = max(1, min(ceiling, SAMPLE_BUDGET // (SAMPLE_BYTES * faces)))
    return {"n_point_per_face": per_face, "n_sample_each": per_face * max(1, CHUNK_POINTS // per_face)}


def _params(max_clusters: int, option: int, with_knn: bool, n_point_per_face: int) -> dict:
    """What the hierarchy is, and so what the job id is derived from. `num_parts` is not here:
    it selects a level of the tree rather than changing it, and is answered by /cut."""
    if max_clusters < 2:
        raise ValueError(f"max_clusters must be at least 2, got {max_clusters}")
    if option not in (0, 1, 2):
        raise ValueError(f"option must be 0 (chain), 1 (face MST) or 2 (component MST), got {option}")
    if n_point_per_face < 1:
        raise ValueError(f"n_point_per_face must be positive, got {n_point_per_face}")
    return {
        "max_clusters": max_clusters,
        "option": option,
        "with_knn": bool(with_knn),
        "n_point_per_face": n_point_per_face,
    }


def _level(num_parts: int, max_clusters: int) -> int:
    if not 1 <= num_parts <= max_clusters:
        raise ValueError(f"num_parts must be between 1 and this job's max_clusters ({max_clusters}), got {num_parts}")
    return num_parts


def _within(job_id: str, name: str) -> Path:
    """Resolved against the job's own directory, so a crafted name cannot climb out of it."""
    root = (Path(JOBS) / job_id).resolve()
    target = (root / name).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        raise FileNotFoundError(f"no {name} for {job_id}")
    return target


def _submit(model_bytes: bytes, filename: str, sample: str, params: dict) -> str:
    """Put the upload on the volume and register the job. Same inputs, same id, so a repeat
    of a request attaches to the work already done for it."""
    suffix = Path(filename).suffix.lower()
    if suffix not in (".glb", ".obj", ".off"):
        raise ValueError(f"PartField reads .glb, .obj and .off meshes, not {suffix or filename!r}")

    digest = hashlib.sha256(model_bytes + json.dumps(params, sort_keys=True).encode()).hexdigest()[:10]
    job_id = f"pf-{re.sub(r'[^a-z0-9-]+', '-', sample.lower()).strip('-') or 'job'}-{digest}"

    data = Path(JOBS) / job_id / "data"
    data.mkdir(parents=True, exist_ok=True)
    (data / f"{UID}{suffix}").write_bytes(model_bytes)
    jobs_volume.commit()

    jobs[job_id] = {
        "status": "pending",
        "stage": "queued",
        "sample": sample,
        "params": params,
        "created_at": time.time(),
    }
    return job_id


def _fail(job_id: str, err: Exception) -> None:
    jobs_volume.commit()
    jobs[job_id] = {
        **jobs[job_id],
        "status": "failed",
        "error": f"{type(err).__name__}: {err}",
        "updated_at": time.time(),
    }


@app.function(image=image, volumes=VOLUMES, gpu=GPU, timeout=60 * 60)
def segment(job_id: str, num_parts: int) -> None:
    """Feature field, then the clustering hierarchy, then one cut of it as a GLB.

    Each step checks for its own output first, so a job that timed out or lost its container
    picks up where it stopped: the features are the expensive half and they are on the volume.
    The level comes in as an argument rather than off the job, which two callers asking the
    same hierarchy for different granularities would otherwise be overwriting under each other.
    """
    work = Path(JOBS) / job_id
    params = jobs[job_id]["params"]

    def stage(name: str) -> None:
        jobs[job_id] = {**jobs[job_id], "stage": name, "updated_at": time.time()}
        print(f"[{job_id}] {name}", flush=True)

    started = time.time()
    marks = {}

    def mark(name: str) -> None:
        marks[name] = round(time.time() - started, 1)

    try:
        # The upload was written by the web container; pull its commit before reading.
        jobs_volume.reload()
        data = work / "data"
        mesh = _load(_source(work))

        stage("checkpoint")
        ckpt = _ensure_ckpt()

        sampling = _sampling(len(mesh.faces), params["n_point_per_face"])
        # Recorded on the job so /cut can report the same provenance without recomputing it.
        jobs[job_id] = {**jobs[job_id], "sampling": sampling}
        print(f"[{job_id}] {len(mesh.faces)} faces, {sampling}", flush=True)

        features = work / "exp_results" / FEATURES
        if not (features / f"part_feat_{UID}_0_batch.npy").is_file():
            stage("feature field")
            _run(
                [
                    sys.executable, f"{REPO}/partfield_inference.py",
                    "-c", f"{REPO}/configs/final/demo.yaml",
                    "--opts",
                    "continue_ckpt", ckpt,
                    "result_name", FEATURES,
                    "dataset.data_path", str(data),
                    # One shape per job, so a worker per CPU only pays to import the
                    # dataloader's stack of mesh libraries again.
                    "dataset.val_num_workers", "0",
                    "n_point_per_face", str(sampling["n_point_per_face"]),
                    "n_sample_each", str(sampling["n_sample_each"]),
                    # Lightning's checkpoint callback is dead weight under `predict`, but it
                    # still stamps a directory. Not on the volume, please.
                    "output_dir", "/tmp/partfield",
                ],
                work,
                "partfield_inference",
            )
            mark("features")

        clustering = work / "clustering"
        cluster_out = clustering / "cluster_out"
        if not (cluster_out / f"{UID}_0_{params['max_clusters']:02d}.npy").is_file():
            stage("clustering")
            # `export_colored_mesh_ply` recolours the whole mesh in a Python loop once per
            # level and writes it out, which on a large shape costs more than the clustering
            # it illustrates — and nothing here reads those PLYs. Turning it off means passing
            # the empty string, since every flag is declared `type=bool` and so any non-empty
            # value is True; the script then lists a directory it only creates when the flag
            # is on, so it gets one anyway.
            (clustering / "ply").mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable, f"{REPO}/run_part_clustering.py",
                "--root", str(features),
                "--dump_dir", str(clustering),
                "--source_dir", str(data),
                "--use_agglo", "True",
                "--export_mesh", "",
                "--max_num_clusters", str(params["max_clusters"]),
                "--option", str(params["option"]),
            ]
            if params["with_knn"]:
                command += ["--with_knn", "True"]
            _run(command, work, "run_part_clustering")
            mark("clustering")

        stage("parts")
        summary = _cut(work, mesh, num_parts, {"params": params, "sampling": sampling})
        mark("parts")

        jobs_volume.commit()
        jobs[job_id] = {
            **jobs[job_id],
            "status": "done",
            "stage": "done",
            "files": _artifacts(num_parts),
            "record": {**summary, "seconds": marks, "total_seconds": round(time.time() - started, 1)},
            "updated_at": time.time(),
        }
    except Exception as err:  # noqa: BLE001 — the message is the only thing the client can act on
        _fail(job_id, err)
        raise


@app.function(image=image, volumes=VOLUMES, timeout=15 * 60)
def cut(job_id: str, num_parts: int) -> None:
    """Another granularity of a job already clustered — no GPU, and no model.

    The hierarchy holds every level up to the job's `max_clusters`, so this is the load of a
    label array and a rebuild of the GLB against it. Seconds, against the ninety the tree took.
    """
    work = Path(JOBS) / job_id
    started = time.time()
    try:
        jobs_volume.reload()
        jobs[job_id] = {**jobs[job_id], "stage": "parts", "updated_at": time.time()}
        record = jobs[job_id]
        summary = _cut(
            work,
            _load(_source(work)),
            num_parts,
            {"params": record["params"], "sampling": record.get("sampling")},
        )

        jobs_volume.commit()
        jobs[job_id] = {
            **jobs[job_id],
            "status": "done",
            "stage": "done",
            "files": _artifacts(num_parts),
            "record": {**summary, "total_seconds": round(time.time() - started, 1)},
            "updated_at": time.time(),
        }
    except Exception as err:  # noqa: BLE001 — the message is the only thing the client can act on
        _fail(job_id, err)
        raise


@app.function(image=image, volumes=VOLUMES, timeout=30 * 60)
def prefetch() -> dict:
    """`modal run modal/partfield_app.py::prefetch` — the checkpoint, before the first job."""
    return {"checkpoint": _ensure_ckpt()}


# The two ends of the job that the web container reaches by having the volume mounted, as
# functions, so `modal run` can get a mesh onto it and an artifact back off it without one.
@app.function(image=image, volumes=VOLUMES, timeout=15 * 60)
def submit(model_bytes: bytes, filename: str, sample: str, params: dict) -> str:
    return _submit(model_bytes, filename, sample, params)


@app.function(image=image, volumes=VOLUMES, timeout=15 * 60)
def fetch(job_id: str, name: str) -> bytes:
    jobs_volume.reload()
    return _within(job_id, name).read_bytes()


@app.function(image=image, volumes=VOLUMES, timeout=900)
@modal.concurrent(max_inputs=8)
@modal.asgi_app()
def web():
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.responses import FileResponse

    api = FastAPI(title="dc-partfield")

    @api.get("/health")
    def health():
        return {"ok": True, "gpu": GPU, "checkpoint": f"{CKPT_REPO}/{CKPT_FILE}", "artifacts": _artifacts(8)}

    @api.post("/segment")
    async def start(
        model: UploadFile = File(...),
        sample: str = Form(""),
        num_parts: int = Form(8),
        max_clusters: int = Form(20),
        # How the face adjacency is bridged where the mesh is not one connected component —
        # and only then, since all three return the plain shared-edge graph unchanged when it
        # already is. 0 chains the components in the order they came in; 1 and 2 bridge them
        # along a spanning tree instead, over every face and over the components respectively.
        # 1 is what PartField shows for fragmented meshes, but it builds a NetworkX graph with
        # ten edges per face and takes a minimum spanning tree of it, which a mesh in the
        # hundreds of thousands of faces does not survive in any useful time. 2 is the same
        # idea at a scale that does.
        option: int = Form(0),
        with_knn: bool = Form(False),
        # A ceiling on the points sampled per triangle to average a feature from, not a
        # setting: a mesh dense enough to overrun the card gets fewer. PartField's demo value.
        n_point_per_face: int = Form(1000),
    ):
        try:
            params = _params(max_clusters, option, with_knn, n_point_per_face)
            level = _level(num_parts, max_clusters)
            job_id = _submit(await model.read(), model.filename or "mesh.glb", sample, params)
        except ValueError as err:
            raise HTTPException(status_code=400, detail=str(err))
        segment.spawn(job_id, level)
        return {"job_id": job_id, "sample": sample, "num_parts": level, "params": params}

    @api.post("/cut")
    async def start_cut(job_id: str = Form(...), num_parts: int = Form(...)):
        """A different number of parts out of a hierarchy /segment already built."""
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail=f"no job {job_id} — call /segment first")
        try:
            level = _level(num_parts, jobs[job_id]["params"]["max_clusters"])
        except ValueError as err:
            raise HTTPException(status_code=400, detail=str(err))

        jobs[job_id] = {**jobs[job_id], "status": "pending", "stage": "queued-cut", "updated_at": time.time()}
        cut.spawn(job_id, level)
        return {"job_id": job_id, "num_parts": level}

    @api.get("/jobs/{job_id}")
    def status(job_id: str):
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail=f"no job {job_id}")
        return jobs[job_id]

    @api.get("/jobs/{job_id}/file/{name:path}")
    def file(job_id: str, name: str):
        jobs_volume.reload()
        try:
            return FileResponse(_within(job_id, name))
        except FileNotFoundError as err:
            raise HTTPException(status_code=404, detail=str(err))

    @api.delete("/jobs/{job_id}")
    def discard(job_id: str):
        """A job keeps its features and every level of its hierarchy — tens of megabytes."""
        shutil.rmtree(Path(JOBS) / job_id, ignore_errors=True)
        jobs_volume.commit()
        jobs.pop(job_id, None)
        return {"discarded": job_id}

    return api


@app.local_entrypoint()
def main(
    path: str,
    parts: int = 8,
    max_clusters: int = 20,
    option: int = 0,
    with_knn: bool = False,
    n_point_per_face: int = 1000,
    out: str = "segmented.glb",
):
    """`modal run modal/partfield_app.py --path chair.glb --parts 8` — the service, in one call."""
    source = Path(path)
    params = _params(max_clusters, option, with_knn, n_point_per_face)
    level = _level(parts, max_clusters)
    job_id = submit.remote(source.read_bytes(), source.name, source.stem, params)
    print(f"job {job_id}")

    segment.remote(job_id, level)
    parts_glb, _, labels_json = _artifacts(level)
    Path(out).write_bytes(fetch.remote(job_id, parts_glb))
    print(json.loads(fetch.remote(job_id, labels_json)))
    print(f"wrote {out}")
