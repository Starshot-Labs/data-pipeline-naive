"""P3-SAM automatic mesh segmentation on Modal.

Deploy with::

    modal deploy modal/p3sam_app.py
    modal run modal/p3sam_app.py::prefetch
    modal run modal/p3sam_app.py --path model.glb

The CUDA extension and all model dependencies are built in Modal. The local browser client
only uploads a mesh, polls the job, and downloads a browser-ready GLB plus face labels.
"""

import hashlib
import json
import os
import re
import shutil
import time
from pathlib import Path

import modal

app = modal.App("dc-p3sam")

REPO = "/root/Hunyuan3D-Part"
P3SAM = f"{REPO}/P3-SAM"
DEMO = f"{P3SAM}/demo"
CACHE = "/cache"
JOBS = "/jobs"

PYTHON = "3.10"
CUDA = "12.4.1"
GPU = "A100-80GB"
ARCH = "8.0"
UPSTREAM_COMMIT = "e96be065375438962375b55326416291342958a7"

P3SAM_REPO = "tencent/Hunyuan3D-Part"
P3SAM_FILE = "p3sam/p3sam.safetensors"
SONATA_REPO = "facebook/sonata"
SONATA_FILE = "sonata.pth"
P3SAM_DIR = f"{CACHE}/p3sam"
P3SAM_CKPT = f"{P3SAM_DIR}/{P3SAM_FILE}"
SONATA_DIR = f"{CACHE}/sonata"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA}-devel-ubuntu22.04", add_python=PYTHON)
    .apt_install("git", "build-essential", "ninja-build", "libgl1", "libglib2.0-0", "libgomp1")
    .env(
        {
            "CC": "gcc",
            "CXX": "g++",
            "TORCH_CUDA_ARCH_LIST": ARCH,
            "MAX_JOBS": "4",
            "HF_HOME": f"{CACHE}/hf",
            "HF_HUB_CACHE": f"{CACHE}/hf/hub",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "PYTHONPATH": f"{REPO}:{P3SAM}:{REPO}/XPart/partgen",
        }
    )
    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install("wheel", "setuptools", "packaging", "ninja")
    .pip_install(
        "numpy==1.26.4",
        "scipy",
        "scikit-learn",
        "scikit-image",
        "trimesh",
        "fpsample",
        "numba",
        "tqdm",
        "addict",
        "easydict",
        "omegaconf",
        "timm",
        "safetensors",
        "huggingface_hub",
        "hf_transfer",
        "spconv-cu124",
        "fastapi[standard]",
        "python-multipart",
    )
    .pip_install("torch-scatter", find_links="https://data.pyg.org/whl/torch-2.4.0+cu124.html")
    .run_commands(
        f"git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-Part.git {REPO}",
        f"cd {REPO} && git checkout {UPSTREAM_COMMIT}",
        f"cd {P3SAM}/utils/chamfer3D && pip install --no-build-isolation .",
    )
)

cache_volume = modal.Volume.from_name("p3sam-cache", create_if_missing=True)
jobs_volume = modal.Volume.from_name("p3sam-jobs", create_if_missing=True)
jobs = modal.Dict.from_name("dc-p3sam-jobs", create_if_missing=True)

VOLUMES = {CACHE: cache_volume, JOBS: jobs_volume}


def _ensure_weights() -> dict[str, str]:
    from huggingface_hub import hf_hub_download

    p3sam = Path(P3SAM_CKPT)
    sonata = Path(SONATA_DIR) / SONATA_FILE
    if not p3sam.is_file() or not sonata.is_file():
        cache_volume.reload()
    if not p3sam.is_file():
        hf_hub_download(repo_id=P3SAM_REPO, filename=P3SAM_FILE, local_dir=P3SAM_DIR)
    if not sonata.is_file():
        hf_hub_download(repo_id=SONATA_REPO, filename=SONATA_FILE, local_dir=SONATA_DIR)
    cache_volume.commit()
    return {"p3sam": str(p3sam), "sonata": str(sonata)}


def _artifacts() -> list[str]:
    return ["parts.glb", "labels.bin", "result.json"]


def _within(job_id: str, name: str) -> Path:
    root = (Path(JOBS) / job_id).resolve()
    target = (root / name).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        raise FileNotFoundError(f"no {name} for {job_id}")
    return target


def _colours(count: int):
    import colorsys
    import numpy as np

    return np.array(
        [
            [
                *[round(channel * 255) for channel in colorsys.hsv_to_rgb((index * 0.61803398875) % 1, 0.68, 0.96)],
                255,
            ]
            for index in range(count)
        ],
        dtype=np.uint8,
    )


def _compact_labels(face_ids):
    import numpy as np

    labels = np.asarray(face_ids, dtype=np.int64)
    valid = labels >= 0
    if not np.any(valid):
        raise RuntimeError("P3-SAM returned no labelled faces")

    compact = np.full(labels.shape, -1, dtype=np.int32)
    _, compact[valid] = np.unique(labels[valid], return_inverse=True)
    if np.any(~valid):
        compact[~valid] = int(compact[valid].max()) + 1
    return compact


def _parts_glb(mesh, labels, colours) -> bytes:
    import numpy as np
    import trimesh

    scene = trimesh.Scene()
    for part, colour in enumerate(colours):
        selected = mesh.faces[labels == part]
        if not len(selected):
            continue
        used, faces = np.unique(selected, return_inverse=True)
        piece = trimesh.Trimesh(
            vertices=mesh.vertices[used],
            faces=faces.reshape(-1, 3),
            vertex_colors=np.tile(colour, (len(used), 1)),
            process=False,
        )
        name = f"part_{part:02d}"
        scene.add_geometry(piece, node_name=name, geom_name=name)
    return scene.export(file_type="glb")


def _submit(model_bytes: bytes, filename: str, sample: str, params: dict) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in (".glb", ".obj", ".ply"):
        raise ValueError(f"P3-SAM reads .glb, .obj and .ply meshes, not {suffix or filename!r}")
    if not model_bytes:
        raise ValueError("the uploaded mesh is empty")

    digest = hashlib.sha256(model_bytes + json.dumps(params, sort_keys=True).encode()).hexdigest()[:10]
    slug = re.sub(r"[^a-z0-9-]+", "-", sample.lower()).strip("-") or "job"
    job_id = f"p3-{slug}-{digest}"
    work = Path(JOBS) / job_id
    work.mkdir(parents=True, exist_ok=True)
    (work / f"source{suffix}").write_bytes(model_bytes)
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
        "stage": "failed",
        "error": f"{type(err).__name__}: {err}",
        "updated_at": time.time(),
    }


@app.cls(
    image=image,
    volumes=VOLUMES,
    gpu=GPU,
    timeout=60 * 60,
    startup_timeout=20 * 60,
    scaledown_window=5 * 60,
    max_containers=1,
)
class P3SAMService:
    @modal.enter()
    def load(self):
        import sys

        try:
            weights = _ensure_weights()
            os.chdir(DEMO)
            if DEMO not in sys.path:
                sys.path.insert(0, DEMO)
            if P3SAM not in sys.path:
                sys.path.insert(0, P3SAM)

            from auto_mask import AutoMask
            from models import sonata

            original_load = sonata.load

            def load_sonata(name="sonata", repo_id=SONATA_REPO, download_root=None, custom_config=None, ckpt_only=False):
                config = {**(custom_config or {}), "enable_flash": False}
                return original_load(
                    name,
                    repo_id=repo_id,
                    download_root=SONATA_DIR,
                    custom_config=config,
                    ckpt_only=ckpt_only,
                )

            sonata.load = load_sonata
            self.automask = AutoMask(weights["p3sam"])
        except Exception as err:
            for job_id in list(jobs.keys()):
                record = jobs[job_id]
                if record.get("status") == "pending":
                    _fail(job_id, err)
            raise

    @modal.method()
    def segment(self, job_id: str) -> None:
        import numpy as np
        import torch
        import trimesh

        work = Path(JOBS) / job_id
        started = time.time()

        def stage(name: str) -> None:
            jobs[job_id] = {**jobs[job_id], "stage": name, "updated_at": time.time()}
            print(f"[{job_id}] {name}", flush=True)

        try:
            jobs_volume.reload()
            source = next(iter(sorted(work.glob("source.*"))), None)
            if source is None:
                raise RuntimeError(f"no source mesh for {job_id}")

            stage("loading mesh")
            mesh = trimesh.load(source, force="mesh", process=False)
            if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
                raise ValueError("the upload contains no triangle mesh")
            input_faces = len(mesh.faces)
            input_vertices = len(mesh.vertices)

            params = jobs[job_id]["params"]
            stage("P3-SAM inference")
            with torch.inference_mode():
                aabb, face_ids, segmented = self.automask.predict_aabb(
                    mesh,
                    seed=params["seed"],
                    is_parallel=False,
                    post_process=params["post_process"],
                    threshold=params["threshold"],
                    save_mid_res=False,
                    show_info=True,
                )

            stage("exporting parts")
            labels = _compact_labels(face_ids)
            count = int(labels.max()) + 1
            colours = _colours(count)
            sizes = np.bincount(labels, minlength=count).tolist()
            (work / "parts.glb").write_bytes(_parts_glb(segmented, labels, colours))
            (work / "labels.bin").write_bytes(labels.astype("<i4").tobytes())

            summary = {
                "job_id": job_id,
                "source": source.name,
                "faces": len(segmented.faces),
                "vertices": len(segmented.vertices),
                "input_faces": input_faces,
                "input_vertices": input_vertices,
                "num_parts": count,
                "part_faces": sizes,
                "part_colors": colours[:, :3].tolist(),
                "aabb": np.asarray(aabb).tolist(),
                "params": params,
                "cleaned_mesh": True,
                "total_seconds": round(time.time() - started, 1),
            }
            (work / "result.json").write_text(json.dumps(summary, indent=2))
            jobs_volume.commit()
            jobs[job_id] = {
                **jobs[job_id],
                "status": "done",
                "stage": "done",
                "files": _artifacts(),
                "record": summary,
                "updated_at": time.time(),
            }
        except Exception as err:
            _fail(job_id, err)
            raise


@app.function(image=image, volumes=VOLUMES, timeout=30 * 60)
def prefetch() -> dict:
    return _ensure_weights()


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

    api = FastAPI(title="dc-p3sam")

    @api.get("/health")
    def health():
        return {
            "ok": True,
            "gpu": GPU,
            "checkpoint": f"{P3SAM_REPO}/{P3SAM_FILE}",
            "upstream_commit": UPSTREAM_COMMIT,
        }

    @api.post("/segment")
    async def start(
        model: UploadFile = File(...),
        sample: str = Form(""),
        post_process: bool = Form(True),
        threshold: float = Form(0.95),
        seed: int = Form(42),
    ):
        if not 0 < threshold <= 1:
            raise HTTPException(status_code=400, detail="threshold must be greater than 0 and at most 1")
        params = {"post_process": bool(post_process), "threshold": float(threshold), "seed": int(seed)}
        try:
            job_id = _submit(await model.read(), model.filename or "mesh.glb", sample, params)
        except ValueError as err:
            raise HTTPException(status_code=400, detail=str(err))
        P3SAMService().segment.spawn(job_id)
        return {"job_id": job_id, "sample": sample, "params": params}

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
        shutil.rmtree(Path(JOBS) / job_id, ignore_errors=True)
        jobs_volume.commit()
        jobs.pop(job_id, None)
        return {"discarded": job_id}

    return api


@app.local_entrypoint()
def main(path: str, post_process: bool = True, threshold: float = 0.95, seed: int = 42, out: str = "p3sam.glb"):
    source = Path(path)
    params = {"post_process": post_process, "threshold": threshold, "seed": seed}
    job_id = submit.remote(source.read_bytes(), source.name, source.stem, params)
    print(f"job {job_id}")
    P3SAMService().segment.remote(job_id)
    Path(out).write_bytes(fetch.remote(job_id, "parts.glb"))
    print(json.loads(fetch.remote(job_id, "result.json")))
    print(f"wrote {out}")
