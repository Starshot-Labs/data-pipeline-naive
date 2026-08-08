"""dc-voxhammer — VoxHammer (github.com/Nelipot-Lee/VoxHammer) as a service.

    modal deploy modal/voxhammer_app.py

Named `_app` rather than `voxhammer.py` deliberately: Modal mounts this module flat at
/root, where the name would shadow the checkout's own `voxhammer` package and turn
`voxhammer.edit_pipeline` into "not a package".

Experiment 2's GPU half. The paper edits a 3D asset in TRELLIS's own latent space: it
inverts the asset, then re-denoises it while pinning the latents and attention keys of
everything outside a mask, so the region the mask releases changes and the rest comes back
bit-for-bit. Placing an object becomes an edit rather than a second mesh plus a transform.

It cannot run on a laptop — 40 GB of VRAM minimum, Linux, and a stack of CUDA extensions —
which is why it lives here next to the rest of this project's GPU work.

The paper's steps run as its own entry points, but split across two HTTP calls so the
inpaint can happen off-GPU on the caller's side:

    POST /render   utils/render_rgb_and_mask.py   5 RGB views + the 2D mask, by depth
                                                   comparison; picks the most-visible view
    (caller)       the object is inserted into that view — see below
    POST /edit     inference.py                   150-view render, DINOv2 features, voxel
                                                   mask, invert + re-denoise, decode

The inpaint that VoxHammer runs with FLUX.1-Fill-dev is deliberately NOT here. FLUX is
conditioned on text alone, so it never sees the object being placed and hallucinates one
from the phrase. `2d_edit.png` is a documented interface — `render_rgb_and_mask.py` produces
the view, `run_edit` consumes the edited copy — so the caller makes it instead, with a
reference-image edit (nano-banana) that is shown the actual object photo. The GPU service
keeps only VoxHammer's own method, unmodified; the swap is at the seam the paper already has.

`render_rgb_and_mask.py` is a subprocess because that is how the README drives it, and
because it keeps `bpyrenderer`'s Blender session away from the one `inference.py` opens
later. The edit is called in-process as `run_complete_pipeline`, the function its own CLI
calls, so that `render_params` can carry `scale=1, offset=(0,0,0)`.

That last detail is the whole ballgame for correctness. `bpy_render` normalizes the model
it renders, while `delete_region_voxel` imports the mask with no normalization at all and
tests it against a fixed 64³ lattice over [-0.5, 0.5]³. Two different frames. The pipeline
sends both meshes already inside that cube (see `pipeline/mask.mjs`), so normalizing again
is exactly what must not happen — and it is skipped by passing the parameters that turn it
off rather than by hoping Blender's bounding box agrees with ours.

Long runs, so the shape is the same as the Trellis endpoint the pipeline already talks to:
POST /edit spawns and returns a job id, /jobs/{id} reports the step it is on, and the
artifacts come off the volume one file at a time.
"""

import asyncio
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

import modal

app = modal.App("dc-voxhammer")

REPO = "/voxhammer"
CACHE = "/cache"
JOBS = "/jobs"
RENDERER = "/root/voxhammer_render.py"

# VoxHammer pins torch 2.4.0/cu118 and spconv-cu118, so the base has to match. python 3.10
# is not a preference either: bpy publishes one wheel per Blender release per minor version,
# and 4.0.0 — the version the requirements pin — is cp310 only.
CUDA = "11.8.0"
PYTHON = "3.10"
# A100 only. Fixed here so the CUDA extensions below compile without a GPU attached, and so
# nvdiffrast's runtime JIT targets the card the worker actually gets.
ARCH = "8.0"
GPU = "A100-80GB"

# The commit utils3d is pinned to in the install instructions. Unpinned it moves under the
# feature extractor's feet — `intrinsics_from_fov_xy`, `project_cv`, `io.read_ply`.
UTILS3D = "9a4eb15e4021b67b12c460c7057d642626897ec8"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA}-devel-ubuntu22.04", add_python=PYTHON)
    .apt_install(
        "git",
        "build-essential",
        "ninja-build",
        "libgl1",
        "libglib2.0-0",
        "libgomp1",
        # Blender ships as a library here and still wants an X11/EGL surface at import.
        "libxrender1",
        "libxi6",
        "libxxf86vm1",
        "libxfixes3",
        "libxkbcommon0",
        "libsm6",
        "libegl1",
        "libopengl0",
    )
    .env(
        {
            "CC": "gcc",
            "CXX": "g++",
            "TORCH_CUDA_ARCH_LIST": ARCH,
            # The 2D mask arrives as an EXR depth pass; cv2 ignores the format without this.
            "OPENCV_IO_ENABLE_OPENEXR": "1",
            # TRELLIS picks its attention backend and sparse-conv algorithm from the
            # environment. xformers rather than flash-attn: the wheel matching torch 2.4
            # exists, so the image builds in minutes instead of compiling flash-attn, and
            # the arithmetic is the same either way. Both variables are needed — the dense
            # and sparse attention modules read their own, and the sparse one imports
            # flash_attn at module scope unless told otherwise.
            "ATTN_BACKEND": "xformers",
            "SPARSE_ATTN_BACKEND": "xformers",
            "SPCONV_ALGO": "native",
            # Every weight lands on the cache volume, so a cold container downloads nothing.
            "HF_HOME": f"{CACHE}/hf",
            "HF_HUB_CACHE": f"{CACHE}/hf/hub",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "TORCH_HOME": f"{CACHE}/torch",
            "PYTHONPATH": REPO,
        }
    )
    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        "xformers==0.0.27.post2",
        index_url="https://download.pytorch.org/whl/cu118",
    )
    .pip_install("wheel", "setuptools", "packaging", "ninja")
    .pip_install(
        "numpy<2",
        "pillow",
        "tqdm",
        "easydict",
        "opencv-python",
        "scipy",
        "rembg",
        "onnxruntime",
        "trimesh",
        "pysdf",
        "open3d",
        "plyfile",
        "xatlas",
        "pyvista",
        "pymeshfix",
        "igraph",
        # `mathutils` is in the requirements but not installed here: the bpy wheel below
        # bundles Blender's own, and the PyPI package would both shadow it and need Eigen
        # headers to build.
        #
        # transformers is pinned to a release contemporary with diffusers 0.34: the current
        # one imports `DTensor` from `torch.distributed.tensor`, which torch 2.4 does not
        # have, and FLUX's text encoders come through it.
        "transformers==4.53.2",
        "sentencepiece",
        "accelerate",
        "diffusers==0.34.0",
        "huggingface_hub",
        "hf_transfer",
        "fastapi[standard]",
        "python-multipart",
    )
    .pip_install("spconv-cu118")
    .pip_install("bpy==4.0.0", extra_index_url="https://download.blender.org/pypi/")
    .pip_install(f"git+https://github.com/EasternJournalist/utils3d.git@{UTILS3D}")
    .pip_install("git+https://github.com/huanngzh/bpy-renderer.git")
    .run_commands("pip install --no-build-isolation git+https://github.com/NVlabs/nvdiffrast.git")
    # The gaussian rasterizer TRELLIS bakes vertex colour with, from the fork the install
    # instructions name. A CUDA extension, compiled here rather than on first use.
    .run_commands(
        "git clone --depth 1 https://github.com/autonomousvision/mip-splatting.git /tmp/mip-splatting",
        "pip install --no-build-isolation /tmp/mip-splatting/submodules/diff-gaussian-rasterization/",
    )
    .pip_install("kaolin", find_links="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu118.html")
    # Last, so iterating on the checkout does not rebuild the stack above it.
    .run_commands(f"git clone --depth 1 https://github.com/Nelipot-Lee/VoxHammer.git {REPO}")
    # Mounted rather than copied, so editing the renderer costs a deploy and not a rebuild.
    .add_local_file(Path(__file__).parent / "voxhammer_render.py", f"{RENDERER}")
)

cache = modal.Volume.from_name("voxhammer-cache", create_if_missing=True)
jobs_volume = modal.Volume.from_name("voxhammer-jobs", create_if_missing=True)
jobs = modal.Dict.from_name("dc-voxhammer-jobs", create_if_missing=True)

VOLUMES = {CACHE: cache, JOBS: jobs_volume}
HF_SECRET = modal.Secret.from_name("huggingface-secret")  # HF_TOKEN, for the gated FLUX weights

TRELLIS_REPO = "microsoft/TRELLIS-image-large"
FLUX_REPO = "black-forest-labs/FLUX.1-Fill-dev"
TRELLIS_DIR = f"{CACHE}/TRELLIS-image-large"

# README step 1: the released pipeline.json omits the two encoders inversion needs — it was
# built for generation, which only ever decodes.
ENCODERS = {
    "sparse_structure_encoder": "ckpts/ss_enc_conv3d_16l8_fp16",
    "slat_encoder": "ckpts/slat_enc_swin8_B_64l8_fp16",
}

# What a finished job hands back, in the order it is produced.
ARTIFACTS = [
    "images/2d_render.png",
    "images/2d_mask.png",
    "images/2d_edit.png",
    "output.glb",
]


def _prepare_trellis() -> str:
    """TRELLIS's weights on the cache volume, with the encoders added to pipeline.json."""
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=TRELLIS_REPO, local_dir=TRELLIS_DIR)
    config_path = Path(TRELLIS_DIR) / "pipeline.json"
    config = json.loads(config_path.read_text())
    models = config["args"]["models"]

    for key, relative in ENCODERS.items():
        if not (Path(TRELLIS_DIR) / f"{relative}.safetensors").exists():
            raise RuntimeError(f"{TRELLIS_REPO} has no {relative}.safetensors — inversion needs it")
        models[key] = relative

    config_path.write_text(json.dumps(config, indent=4))
    return TRELLIS_DIR


# A view catching only a sliver of the region is not usable: the object has to be painted
# into it and then found again by comparing against the clean render. One part in two hundred
# of a 1024² frame is roughly a 72 px square. Measured against real samples, a usable angle
# lands between 2% and 5% and a marginal one an order of magnitude below that, so the floor
# sits in the gap rather than close to either side.
MIN_COVERAGE = 5e-3


def _pick_view(images_dir: Path, cameras: list[dict]) -> tuple[Optional[int], list[dict]]:
    """The best-ranked candidate whose mask actually landed on screen, and what each saw.

    The README leaves the choice to the eye — "select one pair" — which a batch cannot do.
    The caller's model ranks the angles from the phrase and the occupancy grid, which is the
    part it can reason about; whether the region survives the anchor's own geometry is a
    question only the depth comparison answers. So the ranking is the preference and this is
    the veto. Returns no index rather than raising, so the coverage it measured can be
    recorded against the job before the failure is.
    """
    import cv2

    seen = []
    for index, camera in enumerate(cameras):
        mask_path = images_dir / f"mask_{index:04d}.png"
        rendered = (images_dir / f"render_{index:04d}.png").is_file()
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.is_file() else None
        coverage = None if mask is None or not rendered else round(float((mask > 0).mean()), 6)
        seen.append({**camera, "coverage": coverage})

    chosen = next(
        (i for i, candidate in enumerate(seen) if (candidate["coverage"] or 0) >= MIN_COVERAGE),
        None,
    )
    return chosen, seen


def _match_size(reference: Path, target: Path) -> None:
    """The edited view resampled to the render's size, which VoxHammer assumes but never checks.

    `preprocess_image` derives its resize factor from the source view alone — `min(1, 1024 /
    max(src.size))`, which is exactly 1 for a 1024 render, so its resize branch never runs. It
    then takes the union of the two objects' bounding boxes and crops *both* images with it. A
    512 target against a 1024 render is therefore cropped in the wrong coordinate system: the
    object lands at half scale, off centre, in a frame that is three quarters padding, while
    the cross-attention mask is cropped correctly and points somewhere else entirely. The
    structure stage gets a target condition that contradicts the geometry pinned around it and
    predicts no new occupancy, so the edit returns the original asset untouched.
    """
    from PIL import Image

    with Image.open(reference) as source:
        size = source.size
    with Image.open(target) as edited:
        if edited.size == size:
            return
        print(f"resampling 2d_edit {edited.size} -> {size}", flush=True)
        edited.convert("RGB").resize(size, Image.LANCZOS).save(target)


def _cameras(raw: str) -> list[dict]:
    """The caller's ranked angles, best first, checked before a GPU is spent on them."""
    try:
        candidates = json.loads(raw)
    except json.JSONDecodeError as err:
        raise ValueError(f"cameras is not JSON: {err}") from err
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("cameras must be a non-empty list of {elevation, azimuth}")

    clean = []
    for candidate in candidates:
        try:
            elevation = float(candidate["elevation"])
            azimuth = float(candidate["azimuth"])
        except (TypeError, KeyError, ValueError) as err:
            raise ValueError(f"bad camera {candidate!r}: {err}") from err
        # The poles leave the camera's up vector undefined, and looking straight down a
        # cavity is no better than looking down it at 80.
        if not -90 < elevation < 90:
            raise ValueError(f"elevation {elevation} is not between -90 and 90")
        clean.append({"elevation": elevation, "azimuth": azimuth % 360})
    return clean


def _run(command: list[str], stage: str) -> None:
    done = subprocess.run(command, cwd=REPO, capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(f"{stage} failed:\n{done.stdout[-2000:]}\n{done.stderr[-3000:]}")


def _fail(job_id: str, err: Exception) -> None:
    jobs_volume.commit()
    jobs[job_id] = {**jobs[job_id], "status": "failed", "error": f"{type(err).__name__}: {err}", "updated_at": time.time()}


@app.function(image=image, volumes=VOLUMES, gpu=GPU, timeout=30 * 60)
def render(job_id: str) -> None:
    """Step 3: one RGB view and 2D mask per candidate angle, then the pick of them.

    Writes `images/2d_render.png` and `images/2d_mask.png` — the view the caller inserts the
    object into, and the region to insert it in — and stops. No weights, no A100-scale work;
    just Blender. Resumes for free, since a second call finds both files already there.
    """
    work = Path(JOBS) / job_id
    images = work / "images"
    uploads = work / "inputs"

    def stage(name: str) -> None:
        jobs[job_id] = {**jobs[job_id], "stage": name, "updated_at": time.time()}
        print(f"[{job_id}] {name}", flush=True)

    started = time.time()
    try:
        # The uploads were written by the web container. A warm worker still holds the volume
        # as it was when it mounted, where this job's directory does not exist yet, and
        # Blender reports that as "Please select a file" rather than a missing path.
        jobs_volume.reload()

        record = jobs[job_id]
        view, seen = record.get("view"), record.get("cameras_seen")
        if not ((images / "2d_render.png").is_file() and (images / "2d_mask.png").is_file()):
            cameras = record["cameras"]
            stage(f"render {len(cameras)} view(s)")
            _run(
                [
                    "python", RENDERER,
                    "--source_model", str(uploads / "model.glb"),
                    "--mask_model", str(uploads / "mask.glb"),
                    "--output_dir", str(work),
                    "--cameras", json.dumps(cameras),
                ],
                "voxhammer_render",
            )
            index, seen = _pick_view(images, cameras)
            # Recorded before the check, so a job that saw nothing still says what it looked at.
            jobs[job_id] = {**jobs[job_id], "cameras_seen": seen}
            if index is None:
                raise RuntimeError(f"no candidate view shows the mask — is it sealed inside the anchor? {json.dumps(seen)}")

            shutil.copy(images / f"render_{index:04d}.png", images / "2d_render.png")
            shutil.copy(images / f"mask_{index:04d}.png", images / "2d_mask.png")
            view = seen[index]

        jobs_volume.commit()
        jobs[job_id] = {
            **jobs[job_id],
            "status": "done",
            "stage": "rendered",
            "view": view,
            "cameras_seen": seen,
            "files": ["images/2d_render.png", "images/2d_mask.png"],
            "seconds": round(time.time() - started, 1),
            "updated_at": time.time(),
        }
    except Exception as err:  # noqa: BLE001 — the message is the only thing the client can act on
        _fail(job_id, err)
        raise


@app.function(image=image, volumes=VOLUMES, secrets=[HF_SECRET], gpu=GPU, timeout=3 * 60 * 60)
def edit(job_id: str) -> None:
    """Steps 5-6: 150-view render, DINOv2 features, voxel mask, invert + re-denoise, decode.

    Expects `images/{2d_render,2d_mask,2d_edit}.png` — the first two from `render`, the third
    the caller's inserted-object image. Everything in `run_complete_pipeline` is VoxHammer's,
    unchanged; only `render_params` is passed, to keep its renderer from re-normalizing the
    meshes out of the frame the mask was built in.
    """
    import torch

    work = Path(JOBS) / job_id
    images = work / "images"
    uploads = work / "inputs"

    def stage(name: str) -> None:
        jobs[job_id] = {**jobs[job_id], "stage": name, "updated_at": time.time()}
        print(f"[{job_id}] {name}", flush=True)

    started = time.time()
    marks = {}

    def mark(name: str) -> None:
        marks[name] = round(time.time() - started, 1)

    try:
        # The edited image was written by the web container; pull its commit before reading.
        jobs_volume.reload()
        for required in ("2d_render.png", "2d_mask.png", "2d_edit.png"):
            if not (images / required).is_file():
                raise RuntimeError(f"missing images/{required} — call /render and supply the edited image first")
        _match_size(images / "2d_render.png", images / "2d_edit.png")

        stage("load pipeline")
        # Both the voxel masking step and TRELLIS's own configs reach for `assets/…` by
        # relative path, so the repo has to be the working directory, not just on the path.
        os.chdir(REPO)

        # Importing bpy is what puts Blender's bundled mathutils on the path — the same
        # ordering `voxhammer/bpy_render.py` relies on.
        import bpy  # noqa: F401
        from mathutils import Vector

        from inference import run_complete_pipeline
        from trellis.pipelines import TrellisImageTo3DPipeline

        pipeline = TrellisImageTo3DPipeline.from_pretrained(_prepare_trellis())
        pipeline.cuda()
        mark("load_pipeline")

        stage("render, features, voxel mask, edit")
        run_complete_pipeline(
            pipeline=pipeline,
            input_model_path=str(uploads / "model.glb"),
            mask_glb_path=str(uploads / "mask.glb"),
            render_dir=str(work / "render"),
            output_path=str(work / "output.glb"),
            image_dir=str(images),
            is_text=False,
            source_prompt="",
            target_prompt="",
            render_params={"scale": 1.0, "offset": Vector((0.0, 0.0, 0.0))},
        )
        mark("edit")

        del pipeline
        torch.cuda.empty_cache()

        landed = [name for name in ARTIFACTS if (work / name).is_file()]
        if "output.glb" not in landed:
            raise RuntimeError("the pipeline reported success but wrote no output.glb")

        jobs_volume.commit()
        jobs[job_id] = {
            **jobs[job_id],
            "status": "done",
            "stage": "done",
            "files": landed,
            "record": {
                "view": jobs[job_id].get("view"),
                "seconds": marks,
                "total_seconds": round(time.time() - started, 1),
            },
            "updated_at": time.time(),
        }
    except Exception as err:  # noqa: BLE001 — the message is the only thing the client can act on
        _fail(job_id, err)
        raise


@app.function(image=image, volumes=VOLUMES, secrets=[HF_SECRET], timeout=2 * 60 * 60)
def prefetch() -> dict:
    """`modal run modal/voxhammer.py::prefetch` — pull the weights before the first job.

    ~30 GB across three sources, all of which a cold container would otherwise download
    while holding an A100 idle.
    """
    from huggingface_hub import snapshot_download

    _prepare_trellis()
    snapshot_download(repo_id=FLUX_REPO)
    # DINOv2 comes through torch.hub, which caches under TORCH_HOME rather than HF's cache.
    import torch

    torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_reg", pretrained=True)
    cache.commit()
    return {"trellis": TRELLIS_DIR, "flux": FLUX_REPO}


@app.function(image=image, volumes=VOLUMES, timeout=900)
@modal.concurrent(max_inputs=8)
@modal.asgi_app()
def web():
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.responses import FileResponse

    api = FastAPI(title="dc-voxhammer")

    @api.get("/health")
    def health():
        return {"ok": True, "gpu": GPU, "artifacts": ARTIFACTS}

    @api.post("/render")
    async def start_render(
        model: UploadFile = File(...),
        mask: UploadFile = File(...),
        cameras: str = Form(...),
        sample: str = Form(""),
    ):
        """Step 3. Both meshes must already be in TRELLIS's cube — see the module docstring.

        `cameras` is the caller's ranked angles as JSON, best first, in the mask's own frame:
        elevation above the horizontal, azimuth around +Y from the model's +Z front.

        The job id is a hash of the meshes and those angles, so re-posting the same request
        attaches to the render already taken for it while a changed angle starts a new job
        rather than quietly reusing a view taken from somewhere else.
        """
        model_bytes = await model.read()
        mask_bytes = await mask.read()
        try:
            candidates = _cameras(cameras)
        except ValueError as err:
            raise HTTPException(status_code=400, detail=str(err)) from err

        canonical = json.dumps(candidates, sort_keys=True)
        digest = hashlib.sha256(model_bytes + mask_bytes + canonical.encode()).hexdigest()[:10]
        job_id = f"vh-{re.sub(r'[^a-z0-9-]+', '-', sample.lower()) or 'job'}-{digest}"

        # Admission is all blocking work — two 20 MB volume writes, a commit that covers
        # whatever the GPU workers are writing at the same time, and two RPCs. Eight requests
        # share this container's event loop, so running any of it inline stalls every other
        # upload in flight and a batch posted together walks past the client's deadline.
        def write_uploads() -> None:
            uploads = Path(JOBS) / job_id / "inputs"
            uploads.mkdir(parents=True, exist_ok=True)
            (uploads / "model.glb").write_bytes(model_bytes)
            (uploads / "mask.glb").write_bytes(mask_bytes)

        await asyncio.to_thread(write_uploads)
        # Committed before the spawn, not after: the worker mounts the volume as it starts.
        await jobs_volume.commit.aio()
        await jobs.put.aio(
            job_id,
            {
                "status": "pending",
                "stage": "queued",
                "sample": sample,
                "cameras": candidates,
                "created_at": time.time(),
            },
        )
        await render.spawn.aio(job_id)
        return {"job_id": job_id, "sample": sample}

    @api.post("/edit")
    async def start_edit(job_id: str = Form(...), edit_image: UploadFile = File(...)):
        """Steps 5-6 for a job `/render` already prepared, against the caller's edited view."""
        record = await jobs.get.aio(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"no job {job_id} — call /render first")

        images = Path(JOBS) / job_id / "images"
        edited = await edit_image.read()

        def write_edit() -> None:
            images.mkdir(parents=True, exist_ok=True)
            (images / "2d_edit.png").write_bytes(edited)

        await asyncio.to_thread(write_edit)
        await jobs_volume.commit.aio()
        await jobs.put.aio(job_id, {**record, "status": "pending", "stage": "queued-edit", "updated_at": time.time()})
        await edit.spawn.aio(job_id)
        return {"job_id": job_id}

    @api.get("/jobs/{job_id}")
    def status(job_id: str):
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail=f"no job {job_id}")
        return jobs[job_id]

    @api.get("/jobs/{job_id}/file/{name:path}")
    def file(job_id: str, name: str):
        jobs_volume.reload()
        # Resolved against the job's own directory, so a crafted name cannot climb out of it.
        target = (Path(JOBS) / job_id / name).resolve()
        if not str(target).startswith(str((Path(JOBS) / job_id).resolve())) or not target.is_file():
            raise HTTPException(status_code=404, detail=f"no {name} for {job_id}")
        return FileResponse(target)

    @api.delete("/jobs/{job_id}")
    def discard(job_id: str):
        """Jobs keep every intermediate, which is tens of megabytes each."""
        shutil.rmtree(Path(JOBS) / job_id, ignore_errors=True)
        jobs_volume.commit()
        jobs.pop(job_id, None)
        return {"discarded": job_id}

    return api
