"""Mirror and index gated InternScenes on the `InternScenes` Modal Volume v2.

    modal run --detach modal/internscenes_fetch.py --mirror
    modal run --detach modal/internscenes_fetch.py --prepare
    modal run --detach modal/internscenes_fetch.py --mirror --prepare

`mirror` pins the current Hugging Face commit and reproduces every repository path under
`/vol/mirror/<commit>/`. `prepare` extracts the Real2Sim layouts and the two archived asset
libraries needed for per-scene dependency exports, then builds exact scene and asset indexes.
The immutable mirror is retained: derived extraction never substitutes for upstream bytes.

The repository is gated. `huggingface-2` must contain `HUGGINGFACE_ACCESS_KEY` from an
account that accepted the InternScenes license. The token is passed explicitly; it is never
written to the volume or repository.
"""

from __future__ import annotations

import json
import os
import shutil
import tarfile
import time
from pathlib import Path

import modal

app = modal.App("dc-internscenes")
VOL = Path("/vol")
REPO = "InternRobotics/InternScenes"
REVISION = "808c1d0669904b6c775189607afdc39011c87728"
MIRROR_WORKERS = 8
MIRROR_ATTEMPTS = 20
EXPECTED_REAL2SIM_SCENES = 9583
HF_SECRET = modal.Secret.from_name("huggingface-2")
volume = modal.Volume.from_name("InternScenes", version=2)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface_hub[hf_xet]")
    .env({"PYTHONUNBUFFERED": "1", "HF_HUB_DISABLE_XET": "1"})
)

REAL2SIM_ARCHIVES = {
    "Layout_info.tar.gz": "real2sim",
    "asset_library/objaverse": "real2sim/asset_library/objaverse",
    "asset_library/partnet_mobility/partnet_mobility.tar.gz": "real2sim/asset_library/partnet_mobility",
}
GLOBAL_METADATA = (
    "asset_library/uid_2_angle.json",
    "asset_library/uid_2_origin_cate.json",
)
LOOSE_ASSET_PREFIXES = (
    "3D-FUTURE-model",
    "gen_assets",
    "gr100",
    "hssd-models",
    "objaverse_old",
)


def _token() -> str:
    token = os.environ.get("HUGGINGFACE_ACCESS_KEY")
    if not token:
        raise RuntimeError("huggingface-2 has no HUGGINGFACE_ACCESS_KEY")
    return token


def _meta(token: str) -> dict:
    from huggingface_hub import HfApi

    info = HfApi(token=token).dataset_info(REPO, revision=REVISION, files_metadata=True)
    files = []
    for sibling in info.siblings:
        size = sibling.size
        if size is None:
            raise RuntimeError(f"the Hub returned no size for {sibling.rfilename}")
        files.append({"path": sibling.rfilename, "bytes": size, "blob_id": sibling.blob_id})
    return {"repo": REPO, "commit": info.sha, "files": files, "bytes": sum(x["bytes"] for x in files)}


def _atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(path)


def _safe_members(tar: tarfile.TarFile):
    for member in tar:
        path = Path(member.name)
        if path.is_absolute() or ".." in path.parts or member.isdev():
            raise RuntimeError(f"unsafe archive member {member.name!r}")
        if member.issym() or member.islnk():
            raise RuntimeError(f"archive link is not allowed: {member.name!r}")
        yield member


def _extract_tar(path: Path, destination: Path) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    files = bytes_written = 0
    with tarfile.open(path, "r:gz") as tar:
        for member in _safe_members(tar):
            target = destination / member.name
            if member.isfile() and target.exists() and target.stat().st_size == member.size:
                continue
            tar.extract(member, destination, filter="data")
            if member.isfile():
                files += 1
                bytes_written += member.size
    return {"files_written": files, "bytes_written": bytes_written}


def _joined_extract(parts: list[Path], destination: Path) -> dict:
    class Joined:
        def __init__(self, paths):
            self.paths = iter(paths)
            self.current = None

        def read(self, size=-1):
            chunks, remaining = [], size
            while remaining != 0:
                if self.current is None:
                    try:
                        self.current = open(next(self.paths), "rb")
                    except StopIteration:
                        break
                chunk = self.current.read(remaining)
                if chunk:
                    chunks.append(chunk)
                    if remaining > 0:
                        remaining -= len(chunk)
                else:
                    self.current.close()
                    self.current = None
            return b"".join(chunks)

        def close(self):
            if self.current:
                self.current.close()

    source = Joined(parts)
    destination.mkdir(parents=True, exist_ok=True)
    files = bytes_written = 0
    try:
        with tarfile.open(fileobj=source, mode="r|gz") as tar:
            for member in _safe_members(tar):
                target = destination / member.name
                if member.isfile() and target.exists() and target.stat().st_size == member.size:
                    continue
                tar.extract(member, destination, filter="data")
                if member.isfile():
                    files += 1
                    bytes_written += member.size
    finally:
        source.close()
    return {"files_written": files, "bytes_written": bytes_written}


@app.function(
    image=image,
    secrets=[HF_SECRET],
    volumes={str(VOL): volume},
    cpu=8,
    memory=32768,
    timeout=24 * 3600,
)
def mirror() -> dict:
    from huggingface_hub import snapshot_download

    token = _token()
    manifest = _meta(token)
    destination = VOL / "mirror" / manifest["commit"]
    destination.mkdir(parents=True, exist_ok=True)
    print(f"mirroring {len(manifest['files'])} files, {manifest['bytes'] / 10**12:.3f} TB at {manifest['commit']}")
    for attempt in range(1, MIRROR_ATTEMPTS + 1):
        try:
            snapshot_download(
                repo_id=REPO,
                repo_type="dataset",
                revision=manifest["commit"],
                local_dir=destination,
                token=token,
                max_workers=MIRROR_WORKERS,
            )
            break
        except ConnectionError as err:
            if attempt == MIRROR_ATTEMPTS:
                raise
            delay = min(60 * 2 ** (attempt - 1), 900)
            print(f"mirror attempt {attempt}/{MIRROR_ATTEMPTS} hit a network/rate-limit error; retrying in {delay}s: {err}")
            time.sleep(delay)
    wrong = []
    for item in manifest["files"]:
        path = destination / item["path"]
        if not path.is_file() or path.stat().st_size != item["bytes"]:
            wrong.append(item["path"])
    if wrong:
        raise RuntimeError(f"{len(wrong)} mirrored files missing/wrong-sized: {wrong[:20]}")
    _atomic_json(VOL / "manifests" / "repo.json", manifest)
    volume.commit()
    return {"commit": manifest["commit"], "files": len(manifest["files"]), "bytes": manifest["bytes"]}


def _mirror_root() -> tuple[Path, dict]:
    manifest_path = VOL / "manifests" / "repo.json"
    if not manifest_path.is_file():
        raise RuntimeError("mirror first: /vol/manifests/repo.json is missing")
    manifest = json.loads(manifest_path.read_text())
    return VOL / "mirror" / manifest["commit"], manifest


def _copy_loose_assets(mirror_root: Path, extracted: Path) -> None:
    source_root = mirror_root / "asset_library"
    target_root = extracted / "real2sim" / "asset_library"
    for prefix in LOOSE_ASSET_PREFIXES:
        source = source_root / prefix
        if not source.exists():
            raise RuntimeError(f"missing mirrored asset library {source}")
        target = target_root / prefix
        if target.exists():
            continue
        shutil.copytree(source, target, copy_function=os.link)
    for relative in GLOBAL_METADATA:
        source = mirror_root / relative
        target = extracted / "real2sim" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            os.link(source, target)


def _normalize_archive_root(destination: Path, root_name: str) -> None:
    nested = destination / root_name
    if not nested.is_dir():
        return
    siblings = [path for path in destination.iterdir() if path != nested]
    if siblings:
        raise RuntimeError(f"cannot normalize {destination}: unexpected siblings {siblings[:10]}")
    temporary = destination.with_name(f".{destination.name}.unwrap")
    if temporary.exists():
        raise RuntimeError(f"stale normalization directory: {temporary}")
    destination.replace(temporary)
    (temporary / root_name).replace(destination)
    temporary.rmdir()

def _build_index(extracted: Path, commit: str) -> dict:
    real = extracted / "real2sim"
    layout_root = real / "Layout_info"
    scenes = {}
    dependencies = {}
    unknown = set()
    missing = []

    for layout_path in sorted(layout_root.rglob("layout.json")):
        scene_id = layout_path.parent.relative_to(layout_root).as_posix()
        objects = json.loads(layout_path.read_text())
        uids = sorted({str(obj.get("model_uid", "")) for obj in objects if obj.get("model_uid")})
        resolved = []
        for uid in uids:
            prefix = uid.split("/", 1)[0]
            if prefix == "partnet_mobility":
                relative = f"asset_library/{uid}/whole.glb"
            elif prefix in LOOSE_ASSET_PREFIXES or prefix == "objaverse":
                relative = f"asset_library/{uid}.glb"
            else:
                unknown.add(prefix)
                continue
            if not (real / relative).is_file():
                missing.append({"scene": scene_id, "uid": uid, "path": relative})
            resolved.append({"uid": uid, "path": relative})
        structure = sorted(
            str(path.relative_to(real))
            for path in (layout_path.parent / "StructureMesh").glob("*")
            if path.is_file()
        )
        scenes[scene_id] = {
            "layout": str(layout_path.relative_to(real)),
            "structure": structure,
            "objects": len(objects),
            "assets": resolved,
        }
        for item in resolved:
            dependencies[item["uid"]] = item["path"]

    if unknown:
        raise RuntimeError(f"unknown model_uid prefixes: {sorted(unknown)}")
    if missing:
        _atomic_json(VOL / "manifests" / "missing-assets.json", {"missing": missing})
        raise RuntimeError(f"{len(missing)} unresolved model assets; see missing-assets.json")
    if len(scenes) != EXPECTED_REAL2SIM_SCENES:
        raise RuntimeError(f"indexed {len(scenes)} Real2Sim scenes, expected {EXPECTED_REAL2SIM_SCENES}")
    return {"repo": REPO, "commit": commit, "scenes": scenes, "assets": dependencies}


@app.function(
    image=image,
    volumes={str(VOL): volume},
    cpu=8,
    memory=32768,
    timeout=24 * 3600,
)
def prepare() -> dict:
    mirror_root, manifest = _mirror_root()
    extracted = VOL / "extracted"
    complete = VOL / "manifests" / "real2sim-index.json"
    if complete.is_file():
        index = json.loads(complete.read_text())
        if index.get("commit") == manifest["commit"] and len(index.get("scenes", {})) == EXPECTED_REAL2SIM_SCENES:
            return {"commit": index["commit"], "scenes": len(index["scenes"]), "assets": len(index["assets"])}

    print("extracting Real2Sim layouts")
    _extract_tar(mirror_root / "Layout_info.tar.gz", extracted / "real2sim")
    _copy_loose_assets(mirror_root, extracted)

    objaverse_parts = sorted((mirror_root / "asset_library" / "objaverse").glob("objaverse.tar.gz.*"))
    if not objaverse_parts:
        raise RuntimeError("no Objaverse archive parts found")
    print(f"extracting Objaverse from {len(objaverse_parts)} parts")
    objaverse_root = extracted / "real2sim" / "asset_library" / "objaverse"
    _normalize_archive_root(objaverse_root, "objaverse")
    if not next(objaverse_root.glob("*.glb"), None):
        _joined_extract(objaverse_parts, objaverse_root)
        _normalize_archive_root(objaverse_root, "objaverse")

    print("extracting PartNet-Mobility")
    partnet_root = extracted / "real2sim" / "asset_library" / "partnet_mobility"
    _normalize_archive_root(partnet_root, "partnet_mobility")
    if not next(partnet_root.glob("*/whole.glb"), None):
        _extract_tar(
            mirror_root / "asset_library" / "partnet_mobility" / "partnet_mobility.tar.gz",
            partnet_root,
        )
        _normalize_archive_root(partnet_root, "partnet_mobility")



    index = _build_index(extracted, manifest["commit"])
    _atomic_json(complete, index)
    volume.commit()
    return {"commit": index["commit"], "scenes": len(index["scenes"]), "assets": len(index["assets"])}


@app.local_entrypoint()
def main(mirror_data: bool = False, prepare_data: bool = False):
    if not mirror_data and not prepare_data:
        raise SystemExit("pass --mirror, --prepare, or both")
    result = {}
    if mirror_data:
        result["mirror"] = mirror.remote()
    if prepare_data:
        result["prepare"] = prepare.remote()
    print(json.dumps(result, indent=2))






