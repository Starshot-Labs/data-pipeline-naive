"""Export X Real2Sim scenes from the `InternScenes` Modal volume, then download locally.

    python modal/internscenes_client.py 5 --out data/internscenes
    python modal/internscenes_client.py 3rscan/<scan-id> scannet/scene0000_00 --out data/internscenes
    python modal/internscenes_client.py 20 --seed 7 --out data/internscenes

An integer selects that many deterministic random scenes. Explicit IDs select exactly those
scenes. Each bundle contains the original layout JSON, complete StructureMesh directory,
only unique GLBs referenced by `model_uid`, the two orientation metadata files required by
the official composer, and a manifest. Repository-relative paths are preserved.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import uuid
from pathlib import Path

import modal

APP_NAME = "dc-internscenes-export"
VOLUME_NAME = "InternScenes"
VOL = Path("/vol")
volume = modal.Volume.from_name(VOLUME_NAME, version=2)
app = modal.App(APP_NAME)
image = modal.Image.debian_slim(python_version="3.12")
GLOBAL_METADATA = (
    "asset_library/uid_2_angle.json",
    "asset_library/uid_2_origin_cate.json",
)


def _safe_scene_id(scene_id: str) -> str:
    path = Path(scene_id)
    if path.is_absolute() or len(path.parts) < 2 or ".." in path.parts or "\\" in scene_id:
        raise ValueError(f"invalid scene id {scene_id!r}; expected dataset/.../scan_id")
    return "/".join(path.parts)


@app.function(image=image, volumes={str(VOL): volume}, timeout=6 * 3600)
def export(scene_ids: list[str], count: int = 0, seed: int = 0) -> dict:
    index_path = VOL / "manifests" / "real2sim-index.json"
    if not index_path.is_file():
        raise RuntimeError("InternScenes is not prepared; run modal/internscenes_fetch.py --prepare")
    index = json.loads(index_path.read_text())
    available = sorted(index["scenes"])

    if count:
        if count > len(available):
            raise ValueError(f"requested {count} scenes, only {len(available)} are available")
        scene_ids = sorted(random.Random(seed).sample(available, count))
    else:
        scene_ids = [_safe_scene_id(scene_id) for scene_id in scene_ids]
    unknown = sorted(set(scene_ids) - set(available))
    if unknown:
        raise ValueError(f"unknown scene ids: {unknown}")

    export_id = uuid.uuid4().hex
    temporary = VOL / "exports" / f".{export_id}.tmp"
    finished = VOL / "exports" / export_id
    source = VOL / "extracted" / "real2sim"
    temporary.mkdir(parents=True)
    files = set()
    assets = {}

    for scene_id in scene_ids:
        scene = index["scenes"][scene_id]
        relative_paths = [scene["layout"], *scene["structure"]]
        for item in scene["assets"]:
            relative_paths.append(item["path"])
            assets[item["uid"]] = item["path"]
        for relative in relative_paths:
            files.add(relative)

    files.update(GLOBAL_METADATA)
    copied = []
    for relative in sorted(files):
        src = source / relative
        if not src.is_file():
            raise RuntimeError(f"indexed dependency is missing: {relative}")
        dst = temporary / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append({"path": relative, "bytes": src.stat().st_size})

    manifest = {
        "repo": index["repo"],
        "commit": index["commit"],
        "scenes": scene_ids,
        "assets": assets,
        "files": copied,
        "bytes": sum(item["bytes"] for item in copied),
    }
    (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2))
    temporary.replace(finished)
    volume.commit()
    return {"export_id": export_id, **manifest, "file_count": len(copied) + 1}


def _download(export_id: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{export_id}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()

    remote_root = f"exports/{export_id}"
    entries = volume.listdir(remote_root, recursive=True)
    files = [entry for entry in entries if entry.type.name == "FILE"]
    if not files:
        raise RuntimeError(f"Modal export {export_id} is empty")

    for index, entry in enumerate(files, 1):
        relative = Path(entry.path).relative_to(remote_root)
        local = temporary / relative
        local.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {index}/{len(files)}: {relative} ({entry.size / 1024**2:.1f} MiB)")
        with open(local, "wb") as output:
            volume.read_file_into_fileobj(entry.path, output)
        if local.stat().st_size != entry.size:
            raise RuntimeError(f"downloaded file is wrong-sized: {relative}")

    manifest_path = temporary / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("downloaded export has no manifest.json")
    manifest = json.loads(manifest_path.read_text())
    for item in manifest["files"]:
        path = temporary / item["path"]
        if not path.is_file() or path.stat().st_size != item["bytes"]:
            raise RuntimeError(f"downloaded file is missing/wrong-sized: {item['path']}")
    if destination.exists():
        raise FileExistsError(destination)
    temporary.replace(destination)
    return destination


@app.local_entrypoint()
def main():
    pass


def cli() -> None:
    parser = argparse.ArgumentParser(description="Download selected InternScenes scenes and only their assets.")
    parser.add_argument("selection", nargs="+", help="one integer count, or explicit dataset/scan_id values")
    parser.add_argument("--seed", type=int, default=0, help="random seed when selection is a count")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    count, scene_ids = 0, []
    if len(args.selection) == 1:
        try:
            count = int(args.selection[0])
        except ValueError:
            scene_ids = args.selection
        else:
            if count < 1:
                parser.error("scene count must be positive")
    else:
        scene_ids = args.selection

    with app.run(detach=False):
        result = export.remote(scene_ids, count, args.seed)
    destination = args.out / result["export_id"]
    _download(result["export_id"], destination)
    print(json.dumps({"destination": str(destination), **result}, indent=2))


if __name__ == "__main__":
    try:
        cli()
    except (OSError, RuntimeError, ValueError) as err:
        sys.exit(f"error: {err}")


