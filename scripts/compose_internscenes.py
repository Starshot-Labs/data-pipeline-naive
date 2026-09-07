"""Compose downloaded InternScenes bundles with the authors' own composer.

    python scripts/compose_internscenes.py data/internscenes/<export-id>
    python scripts/compose_internscenes.py data/internscenes/<export-id> scannet/scene0000_00

The composition code is the untouched upstream file vendored at
scripts/internscenes/compose_scenes.py (InternRobotics/InternScenes, commit a3614c24). This
wrapper only points that module's directories at the bundle and copies each scene's
StructureMesh where `compose_one_scene` expects it — the output folder, the same step the
official tutorial performs by hand. Results land next to the bundle's source data:

    <bundle>/composed/<scene-id>/glb_scene.glb

Requires: trimesh, numpy, pillow, tqdm.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "internscenes"))
import compose_scenes


def main() -> None:
    parser = argparse.ArgumentParser(description="Compose scenes from an InternScenes export bundle.")
    parser.add_argument("bundle", type=Path, help="a folder downloaded by modal/internscenes_client.py")
    parser.add_argument("scenes", nargs="*", help="scene ids; defaults to every scene in the manifest")
    parser.add_argument("--no-texture", action="store_true")
    args = parser.parse_args()

    bundle = args.bundle.resolve()
    manifest = json.loads((bundle / "manifest.json").read_text())
    scenes = args.scenes or manifest["scenes"]
    unknown = sorted(set(scenes) - set(manifest["scenes"]))
    if unknown:
        raise SystemExit(f"not in this bundle: {unknown}")

    composed = bundle / "composed"
    compose_scenes.ASSET_LIBRARY_FOLDER = str(bundle / "asset_library")
    compose_scenes.SCENE_INFO_DIR = str(bundle / "Layout_info")
    compose_scenes.SCENE_SAVE_DIR = str(composed)
    composer = compose_scenes.SceneComposer()

    for index, scene in enumerate(scenes, 1):
        print(f"[{index}/{len(scenes)}] {scene}")
        structure = composed / scene / "StructureMesh"
        structure.parent.mkdir(parents=True, exist_ok=True)
        if not structure.exists():
            shutil.copytree(bundle / "Layout_info" / scene / "StructureMesh", structure)
        composer.compose_one_scene(
            scene,
            use_texture=not args.no_texture,
            add_floor=True,
            add_wall=True,
            add_ceiling=True,
        )


if __name__ == "__main__":
    main()
