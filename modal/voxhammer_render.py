"""VoxHammer's `utils/render_rgb_and_mask.py` with the camera ring made an argument.

Upstream fixes all five of its views at elevation 15 and varies only azimuth, so a region
that opens upward — inside a mug, a drawer, a box — sits behind the anchor's own wall in
every one of them and the depth-difference mask comes back empty. Nothing else about the
render needs to change, so nothing else here does: `process` is imported and called, and
only `render_model`, the function that places the cameras, is ours.

    python voxhammer_render.py --source_model model.glb --mask_model mask.glb \
        --output_dir /jobs/<id> --cameras '[{"elevation": 70, "azimuth": 0}]'

Angles are in the mask's frame rather than Blender's: +Y up and +Z the model's front, the
axes `pipeline/mask.mjs` builds the box in. `_matrices` is the only place that converts.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, "/voxhammer/utils")

import render_rgb_and_mask as upstream
from bpyrenderer import SceneManager
from bpyrenderer.camera import add_camera
from bpyrenderer.camera.layout import get_camera_positions_on_sphere
from bpyrenderer.engine import init_render_engine
from bpyrenderer.environment import set_env_map
from bpyrenderer.importer import load_file
from bpyrenderer.render_output import enable_color_output, enable_depth_output

RADIUS = 1.8
RESOLUTION = 1024
ENV_MAP = "assets/preset/brown_photostudio_02_1k.exr"


def _matrices(cameras):
    """One camera matrix per angle, in the order given.

    Asked for one at a time because `get_camera_positions_on_sphere` takes a list of
    elevations and a list of azimuths, and how it pairs them is its business, not ours.

    The -90 is the glTF-to-Blender turn. Blender counts azimuth from +X toward +Y with +Z
    up, while the cube the mask is authored in is glTF's — +Y up, +Z front — which puts the
    model's front at Blender's -Y. Upstream shifts its own five by the same amount.
    """
    matrices = []
    for camera in cameras:
        _, mats, _, _ = get_camera_positions_on_sphere(
            center=(0, 0, 0),
            radius=RADIUS,
            elevations=[float(camera["elevation"])],
            azimuths=[float(camera["azimuth"]) - 90.0],
        )
        matrices.append(mats[0])
    return matrices


def render_model(input_model, output_dir, cameras):
    """Upstream's `render_model`, with `cameras` where its hardcoded ring used to be."""
    init_render_engine("CYCLES")
    scene_manager = SceneManager()
    scene_manager.clear(reset_keyframes=True)

    load_file(input_model)

    scene_manager.smooth()
    scene_manager.clear_normal_map()
    scene_manager.set_material_transparency(False)
    # Upstream's comment: important for rendering normals, but may cause render errors.
    scene_manager.set_materials_opaque()
    set_env_map(ENV_MAP)

    matrices = _matrices(cameras)
    for i, matrix in enumerate(matrices):
        add_camera(matrix, "PERSP", add_frame=i < len(matrices) - 1)

    enable_color_output(
        RESOLUTION,
        RESOLUTION,
        output_dir,
        file_format="PNG",
        mode="IMAGE",
        film_transparent=True,
    )
    enable_depth_output(output_dir)
    scene_manager.render()

    # `process` copies this into the image directory unconditionally, so it has to exist.
    meta = {
        "width": RESOLUTION,
        "height": RESOLUTION,
        "locations": [
            {"index": f"{i:04d}", "elevation": camera["elevation"], "azimuth": camera["azimuth"]}
            for i, camera in enumerate(cameras)
        ],
    }
    with open(os.path.join(output_dir, "meta.json"), "w") as handle:
        json.dump(meta, handle, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_model", required=True)
    parser.add_argument("--mask_model", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cameras", required=True, help='[{"elevation": 70, "azimuth": 0}, ...]')
    args = parser.parse_args()

    cameras = json.loads(args.cameras)
    if not cameras:
        raise SystemExit("--cameras is empty")

    # The checkout is cloned unpinned, so fail loudly on a rename rather than silently
    # rendering upstream's own five views.
    for attribute in ("render_model", "process"):
        if not hasattr(upstream, attribute):
            raise SystemExit(f"VoxHammer's render_rgb_and_mask has no {attribute}")

    upstream.render_model = lambda model, output: render_model(model, output, cameras)

    os.makedirs(args.output_dir, exist_ok=True)
    upstream.process(args.source_model, args.mask_model, args.output_dir)
    print(f"rendered {len(cameras)} view(s)", flush=True)
