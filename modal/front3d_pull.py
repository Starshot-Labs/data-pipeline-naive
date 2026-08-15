"""Copy normalized 3D-FRONT rooms off `3D-Front` into data/3d-front/.

    # the first 20 rooms of the full split
    python modal/front3d_pull.py

    python modal/front3d_pull.py --count 50
    python modal/front3d_pull.py --root 3D-FRONT-TEST-SCENE

`modal/front3d_clean.py` normalizes this volume in place: a directory per house still holds each
room's source component directory beside `<room>.glb` and `<room>_full.glb`, and every assembled
node is named exactly like its component filename, including `.glb`. Only the `_full` rooms are copied, and they keep
that name, because the two GLBs are not two encodings of one room. `_full` holds the same
furniture under the same `world` node at identical coordinates — the furniture's longest axis is
1.9 in both — plus optional `floor`, `wall`, `ceil`, and `others` nodes. Some architecture is the
whole house's slab rather than this room's: of the first twenty rooms, nineteen have architecture
reaching past the furniture's extent and one reaches eleven times as far.

Houses and rooms are taken in the volume's own listing order rather than sorted, so "the first
twenty" means the same thing here as it does in the dashboard. A room already on disk at its
volume byte count and already carrying normalized node names is left alone. Equal-sized legacy
files are refreshed, because normalization changes only bytes inside a fixed-size JSON chunk.
Rooms land flat, under their own names, and a run refuses to start if two of the
houses it walked hold one room name.
"""

import argparse
import json
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal

VOLUME = "3D-Front"
ROOT = "3D-FRONT-SCENE"
SUFFIX = "_full.glb"
DEST = Path(__file__).resolve().parent.parent / "data" / "3d-front"

LANES = 8


def normalized(data: bytes) -> bool:
    if len(data) < 20 or data[:4] != b"glTF":
        return False
    json_bytes, kind = struct.unpack_from("<I4s", data, 12)
    if kind != b"JSON" or len(data) < 20 + json_bytes:
        return False
    try:
        doc = json.loads(data[20 : 20 + json_bytes])
        scene = doc["scenes"][doc.get("scene", 0)]
        roots = scene["nodes"]
        if len(roots) != 1:
            return False
        children = doc["nodes"][roots[0]].get("children", [])
        for child in children:
            node = doc["nodes"][child]
            name = node.get("name", "")
            mesh_name = doc["meshes"][node["mesh"]].get("name", "")
            if not name.endswith(".glb") or name != mesh_name:
                return False
        return True
    except (KeyError, IndexError, TypeError, UnicodeDecodeError, json.JSONDecodeError):
        return False


def rooms(vol, root: str, count: int) -> list:
    """The first `count` `_full.glb` rooms under `root`, in the volume's own listing order.

    Houses are walked one at a time. A burst of listings trips the volume's rate limit, and a
    recursive listing is one call but arrives in an interleaved order that is neither the
    dashboard's nor stable, which would make "the first twenty" mean nothing.
    """
    found = []
    for house in vol.listdir(root):
        found += [entry for entry in vol.listdir(house.path) if entry.path.endswith(SUFFIX)]
        if len(found) >= count:
            break
    if not found:
        raise RuntimeError(
            f"{VOLUME}/{root} has no rooms; run modal/front3d_fetch.py first"
        )
    return found[:count]


def one_name_each(picked: list) -> None:
    """Every room lands in the same flat directory, so a shared name must not be silent."""
    seen = {}
    for entry in picked:
        name = Path(entry.path).name
        if name in seen:
            raise RuntimeError(f"{seen[name]} and {entry.path} would both land as {name}")
        seen[name] = entry.path


def pull(vol, entry, dest: Path) -> int:
    """One room into `dest` under its own name, as the bytes transferred to get it there."""
    target = dest / Path(entry.path).name
    label = f"{Path(entry.path).parent.name}/{target.name}"

    if target.exists() and target.stat().st_size == entry.size:
        if normalized(target.read_bytes()):
            print(f"  have {label}")
            return 0
        print(f"  stale {label}; refreshing names")

    partial = target.with_suffix(".part")
    with open(partial, "wb") as out:
        for chunk in vol.read_file(entry.path):
            out.write(chunk)

    written = partial.stat().st_size
    if written != entry.size:
        partial.unlink()
        raise RuntimeError(f"{entry.path} arrived {written} of {entry.size} bytes")
    if not normalized(partial.read_bytes()):
        partial.unlink()
        raise RuntimeError(f"{entry.path} is not normalized; run npm run clean-front3d first")

    partial.replace(target)
    print(f"  got  {label}  {entry.size / 1024 ** 2:.1f} MiB")
    return entry.size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--count", type=int, default=20, help="how many rooms to copy")
    parser.add_argument("--root", default=ROOT, help="the split to walk on the volume")
    parser.add_argument("--dest", type=Path, default=DEST, help="where the rooms land")
    args = parser.parse_args()

    args.dest.mkdir(parents=True, exist_ok=True)
    vol = modal.Volume.from_name(VOLUME)

    started = time.monotonic()
    picked = rooms(vol, args.root, args.count)
    one_name_each(picked)
    if len(picked) < args.count:
        print(f"{args.root} holds only {len(picked)} rooms")
    print(f"{len(picked)} rooms, {sum(entry.size for entry in picked) / 1024 ** 2:.0f} MiB")

    with ThreadPoolExecutor(max_workers=LANES) as pool:
        moved = list(pool.map(lambda entry: pull(vol, entry, args.dest), picked))

    copied = sum(1 for size in moved if size)
    print(
        f"{copied} copied, {len(picked) - copied} already there, "
        f"{sum(moved) / 1024 ** 2:.0f} MiB in {time.monotonic() - started:.0f}s "
        f"-> {args.dest}"
    )


if __name__ == "__main__":
    main()
