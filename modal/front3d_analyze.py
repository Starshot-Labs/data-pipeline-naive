"""Analyze structural-component combinations and the geometry inside every `others.glb`.

This is read-only. The complete report is written locally to `data/3d-front-analysis.json`:

    python -X utf8 -m modal run modal/front3d_analyze.py

Presence is exact from the four filenames in each room directory. `others.glb` is classified from
its triangles, not its name: area by face orientation and by normalized room height, planar area
near the floor and ceiling reference elevations, vertical area spanning the room height, and the
same metrics for `floor.glb`, `ceil.glb`, and `wall.glb` when present. The report retains the raw
continuous metrics for every room; categorical labels are an interpretable summary and remain
explicitly marked as geometric inference rather than source semantics.
"""

from __future__ import annotations

import json
import math
import struct
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal

app = modal.App("dc-front3d-analyze")

VOLUME = "3D-Front"
VOL = "/vol"
SPLIT = "3D-FRONT-SCENE"
GENERICS = ("floor", "wall", "ceil", "others")
ROOM_WORKERS = 4
MAX_CONTAINERS = 32
EPSILON = 1e-12

image = modal.Image.debian_slim(python_version="3.12").env({"PYTHONUNBUFFERED": "1"})
source = modal.Volume.from_name(VOLUME)
volume = source.read_only()

COMPONENT = {
    5120: (1, "b"),
    5121: (1, "B"),
    5122: (2, "h"),
    5123: (2, "H"),
    5125: (4, "I"),
    5126: (4, "f"),
}
COMPONENTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


def _glb(path: Path) -> tuple[dict, memoryview]:
    data = path.read_bytes()
    if len(data) < 28 or data[:4] != b"glTF":
        raise ValueError(f"{path} is not GLB")
    json_bytes, kind = struct.unpack_from("<I4s", data, 12)
    if kind != b"JSON":
        raise ValueError(f"{path} has no leading JSON chunk")
    document = json.loads(data[20 : 20 + json_bytes])
    binary_header = 20 + json_bytes
    binary_bytes, binary_kind = struct.unpack_from("<I4s", data, binary_header)
    if binary_kind != b"BIN\0":
        raise ValueError(f"{path} has no binary chunk")
    start = binary_header + 8
    return document, memoryview(data)[start : start + binary_bytes]


def _accessor(document: dict, binary: memoryview, index: int):
    accessor = document["accessors"][index]
    view = document["bufferViews"][accessor["bufferView"]]
    if accessor.get("sparse"):
        raise ValueError("sparse accessors are unsupported")
    width, code = COMPONENT[accessor["componentType"]]
    components = COMPONENTS[accessor["type"]]
    stride = view.get("byteStride", width * components)
    offset = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    unpack = struct.Struct(f"<{components}{code}").unpack_from
    return [unpack(binary, offset + row * stride) for row in range(accessor["count"])]


def _triangles(path: Path):
    document, binary = _glb(path)
    for mesh in document.get("meshes", []):
        for primitive in mesh.get("primitives", []):
            if (primitive.get("mode", 4)) != 4 or "POSITION" not in primitive.get("attributes", {}):
                continue
            positions = _accessor(document, binary, primitive["attributes"]["POSITION"])
            indices = (
                [row[0] for row in _accessor(document, binary, primitive["indices"])]
                if "indices" in primitive
                else range(len(positions))
            )
            indices = list(indices)
            for at in range(0, len(indices) - 2, 3):
                yield positions[indices[at]], positions[indices[at + 1]], positions[indices[at + 2]]


def _raw_profile(path: Path) -> dict:
    total = horizontal = vertical = sloped = 0.0
    low_horizontal = high_horizontal = middle_horizontal = 0.0
    low_vertical = full_vertical = high_vertical = 0.0
    minimum = [math.inf, math.inf, math.inf]
    maximum = [-math.inf, -math.inf, -math.inf]
    records = []
    triangles = 0

    for a, b, c in _triangles(path):
        for point in (a, b, c):
            for axis in range(3):
                minimum[axis] = min(minimum[axis], point[axis])
                maximum[axis] = max(maximum[axis], point[axis])
        ab = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
        ac = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
        cross = (
            ab[1] * ac[2] - ab[2] * ac[1],
            ab[2] * ac[0] - ab[0] * ac[2],
            ab[0] * ac[1] - ab[1] * ac[0],
        )
        double_area = math.sqrt(sum(value * value for value in cross))
        if double_area <= EPSILON:
            continue
        area = double_area / 2
        normal_y = abs(cross[1]) / double_area
        center_y = (a[1] + b[1] + c[1]) / 3
        y_min = min(a[1], b[1], c[1])
        y_max = max(a[1], b[1], c[1])
        records.append((area, normal_y, center_y, y_min, y_max))
        total += area
        triangles += 1

    span = [maximum[axis] - minimum[axis] for axis in range(3)]
    height = max(span[1], EPSILON)
    for area, normal_y, center_y, y_min, y_max in records:
        level = (center_y - minimum[1]) / height
        vertical_span = (y_max - y_min) / height
        if normal_y >= 0.9:
            horizontal += area
            if level <= 0.15:
                low_horizontal += area
            elif level >= 0.85:
                high_horizontal += area
            else:
                middle_horizontal += area
        elif normal_y <= 0.1:
            vertical += area
            if level <= 0.25:
                low_vertical += area
            if level >= 0.75:
                high_vertical += area
            if vertical_span >= 0.5:
                full_vertical += area
        else:
            sloped += area

    return {
        "triangles": triangles,
        "bounds_min": minimum,
        "bounds_max": maximum,
        "span": span,
        "area": total,
        "horizontal_area": horizontal,
        "vertical_area": vertical,
        "sloped_area": sloped,
        "low_horizontal_area": low_horizontal,
        "high_horizontal_area": high_horizontal,
        "middle_horizontal_area": middle_horizontal,
        "low_vertical_area": low_vertical,
        "high_vertical_area": high_vertical,
        "full_vertical_area": full_vertical,
    }


def _plane_level(profile: dict) -> float | None:
    if not profile or not profile["triangles"]:
        return None
    return (profile["bounds_min"][1] + profile["bounds_max"][1]) / 2


def _profile(path: Path, floor_level: float | None, ceil_level: float | None, room_min: float, room_max: float) -> dict:
    profile = _raw_profile(path)
    height = max(room_max - room_min, EPSILON)
    tolerance = max(height * 0.03, 0.002)
    floor_band = ceil_band = 0.0
    vertical_room_band = 0.0

    for a, b, c in _triangles(path):
        ab = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
        ac = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
        cross = (
            ab[1] * ac[2] - ab[2] * ac[1],
            ab[2] * ac[0] - ab[0] * ac[2],
            ab[0] * ac[1] - ab[1] * ac[0],
        )
        double_area = math.sqrt(sum(value * value for value in cross))
        if double_area <= EPSILON:
            continue
        area = double_area / 2
        normal_y = abs(cross[1]) / double_area
        center_y = (a[1] + b[1] + c[1]) / 3
        y_min, y_max = min(a[1], b[1], c[1]), max(a[1], b[1], c[1])
        if normal_y >= 0.9:
            if floor_level is not None and abs(center_y - floor_level) <= tolerance:
                floor_band += area
            if ceil_level is not None and abs(center_y - ceil_level) <= tolerance:
                ceil_band += area
        elif normal_y <= 0.1 and y_min <= room_min + 0.25 * height and y_max >= room_max - 0.25 * height:
            vertical_room_band += area

    profile.update(
        {
            "floor_reference_area": floor_band,
            "ceil_reference_area": ceil_band,
            "room_spanning_vertical_area": vertical_room_band,
        }
    )
    return profile


def _ratio(value: float, total: float) -> float:
    return value / total if total > EPSILON else 0.0


def _classify(profile: dict, explicit: set[str]) -> dict:
    area = profile["area"]
    horizontal = _ratio(profile["horizontal_area"], area)
    vertical = _ratio(profile["vertical_area"], area)
    floor_signal = max(
        _ratio(profile["floor_reference_area"], area),
        _ratio(profile["low_horizontal_area"], area),
    )
    ceil_signal = max(
        _ratio(profile["ceil_reference_area"], area),
        _ratio(profile["high_horizontal_area"], area),
    )
    wall_signal = max(
        _ratio(profile["room_spanning_vertical_area"], area),
        _ratio(profile["full_vertical_area"], area),
    )

    signals = {"floor": floor_signal, "ceil": ceil_signal, "wall": wall_signal}
    dominant_roles = [role for role in ("floor", "wall", "ceil") if signals[role] >= 0.35]
    substantial_roles = [role for role in ("floor", "wall", "ceil") if signals[role] >= 0.2]
    dominant_label = "+".join(dominant_roles) if dominant_roles else "unresolved_structural_or_decorative"
    composition_label = "+".join(substantial_roles) if substantial_roles else "unresolved_structural_or_decorative"

    ranked = sorted(signals.items(), key=lambda item: item[1], reverse=True)
    margin = ranked[0][1] - ranked[1][1]
    confidence = "high" if ranked[0][1] >= 0.5 and margin >= 0.15 else "medium" if dominant_roles else "low"

    return {
        "label": dominant_label,
        "inferred_roles": dominant_roles,
        "composition_label": composition_label,
        "substantial_roles": substantial_roles,
        "confidence": confidence,
        "signals": {
            "floor": floor_signal,
            "ceil": ceil_signal,
            "wall": wall_signal,
            "horizontal": horizontal,
            "vertical": vertical,
            "sloped": _ratio(profile["sloped_area"], area),
        },
    }


def _analyze_room(split: str, house: str, room_name: str) -> dict:
    room = Path(VOL) / split / house / room_name
    present = {path.stem for path in room.iterdir() if path.is_file() and path.name in {f"{name}.glb" for name in GENERICS}}
    combo = "+".join(name for name in GENERICS if name in present) or "none"
    result = {"house": house, "room": room_name, "combination": combo, "present": sorted(present)}
    if "others" not in present:
        return result

    raw = {name: _raw_profile(room / f"{name}.glb") for name in present}
    floor_level = _plane_level(raw.get("floor"))
    ceil_level = _plane_level(raw.get("ceil"))
    structural_mins = [profile["bounds_min"][1] for profile in raw.values()]
    structural_maxs = [profile["bounds_max"][1] for profile in raw.values()]
    room_min, room_max = min(structural_mins), max(structural_maxs)
    others = _profile(room / "others.glb", floor_level, ceil_level, room_min, room_max)
    classification = _classify(others, present - {"others"})
    result.update(
        {
            "others": {
                "classification": classification,
                "geometry": others,
                "floor_level": floor_level,
                "ceil_level": ceil_level,
                "room_y_range": [room_min, room_max],
            }
        }
    )
    return result


def _house_rooms(split: str, house: str) -> list[str]:
    root = Path(VOL) / split / house
    return sorted(path.name for path in root.iterdir() if path.is_dir())


@app.function(
    image=image,
    volumes={VOL: volume},
    cpu=8,
    memory=4096,
    max_containers=MAX_CONTAINERS,
    timeout=60 * 60,
    retries=2,
)
def analyze_house(split: str, house: str) -> list[dict]:
    rooms = _house_rooms(split, house)
    with ThreadPoolExecutor(max_workers=ROOM_WORKERS) as pool:
        return list(pool.map(lambda room: _analyze_room(split, house, room), rooms))


def _summary(rooms: list[dict]) -> dict:
    combinations = Counter(room["combination"] for room in rooms)
    with_others = [room for room in rooms if "others" in room]
    labels = Counter(room["others"]["classification"]["label"] for room in with_others)
    confidence = Counter(room["others"]["classification"]["confidence"] for room in with_others)
    composition = Counter(room["others"]["classification"]["composition_label"] for room in with_others)
    by_combination = {}
    for combination in combinations:
        subset = [room for room in with_others if room["combination"] == combination]
        if subset:
            by_combination[combination] = dict(
                Counter(room["others"]["classification"]["label"] for room in subset)
            )
    return {
        "rooms": len(rooms),
        "combination_counts": dict(combinations),
        "others_rooms": len(with_others),
        "others_classification_counts": dict(labels),
        "others_composition_counts": dict(composition),
        "others_confidence_counts": dict(confidence),
        "others_classification_by_file_combination": by_combination,
    }


@app.local_entrypoint()
def main(output: str = "data/3d-front-analysis.json", split: str = SPLIT):
    houses = [entry.path.rsplit("/", 1)[1] for entry in source.listdir(split) if entry.type.name == "DIRECTORY"]
    started = time.monotonic()
    nested = analyze_house.map(
        [split] * len(houses),
        houses,
        order_outputs=False,
    )
    rooms = [room for house_rooms in nested for room in house_rooms]
    report = {
        "schema_version": 1,
        "source": f"modal://{VOLUME}/{split}",
        "method": {
            "presence": "exact filenames",
            "others": "triangle-area geometric inference",
            "horizontal_normal_threshold": 0.9,
            "vertical_normal_threshold": 0.1,
            "reference_elevation_tolerance": "max(3% of structural height, 0.002 scene units)",
            "dominant_role_thresholds": {"floor": 0.35, "ceil": 0.35, "wall": 0.35},
            "substantial_component_thresholds": {"floor": 0.2, "ceil": 0.2, "wall": 0.2},
            "warning": "others roles are inferred geometry, not semantic ground truth; unresolved does not mean decorative",
        },
        "elapsed_seconds": round(time.monotonic() - started),
        "summary": _summary(rooms),
        "scenes": rooms,
    }
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2))
    print(json.dumps({**report["summary"], "output": str(target), "elapsed_seconds": report["elapsed_seconds"]}, indent=2))
