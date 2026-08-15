"""Normalize object identifiers inside MIDI's assembled 3D-FRONT room GLBs.

The component GLBs under each room directory are the source of truth:

    <split>/<house>/<room>/<object-id>.glb
    <split>/<house>/<room>.glb
    <split>/<house>/<room>_full.glb

After normalization, every assembled child node and its mesh are named exactly like the source
component's filename, including `.glb`. The plain scene contains the furniture files; the full
scene contains those plus the optional `floor.glb`, `wall.glb`, `ceil.glb`, and `others.glb`
components.

Editing is much faster than reconstruction. Every transform is already baked into the component
geometry, but rebuilding would still deserialize and repack roughly 590 GiB of geometry,
textures, materials, accessors, and buffer views merely to change strings. The normalizer reads
each GLB's small JSON chunk, matches children to components by exact geometry metadata and the
legacy inner name when duplicate geometry needs a tie-breaker, and overwrites that JSON chunk at
its existing byte length whenever it fits. Files without enough padding for the literal `.glb`
suffix are atomically rewritten with a minimally larger JSON chunk while their binary chunk is
copied byte-for-byte.

Cleanup is in-place on `3D-Front`, one Modal container and one writable mount. A map of writers is
not used: this is a v1 Volume, where overlapping commits can publish stale snapshots. Before a
JSON chunk is changed, its original bytes are saved under `/.front3d-name-backups/`; all 46,992
backups require only a few hundred MiB and keep the migration reversible. The original multipart
archives are also left untouched. Each fixed-size patch or atomic rewrite is fsynced and validated. Re-running
skips normalized assemblies, so a partial campaign resumes safely.

Commands:

    # exact read-only audit; this is the default and changes nothing
    modal run modal/front3d_clean.py

    # normalize all 23,496 rooms / 46,992 assembled GLBs in place
    npm run clean-front3d

    # restore every original JSON chunk from the retained backups
    modal run modal/front3d_clean.py --restore

A complete campaign writes `<split>.clean.json`; an interrupted one leaves
`<split>.cleaning.json` with its last completed house and counters.
"""

import json
import math
import os
import struct
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import modal

app = modal.App("dc-front3d-clean")

VOLUME_NAME = "3D-Front"
VOL = "/vol"
DEFAULT_SPLIT = "3D-FRONT-SCENE"
ARCHITECTURE = {"floor.glb", "wall.glb", "ceil.glb", "others.glb"}
BACKUP_ROOT = ".front3d-name-backups"
NORMALIZER_VERSION = 1
ROOM_WORKERS = 4
MAX_AUDIT_CONTAINERS = 32
COMMIT_EVERY_HOUSES = 500
PROGRESS_EVERY_HOUSES = 25

image = modal.Image.debian_slim(python_version="3.12").env({"PYTHONUNBUFFERED": "1"})
volume = modal.Volume.from_name(VOLUME_NAME)
read_only_volume = volume.read_only()


class GLBError(ValueError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> tuple[dict, int, bytes]:
    with open(path, "rb") as source:
        header = source.read(20)
        if len(header) != 20:
            raise GLBError(f"{path} has a truncated GLB header")
        magic, version, total, json_bytes, kind = struct.unpack("<4sIII4s", header)
        if magic != b"glTF" or version != 2 or total != path.stat().st_size or kind != b"JSON":
            raise GLBError(f"{path} is not a canonical GLB 2.0 file")
        raw = source.read(json_bytes)
    try:
        return json.loads(raw), json_bytes, raw
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GLBError(f"{path} has invalid JSON: {exc}") from exc


def _scene_children(doc: dict, path: Path) -> list[int]:
    try:
        scene = doc["scenes"][doc.get("scene", 0)]
        roots = scene["nodes"]
        if len(roots) != 1:
            raise GLBError(f"{path} has {len(roots)} scene roots, expected one")
        children = doc["nodes"][roots[0]].get("children", [])
    except (KeyError, IndexError, TypeError) as exc:
        raise GLBError(f"{path} has an unsupported scene graph") from exc
    if len(children) != len(set(children)):
        raise GLBError(f"{path} repeats a child node")
    return children


def _number(value):
    return "nan" if isinstance(value, float) and math.isnan(value) else value


def _node_fingerprint(doc: dict, node_index: int, path: Path) -> tuple:
    try:
        node = doc["nodes"][node_index]
        mesh = doc["meshes"][node["mesh"]]
        primitives = []
        for primitive in mesh["primitives"]:
            position = doc["accessors"][primitive["attributes"]["POSITION"]]
            indices = doc["accessors"][primitive["indices"]] if "indices" in primitive else None
            primitives.append(
                (
                    position["count"],
                    tuple(map(_number, position["min"])),
                    tuple(map(_number, position["max"])),
                    indices["count"] if indices else None,
                )
            )
        return tuple(primitives)
    except (KeyError, IndexError, TypeError) as exc:
        raise GLBError(f"{path} node {node_index} has unsupported geometry metadata") from exc


def _component_identity(path: Path) -> tuple[tuple, str]:
    doc, _, _ = _read_json(path)
    children = _scene_children(doc, path)
    if len(children) != 1:
        raise GLBError(f"{path} has {len(children)} objects, expected one")
    child = children[0]
    legacy_name = doc["nodes"][child].get("name", "").removesuffix(".obj")
    return _node_fingerprint(doc, child, path), legacy_name


def _component_files(room: Path, full: bool) -> list[Path]:
    files = sorted(path for path in room.iterdir() if path.is_file() and path.suffix.lower() == ".glb")
    return files if full else [path for path in files if path.name not in ARCHITECTURE]


def _mapping(scene_path: Path, components: list[Path]) -> tuple[dict, int, bytes, dict[int, str]]:
    doc, json_bytes, raw = _read_json(scene_path)
    children = _scene_children(doc, scene_path)
    if len(children) != len(components):
        raise GLBError(
            f"{scene_path} has {len(children)} objects but {len(components)} component GLBs"
        )

    assembled = defaultdict(list)
    for child in children:
        assembled[_node_fingerprint(doc, child, scene_path)].append(child)

    identities = {component: _component_identity(component) for component in components}
    matched = {}
    used = set()
    for component in components:
        fingerprint, legacy_name = identities[component]
        candidates = [node for node in assembled.get(fingerprint, []) if node not in used]
        if not candidates:
            raise GLBError(f"{component} has no geometry match in {scene_path}")
        if len(candidates) == 1:
            child = candidates[0]
        else:
            exact = [
                node
                for node in candidates
                if doc["nodes"][node].get("name", "").removesuffix(".obj")
                in {component.name, component.stem, legacy_name}
            ]
            if len(exact) != 1:
                kind = "architecture" if component.name in ARCHITECTURE else "furniture"
                raise GLBError(
                    f"{component} matches {len(candidates)} {kind} objects in {scene_path}, "
                    "and its legacy identifier is not unique"
                )
            child = exact[0]
        used.add(child)
        matched[child] = component.name

    if used != set(children):
        missing = sorted(set(children) - used)
        raise GLBError(f"{scene_path} leaves assembled nodes unmatched: {missing}")
    return doc, json_bytes, raw, matched


def _normalized(scene_path: Path, components: list[Path]) -> tuple[bytes, bytes, int, int, int]:
    doc, json_bytes, raw, matched = _mapping(scene_path, components)
    changed = 0
    for child, expected in matched.items():
        node = doc["nodes"][child]
        mesh = doc["meshes"][node["mesh"]]
        if node.get("name") != expected or mesh.get("name") != expected:
            changed += 1
        node["name"] = expected
        mesh["name"] = expected

    encoded = json.dumps(doc, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    output_json_bytes = max(json_bytes, (len(encoded) + 3) & ~3)
    return (
        encoded + b" " * (output_json_bytes - len(encoded)),
        raw,
        json_bytes,
        changed,
        len(matched),
    )


def _names_match(scene_path: Path, components: list[Path]) -> bool:
    _, _, _, changed, _ = _normalized(scene_path, components)
    return changed == 0


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.writing-{os.getpid()}-{threading.get_ident()}")
    try:
        with open(temporary, "wb") as out:
            out.write(data)
            out.flush()
            os.fsync(out.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_chunk(path: Path, chunk: bytes) -> None:
    size = path.stat().st_size
    with open(path, "r+b", buffering=0) as target:
        target.seek(20)
        written = target.write(chunk)
        if written != len(chunk):
            raise OSError(f"{path} wrote {written} of {len(chunk)} JSON bytes")
        os.fsync(target.fileno())
    if path.stat().st_size != size:
        raise GLBError(f"{path} changed size while normalizing")


def _replace_json_chunk(path: Path, chunk: bytes, old_json_bytes: int) -> None:
    old_size = path.stat().st_size
    new_size = old_size + len(chunk) - old_json_bytes
    temporary = path.with_name(f".{path.name}.rewriting-{os.getpid()}-{threading.get_ident()}")
    try:
        with open(path, "rb") as source, open(temporary, "wb") as out:
            source.seek(20 + old_json_bytes)
            out.write(struct.pack("<4sII", b"glTF", 2, new_size))
            out.write(struct.pack("<I4s", len(chunk), b"JSON"))
            out.write(chunk)
            while data := source.read(8 * 1024 * 1024):
                out.write(data)
            out.flush()
            os.fsync(out.fileno())
        if temporary.stat().st_size != new_size:
            raise GLBError(f"{path} rewrote to {temporary.stat().st_size} of {new_size} bytes")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _backup_path(split: str, house: str, scene_name: str) -> Path:
    return Path(VOL) / BACKUP_ROOT / split / house / f"{scene_name}.json-chunk"


def _restore_chunk(path: Path, backup: Path) -> None:
    raw = backup.read_bytes()
    with open(path, "rb") as source:
        header = source.read(20)
    if len(header) != 20:
        raise GLBError(f"{backup} cannot restore {path}")
    current_json_bytes = struct.unpack_from("<I", header, 12)[0]
    if current_json_bytes == len(raw):
        _write_json_chunk(path, raw)
    else:
        _replace_json_chunk(path, raw, current_json_bytes)
    _read_json(path)


def _recover(path: Path, backup: Path) -> None:
    try:
        _read_json(path)
    except GLBError:
        if not backup.exists():
            raise
        _restore_chunk(path, backup)


def _patch(scene_path: Path, components: list[Path], backup: Path) -> dict:
    _recover(scene_path, backup)
    try:
        normalized, original, old_json_bytes, changed, objects = _normalized(scene_path, components)
    except GLBError:
        if not backup.exists():
            raise
        _restore_chunk(scene_path, backup)
        normalized, original, old_json_bytes, changed, objects = _normalized(scene_path, components)
    if not changed:
        return {"assemblies_skipped": 1, "objects_named": objects}
    if not backup.exists():
        _atomic_write(backup, original)
    elif backup.read_bytes() != original:
        raise GLBError(f"{backup} does not match the current original chunk in {scene_path}")

    grew = len(normalized) > old_json_bytes
    if grew:
        _replace_json_chunk(scene_path, normalized, old_json_bytes)
    else:
        _write_json_chunk(scene_path, normalized)
    try:
        if not _names_match(scene_path, components):
            raise GLBError(f"{scene_path} failed post-write validation")
    except Exception:
        if grew:
            _replace_json_chunk(scene_path, original, len(normalized))
        else:
            _write_json_chunk(scene_path, original)
        raise
    return {
        "assemblies_cleaned": 1,
        "assemblies_rewritten": int(grew),
        "objects_named": objects,
        "names_changed": changed,
        "json_bytes_written": len(normalized),
    }


def _room_paths(split: str, house: str, room_name: str) -> tuple[Path, Path, Path]:
    house_root = Path(VOL) / split / house
    return (
        house_root / room_name,
        house_root / f"{room_name}.glb",
        house_root / f"{room_name}_full.glb",
    )


def _house_names(split: str, limit: int = 0, house: str = "") -> list[str]:
    root = Path(VOL) / split
    if house:
        if not (root / house).is_dir():
            raise RuntimeError(f"{split} has no house {house}")
        return [house]
    houses = [path.name for path in root.iterdir() if path.is_dir()]
    return houses[:limit] if limit else houses


def _room_names(split: str, house: str) -> list[str]:
    root = Path(VOL) / split / house
    return sorted(path.name for path in root.iterdir() if path.is_dir())


def _audit_room(split: str, house: str, room_name: str) -> dict:
    room, plain, full = _room_paths(split, house, room_name)
    furniture = _component_files(room, False)
    all_components = _component_files(room, True)
    _, _, _, plain_changed, plain_objects = _normalized(plain, furniture)
    _, _, _, full_changed, full_objects = _normalized(full, all_components)
    return {
        "rooms": 1,
        "assemblies": 2,
        "plain_clean": int(plain_changed == 0),
        "full_clean": int(full_changed == 0),
        "furniture": len(furniture),
        "architecture": len(all_components) - len(furniture),
        "objects": plain_objects + full_objects,
    }


def _patch_room(split: str, house: str, room_name: str) -> dict:
    room, plain, full = _room_paths(split, house, room_name)
    furniture = _component_files(room, False)
    all_components = _component_files(room, True)
    results = [
        _patch(plain, furniture, _backup_path(split, house, plain.name)),
        _patch(full, all_components, _backup_path(split, house, full.name)),
    ]
    total = _sum(results)
    total["rooms"] = 1
    return total


def _sum(results: list[dict]) -> dict:
    total = Counter()
    for result in results:
        total.update({key: value for key, value in result.items() if isinstance(value, (int, float))})
    return dict(total)


def _write_report(path: Path, report: dict) -> None:
    _atomic_write(path, json.dumps(report, indent=2).encode("utf-8"))


@app.function(
    image=image,
    volumes={VOL: read_only_volume},
    cpu=8,
    memory=4096,
    max_containers=MAX_AUDIT_CONTAINERS,
    timeout=60 * 60,
    retries=2,
)
def audit_house(split: str, house: str) -> dict:
    started = time.monotonic()
    rooms = _room_names(split, house)
    with ThreadPoolExecutor(max_workers=ROOM_WORKERS) as pool:
        total = _sum(list(pool.map(lambda room: _audit_room(split, house, room), rooms)))
    total["houses"] = 1
    total["worker_seconds"] = round(time.monotonic() - started, 3)
    return total


@app.function(
    image=image,
    volumes={VOL: volume},
    cpu=8,
    memory=4096,
    timeout=24 * 60 * 60,
    retries=1,
)
def clean_split(split: str, limit: int = 0, house: str = "") -> dict:
    houses = _house_names(split, limit, house)
    started = time.monotonic()
    total = Counter()
    progress_path = Path(VOL) / f"{split}.cleaning.json"

    for index, house_name in enumerate(houses, 1):
        rooms = _room_names(split, house_name)
        with ThreadPoolExecutor(max_workers=ROOM_WORKERS) as pool:
            total.update(_sum(list(pool.map(lambda room: _patch_room(split, house_name, room), rooms))))
        total["houses"] += 1

        if index % PROGRESS_EVERY_HOUSES == 0 or index == len(houses):
            elapsed = time.monotonic() - started
            progress = {
                "normalizer_version": NORMALIZER_VERSION,
                "volume": VOLUME_NAME,
                "split": split,
                "status": "cleaning",
                "houses_requested": len(houses),
                "houses_completed": index,
                "last_house": house_name,
                "elapsed_seconds": round(elapsed),
                **dict(total),
            }
            _write_report(progress_path, progress)
            rate = total["rooms"] / max(elapsed, 0.001)
            print(
                f"{index}/{len(houses)} houses, {total['rooms']} rooms, "
                f"{total['assemblies_cleaned']} changed, {rate:.1f} rooms/s",
                flush=True,
            )

        if index % COMMIT_EVERY_HOUSES == 0:
            volume.commit()

    complete = not limit and not house
    report = {
        "normalizer_version": NORMALIZER_VERSION,
        "volume": VOLUME_NAME,
        "split": split,
        "status": "complete" if complete else "partial",
        "completed_at": _now(),
        "elapsed_seconds": round(time.monotonic() - started),
        **dict(total),
    }
    if complete:
        _write_report(Path(VOL) / f"{split}.clean.json", report)
        progress_path.unlink(missing_ok=True)
    else:
        _write_report(progress_path, report)
    volume.commit()
    return report


@app.function(
    image=image,
    volumes={VOL: volume},
    cpu=8,
    memory=4096,
    timeout=24 * 60 * 60,
)
def restore_split(split: str, limit: int = 0, house: str = "", drop_backups: bool = False) -> dict:
    backups = Path(VOL) / BACKUP_ROOT / split
    houses = [house] if house else ([path.name for path in backups.iterdir()] if backups.exists() else [])
    houses = houses[:limit] if limit else houses
    restored = 0
    started = time.monotonic()

    for house_name in houses:
        root = backups / house_name
        if not root.is_dir():
            raise RuntimeError(f"no backups for {house_name}")
        for backup in root.glob("*.glb.json-chunk"):
            target = Path(VOL) / split / house_name / backup.name.removesuffix(".json-chunk")
            raw = backup.read_bytes()
            with open(target, "rb") as source:
                header = source.read(20)
            if len(header) != 20:
                raise GLBError(f"{backup} cannot restore {target}")
            current_json_bytes = struct.unpack_from("<I", header, 12)[0]
            if current_json_bytes == len(raw):
                _write_json_chunk(target, raw)
            else:
                _replace_json_chunk(target, raw, current_json_bytes)
            _read_json(target)
            restored += 1
            if drop_backups:
                backup.unlink()
    (Path(VOL) / f"{split}.clean.json").unlink(missing_ok=True)
    (Path(VOL) / f"{split}.cleaning.json").unlink(missing_ok=True)
    volume.commit()
    return {
        "volume": VOLUME_NAME,
        "split": split,
        "houses": len(houses),
        "assemblies_restored": restored,
        "backups_removed": restored if drop_backups else 0,
        "elapsed_seconds": round(time.monotonic() - started),
    }


def _remote_houses(split: str, limit: int, house: str) -> list[str]:
    if house:
        entries = volume.listdir(f"{split}/{house}")
        if not entries:
            raise RuntimeError(f"{split} has no house {house}")
        return [house]
    houses = [
        entry.path.rsplit("/", 1)[1]
        for entry in volume.listdir(split)
        if entry.type.name == "DIRECTORY"
    ]
    return houses[:limit] if limit else houses


@app.local_entrypoint()
def main(
    split: str = DEFAULT_SPLIT,
    apply: bool = False,
    restore: bool = False,
    limit: int = 0,
    house: str = "",
    drop_backups: bool = False,
):
    if apply and restore:
        raise ValueError("choose either --apply or --restore")
    if restore:
        print(json.dumps(restore_split.remote(split, limit, house, drop_backups), indent=2))
        return
    if apply:
        print(json.dumps(clean_split.remote(split, limit, house), indent=2))
        return

    houses = _remote_houses(split, limit, house)
    print(f"auditing {len(houses)} houses from {VOLUME_NAME}/{split}")
    started = time.monotonic()
    results = list(
        audit_house.starmap(
            ((split, name) for name in houses),
            order_outputs=False,
            return_exceptions=True,
            wrap_returned_exceptions=False,
        )
    )
    errors = [result for result in results if isinstance(result, BaseException)]
    good = [result for result in results if isinstance(result, dict)]
    report = {
        "normalizer_version": NORMALIZER_VERSION,
        "mode": "audit",
        "volume": VOLUME_NAME,
        "split": split,
        "houses_requested": len(houses),
        "houses_succeeded": len(good),
        "houses_failed": len(errors),
        "elapsed_seconds": round(time.monotonic() - started),
        **_sum(good),
        "errors": [str(error) for error in errors[:20]],
    }
    print(json.dumps(report, indent=2))
    if errors:
        raise RuntimeError(f"{len(errors)} houses failed exact matching")
