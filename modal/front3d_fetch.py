"""MIDI-3D's repackaging of 3D-FRONT on the `3D-Front` Modal volume.

    # Download and extract the 12.4 GiB test scene archive:
    modal run --detach modal/front3d_fetch.py

    # Download one file without treating a split part as an archive:
    modal run --detach modal/front3d_fetch.py --archive 3D-FRONT-SCENE.partaa

    # Download all 13 full-scene parts and stream-extract them beside the test tree:
    modal run --detach modal/front3d_fetch.py --all-scenes

The resulting layout follows the dataset card:

    /3D-FRONT-TEST-SCENE.tar.gz
    /3D-FRONT-TEST-SCENE/<house>/<room>/<object>.glb
    /3D-FRONT-SCENE.partaa ... /3D-FRONT-SCENE.partam
    /3D-FRONT-SCENE/<house>/<room>/<object>.glb

Downloads use parallel HTTP ranges because the Hub shapes each connection rather than the
aggregate. Against these files from a Modal container, one connection gets roughly 2 MiB/s
and 32 get 60-100 MiB/s. Work is divided into many 256 MiB ranges instead of one large range
per connection, so a slow connection cannot leave every other worker idle for half an hour.
Each range appends to its own temporary file: v1 Volumes are fast at sequential writes but
may rewrite a whole file when a writer seeks into its middle. The temporary files are
assembled in range order and removed only after the result has its expected length.

Every download is resumable. Correctly sized output and range files are retained; short
ranges continue from their current byte count. The Hub's advertised SHA-256 is recorded
next to every finished file and checked on later runs. A pre-existing file without a record
(such as one produced by an older version of this script) is hashed once before it is used.
The all-scenes campaign has one writer for the volume and relies on Modal's periodic
background commits while it runs; explicit commits are reserved for the function boundary,
because a v1 commit can otherwise publish an old whole-volume snapshot over newer writes.

`3D-FRONT-SCENE.partaa` through `partam` are byte slices of one gzip stream, not standalone
archives. All 13 are required in filename order. Extraction reads across them as one stream,
without creating another 484 GiB combined archive. A completion manifest is written only
after gzip reaches a clean EOF. Until then, re-running is safe: already extracted files at
their archive sizes are skipped, though gzip must still scan forward to reach new members.
"""

import hashlib
import json
import shutil
import tarfile
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path
from typing import BinaryIO, Iterable

import modal

app = modal.App("dc-front3d")

VOL = "/vol"
REPO_ID = "huanngzh/3D-Front"
DEFAULT_ARCHIVE = "3D-FRONT-TEST-SCENE.tar.gz"
SCENE_PREFIX = "3D-FRONT-SCENE.part"
SCENE_PARTS = [f"{SCENE_PREFIX}a{letter}" for letter in "abcdefghijklm"]
SCENE_ROOT = "3D-FRONT-SCENE"

image = modal.Image.debian_slim(python_version="3.12").env({"PYTHONUNBUFFERED": "1"})
vol = modal.Volume.from_name("3D-Front", create_if_missing=True)

LANES = 32
RANGE_BYTES = 256 * 1024 * 1024
CHUNK_BYTES = 4 * 1024 * 1024
PROGRESS_EVERY_S = 30
TIMEOUT_S = 120
ATTEMPTS = 20
RETRY_MAX_S = 60
HASH_SUFFIX = ".sha256.json"
COMPLETE_SUFFIX = ".complete.json"


def _gib(count: int) -> float:
    return count / 1024**3


def _size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        while chunk := source.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _url(name: str) -> str:
    return f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/{name}?download=true"


def _remote_metadata(name: str) -> tuple[int, str]:
    """The canonical byte count and LFS SHA-256, before following the CDN redirect."""

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None

    request = urllib.request.Request(_url(name), method="HEAD")
    try:
        urllib.request.build_opener(NoRedirect).open(request, timeout=60)
    except urllib.error.HTTPError as response:
        if response.code != 302:
            raise
        size = response.headers.get("X-Linked-Size")
        digest = response.headers.get("X-Linked-ETag", "").strip('"')
        if size and len(digest) == 64:
            return int(size), digest

    with urllib.request.urlopen(request, timeout=60) as response:
        size = response.headers.get("Content-Length")
        digest = response.headers.get("X-Linked-ETag", "").strip('"')
        if not size or len(digest) != 64:
            raise RuntimeError(f"the Hub returned no size/SHA-256 for {name}")
        return int(size), digest


def _hash_record(path: Path) -> Path:
    return Path(f"{path}{HASH_SUFFIX}")


def _is_verified(path: Path, size: int, digest: str) -> bool:
    if _size(path) != size:
        return False

    record = _hash_record(path)
    if record.exists():
        try:
            saved = json.loads(record.read_text())
        except (OSError, json.JSONDecodeError):
            saved = {}
        if saved == {"bytes": size, "sha256": digest}:
            return True

    print(f"verifying {_gib(size):.2f} GiB already at {path.name}")
    actual = _sha256(path)
    if actual != digest:
        raise RuntimeError(f"{path.name} has SHA-256 {actual}, expected {digest}")
    record.write_text(json.dumps({"bytes": size, "sha256": digest}, indent=2))
    return True


def _pull(url: str, part: Path, first: int, last: int, advance) -> None:
    request = urllib.request.Request(url, headers={"Range": f"bytes={first}-{last}"})
    with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response, open(part, "ab") as out:
        if response.status != 206:
            raise RuntimeError(f"range {first}-{last} returned HTTP {response.status}, not 206")
        remaining = last - first + 1
        while remaining and (chunk := response.read(min(CHUNK_BYTES, remaining))):
            out.write(chunk)
            advance(len(chunk))
            remaining -= len(chunk)
        if remaining:
            raise OSError(f"range {first}-{last} ended {remaining} bytes short")


def _lane(url: str, part: Path, start: int, length: int, advance) -> None:
    """One contiguous slice of a source file into its own resumable range file."""
    if _size(part) > length:
        part.unlink()

    for attempt in range(1, ATTEMPTS + 1):
        if _size(part) == length:
            return
        try:
            _pull(url, part, start + _size(part), start + length - 1, advance)
        except OSError as err:
            delay = min(2 ** (attempt - 1), RETRY_MAX_S)
            print(f"{part.name} attempt {attempt} of {ATTEMPTS}: {err}; retrying in {delay}s")
            time.sleep(delay)
    if _size(part) != length:
        raise RuntimeError(f"{part.name} stopped {_gib(length - _size(part)):.2f} GiB short")


def _assemble(target: Path, parts: list[Path], expected_size: int) -> None:
    started = time.monotonic()
    temporary = Path(f"{target}.assembling")
    temporary.unlink(missing_ok=True)

    with open(temporary, "wb") as out:
        for part in parts:
            with open(part, "rb") as source:
                shutil.copyfileobj(source, out, CHUNK_BYTES)

    if _size(temporary) != expected_size:
        raise RuntimeError(f"assembled {target.name} to {_size(temporary)} of {expected_size} bytes")
    temporary.replace(target)
    for part in parts:
        part.unlink()
    print(f"assembled {target.name} in {time.monotonic() - started:.0f}s")


def _download(name: str, target: Path) -> dict:
    total, digest = _remote_metadata(name)
    if _is_verified(target, total, digest):
        print(f"{name} is verified on the volume, {_gib(total):.2f} GiB")
        return {"archive": name, "archive_bytes": total, "sha256": digest, "download_seconds": 0}

    if target.exists():
        target.unlink()
    _hash_record(target).unlink(missing_ok=True)

    slices = [
        (start, min(RANGE_BYTES, total - start))
        for start in range(0, total, RANGE_BYTES)
    ]
    parts = [Path(f"{target}.range{i:04d}") for i in range(len(slices))]

    done, lock = 0, threading.Lock()

    def advance(count: int) -> None:
        nonlocal done
        with lock:
            done += count

    resumed = sum(min(_size(part), length) for part, (_, length) in zip(parts, slices))
    print(
        f"{name}: {_gib(total):.2f} GiB in {len(slices)} ranges over "
        f"{min(LANES, len(slices))} workers, {_gib(resumed):.2f} GiB resumed"
    )

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=min(LANES, len(slices))) as pool:
        running = [
            pool.submit(_lane, _url(name), part, start, length, advance)
            for part, (start, length) in zip(parts, slices)
        ]
        while True:
            pending = wait(running, timeout=PROGRESS_EVERY_S).not_done
            elapsed = max(time.monotonic() - started, 0.001)
            print(
                f"{name}: {_gib(resumed + done):.2f}/{_gib(total):.2f} GiB at "
                f"{done / 1024**2 / elapsed:.0f} MiB/s, {len(pending)} ranges left"
            )
            if not pending:
                break
        for lane in running:
            lane.result()

    _assemble(target, parts, total)
    print(f"hashing {target.name}")
    actual = _sha256(target)
    if actual != digest:
        target.unlink()
        raise RuntimeError(f"{name} downloaded with SHA-256 {actual}, expected {digest}")
    _hash_record(target).write_text(json.dumps({"bytes": total, "sha256": digest}, indent=2))

    return {
        "archive": name,
        "archive_bytes": total,
        "sha256": digest,
        "download_seconds": round(time.monotonic() - started),
    }


def _extract(tar: tarfile.TarFile, dest: Path) -> dict:
    root, written, skipped, size, read = "", 0, 0, 0, 0
    started = last = time.monotonic()

    for member in tar:
        root = root or member.name.rstrip("/").split("/")[0]
        target = dest / member.name
        if member.isfile() and _size(target) == member.size:
            skipped += 1
        else:
            tar.extract(member, dest, filter="data")
            if member.isfile():
                written += 1
                size += member.size
        read += member.size

        if time.monotonic() - last >= PROGRESS_EVERY_S:
            elapsed = time.monotonic() - started
            print(
                f"{_gib(read):.2f} GiB read in {elapsed:.0f}s — "
                f"{written} files written ({_gib(size):.2f} GiB), {skipped} already there"
            )
            last = time.monotonic()

    return {
        "root": root,
        "files_written": written,
        "files_skipped": skipped,
        "bytes_written": size,
        "archive_bytes_read": read,
        "extract_seconds": round(time.monotonic() - started),
    }


def _extract_archive(archive_path: Path, dest: Path) -> dict:
    with tarfile.open(archive_path, "r:gz") as tar:
        return _extract(tar, dest)


class JoinedParts:
    """A forward-only raw stream over files that are consecutive bytes of one archive."""

    def __init__(self, paths: Iterable[Path]):
        self.paths = iter(paths)
        self.current: BinaryIO | None = None

    def readable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:
        chunks = []
        remaining = size
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
                continue
            self.current.close()
            self.current = None
        return b"".join(chunks)

    def close(self) -> None:
        if self.current is not None:
            self.current.close()
            self.current = None


def _extract_parts(paths: list[Path], dest: Path) -> dict:
    source = JoinedParts(paths)
    try:
        with tarfile.open(fileobj=source, mode="r|gz") as tar:
            return _extract(tar, dest)
    finally:
        source.close()


def _survey(root: Path) -> dict:
    houses = [house for house in root.iterdir() if house.is_dir()]
    files = [path for path in root.rglob("*") if path.is_file()]
    return {
        "houses": len(houses),
        "rooms": sum(1 for house in houses for room in house.iterdir() if room.is_dir()),
        "files": len(files),
        "bytes": sum(path.stat().st_size for path in files),
    }


@app.function(image=image, volumes={VOL: vol}, cpu=4, memory=8192, timeout=24 * 3600)
def fetch(archive: str = DEFAULT_ARCHIVE, keep_archive: bool = True) -> dict:
    archive_path = Path(VOL) / Path(archive).name
    result = _download(archive_path.name, archive_path)

    if any(suffix.startswith(".part") for suffix in archive_path.suffixes):
        print(f"retaining multipart chunk {archive_path.name}")
    else:
        extracted = _extract_archive(archive_path, Path(VOL))
        result.update(extracted)
        result.update(_survey(Path(VOL) / extracted["root"]))
        if not keep_archive:
            archive_path.unlink()
            _hash_record(archive_path).unlink(missing_ok=True)
            result["archive_bytes"] = 0

    vol.commit()
    return result


@app.function(image=image, volumes={VOL: vol}, cpu=4, memory=8192, timeout=24 * 3600)
def fetch_all_scenes() -> dict:
    started = time.monotonic()
    parts = []
    for index, name in enumerate(SCENE_PARTS, 1):
        print(f"scene part {index}/{len(SCENE_PARTS)}: {name}")
        path = Path(VOL) / name
        _download(name, path)
        parts.append(path)

    manifest_path = Path(VOL) / f"{SCENE_ROOT}{COMPLETE_SUFFIX}"
    expected_parts = {
        path.name: json.loads(_hash_record(path).read_text())
        for path in parts
    }
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            manifest = {}
    else:
        manifest = {}

    if manifest.get("parts") == expected_parts and (Path(VOL) / SCENE_ROOT).is_dir():
        print(f"{SCENE_ROOT} is already marked complete")
    else:
        print(f"all {len(parts)} parts verified; stream-extracting {SCENE_ROOT}")
        extracted = _extract_parts(parts, Path(VOL))
        if extracted["root"] != SCENE_ROOT:
            raise RuntimeError(f"multipart archive root is {extracted['root']!r}, expected {SCENE_ROOT!r}")
        manifest = {"parts": expected_parts, "extraction": extracted}
        manifest_path.write_text(json.dumps(manifest, indent=2))

    survey = _survey(Path(VOL) / SCENE_ROOT)
    vol.commit()
    return {
        "parts": len(parts),
        "compressed_bytes": sum(_size(path) for path in parts),
        "elapsed_seconds": round(time.monotonic() - started),
        **survey,
    }


@app.local_entrypoint()
def main(archive: str = DEFAULT_ARCHIVE, keep_archive: bool = True, all_scenes: bool = False):
    result = fetch_all_scenes.remote() if all_scenes else fetch.remote(archive, keep_archive)
    print(json.dumps(result, indent=2))
