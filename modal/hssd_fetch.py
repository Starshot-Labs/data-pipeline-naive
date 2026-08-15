"""All 168 HSSD scene GLBs on the dedicated `HSSD` Modal volume.

    modal run --detach modal/hssd_fetch.py

The volume mirrors the public Hugging Face repository beneath `scenes/`:

    /scenes/102343992.glb
    /scenes/102344022.glb
    ...
    /hssd-scenes.complete.json

The source inventory is discovered from the Hub tree API on every run and pinned to the
repository commit returned with that inventory. Files are LFS/Xet objects, but each is only
75 MiB to 1.43 GiB, so one HTTP range stream per scene is simpler than splitting each scene
again. `WORKERS` scenes download concurrently to distinct files, which is safe on a v2
Volume and fills the bandwidth that the Hub limits per connection.

Downloads go to `.partial` siblings and resume from their current byte count. A completed
file is accepted only when its size and SHA-256 equal the Hub's LFS metadata; existing files
are hashed once per run so a stale manifest can never bless modified bytes. The completion
manifest is written only after every scene has passed that check. Re-running therefore
repairs an interrupted or changed dataset rather than starting over.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import modal

app = modal.App("dc-hssd-fetch")

VOL = "/vol"
REPO_ID = "hssd/hssd-scenes"
TREE = "scenes"
MANIFEST = "hssd-scenes.complete.json"

image = modal.Image.debian_slim(python_version="3.12").env({"PYTHONUNBUFFERED": "1"})
volume = modal.Volume.from_name("HSSD", create_if_missing=True, version=2)

WORKERS = 32
CHUNK_BYTES = 4 * 1024 * 1024
TIMEOUT_S = 120
ATTEMPTS = 20
RETRY_MAX_S = 60
PROGRESS_EVERY_S = 30


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


def _next_link(value: str) -> str | None:
    for item in value.split(","):
        if 'rel="next"' in item:
            return item.partition("<")[2].partition(">")[0]
    return None


def _inventory() -> tuple[str, list[dict]]:
    with urllib.request.urlopen(f"https://huggingface.co/api/datasets/{REPO_ID}", timeout=60) as response:
        commit = json.load(response).get("sha", "")
    if len(commit) != 40:
        raise RuntimeError("the Hub returned no repository commit")

    query = urllib.parse.urlencode({"recursive": "false", "expand": "true", "limit": 100})
    url = f"https://huggingface.co/api/datasets/{REPO_ID}/tree/{commit}/{TREE}?{query}"
    files = []

    while url:
        with urllib.request.urlopen(url, timeout=60) as response:
            page = json.load(response)
            url = _next_link(response.headers.get("Link", ""))
        for entry in page:
            if entry["type"] != "file":
                continue
            lfs = entry.get("lfs")
            if not lfs or len(lfs.get("oid", "")) != 64:
                raise RuntimeError(f"{entry['path']} has no LFS SHA-256 metadata")
            files.append({"path": entry["path"], "bytes": lfs["size"], "sha256": lfs["oid"]})

    if not files:
        raise RuntimeError(f"the Hub returned no files under {TREE}/")
    files.sort(key=lambda item: item["path"])
    return commit, files


def _download_url(path: str, commit: str) -> str:
    quoted = urllib.parse.quote(path, safe="/")
    return f"https://huggingface.co/datasets/{REPO_ID}/resolve/{commit}/{quoted}?download=true"


def _transfer(item: dict, commit: str, advance) -> dict:
    target = Path(VOL) / item["path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(f"{target}.partial")

    if target.exists():
        if _size(target) == item["bytes"] and _sha256(target) == item["sha256"]:
            advance(item["bytes"], True)
            return item
        target.unlink()
    if _size(partial) > item["bytes"]:
        partial.unlink()

    for attempt in range(1, ATTEMPTS + 1):
        have = _size(partial)
        if have == item["bytes"]:
            break
        request = urllib.request.Request(
            _download_url(item["path"], commit),
            headers={"Range": f"bytes={have}-{item['bytes'] - 1}"},
        )
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response, open(partial, "ab") as out:
                if response.status != 206:
                    raise RuntimeError(f"{item['path']} range returned HTTP {response.status}, not 206")
                remaining = item["bytes"] - have
                while remaining and (chunk := response.read(min(CHUNK_BYTES, remaining))):
                    out.write(chunk)
                    advance(len(chunk), False)
                    remaining -= len(chunk)
                if remaining:
                    raise OSError(f"{item['path']} ended {remaining} bytes short")
        except OSError as err:
            delay = min(2 ** (attempt - 1), RETRY_MAX_S)
            print(f"{item['path']} attempt {attempt}/{ATTEMPTS}: {err}; retrying in {delay}s")
            time.sleep(delay)

    if _size(partial) != item["bytes"]:
        raise RuntimeError(f"{item['path']} stopped at {_size(partial)} of {item['bytes']} bytes")
    actual = _sha256(partial)
    if actual != item["sha256"]:
        partial.unlink()
        raise RuntimeError(f"{item['path']} has SHA-256 {actual}, expected {item['sha256']}")
    partial.replace(target)
    return item


@app.function(
    image=image,
    volumes={VOL: volume},
    cpu=8,
    memory=16384,
    timeout=24 * 3600,
)
def fetch_all() -> dict:
    started = time.monotonic()
    commit, inventory = _inventory()
    expected_bytes = sum(item["bytes"] for item in inventory)
    downloaded = verified = 0
    lock = threading.Lock()

    def advance(count: int, was_verified: bool) -> None:
        nonlocal downloaded, verified
        with lock:
            if was_verified:
                verified += count
            else:
                downloaded += count

    print(
        f"{len(inventory)} HSSD scenes at commit {commit[:12]}: "
        f"{_gib(expected_bytes):.2f} GiB, {WORKERS} workers"
    )

    with ThreadPoolExecutor(max_workers=min(WORKERS, len(inventory))) as pool:
        futures = [pool.submit(_transfer, item, commit, advance) for item in inventory]
        pending = set(futures)
        while pending:
            done = {future for future in pending if future.done()}
            pending -= done
            for future in done:
                item = future.result()
                print(f"verified {item['path']} ({len(inventory) - len(pending)}/{len(inventory)})")
            if pending:
                elapsed = max(time.monotonic() - started, 0.001)
                print(
                    f"{_gib(verified + downloaded):.2f}/{_gib(expected_bytes):.2f} GiB accounted for, "
                    f"{downloaded / 1024**2 / elapsed:.0f} MiB/s this run, {len(pending)} scenes left"
                )
                time.sleep(PROGRESS_EVERY_S)

    manifest = {
        "repo": REPO_ID,
        "commit": commit,
        "files": inventory,
        "file_count": len(inventory),
        "bytes": expected_bytes,
    }
    (Path(VOL) / MANIFEST).write_text(json.dumps(manifest, indent=2))
    volume.commit()
    return {
        "commit": commit,
        "files": len(inventory),
        "bytes": expected_bytes,
        "downloaded_bytes": downloaded,
        "verified_existing_bytes": verified,
        "elapsed_seconds": round(time.monotonic() - started),
    }


@app.local_entrypoint()
def main():
    print(json.dumps(fetch_all.remote(), indent=2))
