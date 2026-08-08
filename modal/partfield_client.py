"""Client for `dc-partfield` — NVIDIA PartField part segmentation, over HTTP.

    python partfield_client.py chair.glb --parts 8
    python partfield_client.py chair.glb --parts 8,5,3 --out cuts/

or as a library:

    from partfield_client import segment, cut, save

    coarse = segment("chair.glb", parts=8)
    fine = cut(coarse.job_id, parts=14)   # seconds, and no GPU
    save(fine, "cuts/chair")

The standard library only, one file, nothing from the repo it ships in — copy it wherever it
is needed. `PARTFIELD_BASE_URL` overrides the service it talks to.

PartField predicts a feature field over a shape and clusters the per-triangle features under
the mesh's own face adjacency, so parts fall out without a prompt, a class list or a second
view. What it builds is a *hierarchy* rather than one segmentation, and a result is one level
of it. Which level is right is a property of the shape instead of the model — eight flatters a
chair and shreds a mug — so `segment` pays about ninety seconds of A10G to build the tree and
`cut` reads any other level back out of it in seconds, with no GPU and no model. A job's
`summary["levels"]` says which levels it can still answer.

A job is identified by the mesh, the parameters and the sample name together, so submitting
the same three twice attaches to the work already done rather than repeating it. The flip side
is that two callers sharing a job must not `cut` it to different levels at once: the artifacts
are named for their level and so are safe, but the status they poll is one record and the
second call overwrites what the first is waiting for. Different sample names, different jobs.

Both calls spawn and are polled; there is no synchronous form. What comes back per cut:

    parts_NN.glb    one named, coloured node per part, in the uploaded mesh's own frame
    labels_NN.bin   little-endian int32 per face, the part face i belongs to
    labels_NN.json  face and vertex counts, faces per part, each part's colour, and `levels`
"""

from __future__ import annotations  # the machine running this may be older than 3.10

import argparse
import array
import functools
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

BASE_URL = os.environ.get(
    "PARTFIELD_BASE_URL", "https://starshot-aitools--dc-partfield-web.modal.run"
).rstrip("/")

POLL_SECONDS = 10
JOB_TIMEOUT_SECONDS = 3600
# Meshes run to tens of megabytes, so the upload deadline covers a transfer, not a handshake.
UPLOAD_TIMEOUT_SECONDS = 600
FILE_TIMEOUT_SECONDS = 300


@dataclass
class Cut:
    """One level of a job's hierarchy."""

    job_id: str
    parts: int
    glb: bytes
    labels: array.array
    summary: dict


@functools.lru_cache(maxsize=1)
def _ssl_context() -> ssl.SSLContext:
    """certifi's roots if they are installed, since a Python old enough can have shipped with
    a set that has since expired — and then fails to verify a certificate that is perfectly
    good. Verification stays on either way.
    """
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def _multipart(fields: dict, file: tuple | None) -> tuple[bytes, str]:
    """The service reads its arguments as form data, and the mesh as an uploaded file."""
    boundary = uuid.uuid4().hex
    body = bytearray()
    for name, value in fields.items():
        body += (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'
        ).encode()
    if file is not None:
        filename, blob = file
        body += (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="model"; filename="{filename}"\r\n'
            "Content-Type: application/octet-stream\r\n\r\n"
        ).encode()
        body += blob + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    return bytes(body), f"multipart/form-data; boundary={boundary}"


def _call(
    method: str,
    path: str,
    *,
    fields: dict | None = None,
    file: tuple | None = None,
    timeout: float = 60,
) -> bytes:
    data, content_type = _multipart(fields or {}, file) if fields or file else (None, None)
    request = urllib.request.Request(f"{BASE_URL}{path}", data=data, method=method)
    if content_type:
        request.add_header("Content-Type", content_type)
    try:
        with urllib.request.urlopen(request, timeout=timeout, context=_ssl_context()) as response:
            return response.read()
    except urllib.error.HTTPError as err:
        # FastAPI puts the reason in the body, which is the only thing a caller can act on.
        detail = err.read().decode("utf-8", "replace")[:300]
        raise RuntimeError(f"{method} {path} -> {err.code}: {detail}") from None


def _json(method: str, path: str, **kwargs) -> dict:
    return json.loads(_call(method, path, **kwargs))


def _labels(data: bytes) -> array.array:
    """One int32 per face. Little-endian, like every machine this will run on."""
    labels = array.array("i")
    labels.frombytes(data)
    return labels


def health() -> dict:
    return _json("GET", "/health", timeout=30)


def _await_job(job_id: str, log: Callable[[str], None]) -> dict:
    """Follow a job to completion, reporting each new stage. Returns its final status."""
    deadline = time.time() + JOB_TIMEOUT_SECONDS
    stage = None
    while True:
        if time.time() > deadline:
            raise TimeoutError(f"job {job_id} was still {stage} after {JOB_TIMEOUT_SECONDS}s")
        time.sleep(POLL_SECONDS)

        try:
            status = _json("GET", f"/jobs/{job_id}", timeout=30)
        except (RuntimeError, urllib.error.URLError) as err:
            log(f"poll failed, retrying: {err}")
            continue

        if status.get("stage") != stage:
            stage = status.get("stage")
            log(f"{stage}")
        if status["status"] == "done":
            return status
        if status["status"] == "failed":
            raise RuntimeError(f"job {job_id} failed: {str(status.get('error'))[:500]}")


def _collect(job_id: str, files: list) -> Cut:
    """The three artifacts of a finished cut, taken by extension since they are named for
    the level they are."""
    blobs = {
        name: _call("GET", f"/jobs/{job_id}/file/{urllib.parse.quote(name)}", timeout=FILE_TIMEOUT_SECONDS)
        for name in files
    }

    def of(suffix: str) -> bytes:
        for name, blob in blobs.items():
            if name.endswith(suffix):
                return blob
        raise RuntimeError(f"job {job_id} returned no {suffix}, only {', '.join(blobs) or 'nothing'}")

    summary = json.loads(of(".json"))
    return Cut(job_id, summary["num_parts"], of(".glb"), _labels(of(".bin")), summary)


def segment(
    path,
    *,
    parts: int = 8,
    max_clusters: int = 20,
    option: int = 0,
    with_knn: bool = False,
    points_per_face: int = 1000,
    sample: str | None = None,
    log: Callable[[str], None] = print,
) -> Cut:
    """Build a mesh's part hierarchy and take one cut of it. About ninety seconds of A10G.

    `max_clusters` is how deep the tree goes, and so bounds what `cut` can ask for later.
    `option` only matters for a mesh that is not one connected component, where it chooses how
    the pieces are bridged before clustering: 0 chains them, 2 spans them along a minimum
    spanning tree. `points_per_face` is a ceiling rather than a setting — a mesh dense enough
    to overrun the card is sampled more sparsely.
    """
    source = Path(path)
    spawn = _json(
        "POST",
        "/segment",
        fields={
            "sample": sample or source.stem,
            "num_parts": parts,
            "max_clusters": max_clusters,
            "option": option,
            "with_knn": str(bool(with_knn)).lower(),
            "n_point_per_face": points_per_face,
        },
        file=(source.name, source.read_bytes()),
        timeout=UPLOAD_TIMEOUT_SECONDS,
    )
    job_id = spawn["job_id"]
    log(f"segment job {job_id}")
    return _collect(job_id, _await_job(job_id, log)["files"])


def cut(job_id: str, parts: int, *, log: Callable[[str], None] = print) -> Cut:
    """Another level of a hierarchy `segment` already built. `parts` has to be one of the
    job's `summary["levels"]`."""
    _json("POST", "/cut", fields={"job_id": job_id, "num_parts": parts})
    return _collect(job_id, _await_job(job_id, log)["files"])


def discard(job_id: str) -> dict:
    """Free a job's feature field and hierarchy, which are tens of megabytes each."""
    return _json("DELETE", f"/jobs/{job_id}")


def save(result: Cut, out) -> Path:
    """The cut's three artifacts into `out`, named for their level so cuts accumulate."""
    directory = Path(out)
    directory.mkdir(parents=True, exist_ok=True)
    level = f"{result.parts:02d}"
    (directory / f"parts_{level}.glb").write_bytes(result.glb)
    (directory / f"labels_{level}.bin").write_bytes(result.labels.tobytes())
    (directory / f"labels_{level}.json").write_text(json.dumps(result.summary, indent=2))
    return directory


def main() -> None:
    parser = argparse.ArgumentParser(description="Segment a mesh into parts with PartField.")
    parser.add_argument("mesh", help="a .glb, .obj or .off")
    parser.add_argument("--parts", default="8", help="one level, or several: 8,5,3")
    parser.add_argument("--max-clusters", type=int, default=20)
    parser.add_argument("--option", type=int, default=0, choices=(0, 1, 2))
    parser.add_argument("--with-knn", action="store_true")
    parser.add_argument("--points-per-face", type=int, default=1000)
    parser.add_argument("--out", default=".")
    parser.add_argument("--sample", help="names the job; defaults to the mesh's filename")
    parser.add_argument("--discard", action="store_true", help="free the job when finished")
    args = parser.parse_args()

    levels = [int(level) for level in args.parts.split(",")]
    if any(level < 1 or level > args.max_clusters for level in levels):
        parser.error(f"--parts must be between 1 and --max-clusters ({args.max_clusters})")

    print(f"partfield -> {json.dumps(health())}")

    # Only the first level pays for the hierarchy; the rest are read back out of it.
    result = None
    for parts in levels:
        result = (
            cut(result.job_id, parts)
            if result
            else segment(
                args.mesh,
                parts=parts,
                max_clusters=args.max_clusters,
                option=args.option,
                with_knn=args.with_knn,
                points_per_face=args.points_per_face,
                sample=args.sample,
            )
        )
        out = save(result, args.out)
        print(f"{result.parts} parts over {result.summary['faces']} faces -> {out}")

    if args.discard and result:
        discard(result.job_id)
        print(f"discarded {result.job_id}")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, TimeoutError, OSError) as err:
        sys.exit(f"error: {err}")
