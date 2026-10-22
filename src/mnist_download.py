"""Verified MNIST downloads with per-process proxy control and atomic caching."""

from __future__ import annotations

import gzip
import hashlib
import http.client
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from tqdm import tqdm


MIRRORS = (
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
)


def md5_file(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_opener(no_proxy: bool):
    if no_proxy:
        return urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return urllib.request.build_opener()


def fetch_archive(raw_dir: Path, filename: str, expected_md5: str, *,
                  opener, retries: int = 2, mirrors=MIRRORS) -> dict:
    if retries < 1:
        raise ValueError("retries must be at least 1")
    if Path(filename).name != filename:
        raise ValueError("Archive filename must be a plain basename")
    raw_dir.mkdir(parents=True, exist_ok=True)
    destination = raw_dir / filename
    if destination.is_file() and md5_file(destination) == expected_md5:
        print(f"  Verified cached archive: {filename}", flush=True)
        return {"filename": filename, "source": "verified_local_cache",
                "md5": expected_md5, "bytes": destination.stat().st_size}

    # A partial transfer never replaces the destination. Only verified bytes do.
    partial = raw_dir / (filename + ".part")
    failures = []
    for mirror in mirrors:
        url = mirror + filename
        for attempt in range(1, retries + 1):
            print(f"  Downloading {filename} | attempt {attempt}/{retries} | {mirror}", flush=True)
            try:
                request = urllib.request.Request(url, headers={
                    "User-Agent": "MNIST-course-project/1.0",
                    "Accept-Encoding": "identity",
                })
                with opener.open(request, timeout=60) as response:
                    length = response.headers.get("Content-Length")
                    expected_bytes = int(length) if length is not None else None
                    received = 0
                    with partial.open("wb") as stream, tqdm(
                        total=expected_bytes, unit="B", unit_scale=True,
                        desc=filename, disable=not sys.stderr.isatty(),
                    ) as progress:
                        while chunk := response.read(64 * 1024):
                            stream.write(chunk)
                            received += len(chunk)
                            progress.update(len(chunk))
                if expected_bytes is not None and received != expected_bytes:
                    raise OSError(f"Incomplete response: received {received}, expected {expected_bytes} bytes")
                actual_md5 = md5_file(partial)
                if actual_md5 != expected_md5:
                    raise ValueError(f"MD5 mismatch after {received} bytes: {actual_md5}, expected {expected_md5}")
                partial.replace(destination)
                print(f"  Download verified: {filename} ({received} bytes)", flush=True)
                return {"filename": filename, "source": url,
                        "md5": actual_md5, "bytes": received}
            except (OSError, ValueError, http.client.HTTPException) as exc:
                message = f"{mirror} attempt {attempt}: {type(exc).__name__}: {exc}"
                failures.append(message)
                print(f"  Download failed: {type(exc).__name__}: {exc}", flush=True)
                if attempt < retries:
                    time.sleep(1)
    raise RuntimeError(
        f"Could not obtain a verified {filename}. No invalid download was installed.\n"
        + "\n".join(failures)
        + "\nTry the other proxy mode (--no-proxy or default), or place the original .gz files in data/MNIST/raw."
    )


def prepare_mnist_files(data_dir: Path, resources, *, no_proxy: bool = False) -> list[dict]:
    raw_dir = data_dir / "MNIST" / "raw"
    opener = make_opener(no_proxy)
    print("  Download proxy mode: " + ("direct (system proxies bypassed)" if no_proxy else "system/environment proxies"), flush=True)
    records = []
    for filename, expected_md5 in resources:
        record = fetch_archive(raw_dir, filename, expected_md5, opener=opener)
        archive = raw_dir / filename
        extracted = raw_dir / filename.removesuffix(".gz")
        extraction_partial = raw_dir / (extracted.name + ".part")
        # gzip validates CRC/length on full extraction; keep incomplete output separate.
        with gzip.open(archive, "rb") as source, extraction_partial.open("wb") as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)
        extraction_partial.replace(extracted)
        records.append(record)
    return records
