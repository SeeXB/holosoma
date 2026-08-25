#!/usr/bin/env python3
"""Download a large Hugging Face file with validated resumable HTTP ranges."""

from __future__ import annotations

import argparse
import concurrent.futures
import math
import os
import shutil
import threading
import time
from pathlib import Path

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-size", type=int)
    parser.add_argument("--chunk-size", type=int, default=8 * 1024 * 1024)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=30)
    parser.add_argument("--token-file", type=Path, help="Optional bearer token file; its contents are never logged.")
    return parser.parse_args()


def auth_headers(token_file: Path | None) -> dict[str, str]:
    headers = {"Accept-Encoding": "identity"}
    if token_file is not None:
        token = token_file.expanduser().read_text(encoding="utf-8").strip()
        if not token:
            raise ValueError(f"Empty token file: {token_file}")
        headers["Authorization"] = f"Bearer {token}"
    return headers


def probe_size(url: str, headers: dict[str, str], retries: int) -> int:
    request_headers = {**headers, "Range": "bytes=0-0"}
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with requests.get(url, headers=request_headers, stream=True, timeout=(20, 120)) as response:
                response.raise_for_status()
                content_range = response.headers.get("Content-Range", "")
                if response.status_code != 206 or "/" not in content_range:
                    raise RuntimeError(
                        f"Server did not honor range probe: status={response.status_code}, "
                        f"Content-Range={content_range!r}"
                    )
                return int(content_range.rsplit("/", 1)[1])
        except Exception as exc:  # network/proxy failures are expected to be transient
            last_error = exc
            time.sleep(min(2 ** min(attempt, 4), 15))
    raise RuntimeError(f"Could not probe remote file size after {retries} attempts: {last_error}")


def download_part(
    *,
    url: str,
    headers: dict[str, str],
    part_path: Path,
    start: int,
    end: int,
    total: int,
    retries: int,
) -> str:
    expected = end - start + 1
    if part_path.is_file() and part_path.stat().st_size == expected:
        return "cached"
    temporary = part_path.with_suffix(part_path.suffix + ".tmp")
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request_headers = {**headers, "Range": f"bytes={start}-{end}"}
            with requests.get(url, headers=request_headers, stream=True, timeout=(20, 180)) as response:
                response.raise_for_status()
                expected_range = f"bytes {start}-{end}/{total}"
                actual_range = response.headers.get("Content-Range")
                if response.status_code != 206 or actual_range != expected_range:
                    raise RuntimeError(
                        f"Invalid range response for {start}-{end}: status={response.status_code}, "
                        f"Content-Range={actual_range!r}, expected={expected_range!r}"
                    )
                with temporary.open("wb") as stream:
                    for block in response.iter_content(chunk_size=1024 * 1024):
                        if block:
                            stream.write(block)
            actual = temporary.stat().st_size
            if actual != expected:
                raise IOError(f"Part {start}-{end} has {actual} bytes, expected {expected}")
            os.replace(temporary, part_path)
            return "downloaded"
        except Exception as exc:  # network/proxy failures are expected to be transient
            last_error = exc
            temporary.unlink(missing_ok=True)
            time.sleep(min(2 ** min(attempt, 4), 15))
    raise RuntimeError(f"Failed part {start}-{end} after {retries} attempts: {last_error}")


def main() -> None:
    args = parse_args()
    if args.chunk_size <= 0 or args.workers <= 0 or args.retries <= 0:
        raise ValueError("--chunk-size, --workers, and --retries must be positive")
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    headers = auth_headers(args.token_file)
    remote_size = probe_size(args.url, headers, args.retries)
    if args.expected_size is not None and remote_size != args.expected_size:
        raise ValueError(f"Remote size {remote_size} does not match --expected-size {args.expected_size}")
    if output.is_file() and output.stat().st_size == remote_size:
        print(f"Already complete: {output} ({remote_size} bytes)")
        return

    chunks_dir = output.parent / f".{output.name}.chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    num_parts = math.ceil(remote_size / args.chunk_size)
    lock = threading.Lock()
    completed = 0
    downloaded = 0

    def task(index: int) -> str:
        start = index * args.chunk_size
        end = min(remote_size - 1, start + args.chunk_size - 1)
        return download_part(
            url=args.url,
            headers=headers,
            part_path=chunks_dir / f"part_{index:05d}",
            start=start,
            end=end,
            total=remote_size,
            retries=args.retries,
        )

    print(
        f"Downloading {remote_size} bytes in {num_parts} validated parts "
        f"with {args.workers} workers",
        flush=True,
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(task, index): index for index in range(num_parts)}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            with lock:
                completed += 1
                downloaded += result == "downloaded"
                print(f"parts {completed}/{num_parts} (new {downloaded})", flush=True)

    assembled = output.with_suffix(output.suffix + ".assembling")
    with assembled.open("wb") as destination:
        for index in range(num_parts):
            part = chunks_dir / f"part_{index:05d}"
            with part.open("rb") as source:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
    if assembled.stat().st_size != remote_size:
        raise IOError(f"Assembled file has {assembled.stat().st_size} bytes, expected {remote_size}")
    os.replace(assembled, output)
    print(f"Complete: {output} ({remote_size} bytes)", flush=True)


if __name__ == "__main__":
    main()
