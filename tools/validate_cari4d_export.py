#!/usr/bin/env python3
"""Validate one canonical CARI4D sequence without modifying its data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hoi_pipeline.canonical_validation import validate_and_write
from hoi_pipeline.common import json_ready


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", required=True, type=Path, help="Canonical sequence directory.")
    parser.add_argument("--video-frame-count", type=int, help="Optional source video frame count cross-check.")
    args = parser.parse_args()
    sequence = args.sequence.expanduser().resolve()
    video_frame_count = args.video_frame_count
    if video_frame_count is None:
        sequence_metadata = sequence / "meta" / "sequence.json"
        if sequence_metadata.is_file():
            metadata = json.loads(sequence_metadata.read_text(encoding="utf-8"))
            stored_count = metadata.get("source_video_num_frames")
            if stored_count is not None:
                video_frame_count = int(stored_count)
    report = validate_and_write(sequence, video_frame_count=video_frame_count)
    print(json.dumps(json_ready(report), indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
