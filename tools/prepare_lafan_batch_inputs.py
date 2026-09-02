#!/usr/bin/env python3
"""Prepare a uniformly downsampled LAFAN1 batch for practical retargeting.

The raw LAFAN clips contain 3k--9k frames, which makes the convex retargeter
prohibitively slow for a 39-sequence sweep.  We retain every ``stride``-th
frame, using the same input directory for both methods and recording the
protocol in ``manifest.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


TASKS = (
    "walk1_subject1", "walk3_subject4", "run1_subject2", "run2_subject4",
    "sprint1_subject2", "sprint1_subject4", "jumps1_subject1", "jumps1_subject2",
    "jumps1_subject5", "fallAndGetUp1_subject1", "fallAndGetUp2_subject3",
    "fallAndGetUp3_subject1", "ground1_subject1", "ground1_subject4", "ground2_subject3",
    "dance1_subject1", "dance2_subject4", "fight1_subject2", "fightAndSports1_subject4",
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=Path("src/holosoma_retargeting/holosoma_retargeting/demo_data/lafan"))
    ap.add_argument("--output", type=Path, default=Path("exp/retargeting/lafan_batch/input"))
    ap.add_argument("--stride", type=int, default=20)
    args = ap.parse_args()
    if args.stride < 1:
        raise ValueError("stride must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {"stride": args.stride, "tasks": {}, "source": str(args.source.resolve())}
    for task in TASKS:
        src = args.source / f"{task}.npy"
        if not src.exists():
            raise FileNotFoundError(src)
        arr = np.load(src)
        out = arr[:: args.stride]
        np.save(args.output / f"{task}.npy", out)
        manifest["tasks"][task] = {"source_frames": int(arr.shape[0]), "frames": int(out.shape[0])}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"tasks": len(TASKS), "stride": args.stride, "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
