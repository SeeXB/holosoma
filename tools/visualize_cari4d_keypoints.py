#!/usr/bin/env python3
"""Overlay a CARI4D packed COCO-17 trajectory on its source video."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import cv2
import joblib
import numpy as np


COCO_EDGES = (
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (0, 1), (0, 2), (1, 3), (2, 4),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--packed", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--confidence", type=float, default=0.2)
    args = parser.parse_args()

    packed = joblib.load(args.packed)
    joints = np.asarray(packed["joints2d"], dtype=np.float64)
    if joints.ndim != 4 or joints.shape[1:] != (1, 17, 3):
        raise ValueError(f"Expected joints2d [T,1,17,3], got {joints.shape}")

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {args.video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count != len(joints):
        raise ValueError(f"Video has {frame_count} frames but packed trajectory has {len(joints)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    intermediate = args.output.with_name(args.output.stem + ".mpeg4-tmp.mp4")
    writer = cv2.VideoWriter(
        str(intermediate), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create {intermediate}")

    for frame_index in range(frame_count):
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Could not read video frame {frame_index}")
        keypoints = joints[frame_index, 0]
        for first, second in COCO_EDGES:
            if keypoints[first, 2] >= args.confidence and keypoints[second, 2] >= args.confidence:
                p0 = tuple(np.rint(keypoints[first, :2]).astype(int))
                p1 = tuple(np.rint(keypoints[second, :2]).astype(int))
                cv2.line(frame, p0, p1, (0, 255, 255), 2, cv2.LINE_AA)
        for x, y, confidence in keypoints:
            if confidence >= args.confidence:
                cv2.circle(frame, (int(round(x)), int(round(y))), 4, (0, 0, 255), -1, cv2.LINE_AA)
        writer.write(frame)
    capture.release()
    writer.release()

    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(intermediate),
            "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", "-an", "-y", str(args.output),
        ],
        check=True,
    )
    intermediate.unlink()
    print(f"Saved {frame_count}-frame COCO-17 overlay: {args.output}")


if __name__ == "__main__":
    main()
