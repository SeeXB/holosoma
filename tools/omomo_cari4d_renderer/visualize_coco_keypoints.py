#!/usr/bin/env python3
"""Visualize packed COCO keypoints without changing CARI4D inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import joblib
import numpy as np


COCO_EDGES = (
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--packed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--reference-packed", type=Path)
    parser.add_argument("--confidence-threshold", type=float, default=0.3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = joblib.load(args.packed)
    joints = np.asarray(data["joints2d"], dtype=np.float64)[:, 0]

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count != len(joints):
        raise ValueError(f"Frame mismatch: video={frame_count}, keypoints={len(joints)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    for frame_index in range(frame_count):
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Failed to read frame {frame_index}")
        points = joints[frame_index]
        valid = points[:, 2] >= args.confidence_threshold
        for first, second in COCO_EDGES:
            if valid[first] and valid[second]:
                p1 = tuple(np.rint(points[first, :2]).astype(int))
                p2 = tuple(np.rint(points[second, :2]).astype(int))
                cv2.line(frame, p1, p2, (40, 220, 40), 3, cv2.LINE_AA)
        for joint_index, (x, y, confidence) in enumerate(points):
            if confidence < args.confidence_threshold:
                continue
            color = (40, 40, 255) if joint_index <= 4 else (255, 60, 30)
            cv2.circle(frame, (round(x), round(y)), 5, color, -1, cv2.LINE_AA)
        cv2.putText(
            frame, f"Sapiens COCO | frame {frame_index:06d}", (24, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2, cv2.LINE_AA,
        )
        writer.write(frame)
    cap.release()
    writer.release()

    confidence = joints[:, :, 2]
    inside = (
        (joints[:, :, 0] >= 0) & (joints[:, :, 0] < width)
        & (joints[:, :, 1] >= 0) & (joints[:, :, 1] < height)
    )
    jumps = np.linalg.norm(np.diff(joints[:, :, :2], axis=0), axis=2)
    report: dict[str, object] = {
        "schema": "holosoma.coco_keypoint_diagnostic.v1",
        "video": str(args.video.resolve()),
        "packed": str(args.packed.resolve()),
        "output": str(args.output.resolve()),
        "shape": list(data["joints2d"].shape),
        "confidence_mean": float(confidence.mean()),
        "confidence_min": float(confidence.min()),
        "confidence_p05": float(np.percentile(confidence, 5)),
        "inside_frame_ratio": float(inside.mean()),
        "temporal_jump_px_mean": float(jumps.mean()),
        "temporal_jump_px_p95": float(np.percentile(jumps, 95)),
        "temporal_jump_px_max": float(jumps.max()),
    }
    if args.reference_packed is not None:
        reference = np.asarray(
            joblib.load(args.reference_packed)["joints2d"], dtype=np.float64
        )[:, 0]
        if reference.shape != joints.shape:
            raise ValueError(f"Reference shape mismatch: {reference.shape} vs {joints.shape}")
        error = np.linalg.norm(joints[:, :, :2] - reference[:, :, :2], axis=2)
        report["reference_packed"] = str(args.reference_packed.resolve())
        report["reference_pixel_error_mean"] = float(error.mean())
        report["reference_pixel_error_median"] = float(np.median(error))
        report["reference_pixel_error_p95"] = float(np.percentile(error, 95))

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
