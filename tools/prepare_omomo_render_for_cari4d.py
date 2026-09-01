#!/usr/bin/env python3
"""Legacy sample adapter for color-coded OMOMO renderings.

OMOMO's ``motion_videos`` render the person in blue and the manipulated object
in pink.  For these synthetic videos, color segmentation is both more accurate
and more reproducible than a text-prompted segmentation model.  This utility
is not the general video preprocessing entry point. New callers should use
``prepare_cari4d_video.py`` (SAM3 by default) and explicitly opt into its
``color-key`` backend for similarly color-coded synthetic data. This adapter
only reads RGB pixels; it deliberately does not read OMOMO SMPL-X, object pose,
or retargeted trajectory files and is retained to reproduce the existing sample.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--sequence-name",
        default="OMOMO_Sub03_largebox003",
        help="CARI4D-compatible alias; token 2 must be a known subject such as Sub03.",
    )
    parser.add_argument("--human-hue", nargs=2, type=int, default=(95, 130), metavar=("MIN", "MAX"))
    parser.add_argument("--object-hue", nargs=2, type=int, default=(132, 175), metavar=("MIN", "MAX"))
    parser.add_argument("--min-saturation", type=int, default=35)
    parser.add_argument("--min-value", type=int, default=35)
    parser.add_argument("--morph-kernel", type=int, default=1)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if not args.video.is_file():
        raise FileNotFoundError(args.video)
    tokens = args.sequence_name.split("_")
    if len(tokens) < 3:
        raise ValueError("--sequence-name needs at least three underscore-separated tokens")
    if not (0 <= args.human_hue[0] <= args.human_hue[1] <= 179):
        raise ValueError(f"Invalid --human-hue: {args.human_hue}")
    if not (0 <= args.object_hue[0] <= args.object_hue[1] <= 179):
        raise ValueError(f"Invalid --object-hue: {args.object_hue}")
    if not (0 <= args.min_saturation <= 255 and 0 <= args.min_value <= 255):
        raise ValueError("HSV saturation/value thresholds must be in [0,255]")
    if args.morph_kernel < 1 or args.morph_kernel % 2 == 0:
        raise ValueError("--morph-kernel must be a positive odd integer")


def _color_mask(
    hsv: np.ndarray,
    hue_range: tuple[int, int] | list[int],
    min_saturation: int,
    min_value: int,
    kernel: np.ndarray,
) -> np.ndarray:
    lower = np.array([hue_range[0], min_saturation, min_value], dtype=np.uint8)
    upper = np.array([hue_range[1], 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    if kernel.size > 1:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask.astype(bool)


def main() -> None:
    args = parse_args()
    args.video = args.video.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    _validate_args(args)

    videos_dir = args.output_root / "videos"
    masks_dir = args.output_root / "masks"
    validation_dir = args.output_root / "validation"
    for directory in (videos_dir, masks_dir, validation_dir):
        directory.mkdir(parents=True, exist_ok=True)

    staged_video = videos_dir / f"{args.sequence_name}.0.color.mp4"
    if staged_video.resolve() != args.video:
        shutil.copy2(args.video, staged_video)

    capture = cv2.VideoCapture(str(staged_video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {staged_video}")
    frame_count_claimed = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frame_count_claimed <= 0 or fps <= 0 or width <= 0 or height <= 0:
        raise RuntimeError("Video metadata is invalid")

    overlay_path = validation_dir / "color_masks_overlay.mp4"
    writer = cv2.VideoWriter(
        str(overlay_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width * 2, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create {overlay_path}")

    h5_path = masks_dir / f"{args.sequence_name}_masks_k0.h5"
    kernel = np.ones((args.morph_kernel, args.morph_kernel), dtype=np.uint8)
    human_counts: list[int] = []
    object_counts: list[int] = []
    resolved_overlap_counts: list[int] = []

    try:
        with h5py.File(h5_path, "w") as h5_file:
            group = h5_file.create_group(args.sequence_name)
            frame_index = 0
            while True:
                ok, frame_bgr = capture.read()
                if not ok:
                    break
                hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
                human = _color_mask(
                    hsv, args.human_hue, args.min_saturation, args.min_value, kernel
                )
                object_mask = _color_mask(
                    hsv, args.object_hue, args.min_saturation, args.min_value, kernel
                )
                overlap = human & object_mask
                if overlap.any():
                    # Morphology can expand the two masks into the same boundary
                    # pixels. Resolve only those pixels using their original hue;
                    # the raw HSV observations remain the sole source of labels.
                    hue = hsv[..., 0].astype(np.float32)
                    human_center = 0.5 * (args.human_hue[0] + args.human_hue[1])
                    object_center = 0.5 * (args.object_hue[0] + args.object_hue[1])
                    overlap_to_human = overlap & (
                        np.abs(hue - human_center) <= np.abs(hue - object_center)
                    )
                    object_mask[overlap_to_human] = False
                    human[overlap & ~overlap_to_human] = False
                resolved_overlap_counts.append(int(overlap.sum()))
                human_count = int(human.sum())
                object_count = int(object_mask.sum())
                if human_count < 100 or object_count < 20:
                    raise RuntimeError(
                        f"Implausibly small mask at frame {frame_index}: "
                        f"human={human_count}, object={object_count}"
                    )
                frame_id = f"{frame_index:06d}"
                group.create_dataset(
                    f"{frame_id}-k0.person_mask.png", data=human, compression="gzip"
                )
                group.create_dataset(
                    f"{frame_id}-k0.obj_rend_mask.png", data=object_mask, compression="gzip"
                )
                human_counts.append(human_count)
                object_counts.append(object_count)

                overlay = frame_bgr.copy()
                overlay[human] = (0.55 * overlay[human] + 0.45 * np.array([0, 255, 0])).astype(
                    np.uint8
                )
                overlay[object_mask] = (
                    0.55 * overlay[object_mask] + 0.45 * np.array([0, 165, 255])
                ).astype(np.uint8)
                writer.write(np.concatenate([frame_bgr, overlay], axis=1))
                frame_index += 1
    finally:
        capture.release()
        writer.release()

    if len(human_counts) != frame_count_claimed:
        raise RuntimeError(
            f"Decoded {len(human_counts)} frames, video metadata reports {frame_count_claimed}"
        )

    report = {
        "status": "PASS",
        "source_video": str(args.video),
        "staged_video": str(staged_video),
        "sequence_name": args.sequence_name,
        "frames": len(human_counts),
        "fps": fps,
        "resolution_wh": [width, height],
        "mask_source": "RGB HSV color segmentation of the OMOMO rendered video",
        "uses_omomo_motion_or_object_pose_labels": False,
        "human_hue_opencv": list(args.human_hue),
        "object_hue_opencv": list(args.object_hue),
        "min_saturation": args.min_saturation,
        "min_value": args.min_value,
        "morph_kernel": args.morph_kernel,
        "human_mask_pixels_min_max": [min(human_counts), max(human_counts)],
        "object_mask_pixels_min_max": [min(object_counts), max(object_counts)],
        "morphology_overlap_pixels_total": sum(resolved_overlap_counts),
        "mask_h5": str(h5_path),
        "overlay_video": str(overlay_path),
    }
    report_path = validation_dir / "color_mask_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
