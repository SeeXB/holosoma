#!/usr/bin/env python3
"""Compare official CARI4D SAM3 masks with Blender GT masks."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from common import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam3-h5", type=Path, required=True)
    parser.add_argument("--gt-h5", type=Path, required=True)
    parser.add_argument("--sequence", default="sub03_largebox3")
    parser.add_argument("--selected-frame", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def iou(left: np.ndarray, right: np.ndarray) -> float:
    left = left.astype(bool)
    right = right.astype(bool)
    union = np.logical_or(left, right).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(left, right).sum() / union)


def summarize(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p05": float(np.quantile(values, 0.05)),
        "minimum": float(values.min()),
    }


def main() -> None:
    args = parse_args()
    rows = []
    with h5py.File(args.sam3_h5, "r") as sam3, h5py.File(args.gt_h5, "r") as gt:
        sam_group = sam3[args.sequence]
        gt_group = gt[args.sequence]
        frame_ids = sorted(
            int(key.split("-", 1)[0])
            for key in sam_group.keys()
            if key.endswith(".person_mask.png")
        )
        if not frame_ids:
            raise ValueError(f"No person masks found in {args.sam3_h5}")
        for frame in frame_ids:
            prefix = f"{frame:06d}-k0"
            human_key = f"{prefix}.person_mask.png"
            object_key = f"{prefix}.obj_rend_mask.png"
            sam_human = sam_group[human_key][:].astype(bool)
            gt_human = gt_group[human_key][:].astype(bool)
            sam_object = sam_group[object_key][:].astype(bool)
            gt_object = gt_group[object_key][:].astype(bool)
            rows.append(
                {
                    "frame": frame,
                    "human_iou": iou(sam_human, gt_human),
                    "object_iou": iou(sam_object, gt_object),
                    "human_pred_pixels": int(sam_human.sum()),
                    "human_gt_pixels": int(gt_human.sum()),
                    "object_pred_pixels": int(sam_object.sum()),
                    "object_gt_pixels": int(gt_object.sum()),
                }
            )
    human_values = np.asarray([row["human_iou"] for row in rows])
    object_values = np.asarray([row["object_iou"] for row in rows])
    selected = next(row for row in rows if row["frame"] == args.selected_frame)
    report = {
        "sam3_h5": args.sam3_h5,
        "gt_h5": args.gt_h5,
        "sequence": args.sequence,
        "num_frames": len(rows),
        "human_iou": summarize(human_values),
        "object_iou": summarize(object_values),
        "human_empty_prediction_frames": int(
            sum(row["human_pred_pixels"] == 0 for row in rows)
        ),
        "object_empty_prediction_frames": int(
            sum(row["object_pred_pixels"] == 0 for row in rows)
        ),
        "selected_frame": args.selected_frame,
        "selected_frame_human_iou": selected["human_iou"],
        "selected_frame_object_iou": selected["object_iou"],
        "frames": rows,
    }
    write_json(args.output, report)
    print(args.output.read_text())


if __name__ == "__main__":
    main()
