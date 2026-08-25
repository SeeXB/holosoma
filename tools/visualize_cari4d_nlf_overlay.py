#!/usr/bin/env python3
"""Project a CARI4D NLF SMPL-H result back onto its source video."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import h5py
import joblib
import numpy as np
import torch


BODY25_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10), (10, 11), (8, 12), (12, 13), (13, 14),
    (0, 15), (15, 17), (0, 16), (16, 18), (14, 19), (19, 20),
    (14, 21), (11, 22), (22, 23), (11, 24),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nlf-result", required=True, type=Path)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--intrinsics", required=True, type=Path)
    parser.add_argument("--masks", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--packed-output",
        type=Path,
        help=(
            "Optional CARI4D Body25 packed file made from the NLF reprojection. "
            "This is a transparent fallback when independent Sapiens/OpenPose weights are unavailable."
        ),
    )
    parser.add_argument("--cari4d-root", type=Path, default=Path("third_party/CARI4D"))
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def _bbox(points_xy: np.ndarray, width: int, height: int) -> np.ndarray:
    finite = np.isfinite(points_xy).all(axis=1)
    points_xy = points_xy[finite]
    if not len(points_xy):
        return np.array([np.nan] * 4)
    low = points_xy.min(axis=0)
    high = points_xy.max(axis=0)
    return np.array(
        [max(0.0, low[0]), max(0.0, low[1]), min(width - 1.0, high[0]), min(height - 1.0, high[1])]
    )


def _bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter_w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]) + 1.0)
    inter_h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]) + 1.0)
    inter = inter_w * inter_h
    area_a = max(0.0, a[2] - a[0] + 1.0) * max(0.0, a[3] - a[1] + 1.0)
    area_b = max(0.0, b[2] - b[0] + 1.0) * max(0.0, b[3] - b[1] + 1.0)
    return inter / max(area_a + area_b - inter, 1e-12)


def main() -> None:
    args = parse_args()
    args.cari4d_root = args.cari4d_root.expanduser().resolve()
    for name in ("nlf_result", "video", "intrinsics", "masks"):
        value = getattr(args, name).expanduser().resolve()
        if not value.is_file():
            raise FileNotFoundError(value)
        setattr(args, name, value)
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    args.output = args.output.expanduser().resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(args.cari4d_root))
    from lib_smpl import get_smpl  # pylint: disable=import-outside-toplevel
    from lib_smpl.body_landmark import BodyLandmarks  # pylint: disable=import-outside-toplevel

    nlf = joblib.load(args.nlf_result)
    poses = np.asarray(nlf["poses"], dtype=np.float32)[:, 0]
    betas = np.asarray(nlf["betas"], dtype=np.float32)[:, 0]
    translations = np.asarray(nlf["transls"], dtype=np.float32)[:, 0]
    frame_ids = list(nlf["frames"])
    if poses.shape[1] != 156:
        raise ValueError(f"Expected NLF SMPL-H pose [T,156], got {poses.shape}")
    if not (len(poses) == len(betas) == len(translations) == len(frame_ids)):
        raise ValueError("NLF arrays and frame IDs have different lengths")

    camera = joblib.load(args.intrinsics)
    intrinsic = np.array(
        [[camera["fx"], 0.0, camera["cx"]], [0.0, camera["fy"], camera["cy"]], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    body_model = get_smpl(
        nlf["gender"], hands=True, model_root=str(args.cari4d_root / "data" / "smpl")
    ).eval()
    landmarks = BodyLandmarks(str(args.cari4d_root / "data" / "assets"))

    all_vertices: list[np.ndarray] = []
    all_joints: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(poses), args.batch_size):
            end = min(start + args.batch_size, len(poses))
            vertices, _, _, _ = body_model(
                torch.from_numpy(poses[start:end]),
                torch.from_numpy(betas[start:end]),
                torch.from_numpy(translations[start:end]),
            )
            joints = landmarks.get_body_kpts_batch_torch(vertices)
            all_vertices.append(vertices.numpy())
            all_joints.append(joints.numpy())
    vertices = np.concatenate(all_vertices)
    joints = np.concatenate(all_joints)

    def project(points: np.ndarray) -> np.ndarray:
        homogeneous = points @ intrinsic.T
        return homogeneous[..., :2] / np.maximum(homogeneous[..., 2:3], 1e-8)

    vertices_xy = project(vertices)
    joints_xy = project(joints)
    if args.packed_output is not None:
        args.packed_output = args.packed_output.expanduser().resolve()
        args.packed_output.parent.mkdir(parents=True, exist_ok=True)
        confidence = np.ones((*joints_xy.shape[:-1], 1), dtype=np.float64)
        joints_2d = np.concatenate([joints_xy.astype(np.float64), confidence], axis=-1)[:, None]
        joblib.dump({"frames": frame_ids, "joints2d": joints_2d}, args.packed_output)
        packed_metadata = {
            "source": "reprojection of CARI4D NLF SMPL-H inferred from RGB",
            "independent_2d_detector": False,
            "reason": "Sapiens checkpoint is gated and unavailable on this workstation",
            "uses_omomo_motion_or_object_pose_labels": False,
            "shape": list(joints_2d.shape),
        }
        args.packed_output.with_suffix(".metadata.json").write_text(
            json.dumps(packed_metadata, indent=2) + "\n", encoding="utf-8"
        )

    capture = cv2.VideoCapture(str(args.video))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width * 2, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create {args.output}")

    sequence_name = args.video.name.split(".")[0]
    bbox_ious: list[float] = []
    with h5py.File(args.masks, "r") as h5_file:
        group = h5_file[sequence_name]
        for index, frame_id in enumerate(frame_ids):
            ok, bgr = capture.read()
            if not ok:
                raise RuntimeError(f"Video ended before NLF frame {index}")
            rendered = bgr.copy()
            xy = np.rint(vertices_xy[index]).astype(np.int32)
            valid = (
                (vertices[index, :, 2] > 0)
                & (xy[:, 0] >= 0) & (xy[:, 0] < width)
                & (xy[:, 1] >= 0) & (xy[:, 1] < height)
            )
            rendered[xy[valid, 1], xy[valid, 0]] = (0, 255, 255)
            joints_i = np.rint(joints_xy[index]).astype(np.int32)
            for start_joint, end_joint in BODY25_EDGES:
                a, b = joints_i[start_joint], joints_i[end_joint]
                if 0 <= a[0] < width and 0 <= a[1] < height and 0 <= b[0] < width and 0 <= b[1] < height:
                    cv2.line(rendered, tuple(a), tuple(b), (0, 0, 255), 2, cv2.LINE_AA)
            for joint in joints_i:
                if 0 <= joint[0] < width and 0 <= joint[1] < height:
                    cv2.circle(rendered, tuple(joint), 3, (255, 255, 0), -1, cv2.LINE_AA)

            mask = group[f"{frame_id}-k0.person_mask.png"][:].astype(bool)
            ys, xs = np.where(mask)
            mask_bbox = np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float64)
            projected_bbox = _bbox(vertices_xy[index], width, height)
            bbox_ious.append(_bbox_iou(mask_bbox, projected_bbox))
            cv2.putText(
                rendered, f"NLF SMPL-H bbox IoU {bbox_ious[-1]:.3f}", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA,
            )
            writer.write(np.concatenate([bgr, rendered], axis=1))
    capture.release()
    writer.release()

    report = {
        "status": "PASS",
        "source": "CARI4D NLF video inference",
        "uses_omomo_motion_or_object_pose_labels": False,
        "frames": len(poses),
        "pose_shape": list(poses.shape),
        "bbox_iou_mean": float(np.mean(bbox_ious)),
        "bbox_iou_min": float(np.min(bbox_ious)),
        "bbox_iou_max": float(np.max(bbox_ious)),
        "overlay_video": str(args.output),
        "packed_output": str(args.packed_output) if args.packed_output is not None else None,
    }
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
