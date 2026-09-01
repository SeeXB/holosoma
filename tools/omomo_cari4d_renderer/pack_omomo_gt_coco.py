#!/usr/bin/env python3
"""Pack projected OMOMO GT COCO joints for an explicitly oracle diagnostic.

This does not replace the requested RGB keypoint detector result.  It permits
downstream pristine-CARI4D integration checks while Sapiens access is blocked.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import joblib
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-npz", type=Path, required=True)
    parser.add_argument("--camera-json", type=Path, required=True)
    parser.add_argument("--regressor", type=Path, required=True)
    parser.add_argument("--sequence-alias", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--output-resolution", type=int)
    args = parser.parse_args()

    sequence = np.load(args.sequence_npz)
    vertices = np.asarray(sequence["human_vertices"], dtype=np.float64)
    regressor = np.load(args.regressor).astype(np.float64)
    camera = json.loads(args.camera_json.read_text(encoding="utf-8"))
    world_to_camera = np.asarray(camera["world_to_camera"], dtype=np.float64)
    intrinsics = np.asarray(camera["intrinsics"], dtype=np.float64)
    resolution = int(camera["resolution"])
    if args.output_resolution is not None:
        intrinsics[:2] *= args.output_resolution / resolution
        resolution = args.output_resolution

    joints_world = np.einsum("jv,tvc->tjc", regressor, vertices)
    homogeneous = np.concatenate(
        [joints_world, np.ones((*joints_world.shape[:2], 1))], axis=-1
    )
    joints_camera = homogeneous @ world_to_camera.T
    xyz = joints_camera[..., :3]
    # Blender camera coordinates use +Y upward, while image coordinates use
    # +Y downward. Convert to the OpenCV convention before applying K.
    xyz_opencv = xyz.copy()
    xyz_opencv[..., 1] *= -1.0
    projected = xyz_opencv @ intrinsics.T
    xy = projected[..., :2] / projected[..., 2:3]
    valid = (
        (xyz[..., 2] > 0)
        & (xy[..., 0] >= 0)
        & (xy[..., 0] < resolution)
        & (xy[..., 1] >= 0)
        & (xy[..., 1] < resolution)
    )
    joints2d = np.concatenate([xy, valid[..., None].astype(np.float64)], axis=-1)
    joints2d = joints2d[:, None]

    args.output_root.mkdir(parents=True, exist_ok=True)
    output = args.output_root / f"{args.sequence_alias}_GT-packed.pkl"
    frames = [f"{frame:06d}" for frame in range(len(vertices))]
    joblib.dump({"frames": frames, "joints2d": joints2d}, output)

    overlay_path = None
    if args.video is not None:
        overlay_path = args.output_root / f"{args.sequence_alias}_oracle_coco_overlay.mp4"
        capture = cv2.VideoCapture(str(args.video))
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(
            str(overlay_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        for frame_index in range(len(vertices)):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Video ended at frame {frame_index}")
            for x, y, confidence in joints2d[frame_index, 0]:
                if confidence > 0:
                    cv2.circle(frame, (round(x), round(y)), 5, (0, 0, 255), -1)
            writer.write(frame)
        capture.release()
        writer.release()

    report = {
        "schema": "holosoma.omomo_oracle_coco.v1",
        "status": "DIAGNOSTIC_ONLY_NOT_RGB_INFERENCE",
        "sequence_alias": args.sequence_alias,
        "source_vertices": str(args.sequence_npz.resolve()),
        "source_camera": str(args.camera_json.resolve()),
        "regressor": str(args.regressor.resolve()),
        "output_resolution": resolution,
        "output": str(output.resolve()),
        "shape": list(joints2d.shape),
        "valid_keypoint_ratio": float(valid.mean()),
        "overlay_video": str(overlay_path.resolve()) if overlay_path else None,
        "may_satisfy_final_rgb_keypoint_acceptance": False,
    }
    report_path = args.output_root / f"{args.sequence_alias}_oracle_coco_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
