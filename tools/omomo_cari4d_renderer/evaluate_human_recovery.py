#!/usr/bin/env python3
"""Compare CARI4D SMPL-H fits against OMOMO GT and their 2D observations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import joblib
import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cari4d-root", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--case", action="append", nargs=5, required=True,
        metavar=("NAME", "FIT_PKL", "PACKED_PKL", "MASK_H5", "CAMERA_PKL"),
    )
    return parser.parse_args()


def similarity_align(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Least-squares similarity transform for row-vector point arrays."""
    source_mean = source.mean(axis=0, keepdims=True)
    target_mean = target.mean(axis=0, keepdims=True)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = source_centered.T @ target_centered
    left, singular, right_t = np.linalg.svd(covariance)
    correction = np.ones(3)
    if np.linalg.det(left @ right_t) < 0:
        correction[-1] = -1
    rotation = (left * correction[None]) @ right_t
    scale = float((singular * correction).sum() / np.square(source_centered).sum())
    return scale * source_centered @ rotation + target_mean


def bbox_heights(mask_path: Path, sequence: str, frame_count: int) -> np.ndarray:
    heights = []
    with h5py.File(mask_path, "r") as handle:
        group = handle[sequence]
        for frame in range(frame_count):
            mask = np.asarray(group[f"{frame:06d}-k0.person_mask.png"])
            ys = np.where(mask)[0]
            heights.append(float(ys.max() - ys.min() + 1))
    return np.asarray(heights)


def main() -> None:
    args = parse_args()
    cari4d_root = args.cari4d_root.resolve()
    sys.path.insert(0, str(cari4d_root))
    import smplx  # noqa: PLC0415
    from lib_smpl import SMPL_ASSETS_ROOT, SMPL_MODEL_ROOT  # noqa: PLC0415

    gt_data = np.load(args.gt)
    gt_joints = np.asarray(gt_data["human_joints"][:, :22], dtype=np.float64)
    gt_relative = gt_joints - gt_joints[:, :1]
    frame_count = len(gt_joints)
    coco_regressor = np.load(Path(SMPL_ASSETS_ROOT) / "J_regressor_coco.npy")
    results: dict[str, object] = {
        "schema": "holosoma.omomo_cari4d_human_recovery_comparison.v1",
        "gt": str(args.gt.resolve()),
        "body_joint_count": 22,
        "alignment": (
            "One sequence-level similarity transform after per-frame pelvis removal; "
            "PA-MPJPE additionally uses one similarity transform per frame."
        ),
        "cases": {},
    }

    for name, fit_path, packed_path, mask_path, camera_path in args.case:
        fit = joblib.load(fit_path)
        pose = torch.from_numpy(np.asarray(fit["poses"][:, 0])).float()
        betas = torch.from_numpy(np.asarray(fit["betas"][:, 0])).float()
        translation = torch.from_numpy(np.asarray(fit["transls"][:, 0])).float()
        if len(pose) != frame_count:
            raise ValueError(f"{name}: frame mismatch {len(pose)} != {frame_count}")
        model = smplx.create(
            model_path=SMPL_MODEL_ROOT,
            model_type="smplh",
            gender=fit["gender"],
            use_pca=False,
            batch_size=frame_count,
            flat_hand_mean=True,
        )
        with torch.no_grad():
            body = model(
                betas=betas,
                global_orient=pose[:, :3],
                body_pose=pose[:, 3:66],
                left_hand_pose=pose[:, 66:111],
                right_hand_pose=pose[:, 111:156],
                transl=translation,
                return_verts=True,
            )
        vertices = body.vertices.cpu().numpy().astype(np.float64)
        predicted_joints = body.joints[:, :22].cpu().numpy().astype(np.float64)
        predicted_relative = predicted_joints - predicted_joints[:, :1]

        globally_aligned = similarity_align(
            predicted_relative.reshape(-1, 3), gt_relative.reshape(-1, 3)
        ).reshape(predicted_relative.shape)
        global_error = np.linalg.norm(globally_aligned - gt_relative, axis=2)
        pa_error = np.stack([
            np.linalg.norm(similarity_align(predicted_relative[i], gt_relative[i]) - gt_relative[i], axis=1)
            for i in range(frame_count)
        ])
        acceleration_error = np.linalg.norm(
            np.diff(globally_aligned, n=2, axis=0) - np.diff(gt_relative, n=2, axis=0),
            axis=2,
        )

        packed = joblib.load(packed_path)
        observed = np.asarray(packed["joints2d"][:, 0], dtype=np.float64)
        camera = joblib.load(camera_path)
        intrinsic = np.array([
            [camera["fx"], 0.0, camera["cx"]],
            [0.0, camera["fy"], camera["cy"]],
            [0.0, 0.0, 1.0],
        ])
        coco_3d = np.einsum("jv,fvc->fjc", coco_regressor, vertices)
        homogeneous = coco_3d @ intrinsic.T
        projected = homogeneous[:, :, :2] / homogeneous[:, :, 2:3]
        valid = observed[:, :, 2] >= 0.4
        reprojection_error = np.linalg.norm(projected - observed[:, :, :2], axis=2)
        heights = bbox_heights(Path(mask_path), "OMOMO_Sub03_largebox_003", frame_count)
        normalized_reprojection = reprojection_error / heights[:, None]
        jumps = np.linalg.norm(np.diff(observed[:, :, :2], axis=0), axis=2)
        jump_scale = (heights[:-1] + heights[1:])[:, None] / 2.0

        case_result = {
            "fit": str(Path(fit_path).resolve()),
            "frames": frame_count,
            "person_bbox_height_px_mean": float(heights.mean()),
            "sapiens_confidence_mean": float(observed[:, :, 2].mean()),
            "sapiens_confidence_p05": float(np.percentile(observed[:, :, 2], 5)),
            "sapiens_temporal_jump_normalized_mean": float((jumps / jump_scale).mean()),
            "sapiens_temporal_jump_normalized_p95": float(np.percentile(jumps / jump_scale, 95)),
            "fit_reprojection_error_px_mean": float(reprojection_error[valid].mean()),
            "fit_reprojection_error_px_p95": float(np.percentile(reprojection_error[valid], 95)),
            "fit_reprojection_error_normalized_mean": float(normalized_reprojection[valid].mean()),
            "fit_reprojection_error_normalized_p95": float(np.percentile(normalized_reprojection[valid], 95)),
            "global_similarity_mpjpe_mm": float(global_error.mean() * 1000.0),
            "global_similarity_mpjpe_p95_mm": float(np.percentile(global_error, 95) * 1000.0),
            "pa_mpjpe_mm": float(pa_error.mean() * 1000.0),
            "pa_mpjpe_p95_mm": float(np.percentile(pa_error, 95) * 1000.0),
            "root_relative_acceleration_error_mm_per_frame2": float(acceleration_error.mean() * 1000.0),
        }
        results["cases"][name] = case_result

    case_names = list(results["cases"])
    if len(case_names) == 2:
        old, new = case_names
        improvements = {}
        for metric in (
            "fit_reprojection_error_normalized_mean",
            "global_similarity_mpjpe_mm",
            "pa_mpjpe_mm",
            "root_relative_acceleration_error_mm_per_frame2",
        ):
            old_value = results["cases"][old][metric]
            new_value = results["cases"][new][metric]
            improvements[metric] = float((old_value - new_value) / old_value)
        results["relative_improvement_fraction_old_to_new"] = improvements

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
