#!/usr/bin/env python3
"""Evaluate a normalized single-view Hunyuan mesh against OMOMO largebox GT."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from common import normalized_dimensions, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path(
            "src/holosoma_retargeting/holosoma_retargeting/demo_data/external/omomo/data/captured_objects/"
            "largebox_cleaned_simplified.obj"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = loaded.dump(concatenate=True)
    if not isinstance(loaded, trimesh.Trimesh) or len(loaded.vertices) == 0:
        raise ValueError(f"No triangle mesh found in {path}")
    return loaded


def sorted_obb_extents(mesh: trimesh.Trimesh) -> np.ndarray:
    return np.sort(np.asarray(mesh.bounding_box_oriented.extents, dtype=np.float64))[::-1]


def sample_surface(
    mesh: trimesh.Trimesh, count: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    # trimesh uses NumPy's global RNG internally.
    state = np.random.get_state()
    np.random.seed(int(rng.integers(0, 2**31 - 1)))
    try:
        points, face_indices = trimesh.sample.sample_surface(mesh, count)
    finally:
        np.random.set_state(state)
    return points, np.asarray(mesh.face_normals)[face_indices]


def principal_frame(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - center, full_matrices=False)
    axes = vt.T
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1
    return center, axes


def proper_axis_mappings() -> list[np.ndarray]:
    mappings = []
    for permutation in itertools.permutations(range(3)):
        base = np.eye(3)[:, permutation]
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            matrix = base @ np.diag(signs)
            if np.linalg.det(matrix) > 0.5:
                mappings.append(matrix)
    return mappings


def umeyama(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    source_zero = source - source_center
    target_zero = target - target_center
    covariance = target_zero.T @ source_zero / len(source)
    u, singular, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(u @ vt))
    rotation = u @ correction @ vt
    variance = np.mean(np.sum(source_zero**2, axis=1))
    scale = float(np.sum(singular * np.diag(correction)) / max(variance, 1.0e-12))
    translation = target_center - scale * (rotation @ source_center)
    return scale, rotation, translation


def transform(points: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return scale * (points @ rotation.T) + translation


def similarity_icp(
    predicted: np.ndarray, ground_truth: np.ndarray, iterations: int = 15
) -> tuple[np.ndarray, dict]:
    gt_center, gt_axes = principal_frame(ground_truth)
    pred_center, pred_axes = principal_frame(predicted)
    gt_scale = np.sqrt(np.mean(np.sum((ground_truth - gt_center) ** 2, axis=1)))
    pred_scale = np.sqrt(np.mean(np.sum((predicted - pred_center) ** 2, axis=1)))
    tree = cKDTree(ground_truth)
    best = None
    for mapping in proper_axis_mappings():
        initial_rotation = gt_axes @ mapping @ pred_axes.T
        initial_scale = gt_scale / max(pred_scale, 1.0e-12)
        initial_translation = gt_center - initial_scale * (initial_rotation @ pred_center)
        aligned = transform(predicted, initial_scale, initial_rotation, initial_translation)
        scale, rotation, translation = initial_scale, initial_rotation, initial_translation
        for _ in range(iterations):
            _, nearest = tree.query(aligned, workers=-1)
            delta_scale, delta_rotation, delta_translation = umeyama(
                aligned, ground_truth[nearest]
            )
            aligned = transform(aligned, delta_scale, delta_rotation, delta_translation)
            scale = delta_scale * scale
            rotation = delta_rotation @ rotation
            translation = delta_scale * (delta_rotation @ translation) + delta_translation
        distances, _ = tree.query(aligned, workers=-1)
        score = float(np.mean(distances**2))
        if best is None or score < best[0]:
            best = (score, aligned, scale, rotation, translation)
    assert best is not None
    return best[1], {
        "scale": float(best[2]),
        "rotation": best[3],
        "translation": best[4],
        "one_way_mse": float(best[0]),
    }


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    gt_mesh = load_mesh(args.ground_truth)
    pred_mesh = load_mesh(args.prediction)
    gt_points, gt_normals = sample_surface(gt_mesh, args.samples, rng)
    pred_points, pred_normals = sample_surface(pred_mesh, args.samples, rng)
    aligned_pred, alignment = similarity_icp(pred_points, gt_points)

    gt_tree = cKDTree(gt_points)
    pred_tree = cKDTree(aligned_pred)
    pred_to_gt, pred_nearest = gt_tree.query(aligned_pred, workers=-1)
    gt_to_pred, gt_nearest = pred_tree.query(gt_points, workers=-1)
    normalization = float(sorted_obb_extents(gt_mesh).max())
    chamfer = float(
        (np.mean(pred_to_gt**2) + np.mean(gt_to_pred**2))
        / max(normalization**2, 1.0e-12)
    )

    rotation = np.asarray(alignment["rotation"])
    aligned_pred_normals = pred_normals @ rotation.T
    consistency_forward = np.abs(
        np.sum(aligned_pred_normals * gt_normals[pred_nearest], axis=1)
    )
    consistency_backward = np.abs(
        np.sum(gt_normals * aligned_pred_normals[gt_nearest], axis=1)
    )
    normal_consistency = float(
        0.5 * (consistency_forward.mean() + consistency_backward.mean())
    )

    gt_dimensions = sorted_obb_extents(gt_mesh)
    pred_dimensions = sorted_obb_extents(pred_mesh)
    gt_ratio = normalized_dimensions(gt_dimensions)
    pred_ratio = normalized_dimensions(pred_dimensions)
    aspect_error = float(np.mean(np.abs(pred_ratio - gt_ratio) / np.maximum(gt_ratio, 1.0e-8)))
    passed = bool(aspect_error <= 0.15 and chamfer <= 0.035)
    report = {
        "prediction": args.prediction,
        "ground_truth": args.ground_truth,
        "sample_count": args.samples,
        "gt_oriented_bbox_dimensions": gt_dimensions,
        "prediction_oriented_bbox_dimensions": pred_dimensions,
        "gt_normalized_bbox_ratio": gt_ratio,
        "prediction_normalized_bbox_ratio": pred_ratio,
        "relative_aspect_ratio_error": aspect_error,
        "symmetric_chamfer_squared_normalized": chamfer,
        "normal_consistency_abs": normal_consistency,
        "similarity_alignment": alignment,
        "pass_thresholds": {
            "relative_aspect_ratio_error_max": 0.15,
            "symmetric_chamfer_squared_normalized_max": 0.035,
        },
        "pass": passed,
    }
    write_json(args.output, report)
    print(args.output.read_text())


if __name__ == "__main__":
    main()
