#!/usr/bin/env python3
"""Evaluate a CARI4D canonical sequence against a legacy InterMimic reference.

The reference is used only after reconstruction for diagnostics.  It is never
read by the video preprocessing, CARI4D inference, canonical exporter, or USD
converter.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation


BODY_JOINT_INDICES = np.asarray([*range(18), *range(33, 37)], dtype=np.int64)


def parse_args() -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", required=True, type=Path)
    parser.add_argument("--reference-pt", required=True, type=Path)
    parser.add_argument("--reference-object-mesh", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cari4d-root", type=Path, default=repository / "third_party" / "CARI4D")
    return parser.parse_args()


def fit_similarity(source: np.ndarray, target: np.ndarray, *, estimate_scale: bool) -> tuple[float, np.ndarray, np.ndarray]:
    """Fit target ~= scale * source @ rotation.T + translation."""
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    u, singular_values, vt = np.linalg.svd(source_centered.T @ target_centered)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1] *= -1.0
        rotation = vt.T @ u.T
    scale = (
        float(singular_values.sum() / np.square(source_centered).sum())
        if estimate_scale
        else 1.0
    )
    translation = target_mean - scale * source_mean @ rotation.T
    return scale, rotation, translation


def transform_points(points: np.ndarray, fit: tuple[float, np.ndarray, np.ndarray]) -> np.ndarray:
    scale, rotation, translation = fit
    return scale * points @ rotation.T + translation


def position_errors(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error = np.linalg.norm(prediction - target, axis=-1)
    return {
        "mean_m": float(error.mean()),
        "rmse_m": float(np.sqrt(np.mean(np.square(error)))),
        "median_m": float(np.median(error)),
        "p95_m": float(np.quantile(error, 0.95)),
        "max_m": float(error.max()),
    }


def path_metrics(points: np.ndarray) -> dict[str, float]:
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return {
        "net_displacement_m": float(np.linalg.norm(points[-1] - points[0])),
        "path_length_m": float(steps.sum()),
        "step_mean_m": float(steps.mean()),
        "step_median_m": float(np.median(steps)),
        "step_max_m": float(steps.max()),
    }


def mesh_metrics(path: Path) -> dict[str, object]:
    mesh = trimesh.load(path, process=True)
    return {
        "path": str(path.resolve()),
        "aabb_extent_xyz_m": mesh.bounding_box.extents.tolist(),
        "obb_extent_sorted_m": np.sort(mesh.bounding_box_oriented.primitive.extents).tolist(),
        "volume_m3": float(mesh.volume),
        "watertight": bool(mesh.is_watertight),
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
    }


def load_cari4d_joints(sequence: Path, cari4d_root: Path) -> tuple[np.ndarray, float]:
    tools_dir = Path(__file__).resolve().parents[2] / "tools"
    sys.path.insert(0, str(tools_dir))
    try:
        from export_cari4d_retargeting_input import _load_smplh_joints

        metadata = json.loads((sequence / "human" / "metadata.json").read_text(encoding="utf-8"))
        return _load_smplh_joints(
            sequence / "human" / "smplh.npz",
            gender=metadata["gender"],
            cari4d_root=cari4d_root,
        )
    finally:
        sys.path.pop(0)


def rotation_metrics(
    generated: np.ndarray,
    reference: np.ndarray,
    world_rotation: np.ndarray,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    generated_relative = np.einsum("ij,tjk->tik", generated[0].T, generated)
    reference_relative = np.einsum("ij,tjk->tik", reference[0].T, reference)
    generated_angle = Rotation.from_matrix(generated_relative).magnitude()
    reference_angle = Rotation.from_matrix(reference_relative).magnitude()
    generated_delta = np.einsum("tji,tjk->tik", generated[:-1], generated[1:])
    reference_delta = np.einsum("tji,tjk->tik", reference[:-1], reference[1:])
    generated_step_angle = Rotation.from_matrix(generated_delta).magnitude()
    reference_step_angle = Rotation.from_matrix(reference_delta).magnitude()

    # R_ref[t] ~= A R_generated[t] B, where A is the camera-to-reference
    # rotation obtained from translation alignment and B absorbs the different
    # canonical object frames.
    candidates = np.einsum(
        "tij,jk,tkl->til", np.swapaxes(generated, 1, 2), world_rotation.T, reference
    )
    canonical_rotation = Rotation.from_matrix(candidates).mean().as_matrix()
    aligned = np.einsum("ij,tjk,kl->til", world_rotation, generated, canonical_rotation)
    absolute_error = Rotation.from_matrix(
        np.einsum("tij,tjk->tik", np.swapaxes(aligned, 1, 2), reference)
    ).magnitude()

    aligned_relative = np.einsum(
        "ij,tjk,kl->til", canonical_rotation.T, generated_relative, canonical_rotation
    )
    relative_error = Rotation.from_matrix(
        np.einsum("tij,tjk->tik", np.swapaxes(aligned_relative, 1, 2), reference_relative)
    ).magnitude()
    report = {
        "relative_angle_from_first_generated_final_deg": float(np.degrees(generated_angle[-1])),
        "relative_angle_from_first_reference_final_deg": float(np.degrees(reference_angle[-1])),
        "relative_angle_from_first_generated_max_deg": float(np.degrees(generated_angle.max())),
        "relative_angle_from_first_reference_max_deg": float(np.degrees(reference_angle.max())),
        "relative_angle_profile_correlation": float(np.corrcoef(generated_angle, reference_angle)[0, 1]),
        "angular_step_generated_mean_deg": float(np.degrees(generated_step_angle.mean())),
        "angular_step_reference_mean_deg": float(np.degrees(reference_step_angle.mean())),
        "best_static_two_sided_alignment_error_deg": {
            "mean": float(np.degrees(absolute_error.mean())),
            "median": float(np.degrees(np.median(absolute_error))),
            "p95": float(np.degrees(np.quantile(absolute_error, 0.95))),
            "max": float(np.degrees(absolute_error.max())),
        },
        "relative_orientation_error_deg": {
            "mean": float(np.degrees(relative_error.mean())),
            "median": float(np.degrees(np.median(relative_error))),
            "p95": float(np.degrees(np.quantile(relative_error, 0.95))),
            "max": float(np.degrees(relative_error.max())),
            "final": float(np.degrees(relative_error[-1])),
        },
        "canonical_rotation_reference_from_generated": canonical_rotation.tolist(),
    }
    return report, generated_angle, reference_angle


def main() -> None:
    args = parse_args()
    sequence = args.sequence.expanduser().resolve()
    reference_path = args.reference_pt.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    cari4d_root = args.cari4d_root.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    generated_object_data = np.load(sequence / "object" / "trajectory.npz", allow_pickle=False)
    generated_object = np.asarray(generated_object_data["translation"], dtype=np.float64)
    generated_rotation = np.asarray(generated_object_data["rotation_matrix"], dtype=np.float64)
    generated_joints, human_height = load_cari4d_joints(sequence, cari4d_root)
    generated_joints = np.asarray(generated_joints, dtype=np.float64)

    packed = torch.load(reference_path, map_location="cpu", weights_only=False).detach().numpy()
    if packed.ndim != 2 or packed.shape[1] < 325:
        raise ValueError(f"Expected InterMimic [T,>=325] tensor, got {packed.shape}")
    reference_joints = np.asarray(packed[:, 162:318].reshape(-1, 52, 3), dtype=np.float64)
    reference_object = np.asarray(packed[:, 318:321], dtype=np.float64)
    reference_rotation = Rotation.from_quat(np.asarray(packed[:, 321:325], dtype=np.float64)).as_matrix()
    frame_count = len(generated_object)
    if not (
        len(reference_object) == len(generated_joints) == len(reference_joints) == frame_count
    ):
        raise ValueError(
            "Frame counts differ; refusing to truncate: "
            f"generated object={frame_count}, generated human={len(generated_joints)}, "
            f"reference object={len(reference_object)}, reference human={len(reference_joints)}"
        )

    rigid_fit = fit_similarity(generated_object, reference_object, estimate_scale=False)
    similarity_fit = fit_similarity(generated_object, reference_object, estimate_scale=True)
    generated_object_aligned = transform_points(generated_object, similarity_fit)
    generated_speed = np.linalg.norm(np.diff(generated_object, axis=0), axis=1)
    reference_speed = np.linalg.norm(np.diff(reference_object, axis=0), axis=1)

    generated_pelvis = generated_joints[:, 0]
    reference_pelvis = reference_joints[:, 0]
    scene_fit = fit_similarity(
        np.concatenate([generated_object, generated_pelvis], axis=0),
        np.concatenate([reference_object, reference_pelvis], axis=0),
        estimate_scale=True,
    )
    generated_object_scene = transform_points(generated_object, scene_fit)
    generated_pelvis_scene = transform_points(generated_pelvis, scene_fit)
    relative_vector_error = (
        scene_fit[0] * (generated_object - generated_pelvis) @ scene_fit[1].T
        - (reference_object - reference_pelvis)
    )
    generated_object_pelvis_distance = np.linalg.norm(generated_object - generated_pelvis, axis=1)
    reference_object_pelvis_distance = np.linalg.norm(reference_object - reference_pelvis, axis=1)

    generated_root_centered = generated_joints - generated_joints[:, :1]
    reference_root_centered = reference_joints - reference_joints[:, :1]
    body_fit = fit_similarity(
        generated_root_centered[:, BODY_JOINT_INDICES].reshape(-1, 3),
        reference_root_centered[:, BODY_JOINT_INDICES].reshape(-1, 3),
        estimate_scale=True,
    )
    generated_body_aligned = transform_points(
        generated_root_centered[:, BODY_JOINT_INDICES].reshape(-1, 3), body_fit
    ).reshape(frame_count, len(BODY_JOINT_INDICES), 3)
    body_error_per_frame = np.linalg.norm(
        generated_body_aligned - reference_root_centered[:, BODY_JOINT_INDICES], axis=-1
    ).mean(axis=1)
    all_joint_fit = fit_similarity(
        generated_root_centered.reshape(-1, 3), reference_root_centered.reshape(-1, 3), estimate_scale=True
    )
    all_joint_error = position_errors(
        transform_points(generated_root_centered.reshape(-1, 3), all_joint_fit),
        reference_root_centered.reshape(-1, 3),
    )

    orientation_report, generated_relative_angle, reference_relative_angle = rotation_metrics(
        generated_rotation, reference_rotation, similarity_fit[1]
    )
    distance_difference = generated_object_pelvis_distance - reference_object_pelvis_distance
    report: dict[str, object] = {
        "schema": "holosoma.cari4d_vs_intermimic_diagnostic.v1",
        "status": "PASS",
        "sequence": str(sequence),
        "reference": str(reference_path),
        "reference_role": "evaluation_only_not_reconstruction_input",
        "frames": frame_count,
        "fps": float(np.asarray(generated_object_data["fps"]).item()),
        "generated_coordinate_frame": "cari4d_metric_camera",
        "reference_coordinate_frame": "legacy_intermimic_world",
        "alignment_note": "fixed alignment estimated only for evaluation; canonical outputs were not modified",
        "object_translation": {
            "generated": path_metrics(generated_object),
            "reference": path_metrics(reference_object),
            "speed_profile_correlation": float(np.corrcoef(generated_speed, reference_speed)[0, 1]),
            "rigid_alignment": {
                "scale": rigid_fit[0],
                "rotation": rigid_fit[1].tolist(),
                "translation": rigid_fit[2].tolist(),
                "error": position_errors(transform_points(generated_object, rigid_fit), reference_object),
            },
            "similarity_alignment": {
                "scale": similarity_fit[0],
                "rotation": similarity_fit[1].tolist(),
                "translation": similarity_fit[2].tolist(),
                "error": position_errors(generated_object_aligned, reference_object),
            },
        },
        "object_orientation": orientation_report,
        "human": {
            "cari4d_smplh_height_m": human_height,
            "generated_pelvis_path": path_metrics(generated_pelvis),
            "reference_pelvis_path": path_metrics(reference_pelvis),
            "root_centered_body22_global_similarity_scale": body_fit[0],
            "root_centered_body22_mpjpe_m": float(body_error_per_frame.mean()),
            "root_centered_body22_frame_mpjpe_p95_m": float(np.quantile(body_error_per_frame, 0.95)),
            "root_centered_all52_error": all_joint_error,
            "pose_note": "CARI4D final result has 72 raw pose values; exported finger poses use the recorded GRAB mean-hand prior",
        },
        "shared_human_object_scene_alignment": {
            "scale": scene_fit[0],
            "rotation": scene_fit[1].tolist(),
            "translation": scene_fit[2].tolist(),
            "object_error": position_errors(generated_object_scene, reference_object),
            "pelvis_error": position_errors(generated_pelvis_scene, reference_pelvis),
            "object_minus_pelvis_vector_rmse_m": float(
                np.sqrt(np.mean(np.sum(np.square(relative_vector_error), axis=1)))
            ),
            "object_pelvis_distance_generated_mean_m": float(generated_object_pelvis_distance.mean()),
            "object_pelvis_distance_reference_mean_m": float(reference_object_pelvis_distance.mean()),
            "object_pelvis_distance_mae_m": float(np.mean(np.abs(distance_difference))),
            "object_pelvis_distance_rmse_m": float(np.sqrt(np.mean(np.square(distance_difference)))),
            "object_pelvis_distance_correlation": float(
                np.corrcoef(generated_object_pelvis_distance, reference_object_pelvis_distance)[0, 1]
            ),
        },
        "meshes": {"generated": mesh_metrics(sequence / "object" / "object_metric.obj")},
    }
    if args.reference_object_mesh is not None:
        reference_mesh = args.reference_object_mesh.expanduser().resolve()
        generated_mesh_report = report["meshes"]["generated"]
        reference_mesh_report = mesh_metrics(reference_mesh)
        report["meshes"]["reference"] = reference_mesh_report
        report["meshes"]["obb_extent_ratio_generated_over_reference"] = (
            np.asarray(generated_mesh_report["obb_extent_sorted_m"])
            / np.asarray(reference_mesh_report["obb_extent_sorted_m"])
        ).tolist()
        report["meshes"]["volume_ratio_generated_over_reference"] = float(
            generated_mesh_report["volume_m3"] / reference_mesh_report["volume_m3"]
        )

    report_path = output / "cari4d_vs_intermimic_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    frames = np.arange(frame_count)
    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    ax = fig.add_subplot(2, 3, 1, projection="3d")
    ax.plot(*reference_object.T, label="reference", linewidth=2)
    ax.plot(*generated_object_aligned.T, label="CARI4D (Sim3 aligned)", linewidth=1.5)
    ax.scatter(*reference_object[[0, -1]].T, s=25)
    ax.set_title("Object translation trajectory")
    ax.legend()

    ax = fig.add_subplot(2, 3, 2)
    ax.plot(frames[1:], generated_speed * 100.0, label="CARI4D")
    ax.plot(frames[1:], reference_speed * 100.0, label="reference")
    ax.set_title("Object frame-to-frame displacement")
    ax.set_xlabel("frame")
    ax.set_ylabel("cm/frame")
    ax.legend()

    ax = fig.add_subplot(2, 3, 3)
    ax.plot(frames, np.degrees(generated_relative_angle), label="CARI4D")
    ax.plot(frames, np.degrees(reference_relative_angle), label="reference")
    ax.set_title("Object rotation relative to frame 0")
    ax.set_xlabel("frame")
    ax.set_ylabel("degrees")
    ax.legend()

    ax = fig.add_subplot(2, 3, 4)
    ax.plot(frames, generated_object_pelvis_distance, label="CARI4D")
    ax.plot(frames, reference_object_pelvis_distance, label="reference")
    ax.set_title("Object-pelvis distance")
    ax.set_xlabel("frame")
    ax.set_ylabel("m")
    ax.legend()

    ax = fig.add_subplot(2, 3, 5, projection="3d")
    ax.plot(*reference_pelvis.T, label="reference", linewidth=2)
    ax.plot(*generated_pelvis_scene.T, label="CARI4D (shared Sim3)", linewidth=1.5)
    ax.set_title("Pelvis trajectory (shared scene alignment)")
    ax.legend()

    ax = fig.add_subplot(2, 3, 6)
    ax.plot(frames, body_error_per_frame * 100.0)
    ax.set_title("Root-centered body22 MPJPE")
    ax.set_xlabel("frame")
    ax.set_ylabel("cm")
    fig.suptitle("CARI4D reconstruction vs InterMimic/OMOMO reference (evaluation only)")
    plot_path = output / "cari4d_vs_intermimic_comparison.png"
    fig.savefig(plot_path, dpi=170)
    plt.close(fig)
    print(json.dumps({"report": str(report_path), "plot": str(plot_path)}, indent=2))


if __name__ == "__main__":
    main()
