"""Validation for the canonical CARI4D sequence representation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from hoi_pipeline.common import count_obj_faces, obj_extent, read_obj_vertices, write_json


def _finite(name: str, value: np.ndarray, errors: list[str]) -> None:
    if not np.all(np.isfinite(value)):
        errors.append(f"{name} contains NaN or Inf")


def validate_canonical_sequence(sequence_dir: Path, *, video_frame_count: int | None = None) -> dict[str, Any]:
    human_path = sequence_dir / "human" / "smplh.npz"
    trajectory_path = sequence_dir / "object" / "trajectory.npz"
    pose_path = sequence_dir / "object" / "initial_pose.npy"
    mesh_path = sequence_dir / "object" / "object_metric.obj"
    errors: list[str] = []

    for path in (human_path, trajectory_path, pose_path, mesh_path):
        if not path.is_file():
            errors.append(f"Missing required output: {path}")
    if errors:
        return {"status": "FAIL", "errors": errors}

    with np.load(human_path) as human, np.load(trajectory_path) as obj:
        required_human = {"poses", "trans", "betas", "fps", "frame_ids"}
        required_obj = {
            "translation", "rotation_matrix", "quaternion_wxyz", "transform", "fps", "frame_ids"
        }
        missing_human = sorted(required_human - set(human.files))
        missing_obj = sorted(required_obj - set(obj.files))
        if missing_human:
            errors.append(f"human/smplh.npz missing keys: {missing_human}")
        if missing_obj:
            errors.append(f"object/trajectory.npz missing keys: {missing_obj}")
        if errors:
            return {"status": "FAIL", "errors": errors}

        poses = human["poses"]
        human_trans = human["trans"]
        betas = human["betas"]
        human_ids = human["frame_ids"]
        obj_trans = obj["translation"]
        rotations = obj["rotation_matrix"]
        quaternions = obj["quaternion_wxyz"]
        transforms = obj["transform"]
        obj_ids = obj["frame_ids"]
        human_fps_array = np.asarray(human["fps"])
        obj_fps_array = np.asarray(obj["fps"])
        human_fps = float(human_fps_array.reshape(())) if human_fps_array.size == 1 else float("nan")
        obj_fps = float(obj_fps_array.reshape(())) if obj_fps_array.size == 1 else float("nan")
        if human_fps_array.size != 1 or obj_fps_array.size != 1:
            errors.append(
                f"fps must be scalar, got human={human_fps_array.shape}, object={obj_fps_array.shape}"
            )

    human_frames = len(poses)
    object_frames = len(obj_trans)
    if human_frames == 0 or object_frames == 0:
        errors.append(f"Trajectories must be non-empty, got human={human_frames}, object={object_frames}")
    if poses.shape != (human_frames, 156):
        errors.append(f"poses must have shape [T,156], got {poses.shape}")
    if human_trans.shape != (human_frames, 3):
        errors.append(f"human trans must have shape [T,3], got {human_trans.shape}")
    if betas.shape not in ((human_frames, 10), (10,)):
        errors.append(f"betas must have shape [T,10] or [10], got {betas.shape}")
    if obj_trans.shape != (object_frames, 3):
        errors.append(f"translation must have shape [T,3], got {obj_trans.shape}")
    if rotations.shape != (object_frames, 3, 3):
        errors.append(f"rotation_matrix must have shape [T,3,3], got {rotations.shape}")
    if quaternions.shape != (object_frames, 4):
        errors.append(f"quaternion_wxyz must have shape [T,4], got {quaternions.shape}")
    if transforms.shape != (object_frames, 4, 4):
        errors.append(f"transform must have shape [T,4,4], got {transforms.shape}")
    if human_frames != object_frames:
        errors.append(f"Frame count mismatch: human={human_frames}, object={object_frames}")
    if human_ids.shape != (human_frames,) or obj_ids.shape != (object_frames,):
        errors.append(f"frame_ids shapes must be [T], got human={human_ids.shape}, object={obj_ids.shape}")
    frame_id_shapes_are_valid = human_ids.shape == (human_frames,) and obj_ids.shape == (object_frames,)
    if frame_id_shapes_are_valid:
        if not np.array_equal(human_ids, obj_ids):
            errors.append("Human and object frame_ids differ; no truncation was performed")
        if len(human_ids) > 1 and (
            np.any(np.diff(human_ids) <= 0) or len(np.unique(human_ids)) != len(human_ids)
        ):
            errors.append("frame_ids must be unique and strictly increasing")
    if abs(human_fps - obj_fps) > 1e-9:
        errors.append(f"Human/object fps mismatch: {human_fps} vs {obj_fps}")
    if not np.isfinite(human_fps) or human_fps <= 0.0 or not np.isfinite(obj_fps) or obj_fps <= 0.0:
        errors.append(f"fps must be finite and positive, got human={human_fps}, object={obj_fps}")
    shapes_are_safe = (
        poses.shape == (human_frames, 156)
        and human_trans.shape == (human_frames, 3)
        and betas.shape in ((human_frames, 10), (10,))
        and obj_trans.shape == (object_frames, 3)
        and rotations.shape == (object_frames, 3, 3)
        and quaternions.shape == (object_frames, 4)
        and transforms.shape == (object_frames, 4, 4)
        and human_ids.shape == (human_frames,)
        and obj_ids.shape == (object_frames,)
        and human_frames > 0
        and object_frames > 0
    )
    if not shapes_are_safe:
        return {
            "status": "FAIL",
            "errors": errors,
            "human_frames": human_frames,
            "object_frames": object_frames,
            "video_frames": video_frame_count,
        }
    if video_frame_count is not None and int(human_ids[-1]) >= video_frame_count:
        errors.append(
            f"Last exported frame ID {int(human_ids[-1])} exceeds video frame count {video_frame_count}"
        )
    video_frame_ids_complete = None
    missing_video_frame_count = None
    if video_frame_count is not None:
        expected_video_ids = np.arange(video_frame_count, dtype=human_ids.dtype)
        video_frame_ids_complete = bool(np.array_equal(human_ids, expected_video_ids))
        missing_video_frame_count = int(video_frame_count - len(human_ids))

    for name, array in (
        ("human poses", poses), ("human trans", human_trans), ("human betas", betas),
        ("object translation", obj_trans),
        ("rotation_matrix", rotations), ("quaternion_wxyz", quaternions), ("transform", transforms),
    ):
        _finite(name, array, errors)

    identity = np.eye(3)[None]
    orthogonality_error = np.linalg.norm(np.swapaxes(rotations, 1, 2) @ rotations - identity, axis=(1, 2))
    determinants = np.linalg.det(rotations)
    quaternion_norm_error = np.abs(np.linalg.norm(quaternions, axis=1) - 1.0)
    if len(rotations) and float(orthogonality_error.max()) > 1e-4:
        errors.append(f"Rotation orthogonality error is {orthogonality_error.max():.3g} (>1e-4)")
    if len(rotations) and float(np.max(np.abs(determinants - 1.0))) > 1e-4:
        errors.append(f"Rotation determinant error is {np.max(np.abs(determinants - 1.0)):.3g} (>1e-4)")
    if len(quaternions) and float(quaternion_norm_error.max()) > 1e-4:
        errors.append(f"Quaternion norm error is {quaternion_norm_error.max():.3g} (>1e-4)")
    if not np.allclose(transforms[:, :3, :3], rotations, atol=1e-6):
        errors.append("transform rotations do not match rotation_matrix")
    if not np.allclose(transforms[:, :3, 3], obj_trans, atol=1e-6):
        errors.append("transform translations do not match translation")
    expected_bottom = np.broadcast_to(np.array([0.0, 0.0, 0.0, 1.0]), transforms[:, 3].shape)
    if not np.allclose(transforms[:, 3], expected_bottom, atol=1e-7):
        errors.append("transform bottom rows are not [0,0,0,1]")
    initial_pose = np.load(pose_path)
    if initial_pose.shape != (4, 4) or not np.allclose(initial_pose, transforms[0], atol=1e-7):
        errors.append("initial_pose.npy is not exactly transform[0]")

    lower, upper, extent = obj_extent(mesh_path)
    if not np.all(np.isfinite(extent)) or np.any(extent <= 1e-7):
        errors.append(f"Metric mesh has invalid extent {extent}")
    elif float(extent.max()) > 10.0:
        errors.append(f"Metric mesh extent {extent} exceeds 10 m; likely a unit/scale error")
    translation_norm = np.linalg.norm(obj_trans, axis=1)
    human_translation_norm = np.linalg.norm(human_trans, axis=1)
    if len(translation_norm) and float(translation_norm.max()) > 100.0:
        errors.append(
            f"Object translation magnitude reaches {translation_norm.max():.3f} m; likely a unit/frame error"
        )
    if len(human_translation_norm) and float(human_translation_norm.max()) > 100.0:
        errors.append(
            f"Human translation magnitude reaches {human_translation_norm.max():.3f} m; likely a unit/frame error"
        )
    report: dict[str, Any] = {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "human_frames": human_frames,
        "object_frames": object_frames,
        "video_frames": video_frame_count,
        "frame_ids_cover_entire_video": video_frame_ids_complete,
        "unexported_video_frame_count": missing_video_frame_count,
        "frame_ids_equal": bool(np.array_equal(human_ids, obj_ids)),
        "first_frame_id": int(human_ids[0]) if len(human_ids) else None,
        "last_frame_id": int(human_ids[-1]) if len(human_ids) else None,
        "fps": human_fps,
        "rotation_orthogonality_error_max": float(orthogonality_error.max()) if len(rotations) else None,
        "rotation_determinant_min": float(determinants.min()) if len(rotations) else None,
        "rotation_determinant_max": float(determinants.max()) if len(rotations) else None,
        "quaternion_norm_error_max": float(quaternion_norm_error.max()) if len(quaternions) else None,
        "translation_norm_min_m": float(translation_norm.min()) if len(translation_norm) else None,
        "translation_norm_max_m": float(translation_norm.max()) if len(translation_norm) else None,
        "human_translation_norm_min_m": (
            float(human_translation_norm.min()) if len(human_translation_norm) else None
        ),
        "human_translation_norm_max_m": (
            float(human_translation_norm.max()) if len(human_translation_norm) else None
        ),
        "mesh_bbox_min_m": lower,
        "mesh_bbox_max_m": upper,
        "mesh_extent_xyz_m": extent,
        "mesh_num_vertices": int(len(read_obj_vertices(mesh_path))),
        "mesh_num_faces": count_obj_faces(mesh_path),
    }
    return report


def validate_and_write(sequence_dir: Path, *, video_frame_count: int | None = None) -> dict[str, Any]:
    report = validate_canonical_sequence(sequence_dir, video_frame_count=video_frame_count)
    write_json(sequence_dir / "validation" / "export_report.json", report)
    return report
