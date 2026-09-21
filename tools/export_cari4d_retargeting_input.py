#!/usr/bin/env python3
"""Convert a canonical CARI4D sequence into an explicit retargeting bundle.

This adapter is category- and dataset-agnostic. It never reads an OMOMO or
other dataset trajectory. Because monocular CARI4D output is camera-relative,
the caller must explicitly provide the rigid world-from-camera transform; the
adapter does not silently guess gravity or camera leveling.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

from hoi_pipeline.common import matrix_to_quaternion_wxyz, read_obj_vertices
from omomo_cari4d_renderer.smplh_joint_order import (
    SMPLH_NATIVE_TO_RETARGET,
    SMPLH_RETARGET_JOINT_NAMES,
)


def parse_args() -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", required=True, type=Path, help="Canonical CARI4D sequence root.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--task-name", required=True, help="Output bundle stem; no dataset semantics are assumed.")
    parser.add_argument(
        "--robot-height",
        type=float,
        default=1.32,
        help="Robot reference height in meters, used only to record the recommended retarget scale.",
    )
    parser.add_argument(
        "--world-from-cari4d",
        required=True,
        type=Path,
        help="Explicit rigid 4x4 .npy transform from CARI4D metric camera coordinates to Z-up retarget world.",
    )
    parser.add_argument(
        "--ground-alignment",
        choices=("none", "object-first-frame", "feet-global-minimum"),
        default="none",
        help="Optional explicit shared Z translation applied to both human and object.",
    )
    parser.add_argument(
        "--xy-origin",
        choices=("unchanged", "initial-pelvis"),
        default="unchanged",
        help="Optional shared planar recentering; it does not alter relative human/object geometry.",
    )
    parser.add_argument(
        "--cari4d-root",
        type=Path,
        default=repository / "third_party" / "CARI4D",
    )
    parser.add_argument(
        "--robot-scene-template",
        type=Path,
        default=(
            repository / "src" / "holosoma_retargeting" / "holosoma_retargeting" /
            "demo_data" / "models" / "g1" / "g1_29dof_w_largebox.xml"
        ),
        help="MuJoCo robot+object scene whose object mesh reference will be replaced.",
    )
    return parser.parse_args()


def _load_rigid_transform(path: Path) -> np.ndarray:
    transform = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"--world-from-cari4d must have shape [4,4], got {transform.shape}")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError("--world-from-cari4d has an invalid homogeneous bottom row")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-6
    ):
        raise ValueError("--world-from-cari4d rotation must be right-handed and orthonormal")
    return transform


def _load_smplh_joints(
    human_file: Path,
    *,
    gender: str,
    cari4d_root: Path,
) -> tuple[np.ndarray, float]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("The CARI4D Python environment with PyTorch is required") from exc
    root_text = str(cari4d_root)
    inserted = root_text not in sys.path
    if inserted:
        sys.path.insert(0, root_text)
    try:
        from lib_smpl import get_smpl  # type: ignore[import-not-found]

        data = np.load(human_file, allow_pickle=False)
        poses = np.asarray(data["poses"], dtype=np.float32)
        trans = np.asarray(data["trans"], dtype=np.float32)
        betas = np.asarray(data["betas"], dtype=np.float32)
        if betas.ndim == 1:
            betas = np.broadcast_to(betas, (len(poses), len(betas))).copy()
        if poses.shape != (len(trans), 156) or trans.shape[1:] != (3,) or betas.shape != (len(poses), 10):
            raise ValueError(
                f"Invalid canonical SMPL-H arrays: poses={poses.shape}, trans={trans.shape}, betas={betas.shape}"
            )
        model = get_smpl(gender, True, model_root=str(cari4d_root / "data" / "smpl"))
        joints_chunks: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(poses), 128):
                end = min(len(poses), start + 128)
                _, joints, _, _ = model(
                    torch.from_numpy(poses[start:end]),
                    torch.from_numpy(betas[start:end]),
                    torch.from_numpy(trans[start:end]),
                )
                joints_chunks.append(joints.cpu().numpy())
            zero_pose = torch.zeros((1, 156), dtype=torch.float32)
            zero_trans = torch.zeros((1, 3), dtype=torch.float32)
            mean_beta = torch.from_numpy(betas.mean(axis=0, keepdims=True))
            vertices_tpose, _, _, _ = model(zero_pose, mean_beta, zero_trans)
        native_joints = np.concatenate(joints_chunks, axis=0)
        if native_joints.shape != (len(poses), 52, 3):
            raise ValueError(f"CARI4D SMPL-H model returned unexpected joints {native_joints.shape}")
        # A zero-pose SMPL-H model uses its native +Y vertical convention.
        human_height = float(np.ptp(vertices_tpose.cpu().numpy()[0, :, 1]))
        if not np.isfinite(human_height) or not 1.0 <= human_height <= 2.5:
            raise ValueError(f"Implausible SMPL-H T-pose height: {human_height} m")
        return native_joints[:, SMPLH_NATIVE_TO_RETARGET], human_height
    finally:
        if inserted:
            sys.path.remove(root_text)


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def _write_object_urdf(path: Path, mesh: Path) -> None:
    content = f'''<?xml version="1.0"?>
<robot name="largebox">
  <link name="largebox_link">
    <inertial>
      <mass value="0.1"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.002" ixy="0" ixz="0" iyy="0.002" iyz="0" izz="0.002"/>
    </inertial>
    <visual><geometry><mesh filename="{mesh.name}" scale="1 1 1"/></geometry></visual>
    <collision name="largebox"><geometry><mesh filename="{mesh.name}" scale="1 1 1"/></geometry></collision>
  </link>
</robot>
'''
    path.write_text(content, encoding="utf-8")


def _write_scene(template: Path, destination: Path, object_mesh: Path) -> None:
    if not template.is_file():
        raise FileNotFoundError(template)
    robot_assets = template.parent / "assets"
    text = template.read_text(encoding="utf-8")
    original_compiler = '<compiler angle="radian" meshdir="assets/"/>'
    if original_compiler not in text:
        raise ValueError(f"Scene template compiler layout changed: {template}")
    text = text.replace(
        original_compiler,
        f'<compiler angle="radian" meshdir="{robot_assets.resolve()}"/>',
        1,
    )
    original_mesh = 'file="../../largebox/largebox.obj"'
    if original_mesh not in text:
        raise ValueError(f"Scene template object mesh layout changed: {template}")
    text = text.replace(original_mesh, f'file="{object_mesh.resolve()}"', 1)
    destination.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not np.isfinite(args.robot_height) or args.robot_height <= 0.0:
        raise ValueError("--robot-height must be a positive finite value in meters")
    sequence = args.sequence.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    cari4d_root = args.cari4d_root.expanduser().resolve()
    transform_path = args.world_from_cari4d.expanduser().resolve()
    for path in (sequence / "human" / "smplh.npz", sequence / "human" / "metadata.json",
                 sequence / "object" / "trajectory.npz", sequence / "object" / "object_metric.obj",
                 sequence / "meta" / "sequence.json", transform_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    human_metadata = json.loads((sequence / "human" / "metadata.json").read_text(encoding="utf-8"))
    sequence_metadata = json.loads((sequence / "meta" / "sequence.json").read_text(encoding="utf-8"))
    gender = human_metadata.get("gender")
    if gender not in {"male", "female"}:
        raise ValueError("Canonical human metadata must explicitly contain gender=male/female")

    human_data = np.load(sequence / "human" / "smplh.npz", allow_pickle=False)
    object_data = np.load(sequence / "object" / "trajectory.npz", allow_pickle=False)
    human_frame_ids = np.asarray(human_data["frame_ids"], dtype=np.int64)
    object_frame_ids = np.asarray(object_data["frame_ids"], dtype=np.int64)
    if not np.array_equal(human_frame_ids, object_frame_ids):
        raise ValueError("Human/object frame IDs differ; refusing to truncate")
    fps = float(np.asarray(human_data["fps"]).item())
    if not np.isclose(fps, float(np.asarray(object_data["fps"]).item())):
        raise ValueError("Human/object FPS differ")

    human_camera, human_height = _load_smplh_joints(
        sequence / "human" / "smplh.npz", gender=gender, cari4d_root=cari4d_root
    )
    rotation_camera = np.asarray(object_data["rotation_matrix"], dtype=np.float64)
    translation_camera = np.asarray(object_data["translation"], dtype=np.float64)
    transform = _load_rigid_transform(transform_path)
    human_world = _transform_points(human_camera, transform)
    rotation_world = np.einsum("ij,tjk->tik", transform[:3, :3], rotation_camera)
    translation_world = _transform_points(translation_camera, transform)

    post = np.eye(4, dtype=np.float64)
    if args.xy_origin == "initial-pelvis":
        post[0, 3] = -human_world[0, 0, 0]
        post[1, 3] = -human_world[0, 0, 1]
    if args.ground_alignment == "object-first-frame":
        vertices = read_obj_vertices(sequence / "object" / "object_metric.obj")
        vertices_first = vertices @ rotation_world[0].T + translation_world[0]
        post[2, 3] = -float(vertices_first[:, 2].min())
    elif args.ground_alignment == "feet-global-minimum":
        post[2, 3] = -float(human_world[:, [4, 8], 2].min())
    effective_transform = post @ transform
    human_world = _transform_points(human_camera, effective_transform)
    rotation_world = np.einsum("ij,tjk->tik", effective_transform[:3, :3], rotation_camera)
    translation_world = _transform_points(translation_camera, effective_transform)

    orthogonality = np.linalg.norm(
        np.swapaxes(rotation_world, 1, 2) @ rotation_world - np.eye(3)[None], axis=(1, 2)
    )
    determinants = np.linalg.det(rotation_world)
    if float(orthogonality.max()) > 1e-4 or not np.allclose(determinants, 1.0, atol=1e-4):
        raise ValueError("World object rotations are invalid")
    quaternions = matrix_to_quaternion_wxyz(rotation_world.astype(np.float32))
    object_poses = np.concatenate([quaternions, translation_world.astype(np.float32)], axis=1)
    if not np.isfinite(human_world).all() or not np.isfinite(object_poses).all():
        raise ValueError("Retargeting bundle contains NaN/Inf")

    output.mkdir(parents=True, exist_ok=True)
    bundle = output / f"{args.task_name}.npz"
    object_mesh = output / f"{args.task_name}_object.obj"
    object_urdf = output / f"{args.task_name}_object.urdf"
    scene = output / f"{args.task_name}_g1_29dof_w_largebox.xml"
    shutil.copy2(sequence / "object" / "object_metric.obj", object_mesh)
    _write_object_urdf(object_urdf, object_mesh)
    _write_scene(args.robot_scene_template.expanduser().resolve(), scene, object_mesh)
    np.savez_compressed(
        bundle,
        human_joints=human_world.astype(np.float32),
        object_poses_wxyz_xyz=object_poses.astype(np.float32),
        human_height_m=np.asarray(human_height, dtype=np.float32),
        fps=np.asarray(fps, dtype=np.float32),
        frame_ids=human_frame_ids,
        T_world_from_cari4d=effective_transform.astype(np.float64),
        smplh_joint_names=np.asarray(SMPLH_RETARGET_JOINT_NAMES),
    )
    report = {
        "schema": "holosoma.cari4d_retargeting_input.v1",
        "status": "PASS",
        "task_name": args.task_name,
        "source_canonical_sequence": str(sequence),
        "source_video": sequence_metadata.get("source_video"),
        "source_trajectory": str((sequence / "object" / "trajectory.npz").resolve()),
        "uses_motion_or_object_pose_labels": sequence_metadata.get("uses_motion_or_object_pose_labels"),
        "frames": len(human_frame_ids),
        "fps": fps,
        "gender": gender,
        "human_height_m": human_height,
        "robot_height_m": args.robot_height,
        "recommended_smpl_scale": args.robot_height / human_height,
        "input_coordinate_frame": human_metadata["coordinate_frame"],
        "output_coordinate_frame": "explicit_z_up_retarget_world",
        "T_world_from_cari4d_input_file": str(transform_path),
        "T_world_from_cari4d_effective": effective_transform.tolist(),
        "xy_origin": args.xy_origin,
        "ground_alignment": args.ground_alignment,
        "quaternion_convention": "wxyz_scalar_first",
        "object_pose_layout": "[qw,qx,qy,qz,x,y,z]",
        "joint_order": list(SMPLH_RETARGET_JOINT_NAMES),
        "joint_generation": "CARI4D lib_smpl SMPL-H forward pass, then explicit native-to-retarget permutation",
        "bundle": str(bundle),
        "object_mesh": str(object_mesh),
        "object_urdf": str(object_urdf),
        "mujoco_scene": str(scene),
        "object_translation_range_m": {
            "min": translation_world.min(axis=0).tolist(),
            "max": translation_world.max(axis=0).tolist(),
        },
        "rotation_orthogonality_error_max": float(orthogonality.max()),
        "rotation_determinant_min_max": [float(determinants.min()), float(determinants.max())],
    }
    metadata_path = output / f"{args.task_name}.metadata.json"
    metadata_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
