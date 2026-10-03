"""Rigid palm contact patches without using placeholder finger poses."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import igl
import mujoco
import numpy as np
from scipy.spatial import cKDTree
import trimesh

from holosoma_retargeting.semantic_keyframes.contact_targets import (
    _mesh_similarity, load_retarget_contact_targets,
)


def robot_palm_patch(mesh: trimesh.Trimesh, side: str) -> tuple[np.ndarray, np.ndarray, int]:
    """Area-average the inward palm face of the G1 rubber hand, not fingers.

    G1 link coordinates: hand extends along +X; inward face is -Y on the
    left hand and +Y on the right. This deliberately excludes distal fingers
    and the thumb. It does not define a finger-heading objective.
    """
    if side not in {"left", "right"}:
        raise ValueError("side must be left or right")
    sign = -1.0 if side == "left" else 1.0
    centers, normals = mesh.triangles_center, mesh.face_normals
    selected = (
        (centers[:, 0] >= 0.025) & (centers[:, 0] <= 0.065)
        & (np.abs(centers[:, 2]) <= 0.020) & (normals[:, 1] * sign > 0.75)
    )
    if not selected.any():
        raise ValueError(f"no {side} G1 palm surface triangles found")
    area = mesh.area_faces[selected]
    center = np.average(centers[selected], axis=0, weights=area)
    normal = np.average(normals[selected], axis=0, weights=area)
    normal /= np.linalg.norm(normal)
    return center, normal, int(selected.sum())


def smoothed_surface_normals(
    mesh: trimesh.Trimesh, points: np.ndarray, face_ids: np.ndarray, radius_m: float = 0.025,
) -> np.ndarray:
    """Area-average nearby consistently oriented faces of a scanned surface."""
    tree = cKDTree(mesh.triangles_center)
    normals = []
    for point, face_id in zip(points, face_ids):
        reference = mesh.face_normals[int(face_id)]
        neighbors = np.asarray(tree.query_ball_point(point, radius_m), dtype=int)
        if len(neighbors):
            neighbors = neighbors[mesh.face_normals[neighbors] @ reference > np.cos(np.deg2rad(45))]
        normal = (
            np.average(mesh.face_normals[neighbors], axis=0, weights=mesh.area_faces[neighbors])
            if len(neighbors) else reference.copy()
        )
        normal /= np.linalg.norm(normal)
        normals.append(normal)
    return np.asarray(normals)


def build_palm_patch_targets(
    *, contacts_path: Path, bundle: Path, body_model: Path,
    target_object_mesh: Path, robot_asset_dir: Path, output: Path,
    baseline_npz: Path | None = None, baseline_scene: Path | None = None,
) -> dict:
    if output.exists():
        raise FileExistsError(output)
    contacts = load_retarget_contact_targets(contacts_path)
    if set(contacts.parts).difference({"left_hand", "right_hand"}):
        raise ValueError("palm patches support semantic hand contacts only")
    with np.load(bundle, allow_pickle=False) as data:
        human = np.asarray(data["human_vertices"], dtype=float)
        source_object = np.asarray(data["object_vertices_local"], dtype=float)
        source_faces = np.asarray(data["object_faces"], dtype=np.int32)
        rotations = np.asarray(data["object_rotation"], dtype=float)
        translations = np.asarray(data["object_translation"], dtype=float)
        scales = np.asarray(data["object_scale"], dtype=float)
    contacts.validate(len(human))
    with np.load(body_model, allow_pickle=False) as model:
        dominant = np.argmax(model["weights"], axis=1)
    mesh = trimesh.load(target_object_mesh, force="mesh", process=False)
    if not np.array_equal(source_faces, mesh.faces):
        raise ValueError("object face correspondence changed")
    scale, rotation, translation, error = _mesh_similarity(source_object, mesh)
    if error > 1e-5:
        raise ValueError("object similarity fit failed")
    anchors = np.empty_like(contacts.object_points_local)
    object_normals = np.empty_like(anchors)
    robot_points, robot_normals, description = [], [], {}
    baseline = None
    if (baseline_npz is None) != (baseline_scene is None):
        raise ValueError("baseline_npz and baseline_scene must be supplied together")
    if baseline_npz is not None:
        with np.load(baseline_npz, allow_pickle=False) as data:
            baseline = np.asarray(data["qpos"], dtype=float)
            baseline_fps = float(data["fps"])
        model = mujoco.MjModel.from_xml_path(str(baseline_scene.resolve()))
        if baseline.shape != (len(human), model.nq) or baseline_fps != contacts.metadata["fps"]:
            raise ValueError("baseline trajectory frames/FPS/model do not match contact targets")
        free_joints = np.flatnonzero(model.jnt_type == int(mujoco.mjtJoint.mjJNT_FREE))
        if len(free_joints) != 2:
            raise ValueError("expected robot and object free joints")
        object_body = int(model.jnt_bodyid[free_joints[-1]])
        mj_data = mujoco.MjData(model)
    for index, part in enumerate(contacts.parts):
        side = part.split("_")[0]
        wrist_id = 20 if side == "left" else 21
        # Palm skin is dominated by the wrist joint. Finger-dominated skin
        # (native joints 22..51) is explicitly excluded, even when hand pose
        # entries exist in the SMPL-H file.
        region = np.flatnonzero(dominant == wrist_id)
        if not len(region):
            raise ValueError(f"no native wrist/palm region for {part}")
        palm_world = human[:, region].mean(axis=1)
        local = np.einsum("tji,tj->ti", rotations, palm_world - translations) / scales[:, None]
        _, face_ids, closest = igl.point_mesh_squared_distance(local, source_object, source_faces)
        asset = robot_asset_dir / f"{side}_rubber_hand.obj"
        robot_mesh = trimesh.load(asset, force="mesh", process=False)
        point, normal, face_count = robot_palm_patch(robot_mesh, side)
        robot_points.append(point)
        robot_normals.append(normal)
        if baseline is None:
            anchors[:, index] = scale * np.asarray(closest) @ rotation + translation
        else:
            body = model.body(f"{side}_rubber_hand_link").id
            local = np.empty((len(human), 3))
            for frame, q in enumerate(baseline):
                mj_data.qpos[:] = q
                mujoco.mj_forward(model, mj_data)
                palm = mj_data.xpos[body] + mj_data.xmat[body].reshape(3, 3) @ point
                local[frame] = mj_data.xmat[object_body].reshape(3, 3).T @ (palm - mj_data.xpos[object_body])
            _, face_ids, closest = igl.point_mesh_squared_distance(local, mesh.vertices, mesh.faces)
            anchors[:, index] = closest
        object_normals[:, index] = smoothed_surface_normals(mesh, anchors[:, index], face_ids)
        description[part] = {
            "source_palm_vertices": len(region), "source_native_joint": wrist_id,
            "robot_point_local_m": point.tolist(), "robot_normal_local": normal.tolist(),
            "robot_palm_triangles": face_count,
            "robot_mesh": str(asset.resolve()), "robot_mesh_sha256": hashlib.sha256(asset.read_bytes()).hexdigest(),
        }
    metadata = {
        **contacts.metadata,
        "constraint_mode": "palm_patch",
        "anchor_policy": (
            "project baseline robot palm patch center onto object surface, preserving each hand's contact side; no finger vertices"
            if baseline is not None else
            "project source wrist-dominated palm center onto object surface; no finger vertices"
        ),
        "orientation_policy": "robot inward palm normal opposes local object outward surface normal; twist about the normal is free",
        "finger_pose_or_heading_used": False,
        "arm_joint_reference_lock": False,
        "object_normal_smoothing_radius_m": 0.025,
        "palm_geometry": description,
        "parent_contact_artifact": str(contacts_path.resolve()),
        "parent_contact_sha256": hashlib.sha256(contacts_path.read_bytes()).hexdigest(),
        "baseline_anchor_reference": {
            "npz": str(baseline_npz.resolve()), "sha256": hashlib.sha256(baseline_npz.read_bytes()).hexdigest(),
            "scene": str(baseline_scene.resolve()), "scene_sha256": hashlib.sha256(baseline_scene.read_bytes()).hexdigest(),
            "orientation_preserved_from_baseline": False,
        } if baseline is not None else None,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output, semantic_parts=np.asarray(contacts.parts), active=contacts.active,
        object_points_local=anchors.astype(np.float32), weights=contacts.weights.astype(np.float32),
        source_distance_m=contacts.source_distance_m.astype(np.float32),
        robot_points_local=np.asarray(robot_points, dtype=np.float32),
        robot_normals_local=np.asarray(robot_normals, dtype=np.float32),
        object_normals_local=object_normals.astype(np.float32),
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    load_retarget_contact_targets(output, expected_frames=len(human))
    return metadata
