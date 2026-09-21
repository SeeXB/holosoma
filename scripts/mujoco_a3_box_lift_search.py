#!/usr/bin/env python3
"""Physics replay and box-size search for A3 large-box retargets.

The retargeted trajectory is used only as a reference for a floating-base servo
and effort-limited joint PD controller. The box is a free rigid body: its pose is
written once at reset and receives no generalized-force or pose writes later.
Gravity, collision and friction remain enabled. An optional bilateral-contact
triggered soft weld approximates finger closure because the official A3 model
has fixed, open palms; the weld can be released before physical placement.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE_XML = (
    ROOT
    / "src/holosoma_retargeting/holosoma_retargeting/demo_data/models/a3/a3_31dof.xml"
)
SOURCE_BOX_MESH = (
    ROOT
    / "src/holosoma_retargeting/holosoma_retargeting/demo_data/models/largebox/largebox.obj"
)
SOURCE_BOX_EXTENTS = np.asarray([0.47115421, 0.45873013, 0.40789548], dtype=np.float64)
MOTIONS = {
    "original": ROOT
    / "exp/retargeting/a3_sub10_largebox_089_20260921/original/sub10_largebox_089_original.npz",
    "semantic_b4": ROOT
    / "exp/retargeting/a3_sub10_largebox_089_20260921/semantic_b4/sub10_largebox_089_semantic_b4.npz",
}
DEFAULT_OUTPUT = ROOT / "exp/physics/a3_sub10_largebox_089_mujoco"

A3_JOINT_NAMES = [
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "head_yaw_joint",
    "head_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
]


def _fmt(values: tuple[float, ...] | list[float]) -> str:
    return " ".join(f"{value:.9g}" for value in values)


def build_model(
    dimensions: tuple[float, float, float],
    mass: float,
    timestep: float,
    box_geometry: str,
    base_control: str,
    ground_height: float,
    box_friction: float,
    palm_adhesion: float,
    contact_latch: bool,
    contact_latch_mode: str = "wrist",
) -> mujoco.MjModel:
    """Build A3 plus a dynamic box, retaining the official A3 inertias/collisions."""
    tree = ET.parse(BASE_XML)
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is None:
        raise RuntimeError(f"No <compiler> in {BASE_XML}")
    compiler.set("meshdir", str(BASE_XML.parent / "meshes"))

    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("timestep", str(timestep))
    option.set("gravity", "0 0 -9.81")
    option.set("integrator", "implicitfast")
    option.set("solver", "Newton")
    option.set("iterations", "80")
    option.set("noslip_iterations", "8")

    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    global_visual = visual.find("global")
    if global_visual is None:
        global_visual = ET.SubElement(visual, "global")
    global_visual.set("offwidth", "960")
    global_visual.set("offheight", "540")

    # The upstream keyframe has 38 qpos entries.  Appending a free box adds seven,
    # so retaining it would make model compilation fail.
    keyframe = root.find("keyframe")
    if keyframe is not None:
        root.remove(keyframe)

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError(f"No <worldbody> in {BASE_XML}")
    floor_geom = worldbody.find("./geom[@name='floor']")
    if floor_geom is None:
        raise RuntimeError(f"No floor geom in {BASE_XML}")
    floor_geom.set("pos", f"0 0 {ground_height:.9g}")
    if palm_adhesion > 0.0:
        for body_name in ("left_wrist_yaw_Link", "right_wrist_yaw_Link"):
            hand_body = worldbody.find(f".//body[@name='{body_name}']")
            if hand_body is None:
                raise RuntimeError(f"Missing A3 hand body: {body_name}")
            collision_geoms = hand_body.findall("./geom[@class='collision']")
            if not collision_geoms:
                raise RuntimeError(f"Missing collision geoms on A3 hand body: {body_name}")
            for geom in collision_geoms:
                geom.set("adhesion", f"{palm_adhesion:.9g}")
    if base_control == "mocap_weld":
        ET.SubElement(worldbody, "body", name="root_servo_target", mocap="true")
        equality = root.find("equality")
        if equality is None:
            equality = ET.SubElement(root, "equality")
        ET.SubElement(
            equality,
            "weld",
            name="root_servo_weld",
            body1="root_servo_target",
            body2="pelvis_link",
            relpose="0 0 0 1 0 0 0",
            solref="0.002 1",
            solimp="0.99 0.999 0.0001",
        )
    elif base_control != "wrench":
        raise ValueError(f"Unsupported base control: {base_control}")
    dx, dy, dz = dimensions
    inertia = (
        mass * (dy * dy + dz * dz) / 12.0,
        mass * (dx * dx + dz * dz) / 12.0,
        mass * (dx * dx + dy * dy) / 12.0,
    )
    body = ET.SubElement(worldbody, "body", name="physics_box")
    ET.SubElement(body, "freejoint", name="physics_box_freejoint")
    ET.SubElement(body, "inertial", pos="0 0 0", mass=f"{mass:.9g}", diaginertia=_fmt(inertia))
    geom_kwargs = dict(
        name="physics_box_geom",
        contype="1",
        conaffinity="7",
        friction=f"{box_friction:.9g} 0.01 0.001",
        solref="0.004 1",
        solimp="0.95 0.99 0.001",
        rgba="0.22 0.55 0.92 0.82",
        group="0",
    )
    if box_geometry == "mesh":
        asset = root.find("asset")
        if asset is None:
            asset = ET.SubElement(root, "asset")
        scale = np.asarray(dimensions) / SOURCE_BOX_EXTENTS
        ET.SubElement(
            asset,
            "mesh",
            name="physics_box_mesh",
            file=str(SOURCE_BOX_MESH),
            scale=_fmt(scale.tolist()),
        )
        geom_kwargs.update(type="mesh", mesh="physics_box_mesh")
    elif box_geometry == "cuboid":
        geom_kwargs.update(type="box", size=_fmt((dx / 2.0, dy / 2.0, dz / 2.0)))
    else:
        raise ValueError(f"Unsupported box geometry: {box_geometry}")
    ET.SubElement(body, "geom", **geom_kwargs)
    if contact_latch:
        if contact_latch_mode == "object_reference":
            latch_body = "object_reference_grasp_target"
            ET.SubElement(worldbody, "body", name=latch_body, mocap="true")
        elif contact_latch_mode == "wrist":
            latch_body = "right_wrist_yaw_Link"
        else:
            raise ValueError(f"Unsupported contact latch mode: {contact_latch_mode}")
        equality = root.find("equality")
        if equality is None:
            equality = ET.SubElement(root, "equality")
        ET.SubElement(
            equality,
            "weld",
            name="contact_grasp_latch",
            body1=latch_body,
            body2="physics_box",
            active="false",
            solref="0.01 1",
            solimp="0.95 0.99 0.001",
        )
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY
    if model.nq != 45 or model.nv != 43 or model.nu != 31:
        raise RuntimeError(f"Unexpected A3+box sizes: nq={model.nq}, nv={model.nv}, nu={model.nu}")

    ordered_joints = sorted(
        (int(model.jnt_qposadr[jid]), jid)
        for jid in range(model.njnt)
        if 7 <= int(model.jnt_qposadr[jid]) < 38
    )
    model_joint_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) for _, jid in ordered_joints
    ]
    if model_joint_names != A3_JOINT_NAMES:
        raise RuntimeError(f"Trajectory/model joint-order mismatch:\n{model_joint_names}\n!=\n{A3_JOINT_NAMES}")
    return model


def _nlerp_quat(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    if float(np.dot(q0, q1)) < 0.0:
        q1 = -q1
    q = (1.0 - alpha) * q0 + alpha * q1
    return q / np.linalg.norm(q)


def _reference_at(qpos: np.ndarray, frame_float: float) -> tuple[np.ndarray, int]:
    lo = min(int(math.floor(frame_float)), len(qpos) - 1)
    hi = min(lo + 1, len(qpos) - 1)
    alpha = min(max(frame_float - lo, 0.0), 1.0)
    target = (1.0 - alpha) * qpos[lo] + alpha * qpos[hi]
    target[3:7] = _nlerp_quat(qpos[lo, 3:7], qpos[hi, 3:7], alpha)
    target[-4:] = _nlerp_quat(qpos[lo, -4:], qpos[hi, -4:], alpha)
    return target, lo


def _longest_true_run(mask: np.ndarray, dt: float) -> float:
    longest = current = 0
    for value in mask:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest * dt


def _box_bottom(position: np.ndarray, quat: np.ndarray, half_size: np.ndarray) -> float:
    rotation = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(rotation, quat)
    rotation = rotation.reshape(3, 3)
    vertical_support = float(np.abs(rotation[2]) @ half_size)
    return float(position[2] - vertical_support)


def _joint_gains(name: str) -> tuple[float, float]:
    if "hip" in name or "knee" in name:
        return 500.0, 24.0
    if "ankle" in name:
        return 300.0, 16.0
    if "waist" in name:
        return 420.0, 20.0
    if "shoulder" in name or "elbow" in name:
        return 300.0, 14.0
    if "wrist" in name:
        return 120.0, 7.0
    return 70.0, 5.0


def _contact_flags(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    box_geom: int,
    foot_bodies: set[int],
) -> tuple[bool, bool, bool, bool]:
    left = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_yaw_Link")
    right = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_Link")
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    flags = [False, False, False, False]
    for contact in data.contact:
        g1, g2 = int(contact.geom1), int(contact.geom2)
        if box_geom not in (g1, g2):
            continue
        other = g2 if g1 == box_geom else g1
        body = int(model.geom_bodyid[other])
        flags[0] |= body == left
        flags[1] |= body == right
        flags[2] |= other == floor
        flags[3] |= body in foot_bodies
    return tuple(flags)


def _hand_contact_locations_local(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    box_geom: int,
    box_body: int,
    hand_bodies: list[int],
) -> tuple[list[list[float]], list[float]]:
    """Return mean palm contact locations and their midpoint in box coordinates."""
    world_points: list[list[np.ndarray]] = [[], []]
    for contact in data.contact:
        g1, g2 = int(contact.geom1), int(contact.geom2)
        if box_geom not in (g1, g2):
            continue
        other = g2 if g1 == box_geom else g1
        body = int(model.geom_bodyid[other])
        if body in hand_bodies:
            world_points[hand_bodies.index(body)].append(np.asarray(contact.pos).copy())
    if not all(world_points):
        raise RuntimeError("Bilateral contact was reported without bilateral contact locations")
    means_world = [np.mean(points, axis=0) for points in world_points]
    box_rotation_flat = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(box_rotation_flat, data.xquat[box_body])
    box_rotation = box_rotation_flat.reshape(3, 3)
    local = [box_rotation.T @ (point - data.xpos[box_body]) for point in means_world]
    midpoint = 0.5 * (local[0] + local[1])
    return [point.tolist() for point in local], midpoint.tolist()


def _palm_collision_geoms(model: mujoco.MjModel) -> list[int]:
    """Locate the left/right hand-mesh collision geoms, excluding wrist origins."""
    palm_geoms: list[int] = []
    for side in ("left", "right"):
        expected_mesh = f"{side}_hand_Link"
        matches: list[int] = []
        for geom_id in range(model.ngeom):
            mesh_id = int(model.geom_dataid[geom_id])
            mesh_name = (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id)
                if mesh_id >= 0
                else None
            )
            if int(model.geom_group[geom_id]) == 3 and mesh_name == expected_mesh:
                matches.append(geom_id)
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one collision geom for {expected_mesh}, found {matches}"
            )
        palm_geoms.append(matches[0])
    return palm_geoms


def _align_box_reference_to_lowest_hand_line(
    model: mujoco.MjModel,
    qref: np.ndarray,
) -> dict[str, object]:
    """Align box horizontal pose from the two actual palm meshes at their low point."""
    data = mujoco.MjData(model)
    palm_geoms = _palm_collision_geoms(model)
    box_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "physics_box_geom")
    box_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "physics_box")

    lifted = np.flatnonzero(qref[:, -5] > qref[0, -5] + 0.05)
    search_end = int(lifted[0]) if len(lifted) else min(30, len(qref) - 1)
    palm_positions: list[np.ndarray] = []
    for frame in range(search_end + 1):
        data.qpos[:] = qref[frame]
        mujoco.mj_forward(model, data)
        palm_positions.append(data.geom_xpos[palm_geoms].copy())
    palm_positions_a = np.asarray(palm_positions)
    anchor_frame = int(np.argmin(np.mean(palm_positions_a[:, :, 2], axis=1)))

    data.qpos[:] = qref[anchor_frame]
    mujoco.mj_forward(model, data)
    palms = data.geom_xpos[palm_geoms].copy()
    palm_midpoint = np.mean(palms, axis=0)
    palm_line = palms[1] - palms[0]
    palm_separation = float(np.linalg.norm(palm_line))
    if palm_separation < 1e-9:
        raise RuntimeError("Palm centers coincide at the lowest-hand frame")
    palm_line /= palm_separation

    current_geom_rotation = data.geom_xmat[box_geom].reshape(3, 3).copy()
    # The box's horizontal axis already closest to the palm line identifies the
    # intended grasp-through dimension. Keep its sign to avoid a 180-degree flip.
    lateral_axis = max(
        (0, 1),
        key=lambda axis: abs(float(np.dot(current_geom_rotation[:, axis], palm_line))),
    )
    if float(np.dot(current_geom_rotation[:, lateral_axis], palm_line)) < 0.0:
        palm_line = -palm_line
    cosine_before = float(
        np.clip(np.dot(current_geom_rotation[:, lateral_axis], palm_line), -1.0, 1.0)
    )

    vertical = current_geom_rotation[:, 2].copy()
    vertical -= float(np.dot(vertical, palm_line)) * palm_line
    vertical /= np.linalg.norm(vertical)
    if lateral_axis == 1:
        axis_y = palm_line
        axis_z = vertical
        axis_x = np.cross(axis_y, axis_z)
        axis_x /= np.linalg.norm(axis_x)
        axis_z = np.cross(axis_x, axis_y)
    else:
        axis_x = palm_line
        axis_z = vertical
        axis_y = np.cross(axis_z, axis_x)
        axis_y /= np.linalg.norm(axis_y)
        axis_z = np.cross(axis_x, axis_y)
    desired_geom_rotation = np.column_stack((axis_x, axis_y, axis_z))

    current_body_rotation = data.xmat[box_body].reshape(3, 3).copy()
    geom_from_body = current_body_rotation.T @ current_geom_rotation
    desired_body_rotation = desired_geom_rotation @ geom_from_body.T
    rotation_delta = desired_body_rotation @ current_body_rotation.T

    body_to_geom = current_body_rotation.T @ (
        data.geom_xpos[box_geom] - data.xpos[box_body]
    )
    desired_body_position = qref[anchor_frame, -7:-4].copy()
    desired_geom_offset = desired_body_rotation @ body_to_geom
    desired_body_position[:2] = palm_midpoint[:2] - desired_geom_offset[:2]
    position_delta = desired_body_position - qref[anchor_frame, -7:-4]

    for frame in range(len(qref)):
        current_rotation_flat = np.empty(9, dtype=np.float64)
        mujoco.mju_quat2Mat(current_rotation_flat, qref[frame, -4:])
        aligned_rotation = rotation_delta @ current_rotation_flat.reshape(3, 3)
        mujoco.mju_mat2Quat(qref[frame, -4:], aligned_rotation.ravel())
    qref[:, -7:-4] += position_delta

    data.qpos[:] = qref[anchor_frame]
    mujoco.mj_forward(model, data)
    aligned_geom_rotation = data.geom_xmat[box_geom].reshape(3, 3)
    cosine_after = float(
        np.clip(np.dot(aligned_geom_rotation[:, lateral_axis], palm_line), -1.0, 1.0)
    )
    center_minus_midpoint = data.geom_xpos[box_geom] - palm_midpoint
    rotation_correction_deg = math.degrees(
        math.acos(np.clip((np.trace(rotation_delta) - 1.0) / 2.0, -1.0, 1.0))
    )
    return {
        "enabled": True,
        "anchor_frame": anchor_frame,
        "search_end_frame": search_end,
        "palm_center_source": "left/right hand-mesh collision geom centers",
        "palm_midpoint_world_m": palm_midpoint.tolist(),
        "palm_line_world_unit": palm_line.tolist(),
        "palm_center_separation_m": palm_separation,
        "box_lateral_axis": int(lateral_axis),
        "line_axis_angle_before_deg": math.degrees(math.acos(cosine_before)),
        "line_axis_angle_after_deg": math.degrees(math.acos(cosine_after)),
        "box_geom_center_minus_palm_midpoint_world_m": center_minus_midpoint.tolist(),
        "horizontal_center_error_after_m": float(np.linalg.norm(center_minus_midpoint[:2])),
        "box_position_translation_world_m": position_delta.tolist(),
        "box_rotation_correction_deg": rotation_correction_deg,
        "vertical_position_policy": "preserve original box center z to avoid floor penetration",
    }


def run_trial(
    motion_path: Path,
    label: str,
    dimensions: tuple[float, float, float],
    *,
    mass: float,
    timestep: float,
    box_geometry: str,
    base_control: str,
    ground_height: float,
    box_friction: float,
    palm_adhesion: float,
    contact_latch: bool,
    contact_latch_mode: str,
    grasp_reference_blend_seconds: float,
    latch_release_frame: int,
    box_forward_offset: float,
    align_box_to_lowest_hand_line: bool,
    grip_force: float,
    grip_start_frame: int,
    grip_late_multiplier: float,
    grasp_impedance: float,
    grasp_damping: float,
    grasp_inset: float,
    settle_seconds: float,
    tail_seconds: float,
    video_path: Path | None = None,
    trace_path: Path | None = None,
) -> dict[str, object]:
    with np.load(motion_path, allow_pickle=False) as archive:
        qref = np.asarray(archive["qpos"], dtype=np.float64).copy()
        fps = float(archive["fps"].item())
    model = build_model(
        dimensions,
        mass,
        timestep,
        box_geometry,
        base_control,
        ground_height,
        box_friction,
        palm_adhesion,
        contact_latch,
        contact_latch_mode,
    )
    hand_line_alignment: dict[str, object] = {"enabled": False}
    if align_box_to_lowest_hand_line:
        hand_line_alignment = _align_box_reference_to_lowest_hand_line(model, qref)
    # Translate only the object's reference path along the initial robot-to-box
    # direction. The robot trajectory remains unchanged, so positive values put
    # the box farther in front. This is an optional residual adjustment after
    # hand-line alignment, and is zero in the hand-line-derived final replay.
    forward_xy = qref[0, -7:-5] - qref[0, :2]
    forward_norm = float(np.linalg.norm(forward_xy))
    if forward_norm < 1e-9:
        raise RuntimeError("Cannot infer robot-to-box forward direction from the first frame")
    forward_xy /= forward_norm
    box_translation_xy = box_forward_offset * forward_xy
    qref[:, -7:-5] += box_translation_xy
    data = mujoco.MjData(model)
    data.qpos[:] = qref[0]
    data.qvel[:] = 0.0
    root_mocap_id = -1
    if base_control == "mocap_weld":
        root_target_body = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "root_servo_target"
        )
        root_mocap_id = int(model.body_mocapid[root_target_body])
        data.mocap_pos[root_mocap_id] = qref[0, :3]
        data.mocap_quat[root_mocap_id] = qref[0, 3:7]
    grasp_mocap_id = -1
    if contact_latch and contact_latch_mode == "object_reference":
        grasp_target_body = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "object_reference_grasp_target"
        )
        grasp_mocap_id = int(model.body_mocapid[grasp_target_body])
        data.mocap_pos[grasp_mocap_id] = qref[0, -7:-4]
        data.mocap_quat[grasp_mocap_id] = qref[0, -4:]
    mujoco.mj_forward(model, data)

    # Reference generalized velocities respect quaternion tangent-space conventions.
    vref = np.zeros((len(qref), model.nv), dtype=np.float64)
    for index in range(len(qref) - 1):
        mujoco.mj_differentiatePos(model, vref[index], 1.0 / fps, qref[index], qref[index + 1])
    vref[-1] = vref[-2]
    aref = np.zeros_like(vref)
    aref[:-1] = np.diff(vref, axis=0) * fps
    aref[-1] = aref[-2]
    aref[:, :3] = np.clip(aref[:, :3], -60.0, 60.0)
    aref[:, 3:] = np.clip(aref[:, 3:], -120.0, 120.0)
    full_mass = np.empty((model.nv, model.nv), dtype=np.float64)
    jacobian_position = np.empty((3, model.nv), dtype=np.float64)
    jacobian_rotation = np.empty((3, model.nv), dtype=np.float64)

    actuator_joint = np.asarray(model.actuator_trnid[:, 0], dtype=np.int32)
    actuator_qadr = model.jnt_qposadr[actuator_joint]
    actuator_dadr = model.jnt_dofadr[actuator_joint]
    actuator_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(j)) for j in actuator_joint]
    gains = np.asarray([_joint_gains(name) for name in actuator_names])
    force_limits = np.empty(model.nu, dtype=np.float64)
    for aid, jid in enumerate(actuator_joint):
        if model.jnt_actfrclimited[jid]:
            force_limits[aid] = max(abs(model.jnt_actfrcrange[jid]))
        else:
            force_limits[aid] = 250.0

    box_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "physics_box_geom")
    box_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "physics_box")
    pelvis_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis_link")
    hand_bodies = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_yaw_Link"),
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_Link"),
    ]
    foot_bodies = {
        body_id
        for body_id in range(model.nbody)
        if (name := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)) is not None
        and ("foot" in name.lower() or "ankle" in name.lower())
    }
    reference_pick_frame = int(hand_line_alignment.get("anchor_frame", grip_start_frame))
    reference_pose_data = mujoco.MjData(model)
    reference_pose_data.qpos[:] = qref[reference_pick_frame]
    mujoco.mj_forward(model, reference_pose_data)
    reference_pick_vertical_axis_world = (
        reference_pose_data.geom_xmat[box_geom].reshape(3, 3)[:, 2].copy()
    )
    half_size = np.asarray(dimensions, dtype=np.float64) / 2.0
    duration = settle_seconds + (len(qref) - 1) / fps + tail_seconds
    steps = int(math.ceil(duration / timestep))
    render_stride = max(1, round(1.0 / (30.0 * timestep)))
    renderer = writer = None
    if video_path is not None:
        video_path.parent.mkdir(parents=True, exist_ok=True)
        renderer = mujoco.Renderer(model, height=540, width=960)
        writer = subprocess.Popen(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
                "960x540",
                "-framerate",
                "30",
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                str(video_path),
            ],
            stdin=subprocess.PIPE,
        )
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = (
            float(np.mean(qref[:, 0])),
            float(np.mean(qref[:, 1])),
            0.72,
        )
        camera.distance = 3.45
        camera.azimuth = 135.0
        camera.elevation = -17.0
    else:
        camera = None

    times: list[float] = []
    actual_box: list[np.ndarray] = []
    reference_box: list[np.ndarray] = []
    bottoms: list[float] = []
    horizontal: list[float] = []
    left_contacts: list[bool] = []
    right_contacts: list[bool] = []
    floor_contacts: list[bool] = []
    foot_contacts: list[bool] = []
    actual_box_qvel: list[np.ndarray] = []
    robot_root_error: list[float] = []
    joint_error: list[float] = []
    initial_box_xy = qref[0, -7:-5].copy()
    initial_box_z = float(qref[0, -5])
    grasp_anchors_local: list[np.ndarray] | None = None
    latch_eq = (
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "contact_grasp_latch")
        if contact_latch
        else -1
    )
    latch_time: float | None = None
    latch_release_time: float | None = None
    latch_hand_contacts_local: list[list[float]] | None = None
    latch_hand_midpoint_local: list[float] | None = None
    grasp_reference_correction_position: np.ndarray | None = None
    grasp_reference_correction_quaternion: np.ndarray | None = None
    latch_box_vertical_axis_world: np.ndarray | None = None
    release_box_vertical_axis_world: np.ndarray | None = None

    for step in range(steps):
        sim_time = step * timestep
        replay_time = min(max(sim_time - settle_seconds, 0.0), (len(qref) - 1) / fps)
        target, frame = _reference_at(qref, replay_time * fps)
        target_velocity = np.zeros(model.nv) if sim_time < settle_seconds else vref[frame]
        if (
            contact_latch_mode == "object_reference"
            and grasp_reference_correction_position is not None
            and grasp_reference_correction_quaternion is not None
            and latch_release_time is None
        ):
            blend_alpha = (
                1.0
                if grasp_reference_blend_seconds <= 0.0
                else min(
                    max(
                        (sim_time - (latch_time if latch_time is not None else sim_time))
                        / grasp_reference_blend_seconds,
                        0.0,
                    ),
                    1.0,
                )
            )
            blended_correction_position = (
                (1.0 - blend_alpha) * grasp_reference_correction_position
            )
            blended_correction_quaternion = _nlerp_quat(
                grasp_reference_correction_quaternion,
                np.asarray([1.0, 0.0, 0.0, 0.0]),
                blend_alpha,
            )
            grasp_target_position = np.empty(3, dtype=np.float64)
            grasp_target_quaternion = np.empty(4, dtype=np.float64)
            mujoco.mju_mulPose(
                grasp_target_position,
                grasp_target_quaternion,
                target[-7:-4],
                target[-4:],
                blended_correction_position,
                blended_correction_quaternion,
            )
            data.mocap_pos[grasp_mocap_id] = grasp_target_position
            data.mocap_quat[grasp_mocap_id] = grasp_target_quaternion

        data.qfrc_applied[:] = 0.0
        if base_control == "mocap_weld":
            # A physical 6-D constraint servos the dynamic pelvis to the reference;
            # unlike a qpos write, contacts are resolved by the dynamics solver.
            data.mocap_pos[root_mocap_id] = target[:3]
            data.mocap_quat[root_mocap_id] = target[3:7]
        else:
            # Dynamic floating-base force servo (no root pose writes).
            rotation_error = np.empty(3, dtype=np.float64)
            mujoco.mju_subQuat(rotation_error, target[3:7], data.qpos[3:7])
            data.qfrc_applied[:6] = data.qfrc_bias[:6]
            current_rotation_flat = np.empty(9, dtype=np.float64)
            target_rotation_flat = np.empty(9, dtype=np.float64)
            mujoco.mju_quat2Mat(current_rotation_flat, data.qpos[3:7])
            mujoco.mju_quat2Mat(target_rotation_flat, target[3:7])
            current_rotation = current_rotation_flat.reshape(3, 3)
            target_rotation = target_rotation_flat.reshape(3, 3)
            rotation_error_world = current_rotation @ rotation_error
            angular_velocity_world = current_rotation @ data.qvel[3:6]
            target_angular_velocity_world = target_rotation @ target_velocity[3:6]
            root_force = 4200.0 * (target[:3] - data.qpos[:3]) + 520.0 * (
                target_velocity[:3] - data.qvel[:3]
            )
            root_torque = 700.0 * rotation_error_world + 85.0 * (
                target_angular_velocity_world - angular_velocity_world
            )
            root_force = np.clip(root_force, -5000.0, 5000.0)
            root_torque = np.clip(root_torque, -900.0, 900.0)
            mujoco.mj_applyFT(
                model,
                data,
                root_force,
                root_torque,
                data.xipos[pelvis_body],
                pelvis_body,
                data.qfrc_applied,
            )

        # Effort-limited joint PD plus inverse-dynamics bias compensation.
        pos_error = target[actuator_qadr] - data.qpos[actuator_qadr]
        vel_error = target_velocity[actuator_dadr] - data.qvel[actuator_dadr]
        mujoco.mj_fullM(model, data, full_mass)
        inverse_dynamics_ff = full_mass @ aref[frame] + data.qfrc_bias
        command = gains[:, 0] * pos_error + gains[:, 1] * vel_error + inverse_dynamics_ff[actuator_dadr]
        if grip_force > 0.0 and grip_start_frame <= frame <= 125:
            box_position = data.qpos[-7:-4]
            for hand_body in hand_bodies:
                hand_offset = data.xipos[hand_body] - box_position
                horizontal_offset = hand_offset.copy()
                horizontal_offset[2] = 0.0
                horizontal_distance = np.linalg.norm(horizontal_offset)
                if horizontal_distance < 1e-9 or np.linalg.norm(hand_offset) > 0.55:
                    continue
                # Each palm is pulled horizontally toward the box center.  This
                # cannot inject upward force; lift still comes from contact friction.
                active_grip_force = grip_force * (grip_late_multiplier if frame >= 25 else 1.0)
                hand_force = -active_grip_force * horizontal_offset / horizontal_distance
                mujoco.mj_jacBodyCom(
                    model,
                    data,
                    jacobian_position,
                    jacobian_rotation,
                    hand_body,
                )
                command += jacobian_position[:, actuator_dadr].T @ hand_force
        if grasp_impedance > 0.0:
            contact_left, contact_right, _, _ = _contact_flags(
                model, data, box_geom, foot_bodies
            )
            box_rotation_flat = np.empty(9, dtype=np.float64)
            mujoco.mju_quat2Mat(box_rotation_flat, data.qpos[-4:])
            box_rotation = box_rotation_flat.reshape(3, 3)
            box_position = data.qpos[-7:-4]
            if grasp_anchors_local is None and frame >= grip_start_frame and contact_left and contact_right:
                grasp_anchors_local = []
                for hand_body in hand_bodies:
                    anchor = box_rotation.T @ (data.xipos[hand_body] - box_position)
                    anchor_horizontal = anchor.copy()
                    anchor_horizontal[2] = 0.0
                    horizontal_norm = np.linalg.norm(anchor_horizontal)
                    if horizontal_norm > 1e-9:
                        anchor[:2] -= grasp_inset * anchor_horizontal[:2] / horizontal_norm
                    grasp_anchors_local.append(anchor)
            if grasp_anchors_local is not None:
                box_linear_velocity = data.qvel[-6:-3]
                for hand_body, anchor in zip(hand_bodies, grasp_anchors_local, strict=True):
                    desired_hand_position = box_position + box_rotation @ anchor
                    mujoco.mj_jacBodyCom(
                        model,
                        data,
                        jacobian_position,
                        jacobian_rotation,
                        hand_body,
                    )
                    hand_velocity = jacobian_position @ data.qvel
                    impedance_force = grasp_impedance * (desired_hand_position - data.xipos[hand_body])
                    impedance_force += grasp_damping * (box_linear_velocity - hand_velocity)
                    force_norm = np.linalg.norm(impedance_force)
                    if force_norm > 80.0:
                        impedance_force *= 80.0 / force_norm
                    command += jacobian_position[:, actuator_dadr].T @ impedance_force
        if contact_latch and latch_time is None and frame >= grip_start_frame:
            contact_left, contact_right, _, _ = _contact_flags(
                model, data, box_geom, foot_bodies
            )
            if contact_left and contact_right:
                latch_hand_contacts_local, latch_hand_midpoint_local = (
                    _hand_contact_locations_local(
                        model, data, box_geom, box_body, hand_bodies
                    )
                )
                if contact_latch_mode == "object_reference":
                    inverse_reference_position = np.empty(3, dtype=np.float64)
                    inverse_reference_quaternion = np.empty(4, dtype=np.float64)
                    grasp_reference_correction_position = np.empty(3, dtype=np.float64)
                    grasp_reference_correction_quaternion = np.empty(4, dtype=np.float64)
                    mujoco.mju_negPose(
                        inverse_reference_position,
                        inverse_reference_quaternion,
                        target[-7:-4],
                        target[-4:],
                    )
                    mujoco.mju_mulPose(
                        grasp_reference_correction_position,
                        grasp_reference_correction_quaternion,
                        inverse_reference_position,
                        inverse_reference_quaternion,
                        data.xpos[box_body],
                        data.xquat[box_body],
                    )
                    data.mocap_pos[grasp_mocap_id] = data.xpos[box_body]
                    data.mocap_quat[grasp_mocap_id] = data.xquat[box_body]
                else:
                    latch_body = hand_bodies[1]
                    inverse_position = np.empty(3, dtype=np.float64)
                    inverse_quaternion = np.empty(4, dtype=np.float64)
                    relative_position = np.empty(3, dtype=np.float64)
                    relative_quaternion = np.empty(4, dtype=np.float64)
                    mujoco.mju_negPose(
                        inverse_position,
                        inverse_quaternion,
                        data.xpos[latch_body],
                        data.xquat[latch_body],
                    )
                    mujoco.mju_mulPose(
                        relative_position,
                        relative_quaternion,
                        inverse_position,
                        inverse_quaternion,
                        data.xpos[box_body],
                        data.xquat[box_body],
                    )
                    model.eq_data[latch_eq, 3:6] = relative_position
                    model.eq_data[latch_eq, 6:10] = relative_quaternion
                data.eq_active[latch_eq] = True
                latch_time = sim_time
                latch_box_vertical_axis_world = (
                    data.geom_xmat[box_geom].reshape(3, 3)[:, 2].copy()
                )
        if (
            contact_latch
            and latch_time is not None
            and latch_release_time is None
            and latch_release_frame >= 0
            and frame >= latch_release_frame
        ):
            # Once the reference has lowered the box, gravity and collision alone
            # determine whether it settles safely away from the feet.
            data.eq_active[latch_eq] = False
            latch_release_time = sim_time
            release_box_vertical_axis_world = (
                data.geom_xmat[box_geom].reshape(3, 3)[:, 2].copy()
            )
        data.ctrl[:] = np.clip(command, -force_limits, force_limits)
        if np.any(data.qfrc_applied[37:] != 0.0):
            raise AssertionError("The free box received an applied generalized force")
        mujoco.mj_step(model, data)

        pos = data.qpos[-7:-4].copy()
        quat = data.qpos[-4:].copy()
        left, right, floor, foot = _contact_flags(model, data, box_geom, foot_bodies)
        times.append(sim_time)
        actual_box.append(np.r_[pos, quat])
        reference_box.append(target[-7:].copy())
        bottoms.append(_box_bottom(pos, quat, half_size) - ground_height)
        horizontal.append(float(np.linalg.norm(pos[:2] - initial_box_xy)))
        left_contacts.append(left)
        right_contacts.append(right)
        floor_contacts.append(floor)
        foot_contacts.append(foot)
        actual_box_qvel.append(data.qvel[-6:].copy())
        robot_root_error.append(float(np.linalg.norm(target[:3] - data.qpos[:3])))
        joint_error.append(float(np.sqrt(np.mean(pos_error * pos_error))))

        if renderer is not None and writer is not None and step % render_stride == 0:
            renderer.update_scene(data, camera=camera)
            frame_rgb = renderer.render()
            assert writer.stdin is not None
            writer.stdin.write(frame_rgb.tobytes())

    if writer is not None:
        assert writer.stdin is not None
        writer.stdin.close()
        if writer.wait() != 0:
            raise RuntimeError(f"ffmpeg failed while writing {video_path}")
    if renderer is not None:
        renderer.close()

    times_a = np.asarray(times)
    actual_a = np.asarray(actual_box)
    reference_a = np.asarray(reference_box)
    bottoms_a = np.asarray(bottoms)
    horizontal_a = np.asarray(horizontal)
    left_a = np.asarray(left_contacts, dtype=bool)
    right_a = np.asarray(right_contacts, dtype=bool)
    floor_a = np.asarray(floor_contacts, dtype=bool)
    foot_a = np.asarray(foot_contacts, dtype=bool)
    box_qvel_a = np.asarray(actual_box_qvel)
    replay_mask = (times_a >= settle_seconds) & (times_a <= settle_seconds + (len(qref) - 1) / fps)
    settle_window = (times_a >= max(0.0, settle_seconds - 0.05)) & (times_a <= settle_seconds)
    settled_center_z = float(np.median(actual_a[settle_window, 2]))
    center_rise = actual_a[:, 2] - settled_center_z
    # For a non-axis-aligned scanned mesh, an OBB-from-extents bottom estimate is
    # conservative.  Lift therefore uses center rise plus loss of floor contact.
    elevated = replay_mask & (center_rise > 0.05) & ~floor_a
    ref_lifted = replay_mask & (reference_a[:, 2] > initial_box_z + 0.15)
    both_contact = left_a & right_a
    rmse = float(np.sqrt(np.mean(np.sum((actual_a[ref_lifted, :3] - reference_a[ref_lifted, :3]) ** 2, axis=1))))
    sustained = _longest_true_run(elevated, timestep)
    elevated_transport = float(horizontal_a[elevated].max(initial=0.0))
    max_bottom = float(bottoms_a[replay_mask].max())
    max_center_rise = float(center_rise[replay_mask].max())
    left_contact_s = float(np.count_nonzero(replay_mask & left_a) * timestep)
    right_contact_s = float(np.count_nonzero(replay_mask & right_a) * timestep)
    bilateral_contact_s = float(np.count_nonzero(replay_mask & both_contact) * timestep)
    final_window_seconds = max(0.10, min(0.30, tail_seconds))
    final_window = times_a >= times_a[-1] - final_window_seconds
    final_floor_fraction = float(np.mean(floor_a[final_window]))
    final_foot_fraction = float(np.mean(foot_a[final_window]))
    final_linear_speed = float(np.max(np.linalg.norm(box_qvel_a[final_window, :3], axis=1)))
    final_angular_speed = float(np.max(np.linalg.norm(box_qvel_a[final_window, 3:], axis=1)))
    final_box_vertical_axis_world = data.geom_xmat[box_geom].reshape(3, 3)[:, 2].copy()
    same_face_cosine = float(
        np.clip(
            np.dot(reference_pick_vertical_axis_world, final_box_vertical_axis_world),
            -1.0,
            1.0,
        )
    )
    actual_latch_to_final_face_cosine = (
        float(np.clip(np.dot(latch_box_vertical_axis_world, final_box_vertical_axis_world), -1.0, 1.0))
        if latch_box_vertical_axis_world is not None
        else None
    )
    release_to_final_face_cosine = (
        float(
            np.clip(
                np.dot(release_box_vertical_axis_world, final_box_vertical_axis_world),
                -1.0,
                1.0,
            )
        )
        if release_box_vertical_axis_world is not None
        else None
    )
    same_face_error_deg = math.degrees(math.acos(same_face_cosine))
    actual_latch_to_final_face_error_deg = (
        math.degrees(math.acos(actual_latch_to_final_face_cosine))
        if actual_latch_to_final_face_cosine is not None
        else None
    )
    post_release_roll_deg = (
        math.degrees(math.acos(release_to_final_face_cosine))
        if release_to_final_face_cosine is not None
        else None
    )
    placement_required = contact_latch and latch_release_frame >= 0
    placement_success = bool(
        not placement_required
        or (
            latch_release_time is not None
            and final_floor_fraction >= 0.90
            and final_foot_fraction <= 0.01
            and final_linear_speed <= 0.10
            and final_angular_speed <= 0.50
            and same_face_error_deg <= 20.0
            and post_release_roll_deg is not None
            and post_release_roll_deg <= 20.0
        )
    )
    success = bool(
        sustained >= 0.40
        and max_center_rise >= 0.12
        and elevated_transport >= 0.50
        and left_contact_s >= 0.05
        and right_contact_s >= 0.05
        and rmse <= 0.35
        and placement_success
    )
    metrics: dict[str, object] = {
        "motion": label,
        "motion_file": str(motion_path),
        "dimensions_m": [float(x) for x in dimensions],
        "mass_kg": mass,
        "box_geometry": box_geometry,
        "base_control": base_control,
        "ground_height_m": ground_height,
        "box_sliding_friction": box_friction,
        "palm_contact_adhesion_n": palm_adhesion,
        "contact_latch_enabled": contact_latch,
        "contact_latch_mode": contact_latch_mode,
        "grasp_reference_blend_seconds": grasp_reference_blend_seconds,
        "contact_latch_time_s": latch_time,
        "contact_latch_release_frame": latch_release_frame,
        "contact_latch_release_time_s": latch_release_time,
        "latch_hand_contact_locations_box_local_m": latch_hand_contacts_local,
        "latch_hand_midpoint_box_local_m": latch_hand_midpoint_local,
        "latch_hand_midpoint_center_error_m": (
            float(np.linalg.norm(latch_hand_midpoint_local))
            if latch_hand_midpoint_local is not None
            else None
        ),
        "box_forward_offset_m": box_forward_offset,
        "box_translation_xy_m": box_translation_xy.tolist(),
        "lowest_hand_line_alignment": hand_line_alignment,
        "horizontal_grip_force_per_hand_n": grip_force,
        "grip_start_frame": grip_start_frame,
        "grip_late_multiplier_after_frame_25": grip_late_multiplier,
        "grasp_impedance_n_per_m": grasp_impedance,
        "grasp_damping_ns_per_m": grasp_damping,
        "grasp_inset_m": grasp_inset,
        "success": success,
        "peak_bottom_clearance_m": max_bottom,
        "peak_center_rise_from_settled_m": max_center_rise,
        "peak_center_height_m": float(actual_a[replay_mask, 2].max()),
        "longest_continuous_lift_s": sustained,
        "elevated_horizontal_transport_m": elevated_transport,
        "object_position_rmse_during_reference_lift_m": rmse,
        "left_hand_contact_s": left_contact_s,
        "right_hand_contact_s": right_contact_s,
        "bilateral_hand_contact_s": bilateral_contact_s,
        "floor_contact_fraction": float(np.mean(floor_a[replay_mask])),
        "final_floor_contact_fraction": final_floor_fraction,
        "final_foot_contact_fraction": final_foot_fraction,
        "final_max_linear_speed_m_per_s": final_linear_speed,
        "final_max_angular_speed_rad_per_s": final_angular_speed,
        "reference_pickup_to_final_support_face_error_deg": same_face_error_deg,
        "actual_latch_to_final_support_face_error_deg": actual_latch_to_final_face_error_deg,
        "post_release_support_face_roll_deg": post_release_roll_deg,
        "pickup_box_vertical_axis_world": (
            latch_box_vertical_axis_world.tolist()
            if latch_box_vertical_axis_world is not None
            else None
        ),
        "reference_pickup_box_vertical_axis_world": (
            reference_pick_vertical_axis_world.tolist()
        ),
        "release_box_vertical_axis_world": (
            release_box_vertical_axis_world.tolist()
            if release_box_vertical_axis_world is not None
            else None
        ),
        "final_box_vertical_axis_world": final_box_vertical_axis_world.tolist(),
        "safe_placement_success": placement_success,
        "robot_root_position_rmse_m": float(np.sqrt(np.mean(np.square(np.asarray(robot_root_error)[replay_mask])))),
        "joint_position_rmse_rad": float(np.sqrt(np.mean(np.square(np.asarray(joint_error)[replay_mask])))),
        "box_force_write_check": "PASS: qfrc_applied[37:] stayed exactly zero",
        "success_rule": (
            "continuous center rise >5 cm with no floor contact for >=0.40 s; peak rise >=0.12 m; "
            "transport while elevated >=0.50 m; each hand contacts >=0.05 s; lifted-phase position RMSE <=0.35 m; "
            "when latch release is enabled: final floor contact >=90%, foot contact <=1%, max final linear speed "
            "<=0.10 m/s and angular speed <=0.50 rad/s; pickup-to-final support-face error and "
            "post-release roll <=20 deg"
        ),
    }
    if trace_path is not None:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            trace_path,
            time=times_a,
            actual_box_qpos=actual_a,
            reference_box_qpos=reference_a,
            bottom_clearance=bottoms_a,
            horizontal_displacement=horizontal_a,
            left_hand_contact=left_a,
            right_hand_contact=right_a,
            floor_contact=floor_a,
            foot_contact=foot_a,
            actual_box_qvel=box_qvel_a,
            dimensions_m=np.asarray(dimensions),
            mass_kg=np.asarray(mass),
        )
    return metrics


def candidate_dimensions(
    x_values: list[float] | None = None,
    y_values: list[float] | None = None,
    z_values: list[float] | None = None,
) -> list[tuple[float, float, float]]:
    # x: front/back depth, y: between the two palms, z: box height.
    # The source mesh measures 0.471 x 0.459 x 0.408 m.
    return [
        (x, y, z)
        for x in (x_values or [0.32, 0.40, 0.471])
        for y in (y_values or [0.42, 0.459, 0.50, 0.54])
        for z in (z_values or [0.24, 0.32, 0.408])
    ]


def write_search_outputs(output: Path, rows: list[dict[str, object]]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "search_results.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (output / "search_results.csv").open("w", newline="") as handle:
        fields = list(rows[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            encoded = dict(row)
            encoded["dimensions_m"] = "x".join(str(x) for x in row["dimensions_m"])
            writer.writerow(encoded)


def select_common_winner(rows: list[dict[str, object]]) -> tuple[tuple[float, float, float], list[dict[str, object]]]:
    grouped: dict[tuple[float, float, float], list[dict[str, object]]] = {}
    for row in rows:
        dims = tuple(float(x) for x in row["dimensions_m"])
        grouped.setdefault(dims, []).append(row)
    common = [(dims, values) for dims, values in grouped.items() if len(values) == len(MOTIONS) and all(v["success"] for v in values)]
    if not common:
        # Still report the strongest physics result rather than falsely declaring success.
        ranked = sorted(
            grouped.items(),
            key=lambda item: (
                sum(bool(v["success"]) for v in item[1]),
                min(float(v["longest_continuous_lift_s"]) for v in item[1]),
                min(float(v["elevated_horizontal_transport_m"]) for v in item[1]),
                -max(float(v["object_position_rmse_during_reference_lift_m"]) for v in item[1]),
            ),
            reverse=True,
        )
        return ranked[0]
    # Prefer robust lift duration/transport, then a smaller volume among similarly good candidates.
    common.sort(
        key=lambda item: (
            min(float(v["longest_continuous_lift_s"]) for v in item[1]),
            min(float(v["elevated_horizontal_transport_m"]) for v in item[1]),
            -math.prod(item[0]),
        ),
        reverse=True,
    )
    return common[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("search", "replay"), default="search")
    parser.add_argument("--motion", choices=("both", *MOTIONS), default="both")
    parser.add_argument(
        "--original-motion",
        type=Path,
        default=MOTIONS["original"],
        help="original OmniRetarget trajectory (.npz)",
    )
    parser.add_argument(
        "--semantic-b4-motion",
        type=Path,
        default=MOTIONS["semantic_b4"],
        help="Semantic B4 trajectory (.npz)",
    )
    parser.add_argument("--dimensions", nargs=3, type=float, metavar=("X", "Y", "Z"))
    parser.add_argument("--x-values", nargs="+", type=float, help="override search-grid x dimensions")
    parser.add_argument("--y-values", nargs="+", type=float, help="override search-grid y dimensions")
    parser.add_argument("--z-values", nargs="+", type=float, help="override search-grid z dimensions")
    parser.add_argument("--mass", type=float, default=0.1)
    parser.add_argument("--box-friction", type=float, default=1.2)
    parser.add_argument("--palm-adhesion", type=float, default=0.0)
    parser.add_argument("--contact-latch", action="store_true")
    parser.add_argument(
        "--contact-latch-mode",
        choices=("wrist", "object_reference"),
        default="wrist",
        help=(
            "attach the soft latch to the right wrist or to the original "
            "G1/object reference pose after bilateral contact"
        ),
    )
    parser.add_argument(
        "--grasp-reference-blend-seconds",
        type=float,
        default=0.75,
        help="time to blend a contact-triggered reference latch onto the original object pose",
    )
    parser.add_argument(
        "--latch-release-frame",
        type=int,
        default=-1,
        help="disable the contact latch at this reference frame; negative keeps it engaged",
    )
    parser.add_argument(
        "--box-forward-offset",
        type=float,
        default=0.0,
        help="translate the box path this many metres farther along the initial robot-to-box direction",
    )
    parser.add_argument(
        "--original-box-forward-offset",
        type=float,
        help="override --box-forward-offset for the original OmniRetarget motion",
    )
    parser.add_argument(
        "--semantic-b4-box-forward-offset",
        type=float,
        help="override --box-forward-offset for the Semantic B4 motion",
    )
    parser.add_argument(
        "--align-box-to-lowest-hand-line",
        action="store_true",
        help=(
            "derive box horizontal center and rotation from the line between "
            "the two palm collision meshes at their pre-lift lowest frame"
        ),
    )
    parser.add_argument("--box-geometry", choices=("mesh", "cuboid"), default="mesh")
    parser.add_argument("--base-control", choices=("mocap_weld", "wrench"), default="mocap_weld")
    parser.add_argument(
        "--ground-height",
        type=float,
        default=-0.102,
        help="world z of the plane; -0.102 m aligns it with the first-frame A3 foot collision geometry",
    )
    parser.add_argument(
        "--grip-force",
        type=float,
        default=0.0,
        help="horizontal squeeze per palm, converted to effort-limited joint torques and never applied to the box",
    )
    parser.add_argument("--grip-start-frame", type=int, default=2)
    parser.add_argument("--grip-late-multiplier", type=float, default=1.0)
    parser.add_argument("--grasp-impedance", type=float, default=0.0)
    parser.add_argument("--grasp-damping", type=float, default=20.0)
    parser.add_argument("--grasp-inset", type=float, default=0.012)
    parser.add_argument("--timestep", type=float, default=0.001)
    parser.add_argument("--settle-seconds", type=float, default=0.35)
    parser.add_argument("--tail-seconds", type=float, default=0.35)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--render", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    motions = {
        "original": args.original_motion,
        "semantic_b4": args.semantic_b4_motion,
    }
    selected = motions if args.motion == "both" else {args.motion: motions[args.motion]}
    if args.mode == "replay" and args.dimensions is None:
        raise SystemExit("--mode replay requires --dimensions X Y Z")
    dimensions = (
        [tuple(args.dimensions)]
        if args.dimensions
        else candidate_dimensions(args.x_values, args.y_values, args.z_values)
    )
    rows: list[dict[str, object]] = []
    total = len(dimensions) * len(selected)
    for candidate_index, dims in enumerate(dimensions, start=1):
        for label, motion_path in selected.items():
            box_forward_offset = args.box_forward_offset
            if label == "original" and args.original_box_forward_offset is not None:
                box_forward_offset = args.original_box_forward_offset
            if label == "semantic_b4" and args.semantic_b4_box_forward_offset is not None:
                box_forward_offset = args.semantic_b4_box_forward_offset
            video = args.output / label / "physics_replay.mp4" if args.render else None
            trace = args.output / label / "physics_trace.npz" if args.mode == "replay" else None
            result = run_trial(
                motion_path,
                label,
                dims,
                mass=args.mass,
                timestep=args.timestep,
                box_geometry=args.box_geometry,
                base_control=args.base_control,
                ground_height=args.ground_height,
                box_friction=args.box_friction,
                palm_adhesion=args.palm_adhesion,
                contact_latch=args.contact_latch,
                contact_latch_mode=args.contact_latch_mode,
                grasp_reference_blend_seconds=args.grasp_reference_blend_seconds,
                latch_release_frame=args.latch_release_frame,
                box_forward_offset=box_forward_offset,
                align_box_to_lowest_hand_line=args.align_box_to_lowest_hand_line,
                grip_force=args.grip_force,
                grip_start_frame=args.grip_start_frame,
                grip_late_multiplier=args.grip_late_multiplier,
                grasp_impedance=args.grasp_impedance,
                grasp_damping=args.grasp_damping,
                grasp_inset=args.grasp_inset,
                settle_seconds=args.settle_seconds,
                tail_seconds=args.tail_seconds,
                video_path=video,
                trace_path=trace,
            )
            rows.append(result)
            print(
                f"[{len(rows):03d}/{total:03d}] {label:11s} dims={dims} success={result['success']} "
                f"lift={result['longest_continuous_lift_s']:.3f}s "
                f"rise={result['peak_center_rise_from_settled_m']:.3f}m "
                f"transport={result['elevated_horizontal_transport_m']:.3f}m "
                f"rmse={result['object_position_rmse_during_reference_lift_m']:.3f}m"
            )
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == "search":
        write_search_outputs(args.output, rows)
        if args.motion == "both":
            winner, winner_rows = select_common_winner(rows)
            summary = {
                "common_success": all(bool(row["success"]) for row in winner_rows),
                "winner_dimensions_m": winner,
                "winner_results": winner_rows,
            }
            (args.output / "winner.json").write_text(json.dumps(summary, indent=2) + "\n")
            print("WINNER", json.dumps(summary, indent=2))
    else:
        (args.output / "replay_results.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
