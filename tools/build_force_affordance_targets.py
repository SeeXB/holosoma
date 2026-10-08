#!/usr/bin/env python3
"""Build fixed force-affordance targets directly from an official SMPL input.

The builder never reads a retargeted robot trajectory.  Source wrists choose
natural opposing object faces, source object kinematics drive a friction-cone
wrench design, and robot-local palm points use the original G1 rubber-hand
geometry. The resulting artifact is consumed inside the normal Semantic-B4 SQP
loop without changing its 2/4 iteration schedule.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys

import cvxpy as cp
import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.optimize import least_squares
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/holosoma_retargeting"))

from holosoma_retargeting.config_types.data_type import SMPLH_DEMO_JOINTS  # noqa: E402
from holosoma_retargeting.semantic_keyframes.contact_targets import (  # noqa: E402
    SCHEMA,
    load_retarget_contact_targets,
)
from holosoma_retargeting.semantic_keyframes.runtime import (  # noqa: E402
    load_semantic_plan_projection,
)
from holosoma_retargeting.src.utils import load_intermimic_data  # noqa: E402

SIDES = ("left", "right")
PARTS = ("left_hand", "right_hand")
WRIST_NAMES = {"left": "L_Wrist", "right": "R_Wrist"}
ORIGINAL_ROBOT_BODIES = {
    "left_hand": "left_rubber_hand_link",
    "right_hand": "right_rubber_hand_link",
}
ORIGINAL_PALM_DIRECTIONS = {
    "left": np.array([0.0, -1.0, 0.0]),
    "right": np.array([0.0, 1.0, 0.0]),
}
# Legacy fallback coordinates in the unchanged rubber-hand body frames.
# Production runs use directional_support on the physical hand meshes.
ORIGINAL_PALM_POINTS = {
    "left": np.array([0.0442511839, -0.0095724127, -0.0000468470]),
    "right": np.array([0.0442911360, 0.0096068154, -0.0000482072]),
}
DEFAULT_HAND_MESHES = {
    side: (ROOT / "src/holosoma_retargeting/holosoma_retargeting/models/g1/assets" / f"{side}_rubber_hand.obj")
    for side in SIDES
}
SPHERE_ROBOT_BODIES = {
    "left_hand": "left_sphere_hand_link",
    "right_hand": "right_sphere_hand_link",
}
SPHERE_PALM_DIRECTIONS = {
    "left": np.array([0.0, -1.0, 0.0]),
    "right": np.array([0.0, 1.0, 0.0]),
}
# Surface support points previously validated for the deployment half-sphere.
SPHERE_PALM_POINTS = {
    "left": np.array([0.0, -0.03987813, 0.02498267]),
    "right": np.array([0.0, 0.03987813, 0.02498267]),
}
SPHERE_HAND_MESH = ROOT / "src/holosoma_retargeting/holosoma_retargeting/models/g1/assets" / "half_sphere.obj"


def _hand_geometry(hand_model: str) -> tuple[dict, dict, dict]:
    if hand_model == "original":
        return ORIGINAL_ROBOT_BODIES, ORIGINAL_PALM_POINTS, ORIGINAL_PALM_DIRECTIONS
    if hand_model == "sphere":
        return SPHERE_ROBOT_BODIES, SPHERE_PALM_POINTS, SPHERE_PALM_DIRECTIONS
    raise ValueError(f"unsupported hand model: {hand_model}")


def _directional_support_point(
    mesh: trimesh.Trimesh,
    direction: np.ndarray,
    band_m: float,
) -> tuple[np.ndarray, dict[str, object]]:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    projection = vertices @ direction
    support = float(np.max(projection))
    selected = np.flatnonzero(projection >= support - band_m)
    if not len(selected):
        raise RuntimeError("directional support band selected no hand vertices")
    point = np.mean(vertices[selected], axis=0)
    point += (support - float(point @ direction)) * direction
    overshoot = float(np.max((vertices - point) @ direction))
    return point, {
        "support_band_m": band_m,
        "support_vertex_count": int(len(selected)),
        "support_projection_m": support,
        "directional_support_overshoot_m": overshoot,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


@dataclass(frozen=True)
class Anchor:
    side: str
    point: np.ndarray
    normal: np.ndarray
    source_face: int


def _object_kinematics(
    poses_wxyz_xyz: np.ndarray,
    fps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dt = 1.0 / fps
    poses = np.asarray(poses_wxyz_xyz, dtype=np.float64)
    rotations = Rotation.from_quat(poses[:, :4][:, [1, 2, 3, 0]]).as_matrix()
    position = poses[:, 4:7]
    window = min(9, len(position) if len(position) % 2 else len(position) - 1)
    smooth = savgol_filter(position, window, 3, axis=0, mode="interp") if window >= 5 else position
    velocity = np.gradient(smooth, dt, axis=0, edge_order=2)
    acceleration = np.gradient(velocity, dt, axis=0, edge_order=2)
    omega = np.zeros_like(position)
    for frame in range(len(position) - 1):
        delta = Rotation.from_matrix(rotations[frame + 1] @ rotations[frame].T).as_rotvec()
        omega[frame] = delta / dt
    omega[-1] = omega[-2]
    if window >= 5:
        omega = savgol_filter(omega, window, 3, axis=0, mode="interp")
    alpha = np.gradient(omega, dt, axis=0, edge_order=2)
    return rotations, acceleration, omega, alpha


def _outward_face_normals(mesh: trimesh.Trimesh) -> np.ndarray:
    normals = np.asarray(mesh.face_normals, dtype=np.float64).copy()
    center = np.asarray(mesh.bounding_box.centroid, dtype=np.float64)
    centers = np.asarray(mesh.triangles_center, dtype=np.float64)
    inward = np.einsum("ij,ij->i", normals, centers - center) < 0.0
    normals[inward] *= -1.0
    return normals


def _select_source_anchors(
    mesh: trimesh.Trimesh,
    wrist_local: dict[str, np.ndarray],
    frame: int,
) -> dict[str, Anchor]:
    center = np.asarray(mesh.bounding_box.centroid, dtype=np.float64)
    normals = _outward_face_normals(mesh)
    probabilities = np.asarray(mesh.area_faces, dtype=np.float64).copy()
    probabilities /= probabilities.sum()
    rng = np.random.default_rng(0)
    sampled = normals[rng.choice(len(normals), size=20000, replace=True, p=probabilities)]
    dominant, _ = kmeans2(sampled, 6, minit="++", iter=100, seed=0)
    dominant /= np.linalg.norm(dominant, axis=1, keepdims=True) + 1e-12
    # Select the two sides jointly. Independently taking the extrema of the
    # clustered normals can return non-antipodal clusters on objects with many
    # oblique surfaces (chairs are a common example). For each dominant axis,
    # evaluate the nearest left patch along +axis and the nearest right patch
    # along -axis, then choose the wrist-adjacent antipodal pair.
    best: tuple[float, dict[str, Anchor]] | None = None
    for axis in np.concatenate((dominant, -dominant), axis=0):
        trial: dict[str, Anchor] = {}
        distance_sum = 0.0
        valid = True
        for side, sign in (("left", 1.0), ("right", -1.0)):
            desired = sign * axis
            candidates = np.flatnonzero(normals @ desired >= 0.95)
            if not len(candidates):
                valid = False
                break
            query = np.repeat(wrist_local[side][frame][None], len(candidates), axis=0)
            points = trimesh.triangles.closest_point(mesh.triangles[candidates], query)
            distances = np.linalg.norm(points - query, axis=1)
            local = int(np.argmin(distances))
            point = np.asarray(points[local], dtype=np.float64)
            normal = np.asarray(desired, dtype=np.float64)
            if float(normal @ (point - center)) < -1e-4:
                valid = False
                break
            distance_sum += float(distances[local])
            trial[side] = Anchor(side, point, normal, int(candidates[local]))
        if valid and (best is None or distance_sum < best[0]):
            best = (distance_sum, trial)
    if best is None:
        raise RuntimeError("no wrist-adjacent opposing object-face pair")
    result = best[1]
    if float(result["left"].normal @ result["right"].normal) > -0.95:
        raise RuntimeError("selected grasp faces are not opposing")
    return result


def _solve_wrench(
    poses: np.ndarray,
    fps: float,
    anchors: dict[str, Anchor],
    start: int,
    end: int,
    mass: float,
    friction: float,
    min_normal: float,
    max_normal: float,
) -> tuple[dict[str, object], np.ndarray]:
    rotations, acceleration, omega, alpha = _object_kinematics(poses, fps)
    frames = np.arange(start, end + 1, dtype=np.int64)
    forces = cp.Variable((len(frames), 2, 3), name="contact_force_world")
    gravity = np.array([0.0, 0.0, -9.81], dtype=np.float64)
    inertia_body = np.eye(3, dtype=np.float64) * (0.002 * mass / 0.1)
    objective = []
    constraints = []
    required_force = np.empty((len(frames), 3))
    required_torque = np.empty((len(frames), 3))
    grasp_matrices = []
    for local, frame in enumerate(frames):
        rotation = rotations[frame]
        force_target = mass * (acceleration[frame] - gravity)
        inertia_world = rotation @ inertia_body @ rotation.T
        torque_target = inertia_world @ alpha[frame] + np.cross(omega[frame], inertia_world @ omega[frame])
        required_force[local] = force_target
        required_torque[local] = torque_target
        force_sum = cp.sum(forces[local], axis=0)
        torque_sum = cp.Constant(np.zeros(3))
        rows = []
        for index, side in enumerate(SIDES):
            inward = -(rotation @ anchors[side].normal)
            inward /= np.linalg.norm(inward) + 1e-12
            arm = rotation @ anchors[side].point
            normal_force = cp.Constant(inward) @ forces[local, index]
            tangential = forces[local, index] - normal_force * inward
            constraints += [
                normal_force >= min_normal,
                normal_force <= max_normal,
                cp.norm(tangential, 2) <= friction * normal_force,
            ]
            skew = np.array(
                [
                    [0.0, -arm[2], arm[1]],
                    [arm[2], 0.0, -arm[0]],
                    [-arm[1], arm[0], 0.0],
                ]
            )
            torque_sum = torque_sum + cp.Constant(skew) @ forces[local, index]
            rows.append(np.vstack((np.eye(3), skew)))
        grasp_matrices.append(np.hstack(rows))
        objective += [
            cp.sum_squares(force_sum - force_target),
            cp.sum_squares((torque_sum - torque_target) / 0.1),
            1e-4 * cp.sum_squares(forces[local]),
        ]
    if len(frames) > 1:
        objective.append(2e-3 * cp.sum_squares(forces[1:] - forces[:-1]))
    problem = cp.Problem(cp.Minimize(cp.sum(objective)), constraints)
    problem.solve(solver=cp.CLARABEL)
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"wrench allocation failed: {problem.status}")
    values = np.asarray(forces.value, dtype=np.float64)
    force_errors = []
    torque_errors = []
    normal_forces = {side: [] for side in SIDES}
    friction_use = {side: [] for side in SIDES}
    for local, frame in enumerate(frames):
        wrench = grasp_matrices[local] @ values[local].reshape(-1)
        force_errors.append(np.linalg.norm(wrench[:3] - required_force[local]))
        torque_errors.append(np.linalg.norm(wrench[3:] - required_torque[local]))
        for index, side in enumerate(SIDES):
            inward = -(rotations[frame] @ anchors[side].normal)
            inward /= np.linalg.norm(inward) + 1e-12
            normal = float(inward @ values[local, index])
            tangent = values[local, index] - normal * inward
            normal_forces[side].append(normal)
            friction_use[side].append(np.linalg.norm(tangent) / (friction * normal + 1e-12))
    packed = np.zeros((len(poses), 2, 3), dtype=np.float64)
    packed[frames] = values
    force_errors = np.asarray(force_errors)
    torque_errors = np.asarray(torque_errors)
    report = {
        "solver_status": str(problem.status),
        "design_mass_kg": mass,
        "friction_coefficient": friction,
        "normal_force_bounds_n": [min_normal, max_normal],
        "force_residual_mean_n": float(np.mean(force_errors)),
        "force_residual_p95_n": float(np.quantile(force_errors, 0.95)),
        "force_residual_max_n": float(np.max(force_errors)),
        "torque_residual_mean_nm": float(np.mean(torque_errors)),
        "torque_residual_p95_nm": float(np.quantile(torque_errors, 0.95)),
        "torque_residual_max_nm": float(np.max(torque_errors)),
        "normal_force_mean_n": {side: float(np.mean(normal_forces[side])) for side in SIDES},
        "normal_force_min_n": {side: float(np.min(normal_forces[side])) for side in SIDES},
        "friction_utilization_p95": {side: float(np.quantile(friction_use[side], 0.95)) for side in SIDES},
        "fraction_force_residual_le_1n": float(np.mean(force_errors <= 1.0)),
        "fraction_torque_residual_le_0p1nm": float(np.mean(torque_errors <= 0.1)),
    }
    return report, packed


def _refine_anchors(
    poses: np.ndarray,
    fps: float,
    mesh: trimesh.Trimesh,
    initial: dict[str, Anchor],
    start: int,
    end: int,
    mass: float,
    friction: float,
    min_normal: float,
    max_normal: float,
    iterations: int,
) -> tuple[dict[str, Anchor], dict[str, object], np.ndarray]:
    normals = _outward_face_normals(mesh)
    face_centers = np.asarray(mesh.triangles_center, dtype=np.float64)
    center = np.asarray(mesh.bounding_box.centroid, dtype=np.float64)
    direction = initial["left"].normal - initial["right"].normal
    direction /= np.linalg.norm(direction) + 1e-12
    seed = np.array([0.0, 0.0, 1.0])
    if abs(float(seed @ direction)) > 0.9:
        seed = np.array([1.0, 0.0, 0.0])
    tangent0 = np.cross(direction, seed)
    tangent0 /= np.linalg.norm(tangent0) + 1e-12
    tangent1 = np.cross(direction, tangent0)
    tangent1 /= np.linalg.norm(tangent1) + 1e-12
    tangent = np.stack((tangent0, tangent1), axis=1)
    offsets = {}
    sources = {}
    bounds = {}
    triangles = {}
    for side in SIDES:
        outward = direction if side == "left" else -direction
        ids = np.flatnonzero(normals @ outward >= 0.95)
        if not len(ids):
            raise RuntimeError(f"no planar patch for {side}")
        tri = np.asarray(mesh.triangles[ids], dtype=np.float64)
        coords = tri.reshape(-1, 3) @ tangent
        triangles[side] = tri
        offsets[side] = float(direction @ initial[side].point)
        sources[side] = initial[side].point @ tangent
        bounds[side] = (coords.min(axis=0), coords.max(axis=0))
    coordinates = np.concatenate((sources["left"], sources["right"]))
    lower = np.concatenate((bounds["left"][0], bounds["right"][0]))
    upper = np.concatenate((bounds["left"][1], bounds["right"][1]))
    coordinates = np.clip(coordinates, lower, upper)
    rotations, _, omega, alpha = _object_kinematics(poses, fps)
    inertia_body = np.eye(3) * (0.002 * mass / 0.1)
    active_frames = np.arange(start, end + 1, dtype=np.int64)
    current = dict(initial)
    history = []
    packed = np.zeros((len(poses), 2, 3))
    for iteration in range(iterations):
        audit, packed = _solve_wrench(poses, fps, current, start, end, mass, friction, min_normal, max_normal)

        def residual(values: np.ndarray) -> np.ndarray:
            points = {
                "left": tangent @ values[:2] + offsets["left"] * direction,
                "right": tangent @ values[2:] + offsets["right"] * direction,
            }
            terms = []
            for frame in active_frames[::2]:
                rotation = rotations[frame]
                inertia_world = rotation @ inertia_body @ rotation.T
                required = inertia_world @ alpha[frame] + np.cross(omega[frame], inertia_world @ omega[frame])
                torque = np.zeros(3)
                for index, side in enumerate(SIDES):
                    torque += np.cross(rotation @ points[side], packed[frame, index])
                terms.extend((torque - required) / 0.1)
            terms.extend((values - np.concatenate((sources["left"], sources["right"]))) / 0.20)
            return np.asarray(terms)

        solved = least_squares(
            residual,
            coordinates,
            bounds=(lower, upper),
            max_nfev=120,
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
        )
        coordinates = solved.x
        updated = {}
        for index, side in enumerate(SIDES):
            coord = coordinates[2 * index : 2 * index + 2]
            plane_point = tangent @ coord + offsets[side] * direction
            closest = trimesh.triangles.closest_point(
                triangles[side], np.repeat(plane_point[None], len(triangles[side]), axis=0)
            )
            point = closest[int(np.argmin(np.linalg.norm(closest - plane_point, axis=1)))]
            updated[side] = Anchor(side, np.asarray(point), current[side].normal, current[side].source_face)
        current = updated
        coordinates = np.concatenate(tuple(current[side].point @ tangent for side in SIDES))
        history.append(
            {
                "iteration": iteration,
                "pre_update_wrench_audit": audit,
                "anchor_solver_cost": float(solved.cost),
                "anchor_solver_nfev": int(solved.nfev),
                "anchor_points_local_m": {side: current[side].point.tolist() for side in SIDES},
            }
        )
    final_audit, packed = _solve_wrench(poses, fps, current, start, end, mass, friction, min_normal, max_normal)
    return (
        current,
        {
            "iterations": history,
            "final_wrench_audit": final_audit,
            "explicit_com_or_axis_objective": False,
            "objective": "contact torque residual with fixed friction-cone forces",
            "mesh_center_used_only_for_outward_normal_orientation": center.tolist(),
        },
        packed,
    )


def _smooth_activation(
    frames: int,
    start: int,
    ramp: int,
    end: int | None = None,
) -> np.ndarray:
    gamma = np.zeros(frames, dtype=np.float64)
    end = frames - 1 if end is None else int(end)
    if not 0 <= start <= end < frames:
        raise ValueError("activation interval is outside the sequence")
    if ramp <= 0:
        gamma[start : end + 1] = 1.0
        return gamma
    gamma[min(end + 1, start + ramp) : end + 1] = 1.0
    for frame in range(start, min(end + 1, start + ramp)):
        # Contact begins at start; do not pull the hands into the object
        # before the source trajectory has actually established its grasp.
        x = (frame - start + 1) / ramp
        gamma[frame] = x * x * (3.0 - 2.0 * x)
    return gamma


def build(args: argparse.Namespace) -> dict[str, object]:
    source = args.official_input.resolve()
    plan_path = args.semantic_plan.resolve()
    object_mesh_path = args.object_mesh.resolve()
    interval_path = args.interval_plan.resolve()
    robot_bodies, palm_points, palm_directions = _hand_geometry(args.hand_model)
    hand_mesh_paths = (
        {"left": SPHERE_HAND_MESH.resolve(), "right": SPHERE_HAND_MESH.resolve()}
        if args.hand_model == "sphere"
        else {
            "left": args.left_hand_mesh.resolve(),
            "right": args.right_hand_mesh.resolve(),
        }
    )
    output = args.output.resolve()
    if source.parent.name != "official_inputs":
        raise ValueError("official input must come from omomo/official_inputs; retargeted qpos inputs are not accepted")
    if output.exists():
        raise FileExistsError(output)
    human, poses = load_intermimic_data(str(source))
    events = load_semantic_plan_projection(plan_path)
    declared = {part for event in events for part in event.body_parts}
    if not set(PARTS).issubset(declared):
        raise ValueError("semantic plan must declare both hands")
    interval = json.loads(interval_path.read_text(encoding="utf-8"))
    if interval.get("schema") != "holosoma.force_affordance_interval.v1":
        raise ValueError("interval plan has the wrong schema")
    if interval.get("policy", {}).get("manual_frame_overrides") is not False:
        raise ValueError("interval plan must prove that manual frame overrides are disabled")
    interval_inputs = interval.get("inputs", {})
    expected_interval_hashes = {
        "official_smpl_sha256": _sha256(source),
        "semantic_plan_sha256": _sha256(plan_path),
        "object_mesh_sha256": _sha256(object_mesh_path),
    }
    mismatches = {
        key: (interval_inputs.get(key), value)
        for key, value in expected_interval_hashes.items()
        if interval_inputs.get(key) != value
    }
    if mismatches:
        raise ValueError(f"interval-plan source mismatch: {mismatches}")
    grip_start = int(interval["start_frame"])
    end_frame = int(interval["end_frame"])
    anchor_frame = int(interval["anchor_frame"])
    if int(interval.get("frames", -1)) != len(human):
        raise ValueError("interval-plan frame count differs from official input")
    if not 0 <= grip_start <= anchor_frame <= end_frame < len(human):
        raise ValueError("invalid derived force-affordance interval")
    mesh = trimesh.load(object_mesh_path, force="mesh", process=False)
    hand_meshes = {side: trimesh.load(path, force="mesh", process=False) for side, path in hand_mesh_paths.items()}
    if not isinstance(mesh, trimesh.Trimesh) or any(
        not isinstance(hand_meshes[side], trimesh.Trimesh) for side in SIDES
    ):
        raise ValueError("object and hand assets must be triangle meshes")
    palm_points = {side: np.asarray(point, dtype=np.float64).copy() for side, point in palm_points.items()}
    support_audit: dict[str, dict[str, object]] = {}
    for side in SIDES:
        if args.hand_model == "original" and args.hand_point_policy == "directional_support":
            palm_points[side], support_audit[side] = _directional_support_point(
                hand_meshes[side],
                palm_directions[side],
                args.support_band_mm / 1000.0,
            )
        else:
            vertices = np.asarray(hand_meshes[side].vertices, dtype=np.float64)
            support_audit[side] = {
                "directional_support_overshoot_m": float(np.max((vertices - palm_points[side]) @ palm_directions[side]))
            }
    rotations = Rotation.from_quat(poses[:, :4][:, [1, 2, 3, 0]]).as_matrix()
    wrist_local = {}
    source_distances = np.empty((len(human), 2), dtype=np.float64)
    for index, side in enumerate(SIDES):
        wrist_id = SMPLH_DEMO_JOINTS.index(WRIST_NAMES[side])
        wrist_local[side] = np.einsum("tji,tj->ti", rotations, human[:, wrist_id] - poses[:, 4:7])
        _, distances, _ = trimesh.proximity.closest_point_naive(mesh, wrist_local[side])
        source_distances[:, index] = distances
    initial = _select_source_anchors(mesh, wrist_local, anchor_frame)
    anchors, refinement, forces = _refine_anchors(
        poses,
        args.fps,
        mesh,
        initial,
        grip_start,
        end_frame,
        args.design_mass_kg,
        args.friction,
        args.minimum_normal_force_n,
        args.maximum_normal_force_n,
        args.refinement_iterations,
    )
    activation_1d = _smooth_activation(
        len(human),
        grip_start,
        args.ramp_frames,
        end_frame,
    )
    activation = np.repeat(activation_1d[:, None], 2, axis=1)
    active = activation > 0.0
    object_points = np.empty((len(human), 2, 3), dtype=np.float64)
    object_normals = np.empty_like(object_points)
    robot_points = np.empty((2, 3), dtype=np.float64)
    robot_normals = np.empty((2, 3), dtype=np.float64)
    robot_geometry = {}
    for index, side in enumerate(SIDES):
        anchor = anchors[side]
        object_points[:, index] = anchor.point - args.preload_mm / 1000.0 * anchor.normal
        object_normals[:, index] = anchor.normal
        direction = palm_directions[side]
        point = palm_points[side]
        hand_mesh = hand_meshes[side]
        closest, distance, triangle = trimesh.proximity.closest_point_naive(
            hand_mesh,
            point[None],
        )
        if float(distance[0]) > 0.002:
            raise ValueError(
                f"{side} palm point is not on the original hand mesh: {1000.0 * float(distance[0]):.3f} mm"
            )
        robot_points[index] = point
        robot_normals[index] = direction
        robot_geometry[f"{side}_hand"] = {
            "body": robot_bodies[f"{side}_hand"],
            "point_policy": (
                "fixed validated half-sphere surface support point"
                if args.hand_model == "sphere"
                else "directional support of the physical rubber-hand mesh"
                if args.hand_point_policy == "directional_support"
                else "fixed existing original-G1 palm proxy"
            ),
            "point_local_m": robot_points[index].tolist(),
            "normal_local": robot_normals[index].tolist(),
            "physical_mesh_surface_offset_m": float(distance[0]),
            "closest_surface_point_local_m": closest[0].tolist(),
            "closest_surface_triangle": int(triangle[0]),
            **support_audit[side],
        }
    metadata = {
        "schema": SCHEMA,
        "method": "source-only force-affordance design for Semantic B4",
        "frames": len(human),
        "fps": args.fps,
        "activation_mode": "force_affordance_interval",
        "hand_model": args.hand_model,
        "hand_point_policy": args.hand_point_policy,
        "semantic_parts_with_contact": list(PARTS),
        "force_affordance_robot_bodies": robot_bodies,
        "robot_geometry": robot_geometry,
        "grip_start": grip_start,
        "end_frame": end_frame,
        "anchor_frame": anchor_frame,
        "interval_derivation": interval,
        "activation_ramp": {
            "start_frame": grip_start,
            "full_weight_frame": min(len(human) - 1, grip_start + args.ramp_frames - 1),
            "precontact_activation": False,
            "ramp_frames": args.ramp_frames,
        },
        "preload_mm": args.preload_mm,
        "anchor_policy": "official-SMPL wrist-adjacent opposing faces, then source-object wrench refinement",
        "runtime_policy": "fixed object-local anchors; analytic point/normal Jacobians; no runtime surface search",
        "budget_policy": "inherit Semantic B4 exactly: ordinary=2, exact semantic trigger=4",
        "no_retargeted_robot_trajectory_input": True,
        "official_input_policy": "parent directory must be omomo/official_inputs",
        "initial_anchors": {
            side: {
                "point_local_m": initial[side].point.tolist(),
                "normal_local": initial[side].normal.tolist(),
                "source_face": initial[side].source_face,
            }
            for side in SIDES
        },
        "final_anchors": {
            side: {
                "surface_point_local_m": anchors[side].point.tolist(),
                "preloaded_target_local_m": (
                    anchors[side].point - args.preload_mm / 1000.0 * anchors[side].normal
                ).tolist(),
                "normal_local": anchors[side].normal.tolist(),
            }
            for side in SIDES
        },
        "wrench_refinement": refinement,
        "contact_episodes": [
            {
                "episode_index": 0,
                "start_frame": grip_start,
                "end_frame": end_frame,
                "t_c": grip_start,
                "contact_parts": list(PARTS),
                "contact_geometry_available": list(PARTS),
            }
        ],
        "inputs": {
            "official_smpl": str(source),
            "official_smpl_sha256": _sha256(source),
            "semantic_plan": str(plan_path),
            "semantic_plan_sha256": _sha256(plan_path),
            "object_mesh": str(object_mesh_path),
            "object_mesh_sha256": _sha256(object_mesh_path),
            "interval_plan": str(interval_path),
            "interval_plan_sha256": _sha256(interval_path),
            "hand_meshes": {
                side: {
                    "path": str(hand_mesh_paths[side]),
                    "sha256": _sha256(hand_mesh_paths[side]),
                }
                for side in SIDES
            },
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        semantic_parts=np.asarray(PARTS),
        active=active,
        activation=activation.astype(np.float32),
        object_points_local=object_points.astype(np.float32),
        weights=np.ones(2, dtype=np.float32),
        source_distance_m=source_distances.astype(np.float32),
        robot_points_local=robot_points.astype(np.float32),
        robot_normals_local=robot_normals.astype(np.float32),
        object_normals_local=object_normals.astype(np.float32),
        contact_force_world=forces.astype(np.float32),
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
    _write_json(output.with_suffix(".json"), metadata)
    load_retarget_contact_targets(output, expected_frames=len(human))
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-input", type=Path, required=True)
    parser.add_argument("--semantic-plan", type=Path, required=True)
    parser.add_argument("--object-mesh", type=Path, required=True)
    parser.add_argument("--interval-plan", type=Path, required=True)
    parser.add_argument("--hand-model", choices=("original", "sphere"), default="original")
    parser.add_argument(
        "--hand-point-policy",
        choices=("legacy_proxy", "directional_support"),
        default="legacy_proxy",
    )
    parser.add_argument(
        "--support-band-mm",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "--left-hand-mesh",
        type=Path,
        default=DEFAULT_HAND_MESHES["left"],
    )
    parser.add_argument(
        "--right-hand-mesh",
        type=Path,
        default=DEFAULT_HAND_MESHES["right"],
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--ramp-frames", type=int, default=5)
    parser.add_argument("--preload-mm", type=float, default=0.3)
    parser.add_argument("--design-mass-kg", type=float, default=2.0)
    parser.add_argument("--friction", type=float, default=0.9)
    parser.add_argument("--minimum-normal-force-n", type=float, default=15.0)
    parser.add_argument("--maximum-normal-force-n", type=float, default=40.0)
    parser.add_argument("--refinement-iterations", type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(build(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
