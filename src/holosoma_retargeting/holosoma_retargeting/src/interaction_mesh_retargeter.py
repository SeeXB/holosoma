from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import cvxpy as cp  # type: ignore[import-not-found]
import mujoco  # type: ignore[import-not-found]
import numpy as np
import trimesh
from scipy import sparse as sp  # type: ignore[import-untyped]
from scipy.spatial.transform import Rotation  # type: ignore[import-untyped]
from tqdm import tqdm

try:
    import viser  # type: ignore[import-not-found]
    import yourdfpy  # type: ignore[import-untyped]
    from viser.extras import ViserUrdf  # type: ignore[import-not-found]
except ImportError:  # Visualization is optional for headless benchmark runs.
    viser = None  # type: ignore[assignment]
    yourdfpy = None  # type: ignore[assignment]
    ViserUrdf = None  # type: ignore[assignment,misc]

from holosoma_retargeting.config_types.retargeter import FootLockConfig, SelfCollisionConfig
from holosoma_retargeting.config_types.semantic import SemanticRetargetingConfig
from holosoma_retargeting.semantic_keyframes.profiling import write_rows_csv, write_run_artifacts
from holosoma_retargeting.semantic_keyframes.runtime import (
    BodyVertexMapping,
    SemanticEvent,
    SemanticTimeline,
    SemanticWeightResult,
    build_always_body_vertex_weight_result,
    build_semantic_vertex_weight_result,
    extract_semantic_cross_entity_edges,
    load_semantic_plan_projection,
    make_budget_plan,
    resolve_body_vertex_mapping,
    scale_semantic_weight_result,
    spatiotemporal_semantic_context,
)

# Add src to path for direct execution
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Import with type ignore for mypy compatibility
from mujoco_utils import (  # type: ignore[import-not-found,no-redef]  # noqa: E402
    _world_mesh_from_geom,
)
from utils import (  # type: ignore[import-not-found,no-redef]  # noqa: E402
    calculate_laplacian_coordinates,
    calculate_laplacian_matrix,
    create_interaction_mesh,
    get_adjacency_list,
    transform_points_local_to_world,
    transform_points_world_to_local,
)


OMNI_LAPLACIAN_WEIGHT = 10.0


class InteractionMeshRetargeter:
    """
    A class to perform kinematic retargeting from human motion to a robot,
    preserving spatial relationships using an interaction mesh.
    """

    def __init__(
        self,
        task_constants: ModuleType,
        object_urdf_path: str,
        q_a_init_idx: int = -7,
        activate_foot_sticking: bool = True,
        activate_obj_non_penetration: bool = True,
        activate_joint_limits: bool = True,
        step_size: float = 0.2,
        collision_detection_threshold: float = 0.1,
        penetration_tolerance: float = 1e-3,
        foot_sticking_tolerance: float = 1e-3,
        foot_lock: FootLockConfig | None = None,
        self_collision: SelfCollisionConfig | None = None,
        visualize: bool = False,
        debug: bool = False,
        w_nominal_tracking_init: float = 5.0,
        nominal_tracking_tau: float = 10.0,
        semantic_config: SemanticRetargetingConfig | None = None,
    ):
        """This kinematic retargeter solves the diffIK problem with hard constraints in SQP style.
        During each SQP iteration, the problem is solved with the following constraints and costs:
            1. [Cost] Minimize the Laplacian deformation in the object frame.
            2. [Constraint] Enforce the non-penetration constraints w/ the ground and (if activated) the object.
            3. [Constraint] Enforce the foot sticking constraints if activated.
            4. [Constraint] Enforce the joint limits if activated.
            5. [Constraint] Enforce trust region of dq.
        The constraints are linearized and the costs are quadratic with a trust region.

        Args:
            q_a_init_idx: the index in robot's configuration where the optimization variables start. -7: starts from the
            floating base, -3: starts from the translation of the floating base, 0: starts from the actuated DOF,
            12: starts from waist, 15: starts from left shoulder
            step_size: trust region for each SQP iteration.
            collision_detection_threshold: only start to detect collision
            when the distance is smaller than this threshold.
            penetration_tolerance: tolerance for penetration when enforcing non-penetration constraints.
            foot_sticking_tolerance: tolerance for foot sticking constraints in x, y.
            foot_lock: configuration for explicit frame-range based foot locking constraints.
            nominal_tracking_tau: the time constant for the nominal tracking cost.
        """

        self.robot_model_path = task_constants.ROBOT_URDF_FILE
        self.object_model_path = object_urdf_path
        self.object_name = task_constants.OBJECT_NAME
        self.collision_detection_threshold = collision_detection_threshold
        self.activate_foot_sticking = activate_foot_sticking
        self.activate_obj_non_penetration = activate_obj_non_penetration
        self.activate_joint_limits = activate_joint_limits
        self.foot_links = dict(zip(task_constants.FOOT_STICKING_LINKS, task_constants.FOOT_STICKING_LINKS))
        self.penetration_tolerance = penetration_tolerance
        self.step_size = step_size
        self.visualize = visualize
        self.debug = debug
        self.demo_joints = task_constants.DEMO_JOINTS
        self.laplacian_match_links = task_constants.JOINTS_MAPPING
        self.task_constants = task_constants

        self.smplh_mapped_joint_indices = [self.demo_joints.index(name) for name in self.laplacian_match_links]

        # Setup weights and parameters
        self.laplacian_weights = OMNI_LAPLACIAN_WEIGHT
        self.smooth_weight = 0.2
        # Tolerance for foot sticking constraints in x, y.
        self.foot_sticking_tolerance = foot_sticking_tolerance
        self._init_foot_lock(foot_lock)
        self._self_collision_config = self_collision

        # Setup visualization if requested
        if self.visualize:
            self._setup_visualization()

        # Load Mujoco model
        explicit_scene = getattr(self.task_constants, "SCENE_XML_FILE", None)
        if explicit_scene:
            robot_xml_path = explicit_scene
        elif self.object_name == "ground":
            robot_xml_path = self.robot_model_path.replace(".urdf", ".xml")
        elif self.object_name == "multi_boxes":
            robot_xml_path = self.task_constants.SCENE_XML_FILE
        else:
            robot_xml_path = self.robot_model_path.replace(".urdf", "_w_" + self.object_name + ".xml")

        self.robot_model = mujoco.MjModel.from_xml_path(robot_xml_path)
        print("Loading robot model from: ", robot_xml_path)

        self.robot_data = mujoco.MjData(self.robot_model)
        self._init_self_collision(self._self_collision_config)

        if self.robot_data.qpos.shape[0] > 7 + self.task_constants.ROBOT_DOF:
            self.has_dynamic_object = True
        else:
            self.has_dynamic_object = False
        self.nq = self.robot_model.nq

        self.q_a_init_idx = q_a_init_idx
        self.q_a_indices = np.arange(7 + self.q_a_init_idx, 7 + self.task_constants.ROBOT_DOF)

        self.nq_a = len(self.q_a_indices)

        # Create complete limits with floating base (-inf, inf) and actuated joint limits
        n_floating_base = 7
        joint_names = [self.robot_model.joint(i).name for i in range(self.robot_model.njnt)]
        actuated_joints = [(i, name) for i, name in enumerate(joint_names) if name]  # Filter out None names

        large_number = 1e6
        complete_lower_limits = np.concatenate(
            [-large_number * np.ones(n_floating_base), self.robot_model.jnt_range[[i for i, _ in actuated_joints], 0]]
        )
        complete_upper_limits = np.concatenate(
            [large_number * np.ones(n_floating_base), self.robot_model.jnt_range[[i for i, _ in actuated_joints], 1]]
        )

        self.q_a_lb = complete_lower_limits[self.q_a_indices]
        self.q_a_ub = complete_upper_limits[self.q_a_indices]

        self.q_a_lb[np.array(list(self.task_constants.MANUAL_LB.keys())).astype(int)] = list(
            self.task_constants.MANUAL_LB.values()
        )
        self.q_a_ub[np.array(list(self.task_constants.MANUAL_UB.keys())).astype(int)] = list(
            self.task_constants.MANUAL_UB.values()
        )

        # Prevent too much waist twist
        self.Q_diag = np.zeros(self.nq_a) * 1e-3
        self.Q_diag[np.array(list(self.task_constants.MANUAL_COST.keys())).astype(int)] = list(
            self.task_constants.MANUAL_COST.values()
        )

        self.w_nominal_tracking_init = w_nominal_tracking_init
        self.nominal_tracking_tau = nominal_tracking_tau
        self.track_nominal_indices = task_constants.NOMINAL_TRACKING_INDICES
        self._semantic_config_explicit = semantic_config is not None
        self.semantic_config = semantic_config or SemanticRetargetingConfig()
        self.semantic_config.validate()
        self._convex_solver_calls = 0

    def _init_foot_lock(self, foot_lock: FootLockConfig | None) -> None:
        """Initialize foot lock configuration and normalize window mappings."""
        self.foot_lock = foot_lock or FootLockConfig()
        self._foot_lock_windows: dict[str, tuple[tuple[int, int], ...]] = {"left": (), "right": ()}
        self._foot_lock_z_floors: dict[str, tuple[float, ...]] = {"left": (), "right": ()}
        if self.foot_lock.windows is None:
            return
        for key, windows in self.foot_lock.windows.items():
            key_lower = key.lower()
            side = None
            if key_lower.startswith("l") or ("left" in key_lower):
                side = "left"
            elif key_lower.startswith("r") or ("right" in key_lower):
                side = "right"
            if side is None:
                continue

            normalized_windows: list[tuple[int, int]] = []
            z_floors: list[float] = []
            for window in windows:
                if len(window) == 3:
                    start, end, z = int(window[0]), int(window[1]), float(window[2])
                elif len(window) == 2:
                    start, end = int(window[0]), int(window[1])
                    z = self.foot_lock.z_floor
                else:
                    raise ValueError(f"Invalid foot lock window for {key}: {window}")
                if end < start:
                    raise ValueError(f"Invalid foot lock window with end < start for {key}: {window}")
                normalized_windows.append((start, end))
                z_floors.append(z)
            self._foot_lock_windows[side] = tuple(normalized_windows)
            self._foot_lock_z_floors[side] = tuple(z_floors)

    def _init_self_collision(self, self_collision: SelfCollisionConfig | None) -> None:
        """Initialize self-collision configuration and precompute geom pairs."""
        sc = self_collision or SelfCollisionConfig()
        self._self_collision_enabled = sc.enable and len(sc.pairs) > 0
        self._self_collision_tolerance = sc.tolerance
        self._self_collision_windows: list[tuple[int, int]] | None = sc.windows
        self._self_collision_geom_pairs: list[tuple[int, int]] = []

        self._sc_last_vis_frame = -1

        if not self._self_collision_enabled:
            return

        m = self.robot_model

        # Build body_name → [geom_ids] mapping (only geoms with collision enabled)
        body_to_geoms: dict[str, list[int]] = {}
        for g in range(m.ngeom):
            if m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0:
                continue
            body_id = m.geom_bodyid[g]
            body_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
            body_to_geoms.setdefault(body_name, []).append(g)

        # Build geom pairs from body name pairs
        for body_a, body_b in sc.pairs:
            geoms_a = body_to_geoms.get(body_a, [])
            geoms_b = body_to_geoms.get(body_b, [])
            if not geoms_a:
                print(f"[SelfCollision] Warning: no collision geoms found for body '{body_a}'")
            if not geoms_b:
                print(f"[SelfCollision] Warning: no collision geoms found for body '{body_b}'")
            for ga in geoms_a:
                for gb in geoms_b:
                    self._self_collision_geom_pairs.append((ga, gb))

        print(
            f"[SelfCollision] Initialized with {len(self._self_collision_geom_pairs)} geom pairs "
            f"from {len(sc.pairs)} body pairs, tolerance={sc.tolerance}m"
        )

    def _setup_visualization(self):
        """Setup Viser visualization components."""
        if viser is None or yourdfpy is None or ViserUrdf is None:
            raise RuntimeError("visualize=True requires the optional viser and yourdfpy packages")
        self.server = viser.ViserServer()

        # 1) Ensure a world frame exists (absolute path!)
        try:
            self.server.scene.add_frame("/world", show_axes=False)
        except Exception:
            print("Starting viser")

        # Create parent frames for robot and object
        self.robot_base = self.server.scene.add_frame("/world/robot", show_axes=False)

        print("robot_model_path: ", self.robot_model_path)

        # Load robot URDF
        self.robot_urdf = yourdfpy.URDF.load(
            self.robot_model_path,
            load_meshes=True,
            build_scene_graph=True,
        )

        print("Viser using robot URDF: ", self.robot_model_path)

        # Create ViserUrdf instance for robot, attaching it to the robot_base frame
        self.viser_robot = ViserUrdf(
            self.server,
            urdf_or_path=self.robot_urdf,
            root_node_name="/world/robot",  # This links to the robot_base frame we created
        )

        # Similarly for object
        if self.object_model_path:
            self.object_base = self.server.scene.add_frame("/world/object", show_axes=False)

            self.object_urdf = yourdfpy.URDF.load(
                self.object_model_path,
                load_meshes=True,
                build_scene_graph=True,
            )

            # Create ViserUrdf instance for object, attaching it to the object_base frame
            self.viser_object = ViserUrdf(
                self.server,
                urdf_or_path=self.object_urdf,
                root_node_name="/world/object",  # This links to the object_base frame we created
            )
            print("Viser using object URDF: ", self.object_model_path)

        else:
            self.viser_object = None

        # Check the number of actuated joints and their names
        robot_joint_limits = self.viser_robot.get_actuated_joint_limits()
        print("\nRobot joints:")
        print("Number of actuated joints:", len(robot_joint_limits))
        print("Joint names:", list(robot_joint_limits.keys()))

        # Initialize robot with this configuration
        robot_initial_config = np.zeros(len(robot_joint_limits))
        self.viser_robot.update_cfg(robot_initial_config)

        # Add grid
        self.server.scene.add_grid(
            "/world/grid",
            width=8,
            height=8,
            position=(0.0, 0.0, 0.0),
        )

    def draw_mesh_from_geom(self, model, data, geom_id, geom_name, name="/mesh", color=(50, 150, 255), opacity=0.5):
        """
        Draw a single MuJoCo mesh geom (already baked to world coords) in viser.
        color is [0, 255] RGB ints; opacity is [0,1].
        """
        if not hasattr(self, "server"):
            return
        V, F = _world_mesh_from_geom(model, data, geom_id, geom_name)
        self.server.scene.add_mesh_simple(
            name,
            vertices=V.astype(np.float32),
            faces=F.astype(np.int32),
            position=(0.0, 0.0, 0.0),  # already world-frame
            color=tuple(int(c) for c in color),
            opacity=float(opacity),
        )

    def draw_mesh_pair_with_contact(
        self,
        model,
        data,
        geom_id1,
        geom_id2,
        geom1_name,
        geom2_name,
        fromto=None,
        group_name="pair",
        color1=(50, 150, 255),
        color2=(255, 120, 60),
        opacity=0.45,
        show_segment=True,
    ):
        """
        Draw two meshes and (optionally) a contact/query segment.
        Uses the existing self.draw_keypoints(...) to visualize points.
        """
        # Note: sometime geom does not have mesh, mesh_id will be -1
        if int(model.geom_dataid[geom_id1]) == -1 or int(model.geom_dataid[geom_id2]) == -1:
            return

        base = f"/{group_name}"
        # meshes
        self.draw_mesh_from_geom(model, data, geom_id1, geom1_name, name=f"{base}/mesh1", color=color1, opacity=opacity)
        self.draw_mesh_from_geom(model, data, geom_id2, geom2_name, name=f"{base}/mesh2", color=color2, opacity=opacity)

        # contact points (q: green, c: red) via your draw_keypoints
        if fromto is not None:
            q = np.asarray(fromto[:3], dtype=float)
            c = np.asarray(fromto[3:], dtype=float)

            # your existing helper (rgba expects floats 0..1)
            self.draw_keypoints(q, name=f"{group_name}_q", rgba=(0.0, 1.0, 0.0, 1.0))
            self.draw_keypoints(c, name=f"{group_name}_c", rgba=(1.0, 0.0, 0.0, 1.0))

    def _geom_and_body_name(self, geom_id: int) -> tuple[str, str]:
        geom_name = mujoco.mj_id2name(self.robot_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        body_id = int(self.robot_model.geom_bodyid[geom_id])
        body_name = mujoco.mj_id2name(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        return geom_name, body_name

    def _penetration_pair_type(self, geom_a: int, geom_b: int) -> str:
        name_a, body_a = self._geom_and_body_name(geom_a)
        name_b, body_b = self._geom_and_body_name(geom_b)
        label_a = f"{name_a} {body_a}".lower()
        label_b = f"{name_b} {body_b}".lower()
        a_ground, b_ground = "ground" in label_a, "ground" in label_b
        a_object, b_object = self.object_name.lower() in label_a, self.object_name.lower() in label_b
        a_foot = "foot" in label_a or "ankle" in label_a
        b_foot = "foot" in label_b or "ankle" in label_b
        a_hand, b_hand = "hand" in label_a, "hand" in label_b
        if (a_ground and b_object) or (b_ground and a_object):
            return "object-ground"
        if (a_ground and b_foot) or (b_ground and a_foot):
            return "foot-ground"
        if (a_object and b_hand) or (b_object and a_hand):
            return "hand-box"
        if a_object or b_object:
            return "other-body-box"
        if a_ground or b_ground:
            return "robot-ground-excluding-foot"
        return "self-collision"

    def _hand_box_signed_distances(self) -> dict[str, float | None]:
        m, d = self.robot_model, self.robot_data
        object_geoms: list[int] = []
        hand_geoms: dict[str, list[int]] = {"left": [], "right": []}
        for geom_id in range(m.ngeom):
            geom_name, body_name = self._geom_and_body_name(geom_id)
            label = f"{geom_name} {body_name}".lower()
            if self.object_name.lower() in label:
                object_geoms.append(geom_id)
            if "hand" in label:
                if "left" in label:
                    hand_geoms["left"].append(geom_id)
                if "right" in label:
                    hand_geoms["right"].append(geom_id)
        result: dict[str, float | None] = {}
        fromto = np.zeros(6, dtype=float)
        for side in ("left", "right"):
            distances: list[float] = []
            for hand_geom in hand_geoms[side]:
                for object_geom in object_geoms:
                    fromto[:] = 0.0
                    distances.append(float(mujoco.mj_geomDistance(m, d, hand_geom, object_geom, 10.0, fromto)))
            result[f"{side}_hand_box_signed_distance"] = min(distances) if distances else None
        return result

    def _penetration_audit(
        self,
        q: np.ndarray,
        frame_idx: int,
    ) -> tuple[dict[str, float | int | None], list[dict[str, Any]]]:
        """Audit actual signed geometry distances and separate expected/illegal pairs."""
        m, d = self.robot_model, self.robot_data
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        candidates = self._prefilter_pairs_with_mj_collision(float(self.collision_detection_threshold))
        fromto = np.zeros(6, dtype=float)
        rows: list[dict[str, Any]] = []
        configured_self_pairs = {
            (min(first, second), max(first, second))
            for first, second in self._self_collision_geom_pairs
        }
        for geom_a, geom_b in candidates:
            if m.geom_contype[geom_a] == 0 and m.geom_conaffinity[geom_a] == 0:
                continue
            if m.geom_contype[geom_b] == 0 and m.geom_conaffinity[geom_b] == 0:
                continue
            fromto[:] = 0.0
            signed_distance = float(
                mujoco.mj_geomDistance(
                    m,
                    d,
                    geom_a,
                    geom_b,
                    float(self.collision_detection_threshold),
                    fromto,
                )
            )
            if signed_distance >= 0.0:
                continue
            geom_name_a, body_name_a = self._geom_and_body_name(geom_a)
            geom_name_b, body_name_b = self._geom_and_body_name(geom_b)
            pair_type = self._penetration_pair_type(geom_a, geom_b)
            expected = pair_type in {"foot-ground", "hand-box", "object-ground"}
            normalized_pair = (min(geom_a, geom_b), max(geom_a, geom_b))
            evaluated_self_collision = pair_type == "self-collision" and normalized_pair in configured_self_pairs
            illegal = pair_type in {"other-body-box", "robot-ground-excluding-foot"} or evaluated_self_collision
            rows.append(
                {
                    "frame": frame_idx,
                    "geom_a": geom_name_a,
                    "geom_b": geom_name_b,
                    "body_a": body_name_a,
                    "body_b": body_name_b,
                    "signed_distance": signed_distance,
                    "penetration_depth": -signed_distance,
                    "pair_type": pair_type,
                    "expected_contact": expected,
                    "illegal_penetration": illegal,
                }
            )

        raw_depths = [float(row["penetration_depth"]) for row in rows]
        expected_depths = [
            float(row["penetration_depth"]) for row in rows if bool(row["expected_contact"])
        ]
        illegal_depths = [
            float(row["penetration_depth"]) for row in rows if bool(row["illegal_penetration"])
        ]
        environment_depths = [
            float(row["penetration_depth"])
            for row in rows
            if row["pair_type"] not in {"object-ground", "self-collision"}
        ]
        metrics: dict[str, float | int | None] = {
            "penetration_depth": max(environment_depths, default=0.0),
            "raw_penetration_depth": max(raw_depths, default=0.0),
            "raw_penetration_pair_count": len(rows),
            "expected_contact_penetration_depth": max(expected_depths, default=0.0),
            "expected_contact_penetration_pair_count": len(expected_depths),
            "illegal_penetration_depth": max(illegal_depths, default=0.0),
            "illegal_penetration_pair_count": len(illegal_depths),
            **self._hand_box_signed_distances(),
        }
        return metrics, rows

    def _measure_environment_penetration(self, q: np.ndarray) -> float:
        """Backward-compatible maximum actual environment penetration depth."""
        metrics, _ = self._penetration_audit(q, frame_idx=-1)
        return float(metrics["penetration_depth"] or 0.0)

    def _measure_self_collision_violation(self, frame_idx: int) -> float | None:
        """Return maximum nonlinear configured self-collision distance violation."""
        if not self._self_collision_enabled:
            return None
        if self._self_collision_windows is not None and not any(
            start <= frame_idx <= end for start, end in self._self_collision_windows
        ):
            return None
        threshold = max(float(self.collision_detection_threshold), float(self._self_collision_tolerance) + 1e-6)
        fromto = np.zeros(6, dtype=float)
        violation = 0.0
        for geom_a, geom_b in self._self_collision_geom_pairs:
            distance = mujoco.mj_geomDistance(
                self.robot_model,
                self.robot_data,
                geom_a,
                geom_b,
                threshold,
                fromto,
            )
            violation = max(violation, float(self._self_collision_tolerance) - float(distance))
        return max(0.0, violation)

    def _measure_foot_sticking_error(
        self,
        q: np.ndarray,
        q_t_last: np.ndarray,
        foot_sticking: dict[str, bool],
        frame_idx: int,
    ) -> float:
        """Measure final FK foot error against the prior-frame anchors."""
        if self.q_a_init_idx >= 12:
            return 0.0
        _, current_positions, _ = self._calc_manipulator_jacobians(q, links=self.foot_links, obj_frame=False)
        _, prior_positions, _ = self._calc_manipulator_jacobians(q_t_last, links=self.foot_links, obj_frame=False)
        left_active = any(bool(value) for key, value in foot_sticking.items() if key.lower().startswith("l"))
        right_active = any(bool(value) for key, value in foot_sticking.items() if key.lower().startswith("r"))
        max_error = 0.0
        for link_key, position in current_positions.items():
            key_lower = link_key.lower()
            sticking = ("left" in key_lower and left_active) or ("right" in key_lower and right_active)
            if sticking and self.activate_foot_sticking:
                max_error = max(max_error, float(np.linalg.norm(position[:2] - prior_positions[link_key][:2])))
            z_anchor = self._is_foot_locked_in_window(link_key, frame_idx) if self.foot_lock.enable else None
            if z_anchor is not None:
                max_error = max(max_error, abs(float(position[2]) - z_anchor))
        return max_error

    def _nonlinear_frame_metrics(
        self,
        q: np.ndarray,
        q_t_last: np.ndarray,
        foot_sticking: dict[str, bool],
        frame_idx: int,
        penetration_metrics: dict[str, float | int | None] | None = None,
    ) -> dict[str, float | int | None]:
        """Evaluate final-q physical feasibility rather than convex surrogates."""
        if penetration_metrics is None:
            penetration_metrics, _ = self._penetration_audit(q, frame_idx)
        self_collision = self._measure_self_collision_violation(frame_idx)
        q_a = q[self.q_a_indices]
        joint_violation = float(
            max(
                np.max(np.maximum(self.q_a_lb - q_a, 0.0), initial=0.0),
                np.max(np.maximum(q_a - self.q_a_ub, 0.0), initial=0.0),
            )
        )
        foot_error = self._measure_foot_sticking_error(q, q_t_last, foot_sticking, frame_idx)
        velocity_limit = self.semantic_config.rescue_velocity_limit_per_frame
        velocity_violation = None
        if velocity_limit is not None:
            velocity_violation = max(
                0.0,
                float(np.max(np.abs(q[self.q_a_indices] - q_t_last[self.q_a_indices]))) - velocity_limit,
            )
        return {
            **penetration_metrics,
            "self_collision_violation": self_collision,
            "joint_limit_violation": joint_violation,
            "foot_sticking_error": foot_error,
            "velocity_limit_violation": velocity_violation,
        }

    def _target_interaction_vertices(self, q: np.ndarray, obj_pts_local: np.ndarray) -> np.ndarray:
        """Return target body/object vertices in the same object-local frame as the source mesh."""
        _, robot_points_dict, _ = self._calc_manipulator_jacobians(
            q,
            links=self.laplacian_match_links,
            obj_frame=(self.object_name != "ground"),
        )
        robot_points_local = np.asarray(
            [robot_points_dict[key] for key in self.laplacian_match_links]
        )
        return np.vstack([robot_points_local, obj_pts_local])


    def _interaction_and_laplacian_metrics(
        self,
        q: np.ndarray,
        target_laplacian: np.ndarray,
        adj_list: list[list[int]],
        source_vertices: np.ndarray,
        obj_pts_demo: np.ndarray,
        obj_pts_local: np.ndarray,
        body_mapping: BodyVertexMapping,
        *,
        return_vertex_residuals: bool = False,
    ) -> dict[str, Any] | tuple[dict[str, Any], np.ndarray]:
        """Measure unweighted Laplacian and sampled-surface interaction errors."""
        robot_vertices = self._target_interaction_vertices(q, obj_pts_local)
        final_laplacian = calculate_laplacian_coordinates(robot_vertices, adj_list)
        residual_by_vertex = np.linalg.norm(final_laplacian - target_laplacian, axis=1)
        metrics: dict[str, Any] = {
            "final_laplacian_error": float(np.mean(residual_by_vertex)),
            "left_hand_demo_object_distance": None,
            "left_hand_robot_object_distance": None,
            "left_hand_object_error": None,
            "left_hand_local_laplacian_error": None,
            "left_hand_object_neighbor_laplacian_error": None,
            "right_hand_demo_object_distance": None,
            "right_hand_robot_object_distance": None,
            "right_hand_object_error": None,
            "right_hand_local_laplacian_error": None,
            "right_hand_object_neighbor_laplacian_error": None,
        }
        for semantic_name in ("left_hand", "right_hand"):
            vertex_idx = body_mapping.indices.get(semantic_name)
            if vertex_idx is None or len(obj_pts_demo) == 0 or len(obj_pts_local) == 0:
                continue
            demo_distance = float(np.min(np.linalg.norm(obj_pts_demo - source_vertices[vertex_idx], axis=1)))
            robot_distance = float(np.min(np.linalg.norm(obj_pts_local - robot_vertices[vertex_idx], axis=1)))
            metrics[f"{semantic_name}_demo_object_distance"] = demo_distance
            metrics[f"{semantic_name}_robot_object_distance"] = robot_distance
            metrics[f"{semantic_name}_object_error"] = abs(robot_distance - demo_distance)
            metrics[f"{semantic_name}_local_laplacian_error"] = float(residual_by_vertex[vertex_idx])
            object_neighbors = [
                neighbor_idx
                for neighbor_idx in adj_list[vertex_idx]
                if len(self.laplacian_match_links) <= neighbor_idx < len(robot_vertices)
            ]
            if object_neighbors:
                metrics[f"{semantic_name}_object_neighbor_laplacian_error"] = float(
                    np.mean(residual_by_vertex[object_neighbors])
                )

        for semantic_name in ("pelvis", "left_hand", "right_hand"):
            joint_name = body_mapping.joint_names.get(semantic_name)
            if joint_name is None:
                metrics[f"{semantic_name}_robot_position"] = None
                continue
            link_name = self.laplacian_match_links[joint_name]
            metrics[f"{semantic_name}_robot_position"] = self._get_robot_link_positions(q, [link_name])[0].tolist()
        if return_vertex_residuals:
            return metrics, residual_by_vertex.copy()
        return metrics

    def _semantic_residual_metrics(
        self,
        q: np.ndarray,
        target_laplacian: np.ndarray,
        adj_list: list[list[int]],
        obj_pts_local: np.ndarray,
        source_vertices: np.ndarray,
        part_vertex_indices: tuple[int, ...],
        semantic_edges: tuple[tuple[int, int], ...],
    ) -> dict[str, Any]:
        """Measure semantic residuals independently from solver weights.

        These diagnostics are deliberately nonlinear and unweighted. They are
        used by profiling and the evaluator, never by the solver.
        """
        target_vertices = self._target_interaction_vertices(q, obj_pts_local)
        final_laplacian = calculate_laplacian_coordinates(target_vertices, adj_list)
        laplacian_residual = np.linalg.norm(final_laplacian - target_laplacian, axis=1)
        part = (
            float(np.mean(laplacian_residual[np.asarray(part_vertex_indices, dtype=np.int64)]))
            if part_vertex_indices
            else 0.0
        )
        edge_errors = [
            float(
                np.linalg.norm(
                    (target_vertices[body_idx] - target_vertices[object_idx])
                    - (source_vertices[body_idx] - source_vertices[object_idx])
                )
            )
            for body_idx, object_idx in semantic_edges
        ]
        edge = float(np.mean(edge_errors)) if edge_errors else 0.0
        return {
            "semantic_part_residual": part,
            "semantic_edge_residual": edge,
            "semantic_residual": part + edge,
            "semantic_part_vertex_count": len(part_vertex_indices),
            "semantic_cross_edge_count": len(semantic_edges),
        }

    def _semantic_iteration_metrics(
        self,
        q: np.ndarray,
        target_laplacian: np.ndarray,
        adj_list: list[list[int]],
        obj_pts_local: np.ndarray,
        source_vertices: np.ndarray,
        event: SemanticEvent,
        body_mapping: BodyVertexMapping,
    ) -> dict[str, float | int | None]:
        """Evaluate an accepted SQP iterate with the method-independent metrics."""
        body_indices = tuple(
            dict.fromkeys(
                body_mapping.indices[part]
                for part in event.body_parts
                if part in body_mapping.indices
            )
        )
        if not body_indices:
            raise ValueError(f"event {event.name!r} has no evaluable body vertices")

        target_vertices = self._target_interaction_vertices(q, obj_pts_local)
        final_laplacian = calculate_laplacian_coordinates(target_vertices, adj_list)
        residual = np.linalg.norm(final_laplacian - target_laplacian, axis=1)
        body_array = np.asarray(body_indices, dtype=np.int64)

        local_indices = set(body_indices)
        semantic_edges: set[tuple[int, int]] = set()
        num_body_vertices = len(self.laplacian_match_links)
        for body_idx in body_indices:
            for neighbor_idx in adj_list[body_idx]:
                neighbor = int(neighbor_idx)
                if num_body_vertices <= neighbor < len(target_vertices):
                    local_indices.add(neighbor)
                    semantic_edges.add((body_idx, neighbor))

        edge_errors = [
            float(
                np.linalg.norm(
                    (target_vertices[body_idx] - target_vertices[object_idx])
                    - (source_vertices[body_idx] - source_vertices[object_idx])
                )
            )
            for body_idx, object_idx in sorted(semantic_edges)
        ]
        return {
            "global_laplacian_error": float(np.mean(residual)),
            "part_error": float(np.mean(residual[body_array])),
            "local_error": float(
                np.mean(residual[np.asarray(sorted(local_indices), dtype=np.int64)])
            ),
            "edge_error": float(np.mean(edge_errors)) if edge_errors else None,
            "part_vertex_count": len(body_indices),
            "edge_count": len(edge_errors),
        }

    def _named_geom_pair_signed_distance(
        self,
        q: np.ndarray,
        geom_name_a: str,
        geom_name_b: str,
    ) -> float:
        """Return an exact MuJoCo signed distance for a named diagnostic pair."""
        geom_a = mujoco.mj_name2id(
            self.robot_model,
            mujoco.mjtObj.mjOBJ_GEOM,
            geom_name_a,
        )
        geom_b = mujoco.mj_name2id(
            self.robot_model,
            mujoco.mjtObj.mjOBJ_GEOM,
            geom_name_b,
        )
        if geom_a < 0 or geom_b < 0:
            raise ValueError(
                f"diagnostic geom pair not found: {geom_name_a!r}, {geom_name_b!r}"
            )
        self.robot_data.qpos[:] = q
        mujoco.mj_forward(self.robot_model, self.robot_data)
        fromto = np.zeros(6, dtype=float)
        return float(
            mujoco.mj_geomDistance(
                self.robot_model,
                self.robot_data,
                geom_a,
                geom_b,
                float(self.collision_detection_threshold),
                fromto,
            )
        )

    def _rescue_reasons(self, metrics: dict[str, Any], interaction_relevant: bool) -> list[str]:
        config = self.semantic_config
        reasons: list[str] = []
        if float(metrics["penetration_depth"]) > config.rescue_penetration_tolerance:
            reasons.append("penetration")
        self_collision_violation = metrics.get("self_collision_violation")
        if (
            self_collision_violation is not None
            and float(self_collision_violation) > config.rescue_self_collision_tolerance
        ):
            reasons.append("self_collision")
        if float(metrics["joint_limit_violation"]) > config.rescue_joint_limit_tolerance:
            reasons.append("joint_limits")
        if float(metrics["foot_sticking_error"]) > config.rescue_foot_sticking_tolerance:
            reasons.append("foot_sticking")
        velocity_violation = metrics.get("velocity_limit_violation")
        if velocity_violation is not None and float(velocity_violation) > 0:
            reasons.append("velocity_limits")
        if interaction_relevant:
            hand_errors = [
                value
                for value in (
                    metrics.get("left_hand_object_error"),
                    metrics.get("right_hand_object_error"),
                )
                if value is not None
            ]
            if hand_errors and max(float(value) for value in hand_errors) > config.rescue_hand_object_tolerance:
                reasons.append("hand_object_residual")
        return reasons

    def _semantic_events_for_run(self) -> list[SemanticEvent]:
        """Load semantic JSON only for modes that explicitly use it."""
        config = self.semantic_config
        if not config.uses_semantic_events:
            return []
        if config.semantic_keyframe_path is None:
            raise ValueError(f"semantic_keyframe_path is required for mode={config.mode!r}")
        return load_semantic_plan_projection(config.semantic_keyframe_path)

    def _weight_events_for_run(
        self,
        events: list[SemanticEvent],
        num_frames: int,
    ) -> list[SemanticEvent]:
        """Return the registered Legacy or deterministic full-event support."""
        del num_frames
        if not self.semantic_config.uses_semantic_weights:
            return []
        if self.semantic_config.uses_full_event_weights:
            return list(events)
        registered = set(self.semantic_config.legacy_weight_events)
        return [event for event in events if event.name in registered]

    def _always_hand_matched_amplitude(
        self,
        human_joint_motions: np.ndarray,
        object_poses: np.ndarray,
        object_points_local_demo: np.ndarray | list[np.ndarray],
        weight_events: list[SemanticEvent],
        body_mapping: BodyVertexMapping,
    ) -> tuple[float, float, float]:
        """Match whole-trajectory L1 preference energy to true semantic weighting."""
        config = self.semantic_config
        frame_adjacencies: list[list[list[int]]] = []
        target_energy = 0.0
        for frame_idx in range(len(human_joint_motions)):
            object_quat = object_poses[frame_idx, 3:]
            object_trans = object_poses[frame_idx, :3]
            human_points = human_joint_motions[frame_idx, self.smplh_mapped_joint_indices]
            if self.object_name != "ground":
                human_points = transform_points_world_to_local(object_quat, object_trans, human_points)
            object_points = (
                object_points_local_demo[frame_idx]
                if isinstance(object_points_local_demo, list)
                else object_points_local_demo
            )
            vertices, simplices = create_interaction_mesh(np.vstack([human_points, object_points]))
            adjacency = get_adjacency_list(simplices, len(vertices))
            frame_adjacencies.append(adjacency)
            target_energy += build_semantic_vertex_weight_result(
                frame_idx=frame_idx,
                events=weight_events,
                body_mapping=body_mapping,
                adjacency=adjacency,
                num_human_vertices=len(self.laplacian_match_links),
                num_vertices=len(vertices),
                config=config,
            ).l1_norm_alpha_minus_one

        def always_energy(amplitude: float) -> float:
            return float(
                sum(
                    build_always_body_vertex_weight_result(
                        body_parts=("left_hand", "right_hand"),
                        body_mapping=body_mapping,
                        adjacency=adjacency,
                        num_human_vertices=len(self.laplacian_match_links),
                        num_vertices=len(adjacency),
                        config=config,
                        amplitude=amplitude,
                    ).l1_norm_alpha_minus_one
                    for adjacency in frame_adjacencies
                )
            )

        low, high = 0.0, 1.0
        for _ in range(50):
            midpoint = (low + high) / 2.0
            if always_energy(midpoint) < target_energy:
                low = midpoint
            else:
                high = midpoint
        amplitude = (low + high) / 2.0
        return amplitude, target_energy, always_energy(amplitude)

    def _semantic_control_energy_scale(
        self,
        human_joint_motions: np.ndarray,
        object_poses: np.ndarray,
        object_points_local_demo: np.ndarray | list[np.ndarray],
        target_events: list[SemanticEvent],
        control_events: list[SemanticEvent],
        body_mapping: BodyVertexMapping,
    ) -> tuple[float, float, float, float]:
        """Match a causal control's realized trajectory L1 to Semantic Weight.

        Event multipliers and temporal kernels are evaluated first without any
        modification. A single global scale on the resulting mean-one
        deviations then compensates only for topology and event-overlap energy
        changes. This keeps the control's spatial and temporal preference
        pattern intact while making total preference energy exactly comparable.
        """
        config = self.semantic_config
        target_energy = 0.0
        control_results: list[SemanticWeightResult] = []
        for frame_idx in range(len(human_joint_motions)):
            object_quat = object_poses[frame_idx, 3:]
            object_trans = object_poses[frame_idx, :3]
            human_points = human_joint_motions[frame_idx, self.smplh_mapped_joint_indices]
            if self.object_name != "ground":
                human_points = transform_points_world_to_local(object_quat, object_trans, human_points)
            object_points = (
                object_points_local_demo[frame_idx]
                if isinstance(object_points_local_demo, list)
                else object_points_local_demo
            )
            vertices, simplices = create_interaction_mesh(np.vstack([human_points, object_points]))
            adjacency = get_adjacency_list(simplices, len(vertices))
            target_energy += build_semantic_vertex_weight_result(
                frame_idx=frame_idx,
                events=target_events,
                body_mapping=body_mapping,
                adjacency=adjacency,
                num_human_vertices=len(self.laplacian_match_links),
                num_vertices=len(vertices),
                config=config,
            ).l1_norm_alpha_minus_one
            control_results.append(
                build_semantic_vertex_weight_result(
                    frame_idx=frame_idx,
                    events=control_events,
                    body_mapping=body_mapping,
                    adjacency=adjacency,
                    num_human_vertices=len(self.laplacian_match_links),
                    num_vertices=len(vertices),
                    config=config,
                )
            )
        control_energy = float(sum(result.l1_norm_alpha_minus_one for result in control_results))
        if control_energy <= 0:
            raise ValueError("semantic causal control has zero preference energy and cannot be matched")
        scale = target_energy / control_energy
        matched_energy = float(
            sum(scale_semantic_weight_result(result, scale).l1_norm_alpha_minus_one for result in control_results)
        )
        if not np.isclose(matched_energy, target_energy, rtol=1e-10, atol=1e-10):
            raise RuntimeError(
                f"semantic control energy match failed: target={target_energy}, actual={matched_energy}"
            )
        return scale, target_energy, control_energy, matched_energy

    def retarget_motion(
        self,
        human_joint_motions,
        object_poses,
        object_poses_augmented,
        object_points_local_demo,
        object_points_local,
        foot_sticking_sequences,
        q_a_init=None,
        q_nominal_list=None,
        original=True,
        dest_res_path=None,
    ):
        """
        The main function to retarget an entire motion sequence frame by frame.

        Args:
            human_joint_motions (np.ndarray): (num_frames, num_joints, 3) array.
            object_poses (np.ndarray): (num_frames, 7) array of demo object poses (quat, trans).
            object_poses_augmented (np.ndarray): (num_frames, 7) array of augmented object poses (quat, trans).
            object_points_local_demo (np.ndarray | list[np.ndarray]): Demo object points in local frame.
                Single array for static points, or list of num_frames arrays for per-frame points.
            object_points_local (np.ndarray | list[np.ndarray]): Current object points in local frame.
                Single array for static points, or list of num_frames arrays for per-frame points.
            foot_sticking_sequences (list): List of foot sticking sequences for each frame.
            q_a_init (np.ndarray, optional): Initial robot configuration.
            q_a_nominal (np.ndarray, optional): Nominal robot configuration.

        Returns:
            tuple: (retargeted_motions, obj_pts_demo_list, obj_pts_list, tetrahedra)
        """
        run_start = time.perf_counter()
        self._convex_solver_calls = 0
        num_frames = human_joint_motions.shape[0]
        if num_frames < 1:
            raise ValueError("human_joint_motions must contain at least one frame")
        if isinstance(object_points_local_demo, list):
            assert len(object_points_local_demo) == num_frames, (
                f"object_points_local_demo length {len(object_points_local_demo)} != num_frames {num_frames}"
            )
        if isinstance(object_points_local, list):
            assert len(object_points_local) == num_frames, (
                f"object_points_local length {len(object_points_local)} != num_frames {num_frames}"
            )
        if q_nominal_list is not None:
            q_locked_list = q_nominal_list
        else:
            q_locked_list = np.zeros((num_frames, self.nq))
            q_locked_list[0, self.q_a_indices] = q_a_init

        q_locked_list[:, -7:] = object_poses_augmented
        q = np.copy(q_locked_list[0])
        retargeted_motions = [q]

        config = self.semantic_config
        events = self._semantic_events_for_run()
        weight_events = self._weight_events_for_run(events, num_frames)
        budget_plan = make_budget_plan(num_frames, events, config)
        timeline_events = (
            weight_events
            if config.uses_semantic_weights
            else list(budget_plan.events)
        )
        timeline = SemanticTimeline(
            timeline_events,
            sigma=config.semantic_temporal_sigma,
            phase_weight=config.phase_semantic_weight,
            nearby_radius=config.near_trigger_radius,
        )
        semantic_parts = [part for event in [*events, *weight_events] for part in event.body_parts]
        metric_parts = ["pelvis", "left_hand", "right_hand", *semantic_parts]
        body_mapping = resolve_body_vertex_mapping(list(self.laplacian_match_links), metric_parts)
        if config.uses_final_weighting and body_mapping.missing:
            raise ValueError(
                "Semantic body parts must map to interaction vertices; missing "
                f"{body_mapping.missing}"
            )
        always_hand_amplitude = 1.0
        matched_target_energy: float | None = None
        matched_actual_energy: float | None = None
        semantic_control_energy_scale = 1.0
        semantic_control_target_energy: float | None = None
        semantic_control_raw_energy: float | None = None
        semantic_control_matched_energy: float | None = None
        if config.mode == "uniform2_always_hand_matched":
            always_hand_amplitude, matched_target_energy, matched_actual_energy = (
                self._always_hand_matched_amplitude(
                    human_joint_motions,
                    object_poses,
                    object_points_local_demo,
                    weight_events,
                    body_mapping,
                )
            )
        if config.mode in {"uniform2_random_time_hand", "uniform2_wrong_body"}:
            true_weight_events = [
                event for event in events if event.name in set(config.legacy_weight_events)
            ]
            (
                semantic_control_energy_scale,
                semantic_control_target_energy,
                semantic_control_raw_energy,
                semantic_control_matched_energy,
            ) = self._semantic_control_energy_scale(
                human_joint_motions,
                object_poses,
                object_points_local_demo,
                true_weight_events,
                weight_events,
                body_mapping,
            )

        tetrahedra: list[np.ndarray] = []
        obj_pts_demo_list: list[np.ndarray] = []
        obj_pts_list: list[np.ndarray] = []
        profiles: list[dict[str, Any]] = []
        rescue_log: list[dict[str, Any]] = []
        penetration_pair_rows: list[dict[str, Any]] = []
        frame_costs: list[float] = []
        unweighted_vertex_residuals: list[np.ndarray] = []
        interaction_mesh_adjacencies: list[np.ndarray] = []
        source_interaction_vertices: list[np.ndarray] = []
        target_interaction_vertices: list[np.ndarray] = []
        semantic_edge_zero_frames: list[int] = []
        base_qpos_frames: list[np.ndarray] = []
        semantic_iteration_trace: list[dict[str, Any]] = []
        iteration_trace_diagnostic_time = 0.0

        print(f"\nStarting motion retargeting for {num_frames} frames in mode={config.mode}...")

        with tqdm(range(num_frames)) as pbar:
            for i in pbar:
                frame_start = time.perf_counter()
                mesh_start = time.perf_counter()
                object_quat_demo = object_poses[i, 3:]
                object_trans_demo = object_poses[i, :3]
                human_mapped_joints = human_joint_motions[i, self.smplh_mapped_joint_indices]

                if self.object_name == "ground":
                    human_mapped_joints_in_object = human_mapped_joints
                else:
                    human_mapped_joints_in_object = transform_points_world_to_local(
                        object_quat_demo,
                        object_trans_demo,
                        human_mapped_joints,
                    )

                obj_pts_demo_i = (
                    object_points_local_demo[i]
                    if isinstance(object_points_local_demo, list)
                    else object_points_local_demo
                )
                obj_pts_i = object_points_local[i] if isinstance(object_points_local, list) else object_points_local
                source_vertices, source_tetrahedra = create_interaction_mesh(
                    np.vstack([human_mapped_joints_in_object, obj_pts_demo_i])
                )
                source_interaction_vertices.append(np.asarray(source_vertices, dtype=np.float64))
                tetrahedra.append(source_tetrahedra)
                adj_list = get_adjacency_list(source_tetrahedra, len(source_vertices))
                adjacency_matrix = np.zeros((len(source_vertices), len(source_vertices)), dtype=np.uint8)
                for vertex_idx, neighbors in enumerate(adj_list):
                    adjacency_matrix[vertex_idx, np.asarray(neighbors, dtype=np.int64)] = 1
                interaction_mesh_adjacencies.append(adjacency_matrix)
                target_laplacian = calculate_laplacian_coordinates(source_vertices, adj_list)
                st_context = spatiotemporal_semantic_context(
                    i,
                    weight_events if config.uses_semantic_weights else [],
                    sigma=config.semantic_temporal_sigma,
                    phase_weight=config.phase_semantic_weight,
                    kernel_cutoff=config.semantic_kernel_cutoff,
                    transition_truncated=config.uses_transition_truncated_weight_support,
                )
                part_indices = tuple(
                    dict.fromkeys(
                        body_mapping.indices[part]
                        for part in st_context.body_parts
                        if part in body_mapping.indices
                    )
                )
                semantic_edges = extract_semantic_cross_entity_edges(
                    part_indices,
                    adj_list,
                    len(self.laplacian_match_links),
                )
                if (
                    config.uses_final_weighting
                    and config.semantic_weight_components in {"edge", "part_edge"}
                    and st_context.semantic_importance >= config.semantic_kernel_cutoff
                    and not semantic_edges.edges
                ):
                    semantic_edge_zero_frames.append(i)
                    warnings.warn(
                        f"frame {i}: active semantic body vertices have no cross-entity edge",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                vertex_weights = None
                weight_result: SemanticWeightResult | None = None
                if config.uses_semantic_weights:
                    if config.mode in {"uniform2_always_hand_raw", "uniform2_always_hand_matched"}:
                        weight_result = build_always_body_vertex_weight_result(
                            body_parts=("left_hand", "right_hand"),
                            body_mapping=body_mapping,
                            adjacency=adj_list,
                            num_human_vertices=len(self.laplacian_match_links),
                            num_vertices=len(source_vertices),
                            config=config,
                            amplitude=always_hand_amplitude,
                        )
                    else:
                        weight_result = build_semantic_vertex_weight_result(
                            frame_idx=i,
                            events=weight_events,
                            body_mapping=body_mapping,
                            adjacency=adj_list,
                            num_human_vertices=len(self.laplacian_match_links),
                            num_vertices=len(source_vertices),
                            config=config,
                        )
                        if config.mode in {"uniform2_random_time_hand", "uniform2_wrong_body"}:
                            weight_result = scale_semantic_weight_result(
                                weight_result,
                                semantic_control_energy_scale,
                            )
                    vertex_weights = weight_result.weights
                mesh_time = time.perf_counter() - mesh_start

                if self.debug:
                    object_quat = object_poses_augmented[i, 3:]
                    object_trans = object_poses_augmented[i, :3]
                    obj_pts_demo = transform_points_local_to_world(object_quat_demo, object_trans_demo, obj_pts_demo_i)
                    obj_pts = transform_points_local_to_world(object_quat, object_trans, obj_pts_i)
                    obj_pts_demo_list.append(obj_pts_demo)
                    obj_pts_list.append(obj_pts)
                    human_kpts_handle_list = self.draw_keypoints(human_mapped_joints, name="human_kpts")
                    obj_kpts_demo_handle_list = self.draw_keypoints(
                        obj_pts_demo,
                        name="object_demo_kpts",
                        rgba=(1, 0, 0, 1),
                    )
                    obj_kpts_handle_list = self.draw_keypoints(obj_pts, name="object_kpts", rgba=(0, 1, 1, 1))

                if original:
                    w_nominal_tracking = self.w_nominal_tracking_init
                else:
                    w_nominal_tracking = self.w_nominal_tracking_init * np.exp(-i / self.nominal_tracking_tau)

                base_budget = int(budget_plan.budgets[i])
                final_budget = base_budget
                previous_q = retargeted_motions[-1]
                solver_calls_before_frame = self._convex_solver_calls
                optimization_start = time.perf_counter()
                q_a_nominal_frame = (
                    q_nominal_list[i, self.q_a_indices] if q_nominal_list is not None else None
                )
                iteration_diagnostics: tuple[dict[str, Any], ...] = ()
                exact_trace_events = [
                    event for event in events if i > 0 and event.trigger_frame == i
                ]
                is_diagnostic_trace_frame = i in config.diagnostic_iteration_frames
                if is_diagnostic_trace_frame and not exact_trace_events and events:
                    exact_trace_events = [
                        min(events, key=lambda event: abs(event.trigger_frame - i))
                    ]
                trace_events = (
                    exact_trace_events
                    if config.uses_full_event_weights
                    and (exact_trace_events or is_diagnostic_trace_frame)
                    else []
                )
                trace_kind = (
                    "exact_trigger"
                    if any(event.trigger_frame == i for event in trace_events)
                    else "diagnostic_frame"
                )
                trace_iterates: list[tuple[int, np.ndarray, float, float, float]] = []
                trace_previous_q = np.array(q, copy=True)

                def record_trace_iterate(iteration: int, q_iterate: np.ndarray, iterate_cost: float) -> None:
                    nonlocal trace_previous_q
                    update = np.asarray(q_iterate) - trace_previous_q
                    trace_iterates.append(
                        (
                            iteration,
                            np.array(q_iterate, copy=True),
                            iterate_cost,
                            float(np.linalg.norm(update)),
                            float(np.max(np.abs(update), initial=0.0)),
                        )
                    )
                    trace_previous_q = np.array(q_iterate, copy=True)

                q, cost, actual_iterations = self.iterate(
                    q_locked=q_locked_list[i],
                    q_n=q,
                    q_t_last=previous_q,
                    target_laplacian=target_laplacian,
                    adj_list=adj_list,
                    obj_pts_local=obj_pts_i,
                    foot_sticking=foot_sticking_sequences[i],
                    w_nominal_tracking=w_nominal_tracking,
                    q_a_nominal=q_a_nominal_frame,
                    init_t=i == 0,
                    n_iter=base_budget,
                    frame_idx=i,
                    vertex_residual_weights=vertex_weights,
                    return_iterations=True,
                    iteration_observer=record_trace_iterate if trace_events else None,
                )
                optimization_time = time.perf_counter() - optimization_start
                frame_trace_start = time.perf_counter()
                for (
                    iteration,
                    q_iterate,
                    iterate_cost,
                    q_update_l2_norm,
                    q_update_max_abs,
                ) in trace_iterates:
                    physical_trace: dict[str, Any] = {}
                    if is_diagnostic_trace_frame:
                        penetration_trace, _ = self._penetration_audit(q_iterate, i)
                        shoulder_box_distance = self._named_geom_pair_signed_distance(
                            q_iterate,
                            "left_shoulder_yaw_link",
                            "largebox",
                        )
                        physical_trace = {
                            "distance_query_threshold": float(
                                self.collision_detection_threshold
                            ),
                            "shoulder_box_signed_distance": shoulder_box_distance,
                            "shoulder_box_penetration_depth": max(
                                0.0,
                                -shoulder_box_distance,
                            ),
                            "max_illegal_penetration_depth": float(
                                penetration_trace["illegal_penetration_depth"] or 0.0
                            ),
                            "illegal_penetration_pair_count": int(
                                penetration_trace["illegal_penetration_pair_count"] or 0
                            ),
                        }
                    for event in trace_events:
                        semantic_iteration_trace.append(
                            {
                                "mode": config.mode,
                                "frame": i,
                                "event": event.name,
                                "body_parts": event.body_parts,
                                "trace_kind": trace_kind,
                                "configured_budget": base_budget,
                                "iteration": iteration,
                                "cost": iterate_cost,
                                "q_update_l2_norm": q_update_l2_norm,
                                "q_update_max_abs": q_update_max_abs,
                                **physical_trace,
                                **self._semantic_iteration_metrics(
                                    q_iterate,
                                    target_laplacian,
                                    adj_list,
                                    obj_pts_i,
                                    np.asarray(source_vertices, dtype=np.float64),
                                    event,
                                    body_mapping,
                                ),
                            }
                        )
                frame_trace_time = time.perf_counter() - frame_trace_start
                iteration_trace_diagnostic_time += frame_trace_time
                registered_base_budget = (
                    base_budget
                    if i == 0 or not config.is_uniform2_mainline
                    else config.round2_base_budget
                )
                base_iterations = min(actual_iterations, registered_base_budget)
                configured_semantic_extra = (
                    max(0, final_budget - registered_base_budget)
                    if config.is_uniform2_mainline and i in budget_plan.extra_budget_frames
                    else max(0, final_budget - base_budget)
                )
                semantic_extra_iterations = (
                    max(0, actual_iterations - registered_base_budget)
                    if i in budget_plan.extra_budget_frames
                    else 0
                )
                coarse_cost = float(cost)
                coarse_interaction_metrics: dict[str, Any] | None = None
                base_qpos_frames.append(np.array(q, copy=True))

                check_start = time.perf_counter()
                penetration_metrics, frame_penetration_rows = self._penetration_audit(q, i)
                physical_metrics = self._nonlinear_frame_metrics(
                    q,
                    previous_q,
                    foot_sticking_sequences[i],
                    i,
                    penetration_metrics=penetration_metrics,
                )
                interaction_metrics, frame_vertex_residuals = self._interaction_and_laplacian_metrics(
                    q,
                    target_laplacian,
                    adj_list,
                    source_vertices,
                    obj_pts_demo_i,
                    obj_pts_i,
                    body_mapping,
                    return_vertex_residuals=True,
                )
                semantic_residual_metrics = self._semantic_residual_metrics(
                    q,
                    target_laplacian,
                    adj_list,
                    obj_pts_i,
                    np.asarray(source_vertices, dtype=np.float64),
                    part_indices,
                    semantic_edges.edges,
                )
                metrics = {
                    **physical_metrics,
                    **interaction_metrics,
                    **semantic_residual_metrics,
                }
                collision_check_time = time.perf_counter() - check_start

                interaction_names = set(config.legacy_weight_events)
                interaction_relevant = any(
                    event.name in interaction_names
                    and (
                        event.start_frame <= i <= event.end_frame
                        or abs(i - event.trigger_frame) <= config.near_trigger_radius
                    )
                    for event in events
                )
                semantic_residual_metrics = self._semantic_residual_metrics(
                    q,
                    target_laplacian,
                    adj_list,
                    obj_pts_i,
                    np.asarray(source_vertices, dtype=np.float64),
                    part_indices,
                    semantic_edges.edges,
                )
                metrics.update(semantic_residual_metrics)
                penetration_pair_rows.extend(frame_penetration_rows)
                unweighted_vertex_residuals.append(frame_vertex_residuals)
                target_interaction_vertices.append(self._target_interaction_vertices(q, obj_pts_i))

                last_iteration_diagnostics = iteration_diagnostics[-1] if iteration_diagnostics else {}
                frame_constraint_valid = not self._rescue_reasons(metrics, interaction_relevant=False)
                if self.debug:
                    robot_link_positions = self._get_robot_link_positions(q, self.laplacian_match_links.values())
                    robot_kpts_handle_list = self.draw_keypoints(
                        robot_link_positions,
                        name="robot_kpts",
                        rgba=(0, 1, 0, 1),
                    )

                retargeted_motions.append(q)
                frame_costs.append(float(cost))
                if self.visualize and self.debug:
                    self.draw_q(q)

                semantic_info = timeline.frame_info(i)
                profile_body_parts = (
                    list(st_context.body_parts)
                    if config.uses_semantic_weights
                    else semantic_info.body_parts
                )
                if config.mode in {"uniform2_always_hand_raw", "uniform2_always_hand_matched"}:
                    profile_body_parts = ["left_hand", "right_hand"]
                profiles.append(
                    {
                        "frame_idx": i,
                        "active_event": (
                            list(st_context.active_events)
                            if config.uses_semantic_weights
                            else semantic_info.active_events
                        ),
                        "nearest_trigger": (
                            st_context.dominant_event
                            if config.uses_semantic_weights
                            else semantic_info.nearest_event
                        ),
                        "distance_to_trigger": semantic_info.nearest_trigger_distance,
                        "semantic_score": (
                            st_context.semantic_importance
                            if config.uses_semantic_weights
                            else semantic_info.semantic_score
                        ),
                        "semantic_tau": st_context.tau,
                        "semantic_importance": st_context.semantic_importance,
                        "E_omni": last_iteration_diagnostics.get("post_omni_objective"),
                        "E_part": metrics["semantic_part_residual"],
                        "E_edge": metrics["semantic_edge_residual"],
                        "R_sem": metrics["semantic_residual"],
                        "semantic_edge_count": len(semantic_edges.edges),
                        "semantic_edge_warning": bool(
                            config.uses_final_weighting
                            and config.semantic_weight_components in {"edge", "part_edge"}
                            and st_context.semantic_importance >= config.semantic_kernel_cutoff
                            and not semantic_edges.edges
                        ),
                        "base_budget": base_budget,
                        "configured_budget": base_budget,
                        "final_budget_after_rescue": final_budget,
                        "configured_semantic_extra_iterations": configured_semantic_extra,
                        "base_sqp_iterations": base_iterations,
                        "semantic_extra_iterations": semantic_extra_iterations,
                        "actual_sqp_iterations": actual_iterations,
                        "convex_solver_calls": self._convex_solver_calls - solver_calls_before_frame,
                        "is_semantic_trigger": i in budget_plan.semantic_trigger_frames,
                        "is_random_budget_frame": (
                            config.uses_random_exact_budget and i in budget_plan.extra_budget_frames
                        ),
                        "allocation_reason": (
                            budget_plan.allocation_reasons[i]
                            if budget_plan.allocation_reasons
                            else "frame0_initialization"
                            if i == 0
                            else "ordinary_base"
                        ),
                        "semantic_body_parts": profile_body_parts,
                        "frame_wall_time": time.perf_counter() - frame_start - frame_trace_time,
                        "mesh_time": mesh_time,
                        "optimization_time": optimization_time,
                        "collision_check_time": collision_check_time,
                        "jacobian_time": None,
                        "solver_time": optimization_time,
                        "coarse_total_cost": coarse_cost,
                        "final_total_cost": float(cost),
                        "coarse_left_hand_object_error": (
                            coarse_interaction_metrics.get("left_hand_object_error")
                            if coarse_interaction_metrics is not None
                            else None
                        ),
                        "coarse_right_hand_object_error": (
                            coarse_interaction_metrics.get("right_hand_object_error")
                            if coarse_interaction_metrics is not None
                            else None
                        ),
                        "semantic_vertex_weight_mean": float(vertex_weights.mean()) if vertex_weights is not None else 1.0,
                        "semantic_vertex_weight_max": float(vertex_weights.max()) if vertex_weights is not None else 1.0,
                        "semantic_boosted_vertex_count": (
                            weight_result.boosted_vertex_count if weight_result is not None else 0
                        ),
                        "semantic_weight_l1": (
                            weight_result.l1_norm_alpha_minus_one if weight_result is not None else 0.0
                        ),
                        "semantic_weight_l2": (
                            weight_result.l2_norm_alpha_minus_one if weight_result is not None else 0.0
                        ),
                        "semantic_weight_edge_count": (
                            weight_result.semantic_edge_count if weight_result is not None else 0
                        ),
                        "semantic_weight_edge_endpoint_count": (
                            weight_result.edge_endpoint_count if weight_result is not None else 0
                        ),
                        "semantic_weight_deduplicated_overlap_count": (
                            weight_result.deduplicated_overlap_count if weight_result is not None else 0
                        ),
                        "is_critical_interaction_neighborhood": interaction_relevant,
                        **metrics,
                    }
                )
                pbar.set_postfix(cost=cost, iterations=actual_iterations, budget=final_budget)

        if self.debug:
            for handle_list in (
                human_kpts_handle_list,
                obj_kpts_demo_handle_list,
                obj_kpts_handle_list,
                robot_kpts_handle_list,
            ):
                if handle_list:
                    for handle in handle_list:
                        handle.remove()
                    handle_list.clear()

        trajectory = np.array(retargeted_motions)[1:]
        np.savez(
            dest_res_path,
            qpos=trajectory,
            human_joints=human_joint_motions,
            fps=30,
            cost=frame_costs[-1],
            frame_costs=np.asarray(frame_costs),
            actual_sqp_iterations=np.asarray([row["actual_sqp_iterations"] for row in profiles]),
            base_sqp_iterations=np.asarray([row["base_sqp_iterations"] for row in profiles]),
            semantic_extra_iterations=np.asarray([row["semantic_extra_iterations"] for row in profiles]),
            max_sqp_budgets=np.asarray([row["final_budget_after_rescue"] for row in profiles]),
            convex_solver_calls=np.asarray([row["convex_solver_calls"] for row in profiles]),
            unweighted_vertex_residuals=np.stack(unweighted_vertex_residuals),
            interaction_mesh_adjacency=np.stack(interaction_mesh_adjacencies),
            interaction_mesh_joint_names=np.asarray(list(self.laplacian_match_links), dtype=np.str_),
            interaction_mesh_num_body_vertices=np.asarray(len(self.laplacian_match_links), dtype=np.int64),
            source_interaction_vertices=np.stack(source_interaction_vertices),
            target_interaction_vertices=np.stack(target_interaction_vertices),
            base_qpos=np.stack(base_qpos_frames),
        )
        print("Saving results to path:", dest_res_path)

        total_wall_time = time.perf_counter() - run_start - iteration_trace_diagnostic_time
        if self._semantic_config_explicit:
            destination = Path(dest_res_path)
            profile_dir = config.profile_dir or destination.parent / f"{destination.stem}_profile"
            write_run_artifacts(
                profile_dir,
                profiles,
                rescue_log,
                mode=config.mode,
                total_wall_time=total_wall_time,
                penetration_tolerance=config.rescue_penetration_tolerance,
                penetration_pairs=penetration_pair_rows,
                metadata={
                    "semantic_keyframe_path": str(config.semantic_keyframe_path)
                    if config.semantic_keyframe_path is not None
                    else None,
                    "random_seed": config.random_seed,
                    "weight_event_centers": {
                        event.name: event.trigger_frame for event in weight_events
                    },
                    "semantic_plan_projection": {
                        "source": "historical semantic_v2 JSON",
                        "retained_fields": [
                            "event",
                            "window",
                            "trigger_frame",
                            "body_parts",
                            "trigger",
                            "end",
                            "rationale",
                        ],
                        "ignored_fields": [
                            "criticality",
                            "criticality_level",
                            "criticality_rationale",
                            "failure_if_inaccurate",
                        ],
                    },
                    "eligible_trigger_count": len(budget_plan.semantic_trigger_frames),
                    "eligible_trigger_frames": list(budget_plan.semantic_trigger_frames),
                    "eligible_trigger_events": [
                        {
                            "event": event.name,
                            "frame": event.trigger_frame,
                            "body_parts": event.body_parts,
                        }
                        for event in events
                        if event.trigger_frame in budget_plan.semantic_trigger_frames
                    ],
                    "extra_budget_frames": list(budget_plan.extra_budget_frames),
                    "always_hand_amplitude": always_hand_amplitude,
                    "always_hand_matched_target_l1": matched_target_energy,
                    "always_hand_matched_actual_l1": matched_actual_energy,
                    "semantic_control_energy_scale": semantic_control_energy_scale,
                    "semantic_control_target_l1": semantic_control_target_energy,
                    "semantic_control_raw_l1": semantic_control_raw_energy,
                    "semantic_control_matched_l1": semantic_control_matched_energy,
                    "configured_velocity_check": config.rescue_velocity_limit_per_frame is not None,
                    "self_collision_check_configured": bool(
                        self._self_collision_enabled and self._self_collision_geom_pairs
                    ),
                    "interaction_distance_metric": "nearest existing object sample (surface approximation)",
                    "scheduler": {
                        "frame0_budget": config.frame0_budget,
                        "ordinary_budget": (
                            config.round2_base_budget
                            if config.is_uniform2_mainline
                            else config.uniform_budget
                            if config.mode == "uniform"
                            else config.original_budget
                        ),
                        "exact_trigger_budget": config.exact_trigger_budget,
                        "random_exclusion_radius": config.random_exclusion_radius,
                        "exact_trigger_only": True,
                    },
                    "residual_weights": {
                        "body_weight_multiplier": config.body_weight_multiplier,
                        "object_neighbor_multiplier": config.object_neighbor_multiplier,
                        "semantic_temporal_sigma": config.semantic_temporal_sigma,
                        "phase_semantic_weight": config.phase_semantic_weight,
                        "components": config.semantic_weight_components,
                        "event_strength": "uniform; criticality fields ignored",
                        "normalization": "mean vertex weight = 1 per frame",
                        "body_only_weight_events": list(config.body_only_weight_events),
                        "object_neighbor_propagation": (
                            "disabled only for body_only_weight_events; unchanged for all other events"
                        ),
                        "temporal_support_policy": (
                            "trigger <= frame <= event end and frame < next event trigger"
                            if config.uses_transition_truncated_weight_support
                            else "symmetric Gaussian plus inclusive event phase window"
                        ),
                        "causal_control_energy_match": "single global scale on normalized alpha-1 deviations",
                        "edge_policy": (
                            "cross body-object Delaunay edge endpoints, multiplier 2; "
                            "overlap with Part/object-neighbor weighting is de-duplicated by max"
                        ),
                    },
                    "semantic_weighting_mainline": {
                        "solver_change": "registered residual reweighting only; no additive objective",
                        "weight_support": (
                            "all deterministic projected events"
                            if config.uses_full_event_weights
                            else "historical contact/lift/place/release"
                            if config.uses_semantic_weights
                            else "none"
                        ),
                        "approach_weight_policy": (
                            "pelvis body-only; no approach contribution to object-neighbor vertices"
                            if "approach" in config.body_only_weight_events
                            else "standard body plus object-neighbor propagation"
                        ),
                        "transition_truncated_weight_support": (
                            config.uses_transition_truncated_weight_support
                        ),
                        "edge_zero_frames": semantic_edge_zero_frames,
                        "compute_policy": (
                            "fixed Uniform-2 plus optional registered exact-trigger budget "
                            f"{config.exact_trigger_budget}"
                        ),
                        "iteration_trace_diagnostic_time_excluded_s": iteration_trace_diagnostic_time,
                        "diagnostic_iteration_frames": list(
                            config.diagnostic_iteration_frames
                        ),
                    },
                    "precision_payload": {
                        "residual": "unweighted uniform-Laplacian norm per interaction-mesh vertex",
                        "topology": "per-frame original Delaunay adjacency",
                        "semantic_alpha_used": False,
                    },
                    "rescue_tolerances": {
                        "penetration": config.rescue_penetration_tolerance,
                        "self_collision": config.rescue_self_collision_tolerance,
                        "joint_limit": config.rescue_joint_limit_tolerance,
                        "foot_sticking": config.rescue_foot_sticking_tolerance,
                        "hand_object": config.rescue_hand_object_tolerance,
                        "velocity_per_frame": config.rescue_velocity_limit_per_frame,
                    },
                    "penetration_pair_policy": {
                        "expected": ["foot-ground", "hand-box", "object-ground"],
                        "illegal": [
                            "other-body-box",
                            "robot-ground-excluding-foot",
                            "self-collision",
                        ],
                    },
                },
            )
            if semantic_iteration_trace:
                write_rows_csv(
                    profile_dir / "semantic_iteration_trace.csv",
                    semantic_iteration_trace,
                )
        if self.visualize:
            from viser_utils import create_motion_control_sliders  # type: ignore[import-not-found]  # noqa: PLC0415

            robot_dof = len(self.viser_robot.get_actuated_joint_limits())

            create_motion_control_sliders(
                server=self.server,
                viser_robot=self.viser_robot,
                robot_base_frame=self.robot_base,
                motion_sequence=np.asarray(retargeted_motions)[1:],
                robot_dof=robot_dof,
                viser_object=self.viser_object,
                object_base_frame=getattr(self, "object_base", None) if self.viser_object else None,
                contains_object_in_qpos=bool(self.viser_object) and bool(self.has_dynamic_object),
                initial_fps=30,
                initial_interp_mult=2,
                loop=False,
            )

            # 4) optional: visibility toggle
            with self.server.gui.add_folder("Visibility"):
                show_meshes_cb = self.server.gui.add_checkbox("Show meshes", self.viser_robot.show_visual)

                @show_meshes_cb.on_update
                def _(_):
                    self.viser_robot.show_visual = show_meshes_cb.value
                    if self.viser_object is not None:
                        self.viser_object.show_visual = show_meshes_cb.value

        return (
            trajectory,
            obj_pts_demo_list,
            obj_pts_list,
            tetrahedra,
        )

    def solve_single_iteration(
        self,
        q_locked: np.ndarray,
        q_a_n_last: np.ndarray,
        q_t_last: np.ndarray,
        target_laplacian: np.ndarray,
        adj_list: list[list[int]],
        obj_pts_local: np.ndarray,
        foot_sticking: tuple[bool, bool],
        w_nominal_tracking: float = 0.0,
        q_a_nominal: np.ndarray | None = None,
        verbose=False,
        init_t=False,
        frame_idx: int = 0,
        vertex_residual_weights: np.ndarray | None = None,
        return_diagnostics: bool = False,
    ):
        """The main function to solve a single iteration of the DiffIK problem.
        Args:
            q_locked: the locked robot and object configuration.
            q_a_n_last: the last optimized robot configuration at current time step.
            q_t_last: the robot and object configuration at the last time step.
            foot_sticking: a sequence of booleans indicating whether the foot [left, right] is sticking to the ground.
            smpl_joints: the (possibly scaled) SMPL joint positions to match for IK.
            q_ref: the reference robot configuration.
            smpl_joints_original: the original SMPL joint positions (used for contact matching).
            obj_original: the original object pose (used for contact matching).
            init_t: the current time step is the first time step.
            frame_idx: frame index used by explicit foot lock window constraints.
        """
        assert len(q_a_n_last) == self.nq_a

        # Lock the object pose and set the current robot slice to last accepted solution
        q = np.copy(q_locked)
        q[self.q_a_indices] = q_a_n_last

        # Compute Laplacian pieces
        J_OC_dict, p_OC_dict, _ = self._calc_manipulator_jacobians(
            q, links=self.laplacian_match_links, obj_frame=(self.object_name != "ground")
        )
        robot_link_keys = list(self.laplacian_match_links.keys())
        V_r = len(robot_link_keys)
        V_o = len(obj_pts_local)
        V = V_r + V_o

        # Stack Jacobians for robot points
        J_V = np.zeros((3 * V, self.nq_a))
        for i, key in enumerate(robot_link_keys):
            J_V[3 * i : 3 * (i + 1), :] = J_OC_dict[key]

        robot_pts_local = np.array([p_OC_dict[k] for k in robot_link_keys])
        vertices = np.vstack([robot_pts_local, obj_pts_local])  # (V x 3)

        L = calculate_laplacian_matrix(vertices, adj_list)  # (V x V), EXPECT SPARSE OR SMALL
        if not sp.issparse(L):
            L = sp.csr_matrix(L)

        Kron = sp.kron(L, sp.eye(3, format="csr"), format="csr")
        J_L = Kron @ J_V

        lap0 = L @ vertices
        lap0_vec = lap0.reshape(-1)  # (3V,)
        target_lap_vec = target_laplacian.reshape(-1)  # (3V,)

        if vertex_residual_weights is None:
            w_v = (self.laplacian_weights * np.ones(V)).astype(float)  # (V,)
        else:
            semantic_weights = np.asarray(vertex_residual_weights, dtype=float).reshape(-1)
            if semantic_weights.shape != (V,):
                raise ValueError(f"vertex_residual_weights must have shape {(V,)}, got {semantic_weights.shape}")
            if np.any(semantic_weights <= 0) or not np.all(np.isfinite(semantic_weights)):
                raise ValueError("vertex_residual_weights must be finite and positive")
            w_v = self.laplacian_weights * semantic_weights
        sqrt_w3 = np.sqrt(np.repeat(w_v, 3))

        # Decision variables
        dqa = cp.Variable(len(self.q_a_indices), name="dqa")
        lap_var = cp.Variable(3 * V, name="laplacian")

        # Constraints list
        constraints = []
        original_constraint_objects: dict[str, list[Any]] = {
            "foot_sticking": [],
            "foot_lock": [],
            "nonpenetration": [],
            "self_collision": [],
            "joint_limits": [],
            "step_bound": [],
        }

        # Linear equality
        constraints += [cp.Constant(J_L[:, self.q_a_indices]) @ dqa - lap_var == -lap0_vec]

        # Foot constraints (sticking + foot lock window Z pinning)
        apply_foot_sticking = (self.q_a_init_idx < 12) and self.activate_foot_sticking
        apply_foot_lock = (self.q_a_init_idx < 12) and self.foot_lock.enable
        if apply_foot_sticking or apply_foot_lock:
            J_WF_dict, p_WF_dict, _ = self._calc_manipulator_jacobians(q, links=self.foot_links, obj_frame=False)

            # Foot sticking: constrain XY to stay near previous frame position
            if apply_foot_sticking:
                _, p_WF_t_last_dict, _ = self._calc_manipulator_jacobians(
                    q_t_last, links=self.foot_links, obj_frame=False
                )
                left_key = right_key = None
                for key in foot_sticking:
                    if key.lower().startswith("l"):
                        left_key = key
                    elif key.lower().startswith("r"):
                        right_key = key
                if left_key is None or right_key is None:
                    raise ValueError("foot_sticking must include one left* and one right* key")

                for key, J_WF in J_WF_dict.items():
                    apply_left = ("left" in key) and foot_sticking[left_key]
                    apply_right = ("right" in key) and foot_sticking[right_key]
                    if apply_left or apply_right:
                        p_lb = p_WF_t_last_dict[key] - p_WF_dict[key] - self.foot_sticking_tolerance
                        p_ub = p_lb + 2 * self.foot_sticking_tolerance  # symmetric window

                        Jxy = J_WF[:2, self.q_a_indices]  # (2 x nq_act)
                        new_constraints = [
                            Jxy @ dqa >= p_lb[:2],
                            Jxy @ dqa <= p_ub[:2],
                        ]
                        constraints += new_constraints
                        original_constraint_objects["foot_sticking"].extend(new_constraints)

            # Foot lock windows: pin Z to floor within configured frame ranges
            if apply_foot_lock:
                for key, J_WF in J_WF_dict.items():
                    z_anchor = self._is_foot_locked_in_window(key, frame_idx)
                    if z_anchor is None:
                        continue

                    z_delta = z_anchor - p_WF_dict[key][2]
                    Jz = J_WF[2, self.q_a_indices]
                    new_constraints = [
                        Jz @ dqa >= z_delta - self.foot_lock.tolerance,
                        Jz @ dqa <= z_delta + self.foot_lock.tolerance,
                    ]
                    constraints += new_constraints
                    original_constraint_objects["foot_lock"].extend(new_constraints)

        # Non-penetration constraints.  Respect the public configuration flag;
        # this is also needed for robot/object motions whose linearized contact
        # constraints are infeasible even though the objective remains solvable.
        if self.activate_obj_non_penetration:
            Js, phis = self._update_jacobians_and_phis_from_q(q)
            for key, phi in phis.items():
                Ja_n_full = Js[key]
                Ja_n = Ja_n_full[self.q_a_indices]
                rhs = -phi - self.penetration_tolerance
                new_constraint = Ja_n @ dqa >= rhs
                constraints += [new_constraint]
                original_constraint_objects["nonpenetration"].append(new_constraint)

        # Self-collision constraints
        Js_sc, phis_sc = self._compute_self_collision_constraints(frame_idx)
        for key, phi in phis_sc.items():
            Ja_n_full = Js_sc[key]
            Ja_n = Ja_n_full[self.q_a_indices]
            # Enforce: new_distance >= tolerance  =>  phi + J @ dqa >= tol
            rhs = self._self_collision_tolerance - phi
            new_constraint = Ja_n @ dqa >= rhs
            constraints += [new_constraint]
            original_constraint_objects["self_collision"].append(new_constraint)

        # Joint limits constraints (actuated)
        if self.activate_joint_limits:
            new_constraints = [
                dqa >= (self.q_a_lb - q_a_n_last),
                dqa <= (self.q_a_ub - q_a_n_last),
            ]
            constraints += new_constraints
            original_constraint_objects["joint_limits"].extend(new_constraints)

        # Step size constraints (Lorentz cone)
        step_constraint = cp.SOC(self.step_size, dqa)
        constraints += [step_constraint]
        original_constraint_objects["step_bound"].append(step_constraint)

        # Unchanged OmniRetarget objective. Semantic behavior enters only
        # through the optional per-vertex residual weights above.
        obj_terms = []

        obj_terms.append(cp.sum_squares(cp.multiply(sqrt_w3, lap_var - target_lap_vec)))

        # nominal tracking for selected indices
        if (w_nominal_tracking > 0) and (q_a_nominal is not None):
            idx = np.array(self.track_nominal_indices, dtype=int)
            if idx.size > 0:
                z = dqa[idx] - (q_a_nominal[idx] - q_a_n_last[idx])
                obj_terms.append(w_nominal_tracking * cp.sum_squares(z))

        # Q_diag cost
        Qd = np.asarray(self.Q_diag, dtype=float).reshape(-1)
        obj_terms.append(cp.sum_squares(cp.multiply(np.sqrt(Qd), dqa + q_a_n_last)))

        # Smoothness cost
        dqa_smooth = q_t_last[self.q_a_indices] - q_a_n_last
        if np.isscalar(self.smooth_weight):
            obj_terms.append(self.smooth_weight * cp.sum_squares(dqa - dqa_smooth))
        else:
            Wsmooth = np.asarray(self.smooth_weight, dtype=float)
            if Wsmooth.ndim == 1:
                obj_terms.append(cp.sum_squares(cp.multiply(np.sqrt(Wsmooth), dqa - dqa_smooth)))
            else:
                # if a full matrix was supplied, fall back to quad_form
                obj_terms.append(cp.quad_form(dqa - dqa_smooth, Wsmooth))

        omni_objective = cp.sum(obj_terms)
        problem = cp.Problem(cp.Minimize(omni_objective), constraints)

        lap_pre = lap0_vec - target_lap_vec
        pre_omni = float(np.sum(w_v.repeat(3) * lap_pre * lap_pre))
        if (w_nominal_tracking > 0) and (q_a_nominal is not None):
            idx = np.array(self.track_nominal_indices, dtype=int)
            if idx.size > 0:
                pre_omni += w_nominal_tracking * float(
                    np.sum((q_a_n_last[idx] - q_a_nominal[idx]) ** 2)
                )
        pre_omni += float(np.sum(Qd * q_a_n_last * q_a_n_last))
        if np.isscalar(self.smooth_weight):
            pre_omni += float(self.smooth_weight) * float(np.sum(dqa_smooth * dqa_smooth))
        else:
            Wsmooth = np.asarray(self.smooth_weight, dtype=float)
            if Wsmooth.ndim == 1:
                pre_omni += float(np.sum(Wsmooth * dqa_smooth * dqa_smooth))
            else:
                pre_omni += float(dqa_smooth @ Wsmooth @ dqa_smooth)
        # -------- Solve with Clarabel --------
        solver_kwargs = {"verbose": verbose}
        problem.solve(solver=cp.CLARABEL, **solver_kwargs)
        self._convex_solver_calls += 1
        if (problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)) and init_t:
            constraints = [c for c in constraints if not isinstance(c, cp.constraints.second_order.SOC)]
            problem = cp.Problem(cp.Minimize(cp.sum(obj_terms)), constraints)
            problem.solve(solver=cp.CLARABEL, **solver_kwargs)
            self._convex_solver_calls += 1

        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise RuntimeError(f"CVXPY solve failed: {problem.status}")

        dqa_star = dqa.value
        cost = problem.value

        q_star = np.copy(q)
        q_star[self.q_a_indices] = dqa_star + q_a_n_last
        q_star[3:7] /= np.linalg.norm(q_star[3:7]) + 1e-12

        if return_diagnostics:
            def _dual_magnitude(value: Any) -> float:
                if value is None:
                    return 0.0
                if isinstance(value, (list, tuple)):
                    return max((_dual_magnitude(item) for item in value), default=0.0)
                return float(np.max(np.abs(np.asarray(value, dtype=np.float64)), initial=0.0))

            active_constraint_categories = []
            for name, objects in original_constraint_objects.items():
                if name == "step_bound":
                    active = (
                        abs(float(np.linalg.norm(dqa_star)) - self.step_size)
                        <= 1e-6
                    )
                else:
                    active = any(_dual_magnitude(obj.dual_value) > 1e-7 for obj in objects)
                if active:
                    active_constraint_categories.append(name)
            diagnostics = {
                "pre_omni_objective": pre_omni,
                "post_omni_objective": float(omni_objective.value),
                "step_norm": float(np.linalg.norm(dqa_star)),
                "trust_radius": self.step_size,
                "solver_status": str(problem.status),
                "original_constraint_categories": (
                    "foot_sticking",
                    "foot_lock",
                    "nonpenetration",
                    "self_collision",
                    "joint_limits",
                    "step_bound",
                ),
                "original_constraint_counts": {
                    name: len(objects) for name, objects in original_constraint_objects.items()
                },
                "original_active_constraint_categories": tuple(active_constraint_categories),
            }
            return q_star, cost, diagnostics
        return q_star, cost

    def _is_foot_locked_in_window(self, foot_link_key: str, frame_idx: int) -> float | None:
        """Return z_floor if foot is locked at this frame, else None."""
        key_lower = foot_link_key.lower()
        side = None
        if "left" in key_lower:
            side = "left"
        elif "right" in key_lower:
            side = "right"
        if side is None:
            return None

        for i, (start, end) in enumerate(self._foot_lock_windows.get(side, ())):
            if start <= frame_idx <= end:
                return self._foot_lock_z_floors[side][i]
        return None

    def _compute_self_collision_constraints(self, frame_idx: int):
        """Compute Jacobians and distances for self-collision body pairs.

        Assumes ``mj_forward`` has already been called with the current q
        (done by ``_update_jacobians_and_phis_from_q`` which runs first).

        Returns:
            Js: dict mapping (geom_a, geom_b) -> relative Jacobian (1 x nq)
            phis: dict mapping (geom_a, geom_b) -> signed distance
        """
        if not self._self_collision_enabled:
            return {}, {}

        # Check frame windows
        if self._self_collision_windows is not None:
            if not any(start <= frame_idx <= end for start, end in self._self_collision_windows):
                return {}, {}

        m, d = self.robot_model, self.robot_data
        threshold = float(self.collision_detection_threshold)

        Js, phis = {}, {}
        fromto = np.zeros(6, dtype=float)

        if not hasattr(self, "_geom_names"):
            raise RuntimeError(
                "[SelfCollision] _geom_names not initialized. Please run _prefilter_pairs_with_mj_collision first."
            )

        _first_iter = self._sc_last_vis_frame != frame_idx
        if _first_iter:
            self._sc_last_vis_frame = frame_idx

        for geom_a, geom_b in self._self_collision_geom_pairs:
            fromto[:] = 0.0
            dist = mujoco.mj_geomDistance(m, d, geom_a, geom_b, threshold, fromto)
            if dist <= threshold:
                J_rel = self._compute_jacobian_for_contact_relative(
                    m.geom(geom_a),
                    m.geom(geom_b),
                    self._geom_names[geom_a],
                    self._geom_names[geom_b],
                    fromto,
                    dist,
                )
                key = ("self", geom_a, geom_b)
                Js[key] = J_rel
                phis[key] = float(dist)

        if _first_iter and self.visualize:
            self._draw_self_collision_geoms()

        return Js, phis

    def iterate(
        self,
        q_locked: np.ndarray,
        q_n: np.ndarray,
        q_t_last: np.ndarray,
        target_laplacian: np.ndarray,
        adj_list: list[list[int]],
        obj_pts_local: np.ndarray,
        foot_sticking: tuple[bool, bool],
        w_nominal_tracking: float = 0.0,
        q_a_nominal: np.ndarray | None = None,
        init_t: bool = False,
        n_iter: int = 10,
        min_iterations: int = 1,
        frame_idx: int = 0,
        vertex_residual_weights: np.ndarray | None = None,
        return_iterations: bool = False,
        return_diagnostics: bool = False,
        iteration_observer: Callable[[int, np.ndarray, float], None] | None = None,
    ):
        """Iterate the solver for multiple iterations."""
        if min_iterations < 1 or min_iterations > n_iter:
            raise ValueError(f"min_iterations must be in [1, n_iter], got {min_iterations} for {n_iter}")
        last_cost = np.inf
        actual_iterations = 0
        iteration_diagnostics: list[dict[str, Any]] = []
        for _ in range(n_iter):
            q_a_n_last = q_n[self.q_a_indices]
            iteration_result = self.solve_single_iteration(
                q_locked=q_locked,
                q_a_n_last=q_a_n_last,
                q_t_last=q_t_last,
                target_laplacian=target_laplacian,
                adj_list=adj_list,
                obj_pts_local=obj_pts_local,
                foot_sticking=foot_sticking,
                q_a_nominal=q_a_nominal,
                w_nominal_tracking=w_nominal_tracking,
                init_t=init_t,
                frame_idx=frame_idx,
                vertex_residual_weights=vertex_residual_weights,
                return_diagnostics=return_diagnostics,
            )
            if return_diagnostics:
                q_n, cost, diagnostics = iteration_result
                iteration_diagnostics.append(diagnostics)
            else:
                q_n, cost = iteration_result
            actual_iterations += 1
            if iteration_observer is not None:
                iteration_observer(actual_iterations, q_n, float(cost))
            if actual_iterations >= min_iterations and np.isclose(cost, last_cost):
                break
            last_cost = cost
        if return_iterations and return_diagnostics:
            return q_n, cost, actual_iterations, tuple(iteration_diagnostics)
        if return_iterations:
            return q_n, cost, actual_iterations
        if return_diagnostics:
            return q_n, cost, tuple(iteration_diagnostics)
        return q_n, cost

    def _draw_self_collision_geoms(self):
        """Draw collision cylinders for self-collision geom pairs in viser."""
        if not hasattr(self, "server") or not self._self_collision_enabled:
            return
        m, d = self.robot_model, self.robot_data
        seen_geoms: set[int] = set()
        colors = [(255, 80, 80), (80, 80, 255)]  # red for first body, blue for second
        for geom_a, geom_b in self._self_collision_geom_pairs:
            for idx, gid in enumerate([geom_a, geom_b]):
                if gid in seen_geoms:
                    continue
                seen_geoms.add(gid)
                gtype = int(m.geom_type[gid])
                if gtype not in (3, 5):  # 3 = capsule, 5 = cylinder
                    continue
                radius = float(m.geom_size[gid][0])
                half_len = float(m.geom_size[gid][1])
                cyl = trimesh.creation.capsule(radius=radius, height=2 * half_len, count=[16, 16])
                # World transform from MuJoCo data
                pos = d.geom_xpos[gid]
                rot_mat = d.geom_xmat[gid].reshape(3, 3)
                transform = np.eye(4)
                transform[:3, :3] = rot_mat
                transform[:3, 3] = pos
                cyl.apply_transform(transform)
                body_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[gid]) or ""
                self.server.scene.add_mesh_simple(
                    f"/world/sc_geom/{body_name}_g{gid}",
                    vertices=cyl.vertices.astype(np.float32),
                    faces=cyl.faces.astype(np.int32),
                    color=colors[idx % 2],
                    opacity=0.35,
                )

    def draw_q(self, q: np.ndarray):
        """Draw a single robot configuration."""
        # Update robot joint configurations
        robot_joint_positions = q[7 : 7 + self.task_constants.ROBOT_DOF]
        self.viser_robot.update_cfg(robot_joint_positions)

        # Update robot base pose using set_transform
        robot_quat = q[3:7]  # Base orientation
        robot_pos = q[:3]  # Base position

        # Update robot base frame
        self.robot_base.position = robot_pos
        self.robot_base.wxyz = robot_quat  # Assuming quaternion is in wxyz order

        # Update object pose if it exists
        if hasattr(self, "viser_object") and self.viser_object is not None:
            if self.has_dynamic_object:
                object_quat = q[-4:]
                object_pos = q[-7:-4]
            else:
                object_quat = np.asarray([1, 0, 0, 0])
                object_pos = np.zeros(3)

            # Update object base frame
            self.object_base.position = object_pos
            self.object_base.wxyz = object_quat  # Assuming quaternion is in wxyz order

    def draw_keypoints(self, p, name="keypoint", rgba=(0, 0, 1, 1)):
        """Draw keypoints in visualization."""
        if not hasattr(self, "server"):
            return None

        # Create a sphere mesh using trimesh
        sphere = trimesh.primitives.Sphere(radius=0.02)
        vertices = sphere.vertices
        faces = sphere.faces

        color = tuple(int(c * 255) for c in rgba[:3])
        opacity = float(rgba[3])

        kpts_handle_list = []

        # Draw keypoints
        if len(p.shape) == 1:
            # Single point
            kpts_handle = self.server.scene.add_mesh_simple(
                f"/{name}",
                vertices=vertices,
                faces=faces,
                position=p,
                color=color,
                opacity=opacity,
            )
            kpts_handle_list.append(kpts_handle)
        elif len(p.shape) == 2:
            # Multiple points
            kpts_handle = self.server.scene.add_batched_meshes_simple(
                f"/{name}",
                vertices=vertices,
                faces=faces,
                batched_positions=p,
                batched_wxyzs=np.tile(np.array([1, 0, 0, 0]), (p.shape[0], 1)),
                batched_colors=color,
                opacity=opacity,
            )
            kpts_handle_list.append(kpts_handle)

        return kpts_handle_list

    def visualize_motion(
        self,
        human_joint_motions,
        obj_pts_demo,
        obj_pts,
        retargeted_motions,
        tetrahedra,
        dt=1 / 30,
        visualize_tetrahedra=False,
    ):
        for i in range(len(human_joint_motions)):
            object_pts_demo = obj_pts_demo[i]
            object_pts = obj_pts[i]
            self.draw_keypoints(human_joint_motions[i, self.smplh_mapped_joint_indices], name="human")
            self.draw_keypoints(object_pts_demo, name="object_demo", rgba=(1, 0, 0, 1))
            self.draw_keypoints(object_pts, name="object", rgba=(0, 1, 0, 1))
            self.draw_q(retargeted_motions[i])
            robot_link_positions = self._get_robot_link_positions(
                retargeted_motions[i], self.laplacian_match_links.values()
            )
            self.draw_keypoints(robot_link_positions, name="robot", rgba=(0, 1, 0, 1))
            input()
            if visualize_tetrahedra:
                self.visualize_tetrahedra(
                    np.vstack(
                        [
                            human_joint_motions[i, self.smplh_mapped_joint_indices],
                            object_pts_demo,
                        ]
                    ),
                    tetrahedra[i],
                    name="human_tetrahedra",
                )
                self.visualize_tetrahedra(
                    np.vstack([robot_link_positions, object_pts]),
                    tetrahedra[i],
                    name="robot_tetrahedra",
                    rgba=(0, 1, 1, 1),
                )
            else:
                time.sleep(dt)

    def visualize_tetrahedra(self, vertices, tetrahedra, name="tetrahedra", color=(0, 0, 0, 1)):
        # Convert color to 0-255 range
        color_255 = np.array(color[:3]) * 255

        # Prepare points and colors for all edges
        points = []
        colors = []

        for tet in tetrahedra:
            for i in range(4):
                for j in range(i + 1, 4):
                    u, v = tet[i], tet[j]
                    points.extend([vertices[u], vertices[v]])
                    colors.extend([color_255, color_255])

        # Convert to numpy arrays
        points = np.array(points)
        colors = np.array(colors)

        # Add line segments for all edges at once
        self.server.scene.add_line_segments(
            f"/{name}",
            points=points,
            colors=colors,
            line_width=0.01,
        )

    def _compute_jacobian_for_contact_relative(self, geom1, geom2, geom1_name, geom2_name, fromto, dist):
        # Get closest points from fromto buffer
        pos1 = fromto[:3]  # closest point on geom1
        pos2 = fromto[3:]  # closest point on geom2

        v = pos1 - pos2
        norm_v = np.linalg.norm(v)

        if norm_v > 1e-12:
            nhat_BA_W = np.sign(dist) * (v / norm_v)
        # Degenerate: points coincide. Heuristics fallback.
        # If one side is a plane/ground, use its known normal.
        elif "ground" in geom2_name.lower():
            nhat_BA_W = np.array([0.0, 0.0, 1.0]) * (1.0 if dist >= 0 else -1.0)
        elif "ground" in geom1_name.lower():
            nhat_BA_W = np.array([0.0, 0.0, -1.0]) * (1.0 if dist >= 0 else -1.0)
        else:
            nhat_BA_W = np.array([0.0, 0.0, 0.0])

        J_bodyA = self._calc_contact_jacobian_from_point(geom1.bodyid, pos1, input_world=True)
        J_bodyB = self._calc_contact_jacobian_from_point(geom2.bodyid, pos2, input_world=True)

        # Compute relative Jacobian
        Jc = J_bodyA - J_bodyB

        return nhat_BA_W @ Jc

    def _prefilter_pairs_with_mj_collision(self, threshold: float):
        m, d = self.robot_model, self.robot_data
        ngeom = m.ngeom

        self._geom_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in range(ngeom)]

        if not hasattr(self, "_saved_margins"):
            self._saved_margins = np.empty_like(m.geom_margin)
        self._saved_margins[:] = m.geom_margin

        m.geom_margin[:] = threshold

        # Run collision. This runs broad→narrow and fills d.contact.
        mujoco.mj_collision(m, d)

        # Collect unique candidate pairs that involve at least one masked geom
        candidates = set()
        for k in range(d.ncon):
            c = d.contact[k]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 < 0 or g2 < 0:
                continue
            candidates.add((min(g1, g2), max(g1, g2)))

        # Restore margins to keep physics untouched
        m.geom_margin[:] = self._saved_margins

        return candidates

    def _update_jacobians_and_phis_from_q(self, q: np.ndarray):
        self.robot_data.qpos[:] = q

        mujoco.mj_forward(self.robot_model, self.robot_data)  # kinematics & AABBs valid

        m, d = self.robot_model, self.robot_data
        threshold = float(self.collision_detection_threshold)

        # 1) Fast prefilter via mj_collision with temporary margins
        candidates = self._prefilter_pairs_with_mj_collision(threshold)

        Js, phis = {}, {}
        fromto = np.zeros(6, dtype=float)

        # 2) Precise distance only on candidates (early-exit at threshold)
        contype, conaff = m.geom_contype, m.geom_conaffinity

        def masks_ok(g1, g2):
            if contype[g1] == 0 and conaff[g1] == 0:
                return False
            if contype[g2] == 0 and conaff[g2] == 0:
                return False
            if self.object_name in self._geom_names[g1] and "ground" in self._geom_names[g2]:
                return False
            if "ground" in self._geom_names[g1] and self.object_name in self._geom_names[g2]:
                return False
            return (
                self.object_name in self._geom_names[g1]
                or self.object_name in self._geom_names[g2]
                or "ground" in self._geom_names[g1]
                or "ground" in self._geom_names[g2]
            )

        for g1, g2 in candidates:
            # Optional: keep your own filters here (e.g., skip object-ground, only keep interaction with object/ground)
            if not masks_ok(g1, g2):
                continue

            fromto[:] = 0.0
            dist = mujoco.mj_geomDistance(m, d, g1, g2, threshold, fromto)
            if dist <= threshold:
                J_rel = self._compute_jacobian_for_contact_relative(
                    m.geom(g1), m.geom(g2), self._geom_names[g1], self._geom_names[g2], fromto, dist
                )
                Js[(g1, g2)] = J_rel
                phis[(g1, g2)] = float(dist)

                # For debug
                # self.draw_mesh_pair_with_contact(self.robot_model, self.robot_data, g1, g2,   \
                #     self._geom_names[g1], self._geom_names[g2], fromto=fromto)

        return Js, phis

    def _world_to_body_frame(self, p_w: np.ndarray, body_idx: int) -> np.ndarray:
        """Transform point from world frame to body frame."""
        p_w = np.asarray(p_w).reshape(3)
        body_pos = self.robot_data.xpos[body_idx].reshape(3)
        body_mat = self.robot_data.xmat[body_idx].reshape(3, 3)
        return body_mat.T @ (p_w - body_pos)

    def _get_geometry_name(self, geom_id: int) -> str:
        """Get geometry name from ID."""
        return mujoco.mj_id2name(self.robot_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)

    def _build_transform_qdot_to_qvel_fast(self, use_world_omega=True):
        """
        Return T(q) (nv x nq) such that v = T(q) @ qdot.
        - Free root: qpos=[x,y,z, qw,qx,qy,qz], qvel=[vx,vy,vz, ωx,ωy,ωz]
        where ω and v are WORLD-expressed in MuJoCo.
        - 23 hinge joints: v = qdot.

        If use_world_omega=False, uses BODY-omega mapping (for debugging).
        """
        nq, nv = self.robot_model.nq, self.robot_model.nv
        T = np.zeros((nv, nq), dtype=float)

        # ---- root free joint (assumed joint 0) ----
        j0 = 0
        assert self.robot_model.jnt_type[j0] == mujoco.mjtJoint.mjJNT_FREE
        qadr = self.robot_model.jnt_qposadr[j0]  # 0
        dadr = self.robot_model.jnt_dofadr[j0]  # 0

        # Linear block: v_lin = xyz_dot
        T[dadr : dadr + 3, qadr : qadr + 3] = np.eye(3)

        # Angular block: ω_* = 2 * E_*(q) * quat_dot
        w, x, y, z = self.robot_data.qpos[qadr + 3 : qadr + 7]

        def get_e_world(qw, qx, qy, qz):
            return np.array(
                [
                    [-qx, qw, qz, -qy],
                    [-qy, -qz, qw, qx],
                    [-qz, qy, -qx, qw],
                ]
            )

        def get_e_body(qw, qx, qy, qz):
            return np.array(
                [
                    [-qx, qw, -qz, qy],
                    [-qy, qz, qw, -qx],
                    [-qz, -qy, qx, qw],
                ]
            )

        E_fn = get_e_world if use_world_omega else get_e_body

        # ---- FREE joint #1 (human/root): use model addresses, but this should be the first joint ----
        j_free1 = 0
        assert self.robot_model.jnt_type[j_free1] == mujoco.mjtJoint.mjJNT_FREE
        qadr1 = int(self.robot_model.jnt_qposadr[j_free1])  # expect 0
        dadr1 = int(self.robot_model.jnt_dofadr[j_free1])  # start of its 6 qvel dofs

        qw, qx, qy, qz = self.robot_data.qpos[qadr1 + 3 : qadr1 + 7]
        E1 = 2.0 * E_fn(qw, qx, qy, qz)
        # linear-first: v_W = rdot, ω_W = 2E(q) * quat_dot
        T[dadr1 + 0 : dadr1 + 3, qadr1 + 0 : qadr1 + 3] = np.eye(3)  # v block
        T[dadr1 + 3 : dadr1 + 6, qadr1 + 3 : qadr1 + 7] = E1  # ω block

        if self.has_dynamic_object:
            # ---- FREE joint #2 (object): assume it's the last FREE joint; fill its 6x7 block ----
            # Find it by type (safer than hardcoding tail indices)
            free_joints = [
                j for j in range(self.robot_model.njnt) if self.robot_model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
            ]
            assert len(free_joints) >= 2, "Expected two FREE joints (human + object)."
            j_free2 = free_joints[1]  # second FREE joint
            qadr2 = int(self.robot_model.jnt_qposadr[j_free2])  # expect nq-7
            dadr2 = int(self.robot_model.jnt_dofadr[j_free2])  # its 6 qvel dofs (often at nv-6)

            qw, qx, qy, qz = self.robot_data.qpos[qadr2 + 3 : qadr2 + 7]
            E2 = 2.0 * E_fn(qw, qx, qy, qz)
            T[dadr2 + 0 : dadr2 + 3, qadr2 + 0 : qadr2 + 3] = np.eye(3)  # v block
            T[dadr2 + 3 : dadr2 + 6, qadr2 + 3 : qadr2 + 7] = E2  # ω block

        # ---- remaining hinge/slide joints: v = qdot ----
        for j in range(1, self.robot_model.njnt):
            jt = self.robot_model.jnt_type[j]
            if jt in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                qa = self.robot_model.jnt_qposadr[j]
                da = self.robot_model.jnt_dofadr[j]
                T[da, qa] = 1.0
            elif jt == mujoco.mjtJoint.mjJNT_BALL:
                raise NotImplementedError("BALL joint block not implemented.")

        return T

    def _calc_contact_jacobian_from_point(self, body_idx: int, p_body: np.ndarray, input_world=False):
        """
        Translational Jacobian J(q) (3 x nq) such that
        v_point_world = J(q) @ qdot.

        Fast analytic version: J_qdot = J_v @ T(q)
        """

        p_body = np.asarray(p_body, dtype=float).reshape(3)

        # 1) Make sure kinematics are current once
        mujoco.mj_forward(self.robot_model, self.robot_data)

        # 2) World point (3,1) for mj_jac
        R_WB = self.robot_data.xmat[body_idx].reshape(3, 3)
        p_WB = self.robot_data.xpos[body_idx]

        if input_world:
            p_W = p_body.astype(np.float64).reshape(3, 1)
        else:
            p_W = (p_WB + R_WB @ p_body).astype(np.float64).reshape(3, 1)

        # 3) J_v: translational Jacobian wrt generalized velocities (3 x nv)
        Jp = np.zeros((3, self.robot_model.nv), dtype=np.float64, order="C")
        Jr = np.zeros((3, self.robot_model.nv), dtype=np.float64, order="C")
        mujoco.mj_jac(self.robot_model, self.robot_data, Jp, Jr, p_W, int(body_idx))  # Jp = J_v

        T = self._build_transform_qdot_to_qvel_fast()

        return Jp @ T

    def _calc_manipulator_jacobians(
        self,
        q: np.ndarray,
        links: dict[str, str],
        obj_frame: bool = False,
        point_offsets: np.ndarray | None = None,
    ):
        """Compute position-based Jacobians using MuJoCo."""
        J_XC_dict = {}
        p_XC_dict = {}

        if obj_frame:
            if self.has_dynamic_object:
                obj_quat = q[-4:]
                obj_pos = q[-7:-4]
                obj_rot = Rotation.from_quat([obj_quat[1], obj_quat[2], obj_quat[3], obj_quat[0]]).as_matrix()
                obj_rot_inv = obj_rot.T
            else:
                obj_rot = Rotation.from_quat([0, 0, 0, 1]).as_matrix()
                obj_rot_inv = obj_rot.T
                obj_pos = np.zeros(3)

        q_mujoco = q.copy()
        self.robot_data.qpos[:] = q_mujoco

        mujoco.mj_forward(self.robot_model, self.robot_data)

        for name, link_name in links.items():
            body_id = mujoco.mj_name2id(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, link_name)

            if point_offsets is not None:
                pC_B = point_offsets
            else:
                pC_B = np.zeros(3)

            J = self._calc_contact_jacobian_from_point(body_id, pC_B)
            pos_world = self.robot_data.xpos[body_id]

            if obj_frame:
                p_XC = obj_rot_inv @ (pos_world - obj_pos)
                J_XC = obj_rot_inv @ J
            else:
                p_XC = pos_world
                J_XC = J

            # Store reduced Jacobian and position with hard copies to avoid aliasing
            J_XC_dict[name] = np.array(J_XC[:, self.q_a_indices], dtype=float, copy=True)  # FIX (copy)
            p_XC_dict[name] = np.array(p_XC, dtype=float, copy=True)

        P_WO = {"position": obj_pos, "rotation": obj_rot} if obj_frame else None

        return J_XC_dict, p_XC_dict, P_WO

    def _get_robot_link_positions(self, q, link_names):
        """Get robot link positions for given configuration using Mujoco."""
        mujoco_q = q.copy()

        # Set the configuration
        if mujoco_q.shape != self.robot_data.qpos.shape:
            self.robot_data.qpos = mujoco_q[:-7]  # Exclude object information from q
        else:
            self.robot_data.qpos = mujoco_q
        # Forward kinematics to update all positions
        mujoco.mj_forward(self.robot_model, self.robot_data)

        robot_link_positions = []

        for link_name in link_names:
            # Get body ID from name
            body_id = mujoco.mj_name2id(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, link_name)
            if body_id == -1:
                raise ValueError(f"Body {link_name} not found in Mujoco model")

            # Get position in world frame
            # xpos gives us the position of the body's center of mass in world coordinates
            pos = self.robot_data.xpos[body_id].copy()
            robot_link_positions.append(pos)

        return np.array(robot_link_positions)
