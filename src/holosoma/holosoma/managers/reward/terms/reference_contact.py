"""Semantic-key-part surface contact targets transferred from retargeting.

Targets are object-local and robot anchors are points on the training robot's
collision surface. No nearest-point search or file I/O occurs per step.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from holosoma.managers.reward.base import RewardTermBase
from holosoma.utils.rotations import quat_apply


def contact_position_reward(body_points, object_points, active, sigma):
    """Weighted contact error with a smooth source-activation envelope.

    Boolean ``active`` values retain the original behavior. Float values in
    [0, 1] additionally scale the reward during the contact onset/release ramp.
    """
    squared = (body_points - object_points).square().sum(dim=-1)
    weight = active.to(dtype=squared.dtype)
    positive = weight > 0
    count = positive.sum(dim=-1)
    total_weight = weight.sum(dim=-1)
    mse = (squared * weight).sum(dim=-1) / total_weight.clamp_min(1e-8)
    strength = total_weight / count.clamp_min(1)
    reward = torch.exp(-mse / sigma**2) * strength * (count > 0)
    return reward, mse.sqrt()


def contact_force_gate(object_on_body_forces, outward_normals, activation, minimum_normal_force):
    """Require current compressive object contact on EVERY active target.

    Force vectors must come from an object-filtered sensor. The outward
    object normal is the direction of the object's reaction force on the
    contacting robot part. Tangential/reversed forces cannot open the gate.
    A hard gate retains the position objective and never rewards force growth.
    """
    normal_force = (object_on_body_forces * outward_normals).sum(dim=-1).clamp_min(0.0)
    finite = torch.isfinite(object_on_body_forces).all(dim=-1) & torch.isfinite(normal_force)
    normal_force = torch.where(finite, normal_force, 0.0)
    loaded = finite & (normal_force >= minimum_normal_force)
    active = activation > 0
    gate = active.any(dim=-1) & (loaded | ~active).all(dim=-1)
    return gate, normal_force, loaded & active


def contact_force_shaped_reward(body_points, object_points, activation, normal_force,
                                sigma, approach_fraction, force_scale_n):
    """Small approach signal plus bounded, independent per-part force feedback.

    The scale is a saturation scale, not a binary grasp threshold. A part
    cannot borrow another part's force; inactive parts contribute nothing.
    """
    weight = activation.to(dtype=body_points.dtype)
    active = weight > 0
    count = active.sum(dim=-1).clamp_min(1)
    squared = (body_points - object_points).square().sum(dim=-1)
    proximity = torch.exp(-squared / sigma**2)
    pressure = torch.where(torch.isfinite(normal_force), normal_force.clamp_min(0), 0.0)
    force_fraction = (pressure / force_scale_n).clamp(max=1.0)
    approach = (weight * proximity * approach_fraction).sum(dim=-1) / count
    bonus = (weight * proximity * (1 - approach_fraction) * force_fraction).sum(dim=-1) / count
    return approach + bonus, approach, bonus, force_fraction


class ReferenceContactPosition(RewardTermBase):
    """Track active contact anchors for semantic key parts only."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.sigma = float(cfg.params.get("sigma", 0.05))
        if not np.isfinite(self.sigma) or self.sigma <= 0:
            raise ValueError("Contact sigma must be finite and positive")
        self.ready = False  # Reward manager is constructed before command manager.

    def _initialize(self, env):
        command = env.command_manager.get_state("motion_command")
        if command.motion_cfg.sampling_mode != "semantic_adaptive":
            raise ValueError("ReferenceContactPosition requires semantic_adaptive")
        if command.motion.num_motions != 1:
            raise ValueError("Reference contacts require a single motion")
        path = Path(self.cfg.params["contact_file"])
        expected = self.cfg.params["contact_sha256"]
        if not expected or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Contact artifact SHA256 mismatch")
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"].item()))
            names = data["body_names"].tolist()
            schema = metadata.get("schema")
            if schema != "holosoma.reference_surface_contacts.v2":
                raise ValueError("Unsupported reference contact schema")
            local_body = data["body_points_local"].copy()
            local_object = data["object_points_local"].copy()
            active = data["active"].copy()
            activation = data["activation"].copy() if "activation" in data.files else active.astype(np.float32)
            local_normals = data["object_normals_local"].copy() if "object_normals_local" in data.files else None
        motion_file = Path(command.motion_cfg.motion_file)
        if hashlib.sha256(motion_file.read_bytes()).hexdigest() != metadata["motion_sha256"]:
            raise ValueError("Contact targets were generated for a different motion")
        semantic_file = Path(command.motion_cfg.semantic_file)
        if hashlib.sha256(semantic_file.read_bytes()).hexdigest() != metadata["semantic_sha256"]:
            raise ValueError("Contact targets were generated for a different semantic plan")
        frames = command.motion.time_step_total
        fps = float(np.asarray(command.motion.fps).item())
        if frames != len(active) or not np.isclose(fps, metadata["fps"]):
            raise ValueError("Contact frame count/FPS differs from motion (transitions unsupported)")
        if (
            not names
            or not all(isinstance(name, str) and name for name in names)
            or active.shape != (frames, len(names))
            or active.dtype != np.bool_
            or local_body.shape != (frames, len(names), 3)
            or local_object.shape != local_body.shape
            or activation.shape != active.shape
            or not np.issubdtype(activation.dtype, np.floating)
            or not np.isfinite(activation).all()
            or np.any((activation < 0) | (activation > 1))
            or not np.array_equal(active, activation > 0)
            or not np.isfinite(local_body).all()
            or not np.isfinite(local_object).all()
            or not active.any()
        ):
            raise ValueError("Invalid contact arrays")
        for file, digest in metadata["geometry_sha256"].items():
            if hashlib.sha256(Path(file).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Contact geometry changed: {file}")
        self.body_indexes = torch.tensor(
            [env.simulator.body_names.index(n) for n in names], device=env.device, dtype=torch.long
        )
        self.body = torch.as_tensor(local_body, device=env.device, dtype=torch.float32)
        self.object = torch.as_tensor(local_object, device=env.device, dtype=torch.float32)
        self.active = torch.as_tensor(active, device=env.device)
        self.activation = torch.as_tensor(activation, device=env.device, dtype=torch.float32)
        self.body_names = names
        self.object_normals = None
        if local_normals is not None:
            if (
                local_normals.shape != local_object.shape
                or not np.isfinite(local_normals).all()
                or not np.allclose(np.linalg.norm(local_normals, axis=-1), 1.0, atol=1e-4)
            ):
                raise ValueError("Object contact normals must be finite unit vectors matching anchors")
            self.object_normals = torch.as_tensor(local_normals, device=env.device, dtype=torch.float32)
        self.command = command
        self.ready = True

    def _world_points(self, env):
        if not self.ready:
            self._initialize(env)
        command = self.command
        step = command.time_steps
        position = env.simulator._rigid_body_pos[:, self.body_indexes]
        rotation = env.simulator._rigid_body_rot[:, self.body_indexes]
        body_world = position + quat_apply(rotation, self.body[step], w_last=True)
        # Read actual object pose once. These are actor-frame coordinates, not COM.
        obj = command._simulator_object_states()
        object_rotation = obj[:, None, 3:7].expand(-1, len(self.body_indexes), -1)
        object_world = obj[:, None, :3] + quat_apply(object_rotation, self.object[step], w_last=True)
        return step, body_world, object_world, object_rotation

    def _position_reward(self, step, body_world, object_world):
        command = self.command
        active = self.active[step]
        activation = self.activation[step]
        reward, error = contact_position_reward(body_world, object_world, activation, self.sigma)
        command.metrics["contact/active_fraction"] = active.float().mean(dim=-1)
        command.metrics["contact/activation_mean"] = activation.mean(dim=-1)
        command.metrics["contact/position_error_m_gated"] = error
        command.metrics["contact/position_reward_raw"] = reward
        return reward

    def __call__(self, env, **kwargs):
        step, body_world, object_world, _ = self._world_points(env)
        return self._position_reward(step, body_world, object_world)

    def reset(self, env_ids=None):
        pass


class ReferenceContactForce(ReferenceContactPosition):
    """Position reward gated by current, target-object-filtered contact force."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.minimum_normal_force = float(cfg.params.get("minimum_normal_force_n", 1.0))
        self.object_name = str(cfg.params.get("object_name", "object"))
        if not np.isfinite(self.minimum_normal_force) or self.minimum_normal_force <= 0:
            raise ValueError("minimum_normal_force_n must be finite and strictly positive")

    def _initialize(self, env):
        super()._initialize(env)
        self.ready = False
        if self.object_normals is None:
            raise ValueError("Force-gated contacts require object_normals_local in the contact artifact")
        if not callable(getattr(env.simulator, "get_object_contact_forces", None)):
            raise ValueError("Force-gated contacts require target-object-filtered contact sensors")
        if getattr(self.command, "object_name", self.object_name) != self.object_name:
            raise ValueError("Contact force sensors must filter the motion command's tracked object")
        self.ready = True

    def __call__(self, env, **kwargs):
        step, body_world, object_world, object_rotation = self._world_points(env)
        position_reward = self._position_reward(step, body_world, object_world)
        normal_world = quat_apply(object_rotation, self.object_normals[step], w_last=True)
        forces = env.simulator.get_object_contact_forces(self.body_names, self.object_name)
        activation = self.activation[step]
        gate, normal_force, loaded = contact_force_gate(
            forces, normal_world, activation, self.minimum_normal_force,
        )
        reward = torch.where(gate, position_reward, torch.zeros_like(position_reward))
        active = activation > 0
        count = active.sum(dim=-1).clamp_min(1)
        metrics = self.command.metrics
        metrics["contact/force_gate"] = gate.float()
        metrics["contact/loaded_part_fraction"] = loaded.sum(dim=-1).float() / count
        force_norm = torch.linalg.vector_norm(forces, dim=-1)
        metrics["contact/object_force_norm_mean_n"] = (force_norm * active).sum(dim=-1) / count
        metrics["contact/normal_force_mean_n"] = (normal_force * active).sum(dim=-1) / count
        minimum = normal_force.masked_fill(~active, float("inf")).min(dim=-1).values
        metrics["contact/normal_force_min_n"] = torch.where(active.any(dim=-1), minimum, 0.0)
        metrics["contact/force_reward_raw"] = reward
        return reward


class ReferenceContactForceShaped(ReferenceContactForce):
    """Dense approach shaping, amplified by actual force on each key part.

    Explicitly opt-in: frozen hard-gated checkpoints retain their old reward.
    No actor observation, tracking reward or sampler setting is changed.
    """

    def __init__(self, cfg, env):
        ReferenceContactPosition.__init__(self, cfg, env)
        self.object_name = str(cfg.params.get("object_name", "object"))
        self.approach_fraction = float(cfg.params.get("approach_fraction", 0.1))
        self.force_scale_n = float(cfg.params.get("force_scale_n", 4.0))
        if not np.isfinite(self.approach_fraction) or not 0 < self.approach_fraction < 1:
            raise ValueError("approach_fraction must be finite and strictly between zero and one")
        if not np.isfinite(self.force_scale_n) or self.force_scale_n <= 0:
            raise ValueError("force_scale_n must be finite and strictly positive")

    def __call__(self, env, **kwargs):
        step, body_world, object_world, object_rotation = self._world_points(env)
        self._position_reward(step, body_world, object_world)  # Retain position diagnostics.
        normals = quat_apply(object_rotation, self.object_normals[step], w_last=True)
        forces = env.simulator.get_object_contact_forces(self.body_names, self.object_name)
        activation = self.activation[step]
        _, pressure, _ = contact_force_gate(forces, normals, activation, self.force_scale_n)
        reward, approach, bonus, fraction = contact_force_shaped_reward(
            body_world, object_world, activation, pressure,
            self.sigma, self.approach_fraction, self.force_scale_n,
        )
        active = activation > 0
        count = active.sum(dim=-1).clamp_min(1)
        metrics = self.command.metrics
        metrics["contact/approach_reward_raw"] = approach
        metrics["contact/force_bonus_raw"] = bonus
        metrics["contact/force_reward_raw"] = reward
        metrics["contact/loaded_part_fraction"] = ((pressure > 1e-6) & active).sum(dim=-1) / count
        metrics["contact/force_amplification_fraction"] = (fraction * active).sum(dim=-1) / count
        metrics["contact/normal_force_mean_n"] = (pressure * active).sum(dim=-1) / count
        finite_forces = torch.where(torch.isfinite(forces).all(dim=-1, keepdim=True), forces, 0.0)
        magnitude = torch.linalg.vector_norm(finite_forces, dim=-1)
        metrics["contact/object_force_norm_mean_n"] = (magnitude * active).sum(dim=-1) / count
        return reward
