"""Opt-in surface contact targets extracted from one retargeted motion.

Targets are object-local and effectors are points on the training robot's
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


def contact_position_reward(hand_points, object_points, active, sigma):
    """One exponential of the mean active-hand squared distance; zero off-contact."""
    squared = (hand_points - object_points).square().sum(dim=-1)
    count = active.sum(dim=-1)
    mse = (squared * active).sum(dim=-1) / count.clamp_min(1)
    reward = torch.exp(-mse / sigma**2) * (count > 0)
    return reward, mse.sqrt()


class ReferenceContactPosition(RewardTermBase):
    """Only semantic_adaptive may opt into this additional contact reward."""

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
            local_hand = data["hand_points_local"].copy()
            local_object = data["object_points_local"].copy()
            active = data["active"].copy()
        if metadata.get("schema") != "holosoma.reference_surface_contacts.v1":
            raise ValueError("Unsupported reference contact schema")
        motion_file = Path(command.motion_cfg.motion_file)
        if hashlib.sha256(motion_file.read_bytes()).hexdigest() != metadata["motion_sha256"]:
            raise ValueError("Contact targets were generated for a different motion")
        frames = command.motion.time_step_total
        fps = float(np.asarray(command.motion.fps).item())
        if frames != len(active) or not np.isclose(fps, metadata["fps"]):
            raise ValueError("Contact frame count/FPS differs from motion (transitions unsupported)")
        if (not names or len(set(names)) != len(names)
                or active.shape != (frames, len(names)) or active.dtype != np.bool_
                or local_hand.shape != (frames, len(names), 3)
                or local_object.shape != local_hand.shape
                or not np.isfinite(local_hand).all() or not np.isfinite(local_object).all()
                or not active.any()):
            raise ValueError("Invalid contact arrays")
        for file, digest in metadata["geometry_sha256"].items():
            if hashlib.sha256(Path(file).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Contact geometry changed: {file}")
        self.body_indexes = torch.tensor([env.simulator.body_names.index(n) for n in names],
                                         device=env.device, dtype=torch.long)
        self.hand = torch.as_tensor(local_hand, device=env.device, dtype=torch.float32)
        self.object = torch.as_tensor(local_object, device=env.device, dtype=torch.float32)
        self.active = torch.as_tensor(active, device=env.device)
        self.command = command
        self.ready = True

    def __call__(self, env, **kwargs):
        if not self.ready:
            self._initialize(env)
        command = self.command
        step = command.time_steps
        position = env.simulator._rigid_body_pos[:, self.body_indexes]
        rotation = env.simulator._rigid_body_rot[:, self.body_indexes]
        hand_world = position + quat_apply(rotation, self.hand[step], w_last=True)
        # Read actual object pose once. These are actor-frame coordinates, not COM.
        obj = command._simulator_object_states()
        object_rotation = obj[:, None, 3:7].expand(-1, len(self.body_indexes), -1)
        object_world = obj[:, None, :3] + quat_apply(object_rotation, self.object[step], w_last=True)
        active = self.active[step]
        reward, error = contact_position_reward(hand_world, object_world, active, self.sigma)
        command.metrics["contact/active_fraction"] = active.float().mean(dim=-1)
        command.metrics["contact/position_error_m_gated"] = error
        command.metrics["contact/position_reward_raw"] = reward
        return reward

    def reset(self, env_ids=None):
        pass
