"""Measure bad-tracking causes after training-style WBT resets.

Load this module with ``eval_agent.py --import-file``.  Evaluation is switched
back to training-style phase sampling and resets, while the requested initial
pose noise scale is applied.  The policy, rewards, termination thresholds, and
physics are left unchanged.

Environment variables:

``BAD_TRACKING_DIAG_OUTPUT``
    Output ``.npz`` path.
``BAD_TRACKING_DIAG_NOISE_SCALE``
    Initial-pose noise scale (normally ``1`` or ``0``).
``BAD_TRACKING_DIAG_ALIGN_OBJECT_XY``
    If true, retain all initialization noise but give the object the same XY
    translation as the robot root.  This removes only their independent XY
    displacement.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from holosoma.agents.ppo.ppo import PPO
from holosoma.envs.base_task.base_task import BaseTask
from holosoma.managers.command.terms.wbt import MotionCommand
from holosoma.utils.rotations import quat_error_magnitude
from holosoma.utils.safe_torch_import import torch


OUTPUT_PATH = Path(
    os.environ.get(
        "BAD_TRACKING_DIAG_OUTPUT",
        "diagnostics/reset_contact/bad_tracking_training_noise.npz",
    )
).resolve()
NOISE_SCALE = float(os.environ.get("BAD_TRACKING_DIAG_NOISE_SCALE", "1"))
ALIGN_OBJECT_XY = os.environ.get("BAD_TRACKING_DIAG_ALIGN_OBJECT_XY", "0").lower() in {
    "1",
    "true",
    "yes",
}
TRACE_ENV0 = os.environ.get("BAD_TRACKING_DIAG_TRACE_ENV0", "0").lower() in {"1", "true", "yes"}


def _trace(owner: Any, label: str, **values: Any) -> None:
    if not TRACE_ENV0:
        return
    count = int(getattr(owner, "_bad_tracking_diag_trace_count", 0))
    if count >= 80:
        return
    owner._bad_tracking_diag_trace_count = count + 1
    rendered = " ".join(f"{name}={value}" for name, value in values.items())
    print(f"BAD_TRACKING_TRACE {count:03d} {label} {rendered}", flush=True)


def _ensure_buffers(command: MotionCommand) -> None:
    n = command.num_envs
    device = command.device
    if not hasattr(command, "_bad_tracking_diag_age"):
        command._bad_tracking_diag_age = torch.zeros(n, dtype=torch.long, device=device)
        command._bad_tracking_diag_start_motion_step = torch.zeros(n, dtype=torch.long, device=device)
        command._bad_tracking_diag_start_object_z = torch.zeros(n, device=device)
        command._bad_tracking_diag_object_offset = torch.zeros(n, 3, device=device)
        command._bad_tracking_diag_relative_offset = torch.zeros(n, 3, device=device)
        command._bad_tracking_diag_reset_episode_age = torch.zeros(n, dtype=torch.long, device=device)
        command._bad_tracking_diag_reset_after_bad = torch.zeros(n, dtype=torch.bool, device=device)
        command._bad_tracking_diag_command_step_after_reset = torch.zeros(n, dtype=torch.bool, device=device)
        command._bad_tracking_diag_prephysics_body_z_error = torch.full((n, 4), torch.nan, device=device)
        command._bad_tracking_diag_prephysics_target_body_z = torch.full((n, 4), torch.nan, device=device)
        command._bad_tracking_diag_prephysics_actual_body_z = torch.full((n, 4), torch.nan, device=device)


def _to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy().copy()


if not getattr(MotionCommand, "_bad_tracking_ab_instrumented", False):
    _original_motion_reset = MotionCommand.reset
    _original_motion_step = MotionCommand.step

    def _reset_with_requested_noise(self: MotionCommand, env_ids: torch.Tensor | None) -> None:
        env_ids = self._ensure_index_tensor(env_ids)
        includes_env0 = bool(torch.any(env_ids == 0))
        if includes_env0:
            _trace(
                self,
                "reset_before",
                episode_age=int(self._env.episode_length_buf[0]),
                terminated=bool(self._env.termination_manager.terminated[0]),
                motion_step=int(self.time_steps[0]),
                target_z_max=float(torch.abs(self.body_pos_relative_w[0]).max()),
            )
        saved_scale = float(self.init_pose_cfg.overall_noise_scale)
        object.__setattr__(self.init_pose_cfg, "overall_noise_scale", NOISE_SCALE)
        try:
            _original_motion_reset(self, env_ids)
        finally:
            object.__setattr__(self.init_pose_cfg, "overall_noise_scale", saved_scale)

        if env_ids.numel() == 0 or not self.motion.has_object:
            return
        _ensure_buffers(self)
        robot_offset = self.robot_root_pos_w[env_ids] - self.root_pos_w[env_ids]
        if ALIGN_OBJECT_XY:
            object_states = self._simulator_object_states()[env_ids].clone()
            object_states[:, :2] = self.object_pos_w[env_ids, :2] + robot_offset[:, :2]
            self._env.simulator.set_actor_states([self.object_name], env_ids, object_states)
        actual_object_pos = self.simulator_object_pos_w[env_ids]
        object_offset = actual_object_pos - self.object_pos_w[env_ids]
        self._bad_tracking_diag_age[env_ids] = 0
        self._bad_tracking_diag_start_motion_step[env_ids] = self.time_steps[env_ids]
        self._bad_tracking_diag_start_object_z[env_ids] = self.object_pos_w[env_ids, 2]
        self._bad_tracking_diag_object_offset[env_ids] = object_offset
        self._bad_tracking_diag_relative_offset[env_ids] = object_offset - robot_offset
        self._bad_tracking_diag_reset_episode_age[env_ids] = self._env.episode_length_buf[env_ids]
        self._bad_tracking_diag_reset_after_bad[env_ids] = self._env.termination_manager.terminated[env_ids]
        self._bad_tracking_diag_command_step_after_reset[env_ids] = False
        self._bad_tracking_diag_prephysics_body_z_error[env_ids] = torch.nan
        self._bad_tracking_diag_prephysics_target_body_z[env_ids] = torch.nan
        self._bad_tracking_diag_prephysics_actual_body_z[env_ids] = torch.nan
        if includes_env0:
            _trace(
                self,
                "reset_after",
                episode_age=int(self._env.episode_length_buf[0]),
                terminated=bool(self._env.termination_manager.terminated[0]),
                motion_step=int(self.time_steps[0]),
                target_z_max=float(torch.abs(self.body_pos_relative_w[0]).max()),
            )

    def _step_with_post_reset_snapshot(self: MotionCommand) -> None:
        _trace(
            self,
            "command_step_before",
            episode_age=int(self._env.episode_length_buf[0]),
            motion_step=int(self.time_steps[0]),
            diag_age=int(self._bad_tracking_diag_age[0]) if hasattr(self, "_bad_tracking_diag_age") else -1,
            target_z_max=float(torch.abs(self.body_pos_relative_w[0]).max()),
        )
        _original_motion_step(self)
        _ensure_buffers(self)
        fresh = self._bad_tracking_diag_age == 0
        if not torch.any(fresh):
            return
        bad_tracking = self._env.termination_manager._term_instances.get("bad_tracking")
        if bad_tracking is None:
            return
        body_idx = bad_tracking.bad_motion_body_pos_body_indexes
        target_z = self.body_pos_relative_w[:, body_idx, 2]
        actual_z = self.robot_body_pos_w[:, body_idx, 2]
        self._bad_tracking_diag_command_step_after_reset[fresh] = True
        self._bad_tracking_diag_prephysics_target_body_z[fresh] = target_z[fresh]
        self._bad_tracking_diag_prephysics_actual_body_z[fresh] = actual_z[fresh]
        self._bad_tracking_diag_prephysics_body_z_error[fresh] = torch.abs(target_z[fresh] - actual_z[fresh])
        _trace(
            self,
            "command_step_after",
            episode_age=int(self._env.episode_length_buf[0]),
            motion_step=int(self.time_steps[0]),
            diag_age=int(self._bad_tracking_diag_age[0]),
            target_z_max=float(torch.abs(self.body_pos_relative_w[0]).max()),
            body_z_error_max=float(torch.abs(target_z[0] - actual_z[0]).max()),
        )

    MotionCommand.reset = _reset_with_requested_noise
    MotionCommand.step = _step_with_post_reset_snapshot
    MotionCommand._bad_tracking_ab_instrumented = True


if not getattr(BaseTask, "_bad_tracking_ab_instrumented", False):
    _original_check_termination = BaseTask._check_termination

    def _check_termination_with_reasons(self: BaseTask) -> None:
        _original_check_termination(self)
        command = self.command_manager.get_state("motion_command")
        if command is None or not getattr(command.motion, "has_object", False):
            return
        _ensure_buffers(command)
        command._bad_tracking_diag_age += 1
        _trace(
            command,
            "termination_check",
            episode_age=int(self.episode_length_buf[0]),
            motion_step=int(command.time_steps[0]),
            diag_age=int(command._bad_tracking_diag_age[0]),
            terminated=bool(self.termination_manager.terminated[0]),
            target_z_max=float(torch.abs(command.body_pos_relative_w[0]).max()),
            body_z_error_max=float(
                torch.abs(command.body_pos_relative_w[0, :, 2] - command.robot_body_pos_w[0, :, 2]).max()
            ),
        )

        bad_tracking = self.termination_manager._term_instances.get("bad_tracking")
        if bad_tracking is None:
            return
        done = self.termination_manager.terminated
        if not torch.any(done):
            return

        body_idx = bad_tracking.bad_motion_body_pos_body_indexes
        body_z_error = torch.abs(
            command.body_pos_relative_w[:, body_idx, 2] - command.robot_body_pos_w[:, body_idx, 2]
        ).max(dim=-1).values
        fields = {
            "segment_age_steps": command._bad_tracking_diag_age,
            "episode_age_steps": self.episode_length_buf,
            "start_motion_step": command._bad_tracking_diag_start_motion_step,
            "failure_motion_step": command.time_steps,
            "start_reference_object_z_m": command._bad_tracking_diag_start_object_z,
            "start_object_offset_xyz_m": command._bad_tracking_diag_object_offset,
            "start_relative_offset_xyz_m": command._bad_tracking_diag_relative_offset,
            "reset_episode_age_at_call": command._bad_tracking_diag_reset_episode_age,
            "reset_after_bad_tracking": command._bad_tracking_diag_reset_after_bad,
            "command_step_after_reset": command._bad_tracking_diag_command_step_after_reset,
            "prephysics_body_z_error_m": command._bad_tracking_diag_prephysics_body_z_error,
            "prephysics_target_body_z_m": command._bad_tracking_diag_prephysics_target_body_z,
            "prephysics_actual_body_z_m": command._bad_tracking_diag_prephysics_actual_body_z,
            "failure_object_pos_error_m": torch.linalg.vector_norm(
                command.object_pos_w - command.simulator_object_pos_w, dim=-1
            ),
            "failure_object_ori_error_rad": quat_error_magnitude(
                command.object_quat_w, command.simulator_object_quat_w
            ),
            "failure_ref_z_error_m": torch.abs(command.ref_pos_w[:, 2] - command.robot_ref_pos_w[:, 2]),
            "failure_body_z_error_max_m": body_z_error,
            "failure_target_body_z_m": command.body_pos_relative_w[:, body_idx, 2],
            "failure_actual_body_z_m": command.robot_body_pos_w[:, body_idx, 2],
            "failure_joint_pos_error_max_rad": torch.abs(command.joint_pos - command.robot_joint_pos).max(
                dim=-1
            ).values,
            "bad_ref_pos": bad_tracking.bad_ref_pos(command),
            "bad_ref_ori": bad_tracking.bad_ref_ori(command),
            "bad_motion_body_pos": bad_tracking.bad_motion_body_pos(command),
            "bad_object_pos": bad_tracking.bad_object_pos(command),
            "bad_object_ori": bad_tracking.bad_object_ori(command),
        }
        records = getattr(self, "_bad_tracking_diag_records", None)
        if records is None:
            records = {name: [] for name in fields}
            self._bad_tracking_diag_records = records
        for name, value in fields.items():
            records[name].append(_to_numpy(value[done]))

    BaseTask._check_termination = _check_termination_with_reasons
    BaseTask._bad_tracking_ab_instrumented = True


if not getattr(PPO, "_bad_tracking_ab_instrumented", False):
    _original_eval_pre = PPO._pre_evaluate_policy
    _original_eval_post = PPO._post_evaluate_policy

    def _pre_training_style(self: PPO, reset_env: bool = True) -> None:
        _original_eval_pre(self, reset_env=reset_env)
        # evaluate_policy performs a second reset immediately after this hook.
        # Keeping this false makes that reset and subsequent resets match train.
        self._unwrap_env().is_evaluating = False

    def _post_and_save(self: PPO) -> None:
        _original_eval_post(self)
        env = self._unwrap_env()
        command = env.command_manager.get_state("motion_command")
        records = getattr(env, "_bad_tracking_diag_records", {})
        arrays: dict[str, np.ndarray] = {}
        for name, chunks in records.items():
            if chunks:
                arrays[name] = np.concatenate(chunks, axis=0)

        n_failures = int(len(arrays.get("segment_age_steps", [])))
        reasons = [
            "bad_ref_pos",
            "bad_ref_ori",
            "bad_motion_body_pos",
            "bad_object_pos",
            "bad_object_ori",
        ]
        reason_counts = {
            name: int(np.count_nonzero(arrays.get(name, np.empty(0, dtype=bool)))) for name in reasons
        }
        ages = arrays.get("segment_age_steps", np.empty(0, dtype=np.int64))
        rel = arrays.get("start_relative_offset_xyz_m", np.empty((0, 3), dtype=np.float32))
        object_reason = arrays.get("bad_object_pos", np.zeros(n_failures, dtype=bool)) | arrays.get(
            "bad_object_ori", np.zeros(n_failures, dtype=bool)
        )
        summary: dict[str, Any] = {
            "noise_scale": NOISE_SCALE,
            "align_object_xy_to_robot": ALIGN_OBJECT_XY,
            "num_envs": int(env.num_envs),
            "num_eval_steps": int(getattr(self.config, "max_eval_steps", 0) or 0),
            "failure_count": n_failures,
            "active_segments_at_end": int(env.num_envs),
            "reason_counts_nonexclusive": reason_counts,
            "failure_age_steps": {
                "min": int(ages.min()) if ages.size else None,
                "median": float(np.median(ages)) if ages.size else None,
                "p90": float(np.quantile(ages, 0.9)) if ages.size else None,
                "max": int(ages.max()) if ages.size else None,
                "at_step_1": int(np.count_nonzero(ages <= 1)),
                "within_5": int(np.count_nonzero(ages <= 5)),
                "within_10": int(np.count_nonzero(ages <= 10)),
                "within_25": int(np.count_nonzero(ages <= 25)),
            },
            "object_reason_failures": int(np.count_nonzero(object_reason)),
            "object_reason_start_relative_xy_cm": {
                "mean": float(100 * np.linalg.norm(rel[object_reason, :2], axis=1).mean())
                if np.any(object_reason)
                else None,
                "p90": float(100 * np.quantile(np.linalg.norm(rel[object_reason, :2], axis=1), 0.9))
                if np.any(object_reason)
                else None,
            },
            "motion_file": str(command.motion_cfg.motion_file),
            "training_style_phase_sampling": True,
            "termination_thresholds_unchanged": True,
        }
        arrays["_summary_json"] = np.array(json.dumps(summary))
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(OUTPUT_PATH, **arrays)
        print("BAD_TRACKING_DIAG_SUMMARY=" + json.dumps(summary, sort_keys=True), flush=True)

    PPO._pre_evaluate_policy = _pre_training_style
    PPO._post_evaluate_policy = _post_and_save
    PPO._bad_tracking_ab_instrumented = True
