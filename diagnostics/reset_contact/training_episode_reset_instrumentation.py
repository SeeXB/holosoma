"""Diagnose only normal training-style episode resets in object WBT.

This excludes PPO/``reset_all`` warm-up transitions and motion-clip internal
resampling.  Collection starts only after evaluation setup has completed, and
an episode is enrolled only when ``TerminationManager.terminated`` causes the
ordinary BaseTask reset path.  Actions are sampled from the PPO distribution,
matching training rather than deterministic inference.
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
        "EPISODE_RESET_DIAG_OUTPUT",
        "diagnostics/reset_contact/training_episode_resets.npz",
    )
).resolve()
NOISE_SCALE = float(os.environ.get("EPISODE_RESET_DIAG_NOISE_SCALE", "1"))
ALIGN_OBJECT_XY = os.environ.get("EPISODE_RESET_DIAG_ALIGN_OBJECT_XY", "0").lower() in {
    "1",
    "true",
    "yes",
}


def _cpu(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy().copy()


def _ensure_state(command: MotionCommand) -> None:
    if hasattr(command, "_episode_reset_diag_active"):
        return
    n = command.num_envs
    device = command.device
    command._episode_reset_diag_active = torch.zeros(n, dtype=torch.bool, device=device)
    command._episode_reset_diag_pending_capture = torch.zeros(n, dtype=torch.bool, device=device)
    command._episode_reset_diag_age = torch.zeros(n, dtype=torch.long, device=device)
    command._episode_reset_diag_start_motion_step = torch.zeros(n, dtype=torch.long, device=device)
    command._episode_reset_diag_start_object_z = torch.zeros(n, device=device)
    command._episode_reset_diag_start_object_error = torch.zeros(n, device=device)
    command._episode_reset_diag_start_relative_offset = torch.zeros(n, 3, device=device)
    command._episode_reset_diag_prephysics_body_z_error = torch.zeros(n, device=device)
    command._episode_reset_diag_first_seen = torch.zeros(n, dtype=torch.bool, device=device)
    command._episode_reset_diag_first_body_z_error = torch.full((n,), torch.nan, device=device)
    command._episode_reset_diag_first_object_pos_error = torch.full((n,), torch.nan, device=device)
    command._episode_reset_diag_first_object_ori_error = torch.full((n,), torch.nan, device=device)
    command._episode_reset_diag_first_bad_ref_pos = torch.zeros(n, dtype=torch.bool, device=device)
    command._episode_reset_diag_first_bad_ref_ori = torch.zeros(n, dtype=torch.bool, device=device)
    command._episode_reset_diag_first_bad_body = torch.zeros(n, dtype=torch.bool, device=device)
    command._episode_reset_diag_first_bad_object_pos = torch.zeros(n, dtype=torch.bool, device=device)
    command._episode_reset_diag_first_bad_object_ori = torch.zeros(n, dtype=torch.bool, device=device)


def _append_rows(env: BaseTask, command: MotionCommand, mask: torch.Tensor, *, terminated: bool) -> None:
    if not torch.any(mask):
        return
    bad_tracking = env.termination_manager._term_instances["bad_tracking"]
    body_idx = bad_tracking.bad_motion_body_pos_body_indexes
    object_pos_error = torch.linalg.vector_norm(command.object_pos_w - command.simulator_object_pos_w, dim=-1)
    object_ori_error = quat_error_magnitude(command.object_quat_w, command.simulator_object_quat_w)
    body_z_error = torch.abs(
        command.body_pos_relative_w[:, body_idx, 2] - command.robot_body_pos_w[:, body_idx, 2]
    ).max(dim=-1).values
    fields = {
        "terminated": torch.full((env.num_envs,), terminated, dtype=torch.bool, device=env.device),
        "age_steps": command._episode_reset_diag_age,
        "start_motion_step": command._episode_reset_diag_start_motion_step,
        "end_motion_step": command.time_steps,
        "start_reference_object_z_m": command._episode_reset_diag_start_object_z,
        "start_object_pos_error_m": command._episode_reset_diag_start_object_error,
        "start_relative_offset_xyz_m": command._episode_reset_diag_start_relative_offset,
        "prephysics_body_z_error_max_m": command._episode_reset_diag_prephysics_body_z_error,
        "first_frame_seen": command._episode_reset_diag_first_seen,
        "first_frame_body_z_error_max_m": command._episode_reset_diag_first_body_z_error,
        "first_frame_object_pos_error_m": command._episode_reset_diag_first_object_pos_error,
        "first_frame_object_ori_error_rad": command._episode_reset_diag_first_object_ori_error,
        "first_bad_ref_pos": command._episode_reset_diag_first_bad_ref_pos,
        "first_bad_ref_ori": command._episode_reset_diag_first_bad_ref_ori,
        "first_bad_motion_body_pos": command._episode_reset_diag_first_bad_body,
        "first_bad_object_pos": command._episode_reset_diag_first_bad_object_pos,
        "first_bad_object_ori": command._episode_reset_diag_first_bad_object_ori,
        "end_object_pos_error_m": object_pos_error,
        "end_object_ori_error_rad": object_ori_error,
        "end_body_z_error_max_m": body_z_error,
        "bad_ref_pos": bad_tracking.bad_ref_pos(command),
        "bad_ref_ori": bad_tracking.bad_ref_ori(command),
        "bad_motion_body_pos": bad_tracking.bad_motion_body_pos(command),
        "bad_object_pos": bad_tracking.bad_object_pos(command),
        "bad_object_ori": bad_tracking.bad_object_ori(command),
    }
    records = getattr(env, "_episode_reset_diag_records", None)
    if records is None:
        records = {name: [] for name in fields}
        env._episode_reset_diag_records = records
    for name, value in fields.items():
        records[name].append(_cpu(value[mask]))


if not getattr(MotionCommand, "_training_episode_reset_instrumented", False):
    _original_reset = MotionCommand.reset
    _original_step = MotionCommand.step

    def _reset_with_condition(self: MotionCommand, env_ids: torch.Tensor | None) -> None:
        env_ids = self._ensure_index_tensor(env_ids)
        collecting = bool(getattr(self._env, "_episode_reset_diag_collect", False))
        normal_episode_reset = collecting and bool(torch.any(self._env.termination_manager.terminated[env_ids]))
        normal_mask = self._env.termination_manager.terminated[env_ids].clone() if collecting else None

        saved_scale = float(self.init_pose_cfg.overall_noise_scale)
        object.__setattr__(self.init_pose_cfg, "overall_noise_scale", NOISE_SCALE)
        try:
            _original_reset(self, env_ids)
        finally:
            object.__setattr__(self.init_pose_cfg, "overall_noise_scale", saved_scale)

        if env_ids.numel() and self.motion.has_object and ALIGN_OBJECT_XY:
            robot_offset = self.robot_root_pos_w[env_ids] - self.root_pos_w[env_ids]
            object_states = self._simulator_object_states()[env_ids].clone()
            object_states[:, :2] = self.object_pos_w[env_ids, :2] + robot_offset[:, :2]
            self._env.simulator.set_actor_states([self.object_name], env_ids, object_states)

        if not normal_episode_reset or normal_mask is None:
            return
        _ensure_state(self)
        enrolled = env_ids[normal_mask]
        self._episode_reset_diag_active[enrolled] = True
        self._episode_reset_diag_pending_capture[enrolled] = True
        self._episode_reset_diag_age[enrolled] = 0
        self._episode_reset_diag_first_seen[enrolled] = False
        self._episode_reset_diag_first_body_z_error[enrolled] = torch.nan
        self._episode_reset_diag_first_object_pos_error[enrolled] = torch.nan
        self._episode_reset_diag_first_object_ori_error[enrolled] = torch.nan
        self._episode_reset_diag_first_bad_ref_pos[enrolled] = False
        self._episode_reset_diag_first_bad_ref_ori[enrolled] = False
        self._episode_reset_diag_first_bad_body[enrolled] = False
        self._episode_reset_diag_first_bad_object_pos[enrolled] = False
        self._episode_reset_diag_first_bad_object_ori[enrolled] = False

    def _step_with_reset_snapshot(self: MotionCommand) -> None:
        _original_step(self)
        if not bool(getattr(self._env, "_episode_reset_diag_collect", False)):
            return
        _ensure_state(self)
        pending = self._episode_reset_diag_pending_capture
        if not torch.any(pending):
            return
        bad_tracking = self._env.termination_manager._term_instances["bad_tracking"]
        body_idx = bad_tracking.bad_motion_body_pos_body_indexes
        object_offset = self.simulator_object_pos_w - self.object_pos_w
        robot_offset = self.robot_root_pos_w - self.root_pos_w
        self._episode_reset_diag_start_motion_step[pending] = self.time_steps[pending]
        self._episode_reset_diag_start_object_z[pending] = self.object_pos_w[pending, 2]
        self._episode_reset_diag_start_object_error[pending] = torch.linalg.vector_norm(
            object_offset[pending], dim=-1
        )
        self._episode_reset_diag_start_relative_offset[pending] = (object_offset - robot_offset)[pending]
        self._episode_reset_diag_prephysics_body_z_error[pending] = torch.abs(
            self.body_pos_relative_w[:, body_idx, 2] - self.robot_body_pos_w[:, body_idx, 2]
        ).max(dim=-1).values[pending]
        self._episode_reset_diag_pending_capture[pending] = False

    MotionCommand.reset = _reset_with_condition
    MotionCommand.step = _step_with_reset_snapshot
    MotionCommand._training_episode_reset_instrumented = True


if not getattr(BaseTask, "_training_episode_reset_instrumented", False):
    _original_check = BaseTask._check_termination

    def _check_normal_reset_episodes(self: BaseTask) -> None:
        _original_check(self)
        if not bool(getattr(self, "_episode_reset_diag_collect", False)):
            return
        command = self.command_manager.get_state("motion_command")
        if command is None or not getattr(command.motion, "has_object", False):
            return
        _ensure_state(command)
        active = command._episode_reset_diag_active
        command._episode_reset_diag_age[active] += 1

        bad_tracking = self.termination_manager._term_instances["bad_tracking"]
        body_idx = bad_tracking.bad_motion_body_pos_body_indexes
        first = active & (command._episode_reset_diag_age == 1) & ~command._episode_reset_diag_first_seen
        if torch.any(first):
            command._episode_reset_diag_first_seen[first] = True
            command._episode_reset_diag_first_body_z_error[first] = torch.abs(
                command.body_pos_relative_w[:, body_idx, 2] - command.robot_body_pos_w[:, body_idx, 2]
            ).max(dim=-1).values[first]
            command._episode_reset_diag_first_object_pos_error[first] = torch.linalg.vector_norm(
                command.object_pos_w - command.simulator_object_pos_w, dim=-1
            )[first]
            command._episode_reset_diag_first_object_ori_error[first] = quat_error_magnitude(
                command.object_quat_w, command.simulator_object_quat_w
            )[first]
            command._episode_reset_diag_first_bad_ref_pos[first] = bad_tracking.bad_ref_pos(command)[first]
            command._episode_reset_diag_first_bad_ref_ori[first] = bad_tracking.bad_ref_ori(command)[first]
            command._episode_reset_diag_first_bad_body[first] = bad_tracking.bad_motion_body_pos(command)[first]
            command._episode_reset_diag_first_bad_object_pos[first] = bad_tracking.bad_object_pos(command)[first]
            command._episode_reset_diag_first_bad_object_ori[first] = bad_tracking.bad_object_ori(command)[first]

        ended = active & self.termination_manager.terminated
        _append_rows(self, command, ended, terminated=True)
        command._episode_reset_diag_active[ended] = False

    BaseTask._check_termination = _check_normal_reset_episodes
    BaseTask._training_episode_reset_instrumented = True


if not getattr(PPO, "_training_episode_reset_instrumented", False):
    _original_eval_pre = PPO._pre_evaluate_policy
    _original_pre_step = PPO._pre_eval_env_step
    _original_eval_post = PPO._post_evaluate_policy

    def _pre_training_style(self: PPO, reset_env: bool = True) -> None:
        _original_eval_pre(self, reset_env=reset_env)
        self._unwrap_env().is_evaluating = False

    def _stochastic_policy(self: PPO, device: Any = None):
        self.actor.eval()
        self.actor_obs_normalizer.eval()

        def policy_fn(obs: dict[str, torch.Tensor]) -> torch.Tensor:
            actor_obs = self._normalize_actor_obs(obs["actor_obs"], update=False)
            return self.actor.act({"actor_obs": actor_obs})

        return policy_fn

    def _enable_after_all_reset(self: PPO, actor_state: dict[str, Any]) -> dict[str, Any]:
        result = _original_pre_step(self, actor_state)
        env = self._unwrap_env()
        if not bool(getattr(env, "_episode_reset_diag_collect", False)):
            env._episode_reset_diag_records = None
            env._episode_reset_diag_collect = True
            command = env.command_manager.get_state("motion_command")
            _ensure_state(command)
            command._episode_reset_diag_active.zero_()
            command._episode_reset_diag_pending_capture.zero_()
            command._episode_reset_diag_age.zero_()
        return result

    def _post_and_save(self: PPO) -> None:
        _original_eval_post(self)
        env = self._unwrap_env()
        command = env.command_manager.get_state("motion_command")
        active = command._episode_reset_diag_active
        _append_rows(env, command, active, terminated=False)
        records = getattr(env, "_episode_reset_diag_records", None) or {}
        arrays = {name: np.concatenate(chunks, axis=0) for name, chunks in records.items() if chunks}

        terminated = arrays.get("terminated", np.empty(0, dtype=bool))
        ages = arrays.get("age_steps", np.empty(0, dtype=np.int64))
        first_seen = arrays.get("first_frame_seen", np.empty(0, dtype=bool))
        summary: dict[str, Any] = {
            "noise_scale": NOISE_SCALE,
            "align_object_xy_to_robot": ALIGN_OBJECT_XY,
            "num_envs": int(env.num_envs),
            "episode_reset_count": int(len(ages)),
            "terminated_count": int(np.count_nonzero(terminated)),
            "censored_active_count": int(np.count_nonzero(~terminated)) if terminated.size else 0,
            "first_frame_eligible_count": int(np.count_nonzero(first_seen)),
            "first_frame_termination_count": int(np.count_nonzero(terminated & (ages == 1)))
            if ages.size
            else 0,
            "excludes_reset_all_and_clip_resets": True,
            "stochastic_training_actions": True,
            "horizons": {},
        }
        for horizon in (5, 10, 25, 50, 100):
            eligible = terminated | (ages >= horizon)
            early = terminated & (ages <= horizon)
            summary["horizons"][str(horizon)] = {
                "eligible": int(np.count_nonzero(eligible)),
                "terminated": int(np.count_nonzero(early)),
                "percent": float(100 * np.count_nonzero(early) / np.count_nonzero(eligible))
                if np.count_nonzero(eligible)
                else None,
            }
        if ages.size:
            rel_xy = np.linalg.norm(arrays["start_relative_offset_xyz_m"][:, :2], axis=1)
            finite_pre = np.isfinite(arrays["prephysics_body_z_error_max_m"])
            summary["start_relative_xy_cm"] = {
                "mean": float(100 * rel_xy.mean()),
                "p90": float(100 * np.quantile(rel_xy, 0.9)),
                "max": float(100 * rel_xy.max()),
            }
            summary["start_object_pos_error_cm_max"] = float(100 * arrays["start_object_pos_error_m"].max())
            summary["prephysics_body_z_error_cm"] = {
                "p90": float(100 * np.quantile(arrays["prephysics_body_z_error_max_m"][finite_pre], 0.9)),
                "max": float(100 * arrays["prephysics_body_z_error_max_m"][finite_pre].max()),
            }
            summary["reason_counts_nonexclusive"] = {
                name: int(np.count_nonzero(arrays[name] & terminated))
                for name in (
                    "bad_ref_pos",
                    "bad_ref_ori",
                    "bad_motion_body_pos",
                    "bad_object_pos",
                    "bad_object_ori",
                )
            }
        arrays["_summary_json"] = np.array(json.dumps(summary))
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(OUTPUT_PATH, **arrays)
        print("EPISODE_RESET_DIAG_SUMMARY=" + json.dumps(summary, sort_keys=True), flush=True)

    PPO._pre_evaluate_policy = _pre_training_style
    PPO.get_inference_policy = _stochastic_policy
    PPO._pre_eval_env_step = _enable_after_all_reset
    PPO._post_evaluate_policy = _post_and_save
    PPO._training_episode_reset_instrumented = True
