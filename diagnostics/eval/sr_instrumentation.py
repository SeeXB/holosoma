"""Count first-episode success rates during a multi-environment eval.

The normal WBT termination configuration is left untouched.  The first
termination of each environment is recorded before ``BaseTask`` resets that
environment; reaching the timeout without a non-timeout termination is a
success, while a ``bad_tracking`` termination is a failure.  The result is
written to ``SR_OUTPUT`` (an ``.npz`` file) and printed as JSON.

This module is loaded with ``eval_agent.py --import-file`` and intentionally
does not change policy, randomization, or termination thresholds.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from holosoma.agents.ppo.ppo import PPO
from holosoma.envs.base_task.base_task import BaseTask
from holosoma.utils.safe_torch_import import torch


OUTPUT_PATH = Path(os.environ.get("SR_OUTPUT", "exp/eval/sr.npz")).resolve()


def _to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy().copy()


if not getattr(BaseTask, "_sr_eval_instrumented", False):
    _original_check_termination = BaseTask._check_termination
    _original_reset_all = BaseTask.reset_all

    def _reset_all_for_sr(self: BaseTask):
        # reset_all() itself executes one warm-up step.  Mark collection only
        # after the two reset_all calls used by PPO.evaluate_policy (plus the
        # constructor's initial reset_all), so no warm-up termination can enter
        # the first-episode counters.
        result = _original_reset_all(self)
        count = int(getattr(self, "_sr_reset_all_count", 0)) + 1
        self._sr_reset_all_count = count
        if count >= 3:
            self._sr_collect_terminations = True
        return result

    def _check_termination_with_first_episode(self: BaseTask) -> None:
        _original_check_termination(self)

        # The buffers are populated before reset_envs_idx() and therefore
        # still describe the just-finished episode here.
        if not hasattr(self, "_sr_first_done"):
            n = self.num_envs
            self._sr_first_done = torch.zeros(n, dtype=torch.bool, device=self.device)
            self._sr_first_timeout = torch.zeros(n, dtype=torch.bool, device=self.device)
            self._sr_first_bad_tracking = torch.zeros(n, dtype=torch.bool, device=self.device)
            self._sr_first_step = torch.full((n,), -1, dtype=torch.long, device=self.device)
            self._sr_first_body_z_error = torch.full((n,), float("nan"), device=self.device)
            self._sr_first_ref_z_error = torch.full((n,), float("nan"), device=self.device)
            self._sr_first_object_pos_error = torch.full((n,), float("nan"), device=self.device)
            self._sr_first_object_ori_error = torch.full((n,), float("nan"), device=self.device)
            self._sr_first_reasons = {
                name: torch.zeros(n, dtype=torch.bool, device=self.device)
                for name in (
                    "bad_ref_pos",
                    "bad_ref_ori",
                    "bad_motion_body_pos",
                    "bad_object_pos",
                    "bad_object_ori",
                )
            }
            self._sr_trace = {
                "ref_z_error_m": [],
                "ref_ori_error": [],
                "body_z_error_m": [],
                "object_pos_error_m": [],
                "object_ori_error_rad": [],
                "done": [],
                "timeout": [],
            }

        # ``BaseTask.reset_all()`` performs an internal warm-up step.  During
        # that step MotionCommand's relative-body buffer is still zero until
        # the post-termination task update, so normal thresholds would report
        # a spurious body failure.  PPO marks the actual evaluation loop via
        # ``_sr_collect_terminations`` immediately before its first step.
        if not getattr(self, "_sr_collect_terminations", False):
            return

        command = None
        if getattr(self, "command_manager", None) is not None:
            command = self.command_manager.get_state("motion_command")

        # Record complete raw traces while bad-tracking thresholds are widened
        # to 999 on the command line.  We classify these traces at the paper
        # thresholds after the rollout, avoiding early-reset and warm-up
        # artifacts.
        if command is not None:
            bad_tracking = None
            if self.termination_manager is not None:
                bad_tracking = self.termination_manager._term_instances.get("bad_tracking")
            if bad_tracking is not None:
                body_idx = bad_tracking.bad_motion_body_pos_body_indexes
                self._sr_trace["body_z_error_m"].append(
                    torch.abs(command.body_pos_relative_w[:, body_idx, 2] - command.robot_body_pos_w[:, body_idx, 2])
                    .max(dim=-1)
                    .values.detach()
                    .clone()
                )
            self._sr_trace["ref_z_error_m"].append(
                torch.abs(command.ref_pos_w[:, 2] - command.robot_ref_pos_w[:, 2]).detach().clone()
            )
            from holosoma.managers.observation.terms.wbt import gravity_vector
            from holosoma.utils.rotations import quat_rotate_inverse

            ref_gravity = quat_rotate_inverse(command.ref_quat_w, gravity_vector(self), w_last=True)
            robot_gravity = quat_rotate_inverse(command.robot_ref_quat_w, gravity_vector(self), w_last=True)
            self._sr_trace["ref_ori_error"].append(
                torch.abs(ref_gravity[:, 2] - robot_gravity[:, 2]).detach().clone()
            )
            if getattr(command.motion, "has_object", False):
                self._sr_trace["object_pos_error_m"].append(
                    torch.linalg.vector_norm(
                        command.object_pos_w - command.simulator_object_pos_w, dim=-1
                    ).detach().clone()
                )
                from holosoma.utils.rotations import quat_error_magnitude

                self._sr_trace["object_ori_error_rad"].append(
                    quat_error_magnitude(
                        command.object_quat_w, command.simulator_object_quat_w
                    ).detach().clone()
                )
            self._sr_trace["done"].append(self.reset_buf.bool().detach().clone())
            self._sr_trace["timeout"].append(self.time_out_buf.detach().clone())

        new_done = self.reset_buf.bool() & ~self._sr_first_done
        if not torch.any(new_done):
            return

        self._sr_first_done[new_done] = True
        self._sr_first_timeout[new_done] = self.time_out_buf[new_done]
        self._sr_first_step[new_done] = self.episode_length_buf[new_done]

        bad_tracking = None
        if self.termination_manager is not None:
            bad_tracking = self.termination_manager._term_instances.get("bad_tracking")
        command = None
        if getattr(self, "command_manager", None) is not None:
            command = self.command_manager.get_state("motion_command")

        if bad_tracking is None or command is None:
            self._sr_first_bad_tracking[new_done] = ~self.time_out_buf[new_done]
            return

        body_idx = bad_tracking.bad_motion_body_pos_body_indexes
        self._sr_first_body_z_error[new_done] = torch.abs(
            command.body_pos_relative_w[:, body_idx, 2] - command.robot_body_pos_w[:, body_idx, 2]
        ).max(dim=-1).values[new_done]
        self._sr_first_ref_z_error[new_done] = torch.abs(
            command.ref_pos_w[:, 2] - command.robot_ref_pos_w[:, 2]
        )[new_done]
        if getattr(command.motion, "has_object", False):
            self._sr_first_object_pos_error[new_done] = torch.linalg.vector_norm(
                command.object_pos_w - command.simulator_object_pos_w, dim=-1
            )[new_done]
            from holosoma.utils.rotations import quat_error_magnitude

            self._sr_first_object_ori_error[new_done] = quat_error_magnitude(
                command.object_quat_w, command.simulator_object_quat_w
            )[new_done]

        reason_values = {
            "bad_ref_pos": bad_tracking.bad_ref_pos(command),
            "bad_ref_ori": bad_tracking.bad_ref_ori(command),
            "bad_motion_body_pos": bad_tracking.bad_motion_body_pos(command),
        }
        if getattr(command.motion, "has_object", False):
            reason_values.update(
                {
                    "bad_object_pos": bad_tracking.bad_object_pos(command),
                    "bad_object_ori": bad_tracking.bad_object_ori(command),
                }
            )
        bad = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for name, values in reason_values.items():
            self._sr_first_reasons[name][new_done] = values[new_done]
            bad |= values
        self._sr_first_bad_tracking[new_done] = bad[new_done]

    BaseTask._check_termination = _check_termination_with_first_episode
    BaseTask.reset_all = _reset_all_for_sr
    BaseTask._sr_eval_instrumented = True


if not getattr(PPO, "_sr_eval_instrumented", False):
    _original_pre_eval_env_step = PPO._pre_eval_env_step
    _original_eval_post = PPO._post_evaluate_policy

    def _pre_eval_env_step_with_sr(self: PPO, actor_state: dict[str, Any]) -> dict[str, Any]:
        # This hook is first called after the second reset_all() and before the
        # actual evaluation loop.  All warm-up termination checks have already
        # completed by this point.
        if not getattr(self, "_sr_pre_eval_seen", False):
            self._sr_pre_eval_seen = True
            self._unwrap_env()._sr_collect_terminations = True
        return _original_pre_eval_env_step(self, actor_state)

    def _post_evaluate_policy_with_sr(self: PPO) -> None:
        _original_eval_post(self)
        env = self._unwrap_env()
        n = int(env.num_envs)
        done = _to_numpy(getattr(env, "_sr_first_done", torch.zeros(n, dtype=torch.bool, device=env.device)))
        timeout = _to_numpy(
            getattr(env, "_sr_first_timeout", torch.zeros(n, dtype=torch.bool, device=env.device))
        )
        bad = _to_numpy(
            getattr(env, "_sr_first_bad_tracking", torch.zeros(n, dtype=torch.bool, device=env.device))
        )
        first_step = _to_numpy(
            getattr(env, "_sr_first_step", torch.full((n,), -1, dtype=torch.long, device=env.device))
        )
        reasons = {
            name: _to_numpy(values)
            for name, values in getattr(env, "_sr_first_reasons", {}).items()
        }

        trace = getattr(env, "_sr_trace", {})

        def _trace_array(name: str) -> np.ndarray:
            chunks = trace.get(name, [])
            if not chunks:
                return np.empty((0, n), dtype=np.float32)
            return _to_numpy(torch.stack(chunks, dim=0))

        ref_z = _trace_array("ref_z_error_m")
        ref_ori = _trace_array("ref_ori_error")
        body_z = _trace_array("body_z_error_m")
        object_pos = _trace_array("object_pos_error_m")
        object_ori = _trace_array("object_ori_error_rad")
        trace_timeout = _trace_array("timeout").astype(bool)
        if ref_z.shape[0] == 0:
            max_ref_z = max_ref_ori = max_body_z = max_object_pos = max_object_ori = np.full(n, np.inf)
        else:
            max_ref_z = np.max(ref_z, axis=0)
            max_ref_ori = np.max(ref_ori, axis=0)
            max_body_z = np.max(body_z, axis=0)
            max_object_pos = np.max(object_pos, axis=0)
            max_object_ori = np.max(object_ori, axis=0)
        bad_ref_z = max_ref_z > 0.5
        bad_ref_ori = max_ref_ori > 0.8
        bad_body_z = max_body_z > 0.25
        bad_object_pos = max_object_pos > 1.0
        bad_object_ori = max_object_ori > 0.7853981633974483
        success = ~bad_ref_z & ~bad_ref_ori & ~bad_body_z & ~bad_object_pos & ~bad_object_ori
        # A run is complete only if each environment reached the timeout.  An
        # incomplete trace is reported separately and never counted as a
        # successful episode.
        completed = trace_timeout.any(axis=0) if trace_timeout.shape[0] else np.zeros(n, dtype=bool)
        success &= completed
        failure = completed & ~success
        summary: dict[str, Any] = {
            "num_envs": n,
            "num_completed": int(completed.sum()),
            "num_success": int(success.sum()),
            "num_failure": int(failure.sum()),
            "success_rate": float(success.mean()) if n else 0.0,
            "incomplete": int((~completed).sum()),
            "trace_steps": int(ref_z.shape[0]),
            "first_termination_step": first_step.tolist(),
            "first_body_z_error_m": _to_numpy(
                getattr(env, "_sr_first_body_z_error", torch.full((n,), float("nan"), device=env.device))
            ).tolist(),
            "first_ref_z_error_m": _to_numpy(
                getattr(env, "_sr_first_ref_z_error", torch.full((n,), float("nan"), device=env.device))
            ).tolist(),
            "first_object_pos_error_m": _to_numpy(
                getattr(env, "_sr_first_object_pos_error", torch.full((n,), float("nan"), device=env.device))
            ).tolist(),
            "first_object_ori_error_rad": _to_numpy(
                getattr(env, "_sr_first_object_ori_error", torch.full((n,), float("nan"), device=env.device))
            ).tolist(),
            "timeout": completed.astype(bool).tolist(),
            "bad_tracking": (completed & ~success).astype(bool).tolist(),
            "reason_counts": {
                "bad_ref_pos_z": int(bad_ref_z.sum()),
                "bad_ref_ori": int(bad_ref_ori.sum()),
                "bad_motion_body_pos_z": int(bad_body_z.sum()),
                "bad_object_pos": int(bad_object_pos.sum()),
                "bad_object_ori": int(bad_object_ori.sum()),
            },
            "max_ref_z_error_m": max_ref_z.tolist(),
            "max_ref_ori_error": max_ref_ori.tolist(),
            "max_body_z_error_m": max_body_z.tolist(),
            "max_object_pos_error_m": max_object_pos.tolist(),
            "max_object_ori_error_rad": max_object_ori.tolist(),
            "dt": float(env.dt),
            "horizon_steps": int(getattr(self.config, "max_eval_steps", 0) or 0),
            "max_episode_length_steps": int(getattr(env, "max_episode_length", -1)),
            "collection_started": bool(getattr(env, "_sr_collect_terminations", False)),
            "paper_thresholds": {
                "ref_z_m": 0.5,
                "ref_ori_gravity_z": 0.8,
                "tracked_body_z_m": 0.25,
                "object_pos_m": 1.0,
                "object_ori_rad": 0.7853981633974483,
            },
        }
        arrays: dict[str, np.ndarray] = {
            "success": success,
            "failure": failure,
            "completed": completed,
            "timeout": completed,
            "bad_tracking": completed & ~success,
            "first_termination_step": first_step,
            "max_ref_z_error_m": max_ref_z,
            "max_ref_ori_error": max_ref_ori,
            "max_body_z_error_m": max_body_z,
            "max_object_pos_error_m": max_object_pos,
            "max_object_ori_error_rad": max_object_ori,
            "trace_ref_z_error_m": ref_z,
            "trace_ref_ori_error": ref_ori,
            "trace_body_z_error_m": body_z,
            "trace_object_pos_error_m": object_pos,
            "trace_object_ori_error_rad": object_ori,
            "_summary_json": np.array(json.dumps(summary, sort_keys=True)),
        }
        arrays.update({f"reason_{name}": values for name, values in reasons.items()})
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(OUTPUT_PATH, **arrays)
        print("SR_SUMMARY=" + json.dumps(summary, sort_keys=True), flush=True)

    PPO._pre_eval_env_step = _pre_eval_env_step_with_sr
    PPO._post_evaluate_policy = _post_evaluate_policy_with_sr
    PPO._sr_eval_instrumented = True
