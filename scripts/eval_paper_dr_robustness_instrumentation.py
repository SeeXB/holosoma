"""Instrumentation for the standard Paper-DR robustness evaluation.

This module is loaded with ``eval_agent.py --import-file``.  It reuses the
existing pre-reset tracking snapshot and all-environment recorder, then adds
per-environment push counts and protocol metadata.  The recorder itself does
not alter policy, reward, termination, or simulator state.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import sys

import numpy as np
from loguru import logger

# ``eval_agent.py`` imports this file by path and may not put the repository
# root on ``sys.path``.  Add it so the established diagnostics namespace is
# importable regardless of the launch directory.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# These two modules provide the repository's established pre-reset diagnostic
# channels and all-environment callback extension.
from diagnostics.object_drift import nominal_eval_instrumentation as _nominal  # noqa: F401
from scripts import multienv_recording_instrumentation as _all_envs  # noqa: F401

from holosoma.agents.callbacks.recording import EvalRecordingCallback
from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager
from holosoma.utils.safe_torch_import import torch


if not getattr(WholeBodyTrackingManager, "_paper_dr_push_counter", False):
    _original_push = WholeBodyTrackingManager._push_robots

    def _push_with_counter(self: WholeBodyTrackingManager, env_ids) -> None:
        _original_push(self, env_ids)
        counter = getattr(self, "_paper_eval_push_count", None)
        if counter is None:
            counter = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
            self._paper_eval_push_count = counter
        counter[env_ids] += 1

    WholeBodyTrackingManager._push_robots = _push_with_counter
    WholeBodyTrackingManager._paper_dr_push_counter = True


if not getattr(EvalRecordingCallback, "_paper_dr_robustness_instrumented", False):
    _previous_pre = EvalRecordingCallback.on_pre_evaluate_policy
    _previous_post = EvalRecordingCallback.on_post_eval_env_step
    _previous_save = EvalRecordingCallback._save

    def _pre_paper_dr(self: EvalRecordingCallback) -> None:
        _previous_pre(self)
        env = self._get_env()
        # ``apply_pushes`` consults this explicit protocol flag while the
        # environment remains in eval mode.  It is set after setup and before
        # the first policy step, so startup/reset DR remains unchanged.
        env._paper_eval_allow_pushes = True
        env._paper_eval_push_count = torch.zeros(env.num_envs, dtype=torch.int64, device=env.device)
        self._buffers["push_count"] = []
        self._all_env_buffers["push_count"] = []
        self._paper_eval_completed_episodes = np.zeros(env.num_envs, dtype=np.int64)
        self._metadata.update(
            {
                "eval_protocol": "paper_dr_robustness_v1",
                "episodes_per_env_quota": 10,
                "rollout_envs": int(env.num_envs),
                "paper_object_mass_kg": [0.1, 2.0],
                "paper_object_com_offset_m": 0.08,
                "paper_object_inertia_scale": [0.5, 1.5],
                "paper_push_enabled_during_eval": True,
                "paper_push_interval_s": [1.0, 3.0],
                "paper_push_max_vel": [0.3, 0.3, 0.3, 0.78, 0.78, 0.78],
                "paper_shape_scale": [0.9, 1.1],
                "paper_shape_scale_implemented": False,
                "termination_object_pos_threshold_m": 1.0,
                "termination_object_ori_threshold_rad": 0.7853981633974483,
            }
        )
        self._all_env_metadata.update(
            {
                "eval_protocol": "paper_dr_robustness_v1",
                "episodes_per_env_quota": 10,
                "paper_push_enabled_during_eval": True,
                "paper_shape_scale_implemented": False,
            }
        )

    def _post_paper_dr(self: EvalRecordingCallback, actor_state: dict[str, Any]) -> dict[str, Any]:
        actor_state = _previous_post(self, actor_state)
        env = self._get_env()
        push_count = env._paper_eval_push_count.detach().cpu().numpy().copy()
        self._buffers["push_count"].append(push_count[self.env_id])
        self._all_env_buffers["push_count"].append(push_count)
        self._paper_eval_completed_episodes += self._all_env_buffers["done"][-1].astype(np.int64)
        if self._step_count % 100 == 0:
            logger.info(
                "Paper-DR eval step {}: completed episodes/env min={} max={} (quota=10)",
                self._step_count,
                int(self._paper_eval_completed_episodes.min()),
                int(self._paper_eval_completed_episodes.max()),
            )
        # The protocol scores only each environment's first ten episodes.
        # Once every environment has reached that quota, further rollout adds
        # no scored samples. Request a normal exit so all callbacks still save.
        if np.all(self._paper_eval_completed_episodes >= 10):
            actor_state["stop"] = True
        return actor_state

    def _save_paper_dr(self: EvalRecordingCallback) -> None:
        env = self._get_env()
        counts = env._paper_eval_push_count.detach().cpu().tolist()
        self._metadata["push_events_by_env"] = counts
        self._all_env_metadata["push_events_by_env"] = counts
        _previous_save(self)

    EvalRecordingCallback.on_pre_evaluate_policy = _pre_paper_dr
    EvalRecordingCallback.on_post_eval_env_step = _post_paper_dr
    EvalRecordingCallback._save = _save_paper_dr
    EvalRecordingCallback._paper_dr_robustness_instrumented = True
