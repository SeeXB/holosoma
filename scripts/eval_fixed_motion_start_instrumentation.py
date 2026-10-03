"""Evaluation-only hook that resets original-adaptive WBT clips at a fixed frame.

This is intentionally an import-time diagnostic patch for ``eval_agent.py``.
It does not alter training code or checkpoints.  Set
``HOLOSOMA_EVAL_FIXED_MOTION_START`` to choose the frame (default: 191).
"""

from __future__ import annotations

import os

from holosoma.agents.callbacks.recording import EvalRecordingCallback
from holosoma.managers.command.terms.wbt import MotionCommand
from holosoma.utils.safe_torch_import import torch


START_FRAME = int(os.environ.get("HOLOSOMA_EVAL_FIXED_MOTION_START", "191"))


if not getattr(MotionCommand, "_fixed_eval_motion_start_instrumented", False):
    _original_reset = MotionCommand.reset

    def _reset_at_fixed_frame(self: MotionCommand, env_ids) -> None:
        evaluating = bool(getattr(self._env, "is_evaluating", False))
        sampler = getattr(self, "adaptive_timesteps_sampler", None)
        if not (evaluating and self._uses_original_adaptive and sampler is not None):
            return _original_reset(self, env_ids)

        original_sample = sampler.sample_global_time_steps
        original_eval_flag = self._env.is_evaluating

        def _fixed_sample(num_samples: int):
            frame = max(0, min(START_FRAME, self.motion.time_step_total - 2))
            return torch.full((num_samples,), frame, dtype=torch.long, device=self.device)

        # The historical reset path normally forces frame zero during eval.
        # Temporarily expose its training branch and replace only the sampler
        # output, so robot/object state initialization remains production code.
        self._env.is_evaluating = False
        sampler.sample_global_time_steps = _fixed_sample
        try:
            _original_reset(self, env_ids)
        finally:
            sampler.sample_global_time_steps = original_sample
            self._env.is_evaluating = original_eval_flag

    MotionCommand.reset = _reset_at_fixed_frame
    MotionCommand._fixed_eval_motion_start_instrumented = True


if not getattr(EvalRecordingCallback, "_fixed_eval_motion_start_metadata", False):
    _previous_pre = EvalRecordingCallback.on_pre_evaluate_policy

    def _pre_with_fixed_start_metadata(self: EvalRecordingCallback) -> None:
        _previous_pre(self)
        self._metadata["forced_motion_start_frame"] = START_FRAME
        all_metadata = getattr(self, "_all_env_metadata", None)
        if all_metadata is not None:
            all_metadata["forced_motion_start_frame"] = START_FRAME

    EvalRecordingCallback.on_pre_evaluate_policy = _pre_with_fixed_start_metadata
    EvalRecordingCallback._fixed_eval_motion_start_metadata = True
