"""Extend the read-only eval instrumentation to all environments.

Load this file after ``diagnostics/object_drift/nominal_eval_instrumentation.py``.
The standard callback still writes its env-0 recording.  This extension writes
the pre-reset diagnostic snapshot for every environment to a sibling file whose
name ends in ``_all_envs.npz``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from holosoma.agents.callbacks.recording import EvalRecordingCallback


if not getattr(EvalRecordingCallback, "_all_env_eval_instrumented", False):
    _previous_pre = EvalRecordingCallback.on_pre_evaluate_policy
    _previous_post = EvalRecordingCallback.on_post_eval_env_step
    _previous_save = EvalRecordingCallback._save

    def _pre_all_envs(self: EvalRecordingCallback) -> None:
        _previous_pre(self)
        self._all_env_buffers: dict[str, list[np.ndarray]] = {}
        self._all_env_metadata = {
            "dt": self._metadata["dt"],
            "fps": self._metadata["fps"],
            "num_envs": int(self._get_env().num_envs),
        }

    def _post_all_envs(self: EvalRecordingCallback, actor_state: dict[str, Any]) -> dict[str, Any]:
        actor_state = _previous_post(self, actor_state)
        env = self._get_env()
        snapshot = env._nominal_eval_pre_reset_snapshot
        for name, value in snapshot.items():
            self._all_env_buffers.setdefault(name, []).append(value.detach().cpu().numpy().copy())
        return actor_state

    def _save_all_envs(self: EvalRecordingCallback) -> None:
        _previous_save(self)
        arrays = {
            name: np.stack(values, axis=0)
            for name, values in self._all_env_buffers.items()
            if values
        }
        arrays["_metadata_json"] = np.array(json.dumps(self._all_env_metadata))
        path = Path(self.output_path)
        all_env_path = path.with_name(f"{path.stem}_all_envs.npz")
        np.savez_compressed(all_env_path, **arrays)

    EvalRecordingCallback.on_pre_evaluate_policy = _pre_all_envs
    EvalRecordingCallback.on_post_eval_env_step = _post_all_envs
    EvalRecordingCallback._save = _save_all_envs
    EvalRecordingCallback._all_env_eval_instrumented = True
