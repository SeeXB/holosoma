"""Capture training-style WBT reset samples during checkpoint evaluation.

Load with ``eval_agent.py --import-file ...`` and enable the standard recording
callback.  Before policy rollout begins, this module performs several reset
batches with ``is_evaluating=False`` so ``MotionCommand.reset`` uses the same
adaptive/random phase path and initialization noise as training.  It snapshots
the post-reset robot/object state before any policy action is applied.

This is diagnostic instrumentation only; it does not register or alter an
experiment preset.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from holosoma.agents.callbacks.recording import EvalRecordingCallback
from holosoma.utils.safe_torch_import import torch


OUTPUT_PATH = Path(
    os.environ.get(
        "RESET_CONTACT_OUTPUT",
        "diagnostics/reset_contact/training_style_reset_samples.npz",
    )
).resolve()
NUM_BATCHES = int(os.environ.get("RESET_CONTACT_BATCHES", "4"))


def _cpu(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy().copy()


if not getattr(EvalRecordingCallback, "_reset_contact_instrumented", False):
    _original_pre = EvalRecordingCallback.on_pre_evaluate_policy

    def _pre_with_training_reset_samples(self: EvalRecordingCallback) -> None:
        _original_pre(self)
        env = self._get_env()
        sim = env.simulator
        command = env.command_manager.get_state("motion_command")
        if command is None or not getattr(command.motion, "has_object", False):
            raise RuntimeError("reset-contact diagnostic requires an object WBT motion")

        env_ids = torch.arange(env.num_envs, device=env.device)
        origins = sim.scene.env_origins
        saved_is_evaluating = bool(env.is_evaluating)
        samples: dict[str, list[np.ndarray]] = {
            "batch": [],
            "env_id": [],
            "motion_step": [],
            "root_pos": [],
            "root_quat_xyzw": [],
            "dof_pos": [],
            "object_pos": [],
            "object_quat_xyzw": [],
            "reference_root_pos": [],
            "reference_root_quat_xyzw": [],
            "reference_dof_pos": [],
            "reference_object_pos": [],
            "reference_object_quat_xyzw": [],
        }

        try:
            for batch in range(NUM_BATCHES):
                # This switch is the only behavioral change: evaluation normally
                # forces phase zero, while training samples adaptive/random phases.
                env.is_evaluating = False
                env.reset_envs_idx(env_ids)
                sim.set_actor_root_state_tensor_robots(env_ids, sim.robot_root_states)
                sim.set_dof_state_tensor_robots(env_ids, sim.dof_state)
                sim.refresh_sim_tensors()

                actual_object_states = sim.get_actor_states([command.object_name], env_ids=None)
                n = env.num_envs
                samples["batch"].append(np.full(n, batch, dtype=np.int64))
                samples["env_id"].append(np.arange(n, dtype=np.int64))
                samples["motion_step"].append(_cpu(command.time_steps).astype(np.int64))
                samples["root_pos"].append(_cpu(sim.robot_root_states[:, :3] - origins))
                samples["root_quat_xyzw"].append(_cpu(sim.robot_root_states[:, 3:7]))
                samples["dof_pos"].append(_cpu(sim.dof_pos))
                samples["object_pos"].append(_cpu(actual_object_states[:, :3] - origins))
                samples["object_quat_xyzw"].append(_cpu(actual_object_states[:, 3:7]))
                samples["reference_root_pos"].append(_cpu(command.root_pos_w - origins))
                samples["reference_root_quat_xyzw"].append(_cpu(command.root_quat_w))
                samples["reference_dof_pos"].append(_cpu(command.joint_pos))
                samples["reference_object_pos"].append(_cpu(command.object_pos_w - origins))
                samples["reference_object_quat_xyzw"].append(_cpu(command.object_quat_w))
        finally:
            env.is_evaluating = saved_is_evaluating

        arrays = {name: np.concatenate(values, axis=0) for name, values in samples.items()}
        metadata: dict[str, Any] = {
            "num_envs": int(env.num_envs),
            "num_batches": NUM_BATCHES,
            "num_samples": int(NUM_BATCHES * env.num_envs),
            "motion_file": str(command.motion_cfg.motion_file),
            "training_style_phase_sampling": True,
            "captured_before_policy_action": True,
        }
        arrays["_metadata_json"] = np.array(json.dumps(metadata))
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(OUTPUT_PATH, **arrays)

    EvalRecordingCallback.on_pre_evaluate_policy = _pre_with_training_reset_samples
    EvalRecordingCallback._reset_contact_instrumented = True
