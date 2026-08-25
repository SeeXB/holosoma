"""Read-only instrumentation for deterministic object-WBT checkpoint evaluation.

Load this file with ``eval_agent.py --import-file ...``.  It registers a
nominal randomization manager that keeps the framework's required bookkeeping
terms but disables their perturbations, and augments the standard recording
callback with the state immediately before an environment reset.  No policy,
reward, termination, command, or simulator dynamics are changed.
"""

from __future__ import annotations

from typing import Any

from holosoma.agents.callbacks.recording import EvalRecordingCallback
from holosoma.config_types.randomization import RandomizationManagerCfg, RandomizationTermCfg
from holosoma.config_values.randomization import RANDOMIZATION_REGISTRY
from holosoma.config_values.wbt.g1.randomization import base_reset_terms, base_setup_terms, base_step_terms
from holosoma.envs.base_task.base_task import BaseTask
from holosoma.utils.rotations import quat_error_magnitude
from holosoma.utils.safe_torch_import import torch


def _setup_term_with_enabled(name: str, enabled: bool):
    term = base_setup_terms[name]
    return type(term)(func=term.func, params={**term.params, "enabled": enabled})


# The action and observation paths expect these stateful terms to exist even in
# a non-randomized run.  Their enabled flags/ranges make every operation a no-op.
_nominal_setup_terms = {
    "push_randomizer_state": _setup_term_with_enabled("push_randomizer_state", False),
    "actuator_randomizer_state": base_setup_terms["actuator_randomizer_state"],
    "setup_action_delay_buffers": _setup_term_with_enabled("setup_action_delay_buffers", False),
    "setup_dof_pos_bias": _setup_term_with_enabled("setup_dof_pos_bias", False),
}
RANDOMIZATION_REGISTRY.add(
    "eval-empty",
    RandomizationManagerCfg(
        setup_terms=_nominal_setup_terms,
        reset_terms=base_reset_terms,
        step_terms=base_step_terms,
    ),
)

# The box asset's authored mass is 0.1 kg, while training adds a uniformly sampled
# 1--4 kg offset.  This preset fixes the resulting mass at 2.5 kg without enabling
# any of the other domain-randomization terms, so payload robustness can be tested
# independently from friction, inertia, robot-property, and push perturbations.
RANDOMIZATION_REGISTRY.add(
    "eval-fixed-object-2p5kg",
    RandomizationManagerCfg(
        setup_terms={
            **_nominal_setup_terms,
            "randomize_object_rigid_body_mass_startup": RandomizationTermCfg(
                func="holosoma.managers.randomization.terms.objects:randomize_object_rigid_body_mass_startup",
                params={"mass_distribution_params": [2.4, 2.4]},
            ),
        },
        reset_terms=base_reset_terms,
        step_terms=base_step_terms,
    ),
)


def _clone(value: torch.Tensor) -> torch.Tensor:
    return value.detach().clone()


if not getattr(BaseTask, "_nominal_eval_instrumented", False):
    _original_check_termination = BaseTask._check_termination

    def _check_termination_with_snapshot(self: BaseTask) -> None:
        """Snapshot live states after checks and before reset_envs_idx()."""

        _original_check_termination(self)
        command = self.command_manager.get_state("motion_command")
        if command is None or not getattr(command.motion, "has_object", False):
            return

        actual_object_pos = command.simulator_object_pos_w
        actual_object_quat = command.simulator_object_quat_w
        reference_object_pos = command.object_pos_w
        reference_object_quat = command.object_quat_w

        snapshot: dict[str, torch.Tensor] = {
            "pre_root_pos": _clone(command.robot_root_pos_w),
            "pre_root_quat_xyzw": _clone(command.robot_root_quat_w),
            "pre_dof_pos": _clone(command.robot_joint_pos),
            "pre_actual_tracked_body_pos_w": _clone(command.robot_body_pos_w),
            "pre_reference_tracked_body_pos_w": _clone(command.body_pos_relative_w),
            "object_pos_w": _clone(actual_object_pos),
            "object_quat_xyzw": _clone(actual_object_quat),
            "reference_object_pos_w": _clone(reference_object_pos),
            "reference_object_quat_xyzw": _clone(reference_object_quat),
            "reference_torso_pos_w": _clone(command.ref_pos_w),
            "reference_torso_quat_xyzw": _clone(command.ref_quat_w),
            "actual_torso_pos_w": _clone(command.robot_ref_pos_w),
            "actual_torso_quat_xyzw": _clone(command.robot_ref_quat_w),
            "motion_step": _clone(command.time_steps),
            "episode_step": _clone(self.episode_length_buf),
            "done": _clone(self.reset_buf.bool()),
            "timeout": _clone(self.time_out_buf.bool()),
            "object_pos_error_m": _clone(torch.linalg.vector_norm(reference_object_pos - actual_object_pos, dim=-1)),
            "object_ori_error_rad": _clone(quat_error_magnitude(reference_object_quat, actual_object_quat)),
            "ref_z_error_m": _clone(torch.abs(command.ref_pos_w[:, 2] - command.robot_ref_pos_w[:, 2])),
            "tracked_body_z_error_max_m": _clone(
                torch.max(torch.abs(command.body_pos_relative_w[..., 2] - command.robot_body_pos_w[..., 2]), dim=-1).values
            ),
        }

        false = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        reason_values = {
            "bad_ref_pos": false,
            "bad_ref_ori": false,
            "bad_motion_body_pos": false,
            "bad_object_pos": false,
            "bad_object_ori": false,
        }
        bad_tracking = self.termination_manager._term_instances.get("bad_tracking")
        if bad_tracking is not None:
            reason_values["bad_ref_pos"] = bad_tracking.bad_ref_pos(command)
            reason_values["bad_ref_ori"] = bad_tracking.bad_ref_ori(command)
            reason_values["bad_motion_body_pos"] = bad_tracking.bad_motion_body_pos(command)
            reason_values["bad_object_pos"] = bad_tracking.bad_object_pos(command)
            reason_values["bad_object_ori"] = bad_tracking.bad_object_ori(command)
        snapshot.update({name: _clone(value) for name, value in reason_values.items()})
        self._nominal_eval_pre_reset_snapshot = snapshot

    BaseTask._check_termination = _check_termination_with_snapshot
    BaseTask._nominal_eval_instrumented = True


if not getattr(EvalRecordingCallback, "_nominal_eval_instrumented", False):
    _original_recording_pre = EvalRecordingCallback.on_pre_evaluate_policy
    _original_recording_post = EvalRecordingCallback.on_post_eval_env_step

    def _recording_pre_with_snapshot(self: EvalRecordingCallback) -> None:
        _original_recording_pre(self)
        custom_channels = (
            "pre_root_pos",
            "pre_root_quat_xyzw",
            "pre_dof_pos",
            "pre_actual_tracked_body_pos_w",
            "pre_reference_tracked_body_pos_w",
            "object_pos_w",
            "object_quat_xyzw",
            "reference_object_pos_w",
            "reference_object_quat_xyzw",
            "reference_torso_pos_w",
            "reference_torso_quat_xyzw",
            "actual_torso_pos_w",
            "actual_torso_quat_xyzw",
            "motion_step",
            "episode_step",
            "done",
            "timeout",
            "object_pos_error_m",
            "object_ori_error_rad",
            "ref_z_error_m",
            "tracked_body_z_error_max_m",
            "bad_ref_pos",
            "bad_ref_ori",
            "bad_motion_body_pos",
            "bad_object_pos",
            "bad_object_ori",
        )
        for name in custom_channels:
            self._buffers[name] = []
        self._metadata["nominal_eval_pre_reset_snapshot"] = True

    def _recording_post_with_snapshot(self: EvalRecordingCallback, actor_state: dict[str, Any]) -> dict[str, Any]:
        actor_state = _original_recording_post(self, actor_state)
        env = self._get_env()
        snapshot = getattr(env, "_nominal_eval_pre_reset_snapshot", None)
        if snapshot is None:
            raise RuntimeError("Nominal eval snapshot was not populated before recording")
        for name in tuple(snapshot):
            value = snapshot[name][self.env_id]
            self._buffers[name].append(value.detach().cpu().numpy().copy())
        return actor_state

    EvalRecordingCallback.on_pre_evaluate_policy = _recording_pre_with_snapshot
    EvalRecordingCallback.on_post_eval_env_step = _recording_post_with_snapshot
    EvalRecordingCallback._nominal_eval_instrumented = True
