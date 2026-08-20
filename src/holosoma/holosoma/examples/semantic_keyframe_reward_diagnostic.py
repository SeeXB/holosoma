"""Reference-replay scale diagnostic for semantic-keyframe WBT rewards."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch

from holosoma.config_types.reward import SemanticKeyframeRewardCfg
from holosoma.config_values.wbt.g1.command import motion_config_w_object_transition_truncated_b4
from holosoma.config_values.wbt.g1.reward import (
    g1_29dof_wbt_w_object_semantic_reward,
    semantic_keyframe_config,
)
from holosoma.managers.reward.manager import (
    allocate_fixed_positive_budget,
    compute_base_positive_reward_budget,
)
from holosoma.managers.reward.semantic_keyframes import SemanticKeyframeRuntime

DEFAULT_MOTION = Path(motion_config_w_object_transition_truncated_b4.motion_file)
DEFAULT_SEMANTIC = Path(semantic_keyframe_config.semantic_file)
DEFAULT_OUTPUT = Path(
    "src/holosoma_retargeting/holosoma_retargeting/"
    "benchmark_results_full_event_transition_truncation/rl/semantic_reward_diagnostic"
)


class _CommandManager:
    def __init__(self, command):
        self.command = command

    def get_state(self, name):
        if name != "motion_command":
            return None
        return self.command


def _stats(values: torch.Tensor) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "max": float(values.max()),
    }


def _load_reference_env(motion_file: Path):
    with np.load(motion_file, allow_pickle=False) as payload:
        fps = float(np.asarray(payload["fps"]).reshape(-1)[0])
        body_names = payload["body_names"].tolist()
        tracked_names = list(motion_config_w_object_transition_truncated_b4.body_names_to_track)
        indices = [body_names.index(name) for name in tracked_names]
        body_pos = torch.as_tensor(payload["body_pos_w"][:, indices], dtype=torch.float32)
        body_quat_wxyz = torch.as_tensor(payload["body_quat_w"][:, indices], dtype=torch.float32)
        body_quat = body_quat_wxyz[..., [1, 2, 3, 0]]
        body_lin_vel = torch.as_tensor(payload["body_lin_vel_w"][:, indices], dtype=torch.float32)
        body_ang_vel = torch.as_tensor(payload["body_ang_vel_w"][:, indices], dtype=torch.float32)
        object_pos = torch.as_tensor(payload["object_pos_w"], dtype=torch.float32)
        object_quat_wxyz = torch.as_tensor(payload["object_quat_w"], dtype=torch.float32)
        object_quat = object_quat_wxyz[..., [1, 2, 3, 0]]
        joint_pos = torch.as_tensor(payload["joint_pos"][:, 7:], dtype=torch.float32)

    frame_count = body_pos.shape[0]
    motion = SimpleNamespace(
        fps=np.asarray(fps),
        has_object=True,
        motion_start_idx=torch.tensor([0], dtype=torch.long),
    )
    command = SimpleNamespace(
        motion=motion,
        motion_cfg=SimpleNamespace(body_names_to_track=tracked_names),
        time_steps=torch.arange(frame_count, dtype=torch.long),
        motion_ids=torch.zeros(frame_count, dtype=torch.long),
        body_pos_relative_w=body_pos,
        body_quat_relative_w=body_quat,
        body_pos_w=body_pos,
        body_quat_w=body_quat,
        body_lin_vel_w=body_lin_vel,
        body_ang_vel_w=body_ang_vel,
        robot_body_pos_w=body_pos.clone(),
        robot_body_quat_w=body_quat.clone(),
        robot_body_lin_vel_w=body_lin_vel.clone(),
        robot_body_ang_vel_w=body_ang_vel.clone(),
        object_pos_w=object_pos,
        object_quat_w=object_quat,
        simulator_object_pos_w=object_pos.clone(),
        simulator_object_quat_w=object_quat.clone(),
    )
    env = SimpleNamespace(
        command_manager=_CommandManager(command),
        reward_manager=SimpleNamespace(cfg=g1_29dof_wbt_w_object_semantic_reward),
    )
    return env, command, fps, joint_pos


def run(args: argparse.Namespace) -> dict:
    motion_file = args.motion_file.resolve()
    semantic_file = args.semantic_file.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    env, command, motion_fps, joint_pos = _load_reference_env(motion_file)
    config_values = dict(semantic_keyframe_config.__dict__)
    config_values.update(
        semantic_file=str(semantic_file),
        semantic_fps=args.semantic_fps,
    )
    config = SemanticKeyframeRewardCfg(**config_values)
    runtime = SemanticKeyframeRuntime(config, env)
    result = runtime.evaluate()

    # A perfect reference replay makes every positive bounded Omni term one.
    # W_pos is still inferred from the preset rather than encoded here.
    positive_budget = compute_base_positive_reward_budget(g1_29dof_wbt_w_object_semantic_reward)
    base_positive = torch.full_like(result.active_gate, positive_budget)
    penalty = torch.zeros_like(result.active_gate)
    allocation = allocate_fixed_positive_budget(
        base_positive,
        penalty,
        result.active_gate,
        result.combined_reward,
        positive_budget,
    )
    policy_dt = 1.0 / motion_fps
    exact_ordinary_mask = result.active_gate == 0.0
    semantic_mask = result.active_gate > 0.0
    action_delta = joint_pos[1:] - joint_pos[:-1]
    reference_action_smoothness = torch.square(action_delta).sum(dim=-1)

    summary = {
        "diagnostic_kind": "perfect reference replay; no physics rollout",
        "motion_file": str(motion_file),
        "semantic_file": str(semantic_file),
        "motion_fps": motion_fps,
        "semantic_fps": runtime.semantic_fps,
        "frame_count": int(command.time_steps.numel()),
        "policy_dt": policy_dt,
        "positive_reward_budget": positive_budget,
        "event_count": len(runtime.events),
        "events": [
            {
                "event_id": event.event_id,
                "name_for_logging_only": event.name,
                "trigger_time_s": event.trigger_time_s,
                "trigger_rl_frame": runtime.trigger_rl_frame(event.event_id),
                "support_start_s": event.support_start_s,
                "support_end_s": event.support_end_s,
                "next_trigger_time_s": (
                    event.next_trigger_time_s if np.isfinite(event.next_trigger_time_s) else None
                ),
                "robot_bodies": list(event.body_names),
            }
            for event in runtime.events
        ],
        "components_before_dt": {
            "base_positive": _stats(allocation.base_positive),
            "penalty": _stats(allocation.penalty),
            "activity": _stats(allocation.activity),
            "kf_part": _stats(result.part_reward),
            "kf_rel": _stats(result.rel_reward),
            "kf_dyn": _stats(result.dyn_reward),
            "semantic_combined": _stats(result.combined_reward),
            "valid_objective_count": _stats(result.valid_objective_count),
            "alpha": _stats(allocation.alpha),
            "base_contribution": _stats(allocation.base_contribution),
            "semantic_contribution": _stats(allocation.semantic_contribution),
            "positive_total": _stats(allocation.positive_total),
            "total": _stats(allocation.total),
        },
        "components_after_dt": {
            "base_contribution": _stats(allocation.base_contribution * policy_dt),
            "semantic_contribution": _stats(allocation.semantic_contribution * policy_dt),
            "penalty": _stats(allocation.penalty * policy_dt),
            "total": _stats(allocation.total * policy_dt),
        },
        "ordinary_frame_checks": {
            "exact_zero_activity_frame_count": int(exact_ordinary_mask.sum()),
            "max_abs_alpha": (
                float(allocation.alpha[exact_ordinary_mask].abs().max()) if exact_ordinary_mask.any() else None
            ),
            "max_abs_positive_difference_from_baseline": (
                float(
                    (allocation.positive_total[exact_ordinary_mask] - base_positive[exact_ordinary_mask])
                    .abs()
                    .max()
                )
                if exact_ordinary_mask.any()
                else None
            ),
        },
        "semantic_frame_checks": {
            "frame_count": int(semantic_mask.sum()),
            "min_alpha": float(allocation.alpha[semantic_mask].min()) if semantic_mask.any() else None,
            "max_alpha": float(allocation.alpha[semantic_mask].max()) if semantic_mask.any() else None,
            "theoretical_alpha_upper_bound": 1.0 / (positive_budget + 1.0),
            "max_positive_total": float(allocation.positive_total.max()),
            "positive_budget_not_exceeded": bool(
                torch.all(allocation.positive_total <= positive_budget + 1.0e-6)
            ),
        },
        "reference_errors": {
            "overall_body_tracking_m": 0.0,
            "ordinary_body_tracking_m": 0.0,
            "semantic_part_normalized_energy": float(result.part_error.max()),
            "semantic_relative_normalized_energy": float(result.rel_error.max()),
            "semantic_dynamics_normalized_energy": float(result.dyn_error.max()),
            "global_root_tracking_m": 0.0,
        },
        "reference_action_smoothness": _stats(reference_action_smoothness),
        "not_available_without_physics_rollout": [
            "episode_success_or_termination_rate",
            "contact_and_penetration_metrics",
            "joint_limit_penalty using live simulator limits",
        ],
    }
    serialized = json.dumps(summary, indent=2) + "\n"
    (output_dir / "semantic_reward_budget_diagnostic.json").write_text(serialized, encoding="utf-8")
    # Keep the earlier artifact path usable for downstream notebooks.
    (output_dir / "reward_scale_diagnostic.json").write_text(serialized, encoding="utf-8")

    trace_path = output_dir / "semantic_reward_budget_trace.csv"
    with trace_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "motion_frame",
                "time_s",
                "activity",
                "alpha",
                "kf_part",
                "kf_rel",
                "kf_dyn",
                "semantic_combined",
                "valid_objective_count",
                "base_positive",
                "base_contribution",
                "semantic_contribution",
                "positive_total",
                "penalty",
                "total",
                "active_event_ids",
                "active_event_names_for_logging_only",
            ],
        )
        writer.writeheader()
        for frame in range(command.time_steps.numel()):
            event_ids = [
                event.event_id
                for event in runtime.events
                if float(result.event_gates[frame, event.event_id]) > 0.0
            ]
            writer.writerow(
                {
                    "motion_frame": frame,
                    "time_s": frame / motion_fps,
                    "activity": float(result.active_gate[frame]),
                    "alpha": float(allocation.alpha[frame]),
                    "kf_part": float(result.part_reward[frame]),
                    "kf_rel": float(result.rel_reward[frame]),
                    "kf_dyn": float(result.dyn_reward[frame]),
                    "semantic_combined": float(result.combined_reward[frame]),
                    "valid_objective_count": float(result.valid_objective_count[frame]),
                    "base_positive": float(allocation.base_positive[frame]),
                    "base_contribution": float(allocation.base_contribution[frame]),
                    "semantic_contribution": float(allocation.semantic_contribution[frame]),
                    "positive_total": float(allocation.positive_total[frame]),
                    "penalty": float(allocation.penalty[frame]),
                    "total": float(allocation.total[frame]),
                    "active_event_ids": "|".join(str(event_id) for event_id in event_ids),
                    "active_event_names_for_logging_only": "|".join(
                        runtime.events[event_id].name for event_id in event_ids
                    ),
                }
            )
    (output_dir / "semantic_gate_trace.csv").write_text(
        trace_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    plt.switch_backend("Agg")
    frames = np.arange(command.time_steps.numel())
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
    axes[0].plot(frames, result.active_gate.numpy(), label="A(t): semantic activity", linewidth=1.8)
    axes[0].plot(frames, allocation.alpha.numpy(), label="alpha(t)", linewidth=1.5)
    axes[0].set_ylabel("Gate / mixture")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="upper right")

    axes[1].plot(frames, allocation.base_contribution.numpy(), label="base positive contribution")
    axes[1].plot(frames, allocation.semantic_contribution.numpy(), label="semantic contribution")
    axes[1].plot(frames, allocation.penalty.numpy(), label="penalty contribution")
    axes[1].plot(frames, allocation.total.numpy(), label="total reward", linestyle="--", linewidth=2.0)
    axes[1].axhline(positive_budget, color="black", alpha=0.35, label="fixed positive budget")
    axes[1].set_xlabel("Motion frame")
    axes[1].set_ylabel("Reward before dt")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="center right")
    fig.suptitle("Semantic fixed-budget reward diagnostic (perfect reference replay)")
    fig.savefig(output_dir / "semantic_reward_budget_diagnostic.png", dpi=180)
    plt.close(fig)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-file", type=Path, default=DEFAULT_MOTION)
    parser.add_argument("--semantic-file", type=Path, default=DEFAULT_SEMANTIC)
    parser.add_argument("--semantic-fps", type=float, default=30.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    print(json.dumps(run(_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
