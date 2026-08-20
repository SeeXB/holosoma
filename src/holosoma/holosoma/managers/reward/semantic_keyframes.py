"""Shared runtime for task-agnostic semantic-keyframe WBT rewards."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from holosoma.config_types.reward import RewardManagerCfg, SemanticKeyframeRewardCfg
from holosoma.utils.path import resolve_data_file_path
from holosoma.utils.rotations import quat_apply, quat_error_magnitude, quat_inverse, quat_mul


@dataclass(frozen=True)
class SemanticEvent:
    """One time-resolved semantic event; ``name`` is diagnostic-only metadata."""

    event_id: int
    name: str
    trigger_time_s: float
    support_start_s: float
    support_end_s: float
    next_trigger_time_s: float
    body_indices: tuple[int, ...]
    body_names: tuple[str, ...]


@dataclass
class SemanticRewardOutput:
    part_reward: torch.Tensor
    rel_reward: torch.Tensor
    dyn_reward: torch.Tensor
    active_gate: torch.Tensor
    part_error: torch.Tensor
    rel_error: torch.Tensor
    dyn_error: torch.Tensor
    ordinary_part_error: torch.Tensor
    ordinary_gate: torch.Tensor
    event_gates: torch.Tensor
    event_part_errors: torch.Tensor
    event_rel_errors: torch.Tensor
    event_dyn_errors: torch.Tensor
    part_valid: torch.Tensor
    rel_valid: torch.Tensor
    dyn_valid: torch.Tensor
    combined_reward: torch.Tensor
    valid_objective_count: torch.Tensor


@dataclass(frozen=True)
class OmniTrackingSigmas:
    """Tracking tolerances inherited from the active Omni baseline preset."""

    position: float
    orientation: float
    linear_velocity: float
    angular_velocity: float


_OMNI_SIGMA_TERMS = {
    "position": "holosoma.managers.reward.terms.wbt:motion_relative_body_position_error_exp",
    "orientation": "holosoma.managers.reward.terms.wbt:motion_relative_body_orientation_error_exp",
    "linear_velocity": "holosoma.managers.reward.terms.wbt:motion_global_body_lin_vel",
    "angular_velocity": "holosoma.managers.reward.terms.wbt:motion_global_body_ang_vel",
}


def infer_omni_tracking_sigmas(reward_cfg: RewardManagerCfg) -> OmniTrackingSigmas:
    """Read semantic tolerances from the original Omni tracking terms."""
    resolved: dict[str, float] = {}
    for field_name, func_path in _OMNI_SIGMA_TERMS.items():
        matches = [term for term in reward_cfg.terms.values() if term.func == func_path]
        if len(matches) != 1 or "sigma" not in matches[0].params:
            raise ValueError(f"Expected exactly one Omni term {func_path!r} with a sigma parameter")
        resolved[field_name] = _as_scalar(matches[0].params["sigma"], name=f"Omni {field_name} sigma")
    return OmniTrackingSigmas(**resolved)


def combine_valid_objectives(
    rewards: tuple[torch.Tensor, ...],
    valid: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Equally average valid normalized objectives without treating INVALID as zero."""
    if not rewards or len(rewards) != len(valid):
        raise ValueError("rewards and valid must be non-empty tuples of equal length")
    reward_stack = torch.stack(rewards, dim=-1)
    valid_stack = torch.stack(valid, dim=-1).to(reward_stack.dtype)
    count = valid_stack.sum(dim=-1)
    combined = (reward_stack * valid_stack).sum(dim=-1) / count.clamp_min(1.0)
    return combined, count


def _as_scalar(value: Any, *, name: str) -> float:
    if hasattr(value, "item"):
        value = value.item()
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value!r}")
    return result


def _coerce_config(config: SemanticKeyframeRewardCfg | dict[str, Any]) -> SemanticKeyframeRewardCfg:
    if isinstance(config, SemanticKeyframeRewardCfg):
        return config
    if isinstance(config, dict):
        return SemanticKeyframeRewardCfg(**config)
    raise TypeError(f"Expected SemanticKeyframeRewardCfg or dict, got {type(config)}")


def _window_records(event: dict[str, Any]) -> list[dict[str, Any]]:
    windows = event.get("windows")
    if windows is None:
        window = event.get("window")
        windows = [window] if isinstance(window, dict) else []
    if not isinstance(windows, list) or not windows:
        raise ValueError("Every semantic event must provide a non-empty window/windows field")
    if not all(isinstance(window, dict) for window in windows):
        raise ValueError("Semantic event windows must be JSON objects")
    return windows


def _safe_log_name(name: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_.-]+", "_", name).strip("_")
    return normalized or "unnamed"


class SemanticKeyframeRuntime:
    """Parse semantic metadata once and evaluate all three rewards together."""

    def __init__(self, config: SemanticKeyframeRewardCfg | dict[str, Any], env: Any):
        self.config = _coerce_config(config)
        self._env = env
        self._cached_key: tuple[torch.Tensor, torch.Tensor] | None = None
        self._cached_output: SemanticRewardOutput | None = None
        self.events: tuple[SemanticEvent, ...] = ()
        self.semantic_fps: float | None = None
        self.motion_fps: float | None = None
        self.tracking_sigmas: OmniTrackingSigmas | None = None

        self._validate_numeric_config()
        if not self.config.enabled:
            return
        if not self.config.semantic_file:
            raise ValueError("semantic_file must be explicitly configured when semantic rewards are enabled")

        motion_command = self._motion_command()
        self.motion_fps = _as_scalar(motion_command.motion.fps, name="motion_fps")
        reward_manager = getattr(self._env, "reward_manager", None)
        if reward_manager is None or not isinstance(reward_manager.cfg, RewardManagerCfg):
            raise RuntimeError("Semantic rewards require the active RewardManagerCfg to inherit Omni sigmas")
        self.tracking_sigmas = infer_omni_tracking_sigmas(reward_manager.cfg)
        semantic_path = Path(resolve_data_file_path(self.config.semantic_file))
        if not semantic_path.is_file():
            raise FileNotFoundError(f"Semantic keyframe JSON does not exist: {semantic_path}")
        payload = json.loads(semantic_path.read_text(encoding="utf-8"))
        self.semantic_fps = self._resolve_semantic_fps(payload)
        self.events = self._parse_events(payload, motion_command)

    def _validate_numeric_config(self) -> None:
        sigma_time = float(self.config.sigma_time)
        if not math.isfinite(sigma_time) or sigma_time <= 0.0:
            raise ValueError(f"sigma_time must be finite and positive, got {sigma_time}")
        if self.config.enabled and not any(
            (self.config.enable_part, self.config.enable_rel, self.config.enable_dyn)
        ):
            raise ValueError("At least one semantic objective must be enabled")

    def _motion_command(self) -> Any:
        command = self._env.command_manager.get_state("motion_command")
        if command is None:
            raise RuntimeError("Semantic WBT rewards require a configured motion_command")
        return command

    def _resolve_semantic_fps(self, payload: dict[str, Any]) -> float:
        json_fps = payload.get("fps")
        configured_fps = self.config.semantic_fps
        if json_fps is None and configured_fps is None:
            raise ValueError("Semantic JSON has no fps metadata; semantic_fps must be explicitly configured")
        if json_fps is not None and configured_fps is not None:
            parsed_json_fps = _as_scalar(json_fps, name="semantic JSON fps")
            parsed_config_fps = _as_scalar(configured_fps, name="semantic_fps")
            if not math.isclose(parsed_json_fps, parsed_config_fps, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError(
                    f"semantic_fps={parsed_config_fps} disagrees with semantic JSON fps={parsed_json_fps}"
                )
            return parsed_config_fps
        return _as_scalar(json_fps if json_fps is not None else configured_fps, name="semantic_fps")

    def _mapped_body_names(self, semantic_part: str) -> list[str]:
        mapped = self.config.body_part_mapping.get(semantic_part, [semantic_part])
        if isinstance(mapped, str):
            mapped = [mapped]
        if not mapped:
            raise ValueError(f"Semantic body part {semantic_part!r} maps to no robot bodies")
        return list(mapped)

    def _parse_events(self, payload: dict[str, Any], motion_command: Any) -> tuple[SemanticEvent, ...]:
        raw_events = payload.get("events")
        if not isinstance(raw_events, list):
            raise ValueError("Semantic JSON must contain an events list")
        tracked_names = list(motion_command.motion_cfg.body_names_to_track)
        expanded: list[dict[str, Any]] = []
        for source_index, raw_event in enumerate(raw_events):
            if not isinstance(raw_event, dict):
                raise ValueError("Each semantic event must be a JSON object")
            parts = raw_event.get("body_parts")
            if not isinstance(parts, list) or not parts or not all(isinstance(part, str) for part in parts):
                raise ValueError("Each semantic event must provide a non-empty string body_parts list")
            mapped_names: list[str] = []
            for part in parts:
                mapped_names.extend(self._mapped_body_names(part))
            mapped_names = list(dict.fromkeys(mapped_names))
            missing = [name for name in mapped_names if name not in tracked_names]
            if missing:
                raise ValueError(
                    f"Semantic body mapping references bodies not tracked by WBT: {missing}; tracked={tracked_names}"
                )
            for window_index, window in enumerate(_window_records(raw_event)):
                trigger_frame = window.get("trigger_frame", raw_event.get("trigger_frame"))
                start_frame = window.get("start_frame")
                end_frame = window.get("end_frame")
                if trigger_frame is None or start_frame is None or end_frame is None:
                    raise ValueError("Semantic windows require start_frame, end_frame, and trigger_frame")
                trigger_frame = int(trigger_frame)
                start_frame = int(start_frame)
                end_frame = int(end_frame)
                if start_frame > trigger_frame or trigger_frame > end_frame:
                    raise ValueError(
                        f"Semantic window must satisfy start <= trigger <= end, got "
                        f"{start_frame}, {trigger_frame}, {end_frame}"
                    )
                expanded.append(
                    {
                        "source_index": source_index,
                        "window_index": window_index,
                        "name": str(raw_event.get("event", f"event_{source_index}")),
                        "trigger_time_s": trigger_frame / self.semantic_fps,
                        "support_start_s": start_frame / self.semantic_fps,
                        "support_end_s": end_frame / self.semantic_fps,
                        "body_names": tuple(mapped_names),
                        "body_indices": tuple(tracked_names.index(name) for name in mapped_names),
                    }
                )
        expanded.sort(key=lambda item: (item["trigger_time_s"], item["source_index"], item["window_index"]))
        events: list[SemanticEvent] = []
        for event_id, item in enumerate(expanded):
            next_trigger = (
                float(expanded[event_id + 1]["trigger_time_s"])
                if event_id + 1 < len(expanded)
                else math.inf
            )
            events.append(
                SemanticEvent(
                    event_id=event_id,
                    name=item["name"],
                    trigger_time_s=float(item["trigger_time_s"]),
                    support_start_s=float(item["support_start_s"]),
                    support_end_s=float(item["support_end_s"]),
                    next_trigger_time_s=next_trigger,
                    body_indices=item["body_indices"],
                    body_names=item["body_names"],
                )
            )
        return tuple(events)

    def trigger_rl_frame(self, event_id: int) -> int:
        """Map semantic trigger time to the nearest motion frame."""
        if self.motion_fps is None:
            raise RuntimeError("Runtime is disabled and has no motion FPS")
        return round(self.events[event_id].trigger_time_s * self.motion_fps)

    def gates_at_times(self, times_s: torch.Tensor) -> torch.Tensor:
        """Return transition-truncated Gaussian gates for arbitrary times."""
        if not self.config.enabled or not self.events:
            return torch.zeros((*times_s.shape, 0), dtype=times_s.dtype, device=times_s.device)
        gates = []
        sigma = self.config.sigma_time
        for event in self.events:
            gate = torch.exp(-torch.square(times_s - event.trigger_time_s) / (2.0 * sigma**2))
            support = (times_s >= event.support_start_s) & (times_s <= event.support_end_s)
            if self.config.transition_truncation and math.isfinite(event.next_trigger_time_s):
                support &= times_s < event.next_trigger_time_s
            gates.append(gate * support.to(gate.dtype))
        return torch.stack(gates, dim=-1)

    def _current_times_s(self, motion_command: Any) -> torch.Tensor:
        time_steps = motion_command.time_steps
        if hasattr(motion_command.motion, "motion_start_idx") and hasattr(motion_command, "motion_ids"):
            starts = motion_command.motion.motion_start_idx[motion_command.motion_ids]
            local_frames = time_steps - starts
        else:
            local_frames = time_steps
        return local_frames.to(dtype=torch.float32) / float(self.motion_fps)

    def _cache_key(self, motion_command: Any) -> tuple[torch.Tensor, torch.Tensor] | None:
        episode_lengths = getattr(self._env, "episode_length_buf", None)
        if not isinstance(episode_lengths, torch.Tensor):
            return None
        return episode_lengths.detach().clone(), motion_command.time_steps.detach().clone()

    @staticmethod
    def _same_cache_key(
        left: tuple[torch.Tensor, torch.Tensor] | None,
        right: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> bool:
        return bool(
            left is not None
            and right is not None
            and torch.equal(left[0], right[0])
            and torch.equal(left[1], right[1])
        )

    @staticmethod
    def _aggregate_events(
        gates: torch.Tensor,
        values: torch.Tensor,
        valid_events: torch.Tensor,
    ) -> torch.Tensor:
        valid_gates = gates * valid_events.to(gates.dtype)
        gate_sum = valid_gates.sum(dim=-1)
        eps = torch.finfo(gates.dtype).eps
        return (valid_gates * values).sum(dim=-1) / gate_sum.clamp_min(eps)

    def _empty_output(self, num_envs: int, device: torch.device) -> SemanticRewardOutput:
        zeros = torch.zeros(num_envs, dtype=torch.float32, device=device)
        event_zeros = torch.zeros((num_envs, 0), dtype=torch.float32, device=device)
        return SemanticRewardOutput(
            part_reward=zeros,
            rel_reward=zeros.clone(),
            dyn_reward=zeros.clone(),
            active_gate=zeros.clone(),
            part_error=zeros.clone(),
            rel_error=zeros.clone(),
            dyn_error=zeros.clone(),
            ordinary_part_error=zeros.clone(),
            ordinary_gate=torch.ones_like(zeros),
            event_gates=event_zeros,
            event_part_errors=event_zeros.clone(),
            event_rel_errors=event_zeros.clone(),
            event_dyn_errors=event_zeros.clone(),
            part_valid=torch.zeros(num_envs, dtype=torch.bool, device=device),
            rel_valid=torch.zeros(num_envs, dtype=torch.bool, device=device),
            dyn_valid=torch.zeros(num_envs, dtype=torch.bool, device=device),
            combined_reward=zeros.clone(),
            valid_objective_count=zeros.clone(),
        )

    def evaluate(self) -> SemanticRewardOutput:
        motion_command = self._motion_command()
        num_envs = int(motion_command.time_steps.shape[0])
        device = motion_command.time_steps.device
        if not self.config.enabled:
            return self._empty_output(num_envs, device)

        cache_key = self._cache_key(motion_command)
        if self._same_cache_key(cache_key, self._cached_key) and self._cached_output is not None:
            return self._cached_output

        gates = self.gates_at_times(self._current_times_s(motion_command))
        event_count = len(self.events)
        event_part_rewards = torch.zeros((num_envs, event_count), device=device)
        event_rel_rewards = torch.zeros_like(event_part_rewards)
        event_dyn_rewards = torch.zeros_like(event_part_rewards)
        event_part_errors = torch.zeros_like(event_part_rewards)
        event_rel_errors = torch.zeros_like(event_part_rewards)
        event_dyn_errors = torch.zeros_like(event_part_rewards)
        part_event_valid = torch.ones((num_envs, event_count), dtype=torch.bool, device=device)
        dyn_event_valid = torch.ones_like(part_event_valid)

        ref_pos = motion_command.body_pos_relative_w
        ref_quat = motion_command.body_quat_relative_w
        robot_pos = motion_command.robot_body_pos_w
        robot_quat = motion_command.robot_body_quat_w
        ref_lin_vel = motion_command.body_lin_vel_w
        ref_ang_vel = motion_command.body_ang_vel_w
        robot_lin_vel = motion_command.robot_body_lin_vel_w
        robot_ang_vel = motion_command.robot_body_ang_vel_w

        has_external_entity = bool(getattr(motion_command.motion, "has_object", False))
        rel_event_valid = torch.full_like(part_event_valid, has_external_entity)
        if has_external_entity:
            object_ref_pos = motion_command.object_pos_w
            object_ref_quat = motion_command.object_quat_w
            object_pos = motion_command.simulator_object_pos_w
            object_quat = motion_command.simulator_object_quat_w

        if self.tracking_sigmas is None:
            raise RuntimeError("Enabled semantic runtime has no inherited Omni tracking sigmas")
        sigmas = self.tracking_sigmas

        for event_index, event in enumerate(self.events):
            indices = torch.tensor(event.body_indices, dtype=torch.long, device=device)
            pos_error = torch.sum(torch.square(ref_pos[:, indices] - robot_pos[:, indices]), dim=-1)
            rot_error = torch.square(quat_error_magnitude(ref_quat[:, indices], robot_quat[:, indices]))
            part_pos_reward = torch.exp(-pos_error / sigmas.position**2).mean(dim=-1)
            part_rot_reward = torch.exp(-rot_error / sigmas.orientation**2).mean(dim=-1)
            part_reward = 0.5 * (part_pos_reward + part_rot_reward)
            part_energy = 0.5 * (
                (pos_error / sigmas.position**2).mean(dim=-1)
                + (rot_error / sigmas.orientation**2).mean(dim=-1)
            )
            event_part_errors[:, event_index] = part_energy
            event_part_rewards[:, event_index] = part_reward

            lin_error = torch.sum(torch.square(ref_lin_vel[:, indices] - robot_lin_vel[:, indices]), dim=-1)
            ang_error = torch.sum(torch.square(ref_ang_vel[:, indices] - robot_ang_vel[:, indices]), dim=-1)
            dyn_lin_reward = torch.exp(-lin_error / sigmas.linear_velocity**2).mean(dim=-1)
            dyn_ang_reward = torch.exp(-ang_error / sigmas.angular_velocity**2).mean(dim=-1)
            dyn_reward = 0.5 * (dyn_lin_reward + dyn_ang_reward)
            dyn_energy = 0.5 * (
                (lin_error / sigmas.linear_velocity**2).mean(dim=-1)
                + (ang_error / sigmas.angular_velocity**2).mean(dim=-1)
            )
            event_dyn_errors[:, event_index] = dyn_energy
            event_dyn_rewards[:, event_index] = dyn_reward

            if has_external_entity:
                body_ref_pos = motion_command.body_pos_w[:, indices]
                body_ref_quat = motion_command.body_quat_w[:, indices]
                body_pos = robot_pos[:, indices]
                body_quat = robot_quat[:, indices]
                edge_count = int(indices.numel())
                ref_entity_quat = object_ref_quat[:, None, :].expand(-1, edge_count, -1)
                entity_quat = object_quat[:, None, :].expand(-1, edge_count, -1)
                ref_delta = quat_apply(
                    quat_inverse(ref_entity_quat, w_last=True),
                    body_ref_pos - object_ref_pos[:, None, :],
                    w_last=True,
                )
                delta = quat_apply(
                    quat_inverse(entity_quat, w_last=True),
                    body_pos - object_pos[:, None, :],
                    w_last=True,
                )
                rel_pos_error = torch.sum(torch.square(delta - ref_delta), dim=-1)
                ref_relative_quat = quat_mul(
                    quat_inverse(ref_entity_quat, w_last=True), body_ref_quat, w_last=True
                )
                relative_quat = quat_mul(quat_inverse(entity_quat, w_last=True), body_quat, w_last=True)
                rel_rot_error = torch.square(quat_error_magnitude(ref_relative_quat, relative_quat))
                rel_pos_reward = torch.exp(-rel_pos_error / sigmas.position**2).mean(dim=-1)
                rel_rot_reward = torch.exp(-rel_rot_error / sigmas.orientation**2).mean(dim=-1)
                rel_reward = 0.5 * (rel_pos_reward + rel_rot_reward)
                rel_energy = 0.5 * (
                    (rel_pos_error / sigmas.position**2).mean(dim=-1)
                    + (rel_rot_error / sigmas.orientation**2).mean(dim=-1)
                )
                event_rel_errors[:, event_index] = rel_energy
                event_rel_rewards[:, event_index] = rel_reward

        activity = gates.sum(dim=-1).clamp(min=0.0, max=1.0)
        part_reward = self._aggregate_events(gates, event_part_rewards, part_event_valid)
        rel_reward = self._aggregate_events(gates, event_rel_rewards, rel_event_valid)
        dyn_reward = self._aggregate_events(gates, event_dyn_rewards, dyn_event_valid)
        part_error = self._aggregate_events(gates, event_part_errors, part_event_valid)
        rel_error = self._aggregate_events(gates, event_rel_errors, rel_event_valid)
        dyn_error = self._aggregate_events(gates, event_dyn_errors, dyn_event_valid)
        has_events = event_count > 0
        part_valid = torch.full((num_envs,), has_events and self.config.enable_part, dtype=torch.bool, device=device)
        rel_valid = torch.full(
            (num_envs,),
            has_events and has_external_entity and self.config.enable_rel,
            dtype=torch.bool,
            device=device,
        )
        dyn_valid = torch.full((num_envs,), has_events and self.config.enable_dyn, dtype=torch.bool, device=device)
        combined_reward, valid_objective_count = combine_valid_objectives(
            (part_reward, rel_reward, dyn_reward),
            (part_valid, rel_valid, dyn_valid),
        )
        ordinary_gate = (activity <= torch.finfo(activity.dtype).eps).to(activity.dtype)
        ordinary_error = torch.norm(ref_pos - robot_pos, dim=-1).mean(dim=-1) * ordinary_gate
        output = SemanticRewardOutput(
            part_reward=part_reward,
            rel_reward=rel_reward,
            dyn_reward=dyn_reward,
            active_gate=activity,
            part_error=part_error,
            rel_error=rel_error,
            dyn_error=dyn_error,
            ordinary_part_error=ordinary_error,
            ordinary_gate=ordinary_gate,
            event_gates=gates,
            event_part_errors=event_part_errors,
            event_rel_errors=event_rel_errors,
            event_dyn_errors=event_dyn_errors,
            part_valid=part_valid,
            rel_valid=rel_valid,
            dyn_valid=dyn_valid,
            combined_reward=combined_reward,
            valid_objective_count=valid_objective_count,
        )
        self._cached_key = cache_key
        self._cached_output = output
        return output

    def logging_metrics(self) -> dict[str, torch.Tensor]:
        output = self.evaluate()
        metrics = {
            "semantic/activity": output.active_gate,
            "semantic/part": output.part_reward,
            "semantic/rel": output.rel_reward,
            "semantic/dyn": output.dyn_reward,
            "semantic/combined": output.combined_reward,
            "semantic/valid_objective_count": output.valid_objective_count,
            # Compatibility aliases retained for existing dashboards.
            "semantic/part_reward": output.part_reward,
            "semantic/rel_reward": output.rel_reward,
            "semantic/dyn_reward": output.dyn_reward,
            "semantic/active_gate": output.active_gate,
            "semantic/keyframe_part_error": output.part_error,
            "semantic/keyframe_rel_error": output.rel_error,
            "semantic/keyframe_dyn_error": output.dyn_error,
            "semantic/ordinary_part_error": output.ordinary_part_error,
            "semantic/ordinary_gate": output.ordinary_gate,
        }
        for event in self.events:
            prefix = f"semantic/event_{event.event_id}_{_safe_log_name(event.name)}"
            metrics[f"{prefix}/gate"] = output.event_gates[:, event.event_id]
            metrics[f"{prefix}/part_error"] = output.event_part_errors[:, event.event_id]
            metrics[f"{prefix}/rel_error"] = output.event_rel_errors[:, event.event_id]
            metrics[f"{prefix}/dyn_error"] = output.event_dyn_errors[:, event.event_id]
        return metrics


def get_semantic_keyframe_runtime(
    env: Any,
    semantic_config: SemanticKeyframeRewardCfg | dict[str, Any],
) -> SemanticKeyframeRuntime:
    """Return the one semantic runtime shared by all configured reward terms."""
    config = _coerce_config(semantic_config)
    runtime = getattr(env, "_semantic_keyframe_reward_runtime", None)
    if runtime is None:
        runtime = SemanticKeyframeRuntime(config, env)
        env._semantic_keyframe_reward_runtime = runtime
    elif runtime.config != config:
        raise ValueError("All semantic reward terms must use the same SemanticKeyframeRewardCfg")
    return runtime
