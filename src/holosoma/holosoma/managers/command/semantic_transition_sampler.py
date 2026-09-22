"""Semantic-interval reference-timestep sampling for WBT.

The sampler is deliberately independent from rewards and termination terms.  A
semantic JSON file partitions a single reference motion into every critical
event window and the ordinary gaps around them.  Together these disjoint,
half-open intervals cover the complete sampleable timeline.  Interval failure
statistics are updated from the existing termination signal supplied by
:class:`MotionCommand`.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from holosoma.utils.path import resolve_data_file_path


SAMPLING_MODES = ("original_adaptive", "semantic_uniform", "semantic_adaptive")


@dataclass(frozen=True)
class SemanticTransition:
    """One critical-event interval or ordinary complement interval.

    ``target_step`` is the exclusive sampling boundary and the step that an
    episode must reach to count as successful.  The historical class name is
    retained to avoid breaking callers and metric dashboards.
    """

    transition_id: int
    start_time_s: float
    target_time_s: float
    start_step: int
    target_step: int
    source_event_name: str
    target_event_name: str
    interval_kind: str
    event_name: str | None

    @property
    def is_critical(self) -> bool:
        return self.interval_kind == "critical_event"


def _positive_fps(value: Any, *, field_name: str) -> float:
    if hasattr(value, "item"):
        value = value.item()
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive finite number, got {value!r}") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{field_name} must be a positive finite number, got {value!r}")
    return result


def _integer_frame(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer, got {value!r}")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative, got {value!r}")
    return value


def _event_windows(raw_event: dict[str, Any]) -> list[dict[str, Any]]:
    windows = raw_event.get("windows")
    if windows is None:
        window = raw_event.get("window")
        windows = [window] if isinstance(window, dict) else [raw_event]
    if not isinstance(windows, list) or not windows or not all(isinstance(item, dict) for item in windows):
        raise ValueError("Every semantic event must provide a non-empty windows list")
    return windows


def load_semantic_transitions(
    semantic_file: str | Path,
    *,
    motion_fps: float,
    motion_time_step_total: int,
    semantic_fps: float | None = None,
) -> tuple[SemanticTransition, ...]:
    """Build critical event windows plus ordinary gaps over the whole motion.

    The mapping is always performed in seconds.  If both the JSON metadata and
    an explicit config value are present they must agree; missing FPS metadata
    is therefore never silently replaced by a hard-coded 30/50 FPS assumption.
    Semantic windows use ``[start_frame, end_frame)`` boundaries after FPS
    mapping.  The final motion step is a reachable target rather than a reset
    start, matching :class:`MotionCommand`'s existing last-step guard.
    """

    if not str(semantic_file).strip():
        raise ValueError("semantic_file must be configured for semantic sampling")
    source = Path(resolve_data_file_path(str(semantic_file)))
    if not source.is_file():
        raise FileNotFoundError(f"Semantic sampling file does not exist: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("events"), list):
        raise ValueError(f"{source}: top-level 'events' must be a list")

    json_fps_raw = payload.get("fps")
    if json_fps_raw is None and semantic_fps is None:
        raise ValueError(f"{source}: semantic FPS is missing; provide JSON 'fps' or semantic_fps explicitly")
    if json_fps_raw is not None:
        json_fps = _positive_fps(json_fps_raw, field_name="semantic JSON fps")
    else:
        json_fps = None
    if semantic_fps is not None:
        configured_fps = _positive_fps(semantic_fps, field_name="semantic_fps")
        if json_fps is not None and not math.isclose(json_fps, configured_fps, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(
                f"{source}: semantic_fps={configured_fps} disagrees with semantic JSON fps={json_fps}"
            )
        semantic_rate = configured_fps
    else:
        semantic_rate = json_fps
    assert semantic_rate is not None
    motion_rate = _positive_fps(motion_fps, field_name="motion_fps")
    if motion_time_step_total < 2:
        raise ValueError("motion_time_step_total must contain at least two steps")

    # MotionCommand never retains the final step as a reset start.  Partition
    # [0, sampleable_end) and use sampleable_end as the last reachable target.
    sampleable_end = motion_time_step_total - 1
    events: list[tuple[int, int, int, str]] = []
    for event_index, raw_event in enumerate(payload["events"]):
        if not isinstance(raw_event, dict):
            raise ValueError(f"{source}: events[{event_index}] must be an object")
        event_name = raw_event.get("event", f"event_{event_index}")
        if not isinstance(event_name, str) or not event_name.strip():
            raise ValueError(f"{source}: events[{event_index}].event must be a non-empty string")
        for window_index, window in enumerate(_event_windows(raw_event)):
            start_frame = _integer_frame(
                window.get("start_frame", raw_event.get("start_frame")),
                field_name=f"events[{event_index}].windows[{window_index}].start_frame",
            )
            end_frame = _integer_frame(
                window.get("end_frame", raw_event.get("end_frame")),
                field_name=f"events[{event_index}].windows[{window_index}].end_frame",
            )
            trigger_frame = _integer_frame(
                window.get("trigger_frame", raw_event.get("trigger_frame")),
                field_name=f"events[{event_index}].windows[{window_index}].trigger_frame",
            )
            if not start_frame <= trigger_frame <= end_frame:
                raise ValueError(
                    f"{source}: event {event_name!r} must satisfy start_frame <= trigger_frame <= end_frame"
                )
            start_step = round((start_frame / semantic_rate) * motion_rate)
            target_step = round((end_frame / semantic_rate) * motion_rate)
            if target_step <= start_step:
                raise ValueError(
                    f"{source}: event {event_name!r} window [{start_frame}, {end_frame}] maps to empty "
                    f"motion interval [{start_step}, {target_step})"
                )
            if start_step < 0 or target_step > sampleable_end:
                raise ValueError(
                    f"{source}: event {event_name!r} maps to [{start_step}, {target_step}), outside "
                    f"sampleable motion range [0, {sampleable_end})"
                )
            events.append((start_step, target_step, event_index * 100000 + window_index, event_name))
    events.sort(key=lambda item: (item[0], item[1], item[2]))
    if not events:
        raise ValueError(f"{source}: at least one semantic event window is required")

    transitions: list[SemanticTransition] = []

    def append_interval(start_step: int, target_step: int, *, kind: str, event_name: str | None,
                        source_name: str, target_name: str) -> None:
        transition_id = len(transitions)
        transitions.append(SemanticTransition(
            transition_id=transition_id,
            start_time_s=start_step / motion_rate,
            target_time_s=target_step / motion_rate,
            start_step=start_step,
            target_step=target_step,
            source_event_name=source_name,
            target_event_name=target_name,
            interval_kind=kind,
            event_name=event_name,
        ))

    cursor = 0
    previous_name = "motion_start"
    for start_step, target_step, _, event_name in events:
        if start_step < cursor:
            raise ValueError(
                f"{source}: semantic event windows overlap after FPS mapping; event {event_name!r} "
                f"starts at {start_step} before the previous interval ends at {cursor}"
            )
        if cursor < start_step:
            append_interval(
                cursor, start_step, kind="ordinary", event_name=None,
                source_name=previous_name, target_name=event_name,
            )
        append_interval(
            start_step, target_step, kind="critical_event", event_name=event_name,
            source_name=event_name, target_name=event_name,
        )
        cursor = target_step
        previous_name = event_name
    if cursor < sampleable_end:
        append_interval(
            cursor, sampleable_end, kind="ordinary", event_name=None,
            source_name=previous_name, target_name="motion_end",
        )
    return tuple(transitions)


class SemanticTransitionSampler:
    """Sample full-timeline intervals and maintain per-interval failure EMAs."""

    def __init__(
        self,
        *,
        motion_time_step_total: int,
        num_envs: int,
        device: str,
        motion_fps: float,
        semantic_file: str | Path,
        semantic_fps: float | None,
        sampling_mode: str,
        adaptive_uniform_ratio: float = 0.1,
        adaptive_alpha: float = 0.001,
        critical_interval_weight: float = 2.0,
    ):
        if sampling_mode not in ("semantic_uniform", "semantic_adaptive"):
            raise ValueError(f"SemanticTransitionSampler requires semantic mode, got {sampling_mode!r}")
        if not 0.0 <= adaptive_uniform_ratio <= 1.0:
            raise ValueError("adaptive_uniform_ratio must lie in [0, 1]")
        if not 0.0 < adaptive_alpha <= 1.0:
            raise ValueError("adaptive_alpha must lie in (0, 1]")
        if not math.isfinite(critical_interval_weight) or critical_interval_weight <= 1.0:
            raise ValueError("critical_interval_weight must be finite and greater than 1")
        self.device = device
        self.motion_time_step_total = motion_time_step_total
        self.sampleable_step_total = motion_time_step_total - 1
        self.num_envs = num_envs
        self.sampling_mode = sampling_mode
        self.adaptive_uniform_ratio = float(adaptive_uniform_ratio)
        self.adaptive_alpha = float(adaptive_alpha)
        self.critical_interval_weight = float(critical_interval_weight)
        self.transitions = load_semantic_transitions(
            semantic_file,
            motion_fps=motion_fps,
            motion_time_step_total=motion_time_step_total,
            semantic_fps=semantic_fps,
        )
        self.num_transitions = len(self.transitions)
        self._start_steps = torch.tensor(
            [item.start_step for item in self.transitions], dtype=torch.long, device=device
        )
        self._target_steps = torch.tensor(
            [item.target_step for item in self.transitions], dtype=torch.long, device=device
        )
        self._interval_spans = self._target_steps - self._start_steps
        self._critical_mask = torch.tensor(
            [item.is_critical for item in self.transitions], dtype=torch.bool, device=device
        )
        if torch.any(self._interval_spans <= 0):
            raise ValueError("Semantic intervals must all have positive motion-step spans")
        self.failure_score = torch.zeros(self.num_transitions, dtype=torch.float32, device=device)
        self.success_count = torch.zeros(self.num_transitions, dtype=torch.long, device=device)
        self.failure_count = torch.zeros(self.num_transitions, dtype=torch.long, device=device)
        self.sample_count = torch.zeros(self.num_transitions, dtype=torch.long, device=device)
        self.failure_frame_histogram = torch.zeros(motion_time_step_total, dtype=torch.long, device=device)
        self.failure_transition_histogram = torch.zeros(
            (self.num_transitions, motion_time_step_total), dtype=torch.long, device=device
        )
        self.global_uniform_count = torch.zeros((), dtype=torch.long, device=device)
        self.config_error_count = torch.zeros((), dtype=torch.long, device=device)
        self.assigned_transition_id = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        self.assigned_target_step = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        self.transition_resolved = torch.zeros(num_envs, dtype=torch.bool, device=device)

    @property
    def sampling_probabilities(self) -> torch.Tensor:
        span_mass = self._interval_spans.to(dtype=torch.float32)
        if self.sampling_mode == "semantic_uniform":
            # Uniform per frame, not per interval.  Splitting or joining an
            # ordinary gap therefore cannot change the baseline distribution.
            return span_mass / span_mass.sum()
        semantic_weight = torch.where(
            self._critical_mask,
            torch.full_like(span_mass, self.critical_interval_weight),
            torch.ones_like(span_mass),
        )
        prior_mass = span_mass * semantic_weight
        prior = prior_mass / prior_mass.sum()
        # At cold start this prior already weights each critical frame more
        # heavily.  Failure EMA then multiplies the prior; the configured
        # exploration term keeps every interval sampleable.
        scores = prior * (self.failure_score + self.adaptive_uniform_ratio)
        if not torch.any(scores > 0):
            return prior
        scores = scores + torch.finfo(scores.dtype).eps * prior
        return scores / scores.sum()

    def sample(self, num_samples: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return frame indices, transition IDs, and global-fallback mask."""
        if num_samples < 0:
            raise ValueError("num_samples must be non-negative")
        global_mask = torch.rand(num_samples, device=self.device) < self.adaptive_uniform_ratio
        transition_ids = torch.full((num_samples,), -1, dtype=torch.long, device=self.device)
        frame_indices = torch.empty(num_samples, dtype=torch.long, device=self.device)
        global_count = int(global_mask.sum().item())
        if global_count:
            frame_indices[global_mask] = torch.randint(
                0, self.sampleable_step_total, (global_count,), device=self.device
            )
            self.global_uniform_count += global_count
        transition_mask = ~global_mask
        transition_count = int(transition_mask.sum().item())
        if transition_count:
            sampled_ids = torch.multinomial(self.sampling_probabilities, transition_count, replacement=True)
            starts = self._start_steps[sampled_ids]
            spans = self._interval_spans[sampled_ids]
            offsets = (torch.rand(transition_count, device=self.device) * spans.float()).long()
            frame_indices[transition_mask] = starts + offsets.clamp_max(spans - 1)
            transition_ids[transition_mask] = sampled_ids
            self.sample_count.scatter_add_(0, sampled_ids, torch.ones_like(sampled_ids))
        return frame_indices, transition_ids, global_mask

    def assign(self, env_ids: torch.Tensor, transition_ids: torch.Tensor) -> None:
        if env_ids.numel() != transition_ids.numel():
            raise ValueError("env_ids and transition_ids must have equal lengths")
        self.assigned_transition_id[env_ids] = transition_ids
        self.transition_resolved[env_ids] = False
        valid = transition_ids >= 0
        self.assigned_target_step[env_ids] = torch.where(
            valid,
            self._target_steps[transition_ids.clamp_min(0)],
            torch.full_like(transition_ids, -1),
        )

    def clear_assignment(self, env_ids: torch.Tensor) -> None:
        self.assigned_transition_id[env_ids] = -1
        self.assigned_target_step[env_ids] = -1
        self.transition_resolved[env_ids] = False

    def _record(
        self,
        env_ids: torch.Tensor,
        *,
        failed: torch.Tensor,
        failure_steps: torch.Tensor | None = None,
    ) -> None:
        transition_ids = self.assigned_transition_id[env_ids]
        valid = (transition_ids >= 0) & ~self.transition_resolved[env_ids]
        if not torch.any(valid):
            return
        transition_ids = transition_ids[valid]
        failed = failed[valid]
        counts = torch.bincount(transition_ids, minlength=self.num_transitions).to(dtype=torch.float32)
        failed_counts = torch.bincount(
            transition_ids, weights=failed.to(dtype=torch.float32), minlength=self.num_transitions
        )
        successful_counts = counts - failed_counts
        active = counts > 0
        self.failure_count += failed_counts.to(dtype=torch.long)
        self.success_count += successful_counts.to(dtype=torch.long)
        batch_failure_rate = failed_counts / counts.clamp_min(1.0)
        self.failure_score[active] = (
            (1.0 - self.adaptive_alpha) * self.failure_score[active]
            + self.adaptive_alpha * batch_failure_rate[active]
        )
        failed_env_ids = env_ids[valid][failed]
        if failed_env_ids.numel():
            if failure_steps is None:
                failure_steps = self.assigned_target_step[env_ids][valid][failed]
            else:
                failure_steps = failure_steps[valid][failed]
            failed_steps = failure_steps.clamp(min=0, max=self.motion_time_step_total - 1)
            self.failure_frame_histogram.scatter_add_(
                0, failed_steps, torch.ones_like(failed_steps, dtype=torch.long)
            )
            failed_transition_ids = self.assigned_transition_id[failed_env_ids]
            flat_indices = failed_transition_ids * self.motion_time_step_total + failed_steps
            flat_histogram = self.failure_transition_histogram.view(-1)
            flat_histogram.scatter_add_(0, flat_indices, torch.ones_like(flat_indices, dtype=torch.long))
        valid_env_ids = env_ids[valid]
        self.transition_resolved[valid_env_ids] = True

    def resolve_before_reset(
        self,
        env_ids: torch.Tensor,
        *,
        terminated: torch.Tensor,
        timeouts: torch.Tensor,
        current_steps: torch.Tensor,
    ) -> None:
        """Resolve an assignment before a termination-driven environment reset."""
        if env_ids.numel() == 0:
            return
        target = self.assigned_target_step[env_ids]
        assigned = (self.assigned_transition_id[env_ids] >= 0) & ~self.transition_resolved[env_ids]
        reached = assigned & (current_steps >= target)
        early_failure = assigned & ~reached & terminated
        timeout_before_target = assigned & ~reached & ~terminated & timeouts
        unresolved_external_reset = assigned & ~reached & ~terminated & ~timeouts
        record_mask = reached | early_failure
        self._record(
            env_ids[record_mask],
            failed=early_failure[record_mask],
            failure_steps=current_steps[record_mask],
        )
        config_errors = timeout_before_target | unresolved_external_reset
        if torch.any(config_errors):
            self.config_error_count += int(config_errors.sum().item())
            self.transition_resolved[env_ids[config_errors]] = True

    def mark_reached(self, env_ids: torch.Tensor, current_steps: torch.Tensor) -> None:
        """Record success once an assigned target timestep has been crossed."""
        if env_ids.numel() == 0:
            return
        assigned = (self.assigned_transition_id[env_ids] >= 0) & ~self.transition_resolved[env_ids]
        reached = assigned & (current_steps >= self.assigned_target_step[env_ids])
        self._record(
            env_ids[reached],
            failed=torch.zeros(int(reached.sum().item()), dtype=torch.bool, device=self.device),
        )

    def mark_motion_end(self, env_ids: torch.Tensor, current_steps: torch.Tensor) -> None:
        """Resolve assignments when the reference clip ends unexpectedly."""
        if env_ids.numel() == 0:
            return
        assigned = (self.assigned_transition_id[env_ids] >= 0) & ~self.transition_resolved[env_ids]
        if not torch.any(assigned):
            return
        reached = current_steps[assigned] >= self.assigned_target_step[env_ids][assigned]
        assigned_env_ids = env_ids[assigned]
        self._record(
            assigned_env_ids[reached],
            failed=torch.zeros(int(reached.sum().item()), dtype=torch.bool, device=self.device),
        )
        unresolved = assigned_env_ids[~reached]
        if unresolved.numel():
            self.config_error_count += unresolved.numel()
            self.transition_resolved[unresolved] = True

    def metrics(self) -> dict[str, torch.Tensor]:
        probabilities = self.sampling_probabilities
        sample_float = self.sample_count.float()
        total_samples = sample_float.sum()
        failure_rate = self.failure_count.float() / sample_float.clamp_min(1.0)
        success_rate = self.success_count.float() / sample_float.clamp_min(1.0)
        entropy = -(probabilities * (probabilities + 1e-12).log()).sum()
        entropy = entropy / math.log(max(self.num_transitions, 2))
        top_probability, top_transition = probabilities.max(dim=0)
        global_fraction = self.global_uniform_count.float() / (
            self.global_uniform_count.float() + total_samples
        ).clamp_min(1.0)
        output: dict[str, torch.Tensor] = {
            "semantic_sampling/interval_count": torch.tensor(
                float(self.num_transitions), device=self.device
            ),
            "semantic_sampling/transition_count": torch.tensor(
                float(self.num_transitions), device=self.device
            ),
            "semantic_sampling/critical_interval_count": self._critical_mask.sum().float(),
            "semantic_sampling/ordinary_interval_count": (~self._critical_mask).sum().float(),
            "semantic_sampling/critical_probability_mass": probabilities[self._critical_mask].sum(),
            "semantic_sampling/critical_sample_fraction": (
                sample_float[self._critical_mask].sum() / total_samples.clamp_min(1.0)
            ),
            "semantic_sampling/global_uniform_fraction": global_fraction,
            "semantic_sampling/sample_entropy": entropy,
            "semantic_sampling/top1_transition": top_transition.float(),
            "semantic_sampling/top1_probability": top_probability,
            "semantic_sampling/configuration_error_count": self.config_error_count.float(),
        }
        for transition_id in range(self.num_transitions):
            prefix = f"semantic_sampling/transition_{transition_id}"
            output[f"{prefix}_probability"] = probabilities[transition_id]
            output[f"{prefix}_failure_rate"] = failure_rate[transition_id]
            output[f"{prefix}_success_rate"] = success_rate[transition_id]
            output[f"{prefix}_sample_count"] = self.sample_count[transition_id].float()
            output[f"{prefix}_failure_score"] = self.failure_score[transition_id]
            output[f"{prefix}_is_critical"] = self._critical_mask[transition_id].float()
        return output


__all__ = [
    "SAMPLING_MODES",
    "SemanticTransition",
    "SemanticTransitionSampler",
    "load_semantic_transitions",
]
