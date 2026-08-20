"""Pure runtime primitives for semantic-keyframe-aware retargeting.

This module intentionally has no MuJoCo/CVXPY dependency. JSON parsing,
scheduling, body mapping, and residual weights can therefore be tested without
loading the optimizer stack.
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from holosoma_retargeting.config_types.semantic import SemanticRetargetingConfig
from holosoma_retargeting.semantic_keyframes.pipeline import CRITICALITY_MAPPING


DEFAULT_EVENT_CRITICALITY = {
    "contact": 1.0,
    "lift": 1.0,
    "place": 1.0,
    "release": 1.0,
    "approach": 0.5,
    "arrive": 0.5,
    "carry_mid": 0.3,
    "start": 0.0,
}

BODY_PART_ALIASES: dict[str, tuple[str, ...]] = {
    "left_hand": (
        "L_Wrist",
        "LeftHand",
        "LeftHandMiddle3",
        "L_Hand",
        "left_wrist",
        "left_hand",
    ),
    "right_hand": (
        "R_Wrist",
        "RightHand",
        "RightHandMiddle3",
        "R_Hand",
        "right_wrist",
        "right_hand",
    ),
    "pelvis": ("Pelvis", "Hips", "Spine1", "pelvis", "root"),
    "left_foot": ("L_Toe", "L_Foot", "LeftToeBase", "LeftFoot", "left_foot"),
    "right_foot": ("R_Toe", "R_Foot", "RightToeBase", "RightFoot", "right_foot"),
    "torso": ("Spine3", "Spine2", "Spine1", "Chest", "Torso", "torso"),
}


@dataclass(frozen=True)
class SemanticEvent:
    """One resolved semantic event window."""

    name: str
    trigger_frame: int
    start_frame: int
    end_frame: int
    body_parts: list[str]
    confidence: float
    criticality: float
    criticality_level: int | None = None
    criticality_rationale: str = ""
    failure_if_inaccurate: str = ""
    trigger: dict[str, Any] | None = None
    end: dict[str, Any] | None = None
    rationale: str = ""

    @property
    def score(self) -> float:
        return self.criticality * self.confidence


@dataclass(frozen=True)
class FrameSemanticInfo:
    """Semantic context for a single optimizer frame."""

    frame_idx: int
    active_events: list[str]
    nearest_event: str | None
    nearest_trigger_distance: int | None
    semantic_score: float
    body_parts: list[str]


@dataclass(frozen=True)
class BodyVertexMapping:
    """Resolved semantic aliases to interaction-mesh human vertex indices."""

    indices: dict[str, int]
    joint_names: dict[str, str]
    missing: tuple[str, ...]


@dataclass(frozen=True)
class BudgetPlan:
    """Per-frame maximum iterations and events used to describe that plan."""

    budgets: np.ndarray
    events: tuple[SemanticEvent, ...]
    semantic_trigger_frames: tuple[int, ...] = ()
    extra_budget_frames: tuple[int, ...] = ()
    allocation_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class SemanticWeightResult:
    """Normalized residual weights plus preference-energy diagnostics."""

    weights: np.ndarray
    boosted_vertex_count: int
    l1_norm_alpha_minus_one: float
    l2_norm_alpha_minus_one: float
    semantic_edge_count: int = 0
    edge_endpoint_count: int = 0
    deduplicated_overlap_count: int = 0


@dataclass(frozen=True)
class SpatiotemporalSemanticContext:
    """Temporal context used by Legacy residual weighting and profiling."""

    frame_idx: int
    dominant_event: str | None
    active_events: tuple[str, ...]
    body_parts: tuple[str, ...]
    tau: float
    semantic_importance: float


@dataclass(frozen=True)
class SemanticEdgeSet:
    """Unique directed body-to-object interaction-mesh edges."""

    edges: tuple[tuple[int, int], ...]
    missing_body_vertices: tuple[int, ...]


def _as_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{field_name} must be an integer, got {value!r}")
    return int(value)


def load_semantic_events(
    path: str | Path,
    *,
    event_criticality: Mapping[str, float] | None = None,
    unknown_event_criticality: float = 0.5,
) -> list[SemanticEvent]:
    """Load the current semantic JSON schema into validated dataclasses."""
    criticality = dict(DEFAULT_EVENT_CRITICALITY if event_criticality is None else event_criticality)
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    raw_events = payload.get("events")
    if not isinstance(raw_events, list):
        raise ValueError(f"{source}: top-level 'events' must be a list")

    events: list[SemanticEvent] = []
    for event_idx, raw_event in enumerate(raw_events):
        if not isinstance(raw_event, dict):
            raise ValueError(f"{source}: events[{event_idx}] must be an object")
        name = raw_event.get("event")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{source}: events[{event_idx}].event must be a non-empty string")
        confidence = float(raw_event.get("confidence", 1.0))
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError(f"{source}: confidence for {name!r} must be finite and in [0, 1]")
        raw_body_parts = raw_event.get("body_parts", [])
        if not isinstance(raw_body_parts, list) or not all(isinstance(part, str) for part in raw_body_parts):
            raise ValueError(f"{source}: body_parts for {name!r} must be a list of strings")

        windows = raw_event.get("windows")
        if windows is None:
            # Also accept the flattened schema requested by the optimizer prompt.
            windows = [raw_event]
        if not isinstance(windows, list) or not windows:
            raise ValueError(f"{source}: windows for {name!r} must be a non-empty list")
        raw_level = raw_event.get("criticality_level")
        raw_criticality = raw_event.get("criticality")
        if raw_level is None:
            if raw_criticality is not None:
                raise ValueError(f"{source}: {name!r} has criticality without criticality_level")
            event_level = None
            event_criticality_value = float(criticality.get(name, unknown_event_criticality))
        else:
            if isinstance(raw_level, bool) or not isinstance(raw_level, int) or raw_level not in CRITICALITY_MAPPING:
                raise ValueError(
                    f"{source}: {name!r}.criticality_level must be one of {sorted(CRITICALITY_MAPPING)}"
                )
            expected_criticality = CRITICALITY_MAPPING[raw_level]
            if isinstance(raw_criticality, bool) or not isinstance(raw_criticality, (int, float)):
                raise ValueError(f"{source}: {name!r}.criticality must be numeric")
            if float(raw_criticality) != expected_criticality:
                raise ValueError(
                    f"{source}: {name!r}.criticality must equal {expected_criticality:.2f} for level {raw_level}"
                )
            event_level = raw_level
            event_criticality_value = expected_criticality
        for window_idx, window in enumerate(windows):
            if not isinstance(window, dict):
                raise ValueError(f"{source}: window {window_idx} for {name!r} must be an object")
            trigger_frame = _as_int(window.get("trigger_frame"), f"{name}.trigger_frame")
            start_frame = _as_int(window.get("start_frame"), f"{name}.start_frame")
            end_frame = _as_int(window.get("end_frame"), f"{name}.end_frame")
            if min(start_frame, trigger_frame, end_frame) < 0:
                raise ValueError(f"{source}: frames for {name!r} must be non-negative")
            if not start_frame <= trigger_frame <= end_frame:
                raise ValueError(
                    f"{source}: expected start <= trigger <= end for {name!r}, "
                    f"got {start_frame}, {trigger_frame}, {end_frame}"
                )
            events.append(
                SemanticEvent(
                    name=name,
                    trigger_frame=trigger_frame,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    body_parts=list(dict.fromkeys(raw_body_parts)),
                    confidence=confidence,
                    criticality=event_criticality_value,
                    criticality_level=event_level,
                    criticality_rationale=str(raw_event.get("criticality_rationale", "")),
                    failure_if_inaccurate=str(raw_event.get("failure_if_inaccurate", "")),
                    trigger=raw_event.get("trigger"),
                    end=raw_event.get("end"),
                    rationale=str(raw_event.get("rationale", "")),
                )
            )
    return events


def load_semantic_plan_projection(path: str | Path) -> list[SemanticEvent]:
    """Load the deterministic active-plan projection from historical JSON.

    Only event identity, windows, trigger/end rules, body parts, and rationale
    cross the boundary into the active optimizer.  In particular, criticality
    fields are neither parsed nor validated, so changing them (even to an
    invalid value) cannot change or block an active run.  Projected events have
    unit confidence/strength solely for compatibility with the historical
    ``SemanticEvent`` container.
    """
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    raw_events = payload.get("events")
    if not isinstance(raw_events, list):
        raise ValueError(f"{source}: top-level 'events' must be a list")

    events: list[SemanticEvent] = []
    for event_idx, raw_event in enumerate(raw_events):
        if not isinstance(raw_event, dict):
            raise ValueError(f"{source}: events[{event_idx}] must be an object")
        name = raw_event.get("event")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{source}: events[{event_idx}].event must be a non-empty string")
        raw_body_parts = raw_event.get("body_parts", [])
        if not isinstance(raw_body_parts, list) or not all(isinstance(part, str) for part in raw_body_parts):
            raise ValueError(f"{source}: body_parts for {name!r} must be a list of strings")
        windows = raw_event.get("windows")
        if windows is None:
            windows = [raw_event]
        if not isinstance(windows, list) or not windows:
            raise ValueError(f"{source}: windows for {name!r} must be a non-empty list")
        for window_idx, window in enumerate(windows):
            if not isinstance(window, dict):
                raise ValueError(f"{source}: window {window_idx} for {name!r} must be an object")
            trigger_frame = _as_int(window.get("trigger_frame"), f"{name}.trigger_frame")
            start_frame = _as_int(window.get("start_frame"), f"{name}.start_frame")
            end_frame = _as_int(window.get("end_frame"), f"{name}.end_frame")
            if min(start_frame, trigger_frame, end_frame) < 0:
                raise ValueError(f"{source}: frames for {name!r} must be non-negative")
            if not start_frame <= trigger_frame <= end_frame:
                raise ValueError(
                    f"{source}: expected start <= trigger <= end for {name!r}, "
                    f"got {start_frame}, {trigger_frame}, {end_frame}"
                )
            events.append(
                SemanticEvent(
                    name=name,
                    trigger_frame=trigger_frame,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    body_parts=list(dict.fromkeys(raw_body_parts)),
                    confidence=1.0,
                    criticality=1.0,
                    trigger=raw_event.get("trigger"),
                    end=raw_event.get("end"),
                    rationale=str(raw_event.get("rationale", "")),
                )
            )
    return events


class SemanticTimeline:
    """Compute deterministic per-frame semantic context."""

    def __init__(
        self,
        events: Sequence[SemanticEvent],
        *,
        sigma: float = 2.0,
        phase_weight: float = 0.25,
        nearby_radius: int = 3,
    ) -> None:
        self.events = tuple(events)
        self.sigma = float(sigma)
        self.phase_weight = float(phase_weight)
        self.nearby_radius = int(nearby_radius)
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")

    def temporal_weight(self, frame_idx: int, event: SemanticEvent) -> float:
        distance = frame_idx - event.trigger_frame
        trigger_weight = math.exp(-(distance * distance) / (2.0 * self.sigma * self.sigma))
        phase = self.phase_weight if event.start_frame <= frame_idx <= event.end_frame else 0.0
        return max(trigger_weight, phase)

    def frame_info(self, frame_idx: int) -> FrameSemanticInfo:
        active = [event for event in self.events if event.start_frame <= frame_idx <= event.end_frame]
        nearest = min(self.events, key=lambda event: abs(frame_idx - event.trigger_frame), default=None)
        nearby = [event for event in self.events if abs(frame_idx - event.trigger_frame) <= self.nearby_radius]
        relevant: list[SemanticEvent] = []
        seen_relevant: set[tuple[str, int, int, int]] = set()
        for event in [*active, *nearby]:
            key = (event.name, event.start_frame, event.trigger_frame, event.end_frame)
            if key not in seen_relevant:
                relevant.append(event)
                seen_relevant.add(key)
        body_parts = list(dict.fromkeys(part for event in relevant for part in event.body_parts))
        score = max(
            (event.confidence * self.temporal_weight(frame_idx, event) for event in self.events),
            default=0.0,
        )
        return FrameSemanticInfo(
            frame_idx=frame_idx,
            active_events=list(dict.fromkeys(event.name for event in active)),
            nearest_event=nearest.name if nearest is not None else None,
            nearest_trigger_distance=abs(frame_idx - nearest.trigger_frame) if nearest is not None else None,
            semantic_score=float(score),
            body_parts=body_parts,
        )


def make_budget_plan(
    num_frames: int,
    events: Sequence[SemanticEvent],
    config: SemanticRetargetingConfig,
) -> BudgetPlan:
    """Create the configured official, uniform, semantic, or random plan."""
    if num_frames < 0:
        raise ValueError("num_frames must be non-negative")
    if num_frames == 0:
        return BudgetPlan(np.zeros(0, dtype=np.int64), tuple(events))
    if config.is_uniform2_mainline:
        return make_exact_trigger_budget_plan(
            num_frames,
            events,
            config,
            randomize=config.uses_random_exact_budget,
            enable_extra=config.uses_exact_semantic_budget or config.uses_random_exact_budget,
        )
    if config.mode == "original":
        budgets = np.full(num_frames, config.original_budget, dtype=np.int64)
        budgets[0] = config.frame0_budget
        return BudgetPlan(budgets, tuple(events))
    if config.mode == "uniform":
        budgets = np.full(num_frames, config.uniform_budget, dtype=np.int64)
        budgets[0] = config.frame0_budget
        return BudgetPlan(budgets, ())
    raise ValueError(f"Unsupported active semantic mode: {config.mode!r}")


def make_exact_trigger_budget_plan(
    num_frames: int,
    events: Sequence[SemanticEvent],
    config: SemanticRetargetingConfig,
    *,
    randomize: bool,
    enable_extra: bool = True,
) -> BudgetPlan:
    """Create the registered Uniform-2 exact-trigger or matched random plan."""
    if num_frames < 1:
        return BudgetPlan(np.zeros(0, dtype=np.int64), tuple(events))
    semantic_frames = tuple(
        sorted({event.trigger_frame for event in events if 0 < event.trigger_frame < num_frames})
    )
    budgets = np.full(num_frames, config.round2_base_budget, dtype=np.int64)
    budgets[0] = config.frame0_budget
    reasons = ["ordinary_base"] * num_frames
    reasons[0] = "frame0_initialization"
    extra_frames: tuple[int, ...] = ()

    if enable_extra and randomize:
        excluded: set[int] = {0}
        for event in events:
            for offset in range(-config.random_exclusion_radius, config.random_exclusion_radius + 1):
                candidate = event.trigger_frame + offset
                if 0 <= candidate < num_frames:
                    excluded.add(candidate)
        candidates = np.asarray(
            [frame for frame in range(1, num_frames) if frame not in excluded],
            dtype=np.int64,
        )
        if len(candidates) < len(semantic_frames):
            raise ValueError(
                f"Only {len(candidates)} random ordinary frames remain for "
                f"{len(semantic_frames)} semantic trigger slots"
            )
        rng = np.random.default_rng(config.random_seed)
        extra_frames = tuple(sorted(int(frame) for frame in rng.choice(
            candidates,
            size=len(semantic_frames),
            replace=False,
        )))
        for frame in extra_frames:
            budgets[frame] = config.exact_trigger_budget
            reasons[frame] = "random_extra"
    elif enable_extra:
        extra_frames = semantic_frames
        for frame in extra_frames:
            budgets[frame] = config.exact_trigger_budget
            reasons[frame] = "semantic_extra"

    return BudgetPlan(
        budgets=budgets,
        events=tuple(events),
        semantic_trigger_frames=semantic_frames,
        extra_budget_frames=extra_frames,
        allocation_reasons=tuple(reasons),
    )


def resolve_body_vertex_mapping(
    joint_mapping_keys: Sequence[str],
    body_parts: Iterable[str],
) -> BodyVertexMapping:
    """Resolve semantic aliases against the real JOINTS_MAPPING insertion order."""
    exact = {name: index for index, name in enumerate(joint_mapping_keys)}
    casefolded = {name.casefold(): (name, index) for index, name in enumerate(joint_mapping_keys)}
    indices: dict[str, int] = {}
    names: dict[str, str] = {}
    missing: list[str] = []
    for body_part in dict.fromkeys(body_parts):
        candidates = (body_part, *BODY_PART_ALIASES.get(body_part, ()))
        resolved: tuple[str, int] | None = None
        for candidate in candidates:
            if candidate in exact:
                resolved = (candidate, exact[candidate])
                break
            resolved = casefolded.get(candidate.casefold())
            if resolved is not None:
                break
        if resolved is None:
            missing.append(body_part)
            warnings.warn(
                f"Semantic body part {body_part!r} cannot be mapped to JOINTS_MAPPING keys "
                f"{list(joint_mapping_keys)!r}; it will not be weighted.",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        names[body_part], indices[body_part] = resolved
    return BodyVertexMapping(indices=indices, joint_names=names, missing=tuple(missing))


def spatiotemporal_semantic_context(
    frame_idx: int,
    events: Sequence[SemanticEvent],
    *,
    sigma: float = 2.0,
    phase_weight: float = 0.25,
    kernel_cutoff: float = 1e-3,
    transition_truncated: bool = False,
) -> SpatiotemporalSemanticContext:
    """Compute the dominant Legacy weight strength and active body-part union."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if not 0 <= kernel_cutoff < 1:
        raise ValueError("kernel_cutoff must be in [0, 1)")
    candidates: list[tuple[SemanticEvent, float, float]] = []
    for event in events:
        distance = frame_idx - event.trigger_frame
        tau = math.exp(-(distance * distance) / (2.0 * sigma * sigma))
        phase = phase_weight if event.start_frame <= frame_idx <= event.end_frame else 0.0
        importance = event.confidence * max(tau, phase)
        if transition_truncated and not _event_inside_transition_support(
            frame_idx, event, events
        ):
            importance = 0.0
        candidates.append((event, tau, importance))
    if not candidates:
        return SpatiotemporalSemanticContext(frame_idx, None, (), (), 0.0, 0.0)
    dominant_event, dominant_tau, importance = max(
        candidates,
        key=lambda item: (item[2], -abs(frame_idx - item[0].trigger_frame), -item[0].trigger_frame),
    )
    active = [item for item in candidates if item[2] >= kernel_cutoff]
    body_parts = tuple(dict.fromkeys(part for event, _, _ in active for part in event.body_parts))
    return SpatiotemporalSemanticContext(
        frame_idx=frame_idx,
        dominant_event=dominant_event.name,
        active_events=tuple(event.name for event, _, _ in active),
        body_parts=body_parts,
        tau=float(dominant_tau),
        semantic_importance=float(importance),
    )


def _event_inside_transition_support(
    frame_idx: int,
    event: SemanticEvent,
    events: Sequence[SemanticEvent],
) -> bool:
    """Return whether an event owns this frame before the next transition.

    The support is causal and window-bounded: ``trigger <= frame <= end``.
    A later event trigger is an exclusive upper bound, so the new event owns
    its transition frame.  Looking up the next greater trigger rather than an
    event name keeps the policy generic and deterministic.
    """
    if frame_idx < event.trigger_frame or frame_idx > event.end_frame:
        return False
    next_triggers = [
        other.trigger_frame
        for other in events
        if other.trigger_frame > event.trigger_frame
    ]
    return not next_triggers or frame_idx < min(next_triggers)




def extract_semantic_cross_entity_edges(
    body_vertex_indices: Iterable[int],
    adjacency: Sequence[Sequence[int]],
    num_body_vertices: int,
) -> SemanticEdgeSet:
    """Extract unique body->object edges and diagnose bodies without one."""
    if not 0 < num_body_vertices <= len(adjacency):
        raise ValueError("num_body_vertices must lie inside adjacency")
    edges: set[tuple[int, int]] = set()
    missing: list[int] = []
    for raw_index in dict.fromkeys(int(index) for index in body_vertex_indices):
        if not 0 <= raw_index < num_body_vertices:
            raise ValueError(f"semantic body vertex {raw_index} is outside the body range")
        local_edges = {
            (raw_index, int(neighbor))
            for neighbor in adjacency[raw_index]
            if num_body_vertices <= int(neighbor) < len(adjacency)
        }
        if not local_edges:
            missing.append(raw_index)
        edges.update(local_edges)
    return SemanticEdgeSet(tuple(sorted(edges)), tuple(missing))




def _normalize_semantic_weights(alpha: np.ndarray, boosted: set[int]) -> SemanticWeightResult:
    mean = float(alpha.mean())
    if not math.isfinite(mean) or mean <= 0:
        raise ValueError("semantic vertex weights have a non-positive or non-finite mean")
    normalized = alpha / mean
    delta = normalized - 1.0
    return SemanticWeightResult(
        weights=normalized,
        boosted_vertex_count=len(boosted),
        l1_norm_alpha_minus_one=float(np.linalg.norm(delta, ord=1)),
        l2_norm_alpha_minus_one=float(np.linalg.norm(delta, ord=2)),
    )


def build_semantic_vertex_weight_result(
    *,
    frame_idx: int,
    events: Sequence[SemanticEvent],
    body_mapping: BodyVertexMapping,
    adjacency: Sequence[Sequence[int]],
    num_human_vertices: int,
    num_vertices: int,
    config: SemanticRetargetingConfig,
) -> SemanticWeightResult:
    """Build de-duplicated Part/Edge weights on existing Laplacian residuals.

    Legacy Part remains byte-for-byte equivalent to the historical behavior:
    body residuals use their existing multipliers and directly connected object
    residuals use the fixed multiplier 2.  Explicit Edge weighting emphasizes
    both endpoints of each body-object Delaunay edge with that same multiplier.
    Part+Edge combines contributions with ``max`` so the Legacy object-neighbor
    spillover is never counted twice.
    """
    if num_vertices < 1:
        raise ValueError("num_vertices must be positive")
    if len(adjacency) != num_vertices:
        raise ValueError("adjacency length must equal num_vertices")
    alpha = np.ones(num_vertices, dtype=np.float64)
    boosted: set[int] = set()
    semantic_edges: set[tuple[int, int]] = set()
    edge_endpoints: set[int] = set()
    deduplicated_overlaps: set[tuple[int, int]] = set()
    components = config.semantic_weight_components
    sigma = config.semantic_temporal_sigma
    body_only_events = set(config.body_only_weight_events)
    for event in events:
        if config.uses_transition_truncated_weight_support and not _event_inside_transition_support(
            frame_idx, event, events
        ):
            continue
        distance = frame_idx - event.trigger_frame
        trigger_weight = math.exp(-(distance * distance) / (2.0 * sigma * sigma))
        phase_weight = config.phase_semantic_weight if event.start_frame <= frame_idx <= event.end_frame else 0.0
        temporal = max(trigger_weight, phase_weight) * event.confidence
        if temporal < config.semantic_kernel_cutoff:
            continue
        for body_part in event.body_parts:
            vertex_idx = body_mapping.indices.get(body_part)
            multiplier = config.body_weight_multiplier.get(body_part)
            if vertex_idx is None or multiplier is None:
                continue
            object_neighbors = (
                []
                if event.name in body_only_events
                else [
                    int(neighbor_idx)
                    for neighbor_idx in adjacency[vertex_idx]
                    if num_human_vertices <= int(neighbor_idx) < num_vertices
                ]
            )
            if components in {"part", "part_edge"}:
                alpha[vertex_idx] = max(
                    alpha[vertex_idx],
                    1.0 + temporal * (multiplier - 1.0),
                )
                boosted.add(vertex_idx)
                neighbor_weight = 1.0 + temporal * (config.object_neighbor_multiplier - 1.0)
                for neighbor_idx in object_neighbors:
                    alpha[neighbor_idx] = max(alpha[neighbor_idx], neighbor_weight)
                    boosted.add(neighbor_idx)

            if components in {"edge", "part_edge"}:
                edge_weight = 1.0 + temporal * (config.object_neighbor_multiplier - 1.0)
                for neighbor_idx in object_neighbors:
                    edge = (vertex_idx, neighbor_idx)
                    semantic_edges.add(edge)
                    edge_endpoints.update(edge)
                    if components == "part_edge" and (
                        alpha[vertex_idx] >= edge_weight or alpha[neighbor_idx] >= edge_weight
                    ):
                        deduplicated_overlaps.add(edge)
                    alpha[vertex_idx] = max(alpha[vertex_idx], edge_weight)
                    alpha[neighbor_idx] = max(alpha[neighbor_idx], edge_weight)
                    boosted.update(edge)
    normalized = _normalize_semantic_weights(alpha, boosted)
    return replace(
        normalized,
        semantic_edge_count=len(semantic_edges),
        edge_endpoint_count=len(edge_endpoints),
        deduplicated_overlap_count=len(deduplicated_overlaps),
    )


def build_semantic_vertex_weights(
    *,
    frame_idx: int,
    events: Sequence[SemanticEvent],
    body_mapping: BodyVertexMapping,
    adjacency: Sequence[Sequence[int]],
    num_human_vertices: int,
    num_vertices: int,
    config: SemanticRetargetingConfig,
) -> np.ndarray:
    """Backward-compatible weights-only wrapper."""
    return build_semantic_vertex_weight_result(
        frame_idx=frame_idx,
        events=events,
        body_mapping=body_mapping,
        adjacency=adjacency,
        num_human_vertices=num_human_vertices,
        num_vertices=num_vertices,
        config=config,
    ).weights


def scale_semantic_weight_result(
    result: SemanticWeightResult,
    scale: float,
) -> SemanticWeightResult:
    """Scale mean-one preference deviations while preserving their pattern.

    This is used only by causal controls that must match a pre-registered
    whole-trajectory ``sum(abs(alpha - 1))`` energy. The underlying semantic
    multipliers and temporal kernels are built first; the single global scale
    then removes realized-energy differences caused by topology and event
    overlap.
    """
    if not math.isfinite(scale) or scale < 0:
        raise ValueError(f"semantic weight energy scale must be finite and non-negative, got {scale}")
    weights = 1.0 + scale * (np.asarray(result.weights, dtype=np.float64) - 1.0)
    if np.any(weights <= 0):
        raise ValueError("semantic weight energy matching produced non-positive vertex weights")
    weights /= float(weights.mean())
    delta = weights - 1.0
    return SemanticWeightResult(
        weights=weights,
        boosted_vertex_count=result.boosted_vertex_count,
        l1_norm_alpha_minus_one=float(np.linalg.norm(delta, ord=1)),
        l2_norm_alpha_minus_one=float(np.linalg.norm(delta, ord=2)),
        semantic_edge_count=result.semantic_edge_count,
        edge_endpoint_count=result.edge_endpoint_count,
        deduplicated_overlap_count=result.deduplicated_overlap_count,
    )


def build_always_body_vertex_weight_result(
    *,
    body_parts: Sequence[str],
    body_mapping: BodyVertexMapping,
    adjacency: Sequence[Sequence[int]],
    num_human_vertices: int,
    num_vertices: int,
    config: SemanticRetargetingConfig,
    amplitude: float = 1.0,
) -> SemanticWeightResult:
    """Apply a constant whole-trajectory body preference with mean-one scale."""
    if not 0.0 <= amplitude <= 1.0:
        raise ValueError("amplitude must be in [0, 1]")
    alpha = np.ones(num_vertices, dtype=np.float64)
    boosted: set[int] = set()
    for body_part in dict.fromkeys(body_parts):
        vertex_idx = body_mapping.indices.get(body_part)
        multiplier = config.body_weight_multiplier.get(body_part)
        if vertex_idx is None or multiplier is None:
            continue
        alpha[vertex_idx] = 1.0 + amplitude * (multiplier - 1.0)
        boosted.add(vertex_idx)
        object_weight = 1.0 + amplitude * (config.object_neighbor_multiplier - 1.0)
        for neighbor_idx in adjacency[vertex_idx]:
            if num_human_vertices <= neighbor_idx < num_vertices:
                alpha[neighbor_idx] = max(alpha[neighbor_idx], object_weight)
                boosted.add(neighbor_idx)
    return _normalize_semantic_weights(alpha, boosted)


def randomize_event_times_preserving_support(
    num_frames: int,
    events: Sequence[SemanticEvent],
    seed: int,
    *,
    gaussian_margin: int = 8,
) -> list[SemanticEvent]:
    """Translate windows without changing their length or truncated Gaussian mass."""
    if num_frames < 1:
        return []
    rng = np.random.default_rng(seed)
    randomized: list[SemanticEvent] = []
    used: set[int] = set()
    for event in events:
        before = event.trigger_frame - event.start_frame
        after = event.end_frame - event.trigger_frame
        lower = max(before, gaussian_margin, 1)
        upper = min(num_frames - 1 - after, num_frames - 1 - gaussian_margin)
        candidates = [frame for frame in range(lower, upper + 1) if frame not in used]
        if not candidates:
            raise ValueError(f"Cannot place randomized event {event.name!r} in {num_frames} frames")
        trigger = int(rng.choice(np.asarray(candidates, dtype=np.int64)))
        used.add(trigger)
        randomized.append(
            replace(
                event,
                trigger_frame=trigger,
                start_frame=trigger - before,
                end_frame=trigger + after,
            )
        )
    return randomized




def shuffle_event_body_parts(events: Sequence[SemanticEvent]) -> list[SemanticEvent]:
    """Create the optional deliberately incorrect body-part control."""
    shuffled: list[SemanticEvent] = []
    for event in events:
        body_parts: list[str] = []
        for body_part in event.body_parts:
            if body_part in {"left_hand", "right_hand"}:
                body_parts.append("pelvis")
            elif body_part == "pelvis":
                body_parts.extend(["left_hand", "right_hand"])
            else:
                body_parts.append(body_part)
        shuffled.append(replace(event, body_parts=list(dict.fromkeys(body_parts))))
    return shuffled
