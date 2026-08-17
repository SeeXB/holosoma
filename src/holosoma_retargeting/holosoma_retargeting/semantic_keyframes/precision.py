"""Method-independent precision evaluation for semantic retargeting.

The evaluator consumes only unweighted, uniform-Laplacian vertex residuals and
the original per-frame Delaunay adjacency saved by the retargeter. Optimizer
weights, objectives, budgets, and semantic alpha values are intentionally not
accepted as inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from holosoma_retargeting.semantic_keyframes.runtime import (
    SemanticEvent,
    resolve_body_vertex_mapping,
)


@dataclass(frozen=True)
class FramePartition:
    """Shared exact, neighborhood, and ordinary frame indices."""

    exact: np.ndarray
    keyframe_neighborhood: np.ndarray
    ordinary: np.ndarray


@dataclass(frozen=True)
class PrecisionPayload:
    """Raw method-independent evaluator inputs stored in a trajectory NPZ."""

    residuals: np.ndarray
    adjacency: np.ndarray
    joint_names: tuple[str, ...]
    num_body_vertices: int
    source_vertices: np.ndarray | None = None
    target_vertices: np.ndarray | None = None


@dataclass(frozen=True)
class PrecisionEvaluation:
    """Round-three precision metrics in the residuals' native unit."""

    partition: FramePartition
    global_per_frame: np.ndarray
    semantic_part_per_frame: np.ndarray
    semantic_edge_per_frame: np.ndarray | None
    ordinary: dict[str, float]
    keyframe_global: dict[str, float]
    semantic_part: dict[str, float]
    semantic_local: dict[str, float]
    semantic_edge: dict[str, float] | None
    criticality_weighted_part: dict[str, float]
    criticality_weighted_edge: dict[str, float] | None
    event_rows: tuple[dict[str, Any], ...]
    body_part_rows: tuple[dict[str, Any], ...]


def load_precision_payload(path: str | Path) -> PrecisionPayload:
    """Load and validate the raw evaluator payload from one trajectory."""
    with np.load(path, allow_pickle=False) as payload:
        required = {
            "unweighted_vertex_residuals",
            "interaction_mesh_adjacency",
            "interaction_mesh_joint_names",
            "interaction_mesh_num_body_vertices",
        }
        missing = sorted(required.difference(payload.files))
        if missing:
            raise ValueError(f"{path}: missing Round 3 precision payload keys {missing}")
        residuals = np.asarray(payload["unweighted_vertex_residuals"], dtype=np.float64)
        adjacency = np.asarray(payload["interaction_mesh_adjacency"], dtype=np.uint8)
        joint_names = tuple(str(name) for name in payload["interaction_mesh_joint_names"].tolist())
        num_body_vertices = int(payload["interaction_mesh_num_body_vertices"])
        source_vertices = (
            np.asarray(payload["source_interaction_vertices"], dtype=np.float64)
            if "source_interaction_vertices" in payload.files
            else None
        )
        target_vertices = (
            np.asarray(payload["target_interaction_vertices"], dtype=np.float64)
            if "target_interaction_vertices" in payload.files
            else None
        )
    if residuals.ndim != 2 or residuals.shape[0] < 1:
        raise ValueError(f"{path}: residuals must have shape [frames, vertices]")
    if adjacency.shape != (residuals.shape[0], residuals.shape[1], residuals.shape[1]):
        raise ValueError(f"{path}: adjacency shape {adjacency.shape} does not match residuals {residuals.shape}")
    if not np.all(np.isfinite(residuals)) or np.any(residuals < 0):
        raise ValueError(f"{path}: residuals must be finite and non-negative")
    if not 0 < num_body_vertices <= residuals.shape[1]:
        raise ValueError(f"{path}: invalid body vertex count {num_body_vertices}")
    if len(joint_names) != num_body_vertices:
        raise ValueError(f"{path}: joint-name count does not match body vertex count")
    if (source_vertices is None) != (target_vertices is None):
        raise ValueError(f"{path}: source and target interaction vertices must be stored together")
    if source_vertices is not None:
        expected_shape = (residuals.shape[0], residuals.shape[1], 3)
        if source_vertices.shape != expected_shape or target_vertices.shape != expected_shape:
            raise ValueError(
                f"{path}: interaction vertex shapes must both equal {expected_shape}, got "
                f"{source_vertices.shape} and {target_vertices.shape}"
            )
        if not np.all(np.isfinite(source_vertices)) or not np.all(np.isfinite(target_vertices)):
            raise ValueError(f"{path}: interaction vertices must be finite")
    return PrecisionPayload(
        residuals,
        adjacency,
        joint_names,
        num_body_vertices,
        source_vertices,
        target_vertices,
    )


def build_frame_partition(
    num_frames: int,
    triggers: Sequence[int],
    *,
    radius: int = 3,
) -> FramePartition:
    """Build the pre-registered O/K/K3 partition, excluding frame zero."""
    if num_frames < 1:
        raise ValueError("num_frames must be positive")
    exact = np.asarray(sorted(set(int(trigger) for trigger in triggers)), dtype=np.int64)
    if exact.size == 0 or np.any(exact < 0) or np.any(exact >= num_frames):
        raise ValueError("all semantic triggers must lie inside the trajectory")
    neighborhood = sorted(
        {
            frame
            for trigger in exact
            for frame in range(max(0, int(trigger) - radius), min(num_frames - 1, int(trigger) + radius) + 1)
        }
    )
    excluded = {0, *neighborhood}
    ordinary = np.asarray([frame for frame in range(num_frames) if frame not in excluded], dtype=np.int64)
    return FramePartition(
        exact=exact,
        keyframe_neighborhood=np.asarray(neighborhood, dtype=np.int64),
        ordinary=ordinary,
    )


def _summary(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("precision metric received empty or non-finite values")
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "worst": float(array.max()),
    }


def _window(trigger: int, radius: int, num_frames: int) -> np.ndarray:
    return np.arange(max(0, trigger - radius), min(num_frames - 1, trigger + radius) + 1, dtype=np.int64)


def _aggregate_event_windows(
    per_event: Sequence[dict[str, Any]],
    value_key: str,
) -> dict[str, float]:
    result: dict[str, float] = {}
    for region in ("exact", "pm1", "pm3"):
        values = [row.get(f"{value_key}_{region}") for row in per_event]
        available = [float(value) for value in values if value is not None]
        if not available:
            raise ValueError(f"no events can be evaluated for {value_key}_{region}")
        result[region] = float(np.mean(available))
    for region in ("pm1", "pm3"):
        values = [row.get(f"{value_key}_{region}_worst") for row in per_event]
        available = [float(value) for value in values if value is not None]
        if not available:
            raise ValueError(f"no events can be evaluated for {value_key}_{region}_worst")
        result[f"{region}_worst"] = float(max(available))
    return result


def _criticality_weighted_event_windows(
    per_event: Sequence[dict[str, Any]],
    events: Sequence[SemanticEvent],
    value_key: str,
) -> dict[str, float]:
    by_name = {event.name: event for event in events}
    result: dict[str, float] = {}
    for region in ("exact", "pm1", "pm3"):
        available = [row for row in per_event if row.get(f"{value_key}_{region}") is not None]
        if not available:
            raise ValueError(f"no events can be weighted for {value_key}_{region}")
        weights = np.asarray([by_name[row["event"]].criticality for row in available], dtype=np.float64)
        if np.any(weights < 0) or float(weights.sum()) <= 0:
            raise ValueError("criticality weights must be non-negative with a positive sum")
        result[region] = float(
            np.average(
                np.asarray([row[f"{value_key}_{region}"] for row in available], dtype=np.float64),
                weights=weights,
            )
        )
    return result


def _edge_error_for_frame(
    payload: PrecisionPayload,
    frame: int,
    body_indices: np.ndarray,
) -> tuple[float | None, int]:
    if payload.source_vertices is None or payload.target_vertices is None:
        raise ValueError("semantic edge evaluation requires source and target interaction vertices")
    edge_errors: list[float] = []
    for body_index in body_indices:
        neighbors = np.flatnonzero(payload.adjacency[frame, body_index])
        for object_index in neighbors:
            if object_index < payload.num_body_vertices:
                continue
            source_relative = (
                payload.source_vertices[frame, body_index]
                - payload.source_vertices[frame, object_index]
            )
            target_relative = (
                payload.target_vertices[frame, body_index]
                - payload.target_vertices[frame, object_index]
            )
            edge_errors.append(float(np.linalg.norm(source_relative - target_relative)))
    if not edge_errors:
        return None, 0
    return float(np.mean(edge_errors)), len(edge_errors)


def evaluate_precision(
    payload: PrecisionPayload,
    events: Sequence[SemanticEvent],
    *,
    critical_event_names: Sequence[str] | None = ("contact", "lift", "place", "release"),
) -> PrecisionEvaluation:
    """Evaluate ordinary, keyframe-global, semantic-part, and local precision."""
    residuals = payload.residuals
    num_frames, num_vertices = residuals.shape
    critical_names = set(critical_event_names) if critical_event_names is not None else {event.name for event in events}
    critical = [event for event in events if event.name in critical_names]
    if critical_event_names is not None and {event.name for event in critical} != critical_names:
        raise ValueError(
            f"critical events mismatch: expected {sorted(critical_names)}, got {sorted(event.name for event in critical)}"
        )
    critical.sort(key=lambda event: event.trigger_frame)
    partition = build_frame_partition(num_frames, [event.trigger_frame for event in critical])
    global_per_frame = residuals.mean(axis=1)
    ordinary = _summary(global_per_frame[partition.ordinary])

    mapping = resolve_body_vertex_mapping(
        payload.joint_names,
        [part for event in critical for part in event.body_parts],
    )
    if mapping.missing:
        raise ValueError(f"semantic body parts cannot be evaluated: {mapping.missing}")
    union_indices = np.asarray(sorted({mapping.indices[part] for event in critical for part in event.body_parts}))
    semantic_part_per_frame = residuals[:, union_indices].mean(axis=1)
    semantic_edge_per_frame = None
    if payload.source_vertices is not None:
        semantic_edge_per_frame = np.asarray(
            [
                value if value is not None else np.nan
                for frame in range(num_frames)
                for value in [_edge_error_for_frame(payload, frame, union_indices)[0]]
            ],
            dtype=np.float64,
        )

    event_rows: list[dict[str, Any]] = []
    body_part_rows: list[dict[str, Any]] = []
    for event in critical:
        body_parts = list(dict.fromkeys(event.body_parts))
        body_indices = np.asarray([mapping.indices[part] for part in body_parts], dtype=np.int64)
        event_row: dict[str, Any] = {
            "event": event.name,
            "trigger_frame": event.trigger_frame,
        }
        for radius, region in ((0, "exact"), (1, "pm1"), (3, "pm3")):
            frames = _window(event.trigger_frame, radius, num_frames)
            event_row[f"global_{region}"] = float(global_per_frame[frames].mean())
            event_row[f"global_{region}_worst"] = float(global_per_frame[frames].max())
            body_values = residuals[np.ix_(frames, body_indices)]
            event_row[f"part_{region}"] = float(body_values.mean())
            event_row[f"part_{region}_worst"] = float(body_values.max())

            if payload.source_vertices is not None:
                edge_results = [
                    _edge_error_for_frame(payload, int(frame), body_indices) for frame in frames
                ]
                edge_frame_values = [float(value) for value, _ in edge_results if value is not None]
                event_row[f"edge_{region}_count"] = int(sum(count for _, count in edge_results))
                event_row[f"edge_{region}"] = (
                    float(np.mean(edge_frame_values)) if edge_frame_values else None
                )
                event_row[f"edge_{region}_worst"] = (
                    float(max(edge_frame_values)) if edge_frame_values else None
                )

            local_frame_means: list[float] = []
            local_values: list[float] = []
            for frame in frames:
                local_indices = set(int(index) for index in body_indices)
                for body_index in body_indices:
                    neighbors = np.flatnonzero(payload.adjacency[frame, body_index])
                    local_indices.update(
                        int(index) for index in neighbors if payload.num_body_vertices <= index < num_vertices
                    )
                values = residuals[frame, np.asarray(sorted(local_indices), dtype=np.int64)]
                local_frame_means.append(float(values.mean()))
                local_values.extend(float(value) for value in values)
            event_row[f"local_{region}"] = float(np.mean(local_frame_means))
            event_row[f"local_{region}_worst"] = float(max(local_values))

        for body_part, body_index in zip(body_parts, body_indices):
            exact = float(residuals[event.trigger_frame, body_index])
            pm1_values = residuals[_window(event.trigger_frame, 1, num_frames), body_index]
            pm3_values = residuals[_window(event.trigger_frame, 3, num_frames), body_index]
            body_part_rows.append(
                {
                    "event": event.name,
                    "trigger_frame": event.trigger_frame,
                    "body_part": body_part,
                    "vertex_index": int(body_index),
                    "exact": exact,
                    "pm1": float(pm1_values.mean()),
                    "pm3": float(pm3_values.mean()),
                    "pm1_worst": float(pm1_values.max()),
                    "pm3_worst": float(pm3_values.max()),
                }
            )
        event_rows.append(event_row)

    keyframe_global = _aggregate_event_windows(event_rows, "global")
    semantic_part = _aggregate_event_windows(event_rows, "part")
    semantic_local = _aggregate_event_windows(event_rows, "local")
    edge_available = payload.source_vertices is not None and any(
        row.get("edge_exact") is not None for row in event_rows
    )
    semantic_edge = _aggregate_event_windows(event_rows, "edge") if edge_available else None
    criticality_weighted_part = _criticality_weighted_event_windows(event_rows, critical, "part")
    criticality_weighted_edge = (
        _criticality_weighted_event_windows(event_rows, critical, "edge")
        if edge_available
        else None
    )
    return PrecisionEvaluation(
        partition=partition,
        global_per_frame=global_per_frame,
        semantic_part_per_frame=semantic_part_per_frame,
        semantic_edge_per_frame=semantic_edge_per_frame,
        ordinary=ordinary,
        keyframe_global=keyframe_global,
        semantic_part=semantic_part,
        semantic_local=semantic_local,
        semantic_edge=semantic_edge,
        criticality_weighted_part=criticality_weighted_part,
        criticality_weighted_edge=criticality_weighted_edge,
        event_rows=tuple(event_rows),
        body_part_rows=tuple(body_part_rows),
    )


__all__ = [
    "FramePartition",
    "PrecisionEvaluation",
    "PrecisionPayload",
    "build_frame_partition",
    "evaluate_precision",
    "load_precision_payload",
]
