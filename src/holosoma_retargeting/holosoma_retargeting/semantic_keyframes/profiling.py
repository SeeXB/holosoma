"""Artifact writers and aggregate metrics for semantic retargeting runs."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return "|".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(_json_value(value), sort_keys=True)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _finite(values: Sequence[Any]) -> np.ndarray:
    array = np.asarray([np.nan if value is None else value for value in values], dtype=np.float64)
    return array[np.isfinite(array)]


def summarize_profiles(
    profiles: Sequence[dict[str, Any]],
    *,
    mode: str,
    total_wall_time: float,
    rescue_log: Sequence[dict[str, Any]],
    penetration_tolerance: float = 1e-3,
) -> dict[str, Any]:
    """Aggregate the required runtime, optimization, and feasibility metrics."""
    frame_times = _finite([row.get("frame_wall_time") for row in profiles])
    lap_errors = _finite([row.get("final_laplacian_error") for row in profiles])
    hand_errors = _finite(
        [
            max(
                row.get("left_hand_object_error") or 0.0,
                row.get("right_hand_object_error") or 0.0,
            )
            for row in profiles
            if row.get("is_critical_interaction_neighborhood", False)
        ]
    )
    penetration_depths = _finite([row.get("penetration_depth") for row in profiles])
    raw_penetration_depths = _finite([row.get("raw_penetration_depth") for row in profiles])
    expected_penetration_depths = _finite([row.get("expected_contact_penetration_depth") for row in profiles])
    illegal_penetration_depths = _finite([row.get("illegal_penetration_depth") for row in profiles])
    foot_errors = _finite([row.get("foot_sticking_error") for row in profiles])
    joint_violations = _finite([row.get("joint_limit_violation") for row in profiles])
    self_violations = _finite([row.get("self_collision_violation") for row in profiles])
    velocity_violations = _finite([row.get("velocity_limit_violation") for row in profiles])
    total_iterations = int(sum(int(row.get("actual_sqp_iterations", 0)) for row in profiles))
    iteration_counts = np.asarray([int(row.get("actual_sqp_iterations", 0)) for row in profiles], dtype=np.int64)
    solver_call_counts = np.asarray([int(row.get("convex_solver_calls", 0)) for row in profiles], dtype=np.int64)
    total_base_iterations = int(sum(int(row.get("base_sqp_iterations", 0)) for row in profiles))
    total_semantic_extra_iterations = int(sum(int(row.get("semantic_extra_iterations", 0)) for row in profiles))
    configured_semantic_extra_iterations = int(
        sum(int(row.get("configured_semantic_extra_iterations", 0)) for row in profiles)
    )
    rescued_frames = len({int(row["frame"]) for row in rescue_log})
    importance = np.asarray(
        [float(row.get("semantic_importance", 0.0) or 0.0) for row in profiles],
        dtype=np.float64,
    )
    semantic_active = importance > 0.20
    if semantic_active.size:
        semantic_active[0] = False  # frame 0 always uses the original 50-iteration initializer
    nonzero_iterations = iteration_counts[1:] if iteration_counts.size > 1 else iteration_counts
    objective_magnitudes = {
        name: _finite([row.get(name) for row in profiles if float(row.get("semantic_importance", 0.0) or 0.0) > 0.20])
        for name in (
            "E_omni",
            "E_part",
            "E_edge",
            "R_sem",
        )
    }
    summary = {
        "mode": mode,
        "num_frames": len(profiles),
        "total_wall_time": float(total_wall_time),
        "optimization_wall_time": float(sum(float(row.get("optimization_time", 0.0)) for row in profiles)),
        "mean_frame_time": float(frame_times.mean()) if frame_times.size else None,
        "p50_frame_time": float(np.percentile(frame_times, 50)) if frame_times.size else None,
        "p95_frame_time": float(np.percentile(frame_times, 95)) if frame_times.size else None,
        "total_actual_iterations": total_iterations,
        "total_base_iterations": total_base_iterations,
        "total_semantic_extra_iterations": total_semantic_extra_iterations,
        "configured_semantic_extra_iterations": configured_semantic_extra_iterations,
        "mean_actual_iterations_per_frame": total_iterations / max(len(profiles), 1),
        "median_actual_iterations_per_frame": (
            float(np.median(iteration_counts)) if iteration_counts.size else None
        ),
        "p95_actual_iterations_per_frame": (
            float(np.percentile(iteration_counts, 95)) if iteration_counts.size else None
        ),
        "max_actual_iterations_per_frame": int(iteration_counts.max()) if iteration_counts.size else None,
        "frames_stopped_at_1": int(np.sum(nonzero_iterations == 1)),
        "frames_stopped_at_2": int(np.sum(nonzero_iterations == 2)),
        "frames_using_3": int(np.sum(nonzero_iterations == 3)),
        "frames_using_4": int(np.sum(nonzero_iterations == 4)),
        "ordinary_sqp_total": int(iteration_counts[~semantic_active].sum()),
        "semantic_active_sqp_total": int(iteration_counts[semantic_active].sum()),
        "extra_semantic_sqp_total": int(np.maximum(iteration_counts[semantic_active] - 2, 0).sum()),
        "total_convex_solver_calls": int(solver_call_counts.sum()),
        "mean_E_omni_semantic_active": (
            float(objective_magnitudes["E_omni"].mean()) if objective_magnitudes["E_omni"].size else None
        ),
        "mean_E_part_semantic_active": (
            float(objective_magnitudes["E_part"].mean()) if objective_magnitudes["E_part"].size else None
        ),
        "mean_E_edge_semantic_active": (
            float(objective_magnitudes["E_edge"].mean()) if objective_magnitudes["E_edge"].size else None
        ),
        "mean_semantic_residual_semantic_active": (
            float(objective_magnitudes["R_sem"].mean())
            if objective_magnitudes["R_sem"].size
            else None
        ),
        "laplacian_residual_mean": float(lap_errors.mean()) if lap_errors.size else None,
        "laplacian_residual_median": float(np.median(lap_errors)) if lap_errors.size else None,
        "laplacian_residual_p95": float(np.percentile(lap_errors, 95)) if lap_errors.size else None,
        "laplacian_residual_max": float(lap_errors.max()) if lap_errors.size else None,
        "semantic_hand_object_error_mean": float(hand_errors.mean()) if hand_errors.size else None,
        "semantic_hand_object_error_p95": float(np.percentile(hand_errors, 95)) if hand_errors.size else None,
        "penetration_frame_ratio": float(np.mean(penetration_depths > 0)) if penetration_depths.size else None,
        "penetration_violation_frame_ratio": (
            float(np.mean(penetration_depths > penetration_tolerance)) if penetration_depths.size else None
        ),
        "penetration_tolerance": float(penetration_tolerance),
        "max_penetration_depth": float(penetration_depths.max()) if penetration_depths.size else None,
        "raw_penetration_frame_ratio": (
            float(np.mean(raw_penetration_depths > 0)) if raw_penetration_depths.size else None
        ),
        "max_raw_penetration_depth": (
            float(raw_penetration_depths.max()) if raw_penetration_depths.size else None
        ),
        "expected_contact_penetration_frame_ratio": (
            float(np.mean(expected_penetration_depths > 0)) if expected_penetration_depths.size else None
        ),
        "max_expected_contact_penetration_depth": (
            float(expected_penetration_depths.max()) if expected_penetration_depths.size else None
        ),
        "illegal_penetration_frame_ratio": (
            float(np.mean(illegal_penetration_depths > 0)) if illegal_penetration_depths.size else None
        ),
        "mean_illegal_penetration_depth": (
            float(illegal_penetration_depths[illegal_penetration_depths > 0].mean())
            if np.any(illegal_penetration_depths > 0)
            else 0.0
        ),
        "max_illegal_penetration_depth": (
            float(illegal_penetration_depths.max()) if illegal_penetration_depths.size else None
        ),
        "self_collision_violation_frame_ratio": float(np.mean(self_violations > 0)) if self_violations.size else None,
        "max_self_collision_violation": float(self_violations.max()) if self_violations.size else None,
        "mean_foot_sticking_error": float(foot_errors.mean()) if foot_errors.size else None,
        "max_foot_sticking_error": float(foot_errors.max()) if foot_errors.size else None,
        "max_joint_limit_violation": float(joint_violations.max()) if joint_violations.size else None,
        "max_velocity_limit_violation": float(velocity_violations.max()) if velocity_violations.size else None,
        "num_rescued_frames": rescued_frames,
        "rescue_rate": rescued_frames / max(len(profiles), 1),
        "extra_iterations_from_rescue": int(sum(int(row.get("actual_extra_iterations", 0)) for row in rescue_log)),
        "trajectory_semantic_weight_l1": float(
            sum(float(row.get("semantic_weight_l1", 0.0)) for row in profiles)
        ),
        "trajectory_semantic_weight_l2": float(
            np.sqrt(sum(float(row.get("semantic_weight_l2", 0.0)) ** 2 for row in profiles))
        ),
    }
    return _json_value(summary)


def write_run_artifacts(
    output_dir: str | Path,
    profiles: Sequence[dict[str, Any]],
    rescue_log: Sequence[dict[str, Any]],
    *,
    mode: str,
    total_wall_time: float,
    metadata: dict[str, Any] | None = None,
    penetration_tolerance: float = 1e-3,
    penetration_pairs: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    """Write profile CSV/JSON, rescue CSV, and aggregate run summary."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    normalized_profiles = [_json_value(dict(row)) for row in profiles]
    fieldnames = list(profiles[0]) if profiles else []
    with (output / "profile.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in profiles:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})

    summary = summarize_profiles(
        profiles,
        mode=mode,
        total_wall_time=total_wall_time,
        rescue_log=rescue_log,
        penetration_tolerance=penetration_tolerance,
    )
    profile_payload = {"metadata": _json_value(metadata or {}), "summary": summary, "frames": normalized_profiles}
    (output / "profile.json").write_text(
        json.dumps(profile_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    rescue_fields = ["frame", "old_budget", "new_budget", "reason", "actual_extra_iterations"]
    with (output / "rescue_log.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rescue_fields)
        writer.writeheader()
        for row in rescue_log:
            writer.writerow({key: _csv_value(row.get(key)) for key in rescue_fields})
    penetration_fields = [
        "frame",
        "geom_a",
        "geom_b",
        "body_a",
        "body_b",
        "signed_distance",
        "penetration_depth",
        "pair_type",
        "expected_contact",
        "illegal_penetration",
    ]
    with (output / "penetration_pairs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=penetration_fields)
        writer.writeheader()
        for row in penetration_pairs:
            writer.writerow({key: _csv_value(row.get(key)) for key in penetration_fields})
    (output / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def write_rows_csv(path: str | Path, rows: Sequence[dict[str, Any]]) -> None:
    """Write a heterogeneous list of diagnostic rows with stable first-seen columns."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})
