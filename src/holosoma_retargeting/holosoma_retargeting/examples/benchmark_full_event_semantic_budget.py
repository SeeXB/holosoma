"""Full-event Semantic Weight and exact-trigger refinement benchmark.

The benchmark keeps the historical four-event Legacy trajectory immutable,
introduces a separate all-event weighting mode, and changes only the maximum
Sequential SOCP iterations at nonzero semantic trigger frames.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import tyro

PACKAGE_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = PACKAGE_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from holosoma_retargeting.config_types.data_type import MotionDataConfig  # noqa: E402
from holosoma_retargeting.config_types.retargeting import RetargetingConfig  # noqa: E402
from holosoma_retargeting.config_types.robot import RobotConfig  # noqa: E402
from holosoma_retargeting.config_types.semantic import SemanticRetargetingConfig  # noqa: E402
from holosoma_retargeting.evaluation.eval_retargeting import (  # noqa: E402
    RetargetingEvaluator,
    create_task_constants,
)
from holosoma_retargeting.examples.robot_retarget import main as run_retargeting  # noqa: E402
from holosoma_retargeting.semantic_keyframes.precision import (  # noqa: E402
    PrecisionEvaluation,
    evaluate_precision,
    load_precision_payload,
)
from holosoma_retargeting.semantic_keyframes.runtime import (  # noqa: E402
    SemanticEvent,
    load_semantic_plan_projection,
)
from holosoma_retargeting.src.utils import (  # type: ignore[import-not-found]  # noqa: E402
    extract_foot_sticking_sequence_velocity,
    load_intermimic_data,
)


MM = 1000.0
BUDGETS = (2, 4, 6, 8, 10)
RANDOM_BUDGETS = (6, 10)
RANDOM_SEEDS = tuple(range(5))
INTERACTION_EVENTS = ("contact", "lift", "place", "release")
PHASE_EVENTS = ("approach", "carry_mid", "arrive")


@dataclass(frozen=True)
class RunSpec:
    key: str
    label: str
    mode: str
    budget: int = 2
    seed: int = 0


@dataclass
class BenchmarkConfig:
    """Registered OMOMO/sub3_largebox_003 full-event experiment."""

    task_name: str = "sub3_largebox_003"
    data_path: Path = Path("demo_data/OMOMO_new")
    semantic_keyframe_path: Path = Path(
        "demo_data/semantic_keyframes/sub3_largebox_003_semantic_v2.json"
    )
    output_dir: Path = Path("benchmark_results_full_event_semantic_budget")
    historical_final_dir: Path = Path("benchmark_results_semantic_final")
    force: bool = False
    aggregate_only: bool = False
    refresh_official: bool = False


LEGACY_SPEC = RunSpec(
    "legacy_4event",
    "Legacy-4Event",
    "uniform2_semantic_weight_uniform",
)
CURVE_SPECS = tuple(
    RunSpec(
        f"full_event_b{budget}",
        f"FullEvent-B{budget}",
        (
            "uniform2_semantic_weight_full_event"
            if budget == 2
            else "uniform2_semantic_weight_full_event_budget"
        ),
        budget,
    )
    for budget in BUDGETS
)
RANDOM_SPECS = tuple(
    RunSpec(
        f"random_b{budget}_seed_{seed}",
        f"Random-B{budget} seed={seed}",
        "uniform2_semantic_weight_full_event_random_budget",
        budget,
        seed,
    )
    for budget in RANDOM_BUDGETS
    for seed in RANDOM_SEEDS
)
ORIGINAL_B10_SPEC = RunSpec(
    "original_objective_b10",
    "OriginalObjective-B10",
    "uniform2_original_objective_semantic_budget",
    10,
)


def _trajectory(run_dir: Path, task_name: str) -> Path:
    return run_dir / f"{task_name}_original.npz"


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(field for row in rows for field in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _profile(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads((run_dir / "profile.json").read_text(encoding="utf-8"))
    return payload.get("metadata", {}), payload["summary"], payload["frames"]


def _run(config: BenchmarkConfig, spec: RunSpec) -> Path:
    run_dir = config.output_dir / "runs" / spec.key
    trajectory = _trajectory(run_dir, config.task_name)
    trace = run_dir / "semantic_iteration_trace.csv"
    trace_required = spec.mode.startswith("uniform2_semantic_weight_full_event")
    trace_has_frame34 = False
    if trace.exists() and trace_required:
        trace_has_frame34 = any(
            int(row["frame"]) == 34
            and float(row.get("distance_query_threshold") or np.nan) == 0.1
            for row in _read_csv(trace)
        )
    if (
        trajectory.exists()
        and (run_dir / "profile.json").exists()
        and ((trace.exists() and trace_has_frame34) or not trace_required)
        and not config.force
    ):
        print(f"Reusing completed full-event run: {spec.key}")
        return run_dir
    if config.aggregate_only:
        raise FileNotFoundError(f"missing completed run for --aggregate-only: {run_dir}")
    semantic = SemanticRetargetingConfig(
        mode=spec.mode,  # type: ignore[arg-type]
        semantic_keyframe_path=config.semantic_keyframe_path,
        profile_dir=run_dir,
        exact_trigger_budget=spec.budget,
        random_seed=spec.seed,
        diagnostic_iteration_frames=(34,) if trace_required else (),
    )
    semantic.validate()
    retargeting = RetargetingConfig(
        task_type="object_interaction",
        robot="g1",
        data_format="smplh",
        task_name=config.task_name,
        data_path=config.data_path,
        save_dir=run_dir,
        augmentation=False,
        semantic=semantic,
    )
    print(f"\n=== {spec.label} ({spec.mode}, trigger budget={spec.budget}) ===")
    run_retargeting(retargeting)
    return run_dir


def _official_evaluator() -> RetargetingEvaluator:
    robot = RobotConfig(robot_type="g1")
    motion = MotionDataConfig(data_format="smplh", robot_type="g1")
    constants = create_task_constants(robot, motion, object_name="largebox")
    return RetargetingEvaluator(
        robot_model_path=constants.ROBOT_URDF_FILE,
        object_model_path=constants.OBJECT_URDF_FILE,
        object_name=constants.OBJECT_NAME,
        demo_joints=constants.DEMO_JOINTS,
        joints_mapping=constants.JOINTS_MAPPING,
        visualize=False,
        constants=constants,
    )


def _official(
    config: BenchmarkConfig,
    run_dir: Path,
    evaluator: RetargetingEvaluator,
) -> dict[str, float]:
    cache = run_dir / "official_omniretarget_metrics.json"
    if cache.exists() and not config.refresh_official:
        payload = json.loads(cache.read_text(encoding="utf-8"))
        return {key: float(value) for key, value in payload.items()}
    result = evaluator.evaluate_trajectory(
        config.task_name,
        str(_trajectory(run_dir, config.task_name)),
        str(config.data_path),
    )
    if result is None:
        raise RuntimeError(f"official evaluator failed for {run_dir}")
    penetration = np.asarray(result["penetration_max_depths"], dtype=np.float64)
    skating = np.asarray(result["max_toe_sliding_velocities"], dtype=np.float64)
    metrics = {
        "penetration_duration": float(result["penetration_duration"]),
        "penetration_max_depth_m": float(penetration.max()) if penetration.size else 0.0,
        "foot_skating_duration": float(result["sliding_duration"]),
        "foot_skating_max_velocity": float(skating.max()) if skating.size else 0.0,
        "contact_preservation": float(result["contact_preservation"]),
    }
    cache.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def _event_subset(events: Sequence[SemanticEvent], names: Sequence[str]) -> list[SemanticEvent]:
    selected = [event for event in events if event.name in set(names)]
    if {event.name for event in selected} != set(names):
        raise ValueError(f"missing events: {sorted(set(names) - {event.name for event in selected})}")
    return selected


def _evaluations(path: Path, events: Sequence[SemanticEvent]) -> dict[str, PrecisionEvaluation]:
    payload = load_precision_payload(path)
    noninitial = [event for event in events if event.trigger_frame > 0]
    return {
        "all": evaluate_precision(payload, events, critical_event_names=None),
        "noninitial": evaluate_precision(payload, noninitial, critical_event_names=None),
        "interaction": evaluate_precision(
            payload,
            _event_subset(events, INTERACTION_EVENTS),
            critical_event_names=None,
        ),
        "phase": evaluate_precision(
            payload,
            _event_subset(events, PHASE_EVENTS),
            critical_event_names=None,
        ),
        "start": evaluate_precision(
            payload,
            _event_subset(events, ("start",)),
            critical_event_names=None,
        ),
    }


def _edge_exact(evaluation: PrecisionEvaluation) -> float:
    if evaluation.semantic_edge is None:
        return float("nan")
    return MM * float(evaluation.semantic_edge["exact"])


def _window_metric(
    evaluation: PrecisionEvaluation,
    family: str,
    window: str,
) -> float:
    payload = getattr(evaluation, family)
    if payload is None:
        return float("nan")
    return MM * float(payload[window])


def _metric_row(
    spec: RunSpec,
    run_dir: Path,
    evaluations: dict[str, PrecisionEvaluation],
    official: dict[str, float],
    semantic_frames: set[int],
) -> dict[str, Any]:
    metadata, summary, frames = _profile(run_dir)
    actual = np.asarray([int(frame["actual_sqp_iterations"]) for frame in frames], dtype=np.int64)
    calls = np.asarray(
        [int(frame.get("convex_solver_calls", frame["actual_sqp_iterations"])) for frame in frames],
        dtype=np.int64,
    )
    configured = np.asarray(
        [int(frame.get("configured_budget", frame["actual_sqp_iterations"])) for frame in frames],
        dtype=np.int64,
    )
    trigger_indices = np.asarray(sorted(semantic_frames), dtype=np.int64)
    ordinary_indices = np.asarray(
        [frame for frame in range(1, len(frames)) if frame not in semantic_frames],
        dtype=np.int64,
    )
    all_eval = evaluations["all"]
    interaction = evaluations["interaction"]
    phase = evaluations["phase"]
    start = evaluations["start"]
    noninitial = evaluations["noninitial"]
    return {
        "method": spec.label,
        "run_key": spec.key,
        "mode": spec.mode,
        "trigger_budget": spec.budget,
        "seed": spec.seed if "random" in spec.key else "",
        "configured_sqp_budget": int(configured.sum()),
        "actual_sqp": int(actual.sum()),
        "solver_calls": int(calls.sum()),
        "wall_time_s": float(summary["total_wall_time"]),
        "optimization_time_s": float(summary["optimization_wall_time"]),
        "frame0_actual_sqp": int(actual[0]),
        "semantic_trigger_actual_sqp": int(actual[trigger_indices].sum()),
        "ordinary_actual_sqp": int(actual[ordinary_indices].sum()),
        "ordinary_configured_min": int(configured[ordinary_indices].min()),
        "ordinary_configured_max": int(configured[ordinary_indices].max()),
        "ordinary_mm": MM * all_eval.ordinary["mean"],
        "global_kf_exact_mm": MM * all_eval.keyframe_global["exact"],
        "global_kf_pm1_mm": MM * all_eval.keyframe_global["pm1"],
        "global_kf_pm3_mm": MM * all_eval.keyframe_global["pm3"],
        "all_event_part_exact_mm": MM * all_eval.semantic_part["exact"],
        "all_event_part_pm1_mm": MM * all_eval.semantic_part["pm1"],
        "all_event_part_pm3_mm": MM * all_eval.semantic_part["pm3"],
        "noninitial_part_exact_mm": MM * noninitial.semantic_part["exact"],
        "interaction_part_exact_mm": MM * interaction.semantic_part["exact"],
        "interaction_part_pm1_mm": MM * interaction.semantic_part["pm1"],
        "interaction_part_pm3_mm": MM * interaction.semantic_part["pm3"],
        "interaction_local_exact_mm": MM * interaction.semantic_local["exact"],
        "interaction_local_pm1_mm": MM * interaction.semantic_local["pm1"],
        "interaction_local_pm3_mm": MM * interaction.semantic_local["pm3"],
        "interaction_edge_exact_mm": _window_metric(interaction, "semantic_edge", "exact"),
        "interaction_edge_pm1_mm": _window_metric(interaction, "semantic_edge", "pm1"),
        "interaction_edge_pm3_mm": _window_metric(interaction, "semantic_edge", "pm3"),
        "phase_part_exact_mm": MM * phase.semantic_part["exact"],
        "phase_part_pm1_mm": MM * phase.semantic_part["pm1"],
        "phase_part_pm3_mm": MM * phase.semantic_part["pm3"],
        "phase_local_exact_mm": MM * phase.semantic_local["exact"],
        "phase_local_pm1_mm": MM * phase.semantic_local["pm1"],
        "phase_local_pm3_mm": MM * phase.semantic_local["pm3"],
        "phase_edge_exact_mm": _window_metric(phase, "semantic_edge", "exact"),
        "phase_edge_pm1_mm": _window_metric(phase, "semantic_edge", "pm1"),
        "phase_edge_pm3_mm": _window_metric(phase, "semantic_edge", "pm3"),
        "start_part_exact_mm": MM * start.semantic_part["exact"],
        "start_local_exact_mm": MM * start.semantic_local["exact"],
        "local_exact_mm": MM * all_eval.semantic_local["exact"],
        "local_pm1_mm": MM * all_eval.semantic_local["pm1"],
        "local_pm3_mm": MM * all_eval.semantic_local["pm3"],
        "edge_exact_mm": _edge_exact(all_eval),
        "edge_pm1_mm": _window_metric(all_eval, "semantic_edge", "pm1"),
        "edge_pm3_mm": _window_metric(all_eval, "semantic_edge", "pm3"),
        "penetration_duration": official["penetration_duration"],
        "penetration_max_depth_mm": MM * official["penetration_max_depth_m"],
        "foot_skating_duration": official["foot_skating_duration"],
        "foot_skating_max_velocity": official["foot_skating_max_velocity"],
        "contact_preservation": official["contact_preservation"],
        "weight_event_count": len(metadata.get("weight_event_centers", {})),
        "weight_event_names": "|".join(metadata.get("weight_event_centers", {})),
    }


def _pct_change(value: float, baseline: float) -> float:
    return (float(value) - float(baseline)) / float(baseline) * 100.0


def _improvement(value: float, baseline: float) -> float:
    return -_pct_change(value, baseline)


def _physical_pass(row: dict[str, Any], baseline: dict[str, Any]) -> bool:
    tolerance = 1e-12
    return bool(
        row["penetration_duration"] <= baseline["penetration_duration"] + tolerance
        and row["penetration_max_depth_mm"] <= baseline["penetration_max_depth_mm"] + tolerance
        and row["foot_skating_duration"] <= baseline["foot_skating_duration"] + tolerance
        and row["foot_skating_max_velocity"] <= baseline["foot_skating_max_velocity"] + tolerance
        and row["contact_preservation"] + tolerance >= baseline["contact_preservation"]
    )


def _annotate(
    rows: Sequence[dict[str, Any]],
    u2: dict[str, Any],
    legacy: dict[str, Any],
    full_b2: dict[str, Any],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        row.update(
            {
                "ordinary_change_vs_u2_pct": _pct_change(row["ordinary_mm"], u2["ordinary_mm"]),
                "global_change_vs_u2_pct": _pct_change(
                    row["global_kf_exact_mm"], u2["global_kf_exact_mm"]
                ),
                "all_event_part_improvement_vs_u2_pct": _improvement(
                    row["all_event_part_exact_mm"], u2["all_event_part_exact_mm"]
                ),
                "all_event_part_improvement_vs_legacy_pct": _improvement(
                    row["all_event_part_exact_mm"], legacy["all_event_part_exact_mm"]
                ),
                "all_event_part_improvement_vs_b2_pct": _improvement(
                    row["all_event_part_exact_mm"], full_b2["all_event_part_exact_mm"]
                ),
                "interaction_part_improvement_vs_legacy_pct": _improvement(
                    row["interaction_part_exact_mm"], legacy["interaction_part_exact_mm"]
                ),
                "phase_part_improvement_vs_legacy_pct": _improvement(
                    row["phase_part_exact_mm"], legacy["phase_part_exact_mm"]
                ),
                "physical_pass_vs_u2": _physical_pass(row, u2),
                "registered_guardrail_pass": bool(
                    _pct_change(row["ordinary_mm"], u2["ordinary_mm"]) <= 1.0
                    and _pct_change(row["global_kf_exact_mm"], u2["global_kf_exact_mm"]) <= 0.5
                    and _physical_pass(row, u2)
                ),
            }
        )
        output.append(row)
    return output


def _aggregate_random(rows: Sequence[dict[str, Any]], budget: int) -> dict[str, Any]:
    aggregate: dict[str, Any] = {
        "method": f"Random-B{budget} mean",
        "run_key": f"random_b{budget}_mean",
        "mode": "uniform2_semantic_weight_full_event_random_budget",
        "trigger_budget": budget,
        "seed": "mean",
        "members": "|".join(str(row["run_key"]) for row in rows),
    }
    excluded = {"method", "run_key", "mode", "trigger_budget", "seed", "weight_event_names"}
    for field in rows[0]:
        if field in excluded:
            continue
        values: list[float] = []
        for row in rows:
            value = row.get(field)
            if isinstance(value, (bool, np.bool_)):
                values.append(float(value))
            elif isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value):
                values.append(float(value))
        if values:
            array = np.asarray(values, dtype=np.float64)
            aggregate[field] = float(array.mean())
            aggregate[f"{field}_std"] = float(array.std(ddof=0))
    aggregate["weight_event_names"] = rows[0]["weight_event_names"]
    return aggregate


def _event_map(evaluation: PrecisionEvaluation) -> dict[str, dict[str, Any]]:
    return {str(row["event"]): dict(row) for row in evaluation.event_rows}


def _event_group(event: SemanticEvent) -> str:
    if event.name in INTERACTION_EVENTS:
        return "interaction"
    if event.name in PHASE_EVENTS:
        return "phase"
    return "start"


def _full_event_rows(
    events: Sequence[SemanticEvent],
    evaluations: dict[str, dict[str, PrecisionEvaluation]],
    run_dirs: dict[str, Path],
) -> list[dict[str, Any]]:
    keys = ("uniform_2", "legacy_4event", "full_event_b2")
    maps = {key: _event_map(evaluations[key]["all"]) for key in keys}
    profiles = {key: _profile(run_dirs[key])[2] for key in keys}
    rows: list[dict[str, Any]] = []
    for event in events:
        row: dict[str, Any] = {
            "event": event.name,
            "event_group": _event_group(event),
            "trigger_frame": event.trigger_frame,
            "body_parts": "|".join(event.body_parts),
        }
        for key, prefix in (
            ("uniform_2", "u2"),
            ("legacy_4event", "legacy_4event"),
            ("full_event_b2", "full_event_b2"),
        ):
            metric = maps[key][event.name]
            profile = profiles[key][event.trigger_frame]
            row.update(
                {
                    f"{prefix}_part_exact_mm": MM * metric["part_exact"],
                    f"{prefix}_global_exact_mm": MM * metric["global_exact"],
                    f"{prefix}_local_exact_mm": MM * metric["local_exact"],
                    f"{prefix}_edge_exact_mm": (
                        MM * metric["edge_exact"] if metric.get("edge_exact") is not None else np.nan
                    ),
                    f"{prefix}_actual_sqp": int(profile["actual_sqp_iterations"]),
                }
            )
        full_profile = profiles["full_event_b2"][event.trigger_frame]
        row.update(
            {
                "full_vs_legacy_part_improvement_pct": _improvement(
                    row["full_event_b2_part_exact_mm"], row["legacy_4event_part_exact_mm"]
                ),
                "full_vs_u2_part_improvement_pct": _improvement(
                    row["full_event_b2_part_exact_mm"], row["u2_part_exact_mm"]
                ),
                "full_event_weight_l1": float(full_profile.get("semantic_weight_l1", 0.0)),
                "full_event_weight_active": bool(full_profile.get("semantic_weight_l1", 0.0) > 0),
                "full_event_profile_body_parts": "|".join(full_profile.get("semantic_body_parts", [])),
            }
        )
        rows.append(row)
    return rows


def _curve_event_rows(
    events: Sequence[SemanticEvent],
    evaluations: dict[str, dict[str, PrecisionEvaluation]],
    run_dirs: dict[str, Path],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in CURVE_SPECS:
        event_map = _event_map(evaluations[spec.key]["all"])
        profile = _profile(run_dirs[spec.key])[2]
        for event in events:
            metric = event_map[event.name]
            frame = profile[event.trigger_frame]
            rows.append(
                {
                    "trigger_budget": spec.budget,
                    "run_key": spec.key,
                    "event": event.name,
                    "event_group": _event_group(event),
                    "trigger_frame": event.trigger_frame,
                    "body_parts": "|".join(event.body_parts),
                    "configured_frame_budget": int(frame["configured_budget"]),
                    "actual_sqp": int(frame["actual_sqp_iterations"]),
                    "solver_calls": int(frame.get("convex_solver_calls", frame["actual_sqp_iterations"])),
                    "part_exact_mm": MM * metric["part_exact"],
                    "global_exact_mm": MM * metric["global_exact"],
                    "local_exact_mm": MM * metric["local_exact"],
                    "edge_exact_mm": (
                        MM * metric["edge_exact"] if metric.get("edge_exact") is not None else np.nan
                    ),
                }
            )
    by_event: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_event.setdefault(str(row["event"]), []).append(row)
    for event_rows in by_event.values():
        event_rows.sort(key=lambda row: int(row["trigger_budget"]))
        best_error = min(float(row["part_exact_mm"]) for row in event_rows)
        base_error = float(event_rows[0]["part_exact_mm"])
        saturation = next(
            int(row["trigger_budget"])
            for row in event_rows
            if (float(row["part_exact_mm"]) - best_error) / base_error * 100.0 <= 0.1
        )
        previous: dict[str, Any] | None = None
        for row in event_rows:
            row["part_gain_vs_previous_pct"] = (
                np.nan
                if previous is None
                else _improvement(row["part_exact_mm"], previous["part_exact_mm"])
            )
            row["saturation_budget_within_0_1pct_of_best"] = saturation
            previous = row
    return rows


def _legacy_compatibility(
    config: BenchmarkConfig,
    current_dir: Path,
    historical_dir: Path,
    current_row: dict[str, Any],
    historical_row: dict[str, Any],
) -> dict[str, Any]:
    with np.load(_trajectory(current_dir, config.task_name), allow_pickle=False) as current, np.load(
        _trajectory(historical_dir, config.task_name), allow_pickle=False
    ) as historical:
        qpos_delta = float(np.max(np.abs(current["qpos"] - historical["qpos"])))
        current_sqp = int(np.asarray(current["actual_sqp_iterations"]).sum())
        historical_sqp = int(np.asarray(historical["actual_sqp_iterations"]).sum())
    fields = (
        "ordinary_mm",
        "global_kf_exact_mm",
        "global_kf_pm1_mm",
        "global_kf_pm3_mm",
        "all_event_part_exact_mm",
        "interaction_part_exact_mm",
        "phase_part_exact_mm",
        "local_exact_mm",
        "edge_exact_mm",
        "penetration_duration",
        "penetration_max_depth_mm",
        "foot_skating_duration",
        "foot_skating_max_velocity",
        "contact_preservation",
    )
    deltas = {
        field: abs(float(current_row[field]) - float(historical_row[field]))
        for field in fields
        if np.isfinite(current_row[field]) and np.isfinite(historical_row[field])
    }
    max_delta = max(deltas.values(), default=0.0)
    return {
        "qpos_max_abs_diff": qpos_delta,
        "current_actual_sqp": current_sqp,
        "historical_actual_sqp": historical_sqp,
        "actual_sqp_diff": current_sqp - historical_sqp,
        "evaluator_metric_max_abs_diff": max_delta,
        "evaluator_metric_deltas_json": json.dumps(deltas, sort_keys=True),
        "qpos_zero_drift": qpos_delta == 0.0,
        "sqp_is_440": current_sqp == historical_sqp == 440,
        "evaluator_zero_drift": max_delta == 0.0,
        "legacy_zero_drift": qpos_delta == 0.0 and current_sqp == historical_sqp == 440 and max_delta == 0.0,
    }


def _iteration_trace_rows(run_dirs: dict[str, Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in CURVE_SPECS:
        path = run_dirs[spec.key] / "semantic_iteration_trace.csv"
        for source in _read_csv(path):
            rows.append(
                {
                    "run_key": spec.key,
                    "trigger_budget": spec.budget,
                    "frame": int(source["frame"]),
                    "event": source["event"],
                    "body_parts": source["body_parts"],
                    "trace_kind": source.get("trace_kind", "exact_trigger"),
                    "iteration": int(source["iteration"]),
                    "configured_budget": int(source["configured_budget"]),
                    "part_error_mm": MM * float(source["part_error"]),
                    "local_error_mm": MM * float(source["local_error"]),
                    "edge_error_mm": (
                        MM * float(source["edge_error"]) if source["edge_error"] else np.nan
                    ),
                    "global_laplacian_error_mm": MM * float(source["global_laplacian_error"]),
                    "cost": float(source["cost"]),
                    "q_update_l2_norm": float(source.get("q_update_l2_norm") or np.nan),
                    "q_update_max_abs": float(source.get("q_update_max_abs") or np.nan),
                    "distance_query_threshold_m": float(
                        source.get("distance_query_threshold") or np.nan
                    ),
                    "shoulder_box_signed_distance_mm": (
                        MM * float(source["shoulder_box_signed_distance"])
                        if source.get("shoulder_box_signed_distance")
                        else np.nan
                    ),
                    "shoulder_box_penetration_depth_mm": (
                        MM * float(source["shoulder_box_penetration_depth"])
                        if source.get("shoulder_box_penetration_depth")
                        else np.nan
                    ),
                    "max_illegal_penetration_depth_mm": (
                        MM * float(source["max_illegal_penetration_depth"])
                        if source.get("max_illegal_penetration_depth")
                        else np.nan
                    ),
                    "illegal_penetration_pair_count": (
                        int(source["illegal_penetration_pair_count"])
                        if source.get("illegal_penetration_pair_count")
                        else np.nan
                    ),
                    "part_vertex_count": int(source["part_vertex_count"]),
                    "edge_count": int(source["edge_count"]),
                }
            )
    return rows


def _trajectory_qpos(run_dir: Path, task_name: str) -> np.ndarray:
    with np.load(_trajectory(run_dir, task_name), allow_pickle=False) as payload:
        return np.asarray(payload["qpos"], dtype=np.float64).copy()


def _foot_sliding_detail(
    evaluator: RetargetingEvaluator,
    qpos: np.ndarray,
    contact_sequences: Sequence[dict[str, bool]],
) -> dict[str, Any]:
    """Reproduce the official foot-skating metric while retaining frame ids."""
    toe_positions = np.asarray(
        [
            evaluator._get_robot_link_positions(  # noqa: SLF001 - exact official diagnostic
                q,
                ["left_ankle_roll_sphere_5_link", "right_ankle_roll_sphere_5_link"],
            )
            for q in qpos
        ],
        dtype=np.float64,
    )
    velocities = np.zeros((len(qpos), 2), dtype=np.float64)
    velocities[1:] = np.linalg.norm(np.diff(toe_positions[:, :, :2], axis=0), axis=2)
    contacts = contact_sequences[: len(qpos)]
    sticking = np.asarray(
        [[bool(row["L_Toe"]), bool(row["R_Toe"])] for row in contacts],
        dtype=bool,
    )
    sliding = sticking & (velocities > float(evaluator.sliding_threshold))
    per_frame = np.max(velocities * sliding, axis=1)
    frames = np.flatnonzero(per_frame > 0).astype(int).tolist()
    denominator = int(np.sum(np.any(sticking, axis=1)))
    return {
        "foot_skating_duration": len(frames) / max(denominator, 1),
        "foot_skating_max_velocity": float(per_frame.max(initial=0.0)),
        "foot_skating_frames": frames,
    }


def _penetration_violation_rows(
    run_key: str,
    budget: int,
    run_dir: Path,
    threshold_m: float,
    u2_official_frames: set[int],
) -> list[dict[str, Any]]:
    """Retain every pair that contributes to the unchanged official metric."""
    rows: list[dict[str, Any]] = []
    for source in _read_csv(run_dir / "penetration_pairs.csv"):
        depth = float(source["penetration_depth"])
        pair_type = source["pair_type"]
        if depth <= threshold_m or pair_type in {"object-ground", "self-collision"}:
            continue
        frame = int(source["frame"])
        rows.append(
            {
                "run_key": run_key,
                "trigger_budget": budget,
                "frame": frame,
                "geom_a": source["geom_a"],
                "geom_b": source["geom_b"],
                "pair_type": pair_type,
                "signed_distance_mm": MM * float(source["signed_distance"]),
                "penetration_depth_mm": MM * depth,
                "official_threshold_mm": MM * threshold_m,
                "illegal_penetration": source["illegal_penetration"].lower() == "true",
                "new_vs_u2": frame not in u2_official_frames,
            }
        )
    return rows


def _physical_detail(
    config: BenchmarkConfig,
    spec: RunSpec,
    run_dir: Path,
    evaluator: RetargetingEvaluator,
    contact_sequences: Sequence[dict[str, bool]],
    u2_detail: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _, summary, frames = _profile(run_dir)
    qpos = _trajectory_qpos(run_dir, config.task_name)
    foot = _foot_sliding_detail(evaluator, qpos, contact_sequences)
    threshold = float(evaluator.penetration_tolerance)
    u2_official_frames = (
        set(int(frame) for frame in u2_detail["official_penetration_frames"])
        if u2_detail is not None
        else set()
    )
    violation_rows = _penetration_violation_rows(
        spec.key,
        spec.budget,
        run_dir,
        threshold,
        u2_official_frames,
    )
    official_penetration_frames = sorted({int(row["frame"]) for row in violation_rows})
    illegal_frames = [
        int(frame["frame_idx"])
        for frame in frames
        if float(frame.get("illegal_penetration_depth") or 0.0) > threshold
    ]
    joint_frames = [
        int(frame["frame_idx"])
        for frame in frames
        if float(frame.get("joint_limit_violation") or 0.0) > 1e-6
    ]
    self_collision_frames = [
        int(frame["frame_idx"])
        for frame in frames
        if frame.get("self_collision_violation") is not None
        and float(frame["self_collision_violation"]) > 1e-3
    ]
    foot_frames = [int(frame) for frame in foot["foot_skating_frames"]]
    baseline_sets = {
        "official": set(u2_detail["official_penetration_frames"]) if u2_detail else set(),
        "illegal": set(u2_detail["illegal_penetration_frames"]) if u2_detail else set(),
        "joint": set(u2_detail["joint_limit_frames"]) if u2_detail else set(),
        "self": set(u2_detail["self_collision_frames"]) if u2_detail else set(),
        "foot": set(u2_detail["foot_skating_frames"]) if u2_detail else set(),
    }
    detail = {
        "run_key": spec.key,
        "trigger_budget": spec.budget,
        "official_penetration_duration": len(official_penetration_frames) / max(len(frames), 1),
        "official_penetration_max_depth_mm": max(
            (float(row["penetration_depth_mm"]) for row in violation_rows),
            default=0.0,
        ),
        "official_penetration_frames": official_penetration_frames,
        "illegal_penetration_duration": len(illegal_frames) / max(len(frames), 1),
        "illegal_penetration_max_depth_mm": MM
        * max((float(frame.get("illegal_penetration_depth") or 0.0) for frame in frames), default=0.0),
        "illegal_penetration_frames": illegal_frames,
        "foot_skating_duration": foot["foot_skating_duration"],
        "foot_skating_max_velocity": foot["foot_skating_max_velocity"],
        "foot_skating_frames": foot_frames,
        "contact_preservation": float(
            json.loads((run_dir / "official_omniretarget_metrics.json").read_text())["contact_preservation"]
        ),
        "joint_limit_max_violation": float(summary.get("max_joint_limit_violation") or 0.0),
        "joint_limit_frames": joint_frames,
        "self_collision_configured": bool(
            json.loads((run_dir / "profile.json").read_text())["metadata"].get(
                "self_collision_check_configured",
                False,
            )
        ),
        "self_collision_max_violation": float(summary.get("max_self_collision_violation") or 0.0),
        "self_collision_frames": self_collision_frames,
        "new_official_penetration_frames_vs_u2": sorted(set(official_penetration_frames) - baseline_sets["official"]),
        "new_illegal_penetration_frames_vs_u2": sorted(set(illegal_frames) - baseline_sets["illegal"]),
        "new_joint_limit_frames_vs_u2": sorted(set(joint_frames) - baseline_sets["joint"]),
        "new_self_collision_frames_vs_u2": sorted(set(self_collision_frames) - baseline_sets["self"]),
        "new_foot_skating_frames_vs_u2": sorted(set(foot_frames) - baseline_sets["foot"]),
    }
    return detail, violation_rows


def _physical_csv_row(detail: dict[str, Any]) -> dict[str, Any]:
    row = dict(detail)
    for field in tuple(row):
        if field.endswith("_frames") or "_frames_vs_u2" in field:
            row[field] = json.dumps(row[field])
    return row


def _frame34_rows(
    config: BenchmarkConfig,
    run_dirs: dict[str, Path],
    curve_rows: Sequence[dict[str, Any]],
    trace_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    baselines = {
        key: _trajectory_qpos(run_dirs[key], config.task_name)[34]
        for key in ("uniform_2", "legacy_4event", "full_event_b2")
    }
    by_budget = {int(row["trigger_budget"]): row for row in curve_rows}
    output: list[dict[str, Any]] = []
    for spec in CURVE_SPECS:
        qpos = _trajectory_qpos(run_dirs[spec.key], config.task_name)[34]
        _, _, frames = _profile(run_dirs[spec.key])
        frame = frames[34]
        iterations = sorted(
            [
                row
                for row in trace_rows
                if row["run_key"] == spec.key and int(row["frame"]) == 34
            ],
            key=lambda row: int(row["iteration"]),
        )
        if not iterations:
            raise ValueError(f"missing frame 34 iteration trace for {spec.key}")
        final = iterations[-1]
        summary = by_budget[spec.budget]
        output.append(
            {
                "run_key": spec.key,
                "trigger_budget": spec.budget,
                "frame": 34,
                "configured_trigger_max_iter": spec.budget,
                "frame34_configured_max_iter": int(frame["configured_budget"]),
                "total_actual_sqp": int(summary["actual_sqp"]),
                "frame34_actual_sqp": int(frame["actual_sqp_iterations"]),
                "frame34_solver_calls": int(frame.get("convex_solver_calls", frame["actual_sqp_iterations"])),
                "qpos_json": json.dumps(qpos.tolist()),
                "qpos_max_abs_diff_vs_u2": float(np.max(np.abs(qpos - baselines["uniform_2"]))),
                "qpos_l2_diff_vs_u2": float(np.linalg.norm(qpos - baselines["uniform_2"])),
                "qpos_max_abs_diff_vs_legacy": float(np.max(np.abs(qpos - baselines["legacy_4event"]))),
                "qpos_l2_diff_vs_legacy": float(np.linalg.norm(qpos - baselines["legacy_4event"])),
                "qpos_max_abs_diff_vs_b2": float(np.max(np.abs(qpos - baselines["full_event_b2"]))),
                "qpos_l2_diff_vs_b2": float(np.linalg.norm(qpos - baselines["full_event_b2"])),
                "shoulder_box_signed_distance_mm": final["shoulder_box_signed_distance_mm"],
                "shoulder_box_penetration_depth_mm": final["shoulder_box_penetration_depth_mm"],
                "illegal_at_official_10mm_threshold": bool(
                    float(final["shoulder_box_penetration_depth_mm"]) > 10.0
                ),
                "part_error_mm": final["part_error_mm"],
                "global_laplacian_error_mm": final["global_laplacian_error_mm"],
                "solver_objective": final["cost"],
            }
        )
    return output


def _pareto_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output = [dict(row) for row in rows]
    feasible = [row for row in output if bool(row.get("registered_guardrail_pass", False))]
    for row in output:
        row["compute_part_pareto"] = bool(
            row in feasible
            and not any(
                float(other["actual_sqp"]) <= float(row["actual_sqp"])
                and float(other["all_event_part_exact_mm"]) <= float(row["all_event_part_exact_mm"])
                and (
                    float(other["actual_sqp"]) < float(row["actual_sqp"])
                    or float(other["all_event_part_exact_mm"]) < float(row["all_event_part_exact_mm"])
                )
                for other in feasible
            )
        )
    return output


def _recommended_budget(curve_rows: Sequence[dict[str, Any]]) -> int:
    feasible = [row for row in curve_rows if bool(row["registered_guardrail_pass"])]
    if not feasible:
        return 0
    return min(int(row["trigger_budget"]) for row in feasible)


def _registered_physical_pass(
    detail: dict[str, Any],
    u2_detail: dict[str, Any],
) -> bool:
    tolerance = 1e-12
    return bool(
        not detail["official_penetration_frames"]
        and not detail["illegal_penetration_frames"]
        and not detail["joint_limit_frames"]
        and not detail["self_collision_frames"]
        and float(detail["foot_skating_duration"])
        <= float(u2_detail["foot_skating_duration"]) + tolerance
        and float(detail["foot_skating_max_velocity"])
        <= float(u2_detail["foot_skating_max_velocity"]) + tolerance
        and float(detail["contact_preservation"]) + tolerance
        >= float(u2_detail["contact_preservation"])
    )


def _budget_curve_plots(
    output_dir: Path,
    curve_rows: Sequence[dict[str, Any]],
    frame34_rows: Sequence[dict[str, Any]],
    frame34_trace: Sequence[dict[str, Any]],
) -> None:
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(curve_rows, key=lambda row: int(row["trigger_budget"]))
    budgets = [int(row["trigger_budget"]) for row in ordered]
    curves = (
        ("all_event_part_exact_mm", "budget_vs_all_event_part.png", "All-event Part Exact", "Part Exact (mm)"),
        ("phase_part_exact_mm", "budget_vs_phase_part.png", "Phase Part Exact", "Part Exact (mm)"),
        ("interaction_part_exact_mm", "budget_vs_interaction_part.png", "Interaction Part Exact", "Part Exact (mm)"),
        ("global_kf_exact_mm", "budget_vs_global.png", "Global keyframe error", "Global Exact (mm)"),
        ("actual_sqp", "budget_vs_actual_sqp.png", "Actual SQP", "Actual SQP"),
        ("wall_time_s", "budget_vs_wall_time.png", "Wall time", "Wall time (s)"),
        ("official_penetration_max_depth_mm", "budget_vs_penetration_depth.png", "Official penetration depth", "Depth (mm)"),
    )
    for field, filename, title, ylabel in curves:
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        ax.plot(budgets, [float(row[field]) for row in ordered], marker="o")
        if field == "official_penetration_max_depth_mm":
            ax.axhline(10.0, color="tab:red", linestyle="--", label="official threshold")
            ax.legend()
        ax.set(xlabel="Semantic trigger max SQP", ylabel=ylabel, title=title, xticks=BUDGETS)
        fig.tight_layout()
        fig.savefig(plot_dir / filename, dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for budget in BUDGETS:
        values = sorted(
            [row for row in frame34_trace if int(row["trigger_budget"]) == budget],
            key=lambda row: int(row["iteration"]),
        )
        ax.plot(
            [int(row["iteration"]) for row in values],
            [float(row["shoulder_box_penetration_depth_mm"]) for row in values],
            marker="o",
            label=f"B{budget}",
        )
    ax.axhline(10.0, color="tab:red", linestyle="--", label="official threshold")
    ax.set(
        xlabel="frame 34 SQP iteration",
        ylabel="left shoulder-box penetration (mm)",
        title="Frame 34 nonlinear penetration by accepted iterate",
        xticks=(1, 2),
    )
    ax.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(plot_dir / "frame34_penetration_vs_iteration.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for row in ordered:
        feasible = bool(row["registered_guardrail_pass"])
        ax.scatter(
            float(row["actual_sqp"]),
            float(row["all_event_part_exact_mm"]),
            marker="o" if feasible else "x",
            s=70,
            color="tab:green" if feasible else "tab:red",
        )
        ax.annotate(
            f"B{row['trigger_budget']}",
            (float(row["actual_sqp"]), float(row["all_event_part_exact_mm"])),
            xytext=(5, 4),
            textcoords="offset points",
        )
    ax.set(
        xlabel="Actual SQP",
        ylabel="All-event Part Exact (mm)",
        title="FullEvent quality-compute Pareto (green = registered feasible)",
    )
    fig.tight_layout()
    fig.savefig(plot_dir / "quality_compute_pareto.png", dpi=180)
    plt.close(fig)


def _plots(
    output_dir: Path,
    full_event_rows: Sequence[dict[str, Any]],
    curve_rows: Sequence[dict[str, Any]],
    curve_event_rows: Sequence[dict[str, Any]],
    random_means: dict[int, dict[str, Any]],
    trace_rows: Sequence[dict[str, Any]],
) -> None:
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    names = [str(row["event"]) for row in full_event_rows]
    x = np.arange(len(names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 5))
    for offset, field, label in (
        (-width, "u2_part_exact_mm", "U2"),
        (0.0, "legacy_4event_part_exact_mm", "Legacy-4Event"),
        (width, "full_event_b2_part_exact_mm", "FullEvent-B2"),
    ):
        ax.bar(x + offset, [row[field] for row in full_event_rows], width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set(ylabel="Part Exact (mm, lower is better)", title="Full-event weighting vs historical Legacy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "full_event_vs_legacy_event_errors.png", dpi=180)
    plt.close(fig)

    ordered = sorted(curve_rows, key=lambda row: int(row["trigger_budget"]))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        [row["actual_sqp"] for row in ordered],
        [row["all_event_part_exact_mm"] for row in ordered],
        marker="o",
    )
    for row in ordered:
        ax.annotate(f"B{row['trigger_budget']}", (row["actual_sqp"], row["all_event_part_exact_mm"]), xytext=(4, 4), textcoords="offset points")
    ax.set(xlabel="Actual SQP", ylabel="All-Event Part Exact (mm)", title="Semantic budget quality-compute curve")
    fig.tight_layout()
    fig.savefig(plot_dir / "semantic_budget_curve.png", dpi=180)
    plt.close(fig)

    for field, filename, title in (
        ("interaction_part_exact_mm", "interaction_budget_curve.png", "Interaction-event budget curve"),
        ("phase_part_exact_mm", "phase_budget_curve.png", "Phase-event budget curve"),
    ):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot([row["trigger_budget"] for row in ordered], [row[field] for row in ordered], marker="o")
        ax.set(xlabel="Configured trigger budget", ylabel="Part Exact (mm)", title=title, xticks=BUDGETS)
        fig.tight_layout()
        fig.savefig(plot_dir / filename, dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    positions = np.arange(len(RANDOM_BUDGETS))
    semantic_values = [
        next(float(row["all_event_part_exact_mm"]) for row in ordered if row["trigger_budget"] == budget)
        for budget in RANDOM_BUDGETS
    ]
    random_values = [float(random_means[budget]["all_event_part_exact_mm"]) for budget in RANDOM_BUDGETS]
    random_std = [float(random_means[budget]["all_event_part_exact_mm_std"]) for budget in RANDOM_BUDGETS]
    ax.bar(positions - 0.18, semantic_values, 0.36, label="Semantic timing")
    ax.bar(positions + 0.18, random_values, 0.36, yerr=random_std, capsize=4, label="Random timing")
    ax.set_xticks(positions)
    ax.set_xticklabels([f"B{budget}" for budget in RANDOM_BUDGETS])
    ax.set(ylabel="All-Event Part Exact (mm)", title="Semantic vs random compute placement")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "semantic_vs_random_budget.png", dpi=180)
    plt.close(fig)

    b10_trace = [row for row in trace_rows if int(row["trigger_budget"]) == 10]
    fig, ax = plt.subplots(figsize=(10, 6))
    for event in [row["event"] for row in full_event_rows if row["event"] != "start"]:
        values = sorted(
            [row for row in b10_trace if row["event"] == event],
            key=lambda row: int(row["iteration"]),
        )
        if values:
            ax.plot(
                [row["iteration"] for row in values],
                [row["part_error_mm"] for row in values],
                marker="o",
                label=event,
            )
    ax.set(xlabel="Actual Sequential SOCP iteration", ylabel="Part Error (mm)", title="Per-event B10 iteration trace")
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(plot_dir / "per_event_iteration_trace.png", dpi=180)
    plt.close(fig)


def _stage_a_plot(output_dir: Path, full_event_rows: Sequence[dict[str, Any]]) -> None:
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    names = [str(row["event"]) for row in full_event_rows]
    x = np.arange(len(names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 5))
    for offset, field, label in (
        (-width, "u2_part_exact_mm", "U2"),
        (0.0, "legacy_4event_part_exact_mm", "Legacy-4Event"),
        (width, "full_event_b2_part_exact_mm", "FullEvent-B2"),
    ):
        ax.bar(x + offset, [row[field] for row in full_event_rows], width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set(ylabel="Part Exact (mm, lower is better)", title="Full-event weighting vs historical Legacy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "full_event_vs_legacy_event_errors.png", dpi=180)
    plt.close(fig)


def _physical_failure_rows(
    config: BenchmarkConfig,
    full_dir: Path,
    legacy_dir: Path,
    events: Sequence[SemanticEvent],
    penetration_threshold_m: float,
) -> list[dict[str, Any]]:
    full_profile = _profile(full_dir)[2]
    legacy_profile = _profile(legacy_dir)[2]
    with np.load(_trajectory(full_dir, config.task_name), allow_pickle=False) as full_npz, np.load(
        _trajectory(legacy_dir, config.task_name), allow_pickle=False
    ) as legacy_npz:
        frame_qpos_delta = np.max(
            np.abs(
                np.asarray(full_npz["qpos"], dtype=np.float64)
                - np.asarray(legacy_npz["qpos"], dtype=np.float64)
            ),
            axis=1,
        )
    event_by_frame = {event.trigger_frame: event.name for event in events}
    rows: list[dict[str, Any]] = []
    for source in _read_csv(full_dir / "penetration_pairs.csv"):
        depth = float(source["penetration_depth"])
        pair_type = source["pair_type"]
        if depth <= penetration_threshold_m or pair_type in {"object-ground", "self-collision"}:
            continue
        frame = int(source["frame"])
        full = full_profile[frame]
        legacy = legacy_profile[frame]
        rows.append(
            {
                "frame": frame,
                "exact_trigger_event": event_by_frame.get(frame, ""),
                "active_events": "|".join(full.get("active_event", [])),
                "nearest_trigger": full.get("nearest_trigger"),
                "geom_a": source["geom_a"],
                "geom_b": source["geom_b"],
                "pair_type": pair_type,
                "penetration_depth_mm": MM * depth,
                "official_failure_threshold_mm": MM * penetration_threshold_m,
                "full_event_weight_l1": float(full.get("semantic_weight_l1", 0.0)),
                "full_event_semantic_body_parts": "|".join(full.get("semantic_body_parts", [])),
                "legacy_weight_l1_same_frame": float(legacy.get("semantic_weight_l1", 0.0)),
                "legacy_penetration_depth_mm_same_frame": MM
                * float(legacy.get("penetration_depth", 0.0)),
                "full_vs_legacy_qpos_max_abs_diff_same_frame": float(frame_qpos_delta[frame]),
                "full_vs_legacy_qpos_max_abs_diff_previous_frame": (
                    float(frame_qpos_delta[frame - 1]) if frame > 0 else np.nan
                ),
            }
        )
    return rows


def _report(
    path: Path,
    compatibility: dict[str, Any],
    rows: dict[str, dict[str, Any]],
    full_event_rows: Sequence[dict[str, Any]],
    curve_event_rows: Sequence[dict[str, Any]],
    random_means: dict[int, dict[str, Any]],
    recommended_budget: int,
    full_event_keep: bool,
    budget_keep: bool,
) -> None:
    u2 = rows["uniform_2"]
    legacy = rows["legacy_4event"]
    b2 = rows["full_event_b2"]
    curve = [rows[f"full_event_b{budget}"] for budget in BUDGETS]
    original_b10 = rows["original_objective_b10"]
    phase_events = [row for row in full_event_rows if row["event_group"] == "phase"]
    saturation = {
        event: next(
            int(row["saturation_budget_within_0_1pct_of_best"])
            for row in curve_event_rows
            if row["event"] == event
        )
        for event in (*INTERACTION_EVENTS, *PHASE_EVENTS)
    }
    budget_curve_text = ", ".join(
        f"B{row['trigger_budget']}={row['all_event_part_exact_mm']:.4f} mm/{row['actual_sqp']} SQP"
        for row in curve
    )
    lines = [
        "# Full-Event Semantic Weight + Budget Report",
        "",
        "All optimizer events come from the unchanged deterministic semantic_v2 projection. Criticality is ignored (uniform event strength), the registered Legacy multipliers/kernel are unchanged, and ordinary frames remain configured at two iterations.",
        "",
        "## Decisions",
        "",
        f"- **FullEvent Weight: {'Keep' if full_event_keep else 'Drop'}**",
        f"- **Semantic Budget: {'Keep' if budget_keep else 'Drop'}**",
        f"- **Recommended trigger budget: B{recommended_budget}** (minimum normalized distance to the feasible quality/compute ideal among the registered budgets).",
        "",
        "## Q1–Q10",
        "",
        f"1. **Legacy zero drift:** `{compatibility['legacy_zero_drift']}`; qpos max abs diff {compatibility['qpos_max_abs_diff']:.3g}, evaluator max diff {compatibility['evaluator_metric_max_abs_diff']:.3g}, SQP {compatibility['current_actual_sqp']} (historical {compatibility['historical_actual_sqp']}).",
        f"2. **FullEvent-B2 vs Legacy:** All-event Part {_improvement(b2['all_event_part_exact_mm'], legacy['all_event_part_exact_mm']):+.3f}%, Phase Part {_improvement(b2['phase_part_exact_mm'], legacy['phase_part_exact_mm']):+.3f}%, Interaction Part {_improvement(b2['interaction_part_exact_mm'], legacy['interaction_part_exact_mm']):+.3f}%; Global vs U2 {b2['global_change_vs_u2_pct']:+.3f}%, Ordinary vs U2 {b2['ordinary_change_vs_u2_pct']:+.3f}%.",
        "3. **Phase events:** " + "; ".join(
            f"{row['event']} {_improvement(row['full_event_b2_part_exact_mm'], row['legacy_4event_part_exact_mm']):+.3f}% (weight active={row['full_event_weight_active']})"
            for row in phase_events
        ) + ".",
        f"4. **Budget curve:** {budget_curve_text}.",
        "5. **Interaction saturation** (first registered budget within 0.1% of that event's best observed Part): "
        + ", ".join(f"{event}=B{saturation[event]}" for event in INTERACTION_EVENTS) + ".",
        "6. **Phase-event extra-compute value:** "
        + ", ".join(f"{event}=B{saturation[event]}" for event in PHASE_EVENTS)
        + " under the same operational saturation definition.",
        f"7. **Semantic vs Random:** B6 {rows['full_event_b6']['all_event_part_exact_mm']:.4f} vs {random_means[6]['all_event_part_exact_mm']:.4f}±{random_means[6]['all_event_part_exact_mm_std']:.4f} mm; B10 {rows['full_event_b10']['all_event_part_exact_mm']:.4f} vs {random_means[10]['all_event_part_exact_mm']:.4f}±{random_means[10]['all_event_part_exact_mm_std']:.4f} mm.",
        f"8. **OriginalObjective-B10:** All-event Part {original_b10['all_event_part_exact_mm']:.4f} mm ({_improvement(original_b10['all_event_part_exact_mm'], b2['all_event_part_exact_mm']):+.3f}% vs FullEvent-B2), compared with FullEvent-B10 {rows['full_event_b10']['all_event_part_exact_mm']:.4f} mm.",
        f"9. **Quality-compute operating point:** B{recommended_budget}; see `compute_pareto.csv` for every feasible and nondominated point.",
        f"10. **Final recommendation:** FullEvent Weight={'Keep' if full_event_keep else 'Drop'}, Semantic Budget={'Keep' if budget_keep else 'Drop'}, budget=B{recommended_budget}.",
        "",
        "## Guardrails and definitions",
        "",
        f"FullEvent-B2 guardrail pass = `{b2['registered_guardrail_pass']}` (Ordinary <= +1%, all-event Global KF Exact <= +0.5%, no physical regression vs U2). All-event means all eight projected events including start@0; Phase and Interaction exclude start. `semantic_iteration_trace.csv` records true nonlinear, unweighted geometry after every actual solve; trace diagnostic time is excluded from reported retarget wall time.",
        "",
        "The saturation column is descriptive, not a tuned optimizer threshold: it is the first registered budget whose event Part error lies within 0.1% of the best error observed for that event on this fixed curve.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _curve_report(
    path: Path,
    compatibility: dict[str, Any],
    curve_rows: Sequence[dict[str, Any]],
    frame34_rows: Sequence[dict[str, Any]],
    frame34_trace: Sequence[dict[str, Any]],
    recommended_budget: int,
    full_event_status: str,
    budget_keep: bool,
) -> None:
    ordered = sorted(curve_rows, key=lambda row: int(row["trigger_budget"]))
    frame_ordered = sorted(frame34_rows, key=lambda row: int(row["trigger_budget"]))
    first_physical = next(
        (
            int(row["trigger_budget"])
            for row in ordered
            if bool(row["physical_pass"])
        ),
        None,
    )
    operating = next(
        (row for row in ordered if int(row["trigger_budget"]) == recommended_budget),
        None,
    )
    semantic_curve = ", ".join(
        f"B{row['trigger_budget']} {row['all_event_part_exact_mm']:.4f} mm"
        for row in ordered
    )
    penetration_curve = ", ".join(
        f"B{row['trigger_budget']} {row['shoulder_box_penetration_depth_mm']:.4f} mm"
        for row in frame_ordered
    )
    first_to_second = []
    for budget in BUDGETS:
        iterations = sorted(
            [row for row in frame34_trace if int(row["trigger_budget"]) == budget],
            key=lambda row: int(row["iteration"]),
        )
        if len(iterations) >= 2:
            first_to_second.append(
                f"B{budget} {iterations[0]['shoulder_box_signed_distance_mm']:+.4f}→"
                f"{iterations[-1]['shoulder_box_signed_distance_mm']:+.4f} mm"
            )
    marginal = []
    for previous, current in zip(ordered, ordered[1:]):
        marginal.append(
            f"B{previous['trigger_budget']}→B{current['trigger_budget']}: "
            f"Part {_improvement(current['all_event_part_exact_mm'], previous['all_event_part_exact_mm']):+.3f}%, "
            f"SQP {int(current['actual_sqp']) - int(previous['actual_sqp']):+d}, "
            f"wall {float(current['wall_time_s']) - float(previous['wall_time_s']):+.3f}s"
        )
    if operating is None:
        operating_text = "No budget satisfies the registered physical, Ordinary, Global, and semantic criteria."
    else:
        operating_text = (
            f"B{recommended_budget}: All-event Part {operating['all_event_part_exact_mm']:.4f} mm; "
            f"Phase Part {operating['phase_part_exact_mm']:.4f} mm; "
            f"Interaction Part {operating['interaction_part_exact_mm']:.4f} mm; "
            f"Global {operating['global_kf_exact_mm']:.4f} mm "
            f"({operating['global_change_vs_u2_pct']:+.3f}% vs U2); "
            f"Ordinary {operating['ordinary_mm']:.4f} mm "
            f"({operating['ordinary_change_vs_u2_pct']:+.3f}% vs U2); "
            f"Actual SQP {operating['actual_sqp']}; wall {operating['wall_time_s']:.3f}s."
        )
    lines = [
        "# Full-Event Semantic Budget Curve Report",
        "",
        "The only experimental variable is the maximum number of sequential SQP iterations at the seven nonzero FullEvent semantic trigger frames: 2/4/6/8/10. Frame 0 remains 50, every ordinary frame remains 2, semantic JSON/body parts and all collision settings are unchanged, and no fallback or trajectory replacement is used.",
        "",
        "## Decisions",
        "",
        f"- **FullEvent: {full_event_status}**",
        f"- **Semantic Budget: {'Keep' if budget_keep else 'Drop'}**",
        f"- **Operating point: {'B' + str(recommended_budget) if recommended_budget else 'none'}**",
        "",
        "## Answers",
        "",
        f"1. **Semantic precision curve:** {semantic_curve}. Lower is better. Full per-group and ±1/±3 metrics are in `full_event_budget_curve_summary.csv`.",
        f"   B2 improves All-event/Phase/Interaction Part over Legacy by {ordered[0]['all_event_part_improvement_vs_legacy_pct']:+.3f}%/{ordered[0]['phase_part_improvement_vs_legacy_pct']:+.3f}%/{ordered[0]['interaction_part_improvement_vs_legacy_pct']:+.3f}%. B10 improves them by {ordered[-1]['all_event_part_improvement_vs_legacy_pct']:+.3f}%/{ordered[-1]['phase_part_improvement_vs_legacy_pct']:+.3f}%/{ordered[-1]['interaction_part_improvement_vs_legacy_pct']:+.3f}%.",
        f"2. **Frame 34 penetration curve:** {penetration_curve}. The unchanged official failure threshold is 10 mm. Signed distance across that frame's two accepted iterates is " + "; ".join(first_to_second) + ". Positive means separated; negative means penetrating.",
        f"3. **First physical pass:** {'B' + str(first_physical) if first_physical is not None else 'none through B10'}.",
        f"4. **Selected operating point metrics:** {operating_text}",
        "5. **Marginal value of higher budgets:** " + "; ".join(marginal) + ".",
        f"6. **Best FullEvent operating point:** {'B' + str(recommended_budget) if recommended_budget else 'none'}; selection uses the smallest budget satisfying physical feasibility, Ordinary <= U2+1%, Global <= U2+0.5%, and positive All-event/Phase semantic improvement over the registered baselines.",
        f"7. **Final conclusion:** FullEvent={full_event_status}; Semantic Budget={'Keep' if budget_keep else 'Drop'}.",
        "",
        "## Curve table",
        "",
        "| Budget | All-event Part mm | Phase Part mm | Interaction Part mm | Global mm | Ordinary mm | Actual SQP | Wall s | Frame34 depth mm | Physical |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
        *[
            f"| B{row['trigger_budget']} | {row['all_event_part_exact_mm']:.4f} | {row['phase_part_exact_mm']:.4f} | {row['interaction_part_exact_mm']:.4f} | {row['global_kf_exact_mm']:.4f} | {row['ordinary_mm']:.4f} | {row['actual_sqp']} | {row['wall_time_s']:.3f} | {row['official_penetration_max_depth_mm']:.4f} | {'Pass' if row['physical_pass'] else 'Fail'} |"
            for row in ordered
        ],
        "",
        "## Integrity checks",
        "",
        f"- Legacy zero drift: `{compatibility['legacy_zero_drift']}`; qpos max abs diff {compatibility['qpos_max_abs_diff']:.3g}; current/historical SQP {compatibility['current_actual_sqp']}/{compatibility['historical_actual_sqp']}.",
        "- Physical feasibility uses the unchanged official 10 mm penetration tolerance, official foot-skating/contact metrics, and the recorded nonlinear illegal-penetration/joint-limit/self-collision audits.",
        "- Frame 34 is an ordinary frame, so it always has max_iter=2. Budget affects it causally through the sequential state produced at earlier semantic trigger frames, especially contact@30.",
        "- Diagnostics run only after accepted iterates and their time is excluded from retarget wall time; they do not modify the solver state, objective, collision formulation, step size, or trust region.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(config: BenchmarkConfig) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    semantic_hash_before = _hash(config.semantic_keyframe_path)
    events = load_semantic_plan_projection(config.semantic_keyframe_path)
    semantic_frames = {event.trigger_frame for event in events if event.trigger_frame > 0}
    print("Projected full-event support:", [(event.name, event.trigger_frame, event.body_parts) for event in events])

    historical_dirs = {
        "original": config.historical_final_dir / "runs" / "compat_original",
        "uniform_2": config.historical_final_dir / "runs" / "compat_uniform_2",
        "legacy_historical": config.historical_final_dir / "runs" / "edge_part_uniform",
    }
    for key, run_dir in historical_dirs.items():
        if not _trajectory(run_dir, config.task_name).exists() or not (run_dir / "profile.json").exists():
            raise FileNotFoundError(f"missing historical cache {key}: {run_dir}")

    existing_b2_path = _trajectory(
        config.output_dir / "runs" / CURVE_SPECS[0].key,
        config.task_name,
    )
    pre_curve_b2_qpos = (
        _trajectory_qpos(existing_b2_path.parent, config.task_name)
        if existing_b2_path.exists()
        else None
    )

    # The full curve is mandatory even when an intermediate budget is
    # physically infeasible. Only a solver crash or invalid numerical result
    # may stop it.
    run_dirs = {
        "original": historical_dirs["original"],
        "uniform_2": historical_dirs["uniform_2"],
        "legacy_historical": historical_dirs["legacy_historical"],
        LEGACY_SPEC.key: _run(config, LEGACY_SPEC),
    }
    for spec in CURVE_SPECS:
        run_dirs[spec.key] = _run(config, spec)
    evaluator = _official_evaluator()
    specs_by_key = {
        "original": RunSpec("original", "Original", "original", 10),
        "uniform_2": RunSpec("uniform_2", "Uniform-2", "uniform", 2),
        "legacy_historical": RunSpec(
            "legacy_historical",
            "Legacy historical",
            "uniform2_semantic_weight_uniform",
            2,
        ),
        LEGACY_SPEC.key: LEGACY_SPEC,
        **{spec.key: spec for spec in CURVE_SPECS},
        **{spec.key: spec for spec in RANDOM_SPECS},
        ORIGINAL_B10_SPEC.key: ORIGINAL_B10_SPEC,
    }
    evaluations: dict[str, dict[str, PrecisionEvaluation]] = {}
    metric_rows: dict[str, dict[str, Any]] = {}

    def evaluate_run(key: str) -> None:
        run_dir = run_dirs[key]
        evaluations[key] = _evaluations(_trajectory(run_dir, config.task_name), events)
        metric_rows[key] = _metric_row(
            specs_by_key[key],
            run_dir,
            evaluations[key],
            _official(config, run_dir, evaluator),
            semantic_frames,
        )

    for key in (
        "original",
        "uniform_2",
        "legacy_historical",
        "legacy_4event",
        *(spec.key for spec in CURVE_SPECS),
    ):
        evaluate_run(key)

    compatibility = _legacy_compatibility(
        config,
        run_dirs["legacy_4event"],
        run_dirs["legacy_historical"],
        metric_rows["legacy_4event"],
        metric_rows["legacy_historical"],
    )
    compatibility.update(
        {
            "semantic_v2_sha256_before": semantic_hash_before,
            "semantic_v2_sha256_after_stage_a": _hash(config.semantic_keyframe_path),
            "semantic_v2_unchanged": semantic_hash_before == _hash(config.semantic_keyframe_path),
        }
    )
    _write_csv(config.output_dir / "legacy_4event_backward_compatibility.csv", [compatibility])

    annotated_stage_a = _annotate(
        [metric_rows[key] for key in ("original", "uniform_2", "legacy_4event", "full_event_b2")],
        metric_rows["uniform_2"],
        metric_rows["legacy_4event"],
        metric_rows["full_event_b2"],
    )
    stage_a_by_key = {str(row["run_key"]): row for row in annotated_stage_a}
    metric_rows.update(stage_a_by_key)
    full_event_rows = _full_event_rows(events, evaluations, run_dirs)
    _write_csv(config.output_dir / "full_event_main_summary.csv", annotated_stage_a)
    _write_csv(config.output_dir / "full_event_event_metrics.csv", full_event_rows)
    _stage_a_plot(config.output_dir, full_event_rows)

    all_keys = ("original", "uniform_2", "legacy_4event", *(spec.key for spec in CURVE_SPECS))
    annotated = _annotate(
        [metric_rows[key] for key in all_keys],
        metric_rows["uniform_2"],
        metric_rows["legacy_4event"],
        metric_rows["full_event_b2"],
    )
    by_key = {str(row["run_key"]): row for row in annotated}
    metric_rows.update(by_key)
    curve_rows = [dict(by_key[spec.key]) for spec in CURVE_SPECS]

    curve_event_rows = _curve_event_rows(events, evaluations, run_dirs)
    trace_rows = _iteration_trace_rows(run_dirs)
    human_joints, _ = load_intermimic_data(
        str(config.data_path / f"{config.task_name}.pt")
    )
    contact_sequences = extract_foot_sticking_sequence_velocity(
        human_joints,
        evaluator.demo_joints,
        ["L_Toe", "R_Toe"],
    )
    u2_detail, _ = _physical_detail(
        config,
        specs_by_key["uniform_2"],
        run_dirs["uniform_2"],
        evaluator,
        contact_sequences,
        None,
    )
    physical_details: dict[str, dict[str, Any]] = {}
    all_penetration_rows: list[dict[str, Any]] = []
    original_wall = float(metric_rows["original"]["wall_time_s"])
    for row, spec in zip(curve_rows, CURVE_SPECS):
        detail, penetration_rows = _physical_detail(
            config,
            spec,
            run_dirs[spec.key],
            evaluator,
            contact_sequences,
            u2_detail,
        )
        physical_details[spec.key] = detail
        all_penetration_rows.extend(penetration_rows)
        physical_pass = _registered_physical_pass(detail, u2_detail)
        row.update(
            {
                "speedup_vs_original": original_wall / float(row["wall_time_s"]),
                "official_penetration_duration": detail["official_penetration_duration"],
                "official_penetration_max_depth_mm": detail["official_penetration_max_depth_mm"],
                "illegal_penetration_duration": detail["illegal_penetration_duration"],
                "illegal_penetration_max_depth_mm": detail["illegal_penetration_max_depth_mm"],
                "joint_limit_max_violation": detail["joint_limit_max_violation"],
                "self_collision_max_violation": detail["self_collision_max_violation"],
                "physical_pass": physical_pass,
                "registered_guardrail_pass": bool(
                    physical_pass
                    and float(row["ordinary_change_vs_u2_pct"]) <= 1.0
                    and float(row["global_change_vs_u2_pct"]) <= 0.5
                    and float(row["all_event_part_improvement_vs_u2_pct"]) > 0.0
                    and float(row["all_event_part_improvement_vs_legacy_pct"]) > 0.0
                    and float(row["phase_part_improvement_vs_legacy_pct"]) > 0.0
                ),
            }
        )

    frame34_rows = _frame34_rows(config, run_dirs, curve_rows, trace_rows)
    frame34_trace = [row for row in trace_rows if int(row["frame"]) == 34]
    physical_rows = [_physical_csv_row(physical_details[spec.key]) for spec in CURVE_SPECS]
    compute_rows = [
        {
            "run_key": row["run_key"],
            "trigger_budget": row["trigger_budget"],
            "configured_sqp_budget": row["configured_sqp_budget"],
            "actual_sqp": row["actual_sqp"],
            "solver_calls": row["solver_calls"],
            "wall_time_s": row["wall_time_s"],
            "optimization_time_s": row["optimization_time_s"],
            "speedup_vs_original": row["speedup_vs_original"],
        }
        for row in curve_rows
    ]
    convergence_rows = []
    frame34_by_budget = {int(row["trigger_budget"]): row for row in frame34_rows}
    for row in curve_rows:
        merged = dict(row)
        merged.update(
            {
                f"frame34_{key}": value
                for key, value in frame34_by_budget[int(row["trigger_budget"])].items()
                if key not in {"run_key", "trigger_budget", "qpos_json"}
            }
        )
        convergence_rows.append(merged)

    _write_csv(config.output_dir / "full_event_budget_curve_summary.csv", curve_rows)
    _write_csv(config.output_dir / "full_event_budget_event_metrics.csv", curve_event_rows)
    _write_csv(config.output_dir / "full_event_budget_physical_metrics.csv", physical_rows)
    if all_penetration_rows:
        _write_csv(
            config.output_dir / "full_event_budget_penetration_frames.csv",
            all_penetration_rows,
        )
    _write_csv(config.output_dir / "frame34_budget_diagnostics.csv", frame34_rows)
    _write_csv(config.output_dir / "frame34_iteration_trace.csv", frame34_trace)
    _write_csv(config.output_dir / "full_event_budget_compute.csv", compute_rows)
    _write_csv(
        config.output_dir / "full_event_budget_physical_convergence_curve.csv",
        convergence_rows,
    )
    # Backward-compatible aliases point to the same completed deterministic curve.
    _write_csv(config.output_dir / "semantic_budget_curve_summary.csv", curve_rows)
    _write_csv(config.output_dir / "semantic_budget_curve_event_metrics.csv", curve_event_rows)
    _write_csv(config.output_dir / "semantic_iteration_trace.csv", trace_rows)

    recommended_budget = _recommended_budget(curve_rows)
    if recommended_budget:
        full_event_status = "Keep"
    elif not bool(curve_rows[-1]["physical_pass"]):
        full_event_status = "Need collision-formulation follow-up"
    else:
        full_event_status = "Drop"
    budget_keep = recommended_budget > 2

    _budget_curve_plots(
        config.output_dir,
        curve_rows,
        frame34_rows,
        frame34_trace,
    )
    report_path = config.output_dir / "full_event_budget_curve_report.md"
    _curve_report(
        report_path,
        compatibility,
        curve_rows,
        frame34_rows,
        frame34_trace,
        recommended_budget,
        full_event_status,
        budget_keep,
    )
    (config.output_dir / "full_event_semantic_budget_report.md").write_text(
        report_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    final_hash = _hash(config.semantic_keyframe_path)
    metadata = {
        "task": config.task_name,
        "semantic_keyframe_path": str(config.semantic_keyframe_path),
        "semantic_v2_sha256_before": semantic_hash_before,
        "semantic_v2_sha256_after": final_hash,
        "semantic_v2_unchanged": semantic_hash_before == final_hash,
        "budgets": list(BUDGETS),
        "random_budgets_run": [],
        "original_objective_b10_run": False,
        "interaction_events": list(INTERACTION_EVENTS),
        "phase_events": list(PHASE_EVENTS),
        "eligible_nonzero_events": [event.name for event in events if event.trigger_frame > 0],
        "criticality_used": False,
        "full_event_weight": "all deterministic projected events and JSON body_parts",
        "legacy_weight": "historical contact/lift/place/release only",
        "ordinary_budget": 2,
        "full_event_status": full_event_status,
        "semantic_budget_keep": budget_keep,
        "recommended_budget": recommended_budget,
        "budget_curve_run": True,
        "frame34_is_ordinary_budget_2": True,
        "official_penetration_threshold_m": float(evaluator.penetration_tolerance),
        "b2_pre_curve_qpos_max_abs_diff": (
            float(
                np.max(
                    np.abs(
                        _trajectory_qpos(run_dirs["full_event_b2"], config.task_name)
                        - pre_curve_b2_qpos
                    )
                )
            )
            if pre_curve_b2_qpos is not None
            else None
        ),
    }
    (config.output_dir / "benchmark_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"\nFullEvent: {full_event_status}; "
        f"Semantic Budget: {'Keep' if budget_keep else 'Drop'}; "
        f"recommended {'B' + str(recommended_budget) if recommended_budget else 'none'}"
    )
    print(f"Artifacts: {config.output_dir}")


def cli() -> None:
    run(tyro.cli(BenchmarkConfig))


if __name__ == "__main__":
    cli()
