"""Registered exact-trigger Semantic Budget causal benchmark.

This experiment changes only the temporal location of a small 2 -> 4 solver
budget on top of the immutable Uniform-2 + Legacy-weight objective.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

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
    load_semantic_events,
    load_semantic_plan_projection,
)


MM = 1000.0
RANDOM_SEEDS = tuple(range(5))


@dataclass(frozen=True)
class RunSpec:
    key: str
    label: str
    mode: str
    seed: int = 0


@dataclass
class BenchmarkConfig:
    """Fixed OMOMO/sub3_largebox_003 budget experiment."""

    task_name: str = "sub3_largebox_003"
    data_path: Path = Path("demo_data/OMOMO_new")
    semantic_keyframe_path: Path = Path(
        "demo_data/semantic_keyframes/sub3_largebox_003_semantic_v2.json"
    )
    output_dir: Path = Path("benchmark_results_semantic_budget")
    historical_final_dir: Path = Path("benchmark_results_semantic_final")
    force: bool = False
    aggregate_only: bool = False
    refresh_official: bool = False


NEW_SPECS = (
    RunSpec("legacy_anchor", "Legacy", "uniform2_semantic_weight_uniform"),
    RunSpec(
        "original_objective_semantic_budget",
        "Original-Budget",
        "uniform2_original_objective_semantic_budget",
    ),
    RunSpec(
        "semantic_budget",
        "Semantic-Budget",
        "uniform2_semantic_weight_semantic_budget",
    ),
    *(
        RunSpec(
            f"random_budget_seed_{seed}",
            f"Random-Budget seed={seed}",
            "uniform2_semantic_weight_random_budget",
            seed,
        )
        for seed in RANDOM_SEEDS
    ),
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


def _run(config: BenchmarkConfig, spec: RunSpec) -> Path:
    run_dir = config.output_dir / "runs" / spec.key
    trajectory = _trajectory(run_dir, config.task_name)
    if trajectory.exists() and (run_dir / "profile.json").exists() and not config.force:
        print(f"Reusing completed Semantic Budget run: {spec.key}")
        return run_dir
    if config.aggregate_only:
        raise FileNotFoundError(f"missing completed run for --aggregate-only: {run_dir}")
    semantic = SemanticRetargetingConfig(
        mode=spec.mode,  # type: ignore[arg-type]
        semantic_keyframe_path=config.semantic_keyframe_path,
        profile_dir=run_dir,
        random_seed=spec.seed,
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
    print(f"\n=== {spec.label} ({spec.mode}) ===")
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


def _profile(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads((run_dir / "profile.json").read_text(encoding="utf-8"))
    return payload.get("metadata", {}), payload["summary"], payload["frames"]


def _metric_row(
    spec: RunSpec,
    run_dir: Path,
    primary: PrecisionEvaluation,
    official: dict[str, float],
    semantic_frames: set[int],
) -> dict[str, Any]:
    _, summary, frames = _profile(run_dir)
    actual = np.asarray([int(row["actual_sqp_iterations"]) for row in frames], dtype=np.int64)
    calls = np.asarray(
        [int(row.get("convex_solver_calls", row["actual_sqp_iterations"])) for row in frames],
        dtype=np.int64,
    )
    configured = np.asarray(
        [int(row.get("configured_budget", row.get("base_budget", row["actual_sqp_iterations"]))) for row in frames],
        dtype=np.int64,
    )
    semantic_index = np.asarray(sorted(semantic_frames), dtype=np.int64)
    ordinary_index = np.asarray(
        [frame for frame in range(1, len(frames)) if frame not in semantic_frames],
        dtype=np.int64,
    )
    edge = primary.semantic_edge or {}
    return {
        "method": spec.label,
        "run_key": spec.key,
        "mode": spec.mode,
        "seed": spec.seed if "random_budget" in spec.key else "",
        "configured_sqp_budget": int(configured.sum()),
        "actual_sqp": int(actual.sum()),
        "solver_calls": int(calls.sum()),
        "wall_time_s": float(summary["total_wall_time"]),
        "optimization_time_s": float(summary["optimization_wall_time"]),
        "mean_sqp_per_frame": float(actual.mean()),
        "p95_sqp_per_frame": float(np.percentile(actual, 95)),
        "semantic_frame_sqp": int(actual[semantic_index].sum()),
        "ordinary_frame_sqp": int(actual[ordinary_index].sum()),
        "frame0_sqp": int(actual[0]),
        "ordinary_mm": MM * primary.ordinary["mean"],
        "global_kf_exact_mm": MM * primary.keyframe_global["exact"],
        "global_kf_pm1_mm": MM * primary.keyframe_global["pm1"],
        "global_kf_pm3_mm": MM * primary.keyframe_global["pm3"],
        "part_exact_mm": MM * primary.semantic_part["exact"],
        "part_pm1_mm": MM * primary.semantic_part["pm1"],
        "part_pm3_mm": MM * primary.semantic_part["pm3"],
        "edge_exact_mm": MM * float(edge.get("exact", np.nan)),
        "local_exact_mm": MM * primary.semantic_local["exact"],
        "penetration_duration": official["penetration_duration"],
        "penetration_max_depth_mm": MM * official["penetration_max_depth_m"],
        "foot_skating_duration": official["foot_skating_duration"],
        "foot_skating_max_velocity": official["foot_skating_max_velocity"],
        "contact_preservation": official["contact_preservation"],
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
    original: dict[str, Any],
    u2: dict[str, Any],
    legacy: dict[str, Any],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        extra_calls = int(row["solver_calls"] - legacy["solver_calls"])
        extra_wall = float(row["wall_time_s"] - legacy["wall_time_s"])
        part_gain = _improvement(row["part_exact_mm"], legacy["part_exact_mm"])
        row.update(
            {
                "ordinary_change_vs_u2_pct": _pct_change(row["ordinary_mm"], u2["ordinary_mm"]),
                "global_change_vs_u2_pct": _pct_change(
                    row["global_kf_exact_mm"], u2["global_kf_exact_mm"]
                ),
                "part_improvement_vs_u2_pct": _improvement(row["part_exact_mm"], u2["part_exact_mm"]),
                "ordinary_change_vs_legacy_pct": _pct_change(
                    row["ordinary_mm"], legacy["ordinary_mm"]
                ),
                "global_change_vs_legacy_pct": _pct_change(
                    row["global_kf_exact_mm"], legacy["global_kf_exact_mm"]
                ),
                "part_improvement_vs_legacy_pct": part_gain,
                "part_pm1_improvement_vs_legacy_pct": _improvement(
                    row["part_pm1_mm"], legacy["part_pm1_mm"]
                ),
                "edge_improvement_vs_legacy_pct": _improvement(
                    row["edge_exact_mm"], legacy["edge_exact_mm"]
                ),
                "local_improvement_vs_legacy_pct": _improvement(
                    row["local_exact_mm"], legacy["local_exact_mm"]
                ),
                "extra_solver_calls_vs_legacy": extra_calls,
                "extra_wall_time_vs_legacy_s": extra_wall,
                "part_gain_pct_per_extra_sqp": part_gain / extra_calls if extra_calls > 0 else np.nan,
                "part_gain_pct_per_extra_wall_s": part_gain / extra_wall if extra_wall > 0 else np.nan,
                "sqp_reduction_vs_original_pct": _improvement(row["actual_sqp"], original["actual_sqp"]),
                "sqp_reduction_vs_legacy_pct": _improvement(row["actual_sqp"], legacy["actual_sqp"]),
                "speedup_vs_original": original["wall_time_s"] / row["wall_time_s"],
                "speedup_vs_legacy": legacy["wall_time_s"] / row["wall_time_s"],
                "physical_pass_vs_u2": _physical_pass(row, u2),
                "physical_pass_vs_legacy": _physical_pass(row, legacy),
            }
        )
        output.append(row)
    return output


def _aggregate_random(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {
        "method": "Random-Budget mean",
        "run_key": "random_budget_mean",
        "mode": "uniform2_semantic_weight_random_budget",
        "members": "|".join(str(row["run_key"]) for row in rows),
    }
    excluded = {"method", "run_key", "mode", "seed"}
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
            output[field] = float(array.mean())
            output[f"{field}_std"] = float(array.std(ddof=0))
    return output


def _event_map(evaluation: PrecisionEvaluation) -> dict[str, dict[str, Any]]:
    return {str(row["event"]): dict(row) for row in evaluation.event_rows}


def _frame_log_rows(
    specs: Sequence[RunSpec],
    run_dirs: dict[str, Path],
    events: Sequence[SemanticEvent],
) -> list[dict[str, Any]]:
    event_by_frame: dict[int, list[str]] = {}
    for event in events:
        event_by_frame.setdefault(event.trigger_frame, []).append(event.name)
    semantic_frames = {event.trigger_frame for event in events if event.trigger_frame != 0}
    rows: list[dict[str, Any]] = []
    for spec in specs:
        _, _, frames = _profile(run_dirs[spec.key])
        for frame_idx, frame in enumerate(frames):
            is_random = bool(frame.get("is_random_budget_frame", False))
            is_trigger = frame_idx in semantic_frames
            allocation = frame.get("allocation_reason")
            if allocation is None:
                allocation = "frame0_initialization" if frame_idx == 0 else "ordinary_base"
            rows.append(
                {
                    "method": spec.label,
                    "run_key": spec.key,
                    "frame": frame_idx,
                    "event": "|".join(event_by_frame.get(frame_idx, [])),
                    "is_semantic_trigger": is_trigger,
                    "is_random_budget_frame": is_random,
                    "configured_budget": int(
                        frame.get("configured_budget", frame.get("base_budget", frame["actual_sqp_iterations"]))
                    ),
                    "actual_sqp": int(frame["actual_sqp_iterations"]),
                    "solver_calls": int(frame.get("convex_solver_calls", frame["actual_sqp_iterations"])),
                    "allocation_reason": allocation,
                }
            )
    return rows


def _event_rows(
    events: Sequence[SemanticEvent],
    evaluations: dict[str, PrecisionEvaluation],
    metric_rows: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    maps = {key: _event_map(value) for key, value in evaluations.items()}
    random_keys = [f"random_budget_seed_{seed}" for seed in RANDOM_SEEDS]
    rows: list[dict[str, Any]] = []
    for event in events:
        if event.trigger_frame == 0:
            continue
        legacy = maps["legacy_anchor"][event.name]
        semantic = maps["semantic_budget"][event.name]
        original_budget = maps["original_objective_semantic_budget"][event.name]
        random_part = np.asarray([maps[key][event.name]["part_exact"] for key in random_keys], dtype=np.float64)
        random_global = np.asarray([maps[key][event.name]["global_exact"] for key in random_keys], dtype=np.float64)
        random_local = np.asarray([maps[key][event.name]["local_exact"] for key in random_keys], dtype=np.float64)
        legacy_profile = _profile(metric_rows["_run_dirs"]["legacy_anchor"])[2]
        semantic_profile = _profile(metric_rows["_run_dirs"]["semantic_budget"])[2]
        rows.append(
            {
                "event": event.name,
                "frame": event.trigger_frame,
                "body_parts": "|".join(event.body_parts),
                "legacy_part_exact_mm": MM * legacy["part_exact"],
                "semantic_budget_part_exact_mm": MM * semantic["part_exact"],
                "part_improvement_vs_legacy_pct": _improvement(semantic["part_exact"], legacy["part_exact"]),
                "random_part_exact_mean_mm": MM * float(random_part.mean()),
                "random_part_exact_std_mm": MM * float(random_part.std(ddof=0)),
                "semantic_part_improvement_vs_random_mean_pct": _improvement(
                    semantic["part_exact"], float(random_part.mean())
                ),
                "original_budget_part_exact_mm": MM * original_budget["part_exact"],
                "legacy_global_exact_mm": MM * legacy["global_exact"],
                "semantic_budget_global_exact_mm": MM * semantic["global_exact"],
                "global_change_vs_legacy_pct": _pct_change(
                    semantic["global_exact"], legacy["global_exact"]
                ),
                "random_global_exact_mean_mm": MM * float(random_global.mean()),
                "legacy_local_exact_mm": MM * legacy["local_exact"],
                "semantic_budget_local_exact_mm": MM * semantic["local_exact"],
                "random_local_exact_mean_mm": MM * float(random_local.mean()),
                "legacy_actual_sqp": int(legacy_profile[event.trigger_frame]["actual_sqp_iterations"]),
                "semantic_budget_actual_sqp": int(
                    semantic_profile[event.trigger_frame]["actual_sqp_iterations"]
                ),
            }
        )
    return rows


def _compatibility_row(
    config: BenchmarkConfig,
    current_dir: Path,
    historical_dir: Path,
    current: dict[str, Any],
    historical: dict[str, Any],
    semantic_hash_before: str,
    semantic_hash_after: str,
) -> dict[str, Any]:
    with np.load(_trajectory(current_dir, config.task_name), allow_pickle=False) as new_npz, np.load(
        _trajectory(historical_dir, config.task_name), allow_pickle=False
    ) as old_npz:
        qpos_delta = float(np.max(np.abs(
            np.asarray(new_npz["qpos"], dtype=np.float64)
            - np.asarray(old_npz["qpos"], dtype=np.float64)
        )))
        new_sqp = int(np.asarray(new_npz["actual_sqp_iterations"], dtype=np.int64).sum())
        old_sqp = int(np.asarray(old_npz["actual_sqp_iterations"], dtype=np.int64).sum())
    metric_fields = (
        "ordinary_mm",
        "global_kf_exact_mm",
        "part_exact_mm",
        "edge_exact_mm",
        "local_exact_mm",
    )
    metric_delta = max(abs(float(current[field]) - float(historical[field])) for field in metric_fields)
    return {
        "mode": "uniform2_semantic_weight_uniform",
        "historical_run": str(historical_dir),
        "current_run": str(current_dir),
        "qpos_max_abs_delta": qpos_delta,
        "metric_max_abs_delta_mm": metric_delta,
        "current_actual_sqp": new_sqp,
        "historical_actual_sqp": old_sqp,
        "actual_sqp_delta": new_sqp - old_sqp,
        "semantic_v2_sha256_before": semantic_hash_before,
        "semantic_v2_sha256_after": semantic_hash_after,
        "semantic_v2_unchanged": semantic_hash_before == semantic_hash_after,
        "zero_drift": qpos_delta == 0.0 and metric_delta == 0.0 and new_sqp == old_sqp,
    }


def _plots(
    output_dir: Path,
    main_rows: Sequence[dict[str, Any]],
    frame_rows: Sequence[dict[str, Any]],
    event_rows: Sequence[dict[str, Any]],
) -> None:
    try:
        import matplotlib  # noqa: PLC0415

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for Semantic Budget plots") from exc

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    colors = {
        "Original": "tab:gray",
        "Uniform-2": "tab:blue",
        "Legacy": "tab:green",
        "Original-Budget": "tab:orange",
        "Semantic-Budget": "tab:red",
        "Random-Budget mean": "tab:purple",
    }

    fig, ax = plt.subplots(figsize=(9, 5))
    for row in main_rows:
        ax.scatter(row["actual_sqp"], row["part_exact_mm"], s=55, color=colors[row["method"]])
        ax.annotate(row["method"], (row["actual_sqp"], row["part_exact_mm"]), xytext=(5, 4), textcoords="offset points")
    ax.set(xlabel="Actual SQP", ylabel="Part Exact (mm, lower is better)", title="Compute vs semantic precision")
    fig.tight_layout()
    fig.savefig(plot_dir / "actual_sqp_vs_part_exact.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for row in main_rows:
        ax.scatter(row["global_kf_exact_mm"], row["part_exact_mm"], s=55, color=colors[row["method"]])
        ax.annotate(row["method"], (row["global_kf_exact_mm"], row["part_exact_mm"]), xytext=(5, 4), textcoords="offset points")
    ax.set(xlabel="Global KF Exact (mm, lower is better)", ylabel="Part Exact (mm, lower is better)", title="Global vs semantic precision")
    fig.tight_layout()
    fig.savefig(plot_dir / "global_kf_vs_part_exact.png", dpi=180)
    plt.close(fig)

    by_run: dict[str, list[dict[str, Any]]] = {}
    for row in frame_rows:
        by_run.setdefault(str(row["run_key"]), []).append(dict(row))
    fig, ax = plt.subplots(figsize=(12, 5))
    for key, label, style in (
        ("legacy_anchor", "Legacy actual", "-"),
        ("semantic_budget", "Semantic Budget actual", "-"),
    ):
        values = sorted(by_run[key], key=lambda row: int(row["frame"]))
        ax.step([row["frame"] for row in values], [row["actual_sqp"] for row in values], where="mid", label=label, linestyle=style)
    semantic_values = sorted(by_run["semantic_budget"], key=lambda row: int(row["frame"]))
    ax.step([row["frame"] for row in semantic_values], [row["configured_budget"] for row in semantic_values], where="mid", label="Semantic Budget configured", linestyle="--")
    for row in event_rows:
        ax.axvline(row["frame"], color="tab:red", alpha=0.18, linewidth=1)
        ax.text(row["frame"], 0.97, row["event"], rotation=90, transform=ax.get_xaxis_transform(), va="top", fontsize=8)
    ax.set(xlabel="Frame", ylabel="SQP iterations", title="Legacy and exact-trigger Semantic Budget")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "semantic_budget_vs_frame.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    semantic_locations = [row["frame"] for row in event_rows]
    ax.scatter(semantic_locations, np.zeros(len(semantic_locations)), marker="D", s=55, label="Semantic triggers")
    for seed in RANDOM_SEEDS:
        key = f"random_budget_seed_{seed}"
        locations = [row["frame"] for row in by_run[key] if row["is_random_budget_frame"]]
        ax.scatter(locations, np.full(len(locations), seed + 1), marker="|", s=150, label=f"Random seed {seed}")
    ax.set_yticks(range(6))
    ax.set_yticklabels(["Semantic", *[f"Random {seed}" for seed in RANDOM_SEEDS]])
    ax.set(xlabel="Frame", ylabel="Allocation", title="Semantic vs compute-matched random budget locations")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(plot_dir / "semantic_vs_random_budget_locations.png", dpi=180)
    plt.close(fig)


def _report(
    path: Path,
    rows: dict[str, dict[str, Any]],
    random_mean: dict[str, Any],
    compatibility: dict[str, Any],
    event_rows: Sequence[dict[str, Any]],
    decision: str,
) -> None:
    legacy = rows["legacy_anchor"]
    semantic = rows["semantic_budget"]
    original_budget = rows["original_objective_semantic_budget"]
    best = max(event_rows, key=lambda row: float(row["part_improvement_vs_legacy_pct"]))
    no_gain = [row["event"] for row in event_rows if float(row["part_improvement_vs_legacy_pct"]) <= 0]
    majority = sum(
        float(row["semantic_budget_part_exact_mm"]) < float(row["random_part_exact_mean_mm"])
        for row in event_rows
    )
    lines = [
        "# Semantic Budget Report",
        "",
        "This run tests only exact-trigger 2→4 refinement on top of the registered Uniform-2 baseline. Ordinary frames remain configured at two iterations, and all four iterations at a semantic trigger use one unchanged objective.",
        "",
        "## Decision",
        "",
        f"**Semantic Budget: {decision}.**",
        "",
        "## Q1–Q10",
        "",
        f"1. **Legacy compatibility:** zero drift = `{compatibility['zero_drift']}`; max qpos delta {compatibility['qpos_max_abs_delta']:.3g}, metric delta {compatibility['metric_max_abs_delta_mm']:.3g} mm, SQP {compatibility['current_actual_sqp']}.",
        f"2. **Added compute:** configured {semantic['configured_sqp_budget'] - legacy['configured_sqp_budget']:+.0f} SQP, actual {semantic['extra_solver_calls_vs_legacy']:+.0f} calls, wall {semantic['extra_wall_time_vs_legacy_s']:+.3f} s.",
        f"3. **Semantic quality vs Legacy:** Part Exact {semantic['part_improvement_vs_legacy_pct']:+.3f}%, Part ±1 {semantic['part_pm1_improvement_vs_legacy_pct']:+.3f}%, Local {semantic['local_improvement_vs_legacy_pct']:+.3f}%, Edge {semantic['edge_improvement_vs_legacy_pct']:+.3f}% (positive means improvement).",
        f"4. **Guardrails:** Global vs U2 {semantic['global_change_vs_u2_pct']:+.3f}%, Ordinary vs Legacy {semantic['ordinary_change_vs_legacy_pct']:+.3f}%, physical pass vs Legacy = `{semantic['physical_pass_vs_legacy']}`.",
        f"5. **Event-wise:** largest Part benefit is `{best['event']}` ({best['part_improvement_vs_legacy_pct']:+.3f}%); non-benefiting events: {', '.join(no_gain) if no_gain else 'none'}.",
        f"6. **Random control:** Semantic Part Exact {semantic['part_exact_mm']:.4f} mm vs random mean {random_mean['part_exact_mm']:.4f}±{random_mean['part_exact_mm_std']:.4f} mm; it beats the random mean on {majority}/{len(event_rows)} events.",
        f"7. **Original-Budget control:** Part gain vs U2 {original_budget['part_improvement_vs_u2_pct']:+.3f}% and vs Legacy {original_budget['part_improvement_vs_legacy_pct']:+.3f}%; compare Semantic-Budget {semantic['part_improvement_vs_legacy_pct']:+.3f}% vs Legacy.",
        f"8. **Complementarity:** extra semantic refinement {'shows' if semantic['part_improvement_vs_legacy_pct'] > 0 else 'does not show'} additional gain beyond Legacy weighting on this sequence.",
        f"9. **Efficiency:** {semantic['part_gain_pct_per_extra_sqp']:.6f} Part-improvement percentage points per extra actual SQP and {semantic['part_gain_pct_per_extra_wall_s']:.6f} per extra wall-second.",
        f"10. **Final judgment:** `{decision}` under the registered ordinary/global/physical, ≥1% Part, and random-control criteria.",
        "",
        "## Eligibility and causality",
        "",
        "All non-frame0 triggers from the deterministic projection are eligible: approach, contact, lift, carry_mid, arrive, place, and release. Random candidates exclude frame 0 and every semantic trigger ±3. No VLM call, criticality, Edge objective, adaptive compute, or later-stage optimizer is used.",
        "",
        "The immutable Legacy anchor retains its historical contact/lift/place/release residual support so the required zero-drift test is meaningful. The complete projected event list controls budget eligibility; therefore objective and extra-compute timing are not conflated.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(config: BenchmarkConfig) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    semantic_hash_before = _hash(config.semantic_keyframe_path)
    projected_events = load_semantic_plan_projection(config.semantic_keyframe_path)
    evaluator_events = load_semantic_events(config.semantic_keyframe_path)
    eligible = [event for event in projected_events if event.trigger_frame != 0]
    semantic_frames = {event.trigger_frame for event in eligible}
    print(
        "Eligible semantic triggers:",
        [(event.name, event.trigger_frame, event.body_parts) for event in eligible],
    )

    run_dirs = {spec.key: _run(config, spec) for spec in NEW_SPECS}
    historical_runs = {
        "original": config.historical_final_dir / "runs" / "compat_original",
        "uniform_2": config.historical_final_dir / "runs" / "compat_uniform_2",
        "legacy_historical": config.historical_final_dir / "runs" / "edge_part_uniform",
    }
    for key, run_dir in historical_runs.items():
        if not _trajectory(run_dir, config.task_name).exists() or not (run_dir / "profile.json").exists():
            raise FileNotFoundError(f"missing registered historical cache {key}: {run_dir}")

    all_specs = (
        RunSpec("original", "Original", "original"),
        RunSpec("uniform_2", "Uniform-2", "uniform"),
        *NEW_SPECS,
    )
    run_dirs.update({"original": historical_runs["original"], "uniform_2": historical_runs["uniform_2"]})
    official_evaluator = _official_evaluator()
    metric_rows: dict[str, dict[str, Any]] = {}
    all_event_evaluations: dict[str, PrecisionEvaluation] = {}
    for spec in all_specs:
        run_dir = run_dirs[spec.key]
        payload = load_precision_payload(_trajectory(run_dir, config.task_name))
        primary = evaluate_precision(payload, evaluator_events)
        all_event_evaluations[spec.key] = evaluate_precision(
            payload,
            projected_events,
            critical_event_names=None,
        )
        metric_rows[spec.key] = _metric_row(
            spec,
            run_dir,
            primary,
            _official(config, run_dir, official_evaluator),
            semantic_frames,
        )

    # Evaluate the immutable old c=1 cache with precisely the same evaluator.
    old_payload = load_precision_payload(_trajectory(historical_runs["legacy_historical"], config.task_name))
    old_primary = evaluate_precision(old_payload, evaluator_events)
    old_row = _metric_row(
        RunSpec("legacy_historical", "Legacy historical", "uniform2_semantic_weight_uniform"),
        historical_runs["legacy_historical"],
        old_primary,
        _official(config, historical_runs["legacy_historical"], official_evaluator),
        semantic_frames,
    )

    original = metric_rows["original"]
    u2 = metric_rows["uniform_2"]
    legacy = metric_rows["legacy_anchor"]
    annotated = _annotate(list(metric_rows.values()), original, u2, legacy)
    annotated_by_key = {str(row["run_key"]): row for row in annotated}
    random_rows = [annotated_by_key[f"random_budget_seed_{seed}"] for seed in RANDOM_SEEDS]
    random_mean = _aggregate_random(random_rows)

    main_keys = ("original", "uniform_2", "legacy_anchor", "original_objective_semantic_budget", "semantic_budget")
    main_rows = [annotated_by_key[key] for key in main_keys] + [random_mean]
    _write_csv(config.output_dir / "semantic_budget_main_summary.csv", main_rows)
    _write_csv(config.output_dir / "semantic_budget_random_aggregate.csv", [random_mean, *random_rows])
    _write_csv(config.output_dir / "semantic_budget_compute.csv", annotated)

    frame_specs = (
        RunSpec("original", "Original", "original"),
        RunSpec("uniform_2", "Uniform-2", "uniform"),
        *NEW_SPECS,
    )
    frame_rows = _frame_log_rows(frame_specs, run_dirs, projected_events)
    _write_csv(config.output_dir / "semantic_budget_frame_log.csv", frame_rows)

    event_context: dict[str, Any] = {"_run_dirs": run_dirs}
    event_rows = _event_rows(projected_events, all_event_evaluations, event_context)
    _write_csv(config.output_dir / "semantic_budget_event_metrics.csv", event_rows)

    semantic_hash_after = _hash(config.semantic_keyframe_path)
    compatibility = _compatibility_row(
        config,
        run_dirs["legacy_anchor"],
        historical_runs["legacy_historical"],
        metric_rows["legacy_anchor"],
        old_row,
        semantic_hash_before,
        semantic_hash_after,
    )
    compatibility.update(
        {
            "eligible_trigger_count": len(eligible),
            "eligible_trigger_frames": "|".join(str(event.trigger_frame) for event in eligible),
            "eligible_trigger_events": "|".join(event.name for event in eligible),
        }
    )
    _write_csv(config.output_dir / "semantic_budget_backward_compatibility.csv", [compatibility])

    semantic = annotated_by_key["semantic_budget"]
    majority = sum(
        float(row["semantic_budget_part_exact_mm"]) < float(row["random_part_exact_mean_mm"])
        for row in event_rows
    )
    random_advantage = bool(
        semantic["part_exact_mm"] < random_mean["part_exact_mm"]
        or majority > len(event_rows) / 2
    )
    success = bool(
        compatibility["zero_drift"]
        and semantic["ordinary_change_vs_legacy_pct"] <= 1.0
        and semantic["global_change_vs_u2_pct"] <= 0.5
        and semantic["physical_pass_vs_legacy"]
        and semantic["part_improvement_vs_legacy_pct"] >= 1.0
        and random_advantage
    )
    decision = "Keep" if success else "Drop" if semantic["part_improvement_vs_legacy_pct"] <= 0 else "Unproven"
    _plots(config.output_dir, main_rows, frame_rows, event_rows)
    _report(
        config.output_dir / "semantic_budget_report.md",
        annotated_by_key,
        random_mean,
        compatibility,
        event_rows,
        decision,
    )
    print(f"\nSemantic Budget decision: {decision}")
    print(f"Artifacts: {config.output_dir}")


def cli() -> None:
    run(tyro.cli(BenchmarkConfig))


if __name__ == "__main__":
    cli()
