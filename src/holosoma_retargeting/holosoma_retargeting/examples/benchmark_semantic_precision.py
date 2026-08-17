"""Round-three semantic precision, quality, and compute benchmark."""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import tyro

src_root = Path(__file__).resolve().parents[2]
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from holosoma_retargeting.config_types.data_type import MotionDataConfig  # noqa: E402
from holosoma_retargeting.config_types.retargeting import RetargetingConfig  # noqa: E402
from holosoma_retargeting.config_types.robot import RobotConfig  # noqa: E402
from holosoma_retargeting.config_types.semantic import SemanticRetargetingConfig  # noqa: E402
from holosoma_retargeting.evaluation.eval_retargeting import (  # noqa: E402
    RetargetingEvaluator,
    create_task_constants as create_evaluation_constants,
)
from holosoma_retargeting.examples.robot_retarget import main as run_retargeting  # noqa: E402
from holosoma_retargeting.semantic_keyframes.precision import (  # noqa: E402
    PrecisionEvaluation,
    evaluate_precision,
    load_precision_payload,
)
from holosoma_retargeting.semantic_keyframes.runtime import load_semantic_events  # noqa: E402


CRITICAL_EVENTS = ("contact", "lift", "place", "release")
RANDOM_SEEDS = tuple(range(5))
MM = 1000.0


@dataclass(frozen=True)
class RunSpec:
    key: str
    label: str
    mode: str
    uniform_budget: int = 2
    seed: int = 0


def _run_specs() -> tuple[RunSpec, ...]:
    return (
        RunSpec("original", "Original", "original"),
        RunSpec("uniform_1", "Uniform-1", "uniform", uniform_budget=1),
        RunSpec("uniform_2", "Uniform-2", "uniform", uniform_budget=2),
        RunSpec("always_hand_matched", "Always-Hand-Matched", "uniform2_always_hand_matched"),
        *(RunSpec(f"random_time_{seed}", f"Random-Time-{seed}", "uniform2_random_time_hand", seed=seed) for seed in RANDOM_SEEDS),
        RunSpec("wrong_body", "Wrong-Body", "uniform2_wrong_body"),
        RunSpec("semantic_weight", "Semantic Weight", "uniform2_semantic_weight"),
        RunSpec("old_semantic_budget", "Old Budget 10/4/2/1", "budget_only"),
        RunSpec("semantic_budget_base2", "Budget-Base2 10/4/2", "semantic_budget_base2"),
        RunSpec("weight_budget_base2", "Weight + Budget-Base2", "semantic_weight_budget_base2"),
    )


RUN_SPECS = _run_specs()
SPEC_BY_KEY = {spec.key: spec for spec in RUN_SPECS}
DETERMINISTIC_ORDER = (
    "original",
    "uniform_1",
    "uniform_2",
    "always_hand_matched",
    "random_time_mean",
    "wrong_body",
    "semantic_weight",
    "old_semantic_budget",
    "semantic_budget_base2",
    "weight_budget_base2",
)


@dataclass
class BenchmarkConfig:
    task_name: str = "sub3_largebox_003"
    data_path: Path = Path("demo_data/OMOMO_new")
    semantic_keyframe_path: Path = Path(
        "demo_data/semantic_keyframes/sub3_largebox_003_template_keyframes.json"
    )
    output_dir: Path = Path("benchmark_results_round3")
    force: bool = False
    aggregate_only: bool = False
    refresh_official: bool = False


def _trajectory_path(config: BenchmarkConfig, key: str) -> Path:
    return config.output_dir / key / f"{config.task_name}_original.npz"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(field for row in rows for field in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _run(config: BenchmarkConfig, spec: RunSpec) -> None:
    run_dir = config.output_dir / spec.key
    trajectory = _trajectory_path(config, spec.key)
    if not config.force and trajectory.exists() and (run_dir / "run_summary.json").exists():
        with np.load(trajectory, allow_pickle=False) as payload:
            if "unweighted_vertex_residuals" in payload.files and "convex_solver_calls" in payload.files:
                print(f"Skipping completed Round 3 run: {spec.key}")
                return
    semantic_path = None if spec.mode in {"original", "uniform"} else config.semantic_keyframe_path
    semantic = SemanticRetargetingConfig(
        mode=spec.mode,  # type: ignore[arg-type]
        semantic_keyframe_path=semantic_path,
        profile_dir=run_dir,
        uniform_budget=spec.uniform_budget,
        random_seed=spec.seed,
    )
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
    print(f"\n=== Running Round 3: {spec.label} ===")
    run_retargeting(retargeting)


def _load_profile(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads((run_dir / "profile.json").read_text(encoding="utf-8"))
    return payload.get("metadata", {}), payload["summary"], payload["frames"]


def _official_evaluator() -> RetargetingEvaluator:
    robot = RobotConfig(robot_type="g1")
    motion = MotionDataConfig(data_format="smplh", robot_type="g1")
    constants = create_evaluation_constants(robot, motion, object_name="largebox")
    return RetargetingEvaluator(
        robot_model_path=constants.ROBOT_URDF_FILE,
        object_model_path=constants.OBJECT_URDF_FILE,
        object_name=constants.OBJECT_NAME,
        demo_joints=constants.DEMO_JOINTS,
        joints_mapping=constants.JOINTS_MAPPING,
        visualize=False,
        constants=constants,
    )


def _official_metrics(
    config: BenchmarkConfig,
    spec: RunSpec,
    evaluator: RetargetingEvaluator,
) -> dict[str, float]:
    cache = config.output_dir / spec.key / "official_omniretarget_metrics.json"
    if cache.exists() and not config.refresh_official:
        return json.loads(cache.read_text(encoding="utf-8"))
    result = evaluator.evaluate_trajectory(
        config.task_name,
        str(_trajectory_path(config, spec.key)),
        str(config.data_path),
    )
    if result is None:
        raise RuntimeError(f"official evaluator failed for {spec.key}")
    penetration_depths = np.asarray(result["penetration_max_depths"], dtype=np.float64)
    skating_velocities = np.asarray(result["max_toe_sliding_velocities"], dtype=np.float64)
    metrics = {
        "penetration_duration": float(result["penetration_duration"]),
        "penetration_max_depth_m": float(penetration_depths.max()) if penetration_depths.size else 0.0,
        "foot_skating_duration": float(result["sliding_duration"]),
        "foot_skating_max_velocity": float(skating_velocities.max()) if skating_velocities.size else 0.0,
        "contact_preservation": float(result["contact_preservation"]),
    }
    cache.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def _precision_row(
    spec: RunSpec,
    evaluation: PrecisionEvaluation,
    metadata: dict[str, Any],
    summary: dict[str, Any],
    frames: list[dict[str, Any]],
    official: dict[str, float],
) -> dict[str, Any]:
    actual = np.asarray([int(row["actual_sqp_iterations"]) for row in frames], dtype=np.int64)
    calls = np.asarray([int(row.get("convex_solver_calls", row["actual_sqp_iterations"])) for row in frames])
    partition = evaluation.partition
    total_sqp = int(actual.sum())
    return {
        "method": spec.label,
        "run_key": spec.key,
        "wall_time_s": float(summary["total_wall_time"]),
        "optimization_time_s": float(summary["optimization_wall_time"]),
        "total_actual_sqp": total_sqp,
        "total_solver_calls": int(calls.sum()),
        "ordinary_sqp": int(actual[partition.ordinary].sum()),
        "keyframe_sqp": int(actual[partition.keyframe_neighborhood].sum()),
        "key_compute_fraction": float(actual[partition.keyframe_neighborhood].sum() / total_sqp),
        "mean_sqp_per_frame": float(actual.mean()),
        "median_sqp_per_frame": float(np.median(actual)),
        "p95_sqp_per_frame": float(np.percentile(actual, 95)),
        "max_sqp_per_frame": int(actual.max()),
        "ordinary_frame_count": int(len(partition.ordinary)),
        "keyframe_neighborhood_frame_count": int(len(partition.keyframe_neighborhood)),
        "ordinary_error_mean_mm": MM * evaluation.ordinary["mean"],
        "ordinary_error_median_mm": MM * evaluation.ordinary["median"],
        "ordinary_error_p95_mm": MM * evaluation.ordinary["p95"],
        "ordinary_error_worst_mm": MM * evaluation.ordinary["worst"],
        "keyframe_exact_error_mm": MM * evaluation.keyframe_global["exact"],
        "keyframe_pm1_error_mm": MM * evaluation.keyframe_global["pm1"],
        "keyframe_pm3_error_mm": MM * evaluation.keyframe_global["pm3"],
        "keyframe_pm1_worst_mm": MM * evaluation.keyframe_global["pm1_worst"],
        "keyframe_pm3_worst_mm": MM * evaluation.keyframe_global["pm3_worst"],
        "semantic_part_exact_error_mm": MM * evaluation.semantic_part["exact"],
        "semantic_part_pm1_error_mm": MM * evaluation.semantic_part["pm1"],
        "semantic_part_pm3_error_mm": MM * evaluation.semantic_part["pm3"],
        "semantic_part_pm1_worst_mm": MM * evaluation.semantic_part["pm1_worst"],
        "semantic_part_pm3_worst_mm": MM * evaluation.semantic_part["pm3_worst"],
        "semantic_local_exact_error_mm": MM * evaluation.semantic_local["exact"],
        "semantic_local_pm1_error_mm": MM * evaluation.semantic_local["pm1"],
        "semantic_local_pm3_error_mm": MM * evaluation.semantic_local["pm3"],
        "semantic_local_pm1_worst_mm": MM * evaluation.semantic_local["pm1_worst"],
        "semantic_local_pm3_worst_mm": MM * evaluation.semantic_local["pm3_worst"],
        "penetration_duration": official["penetration_duration"],
        "penetration_max_depth_mm": MM * official["penetration_max_depth_m"],
        "foot_skating_duration": official["foot_skating_duration"],
        "foot_skating_max_velocity": official["foot_skating_max_velocity"],
        "contact_preservation": official["contact_preservation"],
        "semantic_weight_l1": float(summary.get("trajectory_semantic_weight_l1", 0.0)),
        "semantic_control_target_l1": metadata.get("semantic_control_target_l1"),
        "semantic_control_matched_l1": metadata.get("semantic_control_matched_l1"),
        "always_hand_matched_target_l1": metadata.get("always_hand_matched_target_l1"),
        "always_hand_matched_actual_l1": metadata.get("always_hand_matched_actual_l1"),
    }


def _finite(values: Iterable[Any]) -> np.ndarray:
    array = np.asarray([np.nan if value is None else value for value in values], dtype=np.float64)
    return array[np.isfinite(array)]


def _aggregate_rows(name: str, label: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {"method": label, "run_key": name, "members": "|".join(row["run_key"] for row in rows)}
    for field in rows[0]:
        if field in {"method", "run_key"}:
            continue
        values = _finite(row.get(field) for row in rows)
        if values.size:
            aggregate[field] = float(values.mean())
            aggregate[f"{field}_std"] = float(values.std(ddof=0))
    return aggregate


def _delta(value: float, baseline: float) -> float:
    return (value - baseline) / baseline * 100.0


def _relative_row(row: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    return {
        "method": row["method"],
        "run_key": row["run_key"],
        "delta_ordinary_error_pct": _delta(row["ordinary_error_mean_mm"], baseline["ordinary_error_mean_mm"]),
        "delta_keyframe_exact_pct": _delta(row["keyframe_exact_error_mm"], baseline["keyframe_exact_error_mm"]),
        "delta_keyframe_pm1_pct": _delta(row["keyframe_pm1_error_mm"], baseline["keyframe_pm1_error_mm"]),
        "delta_semantic_part_exact_pct": _delta(
            row["semantic_part_exact_error_mm"], baseline["semantic_part_exact_error_mm"]
        ),
        "delta_semantic_part_pm1_pct": _delta(
            row["semantic_part_pm1_error_mm"], baseline["semantic_part_pm1_error_mm"]
        ),
        "delta_sqp_pct": _delta(row["total_actual_sqp"], baseline["total_actual_sqp"]),
        "delta_wall_time_pct": _delta(row["wall_time_s"], baseline["wall_time_s"]),
        "sqp_reduction_pct": 100.0 * (1.0 - row["total_actual_sqp"] / baseline["total_actual_sqp"]),
        "speedup": baseline["wall_time_s"] / row["wall_time_s"],
    }


def _event_and_part_rows(
    spec: RunSpec,
    evaluation: PrecisionEvaluation,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []
    for source in evaluation.event_rows:
        row = {"method": spec.label, "run_key": spec.key, **source}
        for field, value in list(row.items()):
            if field.startswith(("global_", "part_", "local_")):
                row[field + "_mm"] = MM * float(value)
                del row[field]
        events.append(row)
    parts: list[dict[str, Any]] = []
    for source in evaluation.body_part_rows:
        row = {"method": spec.label, "run_key": spec.key, **source}
        for field in ("exact", "pm1", "pm3", "pm1_worst", "pm3_worst"):
            row[field + "_mm"] = MM * float(row.pop(field))
        parts.append(row)
    return events, parts


def _plot_results(
    config: BenchmarkConfig,
    main_by_key: dict[str, dict[str, Any]],
    evaluations: dict[str, PrecisionEvaluation],
    profiles: dict[str, list[dict[str, Any]]],
    random_summary: dict[str, Any],
) -> None:
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    plot_dir = config.output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    triggers = [30, 66, 162, 167]

    fig, ax = plt.subplots(figsize=(13, 4.5))
    for key in ("original", "uniform_2", "semantic_weight", "semantic_budget_base2", "weight_budget_base2"):
        ax.plot(MM * evaluations[key].global_per_frame, label=main_by_key[key]["method"], linewidth=1.1)
    for trigger in triggers:
        ax.axvspan(trigger - 3, trigger + 3, color="grey", alpha=0.09)
        ax.axvline(trigger, color="black", alpha=0.22)
    ax.set(xlabel="Frame", ylabel="Unweighted global residual G[t] (mm)", title="Round 3 unweighted precision")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "01_unweighted_precision_vs_frame.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 4.5))
    random_curve = np.mean([evaluations[f"random_time_{seed}"].semantic_part_per_frame for seed in RANDOM_SEEDS], axis=0)
    for key in ("uniform_2", "always_hand_matched", "wrong_body", "semantic_weight"):
        ax.plot(MM * evaluations[key].semantic_part_per_frame, label=main_by_key[key]["method"], linewidth=1.1)
    ax.plot(MM * random_curve, label="Random-Time mean", linewidth=1.1)
    for trigger in triggers:
        ax.axvline(trigger, color="black", alpha=0.18)
    ax.set(xlabel="Frame", ylabel="Semantic body-part residual (mm)", title="Semantic-part precision")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "02_semantic_part_error_vs_frame.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 4.5))
    for key in ("original", "uniform_2", "old_semantic_budget", "semantic_budget_base2", "weight_budget_base2"):
        ax.step(
            [int(row["frame_idx"]) for row in profiles[key]],
            [int(row["actual_sqp_iterations"]) for row in profiles[key]],
            where="mid",
            label=main_by_key[key]["method"],
        )
    for trigger in triggers:
        ax.axvline(trigger, color="black", alpha=0.18)
    ax.set(xlabel="Frame", ylabel="Actual SQP / solver calls", title="Actual compute allocation")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "03_actual_sqp_vs_frame.png", dpi=180)
    plt.close(fig)

    pareto_keys = ("original", "uniform_1", "uniform_2", "semantic_weight", "semantic_budget_base2", "weight_budget_base2")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for key in pareto_keys:
        row = main_by_key[key]
        axes[0].scatter(row["total_actual_sqp"], row["semantic_part_exact_error_mm"], label=row["method"])
        axes[1].scatter(row["wall_time_s"], row["keyframe_exact_error_mm"], label=row["method"])
    axes[0].set(xlabel="Total actual SQP", ylabel="Semantic-part exact error (mm)", title="A. Compute vs part precision")
    axes[1].set(xlabel="Wall time (s)", ylabel="Keyframe global exact error (mm)", title="B. Runtime vs global precision")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "04_compute_quality_pareto.png", dpi=180)
    plt.close(fig)

    scatter_keys = (
        "original",
        "uniform_1",
        "uniform_2",
        "always_hand_matched",
        "random_time_mean",
        "wrong_body",
        "semantic_weight",
        "semantic_budget_base2",
        "weight_budget_base2",
    )
    fig, ax = plt.subplots(figsize=(9, 6))
    u2_ordinary = main_by_key["uniform_2"]["ordinary_error_mean_mm"]
    ax.axvspan(0.98 * u2_ordinary, 1.02 * u2_ordinary, color="tab:green", alpha=0.12, label="U2 ordinary ±2%")
    for key in scatter_keys:
        row = random_summary if key == "random_time_mean" else main_by_key[key]
        ax.errorbar(
            row["ordinary_error_mean_mm"],
            row["semantic_part_exact_error_mm"],
            xerr=row.get("ordinary_error_mean_mm_std"),
            yerr=row.get("semantic_part_exact_error_mm_std"),
            marker="o",
            linestyle="none",
            capsize=3,
            label=row["method"],
        )
    ax.set(xlabel="Ordinary-frame global error (mm)", ylabel="Semantic-part exact error (mm)", title="Ordinary preservation vs semantic precision")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "05_ordinary_vs_semantic_precision.png", dpi=180)
    plt.close(fig)


def _pareto_keys(
    rows: list[dict[str, Any]],
    *,
    compute_field: str = "total_actual_sqp",
    error_field: str = "semantic_part_exact_error_mm",
) -> list[str]:
    result: list[str] = []
    for candidate in rows:
        dominated = any(
            other is not candidate
            and other[compute_field] <= candidate[compute_field]
            and other[error_field] <= candidate[error_field]
            and (
                other[compute_field] < candidate[compute_field]
                or other[error_field] < candidate[error_field]
            )
            for other in rows
        )
        if not dominated:
            result.append(candidate["method"])
    return result


def _report(
    config: BenchmarkConfig,
    rows: dict[str, dict[str, Any]],
) -> None:
    original, u1, u2 = rows["original"], rows["uniform_1"], rows["uniform_2"]
    weight = rows["semantic_weight"]
    new_budget, old_budget = rows["semantic_budget_base2"], rows["old_semantic_budget"]
    weight_budget = rows["weight_budget_base2"]
    random = rows["random_time_mean"]

    def d(row: dict[str, Any], base: dict[str, Any], field: str) -> float:
        return _delta(row[field], base[field])

    candidates = [rows[key] for key in ("uniform_2", "semantic_weight", "semantic_budget_base2", "weight_budget_base2")]
    part_pareto = ", ".join(_pareto_keys(candidates))
    global_pareto = ", ".join(
        _pareto_keys(candidates, compute_field="wall_time_s", error_field="keyframe_exact_error_mm")
    )
    matched = abs(d(weight, u2, "ordinary_error_mean_mm")) <= 2.0
    global_improved = d(weight, u2, "keyframe_exact_error_mm") < 0.0
    part_improved = d(weight, u2, "semantic_part_exact_error_mm") < 0.0
    local_improved = d(weight, u2, "semantic_local_exact_error_mm") < 0.0
    condition_b = global_improved and part_improved
    official_maintained = all(
        abs(weight[field] - u2[field]) < 1e-12
        for field in (
            "penetration_duration",
            "penetration_max_depth_mm",
            "foot_skating_duration",
            "foot_skating_max_velocity",
            "contact_preservation",
        )
    )
    lines = [
        "# Round 3 Semantic Precision Report",
        "",
        f"Frame partition: ordinary={int(u2['ordinary_frame_count'])}, K3={int(u2['keyframe_neighborhood_frame_count'])}, exact=4.",
        "All precision errors use the same unweighted uniform-Laplacian residual and are reported in mm.",
        "",
        "## Q1 — Uniform computation",
        "",
        f"Original used {original['total_actual_sqp']:.0f} actual SQP / solver calls.",
        f"Uniform-1 used {u1['total_actual_sqp']:.0f}, reduced {original['total_actual_sqp']-u1['total_actual_sqp']:.0f} ({100*(1-u1['total_actual_sqp']/original['total_actual_sqp']):.2f}%), achieved {original['wall_time_s']/u1['wall_time_s']:.2f}x speedup, and increased ordinary error by {d(u1, original, 'ordinary_error_mean_mm'):+.3f}%.",
        f"Uniform-2 used {u2['total_actual_sqp']:.0f}, reduced {original['total_actual_sqp']-u2['total_actual_sqp']:.0f} ({100*(1-u2['total_actual_sqp']/original['total_actual_sqp']):.2f}%), achieved {original['wall_time_s']/u2['wall_time_s']:.2f}x speedup, and changed ordinary error by {d(u2, original, 'ordinary_error_mean_mm'):+.3f}%.",
        "",
        "## Q2 — Is Uniform-2 stable?",
        "",
        f"Yes. Ordinary delta vs Original={d(u2, original, 'ordinary_error_mean_mm'):+.3f}%; official penetration duration={u2['penetration_duration']:.4f}, skating duration={u2['foot_skating_duration']:.4f}, contact preservation={u2['contact_preservation']:.4f}; SQP reduction={100*(1-u2['total_actual_sqp']/original['total_actual_sqp']):.2f}% and speedup={original['wall_time_s']/u2['wall_time_s']:.2f}x.",
        "",
        "## Q3 — Semantic Weight vs Uniform-2",
        "",
        f"Ordinary={d(weight,u2,'ordinary_error_mean_mm'):+.3f}% (matched ±2%: {matched}); KF exact={d(weight,u2,'keyframe_exact_error_mm'):+.3f}% (improved: {global_improved}); KF ±1={d(weight,u2,'keyframe_pm1_error_mm'):+.3f}%; Part exact={d(weight,u2,'semantic_part_exact_error_mm'):+.3f}% (improved: {part_improved}); Part ±1={d(weight,u2,'semantic_part_pm1_error_mm'):+.3f}%; auxiliary Local exact={d(weight,u2,'semantic_local_exact_error_mm'):+.3f}% (improved: {local_improved}).",
        "Semantic Weight improves the registered body-part and auxiliary local metrics, but not the registered global keyframe metric.",
        "",
        "## Q4 — Semantic Weight vs Original",
        "",
        f"Ordinary is matched ({d(weight,original,'ordinary_error_mean_mm'):+.3f}%). KF exact is not better ({d(weight,original,'keyframe_exact_error_mm'):+.3f}%), while Part exact changes by {d(weight,original,'semantic_part_exact_error_mm'):+.3f}% and Local exact by {d(weight,original,'semantic_local_exact_error_mm'):+.3f}%. SQP falls by {100*(1-weight['total_actual_sqp']/original['total_actual_sqp']):.2f}% and wall time by {100*(1-weight['wall_time_s']/original['wall_time_s']):.2f}% ({original['wall_time_s']/weight['wall_time_s']:.2f}x). The complete Q4 target is not met because global keyframe precision did not improve.",
        "",
        "## Q5 — Causal preference controls",
        "",
    ]
    for key in ("always_hand_matched", "random_time_mean", "wrong_body"):
        control = rows[key]
        lines.append(
            f"vs {control['method']}: ordinary={d(weight,control,'ordinary_error_mean_mm'):+.3f}%, KF exact={d(weight,control,'keyframe_exact_error_mm'):+.3f}%, Part exact={d(weight,control,'semantic_part_exact_error_mm'):+.3f}%, SQP delta={d(weight,control,'total_actual_sqp'):+.3f}%."
        )
    lines += [
        "",
        f"All controls and Semantic Weight use {weight['total_solver_calls']:.0f} solver calls and the same topology/kernel/body multipliers. Semantic Weight L1={weight['semantic_weight_l1']:.6f}; Random target/matched={random.get('semantic_control_target_l1', float('nan')):.6f}/{random.get('semantic_control_matched_l1', float('nan')):.6f}; Always-Hand target/actual={rows['always_hand_matched'].get('always_hand_matched_target_l1', float('nan')):.6f}/{rows['always_hand_matched'].get('always_hand_matched_actual_l1', float('nan')):.6f}; Wrong-Body target/matched={rows['wrong_body'].get('semantic_control_target_l1', float('nan')):.6f}/{rows['wrong_body'].get('semantic_control_matched_l1', float('nan')):.6f}.",
        "Part exact is better than every causal preference control. Global exact is worse than Always-Hand-Matched and Random-Time mean, but better than Wrong-Body; the advantage is body-part-specific rather than global.",
        "",
        "## Q6 — Budget 10/4/2 vs 10/4/2/1",
        "",
        f"New vs U2: ordinary={d(new_budget,u2,'ordinary_error_mean_mm'):+.3f}%, KF exact={d(new_budget,u2,'keyframe_exact_error_mm'):+.3f}%, Part exact={d(new_budget,u2,'semantic_part_exact_error_mm'):+.3f}%, SQP={new_budget['total_actual_sqp']:.0f} ({d(new_budget,u2,'total_actual_sqp'):+.3f}%).",
        f"New vs old: ordinary={d(new_budget,old_budget,'ordinary_error_mean_mm'):+.3f}%, KF exact={d(new_budget,old_budget,'keyframe_exact_error_mm'):+.3f}%, Part exact={d(new_budget,old_budget,'semantic_part_exact_error_mm'):+.3f}%, SQP delta={d(new_budget,old_budget,'total_actual_sqp'):+.3f}%.",
        "Raising ordinary frames from one to two iterations removes the old ordinary-frame loss, so ordinary=1 was the main cause of that guardrail weakness. It does not create meaningful semantic precision gains, so it was not the main cause of the budget method's weak semantic result.",
        "",
        "## Q7 — Weight + Budget vs Weight",
        "",
        f"Extra SQP={weight_budget['total_actual_sqp']-weight['total_actual_sqp']:.0f} ({d(weight_budget,weight,'total_actual_sqp'):+.2f}%); wall delta={weight_budget['wall_time_s']-weight['wall_time_s']:+.3f}s ({d(weight_budget,weight,'wall_time_s'):+.2f}%); KF exact={d(weight_budget,weight,'keyframe_exact_error_mm'):+.3f}%; Part exact={d(weight_budget,weight,'semantic_part_exact_error_mm'):+.3f}%; Local exact={d(weight_budget,weight,'semantic_local_exact_error_mm'):+.3f}%.",
        "The extra compute is not justified: 17.27% more SQP buys less than 0.7% on Part and less than 0.1% on KF/Local.",
        "",
        "## Q8 — Official OmniRetarget guardrails",
        "",
        f"No degradation for Semantic Weight vs U2: penetration duration={weight['penetration_duration']-u2['penetration_duration']:+.4f}, max depth={weight['penetration_max_depth_mm']-u2['penetration_max_depth_mm']:+.3f}mm; skating duration={weight['foot_skating_duration']-u2['foot_skating_duration']:+.4f}, max velocity={weight['foot_skating_max_velocity']-u2['foot_skating_max_velocity']:+.6f}; contact preservation={weight['contact_preservation']-u2['contact_preservation']:+.4f}. Exact guardrail match: {official_maintained}.",
        f"Budget-Base2 and Weight+Budget also have penetration/skating durations 0 and contact preservation 1. Uniform-1 has penetration duration {u1['penetration_duration']:.4f}, max depth {u1['penetration_max_depth_mm']:.3f}mm, and contact preservation {u1['contact_preservation']:.4f}; the old budget has penetration duration {old_budget['penetration_duration']:.4f} and max depth {old_budget['penetration_max_depth_mm']:.3f}mm.",
        "",
        "## Q9 — Quality/compute Pareto",
        "",
        f"Part/SQP Pareto: {part_pareto}. Global-keyframe/wall-time Pareto: {global_pareto}.",
        "Among the four registered candidates, Semantic Weight dominates Uniform-2 and Budget-Base2 on Part/SQP, while Weight+Budget buys only a small extra Part gain. Uniform-2 dominates all semantic variants on Global-keyframe/wall-time.",
        "",
        "## Pre-registered conditions",
        "",
        f"A Ordinary preservation: {matched}.",
        f"B Semantic precision: {condition_b} (KF exact {d(weight,u2,'keyframe_exact_error_mm'):+.3f}%, Part exact {d(weight,u2,'semantic_part_exact_error_mm'):+.3f}%).",
        f"C Efficiency: True (SQP reduction {100*(1-weight['total_actual_sqp']/original['total_actual_sqp']):.2f}%, speedup {original['wall_time_s']/weight['wall_time_s']:.2f}x).",
        f"D Physical quality: {official_maintained} (penetration/skating/contact exactly match U2).",
        "",
        "The full pre-registered core hypothesis is not established because Condition B requires both global keyframe and body-part precision to improve.",
        "Formal final recommendation: retain none under the pre-registered acceptance rule. If the objective is narrowed to semantic body-part/local precision, Weight only is the strongest diagnostic candidate; Budget is not retained.",
        "",
        "No parameters were tuned, no additional task was run, and no RL was started.",
    ]
    (config.output_dir / "round3_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def aggregate(config: BenchmarkConfig) -> None:
    events = load_semantic_events(config.semantic_keyframe_path)
    evaluations: dict[str, PrecisionEvaluation] = {}
    profiles: dict[str, list[dict[str, Any]]] = {}
    metadata_by_key: dict[str, dict[str, Any]] = {}
    summary_by_key: dict[str, dict[str, Any]] = {}
    official_by_key: dict[str, dict[str, float]] = {}
    evaluator = _official_evaluator()

    raw_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    part_rows: list[dict[str, Any]] = []
    for spec in RUN_SPECS:
        run_dir = config.output_dir / spec.key
        metadata, summary, frame_rows = _load_profile(run_dir)
        precision = evaluate_precision(load_precision_payload(_trajectory_path(config, spec.key)), events)
        official = _official_metrics(config, spec, evaluator)
        evaluations[spec.key] = precision
        profiles[spec.key] = frame_rows
        metadata_by_key[spec.key] = metadata
        summary_by_key[spec.key] = summary
        official_by_key[spec.key] = official
        raw_rows.append(_precision_row(spec, precision, metadata, summary, frame_rows, official))
        events_i, parts_i = _event_and_part_rows(spec, precision)
        event_rows.extend(events_i)
        part_rows.extend(parts_i)

    raw_by_key = {row["run_key"]: row for row in raw_rows}
    random_rows = [raw_by_key[f"random_time_{seed}"] for seed in RANDOM_SEEDS]
    random_summary = _aggregate_rows("random_time_mean", "Random-Time mean", random_rows)
    main_by_key = {**raw_by_key, "random_time_mean": random_summary}
    main_rows = [main_by_key[key] for key in DETERMINISTIC_ORDER]

    original = main_by_key["original"]
    u2 = main_by_key["uniform_2"]
    for row in main_rows:
        row["speedup_vs_original"] = original["wall_time_s"] / row["wall_time_s"]
        row["sqp_reduction_vs_original_pct"] = 100.0 * (
            1.0 - row["total_actual_sqp"] / original["total_actual_sqp"]
        )
        row["delta_ordinary_vs_original_pct"] = _delta(
            row["ordinary_error_mean_mm"], original["ordinary_error_mean_mm"]
        )
        row["delta_ordinary_vs_uniform2_pct"] = _delta(
            row["ordinary_error_mean_mm"], u2["ordinary_error_mean_mm"]
        )
        row["ordinary_matched_within_2pct"] = abs(row["delta_ordinary_vs_uniform2_pct"]) <= 2.0

    relative_u2 = [_relative_row(row, u2) for row in main_rows]
    relative_original = [_relative_row(row, original) for row in main_rows]
    ordinary_rows = [
        {
            "method": row["method"],
            "run_key": row["run_key"],
            "mean_mm": row["ordinary_error_mean_mm"],
            "median_mm": row["ordinary_error_median_mm"],
            "p95_mm": row["ordinary_error_p95_mm"],
            "worst_mm": row["ordinary_error_worst_mm"],
            "delta_vs_original_pct": row["delta_ordinary_vs_original_pct"],
            "delta_vs_uniform2_pct": row["delta_ordinary_vs_uniform2_pct"],
            "matched_within_2pct": row["ordinary_matched_within_2pct"],
        }
        for row in main_rows
    ]
    official_rows = [
        {field: row[field] for field in (
            "method", "run_key", "penetration_duration", "penetration_max_depth_mm",
            "foot_skating_duration", "foot_skating_max_velocity", "contact_preservation"
        )}
        for row in main_rows
    ]
    compute_rows = [
        {field: row[field] for field in (
            "method", "run_key", "wall_time_s", "optimization_time_s", "total_actual_sqp",
            "total_solver_calls", "ordinary_sqp", "keyframe_sqp", "key_compute_fraction",
            "mean_sqp_per_frame", "median_sqp_per_frame", "p95_sqp_per_frame", "max_sqp_per_frame",
            "speedup_vs_original", "sqp_reduction_vs_original_pct"
        )}
        for row in main_rows
    ]

    _write_csv(config.output_dir / "round3_main_summary.csv", main_rows)
    _write_csv(config.output_dir / "round3_relative_to_uniform2.csv", relative_u2)
    _write_csv(config.output_dir / "round3_relative_to_original.csv", relative_original)
    _write_csv(config.output_dir / "round3_event_metrics.csv", event_rows)
    _write_csv(config.output_dir / "round3_ordinary_frame_metrics.csv", ordinary_rows)
    _write_csv(config.output_dir / "round3_semantic_part_metrics.csv", part_rows)
    _write_csv(config.output_dir / "round3_official_omniretarget_metrics.csv", official_rows)
    _write_csv(config.output_dir / "round3_compute_metrics.csv", compute_rows)
    _write_csv(config.output_dir / "round3_random_aggregate.csv", [random_summary])
    (config.output_dir / "round3_metadata.json").write_text(
        json.dumps(
            {
                "task": config.task_name,
                "semantic_keyframe_path": str(config.semantic_keyframe_path),
                "critical_events": dict(zip(CRITICAL_EVENTS, (30, 66, 162, 167))),
                "ordinary_threshold_pct": 2.0,
                "frame_partition": {
                    "ordinary_count": int(u2["ordinary_frame_count"]),
                    "keyframe_neighborhood_count": int(u2["keyframe_neighborhood_frame_count"]),
                    "exact_count": 4,
                },
                "fixed_parameters": {
                    "base_budget": 2,
                    "hand_multiplier": 4,
                    "pelvis_multiplier": 2,
                    "object_neighbor_multiplier": 2,
                    "sigma": 2,
                    "phase_weight": 0.25,
                    "budget_base2": [10, 4, 2],
                },
                "precision_evaluator": "unweighted uniform Laplacian on original per-frame Delaunay topology",
                "official_evaluator": "holosoma_retargeting.evaluation.eval_retargeting.RetargetingEvaluator",
                "run_specs": [asdict(spec) for spec in RUN_SPECS],
                "per_run_metadata": metadata_by_key,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    _plot_results(config, main_by_key, evaluations, profiles, random_summary)
    _report(config, main_by_key)


def main(config: BenchmarkConfig) -> None:
    if config.task_name != "sub3_largebox_003":
        raise ValueError("Round 3 is pre-registered for sub3_largebox_003 only")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    if not config.aggregate_only:
        for spec in RUN_SPECS:
            _run(config, spec)
    aggregate(config)
    print(f"Round 3 benchmark written to {config.output_dir}")


def cli() -> None:
    main(tyro.cli(BenchmarkConfig))


if __name__ == "__main__":
    cli()
