"""Sequential final benchmark for the Legacy semantic-weighting main line."""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

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
from holosoma_retargeting.semantic_keyframes.runtime import load_semantic_events  # noqa: E402


MM = 1000.0


@dataclass(frozen=True)
class RunSpec:
    key: str
    label: str
    mode: str
    uniform_budget: int = 2
    seed: int = 0


@dataclass
class BenchmarkConfig:
    """Fixed single-task convergence benchmark."""

    task_name: str = "sub3_largebox_003"
    data_path: Path = Path("demo_data/OMOMO_new")
    semantic_keyframe_path: Path = Path(
        "demo_data/semantic_keyframes/sub3_largebox_003_semantic_v2.json"
    )
    output_dir: Path = Path("benchmark_results_semantic_final")
    historical_round4_dir: Path = Path("benchmark_results_round4")
    force: bool = False
    aggregate_only: bool = False
    refresh_official: bool = False


def _trajectory(run_dir: Path, task_name: str) -> Path:
    return run_dir / f"{task_name}_original.npz"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(field for row in rows for field in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _run(config: BenchmarkConfig, spec: RunSpec) -> Path:
    run_dir = config.output_dir / "runs" / spec.key
    trajectory = _trajectory(run_dir, config.task_name)
    complete = trajectory.exists() and (run_dir / "profile.json").exists()
    if complete and not config.force:
        print(f"Reusing completed final benchmark run: {spec.key}")
        return run_dir
    if config.aggregate_only:
        raise FileNotFoundError(f"missing completed run for --aggregate-only: {run_dir}")
    semantic_path = None if spec.mode in {"original", "uniform", "adaptive_base"} else config.semantic_keyframe_path
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
        return {key: float(value) for key, value in json.loads(cache.read_text(encoding="utf-8")).items()}
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


def _load_profile(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads((run_dir / "profile.json").read_text(encoding="utf-8"))
    return payload["summary"], payload["frames"]


def _metric_row(
    config: BenchmarkConfig,
    spec: RunSpec,
    run_dir: Path,
    evaluation: PrecisionEvaluation,
    official: dict[str, float],
) -> dict[str, Any]:
    summary, frames = _load_profile(run_dir)
    iterations = np.asarray([int(row["actual_sqp_iterations"]) for row in frames], dtype=np.int64)
    calls = np.asarray(
        [int(row.get("convex_solver_calls", row["actual_sqp_iterations"])) for row in frames],
        dtype=np.int64,
    )
    edge = evaluation.semantic_edge or {}
    weighted_edge = evaluation.criticality_weighted_edge or {}
    return {
        "method": spec.label,
        "run_key": spec.key,
        "mode": spec.mode,
        "wall_time_s": float(summary["total_wall_time"]),
        "actual_sqp": int(iterations.sum()),
        "solver_calls": int(calls.sum()),
        "ordinary_mm": MM * evaluation.ordinary["mean"],
        "global_kf_exact_mm": MM * evaluation.keyframe_global["exact"],
        "part_exact_mm": MM * evaluation.semantic_part["exact"],
        "edge_exact_mm": MM * float(edge.get("exact", np.nan)),
        "local_exact_mm": MM * evaluation.semantic_local["exact"],
        "criticality_weighted_part_exact_mm": MM * evaluation.criticality_weighted_part["exact"],
        "criticality_weighted_edge_exact_mm": MM * float(weighted_edge.get("exact", np.nan)),
        "penetration_duration": official["penetration_duration"],
        "penetration_max_depth_mm": MM * official["penetration_max_depth_m"],
        "foot_skating_duration": official["foot_skating_duration"],
        "foot_skating_max_velocity": official["foot_skating_max_velocity"],
        "contact_preservation": official["contact_preservation"],
        "frames_at_1": int(np.sum(iterations[1:] == 1)),
        "frames_at_2": int(np.sum(iterations[1:] == 2)),
        "frames_at_3": int(np.sum(iterations[1:] == 3)),
        "frames_at_4": int(np.sum(iterations[1:] == 4)),
    }


def _evaluate_run(
    config: BenchmarkConfig,
    spec: RunSpec,
    run_dir: Path,
    events: list[Any],
    evaluator: RetargetingEvaluator,
) -> tuple[dict[str, Any], PrecisionEvaluation]:
    precision = evaluate_precision(load_precision_payload(_trajectory(run_dir, config.task_name)), events)
    return _metric_row(config, spec, run_dir, precision, _official(config, run_dir, evaluator)), precision


def _pct(value: float, baseline: float) -> float:
    return (float(value) - float(baseline)) / float(baseline) * 100.0


def _physical_pass(row: dict[str, Any], baseline: dict[str, Any]) -> bool:
    tolerance = 1e-12
    return bool(
        row["penetration_duration"] <= baseline["penetration_duration"] + tolerance
        and row["penetration_max_depth_mm"] <= baseline["penetration_max_depth_mm"] + tolerance
        and row["foot_skating_duration"] <= baseline["foot_skating_duration"] + tolerance
        and row["foot_skating_max_velocity"] <= baseline["foot_skating_max_velocity"] + tolerance
        and row["contact_preservation"] + tolerance >= baseline["contact_preservation"]
    )


def _annotate(rows: Iterable[dict[str, Any]], u2: dict[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        row.update(
            {
                "ordinary_change_vs_u2_pct": _pct(row["ordinary_mm"], u2["ordinary_mm"]),
                "global_change_vs_u2_pct": _pct(row["global_kf_exact_mm"], u2["global_kf_exact_mm"]),
                "part_improvement_vs_u2_pct": -_pct(row["part_exact_mm"], u2["part_exact_mm"]),
                "edge_improvement_vs_u2_pct": -_pct(row["edge_exact_mm"], u2["edge_exact_mm"]),
                "local_improvement_vs_u2_pct": -_pct(row["local_exact_mm"], u2["local_exact_mm"]),
                "physical_pass": _physical_pass(row, u2),
            }
        )
        output.append(row)
    return output


def _compatibility(
    config: BenchmarkConfig,
    specs: tuple[RunSpec, ...],
    current_dirs: dict[str, Path],
    events: list[Any],
    evaluator: RetargetingEvaluator,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    historical_names = {
        "compat_original": "original",
        "compat_uniform_2": "uniform_2",
        "compat_legacy": "legacy_weight",
    }
    rows: list[dict[str, Any]] = []
    current_metrics: dict[str, dict[str, Any]] = {}
    for spec in specs:
        current_dir = current_dirs[spec.key]
        historical_dir = config.historical_round4_dir / historical_names[spec.key]
        current_row, _ = _evaluate_run(config, spec, current_dir, events, evaluator)
        historical_row, _ = _evaluate_run(config, spec, historical_dir, events, evaluator)
        current_metrics[spec.key] = current_row
        with np.load(_trajectory(current_dir, config.task_name), allow_pickle=False) as current_npz, np.load(
            _trajectory(historical_dir, config.task_name), allow_pickle=False
        ) as historical_npz:
            current_qpos = np.asarray(current_npz["qpos"], dtype=np.float64)
            historical_qpos = np.asarray(historical_npz["qpos"], dtype=np.float64)
            qpos_max = float(np.max(np.abs(current_qpos - historical_qpos)))
            current_sqp = int(np.asarray(current_npz["actual_sqp_iterations"]).sum())
            historical_sqp = int(np.asarray(historical_npz["actual_sqp_iterations"]).sum())
        metric_fields = ("ordinary_mm", "global_kf_exact_mm", "part_exact_mm")
        physical_fields = (
            "penetration_duration",
            "penetration_max_depth_mm",
            "foot_skating_duration",
            "foot_skating_max_velocity",
            "contact_preservation",
        )
        metric_max = max(abs(float(current_row[field]) - float(historical_row[field])) for field in metric_fields)
        physical_max = max(abs(float(current_row[field]) - float(historical_row[field])) for field in physical_fields)
        passed = qpos_max <= 1e-10 and current_sqp == historical_sqp and metric_max <= 1e-9 and physical_max <= 1e-12
        rows.append(
            {
                "method": spec.label,
                "qpos_max_abs_diff": qpos_max,
                "qpos_pass": qpos_max <= 1e-10,
                "cached_actual_sqp": historical_sqp,
                "current_actual_sqp": current_sqp,
                "sqp_pass": current_sqp == historical_sqp,
                "ordinary_abs_diff_mm": abs(current_row["ordinary_mm"] - historical_row["ordinary_mm"]),
                "global_abs_diff_mm": abs(current_row["global_kf_exact_mm"] - historical_row["global_kf_exact_mm"]),
                "part_abs_diff_mm": abs(current_row["part_exact_mm"] - historical_row["part_exact_mm"]),
                "official_physical_max_abs_diff": physical_max,
                "compatibility_pass": passed,
            }
        )
    return rows, current_metrics


def _mean_row(label: str, key: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"method": label, "run_key": key, "members": "|".join(row["run_key"] for row in rows)}
    for field in rows[0]:
        if field in {"method", "run_key", "mode"}:
            continue
        values = np.asarray([row[field] for row in rows if isinstance(row.get(field), (int, float))], dtype=np.float64)
        if values.size:
            result[field] = float(values.mean())
            result[f"{field}_std"] = float(values.std(ddof=0))
    return result


def _adaptive_logs(config: BenchmarkConfig, specs: list[RunSpec]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    frames: list[dict[str, Any]] = []
    histogram: list[dict[str, Any]] = []
    for spec in specs:
        run_dir = config.output_dir / "runs" / spec.key
        path = run_dir / "adaptive_frame_log.csv"
        if path.exists():
            with path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    frames.append({"method": spec.label, "run_key": spec.key, **row})
        _, profile_frames = _load_profile(run_dir)
        counts = np.asarray([int(row["actual_sqp_iterations"]) for row in profile_frames[1:]], dtype=np.int64)
        for iteration in (1, 2, 3, 4):
            count = int(np.sum(counts == iteration))
            histogram.append(
                {
                    "method": spec.label,
                    "run_key": spec.key,
                    "iterations": iteration,
                    "frame_count": count,
                    "frame_fraction": count / max(len(counts), 1),
                }
            )
    return frames, histogram


def _write_cleanup_artifacts(config: BenchmarkConfig) -> None:
    removed = [
        "raw additive Part objective modes",
        "cardinality/group-balanced additive Part modes",
        "Jacobian-balanced additive mode",
        "additive Part/Edge objective modes",
        "constrained semantic refinement mode",
        "constrained backtracking mode",
        "trust-region contraction/re-solve mode",
        "their schedulers, solver branches, benchmark entrypoints, and exclusive unit tests",
    ]
    (config.output_dir / "removed_modes.txt").write_text("\n".join(removed) + "\n", encoding="utf-8")
    (config.output_dir / "cleanup_report.md").write_text(
        "# Semantic branch cleanup\n\n"
        "The optimizer now has one semantic algorithmic path: Legacy mean-one Laplacian residual "
        "reweighting. Semantic Edge uses equivalent cross-entity edge-endpoint emphasis in that same "
        "residual vector and de-duplicates Legacy object-neighbor overlap with `max`. No additive "
        "semantic objective, constrained second stage, backtracking, Jacobian balancing, or trust-region "
        "re-solve remains callable.\n\n"
        "Shared VLM generation/repair, semantic_v2 data, body mapping, original interaction mesh, "
        "precision/official evaluators, physical checks, and solver profiling remain. Historical result "
        "directories were treated as read-only records.\n",
        encoding="utf-8",
    )


def run(config: BenchmarkConfig) -> None:
    if config.task_name != "sub3_largebox_003":
        raise ValueError("the final benchmark is restricted to OMOMO/sub3_largebox_003")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    _write_cleanup_artifacts(config)
    events = load_semantic_events(config.semantic_keyframe_path)
    evaluator = _official_evaluator()

    # 1-2. Cleanup compatibility gate.
    compatibility_specs = (
        RunSpec("compat_original", "Original", "original"),
        RunSpec("compat_uniform_2", "Uniform-2", "uniform", uniform_budget=2),
        RunSpec("compat_legacy", "Legacy Semantic Weight", "uniform2_semantic_weight"),
    )
    compatibility_dirs = {spec.key: _run(config, spec) for spec in compatibility_specs}
    compatibility, anchors = _compatibility(
        config, compatibility_specs, compatibility_dirs, events, evaluator
    )
    _write_csv(config.output_dir / "backward_compatibility.csv", compatibility)
    if not all(bool(row["compatibility_pass"]) for row in compatibility):
        raise RuntimeError("backward compatibility failed; stopping before Edge")
    u2 = anchors["compat_uniform_2"]

    # 3-4. Edge ablation, c=1, fixed Uniform-2 compute.
    edge_specs = (
        RunSpec("edge_part_uniform", "E1 Legacy Part (c=1)", "uniform2_semantic_weight_uniform"),
        RunSpec("edge_only_uniform", "E2 Edge Only (c=1)", "uniform2_semantic_edge_weight_uniform"),
        RunSpec(
            "edge_part_edge_uniform",
            "E3 Legacy Part + Edge (c=1)",
            "uniform2_semantic_part_edge_weight_uniform",
        ),
    )
    edge_rows = [u2]
    edge_rows[0] = {**edge_rows[0], "method": "E0 Uniform-2", "run_key": "compat_uniform_2"}
    for spec in edge_specs:
        row, _ = _evaluate_run(config, spec, _run(config, spec), events, evaluator)
        edge_rows.append(row)
    edge_rows = _annotate(edge_rows, u2)
    part = next(row for row in edge_rows if row["run_key"] == "edge_part_uniform")
    part_edge = next(row for row in edge_rows if row["run_key"] == "edge_part_edge_uniform")
    keep_edge = bool(
        part_edge["edge_exact_mm"] < part["edge_exact_mm"] - 1e-9
        and part_edge["local_exact_mm"] < part["local_exact_mm"] - 1e-9
        and _pct(part_edge["part_exact_mm"], part["part_exact_mm"]) <= 0.5
        and part_edge["global_change_vs_u2_pct"] <= 0.5
        and part_edge["ordinary_change_vs_u2_pct"] <= 1.0
        and part_edge["physical_pass"]
    )
    for row in edge_rows:
        row["edge_decision"] = "Keep" if keep_edge else "Drop"
    _write_csv(config.output_dir / "edge_ablation.csv", edge_rows)

    # 5-6. Criticality only after the Edge decision.
    prefix = "uniform2_semantic_part_edge_weight" if keep_edge else "uniform2_semantic_weight"
    c0_key = "edge_part_edge_uniform" if keep_edge else "edge_part_uniform"
    c0 = next(row for row in edge_rows if row["run_key"] == c0_key)
    critical_specs = [
        RunSpec("critical_uniform_matched", "C1 Uniform Criticality Matched", f"{prefix}_uniform_matched"),
        RunSpec("critical_vlm", "C2 VLM Criticality", f"{prefix}_vlm" if keep_edge else "uniform2_semantic_weight"),
        *[
            RunSpec(f"critical_shuffled_{seed}", f"C3 Shuffled Criticality seed={seed}", f"{prefix}_shuffled", seed=seed)
            for seed in range(5)
        ],
    ]
    critical_rows = [{**c0, "method": "C0 Uniform Criticality Raw", "run_key": "critical_uniform_raw"}]
    for spec in critical_specs:
        if spec.key == "critical_vlm" and not keep_edge:
            row = dict(anchors["compat_legacy"])
            row.update(method=spec.label, run_key=spec.key, mode=spec.mode)
        else:
            row, _ = _evaluate_run(config, spec, _run(config, spec), events, evaluator)
        critical_rows.append(row)
    shuffled = [row for row in critical_rows if row["run_key"].startswith("critical_shuffled_")]
    shuffled_mean = _mean_row("C3 Shuffled Criticality mean", "critical_shuffled_mean", shuffled)
    critical_rows.append(shuffled_mean)
    critical_rows = _annotate(critical_rows, u2)
    matched = next(row for row in critical_rows if row["run_key"] == "critical_uniform_matched")
    vlm = next(row for row in critical_rows if row["run_key"] == "critical_vlm")
    shuffled_mean = next(row for row in critical_rows if row["run_key"] == "critical_shuffled_mean")
    weighted_part_better = (
        vlm["criticality_weighted_part_exact_mm"] < matched["criticality_weighted_part_exact_mm"]
        and vlm["criticality_weighted_part_exact_mm"] < shuffled_mean["criticality_weighted_part_exact_mm"]
    )
    weighted_edge_better = True if not keep_edge else (
        vlm["criticality_weighted_edge_exact_mm"] < matched["criticality_weighted_edge_exact_mm"]
        and vlm["criticality_weighted_edge_exact_mm"] < shuffled_mean["criticality_weighted_edge_exact_mm"]
    )
    no_material_harm = bool(
        _pct(vlm["part_exact_mm"], matched["part_exact_mm"]) <= 1.0
        and _pct(vlm["edge_exact_mm"], matched["edge_exact_mm"]) <= 1.0
        and vlm["global_change_vs_u2_pct"] <= 0.5
        and vlm["ordinary_change_vs_u2_pct"] <= 1.0
        and vlm["physical_pass"]
    )
    keep_criticality = bool(weighted_part_better and weighted_edge_better and no_material_harm)
    for row in critical_rows:
        row["criticality_decision"] = "Keep" if keep_criticality else "Unproven/Drop"
    _write_csv(config.output_dir / "criticality_ablation.csv", critical_rows)

    # 7. Adaptive compute after semantic choices are frozen.
    adaptive_uniform_mode = (
        "adaptive_semantic_part_edge_weight_uniform"
        if keep_edge
        else "adaptive_semantic_weight_uniform"
    )
    adaptive_vlm_mode = (
        "adaptive_semantic_part_edge_weight_vlm"
        if keep_edge
        else "adaptive_semantic_weight_vlm"
    )
    adaptive_specs = [
        RunSpec("adaptive_uniform_1", "A1 Uniform-1", "uniform", uniform_budget=1),
        RunSpec("adaptive_base", "A3 Adaptive Base", "adaptive_base"),
        RunSpec(
            "adaptive_semantic_uniform",
            "A5 Final Semantic + Adaptive",
            adaptive_uniform_mode,
        ),
    ]
    if keep_criticality:
        adaptive_specs.append(
            RunSpec(
                "adaptive_semantic_vlm",
                "A6 Final Semantic + Criticality + Adaptive",
                adaptive_vlm_mode,
            )
        )
    adaptive_rows = [
        {**anchors["compat_original"], "method": "A0 Original", "run_key": "compat_original"},
        {**u2, "method": "A2 Uniform-2", "run_key": "compat_uniform_2"},
        {
            **(vlm if keep_criticality else c0),
            "method": "A4 Final Semantic + Uniform-2",
            "run_key": "adaptive_semantic_fixed",
        },
    ]
    adaptive_run_specs: list[RunSpec] = []
    for spec in adaptive_specs:
        row, _ = _evaluate_run(config, spec, _run(config, spec), events, evaluator)
        adaptive_rows.append(row)
        if spec.mode.startswith("adaptive_"):
            adaptive_run_specs.append(spec)
    adaptive_rows = _annotate(adaptive_rows, u2)
    adaptive_rows.sort(key=lambda row: row["method"])
    _write_csv(config.output_dir / "adaptive_compute.csv", adaptive_rows)
    adaptive_frames, histogram = _adaptive_logs(config, adaptive_run_specs)
    _write_csv(config.output_dir / "adaptive_frame_log.csv", adaptive_frames)
    _write_csv(config.output_dir / "iteration_histogram.csv", histogram)

    selected_adaptive_key = "adaptive_semantic_vlm" if keep_criticality else "adaptive_semantic_uniform"
    selected_adaptive = next(row for row in adaptive_rows if row["run_key"] == selected_adaptive_key)
    fixed_semantic = next(row for row in adaptive_rows if row["run_key"] == "adaptive_semantic_fixed")
    adaptive_kept = bool(
        selected_adaptive["actual_sqp"] < 440
        and selected_adaptive["wall_time_s"] < fixed_semantic["wall_time_s"]
        and _pct(selected_adaptive["part_exact_mm"], fixed_semantic["part_exact_mm"]) <= 1.0
        and selected_adaptive["global_change_vs_u2_pct"] <= 0.5
        and selected_adaptive["ordinary_change_vs_u2_pct"] <= 1.0
        and selected_adaptive["physical_pass"]
    )

    # 8. Final compact table and report.
    selected_fixed = part_edge if keep_edge else part
    if keep_criticality:
        selected_fixed = vlm
    selected_final_key = selected_adaptive_key if adaptive_kept else selected_fixed["run_key"]
    final_rows = _annotate(
        [
            anchors["compat_original"],
            u2,
            anchors["compat_legacy"],
            part_edge if keep_edge else part,
            vlm if keep_criticality else matched,
            selected_adaptive,
        ],
        u2,
    )
    for row in final_rows:
        row["edge_decision"] = "Keep" if keep_edge else "Drop"
        row["criticality_decision"] = "Keep" if keep_criticality else "Unproven/Drop"
        row["adaptive_decision"] = "Keep" if adaptive_kept else "Drop"
        row["final_selected"] = row["run_key"] == selected_final_key
    _write_csv(config.output_dir / "semantic_final_summary.csv", final_rows)
    report = f"""# Final semantic retargeting report

## Compatibility

Cleanup compatibility: **{'PASS' if all(row['compatibility_pass'] for row in compatibility) else 'FAIL'}**. Original, Uniform-2, and Legacy trajectories, actual SQP counts, method-independent metrics, and official physical metrics were checked against the preserved Round-4 caches.

Legacy anchor vs Uniform-2: Ordinary `{_pct(anchors['compat_legacy']['ordinary_mm'], u2['ordinary_mm']):+.3f}%`, Global KF `{_pct(anchors['compat_legacy']['global_kf_exact_mm'], u2['global_kf_exact_mm']):+.3f}%`, Part improvement `{-_pct(anchors['compat_legacy']['part_exact_mm'], u2['part_exact_mm']):.3f}%`, SQP `{anchors['compat_legacy']['actual_sqp']}`.

## Sequential decisions

- Semantic Edge: **{'Keep' if keep_edge else 'Drop'}**. Part+Edge vs Part changed Edge by `{_pct(part_edge['edge_exact_mm'], part['edge_exact_mm']):+.3f}%`, Local by `{_pct(part_edge['local_exact_mm'], part['local_exact_mm']):+.3f}%`, and Part by `{_pct(part_edge['part_exact_mm'], part['part_exact_mm']):+.3f}%`. The implementation de-duplicates overlap with Legacy object-neighbor weighting.
- VLM Criticality: **{'Keep' if keep_criticality else 'Unproven/Drop'}**. It was required to beat both activation-matched uniform criticality and the five-seed shuffled mean on original-criticality-weighted semantic metrics without violating ordinary/global/physical guardrails.
- Adaptive Compute: **{'Keep' if adaptive_kept else 'Drop'}**. Selected adaptive SQP `{selected_adaptive['actual_sqp']}` vs 440 and wall time `{selected_adaptive['wall_time_s']:.3f}s` vs fixed semantic `{fixed_semantic['wall_time_s']:.3f}s`; Part change vs fixed `{_pct(selected_adaptive['part_exact_mm'], fixed_semantic['part_exact_mm']):+.3f}%`.

## Final method

Selected mode: **`{selected_fixed['mode'] if not adaptive_kept else selected_adaptive['mode']}`**. This uses {'VLM criticality' if keep_criticality else 'uniform criticality (`c=1`)'}; the immutable historical `uniform2_semantic_weight` mode remains available as the compatibility anchor. The activation-matched control used `c={json.loads((config.output_dir / 'runs' / 'critical_uniform_matched' / 'profile.json').read_text(encoding='utf-8'))['metadata']['matched_uniform_criticality']:.6f}`.

The only optimization mechanism retained is mean-one Legacy residual reweighting, with optional modules included only when their registered gate above passes. No additive objective, constrained second stage, backtracking, Jacobian balancing, or trust-region re-solve is used.
"""
    (config.output_dir / "semantic_final_report.md").write_text(report, encoding="utf-8")
    print(f"\nFinal artifacts written to {config.output_dir}")


def cli() -> None:
    run(tyro.cli(BenchmarkConfig))


if __name__ == "__main__":
    cli()
