"""B2/B4/B6/B8/B10 curve for transition-truncated FullEvent weighting."""

from __future__ import annotations

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

from holosoma_retargeting.examples.benchmark_full_event_semantic_budget import (  # noqa: E402
    BenchmarkConfig as CurveBenchmarkConfig,
    RunSpec,
    _annotate,
    _evaluations,
    _event_group,
    _event_map,
    _improvement,
    _metric_row,
    _official,
    _official_evaluator,
    _pct_change,
    _profile,
    _read_csv,
    _run,
    _trajectory,
    _write_csv,
)
from holosoma_retargeting.semantic_keyframes.runtime import (  # noqa: E402
    load_semantic_plan_projection,
)


MM = 1000.0
BUDGETS = (2, 4, 6, 8, 10)
CURVE_SPECS = tuple(
    RunSpec(
        f"transition_truncated_b{budget}",
        f"Transition-Truncated-B{budget}",
        (
            "uniform2_semantic_weight_full_event_transition_truncated"
            if budget == 2
            else "uniform2_semantic_weight_full_event_transition_truncated_budget"
        ),
        budget,
    )
    for budget in BUDGETS
)


@dataclass
class BenchmarkConfig:
    task_name: str = "sub3_largebox_003"
    data_path: Path = Path("demo_data/OMOMO_new")
    semantic_keyframe_path: Path = Path(
        "demo_data/semantic_keyframes/sub3_largebox_003_semantic_v2.json"
    )
    output_dir: Path = Path("benchmark_results_full_event_transition_truncation")
    historical_final_dir: Path = Path("benchmark_results_semantic_final")
    full_event_results_dir: Path = Path("benchmark_results_full_event_semantic_budget")
    force: bool = False
    aggregate_only: bool = False
    refresh_official: bool = False


def _event_rows(
    events: Sequence[Any],
    evaluations: dict[str, dict[str, Any]],
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
                    "run_key": spec.key,
                    "trigger_budget": spec.budget,
                    "event": event.name,
                    "event_group": _event_group(event),
                    "trigger_frame": event.trigger_frame,
                    "body_parts": "|".join(event.body_parts),
                    "actual_sqp_at_trigger": int(frame["actual_sqp_iterations"]),
                    "solver_calls_at_trigger": int(frame["convex_solver_calls"]),
                    "part_exact_mm": MM * float(metric["part_exact"]),
                    "global_exact_mm": MM * float(metric["global_exact"]),
                    "local_exact_mm": MM * float(metric["local_exact"]),
                }
            )
    by_event: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_event.setdefault(str(row["event"]), []).append(row)
    for event_rows in by_event.values():
        event_rows.sort(key=lambda row: int(row["trigger_budget"]))
        base = float(event_rows[0]["part_exact_mm"])
        best = min(float(row["part_exact_mm"]) for row in event_rows)
        for row in event_rows:
            row["part_improvement_vs_b2_pct"] = _improvement(
                row["part_exact_mm"], base
            )
            row["within_0_1pct_of_event_best"] = (
                float(row["part_exact_mm"]) <= best * 1.001
            )
    return rows


def _trace_rows(run_dirs: dict[str, Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in CURVE_SPECS:
        for source in _read_csv(run_dirs[spec.key] / "semantic_iteration_trace.csv"):
            rows.append(
                {
                    "run_key": spec.key,
                    "trigger_budget": spec.budget,
                    "frame": int(source["frame"]),
                    "event": source["event"],
                    "trace_kind": source["trace_kind"],
                    "iteration": int(source["iteration"]),
                    "part_error_mm": MM * float(source["part_error"]),
                    "global_laplacian_error_mm": MM
                    * float(source["global_laplacian_error"]),
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
                }
            )
    return rows


def _registered_feasible(row: dict[str, Any], u2: dict[str, Any]) -> bool:
    tolerance = 1e-12
    return bool(
        float(row["penetration_duration"]) == 0.0
        and float(row["penetration_max_depth_mm"]) == 0.0
        and float(row["foot_skating_duration"])
        <= float(u2["foot_skating_duration"]) + tolerance
        and float(row["foot_skating_max_velocity"])
        <= float(u2["foot_skating_max_velocity"]) + tolerance
        and float(row["contact_preservation"]) + tolerance
        >= float(u2["contact_preservation"])
        and float(row["joint_limit_max_violation"]) <= 1e-6
        and float(row["ordinary_change_vs_u2_pct"]) <= 1.0
        and float(row["global_change_vs_u2_pct"]) <= 0.5
        and float(row["all_event_part_improvement_vs_u2_pct"]) > 0.0
    )


def _select_final(rows: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    feasible = [row for row in rows if bool(row["registered_feasible"])]
    if not feasible:
        return None
    best_part = min(float(row["all_event_part_exact_mm"]) for row in feasible)
    saturated = [
        row
        for row in feasible
        if float(row["all_event_part_exact_mm"]) <= best_part * 1.001
    ]
    return min(
        saturated,
        key=lambda row: (int(row["actual_sqp"]), float(row["all_event_part_exact_mm"])),
    )


def _pareto(rows: Sequence[dict[str, Any]]) -> None:
    feasible = [row for row in rows if bool(row["registered_feasible"])]
    for row in rows:
        row["compute_part_pareto"] = bool(
            row in feasible
            and not any(
                int(other["actual_sqp"]) <= int(row["actual_sqp"])
                and float(other["all_event_part_exact_mm"])
                <= float(row["all_event_part_exact_mm"])
                and (
                    int(other["actual_sqp"]) < int(row["actual_sqp"])
                    or float(other["all_event_part_exact_mm"])
                    < float(row["all_event_part_exact_mm"])
                )
                for other in feasible
            )
        )


def _plots(
    output_dir: Path,
    rows: Sequence[dict[str, Any]],
    event_rows: Sequence[dict[str, Any]],
) -> None:
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    plot_dir = output_dir / "plots_transition_budget"
    plot_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda row: int(row["trigger_budget"]))

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(
        [int(row["actual_sqp"]) for row in ordered],
        [float(row["all_event_part_exact_mm"]) for row in ordered],
        marker="o",
    )
    for row in ordered:
        ax.annotate(
            f"B{row['trigger_budget']}",
            (float(row["actual_sqp"]), float(row["all_event_part_exact_mm"])),
            xytext=(5, 4),
            textcoords="offset points",
        )
    ax.set(xlabel="Actual SQP", ylabel="All-event Part Exact (mm)", title="Truncated budget curve")
    fig.tight_layout()
    fig.savefig(plot_dir / "compute_vs_all_event_part.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for event in ("approach", "contact", "lift", "carry_mid", "arrive", "place", "release"):
        selected = sorted(
            [row for row in event_rows if row["event"] == event],
            key=lambda row: int(row["trigger_budget"]),
        )
        ax.plot(
            [int(row["trigger_budget"]) for row in selected],
            [float(row["part_exact_mm"]) for row in selected],
            marker="o",
            label=event,
        )
    ax.set(xlabel="Exact-trigger max SQP", ylabel="Part Exact (mm)", title="Per-event budget response", xticks=BUDGETS)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(plot_dir / "event_part_budget_curve.png", dpi=180)
    plt.close(fig)


def _report(
    path: Path,
    rows: Sequence[dict[str, Any]],
    selected: dict[str, Any] | None,
    event_rows: Sequence[dict[str, Any]],
) -> None:
    ordered = sorted(rows, key=lambda row: int(row["trigger_budget"]))
    table = [
        "| Budget | SQP | All Part | Phase Part | Interaction Part | Global | Ordinary | Official penetration | Frame34 clearance | Feasible |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in ordered:
        table.append(
            f"| B{row['trigger_budget']} | {row['actual_sqp']} | {row['all_event_part_exact_mm']:.6f} | "
            f"{row['phase_part_exact_mm']:.6f} | {row['interaction_part_exact_mm']:.6f} | "
            f"{row['global_kf_exact_mm']:.6f} | {row['ordinary_mm']:.6f} | "
            f"{row['penetration_max_depth_mm']:.6f} | "
            f"{row['frame34_final_signed_distance_mm']:+.6f} | {row['registered_feasible']} |"
        )
    if selected is None:
        decision = "没有预算通过预注册 guardrail，因此不选择最终方法。"
    else:
        decision = (
            f"选择 **Transition-Truncated-B{selected['trigger_budget']}**：它通过全部 guardrail，"
            "并且是在 All-event Part 距离可行最优值 0.1% 内计算量最低的预算。"
        )
    event_best = []
    for event in ("approach", "contact", "lift", "carry_mid", "arrive", "place", "release"):
        candidates = [row for row in event_rows if row["event"] == event]
        best = min(candidates, key=lambda row: float(row["part_exact_mm"]))
        event_best.append(f"{event}=B{best['trigger_budget']}")
    lines = [
        "# Transition-Truncated Semantic Budget Curve",
        "",
        "唯一变量是七个非零 semantic trigger 的最大串行 SQP 次数 B2/B4/B6/B8/B10。Transition truncation、事件、body parts、multipliers、frame0=50、ordinary=2 和物理设置全部固定。",
        "",
        "## 最终选择",
        "",
        decision,
        "",
        "选择规则：先要求零官方穿透、足部与 contact 不劣于 U2、joint limit 合法、Ordinary ≤ U2+1%、Global ≤ U2+0.5%、All-event Part 优于 U2；然后在 All-event Part 距离可行最优值 0.1% 内选择 Actual SQP 最低者。",
        "",
        "## 主结果（误差单位 mm，越低越好）",
        "",
        *table,
        "",
        "逐事件最低 Part 所在预算：" + ", ".join(event_best) + "。",
        "",
        "完整逐事件曲线、每次实际迭代和最终机器可读选择分别见 `transition_budget_event_metrics.csv`、`transition_budget_iteration_trace.csv` 和 `final_method.json`。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(config: BenchmarkConfig) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    events = load_semantic_plan_projection(config.semantic_keyframe_path)
    semantic_frames = {event.trigger_frame for event in events if event.trigger_frame > 0}
    curve_config = CurveBenchmarkConfig(
        task_name=config.task_name,
        data_path=config.data_path,
        semantic_keyframe_path=config.semantic_keyframe_path,
        output_dir=config.output_dir,
        historical_final_dir=config.historical_final_dir,
        force=config.force,
        aggregate_only=config.aggregate_only,
        refresh_official=config.refresh_official,
    )
    run_dirs = {
        "uniform_2": config.historical_final_dir / "runs" / "compat_uniform_2",
        "legacy_4event": config.full_event_results_dir / "runs" / "legacy_4event",
    }
    for spec in CURVE_SPECS:
        run_dirs[spec.key] = _run(curve_config, spec)
    evaluator = _official_evaluator()
    specs = {
        "uniform_2": RunSpec("uniform_2", "Uniform-2", "uniform", 2),
        "legacy_4event": RunSpec(
            "legacy_4event", "Legacy-4Event", "uniform2_semantic_weight_uniform", 2
        ),
        **{spec.key: spec for spec in CURVE_SPECS},
    }
    evaluations: dict[str, dict[str, Any]] = {}
    raw: dict[str, dict[str, Any]] = {}
    for key, run_dir in run_dirs.items():
        evaluations[key] = _evaluations(_trajectory(run_dir, config.task_name), events)
        raw[key] = _metric_row(
            specs[key], run_dir, evaluations[key], _official(curve_config, run_dir, evaluator), semantic_frames
        )
    annotated = _annotate(
        list(raw.values()),
        raw["uniform_2"],
        raw["legacy_4event"],
        raw[CURVE_SPECS[0].key],
    )
    by_key = {str(row["run_key"]): row for row in annotated}
    curve_rows = [dict(by_key[spec.key]) for spec in CURVE_SPECS]
    u2 = by_key["uniform_2"]
    for row, spec in zip(curve_rows, CURVE_SPECS):
        _, profile_summary, _ = _profile(run_dirs[spec.key])
        row.update(
            {
                "joint_limit_max_violation": float(
                    profile_summary.get("max_joint_limit_violation") or 0.0
                ),
                "self_collision_max_violation": (
                    float(profile_summary["max_self_collision_violation"])
                    if profile_summary.get("max_self_collision_violation") is not None
                    else np.nan
                ),
            }
        )
        row["registered_feasible"] = _registered_feasible(row, u2)
    _pareto(curve_rows)
    selected = _select_final(curve_rows)
    for row in curve_rows:
        row["selected_final"] = bool(
            selected is not None and row["run_key"] == selected["run_key"]
        )

    event_rows = _event_rows(events, evaluations, run_dirs)
    trace_rows = _trace_rows(run_dirs)
    for row, spec in zip(curve_rows, CURVE_SPECS):
        frame34 = [
            trace
            for trace in trace_rows
            if trace["run_key"] == spec.key and int(trace["frame"]) == 34
        ]
        final_frame34 = max(frame34, key=lambda trace: int(trace["iteration"]))
        row["frame34_final_signed_distance_mm"] = float(
            final_frame34["shoulder_box_signed_distance_mm"]
        )
        row["frame34_final_penetration_depth_mm"] = float(
            final_frame34["shoulder_box_penetration_depth_mm"]
        )
    _write_csv(config.output_dir / "transition_budget_curve_summary.csv", curve_rows)
    _write_csv(config.output_dir / "transition_budget_event_metrics.csv", event_rows)
    _write_csv(config.output_dir / "transition_budget_iteration_trace.csv", trace_rows)
    _write_csv(
        config.output_dir / "transition_budget_frame34_trace.csv",
        [row for row in trace_rows if int(row["frame"]) == 34],
    )
    final_payload = {
        "selection_rule": (
            "minimum Actual SQP among registered-feasible methods within 0.1% "
            "of the feasible best All-event Part Exact"
        ),
        "selected": (
            {
                "run_key": selected["run_key"],
                "mode": selected["mode"],
                "exact_trigger_budget": int(selected["trigger_budget"]),
                "actual_sqp": int(selected["actual_sqp"]),
                "solver_calls": int(selected["solver_calls"]),
                "all_event_part_exact_mm": float(selected["all_event_part_exact_mm"]),
                "phase_part_exact_mm": float(selected["phase_part_exact_mm"]),
                "interaction_part_exact_mm": float(selected["interaction_part_exact_mm"]),
                "global_kf_exact_mm": float(selected["global_kf_exact_mm"]),
                "ordinary_mm": float(selected["ordinary_mm"]),
                "penetration_max_depth_mm": float(selected["penetration_max_depth_mm"]),
                "frame34_final_signed_distance_mm": float(
                    selected["frame34_final_signed_distance_mm"]
                ),
            }
            if selected is not None
            else None
        ),
        "budgets": list(BUDGETS),
        "all_runs_actual_sqp_equals_solver_calls": all(
            int(row["actual_sqp"]) == int(row["solver_calls"]) for row in curve_rows
        ),
        "all_runs_transition_support_fixed": True,
        "criticality_used": False,
    }
    (config.output_dir / "final_method.json").write_text(
        json.dumps(final_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _plots(config.output_dir, curve_rows, event_rows)
    _report(
        config.output_dir / "transition_budget_curve_report.md",
        curve_rows,
        selected,
        event_rows,
    )
    print(
        "Selected final method: "
        + (str(selected["run_key"]) if selected is not None else "none")
    )
    print(f"Artifacts: {config.output_dir}")


def cli() -> None:
    run(tyro.cli(BenchmarkConfig))


if __name__ == "__main__":
    cli()
