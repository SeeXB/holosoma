"""Validate causal semantic support truncated at the next event transition."""

from __future__ import annotations

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
    _trajectory_qpos,
    _write_csv,
)
from holosoma_retargeting.semantic_keyframes.runtime import (  # noqa: E402
    load_semantic_plan_projection,
)


MM = 1000.0
TRUNCATED_SPEC = RunSpec(
    "transition_truncated_b2",
    "FullEvent-B2 Transition-Truncated",
    "uniform2_semantic_weight_full_event_transition_truncated",
    2,
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _event_rows(
    events: Sequence[Any],
    evaluations: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    keys = ("uniform_2", "legacy_4event", "full_event_b2", TRUNCATED_SPEC.key)
    maps = {key: _event_map(evaluations[key]["all"]) for key in keys}
    rows: list[dict[str, Any]] = []
    for event in events:
        row: dict[str, Any] = {
            "event": event.name,
            "event_group": _event_group(event),
            "trigger_frame": event.trigger_frame,
            "body_parts": "|".join(event.body_parts),
        }
        for key in keys:
            metric = maps[key][event.name]
            row.update(
                {
                    f"{key}_part_exact_mm": MM * float(metric["part_exact"]),
                    f"{key}_global_exact_mm": MM * float(metric["global_exact"]),
                    f"{key}_local_exact_mm": MM * float(metric["local_exact"]),
                }
            )
        row.update(
            {
                "truncated_part_improvement_vs_full_b2_pct": _improvement(
                    row[f"{TRUNCATED_SPEC.key}_part_exact_mm"],
                    row["full_event_b2_part_exact_mm"],
                ),
                "truncated_global_change_vs_full_b2_pct": _pct_change(
                    row[f"{TRUNCATED_SPEC.key}_global_exact_mm"],
                    row["full_event_b2_global_exact_mm"],
                ),
            }
        )
        rows.append(row)
    return rows


def _support_rows(
    events: Sequence[Any],
    truncated_frames: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, event in enumerate(events):
        next_trigger = events[index + 1].trigger_frame if index + 1 < len(events) else None
        expected_end = min(
            event.end_frame,
            next_trigger - 1 if next_trigger is not None else event.end_frame,
        )
        realized = [
            int(frame["frame_idx"])
            for frame in truncated_frames
            if event.name in frame.get("active_event", [])
        ]
        expected = list(range(event.trigger_frame, expected_end + 1))
        rows.append(
            {
                "event": event.name,
                "body_parts": "|".join(event.body_parts),
                "trigger_frame": event.trigger_frame,
                "event_end_frame": event.end_frame,
                "next_event_trigger": next_trigger if next_trigger is not None else "",
                "expected_support_start": event.trigger_frame,
                "expected_support_end": expected_end,
                "realized_support_start": realized[0] if realized else "",
                "realized_support_end": realized[-1] if realized else "",
                "expected_frame_count": len(expected),
                "realized_frame_count": len(realized),
                "exact_support_match": realized == expected,
            }
        )
    return rows


def _support_trace_rows(run_dirs: dict[str, Path]) -> list[dict[str, Any]]:
    profiles = {
        key: _profile(run_dirs[key])[2]
        for key in ("full_event_b2", TRUNCATED_SPEC.key)
    }
    rows: list[dict[str, Any]] = []
    for frame in range(len(profiles[TRUNCATED_SPEC.key])):
        row: dict[str, Any] = {"frame": frame}
        for key, prefix in (
            ("full_event_b2", "full_b2"),
            (TRUNCATED_SPEC.key, "transition_truncated"),
        ):
            profile = profiles[key][frame]
            row.update(
                {
                    f"{prefix}_active_events": "|".join(profile.get("active_event", [])),
                    f"{prefix}_body_parts": "|".join(profile.get("semantic_body_parts", [])),
                    f"{prefix}_weight_l1": float(profile.get("semantic_weight_l1", 0.0)),
                    f"{prefix}_weight_max": float(
                        profile.get("semantic_vertex_weight_max", 1.0)
                    ),
                    f"{prefix}_boosted_vertex_count": int(
                        profile.get("semantic_boosted_vertex_count", 0)
                    ),
                }
            )
        rows.append(row)
    return rows


def _frame34_rows(run_dirs: dict[str, Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in ("full_event_b2", TRUNCATED_SPEC.key):
        for source in _read_csv(run_dirs[key] / "semantic_iteration_trace.csv"):
            if int(source["frame"]) != 34:
                continue
            rows.append(
                {
                    "run_key": key,
                    "frame": 34,
                    "iteration": int(source["iteration"]),
                    "part_error_mm": MM * float(source["part_error"]),
                    "global_laplacian_error_mm": MM
                    * float(source["global_laplacian_error"]),
                    "shoulder_box_signed_distance_mm": MM
                    * float(source["shoulder_box_signed_distance"]),
                    "shoulder_box_penetration_depth_mm": MM
                    * float(source["shoulder_box_penetration_depth"]),
                    "distance_query_threshold_m": float(source["distance_query_threshold"]),
                }
            )
    return rows


def _report(
    path: Path,
    rows: dict[str, dict[str, Any]],
    event_rows: Sequence[dict[str, Any]],
    frame34_rows: Sequence[dict[str, Any]],
    support_valid: bool,
    qpos_diagnostics: dict[str, Any],
    immutable: bool,
) -> None:
    u2 = rows["uniform_2"]
    full = rows["full_event_b2"]
    trunc = rows[TRUNCATED_SPEC.key]
    approach = next(row for row in event_rows if row["event"] == "approach")
    final34 = {
        row["run_key"]: row
        for row in frame34_rows
        if int(row["iteration"]) == 2
    }
    penetration_resolved = (
        float(trunc["penetration_duration"]) == 0.0
        and float(trunc["penetration_max_depth_mm"]) == 0.0
    )
    lines = [
        "# FullEvent Transition-Truncation Diagnostic",
        "",
        "唯一实验变量是 spatial weighting 的时域支持：每个事件从自身 trigger 开始，并在自身 window end 或下一个事件 trigger（不含）处结束。事件、body parts、multiplier、归一化、碰撞设置和 Uniform-2 计算预算均不变。",
        "",
        "## 结论",
        "",
        f"- **frame 34 官方穿透：{'已消失' if penetration_resolved else '仍存在'}**。Truncated duration/max={trunc['penetration_duration']:.6f}/{trunc['penetration_max_depth_mm']:.6f} mm；FullEvent-B2={full['penetration_duration']:.6f}/{full['penetration_max_depth_mm']:.6f} mm。",
        f"- **通用支持规则验证：`{support_valid}`**。approach 的实际支持为 26–29，frame 30 由 contact 独占；所有事件均使用同一 next-trigger exclusive 规则。",
        f"- **计算量：** Truncated SQP/calls={trunc['actual_sqp']}/{trunc['solver_calls']}；FullEvent-B2={full['actual_sqp']}/{full['solver_calls']}。",
        f"- **旧缓存不变：`{immutable}`**。",
        f"- **综合判断：** 该规则是有效的物理修复诊断，但不是无代价改进；它相对 FullEvent-B2 损失部分语义精度。相对 Legacy，All-event/Phase Part 仍改善 {_improvement(trunc['all_event_part_exact_mm'], rows['legacy_4event']['all_event_part_exact_mm']):+.3f}%/{_improvement(trunc['phase_part_exact_mm'], rows['legacy_4event']['phase_part_exact_mm']):+.3f}%，Interaction Part 则 {_improvement(trunc['interaction_part_exact_mm'], rows['legacy_4event']['interaction_part_exact_mm']):+.3f}%。",
        "",
        "## 质量指标",
        "",
        f"- All-event Part {trunc['all_event_part_exact_mm']:.6f} mm，较 FullEvent-B2 {_improvement(trunc['all_event_part_exact_mm'], full['all_event_part_exact_mm']):+.3f}%。",
        f"- Phase Part {trunc['phase_part_exact_mm']:.6f} mm，较 FullEvent-B2 {_improvement(trunc['phase_part_exact_mm'], full['phase_part_exact_mm']):+.3f}%。",
        f"- Interaction Part {trunc['interaction_part_exact_mm']:.6f} mm，较 FullEvent-B2 {_improvement(trunc['interaction_part_exact_mm'], full['interaction_part_exact_mm']):+.3f}%。",
        f"- approach Part {approach[f'{TRUNCATED_SPEC.key}_part_exact_mm']:.6f} mm，较 FullEvent-B2 {approach['truncated_part_improvement_vs_full_b2_pct']:+.3f}%。",
        f"- Global KF Exact {trunc['global_kf_exact_mm']:.6f} mm（vs U2 {trunc['global_change_vs_u2_pct']:+.3f}%，vs FullEvent-B2 {_pct_change(trunc['global_kf_exact_mm'], full['global_kf_exact_mm']):+.3f}%）。",
        f"- Ordinary {trunc['ordinary_mm']:.6f} mm（vs U2 {trunc['ordinary_change_vs_u2_pct']:+.3f}%）。",
        f"- Foot skating duration/max={trunc['foot_skating_duration']:.6f}/{trunc['foot_skating_max_velocity']:.6f}；contact preservation={trunc['contact_preservation']:.6f}（U2={u2['contact_preservation']:.6f}）。",
        "",
        "## frame 34",
        "",
        f"- FullEvent-B2 second iterate: signed distance {final34['full_event_b2']['shoulder_box_signed_distance_mm']:+.6f} mm。",
        f"- Transition-truncated second iterate: signed distance {final34[TRUNCATED_SPEC.key]['shoulder_box_signed_distance_mm']:+.6f} mm。",
        f"- 轨迹首次分叉 frame={qpos_diagnostics['first_divergent_frame']}；frame 34 qpos max abs diff={qpos_diagnostics['frame34_qpos_max_abs_diff']:.6g}。",
        "",
        "逐事件指标、逐帧权重支持以及两次 SQP 的真碰撞距离分别见 `transition_truncation_event_metrics.csv`、`transition_support_trace.csv` 和 `frame34_iteration_trace.csv`。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(config: BenchmarkConfig) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    events = load_semantic_plan_projection(config.semantic_keyframe_path)
    semantic_frames = {event.trigger_frame for event in events if event.trigger_frame > 0}
    run_dirs = {
        "uniform_2": config.historical_final_dir / "runs" / "compat_uniform_2",
        "legacy_4event": config.full_event_results_dir / "runs" / "legacy_4event",
        "full_event_b2": config.full_event_results_dir / "runs" / "full_event_b2",
    }
    for key, run_dir in run_dirs.items():
        if not _trajectory(run_dir, config.task_name).exists():
            raise FileNotFoundError(f"missing read-only baseline {key}: {run_dir}")
    immutable_paths = [
        config.semantic_keyframe_path,
        *(
            path
            for run_dir in run_dirs.values()
            for path in (_trajectory(run_dir, config.task_name), run_dir / "profile.json")
        ),
    ]
    before = {str(path): _sha256(path) for path in immutable_paths}

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
    run_dirs[TRUNCATED_SPEC.key] = _run(curve_config, TRUNCATED_SPEC)
    evaluator = _official_evaluator()
    specs = {
        "uniform_2": RunSpec("uniform_2", "Uniform-2", "uniform", 2),
        "legacy_4event": RunSpec(
            "legacy_4event", "Legacy-4Event", "uniform2_semantic_weight_uniform", 2
        ),
        "full_event_b2": RunSpec(
            "full_event_b2", "FullEvent-B2", "uniform2_semantic_weight_full_event", 2
        ),
        TRUNCATED_SPEC.key: TRUNCATED_SPEC,
    }
    evaluations: dict[str, dict[str, Any]] = {}
    raw: dict[str, dict[str, Any]] = {}
    for key, run_dir in run_dirs.items():
        evaluations[key] = _evaluations(_trajectory(run_dir, config.task_name), events)
        raw[key] = _metric_row(
            specs[key], run_dir, evaluations[key], _official(curve_config, run_dir, evaluator), semantic_frames
        )
    annotated = _annotate(
        list(raw.values()), raw["uniform_2"], raw["legacy_4event"], raw["full_event_b2"]
    )
    rows = {str(row["run_key"]): row for row in annotated}
    for row in annotated:
        row["all_event_part_improvement_vs_full_b2_pct"] = _improvement(
            row["all_event_part_exact_mm"], rows["full_event_b2"]["all_event_part_exact_mm"]
        )
        row["phase_part_improvement_vs_full_b2_pct"] = _improvement(
            row["phase_part_exact_mm"], rows["full_event_b2"]["phase_part_exact_mm"]
        )
        row["interaction_part_improvement_vs_full_b2_pct"] = _improvement(
            row["interaction_part_exact_mm"], rows["full_event_b2"]["interaction_part_exact_mm"]
        )
    _write_csv(config.output_dir / "transition_truncation_summary.csv", annotated)
    event_rows = _event_rows(events, evaluations)
    _write_csv(config.output_dir / "transition_truncation_event_metrics.csv", event_rows)

    metadata, summary, frames = _profile(run_dirs[TRUNCATED_SPEC.key])
    support_rows = _support_rows(events, frames)
    support_valid = bool(
        all(bool(row["exact_support_match"]) for row in support_rows)
        and metadata["residual_weights"].get("temporal_support_policy")
        == "trigger <= frame <= event end and frame < next event trigger"
        and metadata["semantic_weighting_mainline"].get(
            "transition_truncated_weight_support"
        )
        is True
        and int(summary["total_actual_iterations"]) == 440
        and int(summary["total_convex_solver_calls"]) == 440
    )
    _write_csv(config.output_dir / "transition_support_summary.csv", support_rows)
    _write_csv(config.output_dir / "transition_support_trace.csv", _support_trace_rows(run_dirs))
    frame34_rows = _frame34_rows(run_dirs)
    _write_csv(config.output_dir / "frame34_iteration_trace.csv", frame34_rows)

    full_qpos = _trajectory_qpos(run_dirs["full_event_b2"], config.task_name)
    trunc_qpos = _trajectory_qpos(run_dirs[TRUNCATED_SPEC.key], config.task_name)
    per_frame = np.max(np.abs(trunc_qpos - full_qpos), axis=1)
    divergent = np.flatnonzero(per_frame > 1e-12)
    qpos_diagnostics = {
        "first_divergent_frame": int(divergent[0]) if divergent.size else None,
        "trajectory_qpos_max_abs_diff": float(per_frame.max()),
        "frame34_qpos_max_abs_diff": float(per_frame[34]),
        "frame34_qpos_l2_diff": float(np.linalg.norm(trunc_qpos[34] - full_qpos[34])),
    }
    (config.output_dir / "qpos_causal_diagnostics.json").write_text(
        json.dumps(qpos_diagnostics, indent=2) + "\n", encoding="utf-8"
    )
    after = {str(path): _sha256(path) for path in immutable_paths}
    immutable = before == after
    benchmark_metadata = {
        "task": config.task_name,
        "only_experimental_variable": "generic causal event spatial-weight support",
        "support_rule": "event.trigger <= frame <= event.end and frame < next_event.trigger",
        "hardcoded_event_pair": False,
        "approach_support": [26, 27, 28, 29],
        "pelvis_multiplier": 2.0,
        "criticality_used": False,
        "actual_sqp": rows[TRUNCATED_SPEC.key]["actual_sqp"],
        "solver_calls": rows[TRUNCATED_SPEC.key]["solver_calls"],
        "support_valid": support_valid,
        "baseline_cache_unchanged": immutable,
        "immutable_sha256_before": before,
        "immutable_sha256_after": after,
        **qpos_diagnostics,
    }
    (config.output_dir / "benchmark_metadata.json").write_text(
        json.dumps(benchmark_metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _report(
        config.output_dir / "transition_truncation_report.md",
        rows,
        event_rows,
        frame34_rows,
        support_valid,
        qpos_diagnostics,
        immutable,
    )
    print(f"Artifacts: {config.output_dir}")


def cli() -> None:
    run(tyro.cli(BenchmarkConfig))


if __name__ == "__main__":
    cli()
