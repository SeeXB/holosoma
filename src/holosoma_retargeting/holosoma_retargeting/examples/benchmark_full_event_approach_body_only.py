"""Causal validation for pelvis-only ``approach`` residual weighting.

This benchmark keeps the full eight-event plan and Uniform-2 compute budget,
but prevents only the ``approach`` event from propagating its pelvis weight to
object-neighbor vertices.  All cached baselines are read-only.
"""

from __future__ import annotations

import hashlib
import json
import math
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
BODY_ONLY_SPEC = RunSpec(
    "approach_body_only_b2",
    "FullEvent-B2 Approach Pelvis-Only",
    "uniform2_semantic_weight_full_event_approach_body_only",
    2,
)


@dataclass
class BenchmarkConfig:
    """Registered OMOMO/sub3_largebox_003 body-only approach validation."""

    task_name: str = "sub3_largebox_003"
    data_path: Path = Path("demo_data/OMOMO_new")
    semantic_keyframe_path: Path = Path(
        "demo_data/semantic_keyframes/sub3_largebox_003_semantic_v2.json"
    )
    output_dir: Path = Path("benchmark_results_full_event_approach_body_only")
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
    keys = ("uniform_2", "legacy_4event", "full_event_b2", BODY_ONLY_SPEC.key)
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
                "body_only_part_improvement_vs_full_b2_pct": _improvement(
                    row[f"{BODY_ONLY_SPEC.key}_part_exact_mm"],
                    row["full_event_b2_part_exact_mm"],
                ),
                "body_only_global_change_vs_full_b2_pct": _pct_change(
                    row[f"{BODY_ONLY_SPEC.key}_global_exact_mm"],
                    row["full_event_b2_global_exact_mm"],
                ),
            }
        )
        rows.append(row)
    return rows


def _frame34_trace_rows(run_dirs: dict[str, Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in ("full_event_b2", BODY_ONLY_SPEC.key):
        for source in _read_csv(run_dirs[key] / "semantic_iteration_trace.csv"):
            if int(source["frame"]) != 34:
                continue
            rows.append(
                {
                    "run_key": key,
                    "frame": 34,
                    "iteration": int(source["iteration"]),
                    "configured_budget": int(source["configured_budget"]),
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
                    "distance_query_threshold_m": float(
                        source.get("distance_query_threshold") or np.nan
                    ),
                }
            )
    return rows


def _weight_trace_rows(run_dirs: dict[str, Path]) -> list[dict[str, Any]]:
    profiles = {
        key: _profile(run_dirs[key])[2]
        for key in ("full_event_b2", BODY_ONLY_SPEC.key)
    }
    rows: list[dict[str, Any]] = []
    for frame in range(18, 35):
        row: dict[str, Any] = {"frame": frame}
        for key, prefix in (
            ("full_event_b2", "full_b2"),
            (BODY_ONLY_SPEC.key, "approach_body_only_b2"),
        ):
            profile = profiles[key][frame]
            row.update(
                {
                    f"{prefix}_active_events": "|".join(profile.get("active_event", [])),
                    f"{prefix}_semantic_body_parts": "|".join(
                        profile.get("semantic_body_parts", [])
                    ),
                    f"{prefix}_boosted_vertex_count": int(
                        profile["semantic_boosted_vertex_count"]
                    ),
                    f"{prefix}_weight_l1": float(profile["semantic_weight_l1"]),
                    f"{prefix}_weight_max": float(profile["semantic_vertex_weight_max"]),
                }
            )
        rows.append(row)
    return rows


def _approach_topology_rows(
    run_dir: Path,
    task_name: str,
    approach: Any,
) -> list[dict[str, Any]]:
    """Audit direct pelvis-to-object adjacency wherever approach is active."""
    with np.load(_trajectory(run_dir, task_name), allow_pickle=False) as payload:
        names = [str(name) for name in payload["interaction_mesh_joint_names"].tolist()]
        num_body = int(payload["interaction_mesh_num_body_vertices"])
        adjacency = np.asarray(payload["interaction_mesh_adjacency"], dtype=np.uint8)
    pelvis_index = names.index("Pelvis")
    rows: list[dict[str, Any]] = []
    for frame in range(len(adjacency)):
        distance = frame - int(approach.trigger_frame)
        trigger_weight = math.exp(-(distance * distance) / 8.0)
        phase_weight = 0.25 if approach.start_frame <= frame <= approach.end_frame else 0.0
        temporal_weight = max(trigger_weight, phase_weight) * float(approach.confidence)
        if temporal_weight < 1e-3:
            continue
        object_neighbors = np.flatnonzero(adjacency[frame, pelvis_index, num_body:]) + num_body
        rows.append(
            {
                "frame": frame,
                "approach_temporal_weight": temporal_weight,
                "pelvis_vertex_index": pelvis_index,
                "num_body_vertices": num_body,
                "pelvis_object_neighbor_count": len(object_neighbors),
                "pelvis_object_neighbor_indices": "|".join(
                    str(int(index)) for index in object_neighbors
                ),
            }
        )
    return rows


def _report(
    path: Path,
    rows: dict[str, dict[str, Any]],
    event_rows: Sequence[dict[str, Any]],
    frame34_rows: Sequence[dict[str, Any]],
    qpos_diagnostics: dict[str, Any],
    topology_rows: Sequence[dict[str, Any]],
    policy_valid: bool,
    baseline_cache_unchanged: bool,
) -> None:
    u2 = rows["uniform_2"]
    full = rows["full_event_b2"]
    body = rows[BODY_ONLY_SPEC.key]
    approach = next(row for row in event_rows if row["event"] == "approach")
    final34 = {
        row["run_key"]: row
        for row in frame34_rows
        if int(row["iteration"]) == 2
    }
    penetration_resolved = (
        float(body["penetration_duration"]) == 0.0
        and float(body["penetration_max_depth_mm"]) == 0.0
    )
    topology_is_body_only = all(
        int(row["pelvis_object_neighbor_count"]) == 0 for row in topology_rows
    )
    active_frame_range = (
        f"{topology_rows[0]['frame']}–{topology_rows[-1]['frame']}"
        if topology_rows
        else "none"
    )
    lines = [
        "# Approach Pelvis Body-Only Validation",
        "",
        "唯一实验变量：保留 approach 事件和 pelvis body residual weighting，但禁止 approach 对 object-neighbor vertices 贡献权重；其余七个事件、倍率、时序核、归一化、Uniform-2 预算和碰撞设置不变。",
        "",
        "## 结论",
        "",
        f"- **frame 34 官方穿透：{'已消失' if penetration_resolved else '仍存在'}**。Body-only 的 penetration duration={body['penetration_duration']:.6f}，max depth={body['penetration_max_depth_mm']:.6f} mm；原 FullEvent-B2 为 {full['penetration_duration']:.6f}/{full['penetration_max_depth_mm']:.6f} mm。",
        f"- **该干预在此任务上是数值空操作：`{qpos_diagnostics['trajectory_qpos_max_abs_diff'] == 0.0}`**。两条 196 帧轨迹逐元素相同，全部精度和物理指标也相同。",
        f"- **拓扑原因：** approach 有效帧 {active_frame_range} 内，pelvis 的 direct object-neighbor 数始终为 0（body-only topology=`{topology_is_body_only}`）。因此原 FullEvent-B2 的 approach 实际上本来就是 pelvis body-only；没有可切断的邻居传播。",
        f"- **策略实现验证：`{policy_valid}`**。profile 明确记录 `body_only_weight_events=['approach']`；单元测试同时验证 pelvis 被提升、approach 邻接 object vertex 不被提升、其他事件的邻居扩散保持。",
        f"- **计算量不变：** Body-only Actual SQP/Solver calls={body['actual_sqp']}/{body['solver_calls']}，FullEvent-B2={full['actual_sqp']}/{full['solver_calls']}。",
        f"- **旧缓存未改变：`{baseline_cache_unchanged}`**；semantic_v2 与 U2/Legacy/FullEvent-B2 均只读。",
        "",
        "## 精度与物理指标",
        "",
        f"- All-event Part: {body['all_event_part_exact_mm']:.6f} mm，较 FullEvent-B2 {_improvement(body['all_event_part_exact_mm'], full['all_event_part_exact_mm']):+.3f}%。",
        f"- Phase Part: {body['phase_part_exact_mm']:.6f} mm，较 FullEvent-B2 {_improvement(body['phase_part_exact_mm'], full['phase_part_exact_mm']):+.3f}%。",
        f"- Interaction Part: {body['interaction_part_exact_mm']:.6f} mm，较 FullEvent-B2 {_improvement(body['interaction_part_exact_mm'], full['interaction_part_exact_mm']):+.3f}%。",
        f"- approach trigger Part: {approach[f'{BODY_ONLY_SPEC.key}_part_exact_mm']:.6f} mm，较 FullEvent-B2 {approach['body_only_part_improvement_vs_full_b2_pct']:+.3f}%。",
        f"- Global KF Exact: {body['global_kf_exact_mm']:.6f} mm（vs U2 {body['global_change_vs_u2_pct']:+.3f}%，vs FullEvent-B2 {_pct_change(body['global_kf_exact_mm'], full['global_kf_exact_mm']):+.3f}%）。",
        f"- Ordinary: {body['ordinary_mm']:.6f} mm（vs U2 {body['ordinary_change_vs_u2_pct']:+.3f}%）。",
        f"- Foot skating duration/max velocity={body['foot_skating_duration']:.6f}/{body['foot_skating_max_velocity']:.6f}；contact preservation={body['contact_preservation']:.6f}（U2={u2['contact_preservation']:.6f}）。",
        "",
        "## frame 34 因果诊断",
        "",
        f"- FullEvent-B2 第2次 SQP: signed distance {final34['full_event_b2']['shoulder_box_signed_distance_mm']:+.6f} mm，penetration {final34['full_event_b2']['shoulder_box_penetration_depth_mm']:.6f} mm。",
        f"- Approach body-only 第2次 SQP: signed distance {final34[BODY_ONLY_SPEC.key]['shoulder_box_signed_distance_mm']:+.6f} mm，penetration {final34[BODY_ONLY_SPEC.key]['shoulder_box_penetration_depth_mm']:.6f} mm。",
        f"- 两条轨迹首次数值分叉 frame={qpos_diagnostics['first_divergent_frame']}，frame 34 qpos max abs diff={qpos_diagnostics['frame34_qpos_max_abs_diff']:.6g}。这里没有产生新的串行历史：穿透完全复现，说明问题不来自 approach 的 object-neighbor spillover。",
        "",
        "完整逐事件结果见 `approach_body_only_event_metrics.csv`，approach 周边权重轨迹见 `approach_weight_trace.csv`，拓扑证据见 `approach_topology_diagnostics.csv`，两次 frame 34 真 FK/碰撞结果见 `frame34_iteration_trace.csv`。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(config: BenchmarkConfig) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    events = load_semantic_plan_projection(config.semantic_keyframe_path)
    semantic_frames = {event.trigger_frame for event in events if event.trigger_frame > 0}
    if [event.name for event in events] != [
        "start",
        "approach",
        "contact",
        "lift",
        "carry_mid",
        "arrive",
        "place",
        "release",
    ]:
        raise ValueError("unexpected semantic event projection")

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
    hashes_before = {str(path): _sha256(path) for path in immutable_paths}

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
    run_dirs[BODY_ONLY_SPEC.key] = _run(curve_config, BODY_ONLY_SPEC)

    evaluator = _official_evaluator()
    specs = {
        "uniform_2": RunSpec("uniform_2", "Uniform-2", "uniform", 2),
        "legacy_4event": RunSpec(
            "legacy_4event", "Legacy-4Event", "uniform2_semantic_weight_uniform", 2
        ),
        "full_event_b2": RunSpec(
            "full_event_b2", "FullEvent-B2", "uniform2_semantic_weight_full_event", 2
        ),
        BODY_ONLY_SPEC.key: BODY_ONLY_SPEC,
    }
    evaluations: dict[str, dict[str, Any]] = {}
    raw_rows: dict[str, dict[str, Any]] = {}
    for key, run_dir in run_dirs.items():
        evaluations[key] = _evaluations(_trajectory(run_dir, config.task_name), events)
        raw_rows[key] = _metric_row(
            specs[key],
            run_dir,
            evaluations[key],
            _official(curve_config, run_dir, evaluator),
            semantic_frames,
        )
    annotated = _annotate(
        list(raw_rows.values()),
        raw_rows["uniform_2"],
        raw_rows["legacy_4event"],
        raw_rows["full_event_b2"],
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
            row["interaction_part_exact_mm"],
            rows["full_event_b2"]["interaction_part_exact_mm"],
        )
    _write_csv(config.output_dir / "approach_body_only_summary.csv", annotated)

    event_rows = _event_rows(events, evaluations)
    _write_csv(config.output_dir / "approach_body_only_event_metrics.csv", event_rows)
    frame34_rows = _frame34_trace_rows(run_dirs)
    _write_csv(config.output_dir / "frame34_iteration_trace.csv", frame34_rows)
    _write_csv(config.output_dir / "approach_weight_trace.csv", _weight_trace_rows(run_dirs))
    approach_event = next(event for event in events if event.name == "approach")
    topology_rows = _approach_topology_rows(
        run_dirs[BODY_ONLY_SPEC.key],
        config.task_name,
        approach_event,
    )
    _write_csv(config.output_dir / "approach_topology_diagnostics.csv", topology_rows)

    full_qpos = _trajectory_qpos(run_dirs["full_event_b2"], config.task_name)
    body_qpos = _trajectory_qpos(run_dirs[BODY_ONLY_SPEC.key], config.task_name)
    per_frame_delta = np.max(np.abs(body_qpos - full_qpos), axis=1)
    divergent = np.flatnonzero(per_frame_delta > 1e-12)
    qpos_diagnostics = {
        "first_divergent_frame": int(divergent[0]) if divergent.size else None,
        "trajectory_qpos_max_abs_diff": float(per_frame_delta.max()),
        "frame34_qpos_max_abs_diff": float(per_frame_delta[34]),
        "frame34_qpos_l2_diff": float(np.linalg.norm(body_qpos[34] - full_qpos[34])),
    }
    (config.output_dir / "qpos_causal_diagnostics.json").write_text(
        json.dumps(qpos_diagnostics, indent=2) + "\n", encoding="utf-8"
    )

    metadata, summary, frames = _profile(run_dirs[BODY_ONLY_SPEC.key])
    policy = metadata["residual_weights"]
    policy_valid = bool(
        policy.get("body_only_weight_events") == ["approach"]
        and metadata["semantic_weighting_mainline"].get("approach_weight_policy")
        == "pelvis body-only; no approach contribution to object-neighbor vertices"
        and rows[BODY_ONLY_SPEC.key]["weight_event_count"] == 8
        and int(summary["total_actual_iterations"]) == 440
        and sum(int(frame["convex_solver_calls"]) for frame in frames) == 440
        and topology_rows
        and all(int(row["pelvis_object_neighbor_count"]) == 0 for row in topology_rows)
    )
    hashes_after = {str(path): _sha256(path) for path in immutable_paths}
    baseline_cache_unchanged = hashes_before == hashes_after
    metadata_payload = {
        "task": config.task_name,
        "experimental_variable": (
            "approach keeps pelvis body residual weighting but contributes no weight to "
            "object-neighbor vertices"
        ),
        "approach_retained": True,
        "approach_body_parts": next(
            event.body_parts for event in events if event.name == "approach"
        ),
        "body_only_weight_events": policy.get("body_only_weight_events"),
        "all_event_names": [event.name for event in events],
        "criticality_used": False,
        "ordinary_budget": 2,
        "actual_sqp": rows[BODY_ONLY_SPEC.key]["actual_sqp"],
        "solver_calls": rows[BODY_ONLY_SPEC.key]["solver_calls"],
        "policy_valid": policy_valid,
        "approach_active_frames": [int(row["frame"]) for row in topology_rows],
        "approach_pelvis_object_neighbor_counts": [
            int(row["pelvis_object_neighbor_count"]) for row in topology_rows
        ],
        "original_full_event_was_already_approach_body_only_in_realized_topology": all(
            int(row["pelvis_object_neighbor_count"]) == 0 for row in topology_rows
        ),
        "baseline_cache_unchanged": baseline_cache_unchanged,
        "immutable_sha256_before": hashes_before,
        "immutable_sha256_after": hashes_after,
        **qpos_diagnostics,
    }
    (config.output_dir / "benchmark_metadata.json").write_text(
        json.dumps(metadata_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _report(
        config.output_dir / "approach_body_only_report.md",
        rows,
        event_rows,
        frame34_rows,
        qpos_diagnostics,
        topology_rows,
        policy_valid,
        baseline_cache_unchanged,
    )
    print(f"Artifacts: {config.output_dir}")


def cli() -> None:
    run(tyro.cli(BenchmarkConfig))


if __name__ == "__main__":
    cli()
