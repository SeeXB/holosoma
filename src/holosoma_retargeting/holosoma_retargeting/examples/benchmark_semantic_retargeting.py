"""Run and aggregate the semantic-keyframe-aware OmniRetarget ablations."""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tyro

src_root = Path(__file__).resolve().parents[2]
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from holosoma_retargeting.config_types.retargeting import RetargetingConfig  # noqa: E402
from holosoma_retargeting.config_types.semantic import SemanticRetargetingConfig  # noqa: E402
from holosoma_retargeting.examples.robot_retarget import main as run_retargeting  # noqa: E402
from holosoma_retargeting.semantic_keyframes.runtime import load_semantic_events  # noqa: E402


RUN_SPECS = (
    ("original", "original", None, None),
    ("uniform_1", "uniform", 1, None),
    ("uniform_2", "uniform", 2, None),
    ("uniform_3", "uniform", 3, None),
    ("budget_only", "budget_only", None, None),
    ("weight_only", "weight_only", None, None),
    ("budget_weight", "budget_weight", None, None),
    ("full", "full", None, None),
    ("random_0", "random_semantic", None, 0),
    ("random_1", "random_semantic", None, 1),
    ("random_2", "random_semantic", None, 2),
)
CRITICAL_EVENTS = ("contact", "lift", "place", "release")


@dataclass
class BenchmarkConfig:
    """Configuration for the fixed sub3_largebox_003 benchmark."""

    task_name: str = "sub3_largebox_003"
    data_path: Path = Path("demo_data/OMOMO_new")
    semantic_keyframe_path: Path = Path(
        "demo_data/semantic_keyframes/sub3_largebox_003_template_keyframes.json"
    )
    output_dir: Path = Path("benchmark_results")
    force: bool = False
    """Overwrite known result artifacts by rerunning each method."""

    aggregate_only: bool = False
    """Only rebuild summary tables/plots from already completed method directories."""


def _trajectory_path(run_dir: Path, task_name: str) -> Path:
    return run_dir / f"{task_name}_original.npz"


def _run_spec(config: BenchmarkConfig, spec: tuple[str, str, int | None, int | None]) -> None:
    directory_name, mode, uniform_budget, random_seed = spec
    run_dir = config.output_dir / directory_name
    trajectory = _trajectory_path(run_dir, config.task_name)
    summary_path = run_dir / "run_summary.json"
    if not config.force and trajectory.exists() and summary_path.exists():
        print(f"Skipping completed run: {directory_name}")
        return
    semantic_path = None if mode in {"original", "uniform"} else config.semantic_keyframe_path
    semantic = SemanticRetargetingConfig(
        mode=mode,  # type: ignore[arg-type]
        semantic_keyframe_path=semantic_path,
        profile_dir=run_dir,
        uniform_budget=uniform_budget or 1,
        random_seed=random_seed or 0,
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
    print(f"\n=== Running {directory_name} ===")
    run_retargeting(retargeting)


def _load_profiles(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads((run_dir / "profile.json").read_text(encoding="utf-8"))
    return payload["summary"], payload["frames"]


def _mean_available(values: list[Any]) -> float | None:
    finite = np.asarray([value for value in values if value is not None], dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(finite.mean()) if finite.size else None


def _event_rows(
    method: str,
    profiles: list[dict[str, Any]],
    event_triggers: dict[str, int],
    radius: int = 3,
) -> list[dict[str, Any]]:
    by_frame = {int(row["frame_idx"]): row for row in profiles}
    rows: list[dict[str, Any]] = []
    for event_name in CRITICAL_EVENTS:
        trigger = event_triggers[event_name]
        local = [by_frame[frame] for frame in range(trigger - radius, trigger + radius + 1) if frame in by_frame]
        left_error = _mean_available([row.get("left_hand_object_error") for row in local])
        right_error = _mean_available([row.get("right_hand_object_error") for row in local])
        rows.append(
            {
                "method": method,
                "event": event_name,
                "trigger_frame": trigger,
                "window_start": trigger - radius,
                "window_end": trigger + radius,
                "left_hand_object_error": left_error,
                "right_hand_object_error": right_error,
                "hand_object_error": _mean_available([left_error, right_error]),
                "left_local_laplacian_error": _mean_available(
                    [row.get("left_hand_local_laplacian_error") for row in local]
                ),
                "right_local_laplacian_error": _mean_available(
                    [row.get("right_hand_local_laplacian_error") for row in local]
                ),
                "left_demo_object_distance": _mean_available(
                    [row.get("left_hand_demo_object_distance") for row in local]
                ),
                "right_demo_object_distance": _mean_available(
                    [row.get("right_hand_demo_object_distance") for row in local]
                ),
                "left_robot_object_distance": _mean_available(
                    [row.get("left_hand_robot_object_distance") for row in local]
                ),
                "right_robot_object_distance": _mean_available(
                    [row.get("right_hand_robot_object_distance") for row in local]
                ),
            }
        )
    return rows


def _position_rmse(profiles: list[dict[str, Any]], original: list[dict[str, Any]]) -> float | None:
    values: list[float] = []
    for row, reference in zip(profiles, original):
        for body_part in ("pelvis", "left_hand", "right_hand"):
            position = row.get(f"{body_part}_robot_position")
            reference_position = reference.get(f"{body_part}_robot_position")
            if position is None or reference_position is None:
                continue
            delta = np.asarray(position, dtype=np.float64) - np.asarray(reference_position, dtype=np.float64)
            values.extend(delta.tolist())
    return float(np.sqrt(np.mean(np.square(values)))) if values else None


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_results(
    output_dir: Path,
    profiles: dict[str, list[dict[str, Any]]],
    summaries: list[dict[str, Any]],
    event_rows: list[dict[str, Any]],
    event_triggers: dict[str, int],
    matched_uniform: str,
) -> None:
    try:
        import matplotlib  # noqa: PLC0415

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to generate benchmark plots") from exc

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    semantic_profile = profiles["full"]
    frames = [row["frame_idx"] for row in semantic_profile]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.step(frames, [row["base_budget"] for row in semantic_profile], where="mid", label="Semantic Full")
    for event_name, trigger in event_triggers.items():
        ax.axvline(trigger, color="tab:red", alpha=0.35, linewidth=1)
        ax.text(trigger, 0.98, f"{event_name} {trigger}", rotation=90, transform=ax.get_xaxis_transform(), va="top")
    ax.set(xlabel="Frame", ylabel="SQP max budget", title="Semantic SQP budget by frame")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "01_sqp_max_budget_vs_frame.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4))
    for method, label in (("original", "Original"), (matched_uniform, matched_uniform), ("full", "Semantic Full")):
        ax.plot(
            [row["frame_idx"] for row in profiles[method]],
            [row["actual_sqp_iterations"] for row in profiles[method]],
            label=label,
            linewidth=1.2,
        )
    ax.set(xlabel="Frame", ylabel="Actual SQP iterations", title="Actual SQP iterations by frame")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "02_actual_sqp_iterations_vs_frame.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4))
    for method, label in (("original", "Original"), (matched_uniform, matched_uniform), ("full", "Semantic Full")):
        ax.plot(
            [row["frame_idx"] for row in profiles[method]],
            [row["optimization_time"] for row in profiles[method]],
            label=label,
            linewidth=1.0,
        )
    ax.set(xlabel="Frame", ylabel="Optimization time (s)", title="Frame optimization time")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "03_frame_optimization_time.png", dpi=180)
    plt.close(fig)

    methods_to_plot = ["original", matched_uniform, "budget_only", "budget_weight", "full", "random_0"]
    fig, ax = plt.subplots(figsize=(10, 5))
    for method in methods_to_plot:
        method_rows = [row for row in event_rows if row["method"] == method]
        ax.plot(
            [row["event"] for row in method_rows],
            [row["hand_object_error"] for row in method_rows],
            marker="o",
            label=method,
        )
    ax.set(ylabel="Human-vs-robot hand/object distance error (m)", title="Critical interaction error (trigger +/-3)")
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(plot_dir / "04_critical_hand_object_error.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for row in summaries:
        ax.scatter(row["wall_time"], row["hand_box_error"], s=45)
        ax.annotate(row["method"], (row["wall_time"], row["hand_box_error"]), xytext=(4, 3), textcoords="offset points")
    ax.set(xlabel="Wall time (s)", ylabel="Critical hand/object error (m)", title="Speed-quality tradeoff")
    fig.tight_layout()
    fig.savefig(plot_dir / "05_speed_quality_scatter.png", dpi=180)
    plt.close(fig)


def aggregate_results(config: BenchmarkConfig) -> None:
    """Build fair comparisons, diagnostics, event tables, and required plots."""
    defaults = SemanticRetargetingConfig()
    events = load_semantic_events(config.semantic_keyframe_path)
    event_triggers = {event.name: event.trigger_frame for event in events if event.name in CRITICAL_EVENTS}
    missing = set(CRITICAL_EVENTS) - set(event_triggers)
    if missing:
        raise ValueError(f"Missing critical events in semantic JSON: {sorted(missing)}")

    profiles: dict[str, list[dict[str, Any]]] = {}
    raw_summaries: dict[str, dict[str, Any]] = {}
    event_metrics: list[dict[str, Any]] = []
    for directory_name, _, _, _ in RUN_SPECS:
        run_dir = config.output_dir / directory_name
        raw_summaries[directory_name], profiles[directory_name] = _load_profiles(run_dir)
        event_metrics.extend(_event_rows(directory_name, profiles[directory_name], event_triggers))

    original_qpos = np.load(_trajectory_path(config.output_dir / "original", config.task_name))["qpos"]
    original_wall = float(raw_summaries["original"]["total_wall_time"])
    uniform_methods = ("uniform_1", "uniform_2", "uniform_3")
    full_actual = int(raw_summaries["full"]["total_actual_iterations"])
    budget_only_actual = int(raw_summaries["budget_only"]["total_actual_iterations"])
    budget_weight_actual = int(raw_summaries["budget_weight"]["total_actual_iterations"])
    budget_matched_uniform = min(
        uniform_methods,
        key=lambda method: abs(int(raw_summaries[method]["total_actual_iterations"]) - budget_only_actual),
    )
    budget_weight_matched_uniform = min(
        uniform_methods,
        key=lambda method: abs(int(raw_summaries[method]["total_actual_iterations"]) - budget_weight_actual),
    )
    full_compute_reference = min(
        ("original", *uniform_methods),
        key=lambda method: abs(int(raw_summaries[method]["total_actual_iterations"]) - full_actual),
    )

    summary_rows: list[dict[str, Any]] = []
    for directory_name, mode, _, _ in RUN_SPECS:
        summary = raw_summaries[directory_name]
        method_events = [row for row in event_metrics if row["method"] == directory_name]
        qpos = np.load(_trajectory_path(config.output_dir / directory_name, config.task_name))["qpos"]
        overlap = min(len(qpos), len(original_qpos))
        qpos_delta = qpos[:overlap] - original_qpos[:overlap]
        root_delta = qpos[:overlap, :7] - original_qpos[:overlap, :7]
        penetration_depths = np.asarray(
            [row.get("penetration_depth", np.nan) for row in profiles[directory_name]], dtype=np.float64
        )
        penetration_depths = penetration_depths[np.isfinite(penetration_depths)]
        same_budget_uniform = {
            "budget_only": budget_matched_uniform,
            "budget_weight": budget_weight_matched_uniform,
        }.get(directory_name, "")
        compute_matched_reference = full_compute_reference if directory_name == "full" else ""
        summary_rows.append(
            {
                "method": directory_name,
                "mode": mode,
                "wall_time": summary["total_wall_time"],
                "speedup_vs_original": original_wall / float(summary["total_wall_time"]),
                "actual_sqp": summary["total_actual_iterations"],
                "lap_error": summary["laplacian_residual_mean"],
                "hand_box_error": _mean_available([row["hand_object_error"] for row in method_events]),
                "penetration_frame_ratio": summary["penetration_frame_ratio"],
                "penetration_violation_frame_ratio_1mm": (
                    float(np.mean(penetration_depths > 1e-3)) if penetration_depths.size else None
                ),
                "max_penetration_depth": summary["max_penetration_depth"],
                # The fixed benchmark does not configure self-collision pairs.
                # Keep this unavailable rather than misreporting an unchecked zero.
                "self_collision_violation": None,
                "foot_error": summary["mean_foot_sticking_error"],
                "joint_limit_violation": summary["max_joint_limit_violation"],
                "velocity_limit_violation": summary["max_velocity_limit_violation"],
                "num_rescued_frames": summary["num_rescued_frames"],
                "extra_rescue_iterations": summary["extra_iterations_from_rescue"],
                "qpos_rmse_vs_original": float(np.sqrt(np.mean(np.square(qpos_delta)))),
                "root_rmse_vs_original": float(np.sqrt(np.mean(np.square(root_delta)))),
                "semantic_joint_rmse_vs_original": _position_rmse(profiles[directory_name], profiles["original"]),
                "same_budget_uniform": same_budget_uniform,
                "compute_matched_reference": compute_matched_reference,
            }
        )

    _write_csv(config.output_dir / "summary.csv", summary_rows)
    _write_csv(config.output_dir / "event_metrics.csv", event_metrics)
    metadata = {
        "task_name": config.task_name,
        "semantic_keyframe_path": str(config.semantic_keyframe_path),
        "event_triggers": event_triggers,
        "budget_only_same_budget_uniform": budget_matched_uniform,
        "budget_only_actual_iterations": budget_only_actual,
        "budget_only_uniform_iteration_gap_ratio": abs(
            int(raw_summaries[budget_matched_uniform]["total_actual_iterations"]) - budget_only_actual
        )
        / budget_only_actual,
        "budget_weight_same_budget_uniform": budget_weight_matched_uniform,
        "full_low_budget_uniform_match": None,
        "full_compute_matched_reference": full_compute_reference,
        "full_actual_iterations": full_actual,
        "full_reference_iteration_gap_ratio": abs(
            int(raw_summaries[full_compute_reference]["total_actual_iterations"]) - full_actual
        )
        / full_actual,
        "uniform_actual_iterations": {
            method: raw_summaries[method]["total_actual_iterations"] for method in uniform_methods
        },
        "fixed_scheduler": {
            "frame0_budget": defaults.frame0_budget,
            "critical_trigger_budget": defaults.critical_trigger_budget,
            "near_trigger_budget": defaults.near_trigger_budget,
            "critical_phase_budget": defaults.critical_phase_budget,
            "ordinary_budget": defaults.ordinary_budget,
            "near_trigger_radius": defaults.near_trigger_radius,
        },
        "fixed_residual_weights": {
            "body_weight_multiplier": defaults.body_weight_multiplier,
            "object_neighbor_multiplier": defaults.object_neighbor_multiplier,
            "semantic_temporal_sigma": defaults.semantic_temporal_sigma,
            "phase_semantic_weight": defaults.phase_semantic_weight,
            "normalization": "mean vertex weight = 1 per frame",
        },
        "fixed_rescue_tolerances": {
            "penetration": defaults.rescue_penetration_tolerance,
            "self_collision": defaults.rescue_self_collision_tolerance,
            "joint_limit": defaults.rescue_joint_limit_tolerance,
            "foot_sticking": defaults.rescue_foot_sticking_tolerance,
            "hand_object": defaults.rescue_hand_object_tolerance,
            "velocity_per_frame": defaults.rescue_velocity_limit_per_frame,
        },
        "interaction_distance_metric": "nearest fixed object surface sample; not exact triangle-mesh distance",
        "penetration_violation_tolerance_m": 1e-3,
        "self_collision_metric": "not evaluated because no self-collision pairs are configured",
        "velocity_metric": "not reported unless rescue_velocity_limit_per_frame is configured",
    }
    (config.output_dir / "benchmark_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _plot_results(
        config.output_dir,
        profiles,
        summary_rows,
        event_metrics,
        event_triggers,
        budget_matched_uniform,
    )


def main(config: BenchmarkConfig) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    if not config.aggregate_only:
        for spec in RUN_SPECS:
            _run_spec(config, spec)
    aggregate_results(config)
    print(f"Benchmark summary and plots written to {config.output_dir}")


def cli() -> None:
    """Installed command-line entry point."""
    main(tyro.cli(BenchmarkConfig))


if __name__ == "__main__":
    cli()
