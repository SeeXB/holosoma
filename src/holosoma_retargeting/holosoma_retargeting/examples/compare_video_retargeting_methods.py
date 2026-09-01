"""Compare three retargeting methods on one video-recovered trajectory.

The quality evaluator is deliberately method independent: it consumes the
unweighted interaction-mesh residuals and original per-frame adjacency saved
inside each trajectory.  Collision statistics are read from the MuJoCo
signed-distance audit emitted by the same retargeting run.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PACKAGE_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = PACKAGE_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from holosoma_retargeting.semantic_keyframes.precision import (  # noqa: E402
    evaluate_precision,
    load_precision_payload,
)
from holosoma_retargeting.semantic_keyframes.runtime import (  # noqa: E402
    load_semantic_plan_projection,
)


MM = 1000.0
METHODS = (
    ("original", "Original OmniRetarget", "#d95f59"),
    ("uniform2", "Uniform-2", "#4c78a8"),
    ("semantic_b4", "Final Semantic B4", "#43aa8b"),
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _penetration_summary(rows: Sequence[dict[str, str]], pair_type: str) -> dict[str, Any]:
    selected = [row for row in rows if row["pair_type"] == pair_type]
    by_frame: dict[int, float] = {}
    for row in selected:
        frame = int(row["frame"])
        depth = float(row["penetration_depth"])
        by_frame[frame] = max(depth, by_frame.get(frame, 0.0))
    depths = np.asarray(list(by_frame.values()), dtype=np.float64)
    return {
        "frames_any": len(by_frame),
        "frames_gt_1mm": int(np.count_nonzero(depths > 0.001)),
        "frames_gt_2mm": int(np.count_nonzero(depths > 0.002)),
        "max_depth_mm": MM * float(depths.max()) if depths.size else 0.0,
        "mean_active_frame_depth_mm": MM * float(depths.mean()) if depths.size else 0.0,
    }


def _improvement(candidate: float, baseline: float) -> float:
    return 100.0 * (baseline - candidate) / baseline


def _method_rows(
    experiment_dir: Path,
    task_name: str,
    events: Sequence[Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, np.ndarray]]:
    summaries: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    framewise: dict[str, np.ndarray] = {}
    for key, label, _ in METHODS:
        run_dir = experiment_dir / "runs" / key
        trajectory = run_dir / f"{task_name}_original.npz"
        profile = _read_json(run_dir / "profile.json")["summary"]
        evaluation = evaluate_precision(
            load_precision_payload(trajectory),
            events,
            critical_event_names=None,
        )
        audit_rows = _read_csv(run_dir / "penetration_pairs.csv")
        other_body_box = _penetration_summary(audit_rows, "other-body-box")
        hand_box = _penetration_summary(audit_rows, "hand-box")
        summary = {
            "method": key,
            "label": label,
            "mode": profile["mode"],
            "num_frames": int(profile["num_frames"]),
            "wall_time_s": float(profile["total_wall_time"]),
            "optimization_time_s": float(profile["optimization_wall_time"]),
            "actual_sqp_iterations": int(profile["total_actual_iterations"]),
            "global_exact_mm": MM * float(evaluation.keyframe_global["exact"]),
            "part_exact_mm": MM * float(evaluation.semantic_part["exact"]),
            "local_exact_mm": MM * float(evaluation.semantic_local["exact"]),
            "edge_exact_mm": (
                MM * float(evaluation.semantic_edge["exact"])
                if evaluation.semantic_edge is not None
                else None
            ),
            "ordinary_global_mean_mm": MM * float(evaluation.ordinary["mean"]),
            "other_body_box_frames_any": other_body_box["frames_any"],
            "other_body_box_frames_gt_1mm": other_body_box["frames_gt_1mm"],
            "other_body_box_frames_gt_2mm": other_body_box["frames_gt_2mm"],
            "other_body_box_max_depth_mm": other_body_box["max_depth_mm"],
            "other_body_box_mean_active_depth_mm": other_body_box[
                "mean_active_frame_depth_mm"
            ],
            "hand_box_frames_any": hand_box["frames_any"],
            "hand_box_max_depth_mm": hand_box["max_depth_mm"],
            "illegal_penetration_frame_ratio": float(
                profile["illegal_penetration_frame_ratio"]
            ),
            "max_illegal_penetration_depth_mm": MM
            * float(profile["max_illegal_penetration_depth"]),
            "mean_foot_sticking_error_mm": MM
            * float(profile["mean_foot_sticking_error"]),
            "max_foot_sticking_error_mm": MM
            * float(profile["max_foot_sticking_error"]),
            "max_joint_limit_violation": float(profile["max_joint_limit_violation"]),
        }
        summaries.append(summary)
        framewise[key] = evaluation.global_per_frame * MM
        for row in evaluation.event_rows:
            event_rows.append(
                {
                    "method": key,
                    "label": label,
                    "event": row["event"],
                    "trigger_frame": int(row["trigger_frame"]),
                    "global_exact_mm": MM * float(row["global_exact"]),
                    "part_exact_mm": MM * float(row["part_exact"]),
                    "local_exact_mm": MM * float(row["local_exact"]),
                    "edge_exact_mm": (
                        MM * float(row["edge_exact"])
                        if row.get("edge_exact") is not None
                        else None
                    ),
                    "part_pm1_mm": MM * float(row["part_pm1"]),
                    "part_pm3_mm": MM * float(row["part_pm3"]),
                }
            )
    return summaries, event_rows, framewise


def _plot_quality(rows: Sequence[dict[str, Any]], path: Path) -> None:
    metrics = (
        ("global_exact_mm", "Global"),
        ("part_exact_mm", "Semantic part"),
        ("local_exact_mm", "Local"),
        ("edge_exact_mm", "Body-object edge"),
    )
    x = np.arange(len(metrics), dtype=np.float64)
    width = 0.24
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    for index, ((_, label, color), row) in enumerate(zip(METHODS, rows)):
        values = [float(row[key]) for key, _ in metrics]
        bars = ax.bar(x + (index - 1) * width, values, width, label=label, color=color)
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
    ax.set_xticks(x, [label for _, label in metrics])
    ax.set_ylabel("Method-independent exact-keyframe error (mm, lower is better)")
    ax.set_title("Video-recovered SMPL-H retargeting quality")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_compute(rows: Sequence[dict[str, Any]], path: Path) -> None:
    labels = [row["label"] for row in rows]
    colors = [method[2] for method in METHODS]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7))
    for ax, key, ylabel in (
        (axes[0], "wall_time_s", "Wall time (s)"),
        (axes[1], "actual_sqp_iterations", "Actual SQP iterations"),
    ):
        values = [float(row[key]) for row in rows]
        bars = ax.bar(labels, values, color=colors)
        ax.bar_label(bars, fmt="%.1f" if key == "wall_time_s" else "%.0f", padding=3)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=12)
    fig.suptitle("Compute cost")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_events(
    event_rows: Sequence[dict[str, Any]],
    events: Sequence[Any],
    path: Path,
) -> None:
    event_names = [event.name for event in events]
    x = np.arange(len(event_names), dtype=np.float64)
    width = 0.24
    fig, ax = plt.subplots(figsize=(12, 5.4))
    for index, (key, label, color) in enumerate(METHODS):
        by_event = {
            row["event"]: row for row in event_rows if row["method"] == key
        }
        values = [float(by_event[name]["part_exact_mm"]) for name in event_names]
        ax.bar(x + (index - 1) * width, values, width, label=label, color=color)
    ax.set_xticks(x, event_names, rotation=25, ha="right")
    ax.set_ylabel("Semantic-part exact error (mm, lower is better)")
    ax.set_title("Per-event keyframe precision")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_penetration(rows: Sequence[dict[str, Any]], path: Path) -> None:
    labels = [row["label"] for row in rows]
    colors = [method[2] for method in METHODS]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7))
    metrics = (
        ("other_body_box_max_depth_mm", "Max depth (mm)"),
        ("other_body_box_frames_gt_1mm", "Frames deeper than 1 mm"),
    )
    for ax, (key, ylabel) in zip(axes, metrics):
        values = [float(row[key]) for row in rows]
        bars = ax.bar(labels, values, color=colors)
        ax.bar_label(bars, fmt="%.3f" if "depth" in key else "%.0f", padding=3)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=12)
    fig.suptitle("Robot torso/limb vs box penetration (hands excluded)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_framewise(
    framewise: dict[str, np.ndarray],
    events: Sequence[Any],
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.8))
    for key, label, color in METHODS:
        ax.plot(framewise[key], label=label, color=color, linewidth=1.5)
    for event in events:
        ax.axvline(event.trigger_frame, color="#8a8a8a", linewidth=0.65, alpha=0.35)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Global unweighted residual (mm)")
    ax.set_title("Framewise global retargeting residual; vertical lines are semantic triggers")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _report(
    path: Path,
    rows: Sequence[dict[str, Any]],
    task_name: str,
) -> None:
    original, uniform2, semantic = rows
    lines = [
        "# Video-recovered SMPL-H retargeting comparison",
        "",
        f"Sequence: `{task_name}` (196 frames, 30 fps).",
        "",
        "## Fair comparison settings",
        "",
        "- Same video-recovered SMPL-H and object trajectory, coordinate transform, largebox mesh/URDF, and MuJoCo scene.",
        "- Object non-penetration and joint limits enabled for all methods.",
        "- Same 10 mm foot-sticking tolerance for all methods; the default 1 mm constraint is infeasible on the noisy recovered trajectory at frame 86.",
        "- Quality scores use unweighted evaluator residuals; optimizer semantic weights and criticality values are not scoring weights.",
        "- There is no robot ground truth for this recovered video, so these are cross-method consistency and geometry audits rather than pose-to-GT errors.",
        "",
        "## Results",
        "",
        f"- Final Semantic B4 semantic-part exact error: {semantic['part_exact_mm']:.3f} mm; improvement vs Original {_improvement(semantic['part_exact_mm'], original['part_exact_mm']):+.2f}% and vs Uniform-2 {_improvement(semantic['part_exact_mm'], uniform2['part_exact_mm']):+.2f}%.",
        f"- Final Semantic B4 body-object edge exact error: {semantic['edge_exact_mm']:.3f} mm; improvement vs Original {_improvement(semantic['edge_exact_mm'], original['edge_exact_mm']):+.2f}% and vs Uniform-2 {_improvement(semantic['edge_exact_mm'], uniform2['edge_exact_mm']):+.2f}%.",
        f"- Global exact error: Original {original['global_exact_mm']:.3f} mm, Uniform-2 {uniform2['global_exact_mm']:.3f} mm, Semantic B4 {semantic['global_exact_mm']:.3f} mm.",
        f"- Other-body/box maximum penetration: Original {original['other_body_box_max_depth_mm']:.3f} mm, Uniform-2 {uniform2['other_body_box_max_depth_mm']:.3f} mm, Semantic B4 {semantic['other_body_box_max_depth_mm']:.3f} mm.",
        f"- Wall time / SQP: Original {original['wall_time_s']:.2f} s / {original['actual_sqp_iterations']}; Uniform-2 {uniform2['wall_time_s']:.2f} s / {uniform2['actual_sqp_iterations']}; Semantic B4 {semantic['wall_time_s']:.2f} s / {semantic['actual_sqp_iterations']}.",
        "",
        "All detailed values are in `metrics/comparison_summary.csv` and `metrics/event_metrics.csv`. The MuJoCo triptych uses direct qpos playback and one shared per-frame camera.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compare(args: argparse.Namespace) -> None:
    experiment_dir = args.experiment_dir.resolve()
    metrics_dir = experiment_dir / "metrics"
    plots_dir = metrics_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    events = load_semantic_plan_projection(args.semantic_keyframes.resolve())
    rows, event_rows, framewise = _method_rows(
        experiment_dir,
        args.task_name,
        events,
    )

    original = rows[0]
    for row in rows:
        row["speedup_vs_original"] = original["wall_time_s"] / row["wall_time_s"]
        row["part_improvement_vs_original_pct"] = _improvement(
            row["part_exact_mm"], original["part_exact_mm"]
        )
        row["edge_improvement_vs_original_pct"] = _improvement(
            row["edge_exact_mm"], original["edge_exact_mm"]
        )

    _write_csv(metrics_dir / "comparison_summary.csv", rows)
    _write_csv(metrics_dir / "event_metrics.csv", event_rows)
    payload = {
        "sequence": args.task_name,
        "evaluation": {
            "quality": "unweighted uniform-Laplacian interaction-mesh residuals and original Delaunay adjacency",
            "events": "all projected semantic events; criticality ignored",
            "penetration": "MuJoCo signed-distance audit in the shared custom largebox scene",
        },
        "fairness": {
            "same_input_and_scene": True,
            "foot_sticking_tolerance_m": 0.01,
            "object_nonpenetration_enabled": True,
            "joint_limits_enabled": True,
        },
        "methods": rows,
    }
    (metrics_dir / "comparison_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    _plot_quality(rows, plots_dir / "quality_exact_keyframes.png")
    _plot_compute(rows, plots_dir / "compute_cost.png")
    _plot_events(event_rows, events, plots_dir / "per_event_part_error.png")
    _plot_penetration(rows, plots_dir / "other_body_box_penetration.png")
    _plot_framewise(framewise, events, plots_dir / "framewise_global_residual.png")
    _report(experiment_dir / "REPORT.md", rows, args.task_name)
    print(json.dumps(payload, indent=2))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--semantic-keyframes", type=Path, required=True)
    parser.add_argument("--task-name", default="cari4d_sub3_largebox_003")
    return parser


if __name__ == "__main__":
    compare(_parser().parse_args())
