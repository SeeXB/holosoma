#!/usr/bin/env python3
"""Publish the four native OmniRetarget penetration/foot-skating metrics.

This script consumes the evaluator outputs saved by
``run_current_semantic_retarget_comparison.py``.  It deliberately mirrors the
aggregation in ``evaluation/eval_retargeting.py``:

* duration: mean/std over per-trajectory fractions;
* ``max`` arrays: concatenate all violating-frame values, then mean/std.

The comparison is restricted to the already validated paired trajectories in
``comparison.json``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _summary(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    penetration_duration = np.asarray(
        [row[method]["penetration_duration"] for row in rows], dtype=np.float64
    )
    penetration_depth_cm = np.asarray(
        [
            100.0 * value
            for row in rows
            for value in row[method]["penetration_max_depths_m"]
        ],
        dtype=np.float64,
    )
    skating_duration = np.asarray(
        [row[method]["foot_skating_duration"] for row in rows], dtype=np.float64
    )
    skating_velocity_cm_s = np.asarray(
        [
            100.0 * value
            for row in rows
            for value in row[method]["max_toe_sliding_velocities_m_s"]
        ],
        dtype=np.float64,
    )

    def mean_std(values: np.ndarray) -> tuple[float, float]:
        if values.size == 0:
            return 0.0, 0.0
        return float(values.mean()), float(values.std(ddof=0))

    pen_duration_mean, pen_duration_std = mean_std(penetration_duration)
    pen_depth_mean, pen_depth_std = mean_std(penetration_depth_cm)
    skating_duration_mean, skating_duration_std = mean_std(skating_duration)
    skating_velocity_mean, skating_velocity_std = mean_std(skating_velocity_cm_s)
    return {
        "trajectories": len(rows),
        "penetration": {
            "duration_mean": pen_duration_mean,
            "duration_std": pen_duration_std,
            "max_depth_cm_mean": pen_depth_mean,
            "max_depth_cm_std": pen_depth_std,
            "violating_frames": int(penetration_depth_cm.size),
        },
        "foot_skating": {
            "duration_mean": skating_duration_mean,
            "duration_std": skating_duration_std,
            "max_velocity_cm_s_mean": skating_velocity_mean,
            "max_velocity_cm_s_std": skating_velocity_std,
            "violating_frames": int(skating_velocity_cm_s.size),
        },
    }


def _native_row(method_row: dict[str, Any]) -> dict[str, Any]:
    result = Path(method_row["result"])
    raw_path = result.parent / "native_penetration_raw.json"
    raw = json.loads(raw_path.read_text(encoding="utf-8"))

    penetration_duration = float(raw["duration"])
    if not np.isclose(
        penetration_duration,
        float(method_row["penetration_fraction"]),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(f"Penetration duration mismatch: {raw_path}")

    # The old run artifact did not retain the skating array.  A zero native
    # mean can only come from an empty array because every retained violation
    # is strictly positive.  Reject nonzero cases rather than approximating
    # the paper aggregation from a per-run mean.
    skating_mean = float(method_row["toe_sliding_velocity_m_s"])
    if not np.isclose(skating_mean, 0.0, rtol=0.0, atol=1e-15):
        raise ValueError(
            f"Nonzero foot-skating array was not retained for {result}; "
            "rerun the native evaluator and save its raw values"
        )

    return {
        "trajectory": str(result),
        "trajectory_sha256": _sha256(result),
        "native_raw": str(raw_path),
        "penetration_duration": penetration_duration,
        "penetration_max_depths_m": [float(value) for value in raw["penetration_max_depths_m"]],
        "foot_skating_duration": float(method_row["sliding_fraction"]),
        "max_toe_sliding_velocities_m_s": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()

    run = args.run.resolve()
    source = run / "comparison.json"
    comparison = json.loads(source.read_text(encoding="utf-8"))
    paired = comparison["paired"]
    if not paired:
        raise ValueError(f"No validated pairs in {source}")

    rows: list[dict[str, Any]] = []
    for pair in paired:
        if pair["dataset"] != "omomo":
            continue
        original = pair["original"]
        semantic = pair["semantic_b4"]
        if original["source_vertices_sha256"] != semantic["source_vertices_sha256"]:
            raise ValueError(f"Source vertex mismatch for {pair['task']}")
        if original["source_adjacency_sha256"] != semantic["source_adjacency_sha256"]:
            raise ValueError(f"Source adjacency mismatch for {pair['task']}")
        rows.append(
            {
                "task": pair["task"],
                "original": _native_row(original),
                "semantic_b4": _native_row(semantic),
            }
        )

    summary = {method: _summary(rows, method) for method in ("original", "semantic_b4")}
    payload = {
        "comparison_source": str(source),
        "comparison_source_sha256": _sha256(source),
        "paired_tasks": [row["task"] for row in rows],
        "std_ddof": 0,
        "penetration_tolerance_m": 0.01,
        "foot_sliding_threshold_native": 0.01,
        "aggregation": {
            "duration": "mean/std of per-trajectory fractions",
            "max_columns": (
                "concatenate the evaluator's per-violating-frame maxima across trajectories, "
                "then mean/std; do not zero-pad clean trajectories"
            ),
        },
        "summary": summary,
        "rows": rows,
    }
    json_path = run / "native_omniretarget_metrics.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def cell(mean: float, std: float, digits: int) -> str:
        return f"{mean:.{digits}f} ± {std:.{digits}f}"

    lines = [
        "# Native OmniRetarget interaction metrics",
        "",
        f"Validated paired OMOMO trajectories: {len(rows)}. The same source geometry and collision scene are used by both methods.",
        "",
        "| Method | Penetration Duration ↓ | Max Depth (cm) ↓ | Foot Skating Duration ↓ | Max Vel. (cm/s) ↓ |",
        "|---|---:|---:|---:|---:|",
    ]
    for method, label in (("original", "Original OmniRetarget"), ("semantic_b4", "Semantic B4")):
        values = summary[method]
        penetration = values["penetration"]
        skating = values["foot_skating"]
        lines.append(
            f"| {label} | "
            f"{cell(penetration['duration_mean'], penetration['duration_std'], 4)} | "
            f"{cell(penetration['max_depth_cm_mean'], penetration['max_depth_cm_std'], 2)} | "
            f"{cell(skating['duration_mean'], skating['duration_std'], 4)} | "
            f"{cell(skating['max_velocity_cm_s_mean'], skating['max_velocity_cm_s_std'], 2)} |"
        )
    lines += [
        "",
        "Aggregation follows `evaluation/eval_retargeting.py`: duration is averaged over trajectories; the two `max` arrays are concatenated over violating frames before mean/std. Standard deviation uses `ddof=0`.",
        "",
        "The native foot-skating implementation names the retained XY frame-to-frame displacement a velocity but does not divide by frame time. Both methods produced no retained sliding frames here, so that implementation detail does not affect the reported zero.",
        "",
        f"Penetrating frames: Original {summary['original']['penetration']['violating_frames']}; Semantic B4 {summary['semantic_b4']['penetration']['violating_frames']}. Foot-skating frames: Original {summary['original']['foot_skating']['violating_frames']}; Semantic B4 {summary['semantic_b4']['foot_skating']['violating_frames']}.",
    ]
    report_path = run / "NATIVE_OMNIRETARGET_METRICS.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(report_path)


if __name__ == "__main__":
    main()
