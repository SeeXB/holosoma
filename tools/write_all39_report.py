#!/usr/bin/env python3
"""Render a concise Markdown summary from ALL39_evaluation.json."""

from __future__ import annotations

import json
import glob
import statistics
from pathlib import Path

import math


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "exp/retargeting/ALL39_evaluation.json"
OUT = ROOT / "exp/retargeting/ALL39_REPORT.md"


KEYS = [
    ("opt_cost_last_frame", "opt_cost (last frame)"),
    ("frame_cost_mean", "frame cost (mean)"),
    ("actual_sqp_iterations_mean", "SQP iterations / frame"),
    ("penetration_duration", "penetration duration"),
    ("penetration_max_depth_mean", "penetration depth (mean positive)"),
    ("sliding_duration", "foot sliding duration"),
    ("max_toe_sliding_velocity_mean", "toe sliding velocity"),
    ("contact_preservation", "contact preservation"),
]


def value(block: dict, key: str) -> float:
    return float(block[key]["mean_over_tasks"])


def table(a: dict, b: dict, name_a: str, name_b: str) -> str:
    lines = ["| metric | " + name_a + " | " + name_b + " | relative Δ |"]
    lines.append("|---|---:|---:|---:|")
    for key, label in KEYS:
        x, y = value(a, key), value(b, key)
        delta = "n/a" if x == 0 or not (math.isfinite(x) and math.isfinite(y)) else f"{(y-x)/x*100:+.2f}%"
        sx = "n/a" if not math.isfinite(x) else f"{x:.6f}"
        sy = "n/a" if not math.isfinite(y) else f"{y:.6f}"
        lines.append(f"| {label} | {sx} | {sy} | {delta} |")
    return "\n".join(lines)


def timing(patterns: list[str]) -> tuple[float, float, float, int]:
    paths: list[str] = []
    for pattern in patterns:
        paths.extend(glob.glob(str(ROOT / pattern)))
    rows = [json.loads(Path(p).read_text()) for p in paths]
    return (
        statistics.mean(float(r["total_wall_time"]) for r in rows),
        statistics.mean(float(r["optimization_wall_time"]) for r in rows),
        statistics.mean(float(r["mean_frame_time"]) for r in rows),
        len(rows),
    )


def main() -> None:
    d = json.loads(REPORT.read_text())
    agg = d["aggregate"]
    omo_orig = timing(["exp/retargeting/omomo_batch/runs/original/*/*_profile/run_summary.json", "exp/retargeting/omomo_batch/test_tripod_noobj2/sub12_tripod_041_original_profile/run_summary.json"])
    omo_sem = timing(["exp/retargeting/omomo_batch/runs/semantic_b4/*/run_summary.json", "exp/retargeting/omomo_batch/test_tripod_sem_noobj2/run_summary.json"])
    laf_orig = timing(["exp/retargeting/lafan_batch/runs/original/*/*_profile/run_summary.json"])
    laf_uni = timing(["exp/retargeting/lafan_batch/runs/uniform/*/*_profile/run_summary.json"])
    lines = [
        "# 39-task retargeting/evaluation report",
        "",
        "All metrics are from the repository's official `eval_retargeting.py`; values are macro-averages over tasks (not frame-weighted).",
        "",
        "## Protocol",
        "",
        "- OMOMO: 20 requested object-interaction sequences, Original OmniRetarget vs final Semantic B4; both use the same generated object scenes and `no-activate-foot-sticking` for batch feasibility.",
        "- LAFAN1: 19 requested robot-only sequences, Original vs Uniform-2. The object-semantic B4 mode is not defined for LAFAN because there is no object/event stream; Uniform-2 is therefore the explicitly marked robot-only fallback.",
        "- LAFAN input is uniformly downsampled by stride 20 (`manifest.json`) so the full sweep is tractable; both methods use the same downsampled input.",
        "- `sub12_tripod_041` required the documented no-object-nonpenetration fallback for both methods because its linearized object constraint is infeasible; penetration remains measured by the evaluator.",
        "",
        "## OMOMO (20/20 each)",
        "",
        table(agg["OMOMO"]["original"], agg["OMOMO"]["semantic_b4"], "Original", "Semantic B4"),
        "",
        "## LAFAN1 (19/19 each)",
        "",
        table(agg["LAFAN1"]["original"], agg["LAFAN1"]["uniform"], "Original", "Uniform-2 fallback"),
        "",
        "## Unified 39-task macro-average",
        "",
        "The unified ‘ours’ column is Semantic B4 for OMOMO + Uniform-2 fallback for LAFAN1. Because the two datasets have different objectives and scales, the dataset-specific tables above are the scientifically meaningful comparison.",
        "",
        table(agg["all39_original_unified"], agg["all39_ours_unified"], "Original (39)", "Ours hybrid (39)"),
        "",
        "## Runtime (from run profiles)",
        "",
        "Runtime is separate from the optimization objective. The averages below are per task; OMOMO includes the tripod fallback profile and LAFAN uses the same stride-20 input for both methods.",
        "",
        f"| dataset | method | total wall time (s/task) | optimization time (s/task) | wall time (s/frame) | n |",
        "|---|---|---:|---:|---:|---:|",
        f"| OMOMO | Original | {omo_orig[0]:.3f} | {omo_orig[1]:.3f} | {omo_orig[2]:.4f} | {omo_orig[3]} |",
        f"| OMOMO | Semantic B4 | {omo_sem[0]:.3f} | {omo_sem[1]:.3f} | {omo_sem[2]:.4f} | {omo_sem[3]} |",
        f"| LAFAN1 | Original | {laf_orig[0]:.3f} | {laf_orig[1]:.3f} | {laf_orig[2]:.4f} | {laf_orig[3]} |",
        f"| LAFAN1 | Uniform-2 fallback | {laf_uni[0]:.3f} | {laf_uni[1]:.3f} | {laf_uni[2]:.4f} | {laf_uni[3]} |",
        "",
        "## Artifacts",
        "",
        "- Raw machine-readable results: `ALL39_evaluation.json`.",
        "- OMOMO retarget outputs: `omomo_batch/runs/{original,semantic_b4}/`.",
        "- LAFAN retarget outputs: `lafan_batch/runs/{original,uniform}/`.",
        "",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
