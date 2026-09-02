#!/usr/bin/env python3
"""Write the measured precision/compute comparison report."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "exp/retargeting/precision39"


def _fmt(value: float) -> str:
    return "N/A" if not np.isfinite(value) else f"{value:.3f}"


def _improvement(original: float, value: float) -> str:
    if not np.isfinite(original) or not np.isfinite(value):
        return "N/A"
    return f"{100.0 * (original - value) / original:+.2f}%"


def main() -> None:
    aggregates = json.loads((OUT / "precision39_aggregate.json").read_text())
    rows = json.loads((OUT / "precision39_rows.json").read_text())
    by_key = {(row["dataset"], row["method"]): row for row in aggregates}
    omomo = {method: by_key[("OMOMO", method)] for method in ("original", "uniform2", "semantic_b4")}

    precision_keys = (
        ("global_exact_mm", "Global exact error (mm)"),
        ("semantic_part_exact_mm", "Semantic-part error (mm)"),
        ("local_exact_mm", "Local error (mm)"),
        ("body_object_edge_exact_mm", "Body-object edge error (mm)"),
        ("wall_time_s", "Wall time (s)"),
        ("sqp_iterations", "SQP iterations"),
        ("max_body_box_penetration_mm", "Max body-box penetration (mm)"),
    )
    methods = (("original", "Original"), ("uniform2", "Uniform-2"), ("semantic_b4", "Semantic B4"))
    lines = [
        "# 39-task retargeting precision/compute comparison",
        "",
        "The four precision/error rows and body-box penetration are defined only for the 20 OMOMO object-interaction tasks. Values are macro-averages: first compute one value per task, then average across the 20 tasks.",
        "",
        "## OMOMO (20 tasks; all requested metrics)",
        "",
        "| Metric | Original | Uniform-2 | Semantic B4 |",
        "|---|---:|---:|---:|",
    ]
    for key, label in precision_keys:
        values = [_fmt(float(omomo[method][key])) for method, _ in methods]
        lines.append(f"| {label} | {values[0]} | {values[1]} | {values[2]} |")
    lines += [
        "",
        "Relative improvement (positive means lower/better; negative means worse): Semantic B4 vs Original — "
        + "; ".join(
            f"{label}: {_improvement(float(omomo['original'][key]), float(omomo['semantic_b4'][key]))}"
            for key, label in precision_keys
        )
        + ".",
        "",
        "### Penetration outlier audit",
        "",
        "The penetration row is the mean of each task's maximum `other-body-box` depth; it is not the maximum over all frames/tasks. The per-task maximum distribution is dominated by the infeasible `sub12_tripod_041` fallback and the `sub17_floorlamp_026` geometry/trajectory mismatch:",
        "",
    ]
    for method, label in methods:
        vals = [
            float(row["max_body_box_penetration_mm"])
            for row in rows
            if row.get("dataset") == "OMOMO" and row.get("method") == method and row.get("status") == "ok"
        ]
        lines.append(
            f"- {label}: median {_fmt(float(np.median(vals)))} mm; overall maximum {_fmt(float(np.max(vals)))} mm."
        )
    lines += [
        "",
        "## All 39 tasks (compute-only aggregate)",
        "",
        "LAFAN1 has no object mesh or VLM semantic-event stream, so semantic-part, local semantic, body-object edge, and body-box penetration are N/A there. For the all-39 wall-time/SQP aggregate, Semantic B4 uses the robot-only Uniform-2 fallback on the 19 LAFAN1 tasks; this does not claim semantic B4 was evaluated on LAFAN1.",
        "",
        "| Metric | Original | Uniform-2 | Semantic B4 + robot-only LAFAN fallback |",
        "|---|---:|---:|---:|",
    ]
    lafan_original = by_key[("LAFAN1", "original")]
    lafan_uniform = by_key[("LAFAN1", "uniform")]
    all39_values = {
        "original": (omomo["original"], lafan_original),
        "uniform2": (omomo["uniform2"], lafan_uniform),
        "semantic_b4": (omomo["semantic_b4"], lafan_uniform),
    }
    for key, label in (("wall_time_s", "Wall time (s)"), ("sqp_iterations", "SQP iterations")):
        rendered = []
        for method in ("original", "uniform2", "semantic_b4"):
            om, la = all39_values[method]
            value = (20.0 * float(om[key]) + 19.0 * float(la[key])) / 39.0
            rendered.append(_fmt(value))
        lines.append(f"| {label} | {rendered[0]} | {rendered[1]} | {rendered[2]} |")
    lines += [
        "",
        "Artifacts: `precision39_rows.csv` contains all 98 per-task rows; `precision39_aggregate.json` contains the machine-readable aggregates.",
    ]
    (OUT / "precision39_REPORT.md").write_text("\n".join(lines) + "\n")
    print(OUT / "precision39_REPORT.md")


if __name__ == "__main__":
    main()
