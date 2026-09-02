#!/usr/bin/env python3
"""Aggregate the precision/compute metrics used by the three-method OMOMO table.

The semantic precision evaluator is intentionally independent of optimizer
weights: it reads the unweighted interaction-mesh residuals and original
adjacency stored in each result NPZ.  Object-dependent metrics are reported for
OMOMO; LAFAN has no object/event stream and is reported only for compute metrics.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from holosoma_retargeting.semantic_keyframes.precision import evaluate_precision, load_precision_payload
from holosoma_retargeting.semantic_keyframes.runtime import load_semantic_plan_projection

from prepare_batch_retarget_inputs import TASK_OBJECTS
from prepare_lafan_batch_inputs import TASKS as LAFAN_TASKS


ROOT = Path(__file__).resolve().parents[1]
OMOMO_RUNS = ROOT / "exp/retargeting/omomo_batch/runs"
OMOMO_PLANS = ROOT / "exp/omomo_cari4d"
LAFAN_RUNS = ROOT / "exp/retargeting/lafan_batch/runs"


def _profile(run_dir: Path, task: str, method: str) -> dict[str, Any] | None:
    # Batch runners normally create a profile directory named after the
    # result stem (e.g. ``walk1_subject1_profile``).  Older OMOMO runs used
    # ``*_original_profile`` regardless of the method, so keep that fallback.
    candidates = [
        run_dir / "profile.json",
        run_dir / f"{task}_profile" / "profile.json",
        run_dir / f"{task}_{method}_profile" / "profile.json",
        run_dir / f"{task}_original_profile" / "profile.json",
    ]
    # Tripod was rerun with object non-penetration disabled to make its
    # otherwise infeasible linearized constraints solvable.
    if task == "sub12_tripod_041":
        fallback = {
            "original": ROOT / "exp/retargeting/omomo_batch/test_tripod_noobj2/sub12_tripod_041_original_profile/profile.json",
            "uniform2": ROOT / "exp/retargeting/omomo_batch/test_tripod_uniform_noobj/sub12_tripod_041_original_profile/profile.json",
            "semantic_b4": ROOT / "exp/retargeting/omomo_batch/test_tripod_sem_noobj2/profile.json",
        }[method]
        candidates.append(fallback)
    for path in candidates:
        if path.name == "run_summary.json" and path.exists():
            return json.loads(path.read_text())
        if path.exists():
            payload = json.loads(path.read_text())
            return payload.get("summary", payload)
    return None


def _audit_csv(run_dir: Path, task: str, method: str) -> Path | None:
    candidates = [run_dir / "penetration_pairs.csv", run_dir / f"{task}_original_profile" / "penetration_pairs.csv"]
    if task == "sub12_tripod_041":
        fallback_dir = {
            "original": "test_tripod_noobj2/sub12_tripod_041_original_profile",
            "uniform2": "test_tripod_uniform_noobj/sub12_tripod_041_original_profile",
            "semantic_b4": "test_tripod_sem_noobj2",
        }[method]
        candidates += [
            ROOT / "exp/retargeting/omomo_batch" / fallback_dir / "penetration_pairs.csv",
        ]
    return next((p for p in candidates if p.exists()), None)


def _max_body_box_depth_mm(path: Path | None) -> float:
    if path is None:
        return float("nan")
    depths = []
    with path.open(newline="", encoding="utf-8") as fp:
        for row in csv.DictReader(fp):
            if row.get("pair_type") == "other-body-box":
                depths.append(float(row["penetration_depth"]))
    return 1000.0 * max(depths, default=0.0)


def evaluate_omomo(task: str, method: str) -> dict[str, Any]:
    result = OMOMO_RUNS / method / task / f"{task}_{method}.npz"
    plan_path = OMOMO_PLANS / task / "semantic_keyframes" / f"{task}_dynamic.json"
    row: dict[str, Any] = {"dataset": "OMOMO", "task_name": task, "method": method, "result": str(result)}
    if not result.exists() or not plan_path.exists():
        row["status"] = "missing"
        return row
    try:
        evaluation = evaluate_precision(
            load_precision_payload(result),
            load_semantic_plan_projection(plan_path),
            critical_event_names=None,
        )
        profile = _profile(result.parent, task, method)
        if profile is None:
            raise RuntimeError("profile.json not found")
        row.update({
            "status": "ok",
            "global_exact_mm": 1000.0 * float(evaluation.keyframe_global["exact"]),
            "semantic_part_exact_mm": 1000.0 * float(evaluation.semantic_part["exact"]),
            "local_exact_mm": 1000.0 * float(evaluation.semantic_local["exact"]),
            "body_object_edge_exact_mm": (
                1000.0 * float(evaluation.semantic_edge["exact"])
                if evaluation.semantic_edge is not None else float("nan")
            ),
            "wall_time_s": float(profile["total_wall_time"]),
            "optimization_time_s": float(profile["optimization_wall_time"]),
            "sqp_iterations": int(profile["total_actual_iterations"]),
            "max_body_box_penetration_mm": _max_body_box_depth_mm(_audit_csv(result.parent, task, method)),
        })
    except Exception as exc:
        row.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    return row


def evaluate_lafan(task: str, method: str) -> dict[str, Any]:
    result = LAFAN_RUNS / method / task / f"{task}.npz"
    row: dict[str, Any] = {"dataset": "LAFAN1", "task_name": task, "method": method, "result": str(result)}
    if not result.exists():
        row["status"] = "missing"
        return row
    try:
        profile = _profile(result.parent, task, method)
        if profile is None:
            raise RuntimeError("profile.json not found")
        row.update({"status": "ok", "wall_time_s": float(profile["total_wall_time"]),
                    "optimization_time_s": float(profile["optimization_wall_time"]),
                    "sqp_iterations": int(profile["total_actual_iterations"]),
                    "global_exact_mm": float("nan"), "semantic_part_exact_mm": float("nan"),
                    "local_exact_mm": float("nan"), "body_object_edge_exact_mm": float("nan"),
                    "max_body_box_penetration_mm": float("nan")})
    except Exception as exc:
        row.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    return row


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    vals = [float(r[key]) for r in rows if r.get("status") == "ok" and key in r and np.isfinite(float(r[key]))]
    return float(np.mean(vals)) if vals else float("nan")


def aggregate(rows: list[dict[str, Any]], dataset: str, method: str) -> dict[str, Any]:
    subset = [r for r in rows if (dataset == "all39" or r["dataset"] == dataset)
              and (method == "ours_hybrid" or r["method"] == method)]
    return {
        "dataset": dataset, "method": method,
        "n_requested": len(subset), "n_ok": sum(r.get("status") == "ok" for r in subset),
        **{k: _mean(subset, k) for k in (
            "global_exact_mm", "semantic_part_exact_mm", "local_exact_mm",
            "body_object_edge_exact_mm", "wall_time_s", "optimization_time_s",
            "sqp_iterations", "max_body_box_penetration_mm",
        )},
    }


def main() -> None:
    rows: list[dict[str, Any]] = []
    for method in ("original", "uniform2", "semantic_b4"):
        rows.extend(evaluate_omomo(task, method) for task in TASK_OBJECTS)
    # LAFAN does not have object-semantic precision metrics; still retain its
    # compute rows for the separate 39-task runtime/SQP aggregate.
    for method in ("original", "uniform"):
        rows.extend(evaluate_lafan(task, method) for task in LAFAN_TASKS)

    aggs = [aggregate(rows, "OMOMO", m) for m in ("original", "uniform2", "semantic_b4")]
    aggs += [aggregate(rows, "LAFAN1", m) for m in ("original", "uniform")]
    original39 = [r for r in rows if r["method"] == "original"]
    ours39 = [r for r in rows if (r["dataset"] == "OMOMO" and r["method"] == "semantic_b4") or (r["dataset"] == "LAFAN1" and r["method"] == "uniform")]
    aggs += [aggregate(original39, "all39", "original"), aggregate(ours39, "all39", "ours_hybrid")]

    out_dir = ROOT / "exp/retargeting/precision39"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "precision39_rows.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False, allow_nan=True) + "\n")
    (out_dir / "precision39_aggregate.json").write_text(json.dumps(aggs, indent=2, ensure_ascii=False, allow_nan=True) + "\n")
    fields = sorted({key for row in rows for key in row})
    with (out_dir / "precision39_rows.csv").open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    print(json.dumps({"rows": len(rows), "aggregates": len(aggs), "output": str(out_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
