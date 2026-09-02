#!/usr/bin/env python3
"""Evaluate all requested OMOMO and LAFAN retargeting outputs.

The official evaluator's CLI discovers only ``*_original.npz`` files.  This
script calls the same evaluator class directly so method-specific output names
can be compared without symlinks or accidental overwrites.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from holosoma_retargeting.config_types.data_type import MotionDataConfig
from holosoma_retargeting.config_types.robot import RobotConfig
from holosoma_retargeting.evaluation.eval_retargeting import (
    RetargetingEvaluator,
    create_task_constants,
)
from holosoma_retargeting.src.utils import load_intermimic_data

from prepare_batch_retarget_inputs import TASK_OBJECTS
from prepare_lafan_batch_inputs import TASKS as LAFAN_TASKS


ROOT = Path(__file__).resolve().parents[1]
OMOMO_INPUT = ROOT / "exp/retargeting/omomo_batch/input"
LAFAN_INPUT = ROOT / "exp/retargeting/lafan_batch/input"
OMOMO_RUNS = ROOT / "exp/retargeting/omomo_batch/runs"
LAFAN_RUNS = ROOT / "exp/retargeting/lafan_batch/runs"


def _mean_array(x: Any) -> float:
    a = np.asarray(x, dtype=float).reshape(-1)
    return float(np.mean(a)) if a.size else float("nan")


def evaluate_one(dataset: str, task: str, method: str) -> dict[str, Any]:
    if dataset == "OMOMO":
        object_name = TASK_OBJECTS[task]
        result_path = OMOMO_RUNS / method / task / f"{task}_{method}.npz"
        data_dir = OMOMO_INPUT
        motion_cfg = MotionDataConfig(data_format="smplh", robot_type="g1")
        data_type = "robot_object"
        constants = create_task_constants(RobotConfig(robot_type="g1"), motion_cfg, object_name=object_name)
        constants.SCENE_XML_FILE = str((OMOMO_INPUT / "scenes" / f"g1_29dof_w_{object_name}.xml").resolve())
    else:
        object_name = "ground"
        result_path = LAFAN_RUNS / method / task / f"{task}.npz"
        data_dir = LAFAN_INPUT
        motion_cfg = MotionDataConfig(data_format="lafan", robot_type="g1")
        data_type = "robot_only"
        constants = create_task_constants(RobotConfig(robot_type="g1"), motion_cfg, object_name=object_name)

    if not result_path.exists():
        return {"dataset": dataset, "task_name": task, "method": method, "status": "missing", "result": str(result_path)}

    try:
        evaluator = RetargetingEvaluator(
            robot_model_path=constants.ROBOT_URDF_FILE,
            object_model_path=getattr(constants, "OBJECT_URDF_FILE", None),
            object_name=object_name,
            demo_joints=constants.DEMO_JOINTS,
            joints_mapping=constants.JOINTS_MAPPING,
            visualize=False,
            constants=constants,
        )
        if data_type == "robot_object":
            metrics = evaluator.evaluate_trajectory(task, str(result_path), str(data_dir))
        else:
            metrics = evaluator.evaluate_robot_only_trajectory(task, str(result_path), str(data_dir))
        if metrics is None:
            raise RuntimeError("official evaluator returned None")
        res_npz = np.load(result_path, allow_pickle=True)
        row: dict[str, Any] = {
            "dataset": dataset, "task_name": task, "method": method, "status": "ok",
            "result": str(result_path), "n_frames": int(res_npz["qpos"].shape[0]),
            "fps": float(np.asarray(res_npz["fps"]).reshape(-1)[0]) if np.asarray(res_npz["fps"]).size else float("nan"),
            "opt_cost_last_frame": float(np.asarray(metrics["opt_cost"]).reshape(-1)[0]),
            "frame_cost_mean": _mean_array(res_npz["frame_costs"]),
            "actual_sqp_iterations_mean": _mean_array(res_npz["actual_sqp_iterations"]),
            "base_sqp_iterations_mean": _mean_array(res_npz["base_sqp_iterations"]),
            "semantic_extra_iterations_mean": _mean_array(res_npz["semantic_extra_iterations"]),
            "penetration_duration": float(metrics["penetration_duration"]),
            "penetration_max_depth_mean": _mean_array(metrics["penetration_max_depths"]),
            "sliding_duration": float(metrics["sliding_duration"]),
            "max_toe_sliding_velocity_mean": _mean_array(metrics["max_toe_sliding_velocities"]),
        }
        if "contact_preservation" in metrics:
            row["contact_preservation"] = float(metrics["contact_preservation"])
        return row
    except Exception as exc:  # keep the 39-task report complete
        return {"dataset": dataset, "task_name": task, "method": method, "status": "failed",
                "result": str(result_path), "error": f"{type(exc).__name__}: {exc}"}


def aggregate(rows: list[dict[str, Any]], tasks: list[tuple[str, str]]) -> dict[str, Any]:
    scalar_keys = [
        "opt_cost_last_frame", "frame_cost_mean", "actual_sqp_iterations_mean",
        "base_sqp_iterations_mean", "semantic_extra_iterations_mean", "penetration_duration",
        "penetration_max_depth_mean", "sliding_duration", "max_toe_sliding_velocity_mean",
        "contact_preservation",
    ]
    out: dict[str, Any] = {"n_tasks_requested": len(tasks), "n_tasks_ok": int(sum(r.get("status") == "ok" for r in rows))}
    for k in scalar_keys:
        vals = [float(r[k]) for r in rows if r.get("status") == "ok" and k in r and np.isfinite(float(r[k]))]
        out[k] = {"mean_over_tasks": float(np.mean(vals)) if vals else float("nan"), "std_over_tasks": float(np.std(vals)) if vals else float("nan"), "n": len(vals)}
    return out


def main() -> None:
    jobs = [("OMOMO", t, m) for m in ("original", "semantic_b4") for t in TASK_OBJECTS]
    jobs += [("LAFAN1", t, m) for m in ("original", "uniform") for t in LAFAN_TASKS]
    rows = [evaluate_one(*job) for job in jobs]
    report: dict[str, Any] = {"protocol": {"omomo_method": "Original vs Semantic B4", "lafan_method": "Original vs Uniform-2 (robot-only; no object semantic events)", "lafan_input_stride": 20}, "rows": rows, "aggregate": {}}
    for dataset, methods in (("OMOMO", ("original", "semantic_b4")), ("LAFAN1", ("original", "uniform"))):
        report["aggregate"][dataset] = {}
        tasks = TASK_OBJECTS if dataset == "OMOMO" else {t: "ground" for t in LAFAN_TASKS}
        for method in methods:
            subset = [r for r in rows if r["dataset"] == dataset and r["method"] == method]
            report["aggregate"][dataset][method] = aggregate(subset, [(dataset, t) for t in tasks])
    for method in ("original", "semantic_b4", "uniform"):
        subset = [r for r in rows if r["method"] == method]
        requested = [(r["dataset"], r["task_name"]) for r in subset]
        report["aggregate"]["all39_" + method] = aggregate(subset, requested)
    # The unified 39-task comparison uses the semantic pipeline for OMOMO and
    # the robot-only Uniform-2 fallback for LAFAN (there is no object/event
    # stream in LAFAN for the object-semantic mode).
    ours = [r for r in rows if (r["dataset"] == "OMOMO" and r["method"] == "semantic_b4")
            or (r["dataset"] == "LAFAN1" and r["method"] == "uniform")]
    original = [r for r in rows if r["method"] == "original"]
    report["aggregate"]["all39_original_unified"] = aggregate(original, [(r["dataset"], r["task_name"]) for r in original])
    report["aggregate"]["all39_ours_unified"] = aggregate(ours, [(r["dataset"], r["task_name"]) for r in ours])
    out = ROOT / "exp/retargeting/ALL39_evaluation.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out), "rows": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
