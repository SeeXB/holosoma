#!/usr/bin/env python3
"""Re-resolve existing VLM event functions on official InterMimic trajectories.

The VLM action/function JSON remains unchanged.  Only the trajectory used to
evaluate its declarative predicates is replaced, so the resolved frame windows
match the same official ``.pt`` consumed by retargeting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from holosoma_retargeting.config_types.data_type import SMPLH_DEMO_JOINTS
from holosoma_retargeting.semantic_keyframes.pipeline import (
    dynamic_plan_validation_issues,
    dynamic_resolution_validation_issues,
    execute_dynamic_plan,
    load_retargeting_bundle_signals,
    validate_semantic_keyframe_json,
)
from holosoma_retargeting.src.utils import load_intermimic_data

from prepare_batch_retarget_inputs import TASK_OBJECTS


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "src/holosoma_retargeting/holosoma_retargeting/demo_data"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bundle_name(task: str) -> str:
    return "sub03_largebox3" if task == "sub3_largebox_003" else task


def event_rows(payload: dict) -> list[dict[str, int | str]]:
    return [
        {"event": event["event"], **event["windows"][0]}
        for event in payload["events"]
    ]


def export_signal_bundle(reference: Path, output: Path, object_name: str, fps: float) -> None:
    joints, object_poses = load_intermimic_data(str(reference))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        object_name=np.asarray(object_name),
        fps=np.asarray(fps, dtype=np.float64),
        human_joints=np.asarray(joints, dtype=np.float32),
        smplh_joint_names=np.asarray(SMPLH_DEMO_JOINTS),
        human_joint_layout=np.asarray("smplh_retarget_v1"),
        frame_ids=np.arange(len(joints), dtype=np.int32),
        object_poses_wxyz_xyz=np.asarray(object_poses, dtype=np.float32),
        source_kind=np.asarray("official_intermimic_pt"),
        source_reference_pt=np.asarray(str(reference.resolve())),
        source_reference_sha256=np.asarray(sha256(reference)),
    )


def process_task(args: argparse.Namespace, task: str) -> dict:
    object_name = TASK_OBJECTS[task]
    reference = args.input_root / f"{task}.pt"
    source_dir = args.bundle_root / bundle_name(task)
    source_plan = source_dir / "semantic_keyframes" / f"{task}_dynamic.event_plan.json"
    source_resolved = source_dir / "semantic_keyframes" / f"{task}_dynamic.json"
    source_video = source_dir / "videos" / f"{task}_rerender.mp4"
    task_root = args.output_root / task
    signal_bundle = task_root / "input" / "intermimic_semantic_bundle.npz"
    output = args.plan_root / "omomo" / task / "semantic_plan.json"
    event_output = output.with_name("semantic_plan.event_plan.json")

    for path in (reference, source_plan, source_resolved):
        if not path.is_file():
            raise FileNotFoundError(path)

    export_signal_bundle(reference, signal_bundle, object_name, args.fps)
    plan = json.loads(source_plan.read_text(encoding="utf-8"))
    plan_issues = dynamic_plan_validation_issues(plan)
    if plan_issues:
        raise ValueError(f"event plan is invalid: {plan_issues}")

    signals, _thresholds, fps = load_retargeting_bundle_signals(signal_bundle)
    result = execute_dynamic_plan(plan, signals, fps=fps)
    resolution_issues = dynamic_resolution_validation_issues(
        result, len(next(iter(signals.values())))
    )
    if resolution_issues:
        raise ValueError(f"InterMimic resolution is invalid: {resolution_issues}")

    old_result = json.loads(source_resolved.read_text(encoding="utf-8"))
    result["generation_metadata"] = {
        "planning_mode": "existing_vlm_functions_reresolved_on_official_intermimic_pt",
        "vlm_called": False,
        "action_source_video": str(source_video.resolve()),
        "source_event_plan": str(source_plan.resolve()),
        "source_event_plan_sha256": sha256(source_plan),
        "signal_source_kind": "official_intermimic_pt",
        "signal_source_bundle": str(signal_bundle.resolve()),
        "signal_source_reference_pt": str(reference.resolve()),
        "signal_source_reference_pt_sha256": sha256(reference),
        "threshold_policy": "per-signal trajectory quantiles",
    }
    validate_semantic_keyframe_json(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    event_output.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {
        "task": task,
        "status": "ok",
        "reference_pt": str(reference.resolve()),
        "reference_pt_sha256": sha256(reference),
        "signal_bundle": str(signal_bundle.resolve()),
        "source_event_plan": str(source_plan.resolve()),
        "output": str(output.resolve()),
        "old_events": event_rows(old_result),
        "new_events": event_rows(result),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=sorted(TASK_OBJECTS), required=True)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DATA / "omomo/official_inputs",
    )
    parser.add_argument(
        "--bundle-root",
        type=Path,
        default=DATA / "omomo/bundles",
        help="Existing VLM event-function and video provenance root.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "exp/semantic_plans/dynamic_json_intermimic_v6",
    )
    parser.add_argument("--plan-root", type=Path)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()
    args.input_root = args.input_root.resolve()
    args.bundle_root = args.bundle_root.resolve()
    args.output_root = args.output_root.resolve()
    args.plan_root = (
        args.plan_root.resolve()
        if args.plan_root is not None
        else args.output_root / "retarget_plan_root"
    )

    rows = []
    for task in args.tasks:
        try:
            row = process_task(args, task)
        except Exception as exc:  # Retain per-task diagnostics for batch runs.
            row = {"task": task, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    summary = {
        "schema": "holosoma.intermimic_semantic_reresolution.v1",
        "tasks": args.tasks,
        "results": rows,
    }
    summary_path = args.output_root / "reresolution_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if any(row["status"] != "ok" for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
