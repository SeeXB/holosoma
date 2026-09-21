#!/usr/bin/env python3
"""Versioned G1 anatomy plans for the 18 unfinished OMOMO and 19 LAFAN tasks.

LAFAN uses full-rate native joints, 30 Hz as in the repository conversion
config, with Y/Z exchanged exactly as in its retarget loader. Sparse skeleton
views are visual evidence, not exhaustive verification of every motion cycle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from holosoma_retargeting.config_types.data_type import LAFAN_DEMO_JOINTS
from holosoma_retargeting.semantic_keyframes.local_phases import generate_localized_semantic_keyframes, skeleton_images
from holosoma_retargeting.semantic_keyframes.pipeline import (
    VLMQuotaError,
    dynamic_resolution_validation_issues,
    execute_dynamic_plan,
    load_env_file,
    load_retargeting_bundle_signals,
    validate_semantic_keyframe_json,
)
from prepare_batch_retarget_inputs import TASK_OBJECTS
from prepare_lafan_batch_inputs import TASKS as LAFAN_TASKS

DATA = Path("src/holosoma_retargeting/holosoma_retargeting/demo_data")
VERSION = "g1_anatomy_v2_20260920"
COMPLETED = {"sub3_largebox_003", "sub10_largebox_089"}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def generate_one(dataset, task, root, audit, max_repairs, force, prepare_only=False, stop_event=None):
    if stop_event is not None and stop_event.is_set():
        return {
            "dataset": dataset,
            "task": task,
            "status": "blocked",
            "error": "provider quota exhausted earlier in batch",
        }
    directory = root / dataset / task
    output = directory / "semantic_plan.json"
    try:
        review_path = root / "PLAN_REVIEW_STATUS.json"
        if not prepare_only and review_path.exists() and json.loads(review_path.read_text()).get("block_legacy_regeneration"):
            raise ValueError("Localized phase generation was rejected by the user; use whole-clip trajectory event programs, not phase regeneration")
        if dataset == "omomo":
            original = DATA / "omomo/bundles" / task
            video = original / "videos" / f"{task}_rerender.mp4"
            bundle = original / "input/omomo_gt_sequence.npz"
            source = bundle
        else:
            source = DATA / "lafan" / f"{task}.npy"
            joints = np.load(source)[:, :, [0, 2, 1]]
            directory.mkdir(parents=True, exist_ok=True)
            bundle = directory / "lafan_gt_sequence.npz"
            np.savez_compressed(
                bundle,
                human_joints=joints,
                smplh_joint_names=np.asarray(LAFAN_DEMO_JOINTS),
                human_joint_layout=np.asarray("lafan_z_up_v1"),
                frame_ids=np.arange(len(joints)),
                fps=np.asarray(30.0),
                source_file=np.asarray(str(source)),
                fps_source=np.asarray("repository data_conversion input_fps=30"),
            )
            video = directory / "skeleton_views"
        signals, _, fps = load_retargeting_bundle_signals(bundle)
        if prepare_only:
            if dataset == "lafan":
                skeleton_images(joints, video, task)
            elif not video.is_file():
                raise FileNotFoundError(video)
            return {
                "dataset": dataset,
                "task": task,
                "status": "prepared",
                "source": str(source),
                "bundle": str(bundle),
                "visual_source": str(video),
                "fps": fps,
                "frames": len(next(iter(signals.values()))),
                "available_signals": sorted(signals),
            }
        event_file = output.with_name("semantic_plan.event_plan.json")
        if not force and output.exists() and event_file.exists():
            result = json.loads(output.read_text())
            plan = json.loads(event_file.read_text())
            validate_semantic_keyframe_json(result)
            resolved = execute_dynamic_plan(plan, signals, fps)
            issues = dynamic_resolution_validation_issues(resolved, len(next(iter(signals.values()))))
            if issues or result["events"] != resolved["events"]:
                raise ValueError(f"existing plan validation failed: {issues}")
        else:
            constraints_file = root / "task_body_constraints.json"
            constraints = json.loads(constraints_file.read_text()) if constraints_file.is_file() else {}
            result = generate_localized_semantic_keyframes(
                video=video,
                bundle_file=bundle,
                output=output,
                max_repairs=max_repairs,
                audit_dir=audit / dataset / task,
                body_part_constraints=constraints.get(f"{dataset}/{task}"),
            )
        parts = sorted({part for event in result["events"] for part in event["body_parts"]})
        return {
            "dataset": dataset,
            "task": task,
            "status": "ok",
            "output": str(output),
            "body_parts": parts,
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "fps": fps,
            "frames": len(next(iter(signals.values()))),
            "actions": [
                {"action": e["event"], "body_parts": e["body_parts"], "windows": e["windows"]} for e in result["events"]
            ],
        }
    except Exception as exc:
        if isinstance(exc, VLMQuotaError) and stop_event is not None:
            stop_event.set()
        return {
            "dataset": dataset,
            "task": task,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DATA / "semantic_keyframes" / VERSION)
    parser.add_argument("--audit-root", type=Path, default=Path("exp/semantic_plans") / VERSION)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--max-repairs", type=int, default=5)
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--prepare-only", action="store_true", help="Prepare local inputs without API calls")
    args = parser.parse_args()
    if not args.prepare_only:
        load_env_file(Path("src/holosoma_retargeting/.env"))
    tasks = [("omomo", task) for task in TASK_OBJECTS if task not in COMPLETED] + [
        ("lafan", task) for task in LAFAN_TASKS
    ]
    if args.tasks:
        unknown = set(args.tasks) - {task for _, task in tasks}
        if unknown:
            parser.error(f"unknown or completed tasks: {sorted(unknown)}")
        tasks = [row for row in tasks if row[1] in args.tasks]
    rows = []
    stop_event = threading.Event()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                generate_one,
                ds,
                task,
                args.output_root,
                args.audit_root,
                args.max_repairs,
                args.force,
                args.prepare_only,
                stop_event,
            )
            for ds, task in tasks
        ]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            write_json(
                args.audit_root / ("preflight.json" if args.prepare_only else "summary.json"),
                {
                    "version": VERSION,
                    "total": len(tasks),
                    "completed": len(rows),
                    "results": sorted(rows, key=lambda r: (r["dataset"], r["task"])),
                },
            )
            print(
                json.dumps({k: v for k, v in row.items() if k not in ("traceback", "actions")}, ensure_ascii=False),
                flush=True,
            )
    if any(row["status"] not in ("ok", "prepared") for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
