#!/usr/bin/env python3
"""Run the two registered retargeting methods on the selected OMOMO tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from prepare_batch_retarget_inputs import TASK_OBJECTS


def _run_one(task_name: str, method: str, args: argparse.Namespace) -> dict[str, object]:
    object_name = TASK_OBJECTS[task_name]
    save_dir = args.output_root / "runs" / method / task_name
    # The CLI always writes ``*_original.npz`` for object-interaction runs;
    # rename it after completion so the two methods cannot be confused.
    native_result = save_dir / f"{task_name}_original.npz"
    result = save_dir / f"{task_name}_{method}.npz"
    log_path = save_dir / "retarget.log"
    save_dir.mkdir(parents=True, exist_ok=True)

    command = [
        args.python,
        "-m",
        "holosoma_retargeting.examples.robot_retarget",
        "--task-type",
        "object_interaction",
        "--task-name",
        task_name,
        "--data-format",
        "smplh",
        "--data-path",
        str(args.input_root.resolve()),
        "--save-dir",
        str(save_dir.resolve()),
        "--task-config.object-name",
        object_name,
        "--task-config.scene-xml-file",
        str((args.input_root / "scenes" / f"g1_29dof_w_{object_name}.xml").resolve()),
        "--retargeter.foot-sticking-tolerance",
        str(args.foot_sticking_tolerance),
    ]
    if args.no_foot_sticking:
        command += ["--retargeter.no-activate-foot-sticking"]
    if args.step_size is not None:
        command += ["--retargeter.step-size", str(args.step_size)]
    if method == "original":
        command += ["--semantic.mode", "original"]
    elif method == "uniform2":
        command += ["--semantic.mode", "uniform", "--semantic.profile-dir", str(save_dir.resolve())]
    elif method == "semantic_b4":
        bundle_task = "sub03_largebox3" if task_name == "sub3_largebox_003" else task_name
        plan = args.bundle_root / bundle_task / "semantic_keyframes" / f"{task_name}_dynamic.json"
        command += [
            "--semantic.mode",
            "uniform2_semantic_weight_full_event_transition_truncated_budget",
            "--semantic.exact-trigger-budget",
            "4",
            "--semantic.semantic-keyframe-path",
            str(plan.resolve()),
            "--semantic.profile-dir",
            str(save_dir.resolve()),
        ]
    else:
        raise ValueError(f"unknown method: {method}")
    # Never reuse an old result merely because its filename matches: packing,
    # pose origin, scene or semantic inputs may have changed underneath it.
    fingerprint = hashlib.sha256(json.dumps(command).encode())
    sources = [args.input_root / f"{task_name}.pt",
               args.input_root / "scenes" / f"g1_29dof_w_{object_name}.xml"]
    if method == "semantic_b4":
        sources.append(plan)
    for source in sources:
        fingerprint.update(source.read_bytes())
    signature = fingerprint.hexdigest()
    provenance = save_dir / "retarget_provenance.json"
    if result.is_file() and not args.force:
        saved = json.loads(provenance.read_text()) if provenance.is_file() else {}
        if saved.get("input_signature") == signature:
            return {"task_name": task_name, "method": method, "status": "cached", "result": str(result)}
        return {"task_name": task_name, "method": method, "status": "failed", "result": str(result),
                "error": "Existing result has missing/mismatched input provenance. Use a new output root (or explicit --force)."}
    env = dict(os.environ)
    env["PYTHONPATH"] = str(args.project_root.resolve())
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=args.package_dir,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode == 0 and native_result.is_file():
        native_result.replace(result)
    status = "ok" if completed.returncode == 0 and result.is_file() else "failed"
    if status == "ok":
        provenance.write_text(json.dumps({"input_signature": signature, "command": command,
                                          "sources": [str(p.resolve()) for p in sources]}, indent=2))
    return {
        "task_name": task_name,
        "method": method,
        "status": status,
        "returncode": completed.returncode,
        "result": str(result),
        "log": str(log_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=Path("exp/retargeting/omomo_batch/input"))
    parser.add_argument("--bundle-root", type=Path, default=Path("exp/omomo_cari4d"))
    parser.add_argument("--output-root", type=Path, default=Path("exp/retargeting/omomo_batch"))
    parser.add_argument("--project-root", type=Path, default=Path("src/holosoma_retargeting"))
    parser.add_argument("--package-dir", type=Path,
                        default=Path("src/holosoma_retargeting/holosoma_retargeting"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--foot-sticking-tolerance", type=float, default=0.02)
    parser.add_argument(
        "--no-foot-sticking",
        action="store_true",
        help="Disable foot-sticking constraints for all tasks (useful for infeasible motions).",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--step-size", type=float, default=None,
                        help="Explicit SQP trust-region override; omitted keeps the original setting.")
    parser.add_argument("--tasks", nargs="+", choices=list(TASK_OBJECTS), default=None,
                        help="Optional task subset for validated pilot reruns.")
    parser.add_argument(
        "--method",
        choices=("original", "uniform2", "semantic_b4", "training_pair", "both"),
        default="both",
        help="training_pair runs Original and Semantic B4 together; both retains the three-method benchmark.",
    )
    args = parser.parse_args()
    if args.method == "both":
        methods = ("original", "uniform2", "semantic_b4")
    elif args.method == "training_pair":
        methods = ("original", "semantic_b4")
    else:
        methods = (args.method,)
    selected_tasks = args.tasks or list(TASK_OBJECTS)
    tasks = [(task, method) for method in methods for task in selected_tasks]
    results: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
        futures = {executor.submit(_run_one, task, method, args): (task, method) for task, method in tasks}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    results.sort(key=lambda row: (str(row["method"]), str(row["task_name"])))
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "retarget_batch_summary.json").write_text(
        json.dumps({"tasks": selected_tasks, "results": results}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    failed = [row for row in results if row["status"] == "failed"]
    print(json.dumps({"total": len(results), "failed": len(failed)}, ensure_ascii=False))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
