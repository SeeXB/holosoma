#!/usr/bin/env python3
"""Run robot-only LAFAN1 retargeting for the requested 19 sequences."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from prepare_lafan_batch_inputs import TASKS


def run_one(task: str, method: str, args: argparse.Namespace) -> dict[str, object]:
    save_dir = args.output_root / "runs" / method / task
    result = save_dir / f"{task}.npz"
    log = save_dir / "retarget.log"
    save_dir.mkdir(parents=True, exist_ok=True)
    if result.exists() and not args.force:
        return {"task_name": task, "method": method, "status": "cached", "result": str(result)}
    cmd = [
        args.python, "-m", "holosoma_retargeting.examples.robot_retarget",
        "--task-type", "robot_only", "--task-name", task,
        "--data-format", "lafan", "--data-path", str(args.input_root.resolve()),
        "--save-dir", str(save_dir.resolve()), "--retargeter.no-activate-foot-sticking",
        "--semantic.mode", "original" if method == "original" else "uniform",
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(args.project_root.resolve())
    with log.open("w", encoding="utf-8") as fp:
        p = subprocess.run(cmd, cwd=args.package_dir, env=env, stdout=fp, stderr=subprocess.STDOUT, check=False)
    status = "ok" if p.returncode == 0 and result.exists() else "failed"
    return {"task_name": task, "method": method, "status": status, "returncode": p.returncode,
            "result": str(result), "log": str(log)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", type=Path, default=Path("exp/retargeting/lafan_batch/input"))
    ap.add_argument("--output-root", type=Path, default=Path("exp/retargeting/lafan_batch"))
    ap.add_argument("--project-root", type=Path, default=Path("src/holosoma_retargeting"))
    ap.add_argument("--package-dir", type=Path, default=Path("src/holosoma_retargeting/holosoma_retargeting"))
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--max-workers", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--method", choices=("original", "uniform", "both"), default="both")
    args = ap.parse_args()
    methods = ("original", "uniform") if args.method == "both" else (args.method,)
    jobs = [(t, m) for m in methods for t in TASKS]
    results = []
    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as ex:
        futs = {ex.submit(run_one, t, m, args): (t, m) for t, m in jobs}
        for fut in as_completed(futs):
            row = fut.result(); results.append(row); print(json.dumps(row), flush=True)
    results.sort(key=lambda r: (str(r["method"]), str(r["task_name"])))
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "retarget_batch_summary.json").write_text(
        json.dumps({"tasks": list(TASKS), "results": results}, indent=2) + "\n", encoding="utf-8")
    failed = [r for r in results if r["status"] == "failed"]
    print(json.dumps({"total": len(results), "failed": len(failed)}))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
