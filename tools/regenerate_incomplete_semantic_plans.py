#!/usr/bin/env python3
"""Regenerate only missing/invalid plans with local visual phases and live status."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path

from generate_remaining37_semantic_plans import (
    COMPLETED,
    DATA,
    LAFAN_TASKS,
    TASK_OBJECTS,
    VERSION,
    generate_one,
    write_json,
)
from holosoma_retargeting.semantic_keyframes.pipeline import load_env_file
from monitor_remaining37_semantic_plans import failure_status, publish, snapshot, validate_output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-repairs", type=int, default=2)
    parser.add_argument("--tasks", nargs="+")
    args = parser.parse_args()
    root = DATA / "semantic_keyframes" / VERSION
    audit = Path("exp/semantic_plans") / VERSION
    tasks = [("omomo", task) for task in TASK_OBJECTS if task not in COMPLETED] + [
        ("lafan", task) for task in LAFAN_TASKS
    ]
    with (audit / "monitor.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rows = snapshot(root, tasks)
        pending = [
            row for row in rows if row["status"] != "validated" and (not args.tasks or row["task"] in args.tasks)
        ]
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = audit / "localized_regeneration" / stamp
        protected = {}
        for row in rows:
            if row["status"] == "validated":
                file = Path(row["output"])
                for path in (file, file.with_name(file.stem + ".event_plan.json")):
                    protected[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        write_json(
            run_dir / "manifest.json",
            {
                "tasks": [r["task"] for r in pending],
                "preserved_files_sha256": protected,
                "schema": "holosoma.localized_visual_phases.v1",
                "max_repairs": args.max_repairs,
            },
        )
        load_env_file(Path("src/holosoma_retargeting/.env"))
        stop = threading.Event()
        attempts = {}
        outcomes = []
        deadline = time.time() + 24 * 3600
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            future_rows = {
                pool.submit(
                    generate_one, row["dataset"], row["task"], root, run_dir, args.max_repairs, True, False, stop
                ): row
                for row in pending
            }
            remaining = set(future_rows)
            while remaining:
                finished, remaining = wait(remaining, timeout=15, return_when=FIRST_COMPLETED)
                for future in finished:
                    row = future_rows[future]
                    result = future.result()
                    outcomes.append(result)
                    write_json(run_dir / row["dataset"] / row["task"] / "result.json", result)
                    checked = validate_output(root, row["dataset"], row["task"])
                    row.update(checked)
                    attempts[row["task"]] = 1
                    if result["status"] != "ok":
                        row["status"] = failure_status(result.get("error", ""))
                        row["error"] = result.get("error", "")
                    print(
                        json.dumps(
                            {"task": row["task"], "status": row["status"], "error": row.get("error", "")},
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                active = []
                for future in remaining:
                    if future.running():
                        row = future_rows[future]
                        progress = run_dir / row["dataset"] / row["task"] / "progress.json"
                        info = {"task": row["task"]}
                        if progress.exists():
                            try:
                                info.update(json.loads(progress.read_text()))
                            except json.JSONDecodeError:
                                pass
                        active.append(info)
                publish(
                    audit,
                    rows,
                    "generating_localized_phases",
                    attempts,
                    deadline,
                    active_tasks=active,
                    run_dir=str(run_dir),
                )
                write_json(
                    run_dir / "summary.json",
                    {"total_requested": len(pending), "returned": len(outcomes), "results": outcomes},
                )
        changed = [
            path for path, digest in protected.items() if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest
        ]
        if changed:
            raise RuntimeError(f"pre-existing completed plans changed: {changed}")
        count = sum(r["status"] == "validated" for r in rows)
        publish(
            audit,
            rows,
            "complete_pending_visual_review" if count == 37 else "incomplete_needs_retry",
            attempts,
            deadline,
            run_dir=str(run_dir),
        )
        write_json(run_dir / "preservation_check.json", {"unchanged": len(protected), "changed": changed})
        print(
            json.dumps({"validated": count, "total": 37, "preserved_files": len(protected), "run_dir": str(run_dir)}),
            flush=True,
        )
        if count != 37:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
