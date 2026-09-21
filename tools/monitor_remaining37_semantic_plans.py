#!/usr/bin/env python3
"""Resume missing semantic plans with bounded retries and a local completion report.

This process never starts training or uploads W&B data. It has no chat push
notification channel; MONITOR.md and COMPLETE.md are its local notifications.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import time
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
from holosoma.managers.command.semantic_transition_sampler import load_semantic_transitions
from holosoma.utils.semantic_contacts import (
    G1_BODY_PART_LINKS,
    load_semantic_contact_parts,
    resolve_semantic_contact_links,
)
from holosoma_retargeting.semantic_keyframes.pipeline import (
    dynamic_resolution_validation_issues,
    execute_dynamic_plan,
    load_env_file,
    load_retargeting_bundle_signals,
    validate_semantic_keyframe_json,
)


def now():
    return datetime.now(timezone.utc).isoformat()


def validate_output(root, dataset, task):
    path = root / dataset / task / "semantic_plan.json"
    plan_path = path.with_name("semantic_plan.event_plan.json")
    if not path.is_file() or not plan_path.is_file():
        return {"status": "pending", "output": str(path)}
    try:
        result = json.loads(path.read_text())
        plan = json.loads(plan_path.read_text())
        if result.get("generation_metadata", {}).get("contact_prompt_version"):
            from holosoma_retargeting.semantic_keyframes.local_phases import validate_contact_review

            for segment in plan["segments"]:
                for phase in segment["visual_plan"]["phases"]:
                    validate_contact_review(phase)
            for event in result["events"]:
                validate_contact_review(event)
            if not any(
                review.get(region) in ("left", "right", "both")
                for event in result["events"]
                for review in (event["contact_review"],)
                for region in ("hand", "wrist", "forearm")
            ):
                raise ValueError("object-interaction plan has no identified hand/wrist/forearm contact; visual review required")
        constraints_file = root / "task_body_constraints.json"
        if constraints_file.is_file():
            expected = json.loads(constraints_file.read_text()).get(f"{dataset}/{task}")
            if expected != plan.get("body_part_constraints"):
                raise ValueError("plan does not match the configured task body-part constraint")
        bundle = (
            DATA / "omomo/bundles" / task / "input/omomo_gt_sequence.npz"
            if dataset == "omomo"
            else root / dataset / task / "lafan_gt_sequence.npz"
        )
        signals, _, fps = load_retargeting_bundle_signals(bundle)
        validate_semantic_keyframe_json(result)
        resolved = execute_dynamic_plan(plan, signals, fps)
        issues = dynamic_resolution_validation_issues(resolved, len(next(iter(signals.values()))))
        if issues or result["events"] != resolved["events"] or result["fps"] != resolved["fps"]:
            raise ValueError(f"stored resolution differs from GT or fails timeline validation: {issues}")
        transitions = load_semantic_transitions(
            path, motion_fps=fps, motion_time_step_total=len(next(iter(signals.values())))
        )
        parts = load_semantic_contact_parts(path)
        # Authored names include all G1 semantic groups. Runtime checks the
        # actual simulator bodies again, including fixed-hand merge fallback.
        mapping = resolve_semantic_contact_links(parts, {n for names in G1_BODY_PART_LINKS.values() for n in names})
        return {
            "status": "validated",
            "output": str(path),
            "body_parts": list(parts),
            "transitions": len(transitions),
            "contact_mapping": mapping,
            "visual_review": "pending",
        }
    except Exception as exc:
        return {"status": "invalid", "output": str(path), "error": f"{type(exc).__name__}: {exc}"}


def snapshot(root, tasks):
    return [{"dataset": ds, "task": task, **validate_output(root, ds, task)} for ds, task in tasks]


def failure_status(error):
    if "insufficient_quota" in error or "quota exhausted" in error:
        return "quota_blocked"
    if "VLM failed to produce a valid" in error:
        return "failed_validation"
    return "request_failed"


def restore_progress(audit, rows):
    """Recover failure budgets and diagnostics without mistaking failures for queued work."""
    attempts, validation_failures = {}, {}
    by_task = {row["task"]: row for row in rows}
    round_id = 0
    for directory in sorted((audit / "monitor_attempts").glob("round_*")):
        round_id = max(round_id, int(directory.name.split("_")[-1]))
        for file in sorted(directory.glob("*/*/result.json")):
            outcome = json.loads(file.read_text())
            task = outcome["task"]
            attempts[task] = attempts.get(task, 0) + 1
            if outcome["status"] == "ok":
                continue
            error = outcome.get("error", "unknown generation error")
            if failure_status(error) == "failed_validation":
                validation_failures[task] = validation_failures.get(task, 0) + 1
            if task in by_task and by_task[task]["status"] != "validated":
                by_task[task].update(status=failure_status(error), error=error, result_file=str(file))
    return attempts, validation_failures, round_id


def publish(audit, rows, state, attempts, deadline, **extra):
    valid = sum(row["status"] == "validated" for row in rows)
    if valid == len(rows):
        state = "complete_pending_visual_review"
    payload = {
        "updated_at": now(),
        "pid": os.getpid(),
        "state": state,
        "validated": valid,
        "total": len(rows),
        "deadline_epoch": deadline,
        "new_training_started": False,
        "attempts": attempts,
        "results": rows,
        **extra,
    }
    write_json(audit / "monitor_status.json", payload)
    lines = [
        "# Semantic plan 生成监测",
        "",
        f"更新：{payload['updated_at']}",
        "",
        f"状态：**{state}**；本地校验通过 **{valid}/{len(rows)}**，视觉复核单列，未启动新版训练。",
        "",
        "此文件由后台监测器自动更新；完成后写 COMPLETE.md。历史 REVIEW.md 是上轮快照。",
        "",
    ]
    for key in ("current_task", "last_error", "next_retry_at"):
        if key in extra:
            lines.append(f"- {key}: {extra[key]}")
    for active in extra.get("active_tasks", []):
        lines.append(f"- 正在生成：{active['task']}；片段进度 {active.get('completed_segments', 0)}/{active.get('total_segments', '?')}")
    failed = sum(row["status"] in {"failed_validation", "request_failed", "quota_blocked", "invalid"} for row in rows)
    lines += [
        f"- 已失败/阻塞：{failed}；其余为未完成或正在生成。",
        "",
        "| 数据集 | 任务 | 状态 | 最近失败原因 |",
        "|---|---|---|---|",
    ]
    for row in rows:
        target = os.path.relpath(Path(row["output"]).resolve(), audit.resolve())
        status = f"[{row['status']}]({target})" if row["status"] == "validated" else row["status"]
        if extra.get("current_task") == row["task"]:
            status = "generating" + (f"（上次：{row['status']}）" if row["status"] != "pending" else "")
        error = row.get("error", "").replace("|", "/").replace("\n", " ")
        lines.append(f"| {row['dataset']} | {row['task']} | {status} | {error} |")
    (audit / "MONITOR.md").write_text("\n".join(lines) + "\n")
    if valid == len(rows):
        (audit / "COMPLETE.md").write_text(
            f"# 37 个 semantic plan 已生成\n\n{now()}\n\n"
            "37/37 已通过结构、GT 时间线、采样器与部位映射检查。视觉复核及新版训练尚未进行。\n\n"
            "逐任务文件见 [MONITOR.md](MONITOR.md)。\n"
        )
    return valid


def run(args):
    tasks = [("omomo", task) for task in TASK_OBJECTS if task not in COMPLETED]
    tasks += [("lafan", task) for task in LAFAN_TASKS]
    tasks.sort(key=lambda x: (x[1] != "sub1_largetable_028", x[0], x[1]))
    root = DATA / "semantic_keyframes" / VERSION
    audit = Path("exp/semantic_plans") / VERSION
    audit.mkdir(parents=True, exist_ok=True)
    with (audit / "monitor.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        deadline = time.time() + args.max_hours * 3600
        rows = snapshot(root, tasks)
        previous_status = audit / "monitor_status.json"
        if previous_status.is_file():
            previous = json.loads(previous_status.read_text())
            deadline = previous.get("deadline_epoch", deadline)
        attempts, validation_failures, round_id = restore_progress(audit, rows)
        while time.time() < deadline:
            if publish(audit, rows, "checking", attempts, deadline) == len(tasks):
                return
            pending = [
                row
                for row in rows
                if row["status"] != "validated" and validation_failures.get(row["task"], 0) < args.max_validation_rounds
            ]
            # Finish unattempted tasks before spending another repair round on
            # known failures after a monitor restart.
            pending.sort(key=lambda row: validation_failures.get(row["task"], 0))
            if not pending:
                publish(audit, rows, "needs_plan_repair", attempts, deadline)
                return
            round_id += 1
            last_error = ""
            quota = False
            for row in pending:
                if time.time() >= deadline:
                    break
                ds, task = row["dataset"], row["task"]
                attempts[task] = attempts.get(task, 0) + 1
                publish(audit, rows, "generating", attempts, deadline, current_task=task)
                # Load the authorized endpoint configuration without logging secrets.
                load_env_file(Path("src/holosoma_retargeting/.env"))
                attempt_dir = audit / "monitor_attempts" / f"round_{round_id:03d}"
                outcome = generate_one(ds, task, root, attempt_dir, args.max_repairs, True)
                write_json(attempt_dir / ds / task / "result.json", outcome)
                checked = validate_output(root, ds, task)
                row.update(checked)
                if outcome["status"] != "ok":
                    last_error = outcome.get("error", "unknown generation error")
                    row.update(
                        status=failure_status(last_error),
                        error=last_error,
                        result_file=str(attempt_dir / ds / task / "result.json"),
                    )
                    if "insufficient_quota" in last_error:
                        quota = True
                        break
                    if "VLM failed to produce a valid" in last_error:
                        validation_failures[task] = validation_failures.get(task, 0) + 1
                    else:
                        # Avoid hammering network/auth failures across all 37 inputs.
                        break
                publish(audit, rows, "generating", attempts, deadline)
            if publish(audit, rows, "round_finished", attempts, deadline) == len(tasks):
                return
            retry_at = min(deadline, time.time() + args.poll_seconds)
            publish(
                audit,
                rows,
                "waiting_for_quota" if quota else "waiting_to_retry",
                attempts,
                deadline,
                last_error=last_error,
                next_retry_at=datetime.fromtimestamp(retry_at, timezone.utc).isoformat(),
            )
            while time.time() < retry_at:
                time.sleep(min(30, max(0.01, retry_at - time.time())))
        publish(audit, rows, "monitor_deadline_reached", attempts, deadline)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=float, default=900)
    parser.add_argument("--max-hours", type=float, default=24)
    parser.add_argument("--max-repairs", type=int, default=5)
    parser.add_argument("--max-validation-rounds", type=int, default=2)
    args = parser.parse_args()
    if args.poll_seconds < 60 or args.max_hours <= 0:
        parser.error("poll-seconds must be >=60 and max-hours must be positive")
    run(args)


if __name__ == "__main__":
    main()
