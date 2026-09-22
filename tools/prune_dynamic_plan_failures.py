#!/usr/bin/env python3
"""Conservatively recover one unchanged valid event from over-segmented VLM plans.

Successful source plans are copied byte-for-byte.  For a failed task, every raw
response is parsed and each action is executed alone.  The longest positive
single-action interval is retained without changing any action field, body
part, predicate, signal, threshold, or explanation.  Tasks with no such action
remain failed.  This is intentionally a separate, audited derived plan set.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from holosoma_retargeting.semantic_keyframes.pipeline import (
    _extract_json_value,
    dynamic_plan_validation_issues,
    dynamic_resolution_validation_issues,
    execute_dynamic_plan,
    load_retargeting_bundle_signals,
    normalize_dynamic_plan,
    validate_semantic_keyframe_json,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bundle_name(task: str) -> str:
    return "sub03_largebox3" if task == "sub3_largebox_003" else task


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def singleton_candidates(audit_dir: Path, bundle: Path) -> list[dict[str, object]]:
    signals, _thresholds, fps = load_retargeting_bundle_signals(bundle)
    frame_count = len(next(iter(signals.values())))
    candidates: list[dict[str, object]] = []
    for attempt_path in sorted(audit_dir.glob("semantic_plan.vlm_attempt_*.txt")):
        try:
            plan = normalize_dynamic_plan(_extract_json_value(attempt_path.read_text(encoding="utf-8")))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        for action_index, action in enumerate(plan.get("actions", [])):
            singleton = {"actions": [action]}
            if dynamic_plan_validation_issues(singleton):
                continue
            try:
                resolved = execute_dynamic_plan(singleton, signals, fps=fps)
            except (KeyError, TypeError, ValueError):
                continue
            if dynamic_resolution_validation_issues(resolved, frame_count):
                continue
            window = resolved["events"][0]["windows"][0]
            duration = int(window["end_frame"]) - int(window["start_frame"])
            if duration <= 0:
                continue
            candidates.append({
                "attempt_path": attempt_path,
                "attempt_sha256": digest(attempt_path),
                "action_index": action_index,
                "action": action,
                "resolved": resolved,
                "window": window,
                "duration_frames": duration,
                "frame_count": frame_count,
            })
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-root", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    summary_path = args.generation_root / "generation_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    rows = []
    for source_row in summary["results"]:
        task = source_row["task"]
        source_dir = args.generation_root / "retarget_plan_root" / "omomo" / task
        destination = args.output_root / "omomo" / task
        if source_row["status"] == "ok":
            destination.mkdir(parents=True)
            for name in ("semantic_plan.json", "semantic_plan.event_plan.json"):
                shutil.copy2(source_dir / name, destination / name)
            rows.append({
                "task": task,
                "status": "copied_valid_plan",
                "source_plan_sha256": digest(source_dir / "semantic_plan.event_plan.json"),
                "output": str(destination / "semantic_plan.json"),
            })
            continue

        bundle = args.bundle_root / bundle_name(task) / "input/intermimic_semantic_bundle.npz"
        candidates = singleton_candidates(args.generation_root / "audit" / task, bundle)
        if not candidates:
            rows.append({
                "task": task,
                "status": "failed_no_valid_single_event",
                "source_error": source_row.get("error"),
            })
            continue
        # Prefer the broadest unchanged critical interval.  Remaining ties are
        # deterministic and favor a later repair attempt, then an earlier action.
        selected = max(
            candidates,
            key=lambda item: (
                item["duration_frames"],
                -int(item["window"]["start_frame"]),
                -int(item["action_index"]),
                str(item["attempt_path"]),
            ),
        )
        plan = {"actions": [selected["action"]]}
        resolved = selected["resolved"]
        resolved["generation_metadata"] = {
            "planning_mode": "unchanged_valid_single_event_from_oversegmented_vlm_response",
            "source_generation_summary": str(summary_path.resolve()),
            "source_generation_summary_sha256": digest(summary_path),
            "source_response": str(Path(selected["attempt_path"]).resolve()),
            "source_response_sha256": selected["attempt_sha256"],
            "selection_policy": "maximum_positive_singleton_duration; no event field modified",
            "candidate_count": len(candidates),
            "source_action_index": selected["action_index"],
            "signal_source_bundle": str(bundle.resolve()),
            "signal_source_bundle_sha256": digest(bundle),
        }
        validate_semantic_keyframe_json(resolved)
        write_json(destination / "semantic_plan.event_plan.json", plan)
        write_json(destination / "semantic_plan.json", resolved)
        rows.append({
            "task": task,
            "status": "selected_unchanged_single_event",
            "event": selected["action"]["action"],
            "body_parts": selected["action"]["body_parts"],
            "window": selected["window"],
            "duration_frames": selected["duration_frames"],
            "candidate_count": len(candidates),
            "source_response": str(selected["attempt_path"]),
            "source_response_sha256": selected["attempt_sha256"],
            "output": str(destination / "semantic_plan.json"),
        })

    report = {
        "schema": "holosoma.dynamic_plan_conservative_single_event_selection.v1",
        "policy": (
            "Valid source plans copied exactly. Failed over-segmented plans retain only the longest "
            "positive singleton action from an original VLM response; no action field is modified."
        ),
        "source_generation_summary": str(summary_path.resolve()),
        "source_generation_summary_sha256": digest(summary_path),
        "results": rows,
        "counts": {
            status: sum(row["status"] == status for row in rows)
            for status in sorted({row["status"] for row in rows})
        },
    }
    write_json(args.output_root / "SELECTION_AUDIT.json", report)
    print(json.dumps(report["counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
