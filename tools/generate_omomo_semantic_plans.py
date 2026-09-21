#!/usr/bin/env python3
"""Generate and validate dynamic VLM semantic plans for selected OMOMO tasks."""

from __future__ import annotations

import argparse
import json
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from prepare_batch_retarget_inputs import TASK_OBJECTS

from holosoma_retargeting.semantic_keyframes.pipeline import (
    _extract_json_value,
    dynamic_plan_validation_issues,
    dynamic_resolution_validation_issues,
    execute_dynamic_plan,
    generate_dynamic_semantic_keyframes,
    load_env_file,
    load_retargeting_bundle_signals,
    normalize_dynamic_plan,
    validate_semantic_keyframe_json,
)


def _bundle_name(task: str) -> str:
    return "sub03_largebox3" if task == "sub3_largebox_003" else task


def _paths(bundle_root: Path, task: str) -> tuple[Path, Path, Path]:
    root = bundle_root / _bundle_name(task)
    return (
        root / "videos" / f"{task}_rerender.mp4",
        root / "input" / "omomo_gt_sequence.npz",
        root / "semantic_keyframes" / f"{task}_dynamic.json",
    )


def _existing_plan_is_valid(bundle: Path, output: Path) -> bool:
    event_plan = output.with_name(f"{output.stem}.event_plan.json")
    if not output.is_file() or not event_plan.is_file():
        return False
    try:
        plan = json.loads(event_plan.read_text(encoding="utf-8"))
        if dynamic_plan_validation_issues(plan):
            return False
        signals, _, fps = load_retargeting_bundle_signals(bundle)
        resolved = execute_dynamic_plan(plan, signals, fps=fps)
        return not dynamic_resolution_validation_issues(
            resolved, len(next(iter(signals.values())))
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _recover_existing_vlm_plan(bundle: Path, output: Path, audit_dir: Path | None = None) -> Path | None:
    """Promote a previously saved VLM response when it passes current checks."""
    signals, _, fps = load_retargeting_bundle_signals(bundle)
    frame_count = len(next(iter(signals.values())))
    candidates: list[tuple[tuple[int, int, int, str], Path, dict, dict]] = []
    paths = set(output.parent.glob("*.event_plan.json"))
    paths.update(output.parent.glob("*.vlm_attempt_*.txt"))
    if audit_dir is not None:
        paths.update(audit_dir.glob("*.vlm_attempt_*.txt"))
    for path in sorted(paths):
        try:
            if path.name.endswith(".event_plan.json"):
                plan = json.loads(path.read_text(encoding="utf-8"))
            else:
                plan = normalize_dynamic_plan(_extract_json_value(path.read_text(encoding="utf-8")))
            if dynamic_plan_validation_issues(plan):
                continue
            result = execute_dynamic_plan(plan, signals, fps=fps)
            if dynamic_resolution_validation_issues(result, frame_count):
                continue
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        windows = [event["windows"][0] for event in result["events"]]
        zero_count = sum(window["end_frame"] <= window["start_frame"] for window in windows)
        span = windows[-1]["end_frame"] - windows[0]["start_frame"]
        score = (zero_count, -span, -len(windows), path.name)
        candidates.append((score, path, plan, result))
    if not candidates:
        return None
    _, source, plan, result = min(candidates, key=lambda row: row[0])
    plan_output = output.with_name(f"{output.stem}.event_plan.json")
    result["generation_metadata"] = {
        "planning_mode": "dynamic_vlm_actions_and_functions",
        "signal_source": "explicit_retargeting_bundle",
        "source_bundle": str(bundle.resolve()),
        "recovered_from_existing_vlm_artifact": str(source.resolve()),
        "recovery_policy": "current schema and resolved-timeline validation; minimum zero-duration count",
    }
    validate_semantic_keyframe_json(result)
    plan_output.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return source


def _generate_one(
    task: str,
    bundle_root: Path,
    sample_count: int,
    max_repairs: int,
    force: bool,
    recover_existing: bool,
    recover_only: bool,
    audit_root: Path = Path("exp/omomo_cari4d"),
) -> dict[str, object]:
    video, bundle, output = _paths(bundle_root, task)
    audit_dir = audit_root / _bundle_name(task) / "semantic_keyframes"
    output.parent.mkdir(parents=True, exist_ok=True)
    if not video.is_file() or not bundle.is_file():
        missing = [str(path) for path in (video, bundle) if not path.is_file()]
        return {"task": task, "status": "failed", "error": f"missing inputs: {missing}"}
    if not force and _existing_plan_is_valid(bundle, output):
        return {"task": task, "status": "cached", "output": str(output.resolve())}
    if recover_existing:
        recovered = _recover_existing_vlm_plan(bundle, output, audit_dir)
        if recovered is not None:
            return {
                "task": task,
                "status": "recovered",
                "output": str(output.resolve()),
                "source": str(recovered.resolve()),
            }
    if recover_only:
        return {
            "task": task,
            "status": "failed",
            "error": "no previously saved VLM response passes current resolved-timeline validation",
        }
    try:
        result = generate_dynamic_semantic_keyframes(
            video=video,
            bundle_file=bundle,
            output=output,
            sample_count=sample_count,
            max_repairs=max_repairs,
            audit_dir=audit_dir,
        )
        actions = [
            {
                "action": event["event"],
                "start": event["windows"][0]["start_frame"],
                "trigger": event["windows"][0]["trigger_frame"],
                "end": event["windows"][0]["end_frame"],
            }
            for event in result["events"]
        ]
        return {
            "task": task,
            "status": "ok",
            "output": str(output.resolve()),
            "actions": actions,
        }
    except Exception as exc:  # keep the batch progressing and retain exact failure context
        return {
            "task": task,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, default=Path("src/holosoma_retargeting/holosoma_retargeting/demo_data/omomo/bundles"))
    parser.add_argument("--audit-root", type=Path, default=Path("exp/omomo_cari4d"),
                        help="VLM attempt logs and diagnostics; resolved plans stay under --bundle-root")
    parser.add_argument("--env-file", type=Path,
                        default=Path("src/holosoma_retargeting/.env"))
    parser.add_argument("--tasks", nargs="+", choices=list(TASK_OBJECTS), default=None)
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--sample-count", type=int, default=12)
    parser.add_argument("--max-repairs", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--no-recover-existing",
        action="store_true",
        help="Do not try previously saved VLM responses before making a new API request.",
    )
    parser.add_argument(
        "--recover-only",
        action="store_true",
        help="Validate/promote saved VLM artifacts without making API requests.",
    )
    parser.add_argument("--summary", type=Path,
                        default=Path("exp/omomo_cari4d/selected20_semantic_summary.json"))
    args = parser.parse_args()

    load_env_file(args.env_file)
    tasks = args.tasks or list(TASK_OBJECTS)
    results: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
        futures = {
            executor.submit(
                _generate_one,
                task,
                args.bundle_root,
                args.sample_count,
                args.max_repairs,
                args.force,
                not args.no_recover_existing,
                args.recover_only,
                args.audit_root,
            ): task
            for task in tasks
        }
        for future in as_completed(futures):
            row = future.result()
            results.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)

    order = {task: index for index, task in enumerate(tasks)}
    results.sort(key=lambda row: order[str(row["task"])])
    payload = {
        "schema": "holosoma.omomo_dynamic_semantic_batch.v1",
        "tasks": tasks,
        "results": results,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8")
    failed = [row for row in results if row["status"] == "failed"]
    print(json.dumps({"total": len(results), "failed": len(failed)}, ensure_ascii=False))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
