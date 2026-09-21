#!/usr/bin/env python3
"""Resolve a declarative semantic event plan on a reconstructed trajectory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--event-plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository = Path(__file__).resolve().parents[1]
    package_root = repository / "src" / "holosoma_retargeting"
    sys.path.insert(0, str(package_root))
    from holosoma_retargeting.semantic_keyframes.pipeline import (  # noqa: PLC0415
        dynamic_resolution_validation_issues,
        execute_dynamic_plan,
        load_retargeting_bundle_signals,
        validate_semantic_keyframe_json,
    )

    bundle = args.bundle.expanduser().resolve()
    plan_path = args.event_plan.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not bundle.is_file():
        raise FileNotFoundError(bundle)
    if not plan_path.is_file():
        raise FileNotFoundError(plan_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    signals, _, fps = load_retargeting_bundle_signals(bundle)
    result = execute_dynamic_plan(plan, signals, fps=fps)
    frame_count = len(next(iter(signals.values())))
    resolution_issues = dynamic_resolution_validation_issues(result, frame_count)
    if resolution_issues:
        raise ValueError(f"semantic plan degenerates after GT resolution: {resolution_issues}")
    result["fps"] = fps
    result["generation_metadata"] = {
        "signal_source": "explicit_retargeting_bundle",
        "source_bundle": str(bundle),
        "source_event_plan": str(plan_path),
        "uses_dataset_motion_or_object_pose_labels": False,
        "planning_mode": plan.get("schema", "dynamic_vlm_actions_and_functions"),
        "hand_proxy_joints": ["L_Middle3", "R_Middle3"] if "object_height" in signals else ["LeftHand", "RightHand"],
        "object_position_source": "object_poses_wxyz_xyz[:,4:7]" if "object_height" in signals else None,
    }
    validate_semantic_keyframe_json(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    triggers = {
        event["event"]: event["windows"][0]["trigger_frame"]
        for event in result["events"]
    }
    print(json.dumps({"status": "PASS", "output": str(output), "fps": fps, "triggers": triggers}, indent=2))


if __name__ == "__main__":
    main()
