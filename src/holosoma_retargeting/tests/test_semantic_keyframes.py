from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest
from holosoma_retargeting.semantic_keyframes.pipeline import (
    CRITICALITY_MAPPING,
    dynamic_plan_validation_issues,
    dynamic_resolution_validation_issues,
    execute_dynamic_plan,
    ROLE_END_REQUIREMENTS,
    ROLE_ORDER,
    ROLE_TRIGGER_REQUIREMENTS,
    apply_plan_repairs,
    execute_plan,
    extract_json,
    _extract_json_value,
    plan_validation_issues,
    semantic_json_diff,
    validate_semantic_keyframe_json,
    validate_plan,
    validate_dynamic_plan,
)


def _valid_plan() -> dict[str, Any]:
    events = []
    for index, role in enumerate(ROLE_ORDER):
        primitive, signal, threshold = ROLE_TRIGGER_REQUIREMENTS[role]
        trigger = {
            "primitive": primitive,
            "signal": signal,
            "threshold": threshold,
            "after": ROLE_ORDER[index - 1] if index else None,
        }
        end_primitive, end_signal, end_threshold = ROLE_END_REQUIREMENTS[role]
        end = {
            "primitive": end_primitive,
            "signal": end_signal,
            "threshold": end_threshold,
            "after": role,
        }
        events.append(
            {
                "event": role,
                "body_parts": ["left_hand", "right_hand"] if role in {"contact", "release"} else ["pelvis"],
                "trigger": trigger,
                "end": end,
                "criticality_level": 4 if role in {"contact", "lift", "place", "release"} else 2,
                "criticality_rationale": f"Criticality rationale for {role}.",
                "failure_if_inaccurate": f"Failure mode for {role}.",
                "rationale": f"Detect {role}.",
            }
        )
    return {"events": events}


def test_extract_json_value_preserves_top_level_action_array() -> None:
    value = _extract_json_value('```json\n[{"action":"grasp"},{"action":"carry"}]\n```')
    assert [item["action"] for item in value] == ["grasp", "carry"]


def test_extract_and_validate_fenced_plan() -> None:
    plan = _valid_plan()
    parsed = extract_json(f"prefix\n```json\n{json.dumps(plan)}\n```\n")
    validate_plan(parsed)
    assert parsed == plan


def test_validate_plan_rejects_frame_numbers() -> None:
    plan = _valid_plan()
    plan["events"][0]["frame"] = 0
    with pytest.raises(ValueError, match="forbidden"):
        validate_plan(plan)


def test_execute_plan_resolves_monotonic_trigger_frames() -> None:
    plan = _valid_plan()
    signals = {
        "object_progress": np.array([0.0, 0.1, 0.2, 0.3, 0.6, 0.8, 0.9, 1.0]),
        "pelvis_speed": np.array([2.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]),
        "max_hand_box_surface_distance": np.array([1.0, 0.8, 0.3, 0.2, 0.2, 0.2, 0.4, 0.9]),
        "object_height": np.array([0.0, 0.0, 0.0, 0.6, 0.7, 0.6, 0.1, 0.0]),
        "object_goal_distance": np.array([1.0, 1.0, 0.9, 0.8, 0.6, 0.3, 0.2, 0.1]),
    }
    thresholds = {"contact_enter": 0.4, "lift_enter": 0.5, "moving": 0.5, "near_goal": 0.4,
                  "lift_exit": 0.2, "released": 0.8}

    result = execute_plan(plan, signals, thresholds)
    triggers = [event["windows"][0]["trigger_frame"] for event in result["events"]]

    assert triggers == [0, 1, 2, 3, 4, 5, 6, 7]
    assert result["fps"] == 30
    validate_semantic_keyframe_json(result)
    assert result["events"][0]["criticality"] == CRITICALITY_MAPPING[2]


def test_validate_plan_rejects_invalid_criticality_level() -> None:
    plan = _valid_plan()
    plan["events"][0]["criticality_level"] = 5
    with pytest.raises(ValueError, match="criticality_level"):
        validate_plan(plan)


def test_field_only_repair_preserves_every_validated_field_and_event() -> None:
    plan = _valid_plan()
    plan["events"][0]["body_parts"] = []
    before = json.loads(json.dumps(plan))
    issues = plan_validation_issues(plan)

    assert [(issue["event"], issue["field"]) for issue in issues] == [("start", "body_parts")]
    repaired = apply_plan_repairs(
        plan,
        {
            "repairs": [
                {"event": "start", "field": "body_parts", "action": "replace", "value": ["pelvis"]}
            ]
        },
        issues,
    )

    validate_plan(repaired)
    assert plan == before
    assert repaired["events"][0]["body_parts"] == ["pelvis"]
    repaired_start_without_patch = dict(repaired["events"][0])
    repaired_start_without_patch["body_parts"] = []
    assert repaired_start_without_patch == before["events"][0]
    assert repaired["events"][1:] == before["events"][1:]


def test_field_only_repair_rejects_changes_outside_allowlist() -> None:
    plan = _valid_plan()
    plan["events"][0]["body_parts"] = []
    before = json.loads(json.dumps(plan))
    issues = plan_validation_issues(plan)

    with pytest.raises(ValueError, match="exactly match the allowlist"):
        apply_plan_repairs(
            plan,
            {
                "repairs": [
                    {"event": "start", "field": "body_parts", "action": "replace", "value": ["pelvis"]},
                    {
                        "event": "contact",
                        "field": "body_parts",
                        "action": "replace",
                        "value": ["right_hand"],
                    },
                ]
            },
            issues,
        )

    assert plan == before


def test_resolved_json_rejects_mismatched_criticality() -> None:
    plan = _valid_plan()
    signals = {
        "object_progress": np.arange(8, dtype=float),
        "pelvis_speed": np.arange(8, dtype=float),
        "max_hand_box_surface_distance": np.arange(8, dtype=float),
        "object_height": np.arange(8, dtype=float),
        "object_goal_distance": np.arange(8, dtype=float),
    }
    thresholds = {name: 0.0 for name in ("contact_enter", "lift_enter", "moving", "near_goal", "lift_exit", "released")}
    result = execute_plan(plan, signals, thresholds)
    result["events"][0]["criticality"] = 0.75
    with pytest.raises(ValueError, match="must equal"):
        validate_semantic_keyframe_json(result)


def test_semantic_json_diff_reports_generated_fields() -> None:
    old = {"events": [{"event": "contact", "body_parts": ["left_hand"], "windows": [{"start_frame": 1, "trigger_frame": 2, "end_frame": 3}]}]}
    new = {"events": [{"event": "contact", "body_parts": ["left_hand", "right_hand"], "criticality_level": 4, "criticality": 1.0, "windows": [{"start_frame": 1, "trigger_frame": 2, "end_frame": 4}]}]}
    diff = semantic_json_diff(old, new)
    assert diff["any_change"]
    assert set(diff["events"][0]["changed_fields"]) == {"window", "body_parts", "criticality_level", "criticality"}


def test_dynamic_plan_uses_task_specific_actions_and_gt_functions() -> None:
    plan = {
        "actions": [
            {
                "action": "drag_suitcase",
                "body_parts": ["left_hand", "right_hand"],
                "keyframe_function": {
                    "start": {
                        "primitive": "threshold_crossing_down",
                        "signal": "min_hand_object_distance",
                        "threshold": {"kind": "quantile", "q": 0.2},
                    },
                    "end": {
                        "primitive": "threshold_crossing_up",
                        "signal": "object_progress",
                        "threshold": {"kind": "quantile", "q": 0.8},
                    },
                },
                "criticality_level": 4,
                "rationale": "The hands initiate and finish dragging.",
                "criticality_rationale": "Drag contact determines success.",
                "failure_if_inaccurate": "The suitcase will not move to the goal.",
            }
        ]
    }
    validate_dynamic_plan(plan)
    assert not dynamic_plan_validation_issues(plan)
    signals = {
        "min_hand_object_distance": np.array([1.0, 0.8, 0.2, 0.1, 0.4, 0.7]),
        "object_progress": np.array([0.0, 0.1, 0.3, 0.5, 0.8, 1.0]),
    }
    result = execute_dynamic_plan(plan, signals)
    assert result["events"][0]["event"] == "drag_suitcase"
    assert result["events"][0]["windows"] == [{"start_frame": 2, "end_frame": 4, "trigger_frame": 2}]


def test_dynamic_resolution_rejects_actions_collapsed_to_same_late_frame() -> None:
    result = {
        "events": [
            {"event": "grasp", "windows": [{"start_frame": 95, "trigger_frame": 95, "end_frame": 95}]},
            {"event": "lift", "windows": [{"start_frame": 95, "trigger_frame": 95, "end_frame": 95}]},
            {"event": "carry", "windows": [{"start_frame": 95, "trigger_frame": 95, "end_frame": 95}]},
        ]
    }
    issues = dynamic_resolution_validation_issues(result, frame_count=100)
    messages = " ".join(issue["message"] for issue in issues)
    assert "duplicate window" in messages
    assert "zero duration" in messages
    assert "first action begins" in messages


def test_dynamic_resolution_accepts_ordered_task_phases() -> None:
    result = {
        "events": [
            {"event": "grasp", "windows": [{"start_frame": 10, "trigger_frame": 10, "end_frame": 20}]},
            {"event": "carry", "windows": [{"start_frame": 20, "trigger_frame": 20, "end_frame": 70}]},
            {"event": "place", "windows": [{"start_frame": 70, "trigger_frame": 70, "end_frame": 90}]},
        ]
    }
    assert not dynamic_resolution_validation_issues(result, frame_count=100)


def test_dynamic_executor_does_not_push_next_start_after_previous_end() -> None:
    def action(name: str, start_q: float, end_q: float) -> dict[str, Any]:
        return {
            "action": name,
            "body_parts": ["left_hand"],
            "keyframe_function": {
                "start": {
                    "primitive": "threshold_crossing_up",
                    "signal": "object_progress",
                    "threshold": {"kind": "quantile", "q": start_q},
                },
                "end": {
                    "primitive": "threshold_crossing_up",
                    "signal": "object_progress",
                    "threshold": {"kind": "quantile", "q": end_q},
                },
            },
            "criticality_level": 3,
            "rationale": "Observed phase.",
            "criticality_rationale": "The phase controls the interaction.",
            "failure_if_inaccurate": "The transition is sampled incorrectly.",
        }

    plan = {"actions": [action("lift", 0.1, 0.9), action("carry", 0.5, 0.8)]}
    result = execute_dynamic_plan(
        plan,
        {"object_progress": np.linspace(0.0, 1.0, 11)},
    )
    windows = [event["windows"][0] for event in result["events"]]
    assert windows == [
        {"start_frame": 1, "end_frame": 5, "trigger_frame": 1},
        {"start_frame": 5, "end_frame": 8, "trigger_frame": 5},
    ]


def test_dynamic_plan_accepts_extrema_but_rejects_identical_boundaries() -> None:
    action = {
        "action": "grasp_box",
        "body_parts": ["left_hand"],
        "keyframe_function": {
            "start": {"primitive": "global_minimum", "signal": "left_hand_object_distance", "threshold": None},
            "end": {"primitive": "global_minimum", "signal": "left_hand_object_distance", "threshold": None},
        },
        "criticality_level": 3,
        "rationale": "Grasp the box.",
        "criticality_rationale": "Contact matters.",
        "failure_if_inaccurate": "The box is dropped.",
    }
    plan = {"actions": [action, {**action, "action": "carry_box"}]}
    messages = " ".join(issue["message"] for issue in dynamic_plan_validation_issues(plan))
    assert "start and end rules are identical" in messages
