import numpy as np
import pytest

from holosoma_retargeting.semantic_keyframes.local_phases import (
    SCHEMA, compile_actions, execute_local_plan, validate_contact_review, include_contact_parts, validate_phases,
)


def test_hand_contact_cannot_be_replaced_by_wrist():
    phase = {"body_parts": ["left_wrist", "right_wrist"], "contact_review": {
        "hand": "both", "wrist": "none", "forearm": "none", "evidence": "双手握住物体，腕和小臂没有贴住物体。",
    }}
    with pytest.raises(ValueError, match="omits.*left_hand.*right_hand"):
        validate_contact_review(phase)
    phase["body_parts"] += ["left_hand", "right_hand"]
    validate_contact_review(phase)


def test_forearm_support_does_not_force_a_hand_label_and_survives_replay():
    phase = {"action": "support_table", "body_parts": ["left_elbow", "right_elbow"],
             "contact_review": {"hand": "none", "wrist": "uncertain", "forearm": "both",
                                "evidence": "双侧小臂托住桌面，手掌张开且远离桌面，腕部被遮挡。"},
             "start_sample": 1, "end_sample": 3, "refinement": "visual_boundary",
             "criticality_level": 3, "rationale": "双侧小臂支撑物体。"}
    validate_contact_review(phase)
    plan = {"schema": SCHEMA, "actions": compile_actions({"phases": [phase]}, np.array([0, 10, 20]), 0)}
    event = execute_local_plan(plan, {"object_height": np.ones(21)}, 30)["events"][0]
    assert event["contact_review"] == phase["contact_review"]
    assert event["body_parts"] == ["left_elbow", "right_elbow"]


def test_motion_parts_union_contact_parts_preserves_original_and_is_idempotent():
    phase = {"body_parts": ["right_knee", "right_wrist"], "contact_review": {
        "hand": "both", "wrist": "uncertain", "forearm": "none",
        "evidence": "Both palms visibly grip the box; wrists are occluded; forearms are away.",
    }}
    include_contact_parts(phase)
    validate_contact_review(phase)
    assert phase["body_parts"] == ["right_knee", "right_wrist", "left_hand", "right_hand"]
    assert phase["body_parts_before_contact_review"] == ["right_knee", "right_wrist"]
    assert phase["contact_body_parts"] == ["left_hand", "right_hand"]
    assert phase["body_parts_added_from_contact_review"] == ["left_hand", "right_hand"]
    include_contact_parts(phase)
    assert phase["body_parts_before_contact_review"] == ["right_knee", "right_wrist"]


def test_object_clip_without_identified_contact_is_flagged_not_filled_in():
    phases = [{"action": "walking", "start_sample": start, "end_sample": end,
               "body_parts": ["left_knee", "right_knee"], "criticality_level": 3,
               "rationale": "Walking near the object.",
               "contact_review": {"hand": "uncertain", "wrist": "none", "forearm": "none",
                                  "evidence": "The hands are occluded and contact cannot be established."}}
              for start, end in ((1, 3), (3, 6))]
    with pytest.raises(ValueError, match="No hand/wrist/forearm contact"):
        validate_phases({"phases": phases}, 6, False, require_contact_review=True)
    assert all(not p["contact_body_parts"] for p in phases)
