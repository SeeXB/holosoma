from copy import deepcopy

import numpy as np

from holosoma_retargeting.semantic_keyframes.body_constraints import apply_body_part_constraints
from holosoma_retargeting.semantic_keyframes.local_phases import SCHEMA, compile_actions, execute_local_plan, phase_prompt


def test_bimanual_pairing_preserves_evidence_and_replays_with_same_timing():
    visual = {"phases": [{
        "action": "carry_box", "start_sample": 1, "end_sample": 3,
        "body_parts": ["right_elbow", "left_wrist", "right_hand", "left_shoulder", "right_knee", "pelvis"],
        "refinement": "visual_boundary", "criticality_level": 3, "rationale": "Carries box",
    }]}
    plan = {"schema": SCHEMA, "actions": compile_actions(visual, np.array([0, 10, 20]), 0),
            "segments": [{"visual_plan": deepcopy(visual)}]}
    signals = {"object_height": np.ones(21)}
    baseline = execute_local_plan(plan, signals, 30)
    plan["body_part_constraints"] = {"mode": "bimanual_upper_limbs_v1"}
    constrained = apply_body_part_constraints(plan)
    action = constrained["actions"][0]
    assert action["body_parts_added_by_constraints"] == ["left_elbow", "right_wrist", "left_hand", "right_shoulder"]
    assert "left_knee" not in action["body_parts"]
    assert constrained["segments"] == plan["segments"]
    assert "left_elbow" not in plan["actions"][0]["body_parts"]
    assert apply_body_part_constraints(constrained) == constrained
    resolved = execute_local_plan(constrained, signals, 30)
    assert resolved == execute_local_plan(plan, signals, 30)
    assert resolved["events"][0]["body_parts"] == action["body_parts"]
    assert resolved["events"][0]["windows"] == baseline["events"][0]["windows"]


def test_unknown_or_single_hand_tasks_do_not_gain_exemptions():
    plan = {"actions": [{"body_parts": ["right_elbow", "right_hand"]}]}
    assert apply_body_part_constraints(plan) == plan


def test_user_bimanual_constraint_is_explicit_in_english_prompt():
    plain = phase_prompt(20, False, "largetable")
    constrained = phase_prompt(20, False, "largetable", {"mode": "bimanual_upper_limbs_v1"})
    assert "user confirms a bimanual interaction" in constrained
    assert "even when one side is occluded" in constrained
    assert "user confirms a bimanual interaction" not in plain
    assert constrained.isascii()
