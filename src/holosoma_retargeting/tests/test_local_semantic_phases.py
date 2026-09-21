import numpy as np
import pytest
from holosoma_retargeting.semantic_keyframes.local_phases import (
    SCHEMA,
    compile_actions,
    execute_local_plan,
    resolve_start,
    validate_phases,
)
from holosoma_retargeting.semantic_keyframes.pipeline import execute_dynamic_plan


def phase(start, end, refinement="visual_boundary"):
    return dict(
        action="move_body",
        start_sample=start,
        end_sample=end,
        body_parts=["pelvis"],
        refinement=refinement,
        criticality_level=3,
        rationale="Visible body movement",
    )


def test_numbered_visual_intervals_must_cover_clip_without_gaps():
    with pytest.raises(ValueError, match="start_sample=None"):
        validate_phases({"phases": [phase(0, 6)]}, 6, True)
    with pytest.raises(ValueError, match="less than half"):
        validate_phases({"phases": [phase(1, 2)]}, 6, True)
    with pytest.raises(ValueError, match="start_sample=3"):
        validate_phases({"phases": [phase(1, 3), phase(2, 6)]}, 6, True)


def test_localized_visual_phases_preserve_late_repetitions():
    visual = validate_phases({"phases": [phase(1, 3), phase(3, 6)]}, 6, True)
    actions = compile_actions(visual, np.array([600, 620, 640, 660, 680, 700]), 2)
    signals = {"pelvis_height": np.ones(701)}
    plan = {"schema": SCHEMA, "actions": actions}
    result = execute_dynamic_plan(plan, signals, 30)
    assert [e["windows"][0]["trigger_frame"] for e in result["events"]] == [600, 640]
    assert result["events"][-1]["windows"][0]["end_frame"] == 700
    assert all(e["boundary_source"] == "visual_sample" for e in result["events"])


def test_missing_directional_event_raises_instead_of_nearest_fallback():
    rule = dict(
        primitive="object_rise",
        search_start_frame=10,
        search_end_frame=20,
        minimum_vertical_speed_m_s=0.02,
        visual_anchor_frame=15,
    )
    with pytest.raises(ValueError, match="no directional motion"):
        resolve_start(rule, {"object_height": -np.arange(30) / 30}, 30)
    rule["primitive"] = "object_fall"
    assert 10 <= resolve_start(rule, {"object_height": -np.arange(30) / 30}, 30) <= 20


def test_directional_peak_cannot_escape_local_bracket():
    height = np.zeros(100)
    height[5:] = 2
    height[60:] = 2.1
    rule = dict(
        primitive="object_rise",
        search_start_frame=50,
        search_end_frame=70,
        minimum_vertical_speed_m_s=0.02,
        visual_anchor_frame=60,
    )
    assert resolve_start(rule, {"object_height": height}, 30) in (59, 60)


def test_duplicate_local_triggers_are_rejected():
    actions = compile_actions({"phases": [phase(1, 3), phase(3, 6)]}, np.arange(6) * 20, 0)
    actions[1]["keyframe_function"]["start"] = dict(actions[0]["keyframe_function"]["start"])
    with pytest.raises(ValueError, match="strictly increasing"):
        execute_local_plan({"schema": SCHEMA, "actions": actions}, {"pelvis_height": np.ones(101)}, 30)


def test_adjacent_nonoverlapping_sample_ranges_are_accepted():
    result = validate_phases({"phases": [phase(1, 3), phase(4, 6)]}, 6, True)
    assert result["phases"][1]["start_sample"] == 4


def test_physical_refinement_uses_observed_direction():
    from holosoma_retargeting.semantic_keyframes.local_phases import refine_actions_from_gt

    actions = compile_actions({"phases": [phase(1, 3), phase(3, 6)]}, np.arange(6) * 20, 0)
    signals = {"object_height": np.arange(101) / 30}
    refined = refine_actions_from_gt(actions, signals, 30)
    assert all(a["keyframe_function"]["start"]["primitive"] == "object_rise" for a in refined)
    assert execute_local_plan({"schema": SCHEMA, "actions": refined}, signals, 30)["events"]


def test_singleton_wrapper_is_normalized_without_changing_phase_labels():
    raw = [{"phases": [phase(1, 3), phase(4, 6)]}]
    result = validate_phases(raw, 6, True)
    assert [p["action"] for p in result["phases"]] == ["move_body", "move_body"]
    assert [p["start_sample"] for p in result["phases"]] == [1, 4]


def test_static_gt_keeps_explicit_visual_boundary_not_fake_motion():
    from holosoma_retargeting.semantic_keyframes.local_phases import refine_actions_from_gt

    actions = compile_actions({"phases": [phase(1, 3), phase(3, 6)]}, np.arange(6) * 20, 0)
    refined = refine_actions_from_gt(actions, {"pelvis_height": np.ones(101)}, 30)
    assert all(a["keyframe_function"]["start"]["primitive"] == "visual_boundary" for a in refined)


def test_failed_long_clip_subdivides_without_lowering_coverage(tmp_path, monkeypatch):
    import json
    import re
    from holosoma_retargeting.semantic_keyframes import local_phases as local

    signals = {"pelvis_height": np.ones(600)}
    bundle = tmp_path / "body.npz"
    np.savez(bundle, human_joints=np.zeros((600, 22, 3)))
    monkeypatch.setattr(local.legacy, "load_retargeting_bundle_signals", lambda _: (signals, {}, 30.0))

    def images(joints, directory, *args, **kwargs):
        directory.mkdir(parents=True, exist_ok=True)
        return []

    monkeypatch.setattr(local, "skeleton_images", images)
    requests = []

    def vlm(content, prompt):
        text = content[-1]["text"]
        n = int(re.search(r"contains (\d+) numbered", text).group(1))
        requests.append(n)
        return json.dumps({"phases": [phase(1, 9 if n == 32 else n)]})

    monkeypatch.setattr(local.legacy, "call_vlm", vlm)
    monkeypatch.setenv("OPENAI_MODEL", "mock-test")
    result = local.generate_localized_semantic_keyframes(
        video=tmp_path / "views",
        bundle_file=bundle,
        output=tmp_path / "data/plan.json",
        audit_dir=tmp_path / "audit",
        max_repairs=0,
    )
    assert requests == [32, 16, 16]
    assert len(result["events"]) == 2
    assert [e["windows"][0]["start_frame"] for e in result["events"]] == [0, 300]
    assert result["events"][-1]["windows"][0]["end_frame"] == 599
    assert (tmp_path / "audit/segment_000/split_decision.json").is_file()
    # Resume reuses accepted children and the split decision, without another request.
    local.generate_localized_semantic_keyframes(
        video=tmp_path / "views",
        bundle_file=bundle,
        output=tmp_path / "data/plan.json",
        audit_dir=tmp_path / "audit_retry",
        max_repairs=0,
    )
    assert requests == [32, 16, 16]
