"""Separate keyframe predicates must not turn into continuous-interval detection."""
import json

import numpy as np
import pytest

from holosoma_retargeting.semantic_keyframes import adjacent_delta_boundaries as b
from holosoma_retargeting.semantic_keyframes import pipeline as p


def rule(condition="increasing", threshold=.2, signal="object_height"):
    return dict(signal=signal, condition=condition, threshold=threshold)


def action(start=None, end=None, name="move"):
    return dict(action=name, body_parts=["left_hand"], start=rule() if start is None else start,
                end=rule("decreasing") if end is None else end)


def plan(*actions):
    return dict(schema=b.SCHEMA, actions=list(actions))


def signals(values=(0, 0, 1, 2, 2, 2, 2, 1, 0)):
    return {"object_height":np.array(values, dtype=float)}


def test_two_rules_choose_two_current_frames_without_continuous_interior():
    data = signals()
    result = b.resolve_plan(plan(action()), data, 30)
    event = result["resolved_events"][0]
    assert event["windows"] == [dict(start_frame=2, end_frame=7, trigger_frame=2)]
    assert event["boundary_evidence"]["start"]["adjacent_frames"] == [1, 2]
    assert event["boundary_evidence"]["end"]["adjacent_frames"] == [6, 7]
    # The original increasing run ends at 3. The actual end comes from its own
    # rule at 7, with an interior pause satisfying neither boundary predicate.
    ratios, _ = b.shared.delta_ratios(data["object_height"])
    assert np.all(ratios[4:7] == 0)
    assert p.execute_dynamic_plan(plan(action()), data, 30)["events"] == result["resolved_events"]


def test_end_can_use_a_different_signal():
    data = signals()
    data["left_hand_object_distance"] = np.array([4, 4, 4, 4, 4, 3, 2, 2, 2.])
    result = b.resolve_plan(plan(action(end=rule("decreasing", signal="left_hand_object_distance"))), data, 30)
    assert result["resolved_events"][0]["windows"][0]["end_frame"] == 5


def test_match_inside_existing_true_run_does_not_require_run_start_or_delta_crossing():
    trace = b.resolve_boundary(rule(), signals((0, 1, 2, 3, 3)), minimum=2)
    assert trace["frame"] == 2 and trace["adjacent_frames"] == [1, 2]


def test_single_pair_of_samples_and_point_event_are_valid_without_duration_warning():
    payload = plan(action(end=rule(), name="grasp"))
    result = b.resolve_plan(payload, signals((0, 0, 1, 1)), 30)
    assert result["status"] == "resolved" and not result["issues"]
    event = result["resolved_events"][0]
    assert event["windows"] == [dict(start_frame=2, end_frame=2, trigger_frame=2)]
    assert all("warnings" not in e for e in event["boundary_evidence"].values())
    p.validate_semantic_keyframe_json(dict(fps=30, events=[event]), require_criticality=False)


def test_two_adjacent_boundary_frames_are_valid():
    result = b.resolve_plan(plan(action(end=rule("steady", 0))), signals((0, 0, 1, 1)), 30)
    assert result["resolved_events"][0]["windows"][0] == dict(start_frame=2, end_frame=3, trigger_frame=2)


def test_starts_advance_without_clipping_ends_or_forcing_nonoverlap():
    result = b.resolve_plan(plan(action(name="one"), action(name="two")), signals(), 30)
    assert [e["windows"][0] for e in result["resolved_events"]] == [
        dict(start_frame=2, end_frame=7, trigger_frame=2),
        dict(start_frame=3, end_frame=7, trigger_frame=3),
    ]


def test_missing_start_blocks_end_and_following_order_no_fallback():
    payload = plan(action(start=rule(threshold=.9)), action(name="later"))
    result = b.resolve_plan(payload, signals(), 30)
    assert result["status"] == "unresolved" and not result["resolved_events"]
    assert result["actions"][0]["boundaries"]["end"]["status"] == "blocked_by_start"
    assert result["actions"][1]["boundaries"]["start"]["status"] == "blocked_by_previous_start"
    assert result["actions"][1]["boundaries"]["start"]["frame"] is None
    with pytest.raises(ValueError, match="unresolved"):
        p.execute_dynamic_plan(payload, signals(), 30)


def test_missing_end_does_not_supply_lower_bound_for_later_start():
    result = b.resolve_plan(plan(action(end=rule("decreasing", .9)), action(name="later")), signals(), 30)
    assert result["status"] == "unresolved"
    assert result["actions"][1]["boundaries"]["start"]["frame"] == 3


def test_frame_zero_is_never_fabricated_and_constant_signal_is_undefined():
    assert b.resolve_boundary(rule("steady", 0), signals((0, 1, 1)))["frame"] == 2
    bad = b.resolve_boundary(rule("steady", 0), signals((1, 1, 1)))
    assert bad["status"] == "unresolved" and bad["frame"] is None


def test_amplitude_is_full_raw_signal_span_with_native_frame_rate():
    payload = plan(action())
    data = signals()
    result = b.resolve_plan(payload, data, 30)
    assert result["actions"] == b.resolve_plan(payload, data, 60)["actions"]
    original = result["resolved_events"][0]["windows"]
    assert b.resolve_plan(payload, signals(data["object_height"]*1e-9), 30)["resolved_events"][0]["windows"] == original
    trace = result["actions"][0]["boundaries"]["start"]
    assert trace["signal_span"] == 2 and trace["ratio"] == .5


@pytest.mark.parametrize("extra", [{"time_range_s":[0, 1]}, {"reference":"clip_start"},
    {"frame":3}, {"duration":1}, {"lag":1}, {"crossing":"up"}, {"minimum_steps":2}])
def test_rule_rejects_time_reference_duration_and_extra_parameters(extra):
    assert b.validation_issues(plan(action(start=dict(rule(), **extra))), {"object_height"})


@pytest.mark.parametrize("threshold", [True, -.1, 1.1, float("nan"), float("inf"), "2%"])
def test_bad_boundary_threshold_rejected(threshold):
    assert b.validation_issues(plan(action(end=rule(threshold=threshold))), {"object_height"})


def test_old_action_wide_threshold_is_not_the_new_boundary_contract():
    old = dict(action="lift", body_parts=["left_hand"], signal="object_height", condition="increasing",
               threshold=.02, rationale="visible")
    assert b.validation_issues(plan(old), {"object_height"})
    assert not b.validation_issues(plan(action()), {"object_height"})


def test_invalid_start_and_end_are_reported_together():
    issues = b.validation_issues(plan(action(start=rule(threshold=0), end=rule("bad"))), {"object_height"})
    assert {x["field"] for x in issues} == {"actions[0].start.threshold", "actions[0].end.condition"}


def test_prompt_defines_independent_keyframes_and_accepts_short_events():
    prompt = b.build_prompt(p.DYNAMIC_SIGNALS, 100, 30)
    assert "START FRAME" in prompt and "END FRAME" in prompt
    assert "frames BETWEEN them need not satisfy either predicate" in prompt
    assert "short duration alone is not an error" in prompt
    assert "first current frame t satisfying the start rule" in prompt
    assert "first frame t satisfying the end" in prompt


def test_generation_repairs_old_contract_and_saves_separate_boundary_rules(tmp_path, monkeypatch):
    video = tmp_path/"full.mp4"
    video.write_bytes(b"mock video")
    monkeypatch.setattr(p, "load_retargeting_bundle_signals", lambda _:(signals(), {}, 30))
    monkeypatch.setenv("OPENAI_MODEL", "mock")
    calls = []
    def call(content, prompt):
        calls.append((content, prompt))
        bad = dict(action(), time_range_s=[0, 1])
        return json.dumps(plan(bad if len(calls) == 1 else action()))
    result = b.generate_boundary_plan(video=video, bundle_file=video, output=tmp_path/"out.json", max_repairs=1, call=call)
    assert calls[0][0][0]["type"] == "video_url"
    assert result["generation_metadata"]["numeric_parameters_per_boundary"] == 1
    assert result["generation_metadata"]["interior_predicate_required"] is False
    assert result["generation_metadata"]["short_event_is_error"] is False
    saved = json.loads((tmp_path/"out.event_plan.json").read_text())
    assert set(saved["actions"][0]) == {"action", "body_parts", "start", "end"}
    assert (tmp_path/"audit/attempt_0.validation.json").exists()
    assert all(not ({"rationale", "criticality", "criticality_level", "criticality_source"}&set(e))
               for e in result["events"])


def test_failed_generation_keeps_candidates_but_does_not_export_partial_plan(tmp_path, monkeypatch):
    video = tmp_path/"full.mp4"
    video.write_bytes(b"mock video")
    monkeypatch.setattr(p, "load_retargeting_bundle_signals", lambda _:(signals(), {}, 30))
    output = tmp_path/"out.json"
    with pytest.raises(ValueError, match="whole plan remains unresolved"):
        b.generate_boundary_plan(video=video, bundle_file=video, output=output, max_repairs=0,
                                 call=lambda *_:json.dumps(plan(action(), action(start=rule(threshold=.9)))))
    assert not output.exists() and (tmp_path/"audit/attempt_0.resolution.json").exists()


@pytest.mark.parametrize('extra', [dict(rationale='visible'), dict(criticality=.75),
    dict(criticality_level=3), dict(criticality_source='placeholder'), dict(confidence=1),
    dict(criticality_rationale='legacy'), dict(failure_if_inaccurate='legacy')])
def test_compact_actions_reject_explanations_weights_and_other_unused_fields(extra):
    assert b.validation_issues(plan(dict(action(), **extra)), {'object_height'})


def test_compact_prompt_no_concrete_anatomy_example_and_no_explanation_fields():
    prompt = b.build_prompt(p.DYNAMIC_SIGNALS, 100, 30)
    template = p._extract_json_value(prompt[prompt.index('{"schema"'):])
    example = template['actions'][0]
    assert example['body_parts'] == ['<allowed region>']
    assert set(example) == {'action', 'body_parts', 'start', 'end'}
    assert 'rationale' not in prompt and 'criticality' not in prompt
    assert 'evidence for each selected region internally' in prompt
    assert 'without explanations' in prompt


def test_compact_signal_guide_preserves_full_signal_vocabulary_and_partial_inputs():
    guide = b.compact_signal_guide(p.DYNAMIC_SIGNALS)
    grouped = {f'{part}_{suffix}' for part in p.MAPPABLE_BODY_PARTS
               for suffix in ('height', 'speed', 'object_distance')}
    assert len(grouped) == 54 and len(p.DYNAMIC_SIGNALS-grouped) == 12
    for part in p.MAPPABLE_BODY_PARTS:
        assert part in guide
    for name in p.DYNAMIC_SIGNALS-grouped:
        assert name+':' in guide
    for suffix in ('height', 'speed', 'object_distance'):
        assert '<part>_'+suffix in guide
    assert 'left_elbow_height' in b.compact_signal_guide({'left_elbow_height'})


def test_legacy_plan_still_replays_but_compact_export_has_no_legacy_fields(tmp_path):
    from holosoma_retargeting.semantic_keyframes.runtime import load_semantic_plan_projection
    old = dict(schema=b.LEGACY_SCHEMA, actions=[dict(action(), rationale='old visual evidence')])
    legacy = p.execute_dynamic_plan(old, signals(), 30)
    compact = p.execute_dynamic_plan(plan(action()), signals(), 30)
    assert legacy['events'][0]['windows'] == compact['events'][0]['windows']
    assert legacy['events'][0]['criticality_level'] == 3 and legacy['events'][0]['rationale'] == 'old visual evidence'
    assert 'criticality' not in compact['events'][0] and 'rationale' not in compact['events'][0]
    path = tmp_path/'compact.json'
    path.write_text(json.dumps(compact))
    projected = load_semantic_plan_projection(path)
    assert projected[0].body_parts == ['left_hand']
    assert (projected[0].start_frame, projected[0].end_frame) == (2, 7)
    p.validate_semantic_keyframe_json(compact, require_criticality=False)
    # Historical strict validation remains available; no synthetic weights added.
    with pytest.raises(ValueError, match='criticality_level'):
        p.validate_semantic_keyframe_json(compact)


def test_omitting_legacy_weights_does_not_skip_window_validation():
    payload = b.execute_boundary_plan(plan(action()), signals(), 30)
    payload['events'][0]['windows'][0]['end_frame'] = 0
    with pytest.raises(ValueError, match='start_frame <= trigger_frame <= end_frame'):
        p.validate_semantic_keyframe_json(payload, require_criticality=False)
