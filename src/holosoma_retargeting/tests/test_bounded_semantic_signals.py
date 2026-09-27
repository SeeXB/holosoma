"""Behavioral checks for time-bounded GT detection and failure provenance."""
import json

import numpy as np
import pytest

from holosoma_retargeting.semantic_keyframes import bounded_signals as b
from holosoma_retargeting.semantic_keyframes import pipeline as p


def rule(transition, times=(0.5, 1.5), signal="object_height"):
    return dict(time_range_s=list(times), signal=signal, transition=transition)


def motion():
    # Quiet 1 s, rise 1 s, quiet 1 s, lower 1 s, quiet 1 s.
    t = np.arange(151)/30
    return {"object_height": np.clip(t-1, 0, 1)-np.clip(t-3, 0, 1)}


def plan(start=None, end=None):
    return dict(schema=b.SCHEMA, actions=[dict(
        action="lift", body_parts=["left_hand", "right_hand"], criticality_level=3,
        rationale="Visible upward motion", start=start or rule("increase_onset"),
        end=end or rule("increase_offset", (1.5, 2.5)),
    )])


@pytest.mark.parametrize("transition,bounds,expected", [
    ("increase_onset", (.5, 1.5), 30), ("increase_offset", (1.5, 2.5), 60),
    ("decrease_onset", (2.5, 3.5), 90), ("decrease_offset", (3.5, 4.5), 120),
])
def test_detects_signed_motion_boundaries_not_velocity_peaks(transition, bounds, expected):
    result = b.resolve_boundary(rule(transition, bounds), motion(), 30)
    assert result["status"] == "gt_resolved"
    assert abs(result["frame"]-expected) <= 2
    assert result["enter_threshold"] > result["exit_threshold"] > 0


def test_never_rescues_wrong_direction_or_searches_outside_interval():
    for spec in [rule("decrease_onset"), rule("increase_onset", (3.5, 4.5))]:
        result = b.resolve_boundary(spec, motion(), 30)
        assert result["status"] == "unresolved"
        assert result["frame"] is None
        assert not result["candidate_frames"]


def test_static_noise_and_isolated_outlier_do_not_fabricate_motion():
    for values in [np.ones(151), np.random.default_rng(8).normal(0, 1e-5, 151)]:
        values[30] += 1.0
        result = b.resolve_boundary(rule("increase_onset"), {"object_height": values}, 30)
        assert result["status"] == "unresolved"


def test_repetition_is_local_and_threshold_does_not_use_distant_clip_outlier():
    height = np.r_[motion()["object_height"], motion()["object_height"]]
    specification = rule("increase_onset", (5.6, 6.5))
    first = b.resolve_boundary(specification, {"object_height": height}, 30)
    height[:30] = 10000
    second = b.resolve_boundary(specification, {"object_height": height}, 30)
    assert first["frame"] == second["frame"]
    assert 178 <= first["frame"] <= 183
    assert first["enter_threshold"] == second["enter_threshold"]


def test_already_moving_does_not_invent_onset_or_quiet_baseline():
    result = b.resolve_boundary(rule("increase_onset"), {"object_height": np.arange(151)/30}, 30)
    assert result["status"] == "unresolved"
    assert "baseline" in result["reason"]


def test_speed_tests_activity_not_acceleration():
    speed = np.zeros(151)
    speed[30:60] = .5
    signals = {"object_speed": speed}
    onset = b.resolve_boundary(rule("motion_onset", signal="object_speed"), signals, 30)
    offset = b.resolve_boundary(rule("motion_offset", (1.5, 2.5), "object_speed"), signals, 30)
    assert abs(onset["frame"]-30) <= 2
    assert abs(offset["frame"]-60) <= 2


def test_clip_edges_are_explicitly_censored_not_gt_confirmed():
    payload = plan(rule("clip_start", (0, 0), None), rule("clip_end", (5, 5), None))
    result = b.resolve_plan(payload, motion(), 30)
    assert result["status"] == "resolved"
    assert result["actions"][0]["boundaries"]["start"]["status"] == "clip_censored"
    assert result["resolved_events"][0]["windows"][0] == dict(start_frame=0, end_frame=150, trigger_frame=0)


@pytest.mark.parametrize("spec", [
    rule("global_minimum"), rule("increase_onset", signal="unknown"),
    rule("increase_onset", signal="object_speed"), rule("motion_onset"),
    rule("decrease_onset", signal="object_rotate"), rule("increase_onset", (0, 5)),
    dict(rule("increase_onset"), threshold=.5), rule("clip_start", (.1, .1), None),
    rule("increase_onset", (float("nan"), 2)),
])
def test_invalid_vlm_contract_rejected(spec):
    with pytest.raises(ValueError):
        b.validate_plan(plan(start=spec), {"object_height", "object_speed", "object_rotate"}, 5)


def test_dispatch_preserves_legacy_compatibility_and_requires_whole_plan():
    accepted = p.execute_dynamic_plan(plan(), motion(), 30)
    assert len(accepted["events"]) == 1
    invalid = plan()
    invalid["actions"].append(dict(invalid["actions"][0], action="unsupported",
                                   start=rule("decrease_onset", (3.5, 4.5)),
                                   end=rule("clip_end", (5, 5), None)))
    diagnostic = b.resolve_plan(invalid, motion(), 30)
    assert diagnostic["status"] == "unresolved"
    assert len(diagnostic["actions"]) == 2
    with pytest.raises(ValueError, match="unresolved bounded plan"):
        p.execute_dynamic_plan(invalid, motion(), 30)


def test_generation_sends_full_video_and_keeps_failed_candidate(tmp_path, monkeypatch):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"test video")
    bundle = tmp_path / "bundle.npz"
    bundle.write_bytes(b"test bundle")
    monkeypatch.setattr(p, "load_retargeting_bundle_signals", lambda _: (motion(), {}, 30))
    monkeypatch.setenv("OPENAI_MODEL", "mock")
    calls = []

    def call(content, prompt):
        calls.append((content, prompt))
        return json.dumps(plan(start=rule("decrease_onset")) if len(calls)==1 else plan())

    output = tmp_path / "result.json"
    result = b.generate_bounded_plan(video=video, bundle_file=bundle, output=output, call=call, max_repairs=1)
    assert len(calls) == 2
    assert calls[0][0][0]["type"] == "video_url"
    assert (tmp_path / "audit/attempt_0.resolution.json").is_file()
    assert result["generation_metadata"]["vlm_attempt_count"] == 2
    assert "confidence" not in result["events"][0]


def test_unresolved_generation_never_exports_partial_semantic_plan(tmp_path, monkeypatch):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"test")
    monkeypatch.setattr(p, "load_retargeting_bundle_signals", lambda _: (motion(), {}, 30))
    output = tmp_path / "semantic_plan.json"
    with pytest.raises(ValueError, match="whole plan remains unresolved"):
        b.generate_bounded_plan(video=video, bundle_file=video, output=output, max_repairs=0,
                                call=lambda *_: json.dumps(plan(start=rule("decrease_onset"))))
    assert not output.exists()
    assert (tmp_path / "audit/attempt_0.resolution.json").is_file()
