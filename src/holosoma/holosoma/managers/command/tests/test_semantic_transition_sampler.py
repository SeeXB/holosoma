from __future__ import annotations

import json
from dataclasses import replace

import pytest
import torch

from holosoma.config_values.wbt.g1.paper_dr import (
    g1_29dof_wbt_w_object_b4_s0_original_adaptive_paper_dr,
    g1_29dof_wbt_w_object_b4_s1_semantic_uniform_paper_dr,
    g1_29dof_wbt_w_object_b4_s2_semantic_adaptive_paper_dr,
)
from holosoma.config_values.wbt.g1.reward import g1_29dof_wbt_reward_w_object
from holosoma.managers.command.semantic_transition_sampler import (
    SemanticTransitionSampler,
    load_semantic_transitions,
)

pytestmark = pytest.mark.no_sim


def _write_plan(path, *, names=("first", "second", "third"), fps=20):
    payload = {
        "fps": fps,
        "events": [
            {
                "event": names[0],
                "body_parts": ["pelvis"],
                "windows": [{"start_frame": 4, "trigger_frame": 6, "end_frame": 8}],
            },
            {
                "event": names[1],
                "body_parts": ["left_hand"],
                "windows": [{"start_frame": 12, "trigger_frame": 15, "end_frame": 18}],
            },
            {
                "event": names[2],
                "body_parts": ["right_hand"],
                "windows": [{"start_frame": 24, "trigger_frame": 27, "end_frame": 30}],
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _sampler(path, mode="semantic_adaptive", *, alpha=1.0, ratio=0.1, critical_weight=2.0):
    return SemanticTransitionSampler(
        motion_time_step_total=100,
        num_envs=4,
        device="cpu",
        motion_fps=50.0,
        semantic_file=path,
        semantic_fps=None,
        sampling_mode=mode,
        adaptive_uniform_ratio=ratio,
        adaptive_alpha=alpha,
        critical_interval_weight=critical_weight,
    )


def test_event_windows_and_ordinary_gaps_partition_full_timeline(tmp_path):
    path = _write_plan(tmp_path / "semantic.json", fps=20)
    transitions = load_semantic_transitions(path, motion_fps=50, motion_time_step_total=100)
    assert [(item.start_step, item.target_step, item.interval_kind) for item in transitions] == [
        (0, 10, "ordinary"),
        (10, 20, "critical_event"),
        (20, 30, "ordinary"),
        (30, 45, "critical_event"),
        (45, 60, "ordinary"),
        (60, 75, "critical_event"),
        (75, 99, "ordinary"),
    ]
    covered = []
    for item in transitions:
        covered.extend(range(item.start_step, item.target_step))
    assert covered == list(range(99))
    sampler = _sampler(path, ratio=0.0)
    frames, ids, global_mask = sampler.sample(2000)
    assert not global_mask.any()
    for transition_id, transition in enumerate(sampler.transitions):
        selected = frames[ids == transition_id]
        assert torch.all(selected >= transition.start_step)
        assert torch.all(selected < transition.target_step)


def test_semantic_uniform_is_uniform_per_frame_and_adaptive_prefers_critical_frames(tmp_path):
    path = _write_plan(tmp_path / "semantic.json")
    uniform = _sampler(path, mode="semantic_uniform", ratio=0.0)
    adaptive = _sampler(path, mode="semantic_adaptive", ratio=0.0)
    spans = torch.tensor([item.target_step - item.start_step for item in uniform.transitions])
    assert torch.allclose(uniform.sampling_probabilities, spans.float() / spans.sum())
    per_frame = adaptive.sampling_probabilities / spans.float()
    critical = torch.tensor([item.is_critical for item in adaptive.transitions])
    assert torch.allclose(
        per_frame[critical],
        torch.full_like(per_frame[critical], per_frame[~critical][0] * 2.0),
    )
    assert torch.allclose(per_frame[~critical], torch.full_like(per_frame[~critical], per_frame[~critical][0]))


def test_adaptive_failure_score_increases_sampling_probability(tmp_path):
    path = _write_plan(tmp_path / "semantic.json")
    sampler = _sampler(path, alpha=1.0, ratio=0.1)
    env_ids = torch.tensor([0, 1, 2, 3])
    sampler.assign(env_ids, torch.tensor([1, 1, 3, 3]))
    sampler.resolve_before_reset(
        env_ids,
        terminated=torch.tensor([True, True, False, False]),
        timeouts=torch.zeros(4, dtype=torch.bool),
        current_steps=torch.tensor([15, 15, 45, 45]),
    )
    assert sampler.failure_score[1].item() == pytest.approx(1.0)
    assert sampler.sampling_probabilities[1] > sampler.sampling_probabilities[3]
    assert sampler.failure_frame_histogram[15].item() == 2
    assert sampler.failure_transition_histogram[1, 15].item() == 2


def test_success_is_recorded_once_after_target(tmp_path):
    path = _write_plan(tmp_path / "semantic.json")
    sampler = _sampler(path, alpha=1.0, ratio=0.0)
    env_ids = torch.tensor([0])
    sampler.assign(env_ids, torch.tensor([1]))
    sampler.mark_reached(env_ids, torch.tensor([20]))
    sampler.mark_reached(env_ids, torch.tensor([30]))
    assert sampler.success_count[1].item() == 1
    assert sampler.failure_count.sum().item() == 0
    assert sampler.sample_count.sum().item() == 0


def test_timeout_before_target_is_configuration_error_not_failure(tmp_path):
    path = _write_plan(tmp_path / "semantic.json")
    sampler = _sampler(path, alpha=1.0, ratio=0.0)
    env_ids = torch.tensor([0])
    sampler.assign(env_ids, torch.tensor([3]))
    sampler.resolve_before_reset(
        env_ids,
        terminated=torch.tensor([False]),
        timeouts=torch.tensor([True]),
        current_steps=torch.tensor([40]),
    )
    assert sampler.failure_count.sum().item() == 0
    assert sampler.config_error_count.item() == 1


def test_global_uniform_fallback_covers_tail(tmp_path):
    path = _write_plan(tmp_path / "semantic.json")
    sampler = _sampler(path, ratio=1.0)
    frames, ids, global_mask = sampler.sample(2000)
    assert global_mask.all()
    assert (ids == -1).all()
    assert frames.max().item() >= 95
    assert frames.max().item() < 99


def test_event_names_do_not_change_sampling_distribution(tmp_path):
    first = _write_plan(tmp_path / "first.json", names=("a", "b", "c"))
    second = _write_plan(tmp_path / "second.json", names=("other_a", "other_b", "other_c"))
    torch.manual_seed(123)
    sampler_a = _sampler(first, ratio=0.0)
    frames_a, ids_a, _ = sampler_a.sample(1000)
    torch.manual_seed(123)
    sampler_b = _sampler(second, ratio=0.0)
    frames_b, ids_b, _ = sampler_b.sample(1000)
    assert torch.equal(frames_a, frames_b)
    assert torch.equal(ids_a, ids_b)


def test_single_event_still_produces_event_and_complement_intervals(tmp_path):
    path = tmp_path / "single.json"
    path.write_text(json.dumps({
        "fps": 20,
        "events": [{
            "event": "only_event",
            "windows": [{"start_frame": 8, "trigger_frame": 10, "end_frame": 16}],
        }],
    }), encoding="utf-8")
    intervals = load_semantic_transitions(path, motion_fps=50, motion_time_step_total=100)
    assert [(item.start_step, item.target_step, item.interval_kind) for item in intervals] == [
        (0, 20, "ordinary"), (20, 40, "critical_event"), (40, 99, "ordinary"),
    ]
    assert intervals[1].event_name == "only_event"


def test_overlapping_event_windows_are_rejected_instead_of_double_weighted(tmp_path):
    path = tmp_path / "overlap.json"
    path.write_text(json.dumps({
        "fps": 20,
        "events": [
            {"event": "a", "windows": [{"start_frame": 4, "trigger_frame": 6, "end_frame": 12}]},
            {"event": "b", "windows": [{"start_frame": 10, "trigger_frame": 11, "end_frame": 16}]},
        ],
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="overlap"):
        load_semantic_transitions(path, motion_fps=50, motion_time_step_total=100)


def test_critical_interval_weight_must_actually_prefer_critical_frames(tmp_path):
    path = _write_plan(tmp_path / "semantic.json")
    with pytest.raises(ValueError, match="greater than 1"):
        _sampler(path, critical_weight=1.0)


def test_semantic_fps_mismatch_and_missing_metadata_are_explicit(tmp_path):
    path = _write_plan(tmp_path / "semantic.json", fps=20)
    with pytest.raises(ValueError, match="disagrees"):
        load_semantic_transitions(path, motion_fps=50, motion_time_step_total=100, semantic_fps=30)
    no_fps = tmp_path / "no_fps.json"
    no_fps.write_text(
        json.dumps({"events": [{"event": "a", "trigger_frame": 0}, {"event": "b", "trigger_frame": 1}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="FPS"):
        load_semantic_transitions(no_fps, motion_fps=50, motion_time_step_total=100)


def test_paper_sampling_presets_use_baseline_reward_and_distinct_modes():
    s0 = g1_29dof_wbt_w_object_b4_s0_original_adaptive_paper_dr
    s1 = g1_29dof_wbt_w_object_b4_s1_semantic_uniform_paper_dr
    s2 = g1_29dof_wbt_w_object_b4_s2_semantic_adaptive_paper_dr
    assert s0.reward.semantic_keyframe is None
    assert s1.reward.semantic_keyframe is None
    assert s2.reward.semantic_keyframe is None
    motions = [
        s0.command.setup_terms["motion_command"].params["motion_config"],
        s1.command.setup_terms["motion_command"].params["motion_config"],
        s2.command.setup_terms["motion_command"].params["motion_config"],
    ]
    assert [motion.sampling_mode for motion in motions] == [
        "original_adaptive",
        "semantic_uniform",
        "semantic_adaptive",
    ]
    assert motions[2].semantic_critical_interval_weight == pytest.approx(2.0)
    assert s0.reward.terms == s1.reward.terms == s2.reward.terms == g1_29dof_wbt_reward_w_object.terms
    assert s0 == replace(s1, command=s0.command) == replace(s2, command=s0.command)
