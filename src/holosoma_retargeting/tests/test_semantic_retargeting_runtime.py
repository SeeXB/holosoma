from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from holosoma_retargeting.config_types.semantic import SemanticRetargetingConfig
from holosoma_retargeting.semantic_keyframes.precision import PrecisionPayload, evaluate_precision
from holosoma_retargeting.semantic_keyframes.runtime import (
    SemanticEvent,
    SemanticTimeline,
    build_always_body_vertex_weight_result,
    build_semantic_vertex_weight_result,
    build_semantic_vertex_weights,
    extract_semantic_cross_entity_edges,
    load_semantic_events,
    load_semantic_plan_projection,
    make_budget_plan,
    randomize_event_times_preserving_support,
    resolve_body_vertex_mapping,
    scale_semantic_weight_result,
    spatiotemporal_semantic_context,
)


def _event(
    name: str,
    trigger: int,
    start: int,
    end: int,
    body_parts: list[str] | None = None,
    confidence: float = 1.0,
    criticality: float = 1.0,
) -> SemanticEvent:
    return SemanticEvent(
        name=name,
        trigger_frame=trigger,
        start_frame=start,
        end_frame=end,
        body_parts=body_parts or ["left_hand"],
        confidence=confidence,
        criticality=criticality,
    )


def test_semantic_json_parser_preserves_schema_and_confidence(tmp_path: Path) -> None:
    path = tmp_path / "events.json"
    path.write_text(
        json.dumps(
            {
                "events": [
                    {
                        "event": "contact",
                        "body_parts": ["right_hand", "left_hand"],
                        "confidence": 0.75,
                        "windows": [{"start_frame": 8, "trigger_frame": 10, "end_frame": 14}],
                        "trigger": {"signal": "distance"},
                        "end": {"signal": "height"},
                        "rationale": "contact rationale",
                    },
                    {
                        "event": "unknown_event",
                        "body_parts": ["pelvis"],
                        "confidence": 0.4,
                        "start_frame": 20,
                        "trigger_frame": 21,
                        "end_frame": 22,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    events = load_semantic_events(path)
    assert events[0].score == pytest.approx(0.75)
    assert events[0].trigger == {"signal": "distance"}
    assert events[0].end == {"signal": "height"}
    assert events[1].criticality == pytest.approx(0.5)
    assert events[1].score == pytest.approx(0.2)


def test_bundled_sub3_keyframe_sanity() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "holosoma_retargeting"
        / "demo_data"
        / "semantic_keyframes"
        / "sub3_largebox_003_template_keyframes.json"
    )
    events = load_semantic_events(path)
    by_name = {event.name: event for event in events}
    assert {name: by_name[name].trigger_frame for name in by_name} == {
        "start": 0,
        "approach": 26,
        "contact": 30,
        "lift": 66,
        "carry_mid": 117,
        "arrive": 160,
        "place": 162,
        "release": 167,
    }
    for name in ("contact", "lift", "place", "release"):
        assert set(by_name[name].body_parts) == {"left_hand", "right_hand"}


def test_scheduler_compatibility_and_fixed_uniform2_budget() -> None:
    event = _event("contact", 4, 2, 8)
    original = make_budget_plan(12, [], SemanticRetargetingConfig(mode="original"))
    assert original.budgets.tolist() == [50] + [10] * 11
    uniform = make_budget_plan(12, [event], SemanticRetargetingConfig(mode="uniform", uniform_budget=2))
    assert uniform.budgets.tolist() == [50] + [2] * 11
    for mode in (
        "uniform2_semantic_weight_uniform",
    ):
        plan = make_budget_plan(12, [event], SemanticRetargetingConfig(mode=mode))
        assert plan.budgets.tolist() == [50] + [2] * 11


@pytest.mark.parametrize(
    "retired_mode",
    (
        "uniform2_semantic_weight_shuffled",
        "uniform2_semantic_weight_uniform_matched",
        "uniform2_semantic_edge_weight_uniform",
        "adaptive_semantic_part",
    ),
)
def test_retired_criticality_edge_and_adaptive_modes_are_rejected(retired_mode: str) -> None:
    config = SemanticRetargetingConfig(mode=retired_mode)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unsupported active semantic mode"):
        config.validate()


def test_overlapping_windows_timeline_is_deterministic() -> None:
    events = [_event("contact", 10, 5, 14), _event("lift", 12, 11, 20)]
    info = SemanticTimeline(events).frame_info(12)
    assert info.active_events == ["contact", "lift"]
    assert info.nearest_event == "lift"


def test_body_mapping_and_legacy_weights() -> None:
    keys = ["Pelvis", "L_Hip", "L_Wrist", "R_Wrist"]
    with pytest.warns(RuntimeWarning, match="cannot be mapped"):
        mapping = resolve_body_vertex_mapping(keys, ["pelvis", "left_hand", "right_hand", "tail"])
    assert mapping.indices == {"pelvis": 0, "left_hand": 2, "right_hand": 3}
    assert mapping.missing == ("tail",)

    adjacency = [[1], [0, 2, 3, 4], [1, 5], [1], [1], [2]]
    hand_mapping = resolve_body_vertex_mapping(["Pelvis", "L_Wrist", "R_Wrist"], ["left_hand"])
    weights = build_semantic_vertex_weights(
        frame_idx=10,
        events=[_event("contact", 10, 8, 14)],
        body_mapping=hand_mapping,
        adjacency=adjacency,
        num_human_vertices=3,
        num_vertices=6,
        config=SemanticRetargetingConfig(mode="uniform2_semantic_weight_uniform"),
    )
    assert weights.mean() == pytest.approx(1.0, abs=1e-12)
    assert weights[1] > weights[3] == pytest.approx(weights[4])
    assert weights[3] > weights[5]


def test_cross_entity_edge_extraction() -> None:
    adjacency = [[1, 3], [0, 2, 3, 4], [1], [0, 1], [1]]
    edge_set = extract_semantic_cross_entity_edges([0, 1, 2], adjacency, num_body_vertices=3)
    assert edge_set.edges == ((0, 3), (1, 3), (1, 4))
    assert edge_set.missing_body_vertices == (2,)


def test_temporal_context_ignores_criticality_and_uses_confidence_phase() -> None:
    event = _event("lift", 10, 0, 20, confidence=0.8, criticality=0.75)
    exact = spatiotemporal_semantic_context(10, [event])
    offset = spatiotemporal_semantic_context(12, [event])
    assert exact.semantic_importance == pytest.approx(0.8)
    assert offset.semantic_importance == pytest.approx(0.8 * np.exp(-0.5))


def test_active_projection_ignores_invalid_criticality_fields(tmp_path: Path) -> None:
    path = tmp_path / "semantic_v2.json"
    path.write_text(
        json.dumps(
            {
                "events": [
                    {
                        "event": "contact",
                        "body_parts": ["left_hand", "right_hand"],
                        "criticality": "intentionally invalid",
                        "criticality_level": 99,
                        "criticality_rationale": {"ignored": True},
                        "failure_if_inaccurate": ["ignored"],
                        "windows": [{"start_frame": 8, "trigger_frame": 10, "end_frame": 14}],
                        "trigger": {"signal": "distance"},
                        "rationale": "retained",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    event = load_semantic_plan_projection(path)[0]
    assert (event.start_frame, event.trigger_frame, event.end_frame) == (8, 10, 14)
    assert event.body_parts == ["left_hand", "right_hand"]
    assert event.trigger == {"signal": "distance"}
    assert event.rationale == "retained"
    assert event.criticality == 1.0
    assert event.criticality_level is None


def test_exact_semantic_and_random_budget_are_compute_matched() -> None:
    events = [
        _event("start", 0, 0, 2),
        _event("approach", 10, 8, 12),
        _event("contact", 30, 28, 35),
        _event("lift", 50, 48, 60),
    ]
    semantic = make_budget_plan(
        80,
        events,
        SemanticRetargetingConfig(mode="uniform2_semantic_weight_semantic_budget"),
    )
    random = make_budget_plan(
        80,
        events,
        SemanticRetargetingConfig(mode="uniform2_semantic_weight_random_budget", random_seed=3),
    )
    assert semantic.semantic_trigger_frames == (10, 30, 50)
    assert semantic.extra_budget_frames == semantic.semantic_trigger_frames
    assert semantic.budgets.tolist().count(4) == 3
    assert semantic.allocation_reasons[10] == "semantic_extra"
    assert int(semantic.budgets.sum()) == int(random.budgets.sum())
    assert len(random.extra_budget_frames) == 3
    assert all(
        abs(frame - trigger) > 3
        for frame in random.extra_budget_frames
        for trigger in (0, 10, 30, 50)
    )
    assert all(random.allocation_reasons[frame] == "random_extra" for frame in random.extra_budget_frames)


def test_random_time_control_preserves_window_and_temporal_integral() -> None:
    events = [
        _event("contact", 30, 30, 66, ["left_hand", "right_hand"]),
        _event("lift", 66, 66, 117, ["left_hand", "right_hand"]),
    ]
    randomized = randomize_event_times_preserving_support(196, events, seed=2)
    true_timeline = SemanticTimeline(events)
    random_timeline = SemanticTimeline(randomized)
    for true_event, random_event in zip(events, randomized):
        assert true_event.end_frame - true_event.start_frame == random_event.end_frame - random_event.start_frame
        assert sum(random_timeline.temporal_weight(frame, random_event) for frame in range(196)) == pytest.approx(
            sum(true_timeline.temporal_weight(frame, true_event) for frame in range(196)), abs=1e-12
        )


def test_weight_energy_scaling_is_exact_and_mean_one() -> None:
    adjacency = [[1], [0, 2, 3, 4], [1, 5], [1], [1], [2]]
    mapping = resolve_body_vertex_mapping(["Pelvis", "L_Wrist", "R_Wrist"], ["left_hand"])
    result = build_always_body_vertex_weight_result(
        body_parts=["left_hand"],
        body_mapping=mapping,
        adjacency=adjacency,
        num_human_vertices=3,
        num_vertices=6,
        config=SemanticRetargetingConfig(mode="uniform2_semantic_weight_uniform"),
    )
    scaled = scale_semantic_weight_result(result, 1.25)
    assert scaled.weights.mean() == pytest.approx(1.0, abs=1e-12)
    assert scaled.l1_norm_alpha_minus_one == pytest.approx(1.25 * result.l1_norm_alpha_minus_one)


def test_precision_evaluator_uses_raw_geometry_and_original_criticality() -> None:
    num_frames = 40
    residuals = np.ones((num_frames, 5), dtype=np.float64)
    adjacency = np.zeros((num_frames, 5, 5), dtype=np.uint8)
    adjacency[:, 1, 3] = adjacency[:, 3, 1] = 1
    adjacency[:, 2, 4] = adjacency[:, 4, 2] = 1
    source = np.zeros((num_frames, 5, 3), dtype=np.float64)
    target = source.copy()
    target[:, 1, 0] = 1.0
    target[:, 2, 0] = 2.0
    events = [
        _event("contact", 5, 4, 6, ["left_hand"], criticality=1.0),
        _event("lift", 13, 12, 14, ["right_hand"], criticality=0.25),
        _event("place", 21, 20, 22, ["left_hand"], criticality=1.0),
        _event("release", 29, 28, 30, ["right_hand"], criticality=0.25),
    ]
    payload = PrecisionPayload(residuals, adjacency, ("Pelvis", "L_Wrist", "R_Wrist"), 3, source, target)
    result = evaluate_precision(payload, events)
    assert result.semantic_edge is not None
    assert result.semantic_edge["exact"] == pytest.approx(1.5)
    assert result.criticality_weighted_edge is not None
    assert result.criticality_weighted_edge["exact"] == pytest.approx(1.2)
