from __future__ import annotations

import numpy as np
import pytest

from holosoma_retargeting.semantic_keyframes.contact_targets import (
    contact_objective_schedule,
    hysteretic_contact_mask,
    semantic_contact_mask,
)


def test_hysteretic_contact_enters_at_two_cm_and_holds_to_three_cm() -> None:
    distances = np.array([0.04, 0.025, 0.019, 0.026, 0.031, 0.025, 0.018, 0.029, 0.04])
    assert hysteretic_contact_mask(distances).tolist() == [
        False,
        False,
        True,
        True,
        False,
        False,
        True,
        True,
        False,
    ]


def test_hysteretic_contact_rejects_invalid_threshold_order() -> None:
    with pytest.raises(ValueError, match="onset_threshold"):
        hysteretic_contact_mask(np.array([0.0]), 0.03, 0.02)


def test_semantic_participation_is_not_vetoed_by_missing_left_skin_contact() -> None:
    distances = np.array([[0.10, 0.05], [0.09, 0.019], [0.08, 0.029], [0.09, 0.04]])
    assert semantic_contact_mask(distances, activation_mode="semantic_group").tolist() == [
        [False, False], [True, True], [True, True], [False, False]
    ]
    assert semantic_contact_mask(distances).tolist() == [
        [False, False], [False, True], [False, True], [False, False]
    ]


def test_semantic_group_still_requires_contact_timing_evidence() -> None:
    distances = np.full((3, 2), 0.08)
    assert not semantic_contact_mask(distances, activation_mode="semantic_group").any()


@pytest.mark.parametrize("distances", [np.array([0.0]), np.zeros((3, 0)), np.array([[np.nan]])])
def test_semantic_contact_mask_rejects_invalid_distances(distances: np.ndarray) -> None:
    with pytest.raises(ValueError, match="finite distances_m"):
        semantic_contact_mask(distances)


def test_semantic_contact_mask_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="activation_mode"):
        semantic_contact_mask(np.zeros((3, 2)), activation_mode="whole_body")


def test_contact_objective_schedule_ramps_only_before_each_onset() -> None:
    active = np.zeros((12, 2), dtype=bool)
    active[6:, 0] = True
    active[9:, 1] = True

    objective, target_frames, strength = contact_objective_schedule(active, 5)

    # Contact truth is not mutated. Each five-frame approach points at that
    # part's own onset target; only the onset itself retains unit weight.
    assert not active[:6, 0].any()
    assert objective[:, 0].tolist() == [False, True, True, True, True, True, True, False, False, False, False, False]
    assert objective[:, 1].tolist() == [False, False, False, False, True, True, True, True, True, True, False, False]
    np.testing.assert_array_equal(target_frames[1:6, 0], 6)
    np.testing.assert_array_equal(target_frames[4:9, 1], 9)
    assert np.all(np.diff(strength[1:7, 0]) > 0.0)
    assert np.all(np.diff(strength[4:10, 1]) > 0.0)
    assert strength[6, 0] == 1.0
    assert strength[9, 1] == 1.0
    assert not strength[7:, 0].any()
    assert not strength[10:, 1].any()


def test_zero_approach_optimizes_onsets_only() -> None:
    active = np.array([[False, True], [True, True], [True, False]])
    objective, target_frames, strength = contact_objective_schedule(active, 0)
    np.testing.assert_array_equal(objective, [[False, True], [True, False], [False, False]])
    np.testing.assert_array_equal(target_frames, [[0, 0], [1, 1], [2, 2]])
    np.testing.assert_array_equal(strength, objective.astype(float))


def test_gaussian_schedule_is_continuous_around_contact_onset() -> None:
    active = np.zeros((144, 2), dtype=bool)
    active[44:, :] = True

    objective, target_frames, strength = contact_objective_schedule(
        active,
        0,
        temporal_schedule="gaussian",
        gaussian_sigma_frames=10.0,
        gaussian_min_relative_weight=1e-6,
    )

    assert objective[0].all()
    assert objective[96].all()
    assert not objective[97:].any()
    np.testing.assert_array_equal(target_frames[objective], 44)
    assert strength[44, 0] == 1.0
    assert strength[0, 0] == pytest.approx(np.exp(-0.5 * (44.0 / 10.0) ** 2))
    assert strength[39, 0] == pytest.approx(strength[49, 0])
    assert np.all(np.diff(strength[:45, 0]) > 0.0)
    assert np.all(np.diff(strength[44:97, 0]) < 0.0)
    assert not active[:44].any()


def test_smooth_window_has_compact_zero_slope_rise_and_release() -> None:
    active = np.zeros((70, 1), dtype=bool)
    active[44:, 0] = True

    objective, target_frames, strength = contact_objective_schedule(
        active,
        10,
        temporal_schedule="smooth_window",
        release_frames=5,
    )

    assert not objective[:34].any()
    assert objective[34:49].all()
    assert not objective[49:].any()
    np.testing.assert_array_equal(target_frames[34:49, 0], 44)
    assert strength[34, 0] == 0.0
    assert strength[35, 0] == pytest.approx(0.00856)
    assert strength[39, 0] == pytest.approx(0.5)
    assert strength[44, 0] == 1.0
    assert strength[45, 0] == pytest.approx(0.94208)
    assert strength[48, 0] == pytest.approx(0.05792)


@pytest.mark.parametrize("approach_frames", [-1, 1.5])
def test_contact_objective_schedule_rejects_invalid_window(approach_frames) -> None:
    with pytest.raises(ValueError, match="approach_frames"):
        contact_objective_schedule(np.zeros((2, 1), dtype=bool), approach_frames)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"temporal_schedule": "step"}, "temporal_schedule"),
        ({"gaussian_sigma_frames": 0.0}, "gaussian_sigma_frames"),
        ({"gaussian_min_relative_weight": 1.0}, "gaussian_min_relative_weight"),
        ({"release_frames": -1}, "release_frames"),
    ],
)
def test_contact_objective_schedule_rejects_invalid_gaussian_settings(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        contact_objective_schedule(np.zeros((2, 1), dtype=bool), 0, **kwargs)
