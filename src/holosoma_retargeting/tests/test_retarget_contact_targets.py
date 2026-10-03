from __future__ import annotations

import numpy as np
import pytest

from holosoma_retargeting.semantic_keyframes.contact_targets import (
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
