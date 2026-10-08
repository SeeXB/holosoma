from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))

from derive_force_affordance_interval import derive_interval


def _event(name: str, start: int, end: int) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        start_frame=start,
        end_frame=end,
        body_parts=("left_hand", "right_hand"),
    )


def test_contact_state_bridges_semantic_gap_and_detects_release() -> None:
    frames = 24
    distances = np.full((frames, 2), 0.07)
    distances[18] = 0.074
    distances[19:] = np.column_stack(
        (
            np.linspace(0.10, 0.20, frames - 19),
            np.linspace(0.11, 0.21, frames - 19),
        )
    )
    heights = np.zeros(frames)
    heights[11] = 1.0
    result = derive_interval(
        source_distance_m=distances,
        object_height_m=heights,
        semantic_events=(
            _event("lifting", 3, 7),
            _event("lowering", 9, 14),
        ),
        hold_quantile=0.95,
        release_margin_m=0.005,
        release_persistence_frames=3,
    )
    assert result["start_frame"] == 3
    assert result["semantic_end_frame"] == 14
    assert result["end_frame"] == 18
    assert result["release_start_frame"] == 19
    assert result["anchor_frame"] == 11
    assert result["policy"]["manual_frame_overrides"] is False


def test_contact_onset_requires_persistent_bilateral_neighborhood_entry() -> None:
    frames = 20
    distances = np.full((frames, 2), 0.20)
    distances[6] = 0.05  # Single-frame near-contact must not start the interval.
    distances[9:] = 0.05
    heights = np.zeros(frames)
    heights[12] = 1.0
    result = derive_interval(
        source_distance_m=distances,
        object_height_m=heights,
        semantic_events=(_event("pick and lift", 3, 15),),
        contact_onset_threshold_m=0.06,
        contact_onset_persistence_frames=3,
    )
    assert result["semantic_start_frame"] == 3
    assert result["start_frame"] == 9
    assert result["anchor_frame"] == 12
    assert result["evidence"]["contact_onset_run_frames"] == [9, 10, 11]
    assert result["parameters"]["contact_onset_threshold_m"] == 0.06
