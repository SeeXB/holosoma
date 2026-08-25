"""CPU-only tests for multi-frame collision geometry helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "tools"))
import estimate_multiframe_collision_box as collision  # noqa: E402


def test_voxel_support_removes_single_frame_outliers() -> None:
    shared = np.asarray(
        [
            [x, y, z]
            for x in np.linspace(-0.5, 0.5, 5)
            for y in np.linspace(-1.0, 1.0, 5)
            for z in np.linspace(-1.5, 1.5, 5)
        ],
        dtype=np.float64,
    )
    points = []
    frame_ids = []
    for frame in range(4):
        points.append(shared)
        points.append(np.asarray([[10.0 + frame, 10.0, 10.0]]))
        frame_ids.extend([frame] * (len(shared) + 1))
    filtered, downsampled, report = collision.voxel_frame_support_filter(
        np.concatenate(points), np.asarray(frame_ids), voxel_size=0.05, min_frame_support=2
    )
    assert len(filtered) == len(shared) * 4
    assert len(downsampled) == len(shared)
    assert report["retained_voxels"] == len(shared)
    assert np.max(np.abs(filtered)) <= 1.5


def test_robust_obb_exports_expected_box_dimensions() -> None:
    rng = np.random.default_rng(4)
    points = rng.uniform([-0.5, -1.0, -1.5], [0.5, 1.0, 1.5], size=(20_000, 3))
    box, report = collision.fit_robust_obb(points, points[::10], tail_quantile=0.0)
    np.testing.assert_allclose(report["extent_sorted_m"], [1.0, 2.0, 3.0], atol=0.04)
    np.testing.assert_allclose(np.sort(box.extents), [1.0, 2.0, 3.0], atol=0.04)
