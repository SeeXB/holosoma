from __future__ import annotations

import pytest
import torch

from holosoma.managers.command.terms.wbt import AdaptiveTimestepsSampler


pytestmark = pytest.mark.no_sim


def test_default_bin_count_is_unchanged() -> None:
    sampler = AdaptiveTimestepsSampler(239, "cpu", 50)
    assert sampler.num_bins == 5


def test_four_bins_make_frame_191_an_interior_sample() -> None:
    sampler = AdaptiveTimestepsSampler(
        239,
        "cpu",
        50,
        adaptive_uniform_ratio=0.0,
        adaptive_num_bins=4,
    )
    sampler.bin_failed_count[-1] = 1.0
    torch.manual_seed(7)
    frames = sampler.sample_global_time_steps(8192)

    assert sampler.num_bins == 4
    assert frames.min().item() == 179
    assert frames.max().item() == 238
    assert torch.any(frames == 191)


def test_explicit_bin_count_must_be_positive() -> None:
    with pytest.raises(ValueError, match="adaptive_num_bins"):
        AdaptiveTimestepsSampler(239, "cpu", 50, adaptive_num_bins=0)
