"""Configuration types for reward manager."""

from __future__ import annotations

from dataclasses import field
from typing import Any

from pydantic.dataclasses import dataclass


def _default_semantic_body_part_mapping() -> dict[str, list[str]]:
    """Default abstract semantic-part mapping for the Unitree G1 WBT model."""
    return {
        "pelvis": ["pelvis"],
        "torso": ["torso_link"],
        "left_hand": ["left_wrist_yaw_link"],
        "right_hand": ["right_wrist_yaw_link"],
        "left_foot": ["left_ankle_roll_link"],
        "right_foot": ["right_ankle_roll_link"],
    }


@dataclass(frozen=True)
class SemanticKeyframeRewardCfg:
    """Task-agnostic semantic-keyframe reward configuration.

    Event names are deliberately absent from this configuration: temporal
    activation depends only on event times/windows, and spatial activation
    depends only on the declared body parts.
    """

    enabled: bool = False
    semantic_file: str = ""
    semantic_fps: float | None = None
    sigma_time: float = 0.12

    enable_part: bool = True
    enable_rel: bool = True
    enable_dyn: bool = True

    body_part_mapping: dict[str, list[str]] = field(default_factory=_default_semantic_body_part_mapping)
    transition_truncation: bool = True


@dataclass(frozen=True)
class RewardTermCfg:
    """Configuration for a single reward term."""

    func: str
    """Import path to the reward function or class"""
    """(e.g. ``holosoma.managers.reward.terms.locomotion:tracking_lin_vel``)."""

    params: dict[str, Any] = field(default_factory=dict)
    """Additional parameters forwarded to the reward term."""

    weight: float = 1.0
    """Weight applied to the reward term (manager multiplies by ``dt``)."""

    tags: list[str] = field(default_factory=list)
    """Tags for categorizing reward terms (e.g., ["penalty", "tracking"])."""


@dataclass(frozen=True)
class RewardManagerCfg:
    """Configuration for the reward manager."""

    terms: dict[str, RewardTermCfg] = field(default_factory=dict)
    """Mapping of reward term name to configuration."""

    only_positive_rewards: bool = False
    """If ``True``, clip the total reward to be non-negative."""

    semantic_keyframe: SemanticKeyframeRewardCfg | None = None
    """One shared semantic runtime configuration for all semantic terms."""
