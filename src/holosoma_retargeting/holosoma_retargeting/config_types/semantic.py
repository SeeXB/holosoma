"""Configuration for semantic-keyframe-aware OmniRetarget runs.

The active research path is deliberately small: a Uniform-2 base, the
registered Legacy residual weighting, and an optional exact-trigger 2 -> 4
budget.  Older non-criticality modes remain readable so historical commands
and cached artifacts can still be inspected, but criticality/Edge/adaptive
ablation modes are no longer accepted here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, get_args


SemanticMode = Literal[
    "original",
    "uniform",
    "uniform2_semantic_weight_uniform",
    "uniform2_semantic_weight_semantic_budget",
    "uniform2_semantic_weight_random_budget",
    "uniform2_original_objective_semantic_budget",
]
ACTIVE_SEMANTIC_MODES = tuple(str(mode) for mode in get_args(SemanticMode))


@dataclass
class SemanticRetargetingConfig:
    """Fixed semantic runtime configuration.

    ``uniform2_semantic_weight_uniform`` is the immutable Legacy anchor.  The
    two weighted budget modes use the exact same objective and differ only in
    where their seven (for the registered sequence) extra 2-iteration slots
    are placed.  ``uniform2_original_objective_semantic_budget`` uses the same
    semantic timing allocation without residual weighting.
    """

    mode: SemanticMode = "original"
    semantic_keyframe_path: Path | None = None
    profile_dir: Path | None = None
    random_seed: int = 0

    # Active scheduler constants.
    frame0_budget: int = 50
    original_budget: int = 10
    uniform_budget: int = 2
    near_trigger_radius: int = 3
    round2_base_budget: int = 2

    # This round is pre-registered at exactly 2 -> 4; it is not a tuning knob.
    exact_trigger_budget: int = 4
    random_exclusion_radius: int = 3

    # Registered Legacy weighting constants.  These must not drift.
    semantic_temporal_sigma: float = 2.0
    phase_semantic_weight: float = 0.25
    semantic_kernel_cutoff: float = 1e-3
    body_weight_multiplier: dict[str, float] = field(
        default_factory=lambda: {
            "left_hand": 4.0,
            "right_hand": 4.0,
            "pelvis": 2.0,
            "left_foot": 2.0,
            "right_foot": 2.0,
            "torso": 2.0,
        }
    )
    object_neighbor_multiplier: float = 2.0

    # Historical Legacy support set.  This is event selection, not a
    # criticality scalar: every selected event has exactly equal strength.
    legacy_weight_events: tuple[str, ...] = (
        "contact",
        "lift",
        "place",
        "release",
    )

    # Existing physical rescue/audit settings retained for old non-active
    # modes and for method-independent diagnostics.
    rescue_penetration_tolerance: float = 1e-3
    rescue_self_collision_tolerance: float = 1e-3
    rescue_joint_limit_tolerance: float = 1e-6
    rescue_foot_sticking_tolerance: float = 1e-3
    rescue_hand_object_tolerance: float = 0.05
    rescue_velocity_limit_per_frame: float | None = None

    @property
    def uses_semantic_events(self) -> bool:
        return self.mode not in {"original", "uniform"}

    @property
    def uses_semantic_weights(self) -> bool:
        return self.mode in {
            "uniform2_semantic_weight_uniform",
            "uniform2_semantic_weight_semantic_budget",
            "uniform2_semantic_weight_random_budget",
        }

    @property
    def uses_exact_semantic_budget(self) -> bool:
        return self.mode in {
            "uniform2_semantic_weight_semantic_budget",
            "uniform2_original_objective_semantic_budget",
        }

    @property
    def uses_random_exact_budget(self) -> bool:
        return self.mode == "uniform2_semantic_weight_random_budget"

    @property
    def is_uniform2_mainline(self) -> bool:
        return self.mode in {
            "uniform2_semantic_weight_uniform",
            "uniform2_semantic_weight_semantic_budget",
            "uniform2_semantic_weight_random_budget",
            "uniform2_original_objective_semantic_budget",
        }

    @property
    def uses_final_weighting(self) -> bool:
        return self.uses_semantic_weights

    @property
    def semantic_weight_components(self) -> str:
        # Edge weighting and new objectives are intentionally absent.
        return "part"

    def validate(self) -> None:
        if self.mode not in ACTIVE_SEMANTIC_MODES:
            raise ValueError(
                f"unsupported active semantic mode {self.mode!r}; expected one of {ACTIVE_SEMANTIC_MODES}"
            )
        if self.frame0_budget != 50:
            raise ValueError("frame0_budget is registered at 50")
        if self.round2_base_budget != 2:
            raise ValueError("round2_base_budget is registered at 2")
        if self.exact_trigger_budget != 4:
            raise ValueError("exact_trigger_budget is registered at 4")
        if self.random_exclusion_radius != 3:
            raise ValueError("random_exclusion_radius is registered at 3")
        if self.semantic_temporal_sigma != 2.0:
            raise ValueError("semantic_temporal_sigma is registered at 2.0")
        if self.phase_semantic_weight != 0.25:
            raise ValueError("phase_semantic_weight is registered at 0.25")
        if self.object_neighbor_multiplier != 2.0:
            raise ValueError("object_neighbor_multiplier is registered at 2.0")
        if self.random_seed < 0:
            raise ValueError("random_seed must be non-negative")
        if self.mode in {
            "uniform2_semantic_weight_uniform",
            "uniform2_semantic_weight_semantic_budget",
            "uniform2_semantic_weight_random_budget",
            "uniform2_original_objective_semantic_budget",
        } and self.semantic_keyframe_path is None:
            raise ValueError(f"semantic_keyframe_path is required for mode={self.mode!r}")
