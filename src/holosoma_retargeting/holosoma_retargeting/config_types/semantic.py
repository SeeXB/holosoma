"""Configuration for semantic-keyframe-aware OmniRetarget runs.

The active research path is deliberately small: a Uniform-2 base, the
registered four-event Legacy residual weighting, a distinct full-event
residual weighting, and optional exact-trigger refinement budgets.  Older
non-criticality modes remain readable so historical commands and cached
artifacts can still be inspected, but criticality/Edge/adaptive ablation modes
are no longer accepted here.
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
    "uniform2_semantic_weight_full_event",
    "uniform2_semantic_weight_full_event_approach_body_only",
    "uniform2_semantic_weight_full_event_transition_truncated",
    "uniform2_semantic_weight_full_event_transition_truncated_budget",
    "uniform2_semantic_weight_full_event_budget",
    "uniform2_semantic_weight_full_event_random_budget",
    "uniform2_original_objective_semantic_budget",
]
ACTIVE_SEMANTIC_MODES = tuple(str(mode) for mode in get_args(SemanticMode))
FINAL_SEMANTIC_MODE: SemanticMode = (
    "uniform2_semantic_weight_full_event_transition_truncated_budget"
)
FINAL_SEMANTIC_EXACT_TRIGGER_BUDGET = 4


@dataclass
class SemanticRetargetingConfig:
    """Fixed semantic runtime configuration.

    ``uniform2_semantic_weight_uniform`` is the immutable Legacy anchor.  The
    historical weighted budget modes retain that same four-event objective.
    The full-event modes instead consume every event and body-part assignment
    from the deterministic semantic plan, without name-based filtering.
    ``uniform2_original_objective_semantic_budget`` uses the same semantic
    timing allocation without residual weighting.  The registered final
    experiment recipe is transition-truncated full-event weighting with an
    exact-trigger budget of four; defaults remain backward-compatible.
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

    # Registered refinement curve. Arbitrary values remain invalid.
    exact_trigger_budget: int = 4
    random_exclusion_radius: int = 3

    # Optional profiling-only frames.  These frames do not alter the budget
    # plan or any solver parameter; they only retain accepted SQP iterates for
    # nonlinear post-solve diagnostics.
    diagnostic_iteration_frames: tuple[int, ...] = ()

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

    # Opt-in experimental correction; historical B4 and Original are unchanged.
    geometry_projection: bool = False
    geometry_projection_max_iterations: int = 10
    geometry_projection_numerical_tolerance: float = 1e-5

    # Opt-in lightweight feasibility continuation.  It reuses the final SQP
    # collision active set and bounds the extra work per frame.
    active_pair_nonpenetration_refinement: bool = False
    active_pair_max_iterations: int = 10
    active_pair_acceptance_tolerance: float = 10e-3
    active_pair_prediction_margin: float = 12e-3
    active_pair_motion_threshold: float = 5e-3
    active_pair_stagnation_tolerance: float = 1e-10

    @property
    def uses_semantic_events(self) -> bool:
        return self.mode not in {"original", "uniform"}

    @property
    def uses_semantic_weights(self) -> bool:
        return self.mode in {
            "uniform2_semantic_weight_uniform",
            "uniform2_semantic_weight_semantic_budget",
            "uniform2_semantic_weight_random_budget",
            "uniform2_semantic_weight_full_event",
            "uniform2_semantic_weight_full_event_approach_body_only",
            "uniform2_semantic_weight_full_event_transition_truncated",
            "uniform2_semantic_weight_full_event_transition_truncated_budget",
            "uniform2_semantic_weight_full_event_budget",
            "uniform2_semantic_weight_full_event_random_budget",
        }

    @property
    def uses_full_event_weights(self) -> bool:
        return self.mode in {
            "uniform2_semantic_weight_full_event",
            "uniform2_semantic_weight_full_event_approach_body_only",
            "uniform2_semantic_weight_full_event_transition_truncated",
            "uniform2_semantic_weight_full_event_transition_truncated_budget",
            "uniform2_semantic_weight_full_event_budget",
            "uniform2_semantic_weight_full_event_random_budget",
        }

    @property
    def body_only_weight_events(self) -> tuple[str, ...]:
        """Events whose body residual must not spill into object neighbors."""
        if self.mode == "uniform2_semantic_weight_full_event_approach_body_only":
            return ("approach",)
        return ()

    @property
    def uses_transition_truncated_weight_support(self) -> bool:
        """Whether event weights are causal and stop at the next transition."""
        return self.mode in {
            "uniform2_semantic_weight_full_event_transition_truncated",
            "uniform2_semantic_weight_full_event_transition_truncated_budget",
        }

    @property
    def uses_exact_semantic_budget(self) -> bool:
        return self.mode in {
            "uniform2_semantic_weight_semantic_budget",
            "uniform2_semantic_weight_full_event_budget",
            "uniform2_semantic_weight_full_event_transition_truncated_budget",
            "uniform2_original_objective_semantic_budget",
        }

    @property
    def uses_random_exact_budget(self) -> bool:
        return self.mode in {
            "uniform2_semantic_weight_random_budget",
            "uniform2_semantic_weight_full_event_random_budget",
        }

    @property
    def is_uniform2_mainline(self) -> bool:
        return self.mode in {
            "uniform2_semantic_weight_uniform",
            "uniform2_semantic_weight_semantic_budget",
            "uniform2_semantic_weight_random_budget",
            "uniform2_semantic_weight_full_event",
            "uniform2_semantic_weight_full_event_approach_body_only",
            "uniform2_semantic_weight_full_event_transition_truncated",
            "uniform2_semantic_weight_full_event_transition_truncated_budget",
            "uniform2_semantic_weight_full_event_budget",
            "uniform2_semantic_weight_full_event_random_budget",
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
        if self.geometry_projection and self.mode != FINAL_SEMANTIC_MODE:
            raise ValueError("geometry_projection is only supported for the explicit B4 variant")
        if self.geometry_projection_max_iterations < 1:
            raise ValueError("geometry_projection_max_iterations must be positive")
        if not 0 < self.geometry_projection_numerical_tolerance <= 1e-5:
            raise ValueError("geometry projection numerical tolerance must be in (0, 1e-5]")
        if self.active_pair_nonpenetration_refinement and self.mode != FINAL_SEMANTIC_MODE:
            raise ValueError(
                "active_pair_nonpenetration_refinement is only supported for the explicit B4 variant"
            )
        if self.active_pair_max_iterations < 1:
            raise ValueError("active_pair_max_iterations must be positive")
        if self.active_pair_acceptance_tolerance <= 0:
            raise ValueError("active_pair_acceptance_tolerance must be positive")
        if self.active_pair_prediction_margin < self.active_pair_acceptance_tolerance:
            raise ValueError(
                "active_pair_prediction_margin must be at least the acceptance tolerance"
            )
        if self.active_pair_prediction_margin <= 0:
            raise ValueError("active_pair_prediction_margin must be positive")
        if self.active_pair_motion_threshold <= 0:
            raise ValueError("active_pair_motion_threshold must be positive")
        if self.active_pair_stagnation_tolerance <= 0:
            raise ValueError("active_pair_stagnation_tolerance must be positive")
        if self.mode not in ACTIVE_SEMANTIC_MODES:
            raise ValueError(
                f"unsupported active semantic mode {self.mode!r}; expected one of {ACTIVE_SEMANTIC_MODES}"
            )
        if self.frame0_budget != 50:
            raise ValueError("frame0_budget is registered at 50")
        if self.round2_base_budget != 2:
            raise ValueError("round2_base_budget is registered at 2")
        if self.exact_trigger_budget not in {2, 4, 6, 8, 10}:
            raise ValueError("exact_trigger_budget must be one of {2, 4, 6, 8, 10}")
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
        if any(frame < 0 for frame in self.diagnostic_iteration_frames):
            raise ValueError("diagnostic_iteration_frames must be non-negative")
        if len(set(self.diagnostic_iteration_frames)) != len(self.diagnostic_iteration_frames):
            raise ValueError("diagnostic_iteration_frames must be unique")
        if self.mode in {
            "uniform2_semantic_weight_uniform",
            "uniform2_semantic_weight_semantic_budget",
            "uniform2_semantic_weight_random_budget",
            "uniform2_semantic_weight_full_event",
            "uniform2_semantic_weight_full_event_approach_body_only",
            "uniform2_semantic_weight_full_event_transition_truncated",
            "uniform2_semantic_weight_full_event_transition_truncated_budget",
            "uniform2_semantic_weight_full_event_budget",
            "uniform2_semantic_weight_full_event_random_budget",
            "uniform2_original_objective_semantic_budget",
        } and self.semantic_keyframe_path is None:
            raise ValueError(f"semantic_keyframe_path is required for mode={self.mode!r}")
