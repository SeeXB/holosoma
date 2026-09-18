"""Canonical SMPL-H joint ordering used by the retargeting pipeline."""

from __future__ import annotations

import numpy as np


# ``human_body_prior.BodyModel.Jtr`` follows the native SMPL-H kinematic-tree
# order.  OmniRetarget's ``SMPLH_DEMO_JOINTS`` uses limbs grouped by side, so
# coordinates must be permuted before they are labelled or packed.
SMPLH_NATIVE_TO_RETARGET = np.asarray(
    [
        0, 1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12, 15, 13, 16, 18, 20,
        *range(22, 37), 14, 17, 19, 21, *range(37, 52),
    ],
    dtype=np.int64,
)

SMPLH_RETARGET_JOINT_NAMES = (
    "Pelvis", "L_Hip", "L_Knee", "L_Ankle", "L_Toe",
    "R_Hip", "R_Knee", "R_Ankle", "R_Toe",
    "Torso", "Spine", "Chest", "Neck", "Head",
    "L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist",
    "L_Index1", "L_Index2", "L_Index3",
    "L_Middle1", "L_Middle2", "L_Middle3",
    "L_Pinky1", "L_Pinky2", "L_Pinky3",
    "L_Ring1", "L_Ring2", "L_Ring3",
    "L_Thumb1", "L_Thumb2", "L_Thumb3",
    "R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist",
    "R_Index1", "R_Index2", "R_Index3",
    "R_Middle1", "R_Middle2", "R_Middle3",
    "R_Pinky1", "R_Pinky2", "R_Pinky3",
    "R_Ring1", "R_Ring2", "R_Ring3",
    "R_Thumb1", "R_Thumb2", "R_Thumb3",
)

SMPLH_RETARGET_LAYOUT = "smplh_retarget_v1"


def reorder_smplh_native_to_retarget(joints: np.ndarray) -> np.ndarray:
    """Convert BodyModel native ``Jtr`` columns to OmniRetarget order."""
    values = np.asarray(joints)
    if values.ndim != 3 or values.shape[1:] != (52, 3):
        raise ValueError(f"Expected native SMPL-H joints [T,52,3], got {values.shape}")
    return values[:, SMPLH_NATIVE_TO_RETARGET]


def validate_retarget_smplh_geometry(joints: np.ndarray) -> None:
    """Reject the native-as-retarget ordering regression using limb topology."""
    values = np.asarray(joints)
    if values.ndim != 3 or values.shape[1:] != (52, 3) or not np.isfinite(values).all():
        raise ValueError(f"Expected finite retarget SMPL-H joints [T,52,3], got {values.shape}")
    lengths = lambda a, b: float(np.median(np.linalg.norm(values[:, a] - values[:, b], axis=1)))
    left_thigh, right_thigh = lengths(1, 2), lengths(5, 6)
    left_shin, right_shin = lengths(2, 3), lengths(6, 7)
    limb_lengths = np.asarray([left_thigh, right_thigh, left_shin, right_shin])
    if np.any((limb_lengths < 0.15) | (limb_lengths > 0.75)):
        raise ValueError(
            "SMPL-H joint topology is implausible; BodyModel native Jtr was likely "
            "labelled as OmniRetarget order without applying SMPLH_NATIVE_TO_RETARGET"
        )
    if max(left_thigh, right_thigh) / min(left_thigh, right_thigh) > 1.5:
        raise ValueError("SMPL-H left/right thigh lengths are inconsistent; check joint ordering")
    if max(left_shin, right_shin) / min(left_shin, right_shin) > 1.5:
        raise ValueError("SMPL-H left/right shin lengths are inconsistent; check joint ordering")
