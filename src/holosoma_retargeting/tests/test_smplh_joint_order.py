"""Regression tests for the OMOMO BodyModel-to-retarget joint permutation."""

from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools/omomo_cari4d_renderer"))

from smplh_joint_order import (  # noqa: E402
    SMPLH_NATIVE_TO_RETARGET,
    SMPLH_RETARGET_JOINT_NAMES,
    reorder_smplh_native_to_retarget,
    validate_retarget_smplh_geometry,
)
sys.path.insert(0, str(ROOT / "src/holosoma_retargeting"))
from holosoma_retargeting.semantic_keyframes.pipeline import (  # noqa: E402
    load_retargeting_bundle_signals,
)


def test_native_to_retarget_permutation_matches_known_smplh_topology():
    assert len(SMPLH_NATIVE_TO_RETARGET) == 52
    assert len(np.unique(SMPLH_NATIVE_TO_RETARGET)) == 52
    expected = {
        "Pelvis": 0,
        "L_Hip": 1,
        "L_Knee": 4,
        "L_Ankle": 7,
        "L_Toe": 10,
        "R_Hip": 2,
        "R_Knee": 5,
        "R_Ankle": 8,
        "R_Toe": 11,
        "L_Wrist": 20,
        "R_Wrist": 21,
    }
    for name, native_index in expected.items():
        target_index = SMPLH_RETARGET_JOINT_NAMES.index(name)
        assert SMPLH_NATIVE_TO_RETARGET[target_index] == native_index


def test_reorder_moves_native_columns_before_applying_names():
    native = np.broadcast_to(np.arange(52)[None, :, None], (2, 52, 3)).copy()
    reordered = reorder_smplh_native_to_retarget(native)
    np.testing.assert_array_equal(reordered[:, :, 0], np.tile(SMPLH_NATIVE_TO_RETARGET, (2, 1)))


def test_geometry_guard_rejects_native_columns_labelled_as_retarget():
    path = ROOT / "exp/omomo_cari4d/sub9_clothesstand_058/input/omomo_gt_sequence.npz"
    if not path.is_file():
        pytest.skip("local OMOMO diagnostic bundle is unavailable")
    with np.load(path, allow_pickle=False) as data:
        corrected = data["human_joints"]
    old_native_order = np.empty_like(corrected)
    old_native_order[:, SMPLH_NATIVE_TO_RETARGET] = corrected
    with pytest.raises(ValueError, match="joint topology"):
        validate_retarget_smplh_geometry(old_native_order)
    validate_retarget_smplh_geometry(reorder_smplh_native_to_retarget(old_native_order))


def test_semantic_loader_rejects_unmarked_omomo_joint_order(tmp_path):
    path = tmp_path / "old_omomo.npz"
    np.savez(
        path,
        sequence_name=np.asarray("sub9_clothesstand_058"),
        human_joints=np.zeros((2, 52, 3)),
        object_poses_wxyz_xyz=np.tile([1, 0, 0, 0, 0, 0, 0], (2, 1)),
        smplh_joint_names=np.asarray(SMPLH_RETARGET_JOINT_NAMES),
        frame_ids=np.arange(2),
        fps=np.asarray(30),
    )
    with pytest.raises(ValueError, match="lacks verified smplh_retarget_v1"):
        load_retargeting_bundle_signals(path)
