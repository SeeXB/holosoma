"""Regression tests exercising the real legacy reader, not a copied inverse."""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src/holosoma_retargeting"))
from prepare_batch_retarget_inputs import _write_pt, canonicalize_object_poses
from holosoma_retargeting.src.utils import load_intermimic_data


def sample():
    rng = np.random.default_rng(42)
    human = rng.normal(size=(8, 52, 3)).astype(np.float32)
    quat = rng.normal(size=(8, 4)).astype(np.float32)
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    # Large, distinct xyz values prevent accidental ambiguity with quaternion columns.
    pos = rng.uniform(2, 5, size=(8, 3)).astype(np.float32)
    return human, np.concatenate([quat, pos], axis=1)


def test_real_reader_roundtrip(tmp_path):
    human, pose = sample()
    path = tmp_path / "valid.pt"
    _write_pt(path, human, pose)
    h, p = load_intermimic_data(str(path))
    np.testing.assert_array_equal(h, human)
    np.testing.assert_array_equal(p, pose)
    packed = torch.load(path, weights_only=False).numpy()
    np.testing.assert_array_equal(packed[:, 318:321], pose[:, 4:])
    np.testing.assert_array_equal(packed[:, 321:325], pose[:, [1, 2, 3, 0]])


@pytest.mark.parametrize("invalid", ["nan", "quaternion", "shape"])
def test_writer_rejects_invalid_before_creating_file(tmp_path, invalid):
    human, pose = sample()
    if invalid == "nan":
        pose[0, 4] = np.nan
    elif invalid == "quaternion":
        pose[0, :4] *= 2
    else:
        pose = pose[:, :-1]
    path = tmp_path / "invalid.pt"
    with pytest.raises(ValueError):
        _write_pt(path, human, pose)
    assert not path.exists()


def test_reader_rejects_old_broken_layout(tmp_path):
    human, pose = sample()
    packed = torch.zeros((8, 591))
    packed[:, 162:318] = torch.from_numpy(human.reshape(8, -1))
    packed[:, 318:325] = torch.from_numpy(pose[:, [1, 2, 3, 4, 5, 6, 0]])
    path = tmp_path / "old_broken.pt"
    torch.save(packed, path)
    with pytest.raises(ValueError, match="packed xyz_xyzw"):
        load_intermimic_data(str(path))


def test_canonical_origin_shift_preserves_world_vertices():
    from scipy.spatial.transform import Rotation
    rng = np.random.default_rng(123)
    source = rng.normal(size=(40, 3)) + [2, -3, 4]
    scale = 0.3
    asset = scale * (source-source.mean(0)) + [0.02, 0.03, -0.04]
    _, poses = sample()
    aligned, metadata = canonicalize_object_poses(poses, source, np.full(len(poses), scale), asset)
    rotations = Rotation.from_quat(poses[:, [1,2,3,0]]).as_matrix()
    expected = scale * np.einsum('tij,vj->tvi', rotations, source) + poses[:, None, 4:]
    actual = np.einsum('tij,vj->tvi', rotations, asset) + aligned[:, None, 4:]
    np.testing.assert_allclose(actual, expected, atol=5e-7)
    np.testing.assert_array_equal(aligned[:, :4], poses[:, :4])
    assert metadata['mesh_fit_max_error_m'] < 1e-12


def test_canonical_alignment_rejects_wrong_mesh():
    _, poses = sample()
    source = np.arange(18).reshape(6, 3).astype(float)
    wrong = source.copy()
    wrong[0, 0] += 1
    with pytest.raises(ValueError, match="uniformly scaled"):
        canonicalize_object_poses(poses, source, np.ones(len(poses)), wrong)
