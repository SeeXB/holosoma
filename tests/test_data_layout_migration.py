"""The data move must preserve live readers and links to external datasets."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.reorganize_experiment_inputs import migrate


def test_move_preserves_live_file_and_historical_path(tmp_path):
    old = tmp_path / "exp/task/input"
    old.mkdir(parents=True)
    source = old / "motion.npz"
    source.write_bytes(b"reference bytes")
    inode = source.stat().st_ino
    with source.open("rb") as reader:
        report = migrate([(Path("exp/task/input"), Path("data/task"))], tmp_path)
        assert reader.read() == b"reference bytes"
    assert old.is_symlink()
    assert source.read_bytes() == b"reference bytes"
    assert (tmp_path / "data/task/motion.npz").stat().st_ino == inode
    assert report[0]["verified_identity"]


def test_move_preserves_relative_external_and_internal_links(tmp_path):
    root = tmp_path / "repo"
    old = root / "exp/input"
    old.mkdir(parents=True)
    outside = tmp_path / "external"
    outside.mkdir()
    (outside / "data").write_bytes(b"external")
    (root / "exp/another").mkdir()
    (root / "exp/another/data").write_bytes(b"internal")
    (old / "external").symlink_to("../../../external", target_is_directory=True)
    (old / "internal").symlink_to("../another/data")
    migrate([(Path("exp/input"), Path("src/pkg/data/input")),
             (Path("exp/another"), Path("src/pkg/data/another"))], root)
    assert (root / "src/pkg/data/input/external").is_symlink()
    assert (root / "src/pkg/data/input/external").resolve() == outside
    assert (root / "src/pkg/data/input/internal").read_bytes() == b"internal"
    assert (outside / "data").read_bytes() == b"external"


def test_conflicting_destination_rejected_before_any_move(tmp_path):
    (tmp_path / "a").write_bytes(b"a")
    (tmp_path / "b").write_bytes(b"b")
    (tmp_path / "existing").mkdir()
    with pytest.raises(FileExistsError):
        migrate([(Path("a"), Path("new/a")), (Path("b"), Path("existing"))], tmp_path)
    assert not (tmp_path / "new").exists()
    assert not (tmp_path / "a").is_symlink()
