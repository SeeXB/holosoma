from pathlib import Path
from types import SimpleNamespace
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
import run_omomo_batch_retarget as runner


def configuration(tmp_path):
    inputs = tmp_path / "input"
    (inputs / "scenes").mkdir(parents=True)
    (inputs / "sub10_largebox_089.pt").write_bytes(b"first_input")
    (inputs / "scenes/g1_29dof_w_largebox.xml").write_text("<mujoco/>")
    return SimpleNamespace(input_root=inputs, output_root=tmp_path / "out", bundle_root=tmp_path,
                           project_root=tmp_path, package_dir=tmp_path, python=sys.executable,
                           no_foot_sticking=False, foot_sticking_tolerance=0.02, step_size=None, force=False)


def test_missing_provenance_does_not_reuse_or_overwrite_result(tmp_path, monkeypatch):
    args = configuration(tmp_path)
    result = args.output_root / "runs/original/sub10_largebox_089/sub10_largebox_089_original.npz"
    result.parent.mkdir(parents=True)
    result.write_bytes(b"old_result")
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not run")))
    row = runner._run_one("sub10_largebox_089", "original", args)
    assert row["status"] == "failed"
    assert result.read_bytes() == b"old_result"


def test_matching_cache_reused_but_changed_input_rejected(tmp_path, monkeypatch):
    args = configuration(tmp_path)
    calls = []

    def execute(command, **kwargs):
        calls.append(command)
        output = Path(command[command.index("--save-dir") + 1])
        (output / "sub10_largebox_089_original.npz").write_bytes(b"new_result")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", execute)
    assert runner._run_one("sub10_largebox_089", "original", args)["status"] == "ok"
    assert runner._run_one("sub10_largebox_089", "original", args)["status"] == "cached"
    (args.input_root / "sub10_largebox_089.pt").write_bytes(b"corrected_input")
    assert runner._run_one("sub10_largebox_089", "original", args)["status"] == "failed"
    assert len(calls) == 1
