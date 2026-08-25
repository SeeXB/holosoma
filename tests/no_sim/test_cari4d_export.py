"""CPU-only tests for the source-format CARI4D adapter."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def _write_cube(
    path: Path,
    scale: float = 1.0,
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    *,
    invalid_zero_normals: bool = False,
) -> None:
    vertices = np.asarray(
        [
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
        ],
        dtype=float,
    )
    vertices = vertices * scale + np.asarray(center)
    faces = ((1, 2, 3, 4), (5, 8, 7, 6), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 8, 4), (5, 1, 4, 8))
    if invalid_zero_normals:
        face_lines = [
            "f "
            + " ".join(f"{index}//{0 if offset % 2 else 1}" for offset, index in enumerate(face))
            + "\n"
            for face in faces
        ]
    else:
        face_lines = ["f " + " ".join(map(str, face)) + "\n" for face in faces]
    lines = [*(f"v {x} {y} {z}\n" for x, y, z in vertices), "vn 0 0 1\n", *face_lines]
    path.write_text("".join(lines), encoding="utf-8")


def test_exports_real_cari4d_pr_fields_and_prevents_double_scale(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[2]
    result_path = tmp_path / "sequence.pth"
    normalized = tmp_path / "normalized.obj"
    metric = tmp_path / "metric.obj"
    scale_json = tmp_path / "scale.json"
    output = tmp_path / "sequence"
    frame_count = 5
    _write_cube(normalized, center=(1.0, -0.5, 2.0))
    _write_cube(metric, scale=0.2, center=(0.2, -0.1, 0.4), invalid_zero_normals=True)
    scale_json.write_text(json.dumps({"best_scale": 0.2}), encoding="utf-8")
    transforms = torch.eye(4).repeat(frame_count, 1, 1)
    transforms[:, 2, 3] = torch.linspace(0.8, 1.2, frame_count)
    prediction = {
        "pose_abs": transforms,
        "smpl_pose": torch.zeros(frame_count, 72),
        "smpl_t": torch.zeros(frame_count, 3),
        "betas": torch.zeros(frame_count, 10),
        "frames": [f"sequence/{index:06d}" for index in range(frame_count)],
    }
    torch.save({"gt": {}, "pr": prediction, "in": {}}, result_path)

    subprocess.run(
        [
            sys.executable, str(repository / "tools" / "export_cari4d_sequence.py"),
            "--cari4d-result", str(result_path), "--object-mesh-normalized", str(normalized),
            "--object-mesh-metric", str(metric), "--scale-json", str(scale_json),
            "--output", str(output), "--fps", "29.97",
        ],
        cwd=repository,
        check=True,
    )
    with np.load(output / "human" / "smplh.npz") as human:
        assert human["poses"].shape == (frame_count, 156)
        assert human["poses_cari4d_raw"].shape == (frame_count, 72)
    with np.load(output / "object" / "trajectory.npz") as trajectory:
        np.testing.assert_allclose(trajectory["transform"], transforms.numpy(), atol=1e-7)
        np.testing.assert_allclose(trajectory["quaternion_wxyz"][:, 0], 1.0, atol=1e-7)
    metadata = json.loads((output / "object" / "metadata.json").read_text(encoding="utf-8"))
    np.testing.assert_allclose(metadata["extent_xyz_m"], [0.4, 0.4, 0.4])
    assert metadata["source_metric_mesh_already_scaled"] is True
    assert metadata["scale_applied_by_exporter"] is False
    assert metadata["double_scale_check"] == "PASS"
    assert metadata["obj_invalid_zero_normal_face_indices_repaired"] == 6
    report = json.loads((output / "validation" / "export_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "PASS"


def test_rejects_inconsistent_metric_extent(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[2]
    result_path = tmp_path / "sequence.pth"
    normalized = tmp_path / "normalized.obj"
    wrong_metric = tmp_path / "wrong_metric.obj"
    scale_json = tmp_path / "scale.json"
    _write_cube(normalized)
    _write_cube(wrong_metric, scale=0.4)
    scale_json.write_text(json.dumps({"best_scale": 0.2}), encoding="utf-8")
    prediction = {
        "pose_abs": torch.eye(4)[None], "smpl_pose": torch.zeros(1, 156),
        "smpl_t": torch.zeros(1, 3), "betas": torch.zeros(1, 10), "frames": ["seq/000000"],
    }
    torch.save({"gt": {}, "pr": prediction, "in": {}}, result_path)
    completed = subprocess.run(
        [
            sys.executable, str(repository / "tools" / "export_cari4d_sequence.py"),
            "--cari4d-result", str(result_path), "--object-mesh-normalized", str(normalized),
            "--object-mesh-metric", str(wrong_metric), "--scale-json", str(scale_json),
            "--output", str(tmp_path / "output"), "--fps", "30",
        ],
        cwd=repository,
        text=True,
        capture_output=True,
    )
    assert completed.returncode != 0
    assert "double-scaled mesh" in completed.stderr


def test_explicitly_recovers_missing_archive_scale_from_corresponding_meshes(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[2]
    result_path = tmp_path / "sequence.pth"
    normalized = tmp_path / "normalized.obj"
    metric = tmp_path / "metric.obj"
    output = tmp_path / "output"
    _write_cube(normalized, center=(0.25, -0.5, 1.0))
    _write_cube(metric, scale=0.3, center=(0.075, -0.15, 0.3))
    prediction = {
        "pose_abs": torch.eye(4)[None], "smpl_pose": torch.zeros(1, 72),
        "smpl_t": torch.zeros(1, 3), "betas": torch.zeros(1, 10), "frames": ["seq/000000"],
    }
    torch.save({"gt": {}, "pr": prediction, "in": {}}, result_path)
    subprocess.run(
        [
            sys.executable, str(repository / "tools" / "export_cari4d_sequence.py"),
            "--cari4d-result", str(result_path), "--object-mesh-normalized", str(normalized),
            "--object-mesh-metric", str(metric), "--derive-scale-from-metric-mesh",
            "--output", str(output), "--fps", "30",
        ],
        cwd=repository,
        check=True,
    )
    metadata = json.loads((output / "object" / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["cari4d_metric_scale"] == pytest.approx(0.3)
    assert metadata["source_scale_json"] is None
    assert metadata["cari4d_scale_provenance"] == (
        "derived_from_explicit_corresponding_cari4d_normalized_and_metric_meshes"
    )
    assert metadata["derived_scale_max_vertex_error_m"] < 1e-7


def test_known_metric_asset_bypasses_scale_estimation(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[2]
    result_path = tmp_path / "sequence.pth"
    metric = tmp_path / "known_metric.obj"
    output = tmp_path / "output"
    _write_cube(metric, scale=0.15, center=(0.02, -0.03, 0.04))
    prediction = {
        "pose_abs": torch.eye(4)[None], "smpl_pose": torch.zeros(1, 72),
        "smpl_t": torch.zeros(1, 3), "betas": torch.zeros(1, 10), "frames": ["seq/000000"],
    }
    torch.save({"gt": {}, "pr": prediction, "in": {}}, result_path)
    subprocess.run(
        [
            sys.executable, str(repository / "tools" / "export_cari4d_sequence.py"),
            "--cari4d-result", str(result_path), "--object-mesh-metric", str(metric),
            "--object-mesh-is-metric", "--output", str(output), "--fps", "30",
        ],
        cwd=repository,
        check=True,
    )
    metadata = json.loads((output / "object" / "metadata.json").read_text(encoding="utf-8"))
    np.testing.assert_allclose(metadata["extent_xyz_m"], [0.3, 0.3, 0.3])
    assert metadata["cari4d_metric_scale"] == 1.0
    assert metadata["cari4d_scale_provenance"] == "known_metric_asset_no_scale_estimation"
    assert metadata["source_scale_json"] is None
    assert metadata["scale_applied_by_exporter"] is False
    assert metadata["double_scale_check"] == "PASS"
