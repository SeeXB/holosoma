from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))

from build_force_affordance_targets import _smooth_activation
from run_current_semantic_retarget_comparison import (
    validate_force_affordance_artifact,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_artifact(
    path: Path,
    *,
    source: Path,
    plan: Path,
    include_qpos: bool = False,
    source_only: bool = True,
) -> None:
    interval = path.with_suffix(".interval.json")
    interval.write_text(
        json.dumps(
            {
                "schema": "holosoma.force_affordance_interval.v1",
                "policy": {"manual_frame_overrides": False},
            }
        )
    )
    metadata = {
        "activation_mode": "force_affordance_interval",
        "no_retargeted_robot_trajectory_input": source_only,
        "force_affordance_robot_bodies": {
            "left_hand": "left_rubber_hand_link",
            "right_hand": "right_rubber_hand_link",
        },
        "interval_derivation": {
            "schema": "holosoma.force_affordance_interval.v1",
            "policy": {"manual_frame_overrides": False},
        },
        "grip_start": 0,
        "end_frame": 3,
        "inputs": {
            "official_smpl_sha256": _sha256(source),
            "semantic_plan_sha256": _sha256(plan),
            "interval_plan": str(interval),
            "interval_plan_sha256": _sha256(interval),
        },
    }
    values = {
        "metadata_json": np.asarray(json.dumps(metadata)),
        "active": np.ones((4, 2), dtype=bool),
    }
    if include_qpos:
        values["qpos"] = np.zeros((4, 43), dtype=np.float32)
    np.savez_compressed(path, **values)


def test_force_activation_starts_at_contact_not_before() -> None:
    gamma = _smooth_activation(frames=9, start=3, ramp=3)
    np.testing.assert_array_equal(gamma[:3], np.zeros(3))
    np.testing.assert_allclose(gamma[3:6], [7 / 27, 20 / 27, 1.0])
    np.testing.assert_array_equal(gamma[6:], np.ones(3))


def test_force_activation_stops_at_derived_release() -> None:
    gamma = _smooth_activation(frames=9, start=2, ramp=2, end=5)
    np.testing.assert_array_equal(gamma[:2], np.zeros(2))
    np.testing.assert_allclose(gamma[2:6], [0.5, 1.0, 1.0, 1.0])
    np.testing.assert_array_equal(gamma[6:], np.zeros(3))


def test_source_only_artifact_validation_rejects_robot_qpos(tmp_path: Path) -> None:
    official = tmp_path / "official_inputs"
    official.mkdir()
    source = official / "sub7_smallbox_023.pt"
    source.write_bytes(b"official-smpl")
    plan = tmp_path / "semantic_plan.json"
    plan.write_text("{}")

    valid = tmp_path / "valid.npz"
    _write_artifact(valid, source=source, plan=plan)
    metadata = validate_force_affordance_artifact(
        valid,
        official_source=source,
        semantic_plan=plan,
    )
    assert metadata["no_retargeted_robot_trajectory_input"] is True

    invalid = tmp_path / "contains_qpos.npz"
    _write_artifact(invalid, source=source, plan=plan, include_qpos=True)
    with pytest.raises(ValueError, match="robot trajectory arrays"):
        validate_force_affordance_artifact(
            invalid,
            official_source=source,
            semantic_plan=plan,
        )


def test_source_only_artifact_validation_checks_official_source_hash(
    tmp_path: Path,
) -> None:
    official = tmp_path / "official_inputs"
    official.mkdir()
    source = official / "sub7_smallbox_023.pt"
    source.write_bytes(b"official-smpl")
    plan = tmp_path / "semantic_plan.json"
    plan.write_text("{}")
    artifact = tmp_path / "targets.npz"
    _write_artifact(artifact, source=source, plan=plan)

    source.write_bytes(b"changed-source")
    with pytest.raises(ValueError, match="source mismatch"):
        validate_force_affordance_artifact(
            artifact,
            official_source=source,
            semantic_plan=plan,
        )
