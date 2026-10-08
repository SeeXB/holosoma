"""Load and validate source-derived anchor-force retargeting targets."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "holosoma.retarget_surface_contacts.v2"
SUPPORTED_SCHEMAS = {SCHEMA}


@dataclass(frozen=True)
class RetargetContactTargets:
    parts: tuple[str, ...]
    active: np.ndarray
    object_points_local: np.ndarray
    weights: np.ndarray
    source_distance_m: np.ndarray
    metadata: dict[str, Any]
    activation: np.ndarray | None = None
    robot_points_local: np.ndarray | None = None
    robot_normals_local: np.ndarray | None = None
    object_normals_local: np.ndarray | None = None

    def validate(self, expected_frames: int | None = None) -> None:
        frames, count = self.active.shape
        if expected_frames is not None and frames != expected_frames:
            raise ValueError(f"contact target frames {frames} != motion frames {expected_frames}")
        if len(self.parts) != count:
            raise ValueError("semantic_parts and active target count differ")
        if self.object_points_local.shape != (frames, count, 3):
            raise ValueError("object_points_local shape does not match active")
        if self.source_distance_m.shape != (frames, count):
            raise ValueError("source_distance_m shape does not match active")
        if self.weights.shape != (count,) or np.any(self.weights <= 0.0):
            raise ValueError("weights must contain one positive value per contact target")
        if self.activation is not None:
            if self.activation.shape != (frames, count):
                raise ValueError("activation shape does not match active")
            if not np.isfinite(self.activation).all() or np.any(self.activation < 0.0) or np.any(self.activation > 1.0):
                raise ValueError("activation must be finite and in [0, 1]")
            if not np.array_equal(self.active, self.activation > 0.0):
                raise ValueError("active must equal activation > 0")
        if not np.isfinite(self.object_points_local).all():
            raise ValueError("object_points_local contains NaN/Inf")
        if not np.isfinite(self.source_distance_m).all():
            raise ValueError("source_distance_m contains NaN/Inf")
        patch_fields = (self.robot_points_local, self.robot_normals_local, self.object_normals_local)
        if any(value is not None for value in patch_fields):
            if any(value is None for value in patch_fields):
                raise ValueError("anchor-force requires points and both normal arrays")
            for name, value, shape in (
                ("robot_points_local", self.robot_points_local, (count, 3)),
                ("robot_normals_local", self.robot_normals_local, (count, 3)),
                ("object_normals_local", self.object_normals_local, (frames, count, 3)),
            ):
                assert value is not None
                if value.shape != shape or not np.isfinite(value).all():
                    raise ValueError(f"invalid {name} shape/values")
                if "normals" in name and not np.allclose(np.linalg.norm(value, axis=-1), 1.0, atol=1e-5):
                    raise ValueError(f"{name} must contain unit normals")


def load_retarget_contact_targets(
    path: str | Path,
    *,
    expected_frames: int | None = None,
) -> RetargetContactTargets:
    source = Path(path)
    with np.load(source, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        if metadata.get("schema") not in SUPPORTED_SCHEMAS:
            raise ValueError(f"unsupported retarget contact schema: {metadata.get('schema')!r}")
        result = RetargetContactTargets(
            parts=tuple(str(value) for value in data["semantic_parts"].tolist()),
            active=np.asarray(data["active"], dtype=bool),
            object_points_local=np.asarray(data["object_points_local"], dtype=np.float64),
            weights=np.asarray(data["weights"], dtype=np.float64),
            source_distance_m=np.asarray(data["source_distance_m"], dtype=np.float64),
            metadata=metadata,
            activation=(
                np.asarray(data["activation"], dtype=np.float64)
                if "activation" in data
                else np.asarray(data["active"], dtype=np.float64)
            ),
            robot_points_local=np.asarray(data["robot_points_local"], dtype=np.float64)
            if "robot_points_local" in data
            else None,
            robot_normals_local=np.asarray(data["robot_normals_local"], dtype=np.float64)
            if "robot_normals_local" in data
            else None,
            object_normals_local=np.asarray(data["object_normals_local"], dtype=np.float64)
            if "object_normals_local" in data
            else None,
        )
    result.validate(expected_frames)
    return result
