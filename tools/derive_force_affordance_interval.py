#!/usr/bin/env python3
"""Derive force-affordance timing from official source data.

The contact state starts at the first semantic event that declares both hands.
It remains active across semantic-event gaps and past the final action boundary
until a persistent source-hand departure is observed. The force anchor is the
highest object frame inside that derived interval. No frame-number override is
accepted by this tool.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/holosoma_retargeting"))

from holosoma_retargeting.config_types.data_type import SMPLH_DEMO_JOINTS  # noqa: E402
from holosoma_retargeting.semantic_keyframes.runtime import (  # noqa: E402
    load_semantic_plan_projection,
)
from holosoma_retargeting.src.utils import load_intermimic_data  # noqa: E402


SIDES = ("left", "right")
WRIST_NAMES = {"left": "L_Wrist", "right": "R_Wrist"}
REQUIRED_PARTS = {"left_hand", "right_hand"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _median3(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 3:
        return values.copy()
    padded = np.pad(values, (1, 1), mode="edge")
    return np.median(
        np.stack((padded[:-2], padded[1:-1], padded[2:]), axis=1),
        axis=1,
    )


def source_wrist_distances(
    human: np.ndarray,
    object_poses_wxyz_xyz: np.ndarray,
    object_mesh: trimesh.Trimesh,
) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    rotations = Rotation.from_quat(object_poses_wxyz_xyz[:, :4][:, [1, 2, 3, 0]]).as_matrix()
    object_position = object_poses_wxyz_xyz[:, 4:7]
    distances = np.empty((len(human), 2), dtype=np.float64)
    for index, side in enumerate(SIDES):
        wrist_id = SMPLH_DEMO_JOINTS.index(WRIST_NAMES[side])
        wrist_local = np.einsum(
            "tji,tj->ti",
            rotations,
            human[:, wrist_id] - object_position,
        )
        _, values, _ = trimesh.proximity.closest_point_naive(
            object_mesh,
            wrist_local,
        )
        distances[:, index] = values
    return distances


def derive_interval(
    *,
    source_distance_m: np.ndarray,
    object_height_m: np.ndarray,
    semantic_events: tuple[object, ...],
    hold_quantile: float = 0.95,
    release_margin_m: float = 0.005,
    release_persistence_frames: int = 3,
    contact_onset_threshold_m: float | None = None,
    contact_onset_persistence_frames: int = 3,
) -> dict[str, object]:
    distances = np.asarray(source_distance_m, dtype=np.float64)
    heights = np.asarray(object_height_m, dtype=np.float64)
    if distances.ndim != 2 or distances.shape[1] != 2:
        raise ValueError("source_distance_m must have shape [frames, 2]")
    if heights.shape != (len(distances),):
        raise ValueError("object_height_m must have one value per frame")
    if not np.isfinite(distances).all() or not np.isfinite(heights).all():
        raise ValueError("source timing signals must be finite")
    if not 0.5 <= hold_quantile <= 1.0:
        raise ValueError("hold_quantile must be in [0.5, 1.0]")
    if release_margin_m <= 0.0 or release_persistence_frames < 1:
        raise ValueError("release settings must be positive")
    if contact_onset_threshold_m is not None and contact_onset_threshold_m <= 0.0:
        raise ValueError("contact onset threshold must be positive")
    if contact_onset_persistence_frames < 1:
        raise ValueError("contact onset persistence must be positive")

    bilateral = [event for event in semantic_events if REQUIRED_PARTS.issubset(set(event.body_parts))]
    if not bilateral:
        raise ValueError("semantic plan has no event declaring both hands")
    semantic_start = min(int(event.start_frame) for event in bilateral)
    semantic_end = max(int(event.end_frame) for event in bilateral)
    if not 0 <= semantic_start <= semantic_end < len(distances):
        raise ValueError("semantic hand interval is outside the source trajectory")

    filtered_by_side = np.column_stack(tuple(_median3(distances[:, index]) for index in range(2)))
    bilateral_score = np.max(filtered_by_side, axis=1)
    contact_start = semantic_start
    contact_onset_run: list[int] = []
    if contact_onset_threshold_m is not None:
        inside = np.all(filtered_by_side <= contact_onset_threshold_m, axis=1)
        latest_start = semantic_end - contact_onset_persistence_frames + 1
        contact_start = -1
        for frame in range(semantic_start, latest_start + 1):
            if bool(np.all(inside[frame : frame + contact_onset_persistence_frames])):
                contact_start = int(frame)
                contact_onset_run = list(range(frame, frame + contact_onset_persistence_frames))
                break
        if contact_start < 0:
            raise ValueError("no persistent bilateral contact onset inside the semantic interval")

    hold_values = bilateral_score[contact_start : semantic_end + 1]
    reference = float(np.quantile(hold_values, hold_quantile))
    release_threshold = reference + float(release_margin_m)

    release_start = None
    search_start = semantic_end + 1
    departed = bilateral_score > release_threshold
    for frame in range(
        search_start,
        len(distances) - release_persistence_frames + 1,
    ):
        if bool(np.all(departed[frame : frame + release_persistence_frames])):
            release_start = int(frame)
            break
    end_frame = len(distances) - 1 if release_start is None else release_start - 1
    anchor_frame = contact_start + int(np.argmax(heights[contact_start : end_frame + 1]))
    if not contact_start <= anchor_frame <= end_frame:
        raise RuntimeError("derived anchor lies outside the contact interval")

    return {
        "schema": "holosoma.force_affordance_interval.v1",
        "policy": {
            "start": (
                "first persistent bilateral entry into the configured source-hand "
                "surface-distance neighborhood inside the semantic interval"
                if contact_onset_threshold_m is not None
                else "first frame of the earliest semantic event declaring both hands"
            ),
            "continuation": (
                "open contact state bridges semantic-event gaps and extends beyond the last semantic action boundary"
            ),
            "release": ("first persistent source departure above the semantic-hold bilateral wrist-distance envelope"),
            "anchor": "highest object-translation-z frame inside the derived interval",
            "manual_frame_overrides": False,
        },
        "start_frame": contact_start,
        "semantic_start_frame": semantic_start,
        "semantic_end_frame": semantic_end,
        "end_frame": int(end_frame),
        "release_start_frame": release_start,
        "anchor_frame": int(anchor_frame),
        "parameters": {
            "hold_quantile": float(hold_quantile),
            "release_margin_m": float(release_margin_m),
            "release_persistence_frames": int(release_persistence_frames),
            "contact_onset_threshold_m": (
                None if contact_onset_threshold_m is None else float(contact_onset_threshold_m)
            ),
            "contact_onset_persistence_frames": int(contact_onset_persistence_frames),
        },
        "evidence": {
            "bilateral_semantic_events": [
                {
                    "name": str(event.name),
                    "start_frame": int(event.start_frame),
                    "end_frame": int(event.end_frame),
                    "body_parts": list(event.body_parts),
                }
                for event in bilateral
            ],
            "semantic_hold_score_quantile_m": reference,
            "release_threshold_m": release_threshold,
            "release_run_frames": (
                []
                if release_start is None
                else list(
                    range(
                        release_start,
                        release_start + release_persistence_frames,
                    )
                )
            ),
            "contact_onset_run_frames": contact_onset_run,
            "filtered_wrist_distance_m": filtered_by_side.tolist(),
            "bilateral_score_m": bilateral_score.tolist(),
            "object_height_m": heights.tolist(),
        },
    }


def build(args: argparse.Namespace) -> dict[str, object]:
    official_input = args.official_input.resolve()
    semantic_plan = args.semantic_plan.resolve()
    object_mesh_path = args.object_mesh.resolve()
    output = args.output.resolve()
    if official_input.parent.name != "official_inputs":
        raise ValueError(
            "official input must come from omomo/official_inputs; retargeted trajectories are not accepted"
        )
    if output.exists():
        raise FileExistsError(output)

    human, object_poses = load_intermimic_data(str(official_input))
    events = load_semantic_plan_projection(semantic_plan)
    object_mesh = trimesh.load(object_mesh_path, force="mesh", process=False)
    if not isinstance(object_mesh, trimesh.Trimesh):
        raise ValueError(f"expected one object triangle mesh: {object_mesh_path}")
    distances = source_wrist_distances(human, object_poses, object_mesh)
    result = derive_interval(
        source_distance_m=distances,
        object_height_m=object_poses[:, 6],
        semantic_events=events,
        hold_quantile=args.hold_quantile,
        release_margin_m=args.release_margin_mm / 1000.0,
        release_persistence_frames=args.release_persistence_frames,
        contact_onset_threshold_m=(
            None if args.contact_onset_threshold_mm is None else args.contact_onset_threshold_mm / 1000.0
        ),
        contact_onset_persistence_frames=args.contact_onset_persistence_frames,
    )
    result["frames"] = int(len(human))
    result["inputs"] = {
        "official_smpl": str(official_input),
        "official_smpl_sha256": _sha256(official_input),
        "semantic_plan": str(semantic_plan),
        "semantic_plan_sha256": _sha256(semantic_plan),
        "object_mesh": str(object_mesh_path),
        "object_mesh_sha256": _sha256(object_mesh_path),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-input", type=Path, required=True)
    parser.add_argument("--semantic-plan", type=Path, required=True)
    parser.add_argument("--object-mesh", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hold-quantile", type=float, default=0.95)
    parser.add_argument("--release-margin-mm", type=float, default=5.0)
    parser.add_argument("--release-persistence-frames", type=int, default=3)
    parser.add_argument("--contact-onset-threshold-mm", type=float)
    parser.add_argument("--contact-onset-persistence-frames", type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(build(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
