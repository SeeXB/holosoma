"""Semantic-key-part surface contacts for retargeting.

Only body parts named by the semantic plan are considered.  By default, each
part's contact state is inferred independently from SMPL-H skin proximity.
The opt-in semantic_group mode instead treats the declared parts as contact
participants: proximity of any participant establishes the group's timing,
without vetoing another participant because of a reconstructed skin gap.
Projected anchors for inferred participants are not measured source contacts.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import igl  # type: ignore[import-not-found]
import numpy as np
import trimesh

from holosoma_retargeting.semantic_keyframes.runtime import load_semantic_plan_projection


SCHEMA = "holosoma.retarget_surface_contacts.v1"

# Native SMPL-H order used by the 52-column LBS weight matrix.  A semantic
# region is the set of vertices whose dominant skinning joint is in the group.
SMPLH_NATIVE_JOINT_GROUPS: dict[str, tuple[int, ...]] = {
    "pelvis": (0,),
    "waist": (3, 6, 9),
    "torso": (3, 6, 9),
    "left_hip": (1,),
    "right_hip": (2,),
    "left_knee": (4,),
    "right_knee": (5,),
    "left_ankle": (7,),
    "right_ankle": (8,),
    "left_foot": (7, 10),
    "right_foot": (8, 11),
    "left_shoulder": (13, 16),
    "right_shoulder": (14, 17),
    "left_elbow": (18,),
    "right_elbow": (19,),
    "left_wrist": (20,),
    "right_wrist": (21,),
    "left_hand": (20, *range(22, 37)),
    "right_hand": (21, *range(37, 52)),
}

ALIASES = {
    "left_forearm": "left_elbow",
    "right_forearm": "right_elbow",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hysteretic_contact_mask(
    distances_m: np.ndarray,
    onset_threshold_m: float = 0.02,
    hold_threshold_m: float = 0.03,
) -> np.ndarray:
    """Enter contact at ``onset`` and retain it while within ``hold``."""
    values = np.asarray(distances_m, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"distances_m must be one-dimensional, got {values.shape}")
    if not 0.0 < onset_threshold_m <= hold_threshold_m:
        raise ValueError("expected 0 < onset_threshold_m <= hold_threshold_m")
    active = np.zeros(len(values), dtype=bool)
    held = False
    for frame, distance in enumerate(values):
        held = bool(distance <= (hold_threshold_m if held else onset_threshold_m))
        active[frame] = held
    return active


def _inclusive_spans(mask: np.ndarray) -> list[tuple[int, int]]:
    indices = np.flatnonzero(mask)
    if not len(indices):
        return []
    starts = indices[np.r_[True, np.diff(indices) > 1]]
    ends = indices[np.r_[np.diff(indices) > 1, True]]
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def semantic_contact_mask(
    distances_m: np.ndarray,
    *,
    activation_mode: str = "per_part",
    onset_threshold_m: float = 0.02,
    hold_threshold_m: float = 0.03,
) -> np.ndarray:
    """Separate semantic contact participation from geometric timing evidence.

    semantic_group is for a declared simultaneous contact group, not arbitrary
    important parts.  The builder requires the same group in every event;
    action-dependent groups need their own temporal annotation first.
    """
    values = np.asarray(distances_m, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] == 0 or not np.isfinite(values).all():
        raise ValueError("expected finite distances_m [frames, nonempty parts]")
    if activation_mode == "semantic_group":
        timing = hysteretic_contact_mask(
            values.min(axis=1), onset_threshold_m, hold_threshold_m
        )
        return np.broadcast_to(timing[:, None], values.shape).copy()
    if activation_mode == "per_part":
        return np.column_stack([
            hysteretic_contact_mask(values[:, index], onset_threshold_m, hold_threshold_m)
            for index in range(values.shape[1])
        ])
    raise ValueError(f"unsupported activation_mode: {activation_mode!r}")


def _mesh_similarity(source_vertices: np.ndarray, target_mesh: trimesh.Trimesh) -> tuple[float, np.ndarray, np.ndarray, float]:
    """Fit target = scale * source @ rotation + translation."""
    source = np.asarray(source_vertices, dtype=np.float64)
    target = np.asarray(target_mesh.vertices, dtype=np.float64)
    if source.shape != target.shape:
        raise ValueError(
            "source and target object meshes must share vertex correspondence; "
            f"got {source.shape} and {target.shape}"
        )
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    centered_source = source - source_center
    centered_target = target - target_center
    left, singular_values, right = np.linalg.svd(centered_source.T @ centered_target)
    rotation = left @ right
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right
    scale = float(singular_values.sum() / np.sum(centered_source * centered_source))
    translation = target_center - scale * source_center @ rotation
    fitted = scale * source @ rotation + translation
    max_error = float(np.max(np.linalg.norm(fitted - target, axis=1), initial=0.0))
    return scale, rotation, translation, max_error


@dataclass(frozen=True)
class RetargetContactTargets:
    parts: tuple[str, ...]
    active: np.ndarray
    object_points_local: np.ndarray
    weights: np.ndarray
    source_distance_m: np.ndarray
    metadata: dict[str, Any]
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
        if not np.isfinite(self.object_points_local).all():
            raise ValueError("object_points_local contains NaN/Inf")
        if not np.isfinite(self.source_distance_m).all():
            raise ValueError("source_distance_m contains NaN/Inf")
        patch_fields = (self.robot_points_local, self.robot_normals_local, self.object_normals_local)
        if any(value is not None for value in patch_fields):
            if any(value is None for value in patch_fields):
                raise ValueError("palm patch requires points and both normal arrays")
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
        if metadata.get("schema") != SCHEMA:
            raise ValueError(f"unsupported retarget contact schema: {metadata.get('schema')!r}")
        result = RetargetContactTargets(
            parts=tuple(str(value) for value in data["semantic_parts"].tolist()),
            active=np.asarray(data["active"], dtype=bool),
            object_points_local=np.asarray(data["object_points_local"], dtype=np.float64),
            weights=np.asarray(data["weights"], dtype=np.float64),
            source_distance_m=np.asarray(data["source_distance_m"], dtype=np.float64),
            metadata=metadata,
            robot_points_local=np.asarray(data["robot_points_local"], dtype=np.float64) if "robot_points_local" in data else None,
            robot_normals_local=np.asarray(data["robot_normals_local"], dtype=np.float64) if "robot_normals_local" in data else None,
            object_normals_local=np.asarray(data["object_normals_local"], dtype=np.float64) if "object_normals_local" in data else None,
        )
    result.validate(expected_frames)
    return result


def contact_objective_schedule(
    active: np.ndarray,
    approach_frames: int,
    temporal_schedule: str = "onset_window",
    gaussian_sigma_frames: float = 10.0,
    gaussian_min_relative_weight: float = 1e-6,
    release_frames: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a contact-objective schedule without changing contact labels.

    ``onset_window`` is the legacy bounded pre-contact experiment.
    ``smooth_window`` uses a compact quintic smoothstep rise before onset and
    an optional smooth release after onset. ``gaussian`` keeps the term
    present with relative weight ``exp(-0.5*((t-x)/sigma)**2)`` on both sides
    of each onset ``x``. The complete measured ``active`` mask remains
    untouched for later evaluation.
    """
    mask = np.asarray(active, dtype=bool)
    if mask.ndim != 2 or mask.shape[0] < 1:
        raise ValueError("active must be a nonempty frames-by-parts mask")
    if not isinstance(approach_frames, (int, np.integer)) or approach_frames < 0:
        raise ValueError("approach_frames must be a non-negative integer")
    if temporal_schedule not in {"onset_window", "smooth_window", "gaussian"}:
        raise ValueError(
            "temporal_schedule must be onset_window, smooth_window or gaussian"
        )
    if not np.isfinite(gaussian_sigma_frames) or gaussian_sigma_frames <= 0.0:
        raise ValueError("gaussian_sigma_frames must be positive and finite")
    if not 0.0 <= gaussian_min_relative_weight < 1.0:
        raise ValueError("gaussian_min_relative_weight must be in [0, 1)")
    if not isinstance(release_frames, (int, np.integer)) or release_frames < 0:
        raise ValueError("release_frames must be a non-negative integer")
    previous = np.vstack((np.zeros((1, mask.shape[1]), dtype=bool), mask[:-1]))
    onsets = mask & ~previous
    objective = np.zeros_like(mask)
    target_frames = np.broadcast_to(
        np.arange(len(mask), dtype=np.int64)[:, None], mask.shape
    ).copy()
    progress = np.zeros(mask.shape, dtype=np.float64)
    if temporal_schedule == "smooth_window":
        for onset, part in np.argwhere(onsets):
            onset = int(onset)
            part = int(part)
            start = max(0, onset - approach_frames)
            if start == onset:
                rise_frames = np.asarray([onset], dtype=np.int64)
                rise = np.ones(1, dtype=np.float64)
            else:
                rise_frames = np.arange(start, onset + 1, dtype=np.int64)
                u = (rise_frames - start) / float(onset - start)
                rise = 6.0 * u**5 - 15.0 * u**4 + 10.0 * u**3
            objective[rise_frames, part] = True
            target_frames[rise_frames, part] = onset
            progress[rise_frames, part] = np.maximum(
                progress[rise_frames, part], rise
            )
            if release_frames:
                stop = min(len(mask) - 1, onset + release_frames)
                release_indices = np.arange(onset + 1, stop + 1, dtype=np.int64)
                u = (release_indices - onset) / float(release_frames)
                fall = 1.0 - (6.0 * u**5 - 15.0 * u**4 + 10.0 * u**3)
                stronger = fall > progress[release_indices, part]
                selected = release_indices[stronger]
                objective[selected, part] = True
                target_frames[selected, part] = onset
                progress[selected, part] = fall[stronger]
        return objective, target_frames, progress
    if temporal_schedule == "gaussian":
        frames = np.arange(len(mask), dtype=np.float64)
        for onset, part in np.argwhere(onsets):
            values = np.exp(
                -0.5 * ((frames - float(onset)) / gaussian_sigma_frames) ** 2
            )
            stronger = (
                (values >= gaussian_min_relative_weight)
                & (values > progress[:, part])
            )
            objective[stronger, part] = True
            target_frames[stronger, part] = int(onset)
            progress[stronger, part] = values[stronger]
        return objective, target_frames, progress
    objective[:] = onsets
    progress[:] = onsets
    if approach_frames == 0:
        return objective, target_frames, progress
    for onset, part in np.argwhere(onsets):
        start = max(0, int(onset) - approach_frames)
        denominator = int(onset) - start + 1
        for frame in range(start, int(onset)):
            if mask[frame, part]:
                continue
            u = (frame - start + 1) / denominator
            # Equal progress increments distribute the target pose path across
            # the short serial window.
            ramp = u
            if ramp > progress[frame, part]:
                objective[frame, part] = True
                target_frames[frame, part] = int(onset)
                progress[frame, part] = ramp
    return objective, target_frames, progress


def build_retarget_contact_targets(
    *,
    bundle: str | Path,
    body_model: str | Path,
    target_object_mesh: str | Path,
    semantic_plan: str | Path,
    output: str | Path,
    onset_threshold_m: float = 0.02,
    hold_threshold_m: float = 0.03,
    activation_mode: str = "per_part",
) -> RetargetContactTargets:
    """Extract contact spans and tracked object-local surface projections."""
    bundle = Path(bundle).resolve()
    body_model = Path(body_model).resolve()
    target_object_mesh = Path(target_object_mesh).resolve()
    semantic_plan = Path(semantic_plan).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)

    events = load_semantic_plan_projection(semantic_plan)
    parts = tuple(
        dict.fromkeys(
            ALIASES.get(part, part)
            for event in events
            for part in event.body_parts
        )
    )
    unknown = sorted(set(parts).difference(SMPLH_NATIVE_JOINT_GROUPS))
    if unknown:
        raise ValueError(f"unsupported SMPL-H semantic contact parts: {unknown}")
    if activation_mode == "semantic_group" and any(
        {ALIASES.get(part, part) for part in event.body_parts} != set(parts)
        for event in events
    ):
        raise ValueError("semantic_group requires the same simultaneous contact parts in every event")

    with np.load(bundle, allow_pickle=False) as data:
        human_vertices = np.asarray(data["human_vertices"], dtype=np.float64)
        object_vertices_local = np.asarray(data["object_vertices_local"], dtype=np.float64)
        object_faces = np.asarray(data["object_faces"], dtype=np.int32)
        object_rotation = np.asarray(data["object_rotation"], dtype=np.float64)
        object_translation = np.asarray(data["object_translation"], dtype=np.float64)
        object_scale = np.asarray(data["object_scale"], dtype=np.float64).reshape(-1)
        frame_ids = np.asarray(data["frame_ids"], dtype=np.int64)
        fps = float(np.asarray(data["fps"]).item())
    frames = len(human_vertices)
    if object_rotation.shape != (frames, 3, 3) or object_translation.shape != (frames, 3):
        raise ValueError("bundle object transforms are not frame-aligned")
    if object_scale.shape != (frames,) or frame_ids.shape != (frames,):
        raise ValueError("bundle object scale/frame_ids are not frame-aligned")

    with np.load(body_model, allow_pickle=True) as model:
        lbs_weights = np.asarray(model["weights"], dtype=np.float64)
    if lbs_weights.shape != (human_vertices.shape[1], 52):
        raise ValueError(f"expected SMPL-H weights [V,52], got {lbs_weights.shape}")
    dominant_joint = np.argmax(lbs_weights, axis=1)

    target_mesh = trimesh.load(target_object_mesh, force="mesh", process=False)
    if not isinstance(target_mesh, trimesh.Trimesh):
        raise ValueError(f"expected one target object mesh: {target_object_mesh}")
    if not np.array_equal(object_faces, np.asarray(target_mesh.faces, dtype=np.int32)):
        raise ValueError("source and target object meshes must share face correspondence")
    mesh_scale, mesh_rotation, mesh_translation, mesh_fit_max_error = _mesh_similarity(
        object_vertices_local,
        target_mesh,
    )
    if mesh_fit_max_error > 1e-5:
        raise ValueError(
            f"source-to-target object mesh is not an exact similarity (max {mesh_fit_max_error:.6g} m)"
        )

    active = np.zeros((frames, len(parts)), dtype=bool)
    distances = np.zeros((frames, len(parts)), dtype=np.float64)
    anchors = np.zeros((frames, len(parts), 3), dtype=np.float64)
    part_vertex_counts: dict[str, int] = {}
    span_metadata: dict[str, list[dict[str, Any]]] = {}
    surface_data = {}
    for part_index, part in enumerate(parts):
        vertex_indices = np.flatnonzero(
            np.isin(dominant_joint, SMPLH_NATIVE_JOINT_GROUPS[part])
        )
        if not len(vertex_indices):
            raise ValueError(f"semantic part {part!r} has no dominant SMPL-H surface vertices")
        part_vertex_counts[part] = int(len(vertex_indices))
        world = human_vertices[:, vertex_indices]
        unscaled_local = np.einsum(
            "tji,tkj->tki",
            object_rotation,
            world - object_translation[:, None, :],
        ) / object_scale[:, None, None]
        squared, _, closest = igl.point_mesh_squared_distance(
            unscaled_local.reshape(-1, 3),
            object_vertices_local,
            object_faces,
        )
        squared = np.asarray(squared, dtype=np.float64).reshape(frames, len(vertex_indices))
        closest = np.asarray(closest, dtype=np.float64).reshape(frames, len(vertex_indices), 3)
        per_vertex_distance_m = np.sqrt(squared) * object_scale[:, None]
        closest_vertex = np.argmin(squared, axis=1)
        distances[:, part_index] = per_vertex_distance_m[np.arange(frames), closest_vertex]
        surface_data[part] = (vertex_indices, per_vertex_distance_m, closest_vertex, closest)

    active = semantic_contact_mask(
        distances,
        activation_mode=activation_mode,
        onset_threshold_m=onset_threshold_m,
        hold_threshold_m=hold_threshold_m,
    )
    for part_index, part in enumerate(parts):
        vertex_indices, per_vertex_distance_m, closest_vertex, closest = surface_data[part]
        part_spans = []
        for start, end in _inclusive_spans(active[:, part_index]):
            selected_vertex = int(closest_vertex[start])
            selected_vertices = [selected_vertex]
            switches = []
            for frame in range(start + 1, end + 1):
                if per_vertex_distance_m[frame, selected_vertex] > hold_threshold_m:
                    candidates = np.flatnonzero(
                        per_vertex_distance_m[frame] <= hold_threshold_m
                    )
                    if not len(candidates):
                        if activation_mode != "semantic_group":
                            raise RuntimeError("active contact frame has no vertex inside hold threshold")
                        # Semantic participation is authoritative here.  Keep
                        # a nearby surface projection without claiming that
                        # the reconstructed skin actually contacted it.
                        minimum = per_vertex_distance_m[frame, closest_vertex[frame]]
                        candidates = np.flatnonzero(
                            per_vertex_distance_m[frame] <= minimum + 0.005
                        )
                    previous_anchor = (
                        mesh_scale * closest[frame - 1, selected_vertex] @ mesh_rotation
                        + mesh_translation
                    )
                    candidate_anchors = (
                        mesh_scale * closest[frame, candidates] @ mesh_rotation
                        + mesh_translation
                    )
                    replacement = int(
                        candidates[
                            np.argmin(
                                np.linalg.norm(candidate_anchors - previous_anchor, axis=1)
                            )
                        ]
                    )
                    switches.append(
                        {
                            "frame": frame,
                            "from_source_vertex": int(vertex_indices[selected_vertex]),
                            "to_source_vertex": int(vertex_indices[replacement]),
                        }
                    )
                    selected_vertex = replacement
                selected_vertices.append(selected_vertex)
            source_anchors = closest[
                np.arange(start, end + 1), np.asarray(selected_vertices, dtype=np.int64)
            ]
            target_anchors = (
                mesh_scale * source_anchors @ mesh_rotation + mesh_translation
            )
            anchors[start : end + 1, part_index] = target_anchors
            part_spans.append(
                {
                    "start_frame": start,
                    "end_frame": end,
                    "onset_distance_m": float(distances[start, part_index]),
                    "inferred_participation_frames": (
                        np.flatnonzero(distances[start : end + 1, part_index] > hold_threshold_m)
                        + start
                    ).tolist(),
                    "onset_object_anchor_local_m": target_anchors[0].tolist(),
                    "end_object_anchor_local_m": target_anchors[-1].tolist(),
                    "anchor_path_length_m": float(np.linalg.norm(np.diff(target_anchors, axis=0), axis=1).sum()),
                    "source_surface_vertex_at_onset": int(vertex_indices[selected_vertices[0]]),
                    "source_surface_vertex_switches": switches,
                }
            )
        span_metadata[part] = part_spans

    keep = active.any(axis=0)
    if not keep.any():
        raise ValueError(
            f"no contacts entered at {onset_threshold_m:.3f} m for semantic parts {list(parts)}"
        )
    kept_parts = tuple(part for part, selected in zip(parts, keep) if selected)
    active = active[:, keep]
    distances = distances[:, keep]
    anchors = anchors[:, keep]
    weights = np.ones(len(kept_parts), dtype=np.float64)
    metadata: dict[str, Any] = {
        "schema": SCHEMA,
        "fps": fps,
        "frames": frames,
        "frame_ids": frame_ids.tolist(),
        "semantic_parts_considered": list(parts),
        "semantic_parts_with_contact": list(kept_parts),
        "selection": "only semantic-plan body_parts; dominant SMPL-H LBS surface regions",
        "activation_mode": activation_mode,
        "activation_evidence": (
            "semantic plan declares simultaneous contact participants; minimum source skin distance across the group supplies timing only"
            if activation_mode == "semantic_group" else
            "each semantic part independently passes source skin distance hysteresis"
        ),
        "semantic_keyframe_intervals_changed": False,
        "onset_threshold_m": onset_threshold_m,
        "hold_threshold_m": hold_threshold_m,
        "anchor_policy": (
            "track a source surface vertex; if source contact is missing, use its nearby closest object-surface projection (not observed contact)"
            if activation_mode == "semantic_group" else
            "track one source SMPL-H surface vertex per span; switch only after it exceeds the hold threshold"
        ),
        "part_vertex_counts": part_vertex_counts,
        "active_frames_per_part": {
            part: int(active[:, index].sum()) for index, part in enumerate(kept_parts)
        },
        "active_spans": {part: span_metadata[part] for part in kept_parts},
        "source_to_target_object_similarity": {
            "scale": mesh_scale,
            "rotation": mesh_rotation.tolist(),
            "translation": mesh_translation.tolist(),
            "max_vertex_error_m": mesh_fit_max_error,
        },
        "inputs": {
            "bundle": str(bundle),
            "bundle_sha256": _sha256(bundle),
            "body_model": str(body_model),
            "body_model_sha256": _sha256(body_model),
            "target_object_mesh": str(target_object_mesh),
            "target_object_mesh_sha256": _sha256(target_object_mesh),
            "semantic_plan": str(semantic_plan),
            "semantic_plan_sha256": _sha256(semantic_plan),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        semantic_parts=np.asarray(kept_parts),
        active=active,
        object_points_local=anchors.astype(np.float32),
        weights=weights.astype(np.float32),
        source_distance_m=distances.astype(np.float32),
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    result = load_retarget_contact_targets(output, expected_frames=frames)
    return result
