#!/usr/bin/env python3
"""Transfer Semantic-B4 retarget contacts to the RL sphere-hand geometry.

The force/affordance object anchors and activation interval come directly from
the retarget contact artifact. This adapter never searches a retargeted robot
trajectory for contacts. Its only geometric operation is to locate the palm
support point on each half-sphere collision mesh. Isaac merges each fixed
sphere link into its wrist articulation body, so the adapter expresses that
same physical sphere point in the parent wrist frame for runtime indexing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh


SOURCE_SCHEMA = "holosoma.retarget_surface_contacts.v2"
OUTPUT_SCHEMA = "holosoma.reference_surface_contacts.v2"
PART_TO_SPHERE_LINK = {
    "left_hand": "left_sphere_hand_link",
    "right_hand": "right_sphere_hand_link",
}
PALM_DIRECTION = {
    "left_hand": np.array([0.0, -1.0, 0.0]),
    "right_hand": np.array([0.0, 1.0, 0.0]),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _origin(element: ET.Element | None) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    if element is None:
        return transform
    xyz = np.fromstring(element.get("xyz", "0 0 0"), sep=" ")
    rpy = np.fromstring(element.get("rpy", "0 0 0"), sep=" ")
    if xyz.shape != (3,) or rpy.shape != (3,):
        raise ValueError("URDF origin must contain three xyz and rpy values")
    transform[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    transform[:3, 3] = xyz
    return transform


def _mesh_path(urdf: Path, filename: str) -> Path:
    if filename.startswith("package://"):
        raise ValueError("package:// sphere meshes are unsupported; use a relative path")
    path = Path(filename)
    return (path if path.is_absolute() else urdf.parent / path).resolve()


def _sphere_adapter(
    urdf: Path,
    semantic_part: str,
    support_band_m: float,
) -> tuple[str, str, np.ndarray, Path, dict[str, object]]:
    sphere_link = PART_TO_SPHERE_LINK.get(semantic_part)
    if sphere_link is None:
        raise ValueError(f"No sphere-hand mapping for semantic part {semantic_part!r}")
    root = ET.parse(urdf).getroot()
    link = next((item for item in root.findall("link") if item.get("name") == sphere_link), None)
    if link is None:
        raise ValueError(f"Sphere link missing from training URDF: {sphere_link}")
    collision = link.find("collision")
    if collision is None:
        raise ValueError(f"Sphere link has no collision: {sphere_link}")
    mesh_element = collision.find("geometry/mesh")
    if mesh_element is None or not mesh_element.get("filename"):
        raise ValueError(f"Sphere collision is not a mesh: {sphere_link}")
    mesh_path = _mesh_path(urdf, mesh_element.get("filename"))
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Sphere collision did not load as one triangle mesh: {mesh_path}")
    scale = np.fromstring(mesh_element.get("scale", "1 1 1"), sep=" ")
    if scale.shape != (3,) or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError(f"Invalid sphere mesh scale on {sphere_link}")
    vertices = np.asarray(mesh.vertices, dtype=np.float64) * scale
    collision_to_link = _origin(collision.find("origin"))
    vertices = vertices @ collision_to_link[:3, :3].T + collision_to_link[:3, 3]
    direction = PALM_DIRECTION[semantic_part]
    projection = vertices @ direction
    support = float(projection.max())
    selected = np.flatnonzero(projection >= support - support_band_m)
    if not len(selected):
        raise ValueError(f"No support vertices found on {sphere_link}")
    point_sphere = vertices[selected].mean(axis=0)
    point_sphere += (support - float(point_sphere @ direction)) * direction

    joints = [
        joint
        for joint in root.findall("joint")
        if joint.find("child") is not None and joint.find("child").get("link") == sphere_link
    ]
    if len(joints) != 1 or joints[0].get("type") != "fixed":
        raise ValueError(f"Sphere link must have exactly one fixed parent joint: {sphere_link}")
    joint = joints[0]
    runtime_body = joint.find("parent").get("link")
    sphere_to_runtime = _origin(joint.find("origin"))
    point_runtime = sphere_to_runtime[:3, :3] @ point_sphere + sphere_to_runtime[:3, 3]

    audit = {
        "semantic_part": semantic_part,
        "sphere_link": sphere_link,
        "runtime_body": runtime_body,
        "sphere_mesh": str(mesh_path),
        "sphere_mesh_sha256": _sha256(mesh_path),
        "support_direction_sphere_local": direction.tolist(),
        "support_band_m": support_band_m,
        "support_vertex_count": int(len(selected)),
        "support_point_sphere_local_m": point_sphere.tolist(),
        "sphere_to_runtime_transform": sphere_to_runtime.tolist(),
        "support_point_runtime_local_m": point_runtime.tolist(),
    }
    return sphere_link, runtime_body, point_runtime, mesh_path, audit


def _resample(values: np.ndarray, source_fps: float, times: np.ndarray) -> np.ndarray:
    source_times = np.arange(len(values), dtype=np.float64) / source_fps
    flat = values.reshape(len(values), -1)
    result = np.column_stack([np.interp(times, source_times, flat[:, column]) for column in range(flat.shape[1])])
    return result.reshape((len(times),) + values.shape[1:])


def build(args: argparse.Namespace) -> dict[str, object]:
    source = args.retarget_targets.resolve()
    motion = args.motion.resolve()
    semantic = args.semantic_file.resolve()
    urdf = args.robot_urdf.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    for path in (source, motion, semantic, urdf):
        if not path.is_file():
            raise FileNotFoundError(path)

    with np.load(source, allow_pickle=False) as data:
        source_metadata = json.loads(str(data["metadata_json"].item()))
        if source_metadata.get("schema") != SOURCE_SCHEMA:
            raise ValueError(f"Expected {SOURCE_SCHEMA} source artifact")
        parts = data["semantic_parts"].tolist()
        object_points_source = np.asarray(data["object_points_local"], dtype=np.float64)
        object_normals_source = (
            np.asarray(data["object_normals_local"], dtype=np.float64)
            if "object_normals_local" in data.files else None
        )
        active_source = np.asarray(data["active"])
        activation_source = np.asarray(
            data["activation"] if "activation" in data.files else active_source,
            dtype=np.float64,
        )
    if parts != ["left_hand", "right_hand"]:
        raise ValueError(f"Expected ordered bilateral hand targets, got {parts}")
    source_frames = int(source_metadata.get("frames", -1))
    source_fps = float(source_metadata.get("fps", np.nan))
    expected_shape = (source_frames, len(parts))
    if (
        source_frames <= 1
        or not np.isfinite(source_fps)
        or source_fps <= 0
        or object_points_source.shape != expected_shape + (3,)
        or active_source.shape != expected_shape
        or active_source.dtype != np.bool_
        or activation_source.shape != expected_shape
        or not np.isfinite(object_points_source).all()
        or not np.isfinite(activation_source).all()
        or np.any((activation_source < 0) | (activation_source > 1))
        or not np.array_equal(active_source, activation_source > 0)
    ):
        raise ValueError("Invalid retarget contact arrays")
    if not source_metadata.get("no_retargeted_robot_trajectory_input", False):
        raise ValueError("Source artifact does not prove source-only contact generation")
    expected_semantic_hash = source_metadata.get("inputs", {}).get("semantic_plan_sha256")
    if not expected_semantic_hash or _sha256(semantic) != expected_semantic_hash:
        raise ValueError("Semantic plan differs from the retarget contact source")

    with np.load(motion, allow_pickle=False) as data:
        frames = len(data["joint_pos"])
        fps = float(np.asarray(data["fps"]).item())
    if frames <= 1 or not np.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid RL motion frame count or FPS")
    destination_times = np.arange(frames, dtype=np.float64) / fps
    source_last_time = (source_frames - 1) / source_fps
    if destination_times[-1] > source_last_time + 1e-9:
        raise ValueError(
            f"RL motion duration {destination_times[-1]:.6f}s exceeds contact source duration {source_last_time:.6f}s"
        )
    object_points = _resample(object_points_source, source_fps, destination_times)
    extra_arrays = {}
    if object_normals_source is not None:
        lengths = np.linalg.norm(object_normals_source, axis=-1)
        if (
            object_normals_source.shape != object_points_source.shape
            or not np.isfinite(object_normals_source).all()
            or not np.allclose(lengths, 1.0, atol=1e-4)
        ):
            raise ValueError("Source object normals must be finite unit vectors matching anchors")
        object_normals = _resample(object_normals_source, source_fps, destination_times)
        lengths = np.linalg.norm(object_normals, axis=-1, keepdims=True)
        if np.any(lengths < 1e-8):
            raise ValueError("Resampled object normals cannot be zero")
        extra_arrays["object_normals_local"] = (object_normals / lengths).astype(np.float32)
    activation = _resample(activation_source, source_fps, destination_times)
    activation = np.clip(activation, 0.0, 1.0)
    active = activation > 1e-8

    sphere_links = []
    runtime_bodies = []
    runtime_points = []
    sphere_meshes = []
    sphere_audit = []
    for part in parts:
        sphere_link, runtime_body, point, mesh_path, audit = _sphere_adapter(urdf, part, args.support_band_mm / 1000.0)
        sphere_links.append(sphere_link)
        runtime_bodies.append(runtime_body)
        runtime_points.append(point)
        sphere_meshes.append(mesh_path)
        sphere_audit.append(audit)
    body_points = np.broadcast_to(np.asarray(runtime_points, dtype=np.float64)[None], (frames, len(parts), 3)).copy()

    geometry_paths = {source, urdf, *sphere_meshes}
    metadata = {
        "schema": OUTPUT_SCHEMA,
        "method": "retarget force/affordance contact transfer with sphere-hand adaptation",
        "frames": frames,
        "fps": fps,
        "motion_sha256": _sha256(motion),
        "semantic_sha256": _sha256(semantic),
        "semantic_parts": parts,
        "source_retarget_targets": str(source),
        "source_retarget_targets_sha256": _sha256(source),
        "source_schema": SOURCE_SCHEMA,
        "source_frames": source_frames,
        "source_fps": source_fps,
        "time_resampling": "linear interpolation at physical timestamps",
        "contact_definition": "source object-local anchors and activation; no RL trajectory contact search",
        "hand_adaptation": "half-sphere directional support expressed in its merged fixed-joint wrist runtime body",
        "source_robot_points_ignored": True,
        "sphere_support": sphere_audit,
        "geometry_sha256": {str(path): _sha256(path) for path in sorted(geometry_paths, key=str)},
    }
    if extra_arrays:
        metadata["normal_definition"] = "source outward object surface normals; resampled and normalized"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp.npz")
    np.savez_compressed(
        temporary,
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
        semantic_parts=np.asarray(parts),
        source_geom_names=np.asarray(sphere_links),
        body_names=np.asarray(runtime_bodies),
        body_points_local=body_points.astype(np.float32),
        object_points_local=object_points.astype(np.float32),
        active=active.astype(np.bool_),
        activation=activation.astype(np.float32),
        **extra_arrays,
    )
    temporary.replace(output)
    metadata_path = output.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retarget-targets", type=Path, required=True)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--semantic-file", type=Path, required=True)
    parser.add_argument("--robot-urdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--support-band-mm", type=float, default=0.001)
    args = parser.parse_args()
    if not 0 <= args.support_band_mm <= 1.0:
        raise ValueError("support-band-mm must be in [0, 1]")
    print(json.dumps(build(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
