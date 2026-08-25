#!/usr/bin/env python3
"""Export CARI4D's native inference result to the canonical HOI representation."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np

from hoi_pipeline.canonical_validation import validate_and_write
from hoi_pipeline.common import (
    copy_obj_with_materials,
    count_obj_faces,
    find_best_scale,
    matrix_to_quaternion_wxyz,
    obj_extent,
    parse_cari4d_frame_ids,
    read_obj_vertices,
    relative_extent_error,
    transform_obj_vertices,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Adapt the real CARI4D pr.pose_abs/pr.smpl_pose/pr.smpl_t/pr.betas/pr.frames serialization "
            "to a stable, metric canonical sequence."
        )
    )
    parser.add_argument("--cari4d-result", type=Path, help="Final CARI4D optimization .pth result.")
    parser.add_argument("--native-manifest", type=Path, help="Manifest emitted by run_video_to_hoi.sh.")
    parser.add_argument("--object-mesh-normalized", type=Path, help="CARI4D input normalized object OBJ.")
    parser.add_argument("--object-mesh-metric", type=Path, help="CARI4D metric OBJ (_align.obj), already scaled.")
    parser.add_argument("--scale-json", type=Path, help="CARI4D scale-estimation JSON containing best_scale.")
    parser.add_argument(
        "--object-mesh-is-metric",
        action="store_true",
        help=(
            "The supplied known object asset is already in meters. This explicitly bypasses CARI4D scale "
            "estimation and prevents double scaling; --object-mesh-metric is required."
        ),
    )
    parser.add_argument(
        "--derive-scale-from-metric-mesh",
        action="store_true",
        help=(
            "Explicit archive-recovery mode when CARI4D distributed both corresponding normalized/metric meshes "
            "but omitted its scale JSON. Requires exact V_metric ~= s * V_normalized correspondence. Production "
            "custom-video exports should use --scale-json."
        ),
    )
    parser.add_argument("--output", required=True, type=Path, help="Canonical output sequence directory.")
    parser.add_argument("--video", type=Path, help="Source video; used to read exact FPS and frame count.")
    parser.add_argument("--fps", type=float, help="Explicit FPS when the source video is unavailable.")
    parser.add_argument(
        "--cari4d-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "third_party" / "CARI4D",
        help="CARI4D checkout used to obtain its exact 72-to-156 pose convention.",
    )
    parser.add_argument("--sequence-name", help="Optional display name; defaults to the result stem.")
    parser.add_argument(
        "--gender",
        choices=("male", "female"),
        help="SMPL-H gender. Read from a wrapper manifest when available; otherwise pass explicitly.",
    )
    parser.add_argument(
        "--upstream-provenance-json",
        type=Path,
        action="append",
        default=[],
        help=(
            "Optional preprocessing provenance JSON. May be repeated. The records are copied into sequence "
            "metadata so video-derived results cannot be confused with external motion/pose labels."
        ),
    )
    parser.add_argument(
        "--scale-relative-tolerance",
        type=float,
        default=1e-3,
        help="Maximum per-axis relative extent error accepted in the double-scale check.",
    )
    return parser.parse_args()


def _absolute(path: Path | None) -> Path | None:
    return path.expanduser().resolve() if path is not None else None


def _load_native_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    required = {"cari4d_result", "object_mesh_metric"}
    if not manifest.get("object_mesh_is_metric", False):
        required.update({"object_mesh_normalized", "scale_json"})
    missing = sorted(required - set(manifest))
    if missing:
        raise KeyError(f"Native CARI4D manifest is missing fields: {missing}")
    return manifest


def _resolve_inputs(args: argparse.Namespace) -> None:
    if args.object_mesh_is_metric and args.derive_scale_from_metric_mesh:
        raise ValueError("Use either --object-mesh-is-metric or --derive-scale-from-metric-mesh, not both")
    if args.object_mesh_is_metric and args.scale_json is not None:
        raise ValueError("--object-mesh-is-metric bypasses scale estimation; do not pass --scale-json")
    if args.derive_scale_from_metric_mesh and args.scale_json is not None:
        raise ValueError("Use either --scale-json or --derive-scale-from-metric-mesh, not both")
    if args.derive_scale_from_metric_mesh and args.native_manifest is not None:
        raise ValueError("--derive-scale-from-metric-mesh is only valid with explicit artifact paths, not a manifest")
    if args.native_manifest is not None:
        manifest_path = _absolute(args.native_manifest)
        manifest = _load_native_manifest(manifest_path)
        base = manifest_path.parent

        def resolve_manifest(key: str) -> Path:
            candidate = Path(manifest[key]).expanduser()
            return (base / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()

        args.cari4d_result = args.cari4d_result or resolve_manifest("cari4d_result")
        if args.object_mesh_normalized is None and manifest.get("object_mesh_normalized"):
            args.object_mesh_normalized = resolve_manifest("object_mesh_normalized")
        args.object_mesh_metric = args.object_mesh_metric or resolve_manifest("object_mesh_metric")
        args.object_mesh_is_metric = bool(args.object_mesh_is_metric or manifest.get("object_mesh_is_metric", False))
        if args.scale_json is None and manifest.get("scale_json"):
            args.scale_json = resolve_manifest("scale_json")
        if args.video is None and manifest.get("video"):
            args.video = resolve_manifest("video")
        if args.gender is None and manifest.get("gender"):
            args.gender = str(manifest["gender"])
        for raw_path in manifest.get("upstream_provenance_json", []):
            candidate = Path(raw_path).expanduser()
            resolved = (base / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
            if resolved not in args.upstream_provenance_json:
                args.upstream_provenance_json.append(resolved)
    if args.object_mesh_is_metric and args.scale_json is not None:
        raise ValueError("Known metric manifests must not provide scale_json; that would make scale provenance ambiguous")
    required: dict[str, Path | None] = {"--cari4d-result": args.cari4d_result}
    if args.object_mesh_is_metric:
        required["--object-mesh-metric"] = args.object_mesh_metric
    elif args.derive_scale_from_metric_mesh:
        required["--object-mesh-normalized"] = args.object_mesh_normalized
        required["--object-mesh-metric"] = args.object_mesh_metric
    else:
        required["--object-mesh-normalized"] = args.object_mesh_normalized
        required["--scale-json"] = args.scale_json
    missing = [flag for flag, value in required.items() if value is None]
    if missing:
        raise ValueError(
            "Missing explicit CARI4D artifacts: " + ", ".join(missing) + ". "
            "The final .pth does not serialize mesh/scale paths, so inferring them from filenames would be unsafe."
        )
    for name in ("cari4d_result", "object_mesh_normalized", "object_mesh_metric", "scale_json", "video"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, _absolute(value))
    args.upstream_provenance_json = [_absolute(path) for path in args.upstream_provenance_json]
    required_files = ["cari4d_result"]
    if args.object_mesh_is_metric:
        required_files.append("object_mesh_metric")
    else:
        required_files.append("object_mesh_normalized")
        required_files.append("object_mesh_metric" if args.derive_scale_from_metric_mesh else "scale_json")
    for name in required_files:
        value = getattr(args, name)
        if not value.is_file():
            raise FileNotFoundError(f"Input does not exist: {value}")
    if args.object_mesh_metric is not None and not args.object_mesh_metric.is_file():
        raise FileNotFoundError(f"Input does not exist: {args.object_mesh_metric}")
    if args.video is not None and not args.video.is_file():
        raise FileNotFoundError(f"Input video does not exist: {args.video}")
    for path in args.upstream_provenance_json:
        if not path.is_file():
            raise FileNotFoundError(f"Upstream provenance JSON does not exist: {path}")


def _load_torch_result(path: Path, cari4d_root: Path) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required only for reading CARI4D's native .pth result") from exc
    # Final opt_refineout checkpoints serialize a TrainState instance under pr.train_state.
    # Unpickling therefore needs the exact selected CARI4D checkout on sys.path even though the
    # canonical adapter consumes only numeric pr fields. This is source-derived, not a field guess.
    root_text = str(cari4d_root)
    inserted = root_text not in sys.path
    if inserted:
        sys.path.insert(0, root_text)
    try:
        try:
            result = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            result = torch.load(path, map_location="cpu")
    finally:
        if inserted:
            sys.path.remove(root_text)
    if not isinstance(result, dict) or "pr" not in result or not isinstance(result["pr"], dict):
        raise ValueError(f"{path} is not CARI4D's actual top-level {{'gt','pr','in'}} result format")
    prediction = result["pr"]
    required = {"pose_abs", "smpl_pose", "smpl_t", "betas", "frames"}
    missing = sorted(required - set(prediction))
    if missing:
        raise KeyError(f"CARI4D result pr dict is missing fields: {missing}")
    return prediction


def _numpy(value: Any, *, name: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    result = np.asarray(value)
    if result.dtype == object:
        raise TypeError(f"CARI4D field {name} has unsupported object dtype")
    return result


def _load_grab_hand_prior(cari4d_root: Path) -> np.ndarray:
    prior_file = cari4d_root / "lib_smpl" / "th_hand_prior.py"
    if not prior_file.is_file():
        raise FileNotFoundError(f"Cannot expand CARI4D 72D pose: missing official hand prior {prior_file}")
    package_name = "_cari4d_lib_smpl_adapter"
    package = types.ModuleType(package_name)
    package.__path__ = [str(prior_file.parent)]
    sys.modules[package_name] = package
    spec = importlib.util.spec_from_file_location(f"{package_name}.th_hand_prior", prior_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load CARI4D hand prior module: {prior_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        for module_name in tuple(sys.modules):
            if module_name == package_name or module_name.startswith(package_name + "."):
                sys.modules.pop(module_name, None)
    prior = np.asarray(module.GRAB_MEAN_HAND, dtype=np.float32)
    if prior.shape != (90,):
        raise ValueError(f"Unexpected CARI4D GRAB_MEAN_HAND shape {prior.shape}; expected (90,)")
    return prior


def _standardize_smplh_pose(raw_pose: np.ndarray, cari4d_root: Path) -> tuple[np.ndarray, str]:
    if raw_pose.ndim != 2 or raw_pose.shape[1] not in (72, 156):
        raise ValueError(f"CARI4D pr.smpl_pose must have shape [T,72] or [T,156], got {raw_pose.shape}")
    if raw_pose.shape[1] == 156:
        return raw_pose.astype(np.float32), "unchanged_cari4d_156d"
    # Exact mapping implemented by CARI4D lib_smpl.pose72to156. The 90D hand prior is loaded
    # from the selected CARI4D checkout instead of embedding a possibly stale copy here.
    poses = np.zeros((len(raw_pose), 156), dtype=np.float32)
    poses[:, 66:] = _load_grab_hand_prior(cari4d_root)
    poses[:, :69] = raw_pose[:, :69]
    poses[:, 114:117] = raw_pose[:, 69:72]
    return poses, "cari4d_pose72to156_with_grab_mean_hand_prior"


def _read_video_info(path: Path) -> tuple[float, int]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required to read FPS/frame count from --video; otherwise pass --fps") from exc
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open source video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    capture.release()
    if not np.isfinite(fps) or fps <= 0.0 or frame_count <= 0:
        raise ValueError(f"Invalid video metadata for {path}: fps={fps}, frames={frame_count}")
    return fps, frame_count


def _fps_and_video_frames(args: argparse.Namespace) -> tuple[float, int | None]:
    if args.video is not None:
        video_fps, frame_count = _read_video_info(args.video)
        if args.fps is not None and not np.isclose(args.fps, video_fps, rtol=1e-5, atol=1e-5):
            raise ValueError(f"--fps={args.fps} disagrees with video FPS {video_fps}")
        return video_fps, frame_count
    if args.fps is None or not np.isfinite(args.fps) or args.fps <= 0:
        raise ValueError(
            "CARI4D does not serialize FPS. Pass --video or an explicit positive --fps; no default is assumed."
        )
    return float(args.fps), None


def _export_metric_mesh(args: argparse.Namespace, object_dir: Path) -> dict[str, Any]:
    normalized_mesh = args.object_mesh_normalized
    source_metric_mesh = args.object_mesh_metric
    if args.object_mesh_is_metric:
        metric_vertices = read_obj_vertices(source_metric_mesh)
        metric_extent = metric_vertices.max(axis=0) - metric_vertices.min(axis=0)
        center = metric_vertices.mean(axis=0)
        original_output = object_dir / "object_original.obj"
        metric_output = object_dir / "object_metric.obj"
        copy_obj_with_materials(source_metric_mesh, original_output)
        sanitized_faces = transform_obj_vertices(
            source_metric_mesh,
            metric_output,
            offset=-center,
            sanitize_invalid_face_normals=True,
        )
        _, _, canonical_extent = obj_extent(metric_output)
        print(f"Known metric extent:    {metric_extent.tolist()} m")
        print("Estimated scale:        BYPASSED (known metric asset)")
        print(f"Canonical metric extent: {canonical_extent.tolist()} m")
        print("Double-scale check:     PASS (no scale operation performed)")
        return {
            "mesh_unit": "meter",
            "scale_baked_into_mesh": True,
            "cari4d_metric_scale": 1.0,
            "cari4d_scale_is_scalar": True,
            "cari4d_scale_provenance": "known_metric_asset_no_scale_estimation",
            "derived_scale_max_vertex_error_m": None,
            "source_mesh": str(source_metric_mesh),
            "source_metric_mesh": str(source_metric_mesh),
            "metric_mesh": str(metric_output.resolve()),
            "source_metric_mesh_already_scaled": True,
            "scale_applied_by_exporter": False,
            "double_scale_check": "PASS",
            "double_scale_relative_extent_error_xyz": np.zeros(3),
            "double_scale_max_vertex_error_m": 0.0,
            "original_extent_xyz_normalized": None,
            "raw_metric_extent_xyz_m": metric_extent,
            "extent_xyz_m": canonical_extent,
            "canonical_object_frame": "cari4d_centered_metric_mesh",
            "canonicalization": "vertices_metric - arithmetic_mean(vertices_metric)",
            "canonical_center_in_source_metric_mesh_m": center,
            "T_canonical_from_source_metric_mesh": np.block(
                [[np.eye(3), -center[:, None]], [np.zeros((1, 3)), np.ones((1, 1))]]
            ),
            "pose_semantics": "T_cari4d_camera_from_object_canonical",
            "num_vertices": len(metric_vertices),
            "num_faces": count_obj_faces(metric_output),
            "obj_invalid_zero_normal_face_indices_repaired": sanitized_faces,
            "obj_repair_semantics": "drop invalid face-normal references only; preserve vertices, UVs, and faces",
            "source_scale_json": None,
        }
    normalized_vertices = read_obj_vertices(normalized_mesh)
    derived_scale_max_vertex_error = None
    if args.derive_scale_from_metric_mesh:
        metric_vertices_for_scale = read_obj_vertices(source_metric_mesh)
        if len(metric_vertices_for_scale) != len(normalized_vertices):
            raise ValueError(
                "Cannot recover CARI4D archive scale: normalized and metric meshes have different vertex counts "
                f"({len(normalized_vertices)} vs {len(metric_vertices_for_scale)})"
            )
        denominator = float(np.sum(normalized_vertices * normalized_vertices))
        if denominator <= np.finfo(np.float64).eps:
            raise ValueError("Cannot recover scale from a degenerate normalized object mesh")
        scalar_scale = float(np.sum(normalized_vertices * metric_vertices_for_scale) / denominator)
        if not np.isfinite(scalar_scale) or scalar_scale <= 0.0:
            raise ValueError(f"Recovered object scale must be finite and positive, got {scalar_scale}")
        expected_vertices = normalized_vertices * scalar_scale
        derived_scale_max_vertex_error = float(np.max(np.abs(metric_vertices_for_scale - expected_vertices)))
        derived_error_limit = max(
            1e-6,
            args.scale_relative_tolerance * float(np.max(np.abs(expected_vertices))),
        )
        if derived_scale_max_vertex_error > derived_error_limit:
            raise ValueError(
                "Cannot recover one uniform CARI4D scale from the mesh pair: "
                f"max |V_metric - s*V_normalized|={derived_scale_max_vertex_error} m, "
                f"allowed={derived_error_limit} m"
            )
        scale = np.repeat(np.asarray(scalar_scale, dtype=np.float64), 3)
        scale_provenance = "derived_from_explicit_corresponding_cari4d_normalized_and_metric_meshes"
    else:
        scale_data = json.loads(args.scale_json.read_text(encoding="utf-8"))
        scale = find_best_scale(scale_data)
        scale_provenance = "cari4d_scale_json.best_scale"
    original_extent = normalized_vertices.max(axis=0) - normalized_vertices.min(axis=0)
    expected_metric_extent = original_extent * scale

    original_output = object_dir / "object_original.obj"
    metric_output = object_dir / "object_metric.obj"
    copy_obj_with_materials(normalized_mesh, original_output)

    if source_metric_mesh is not None:
        metric_vertices = read_obj_vertices(source_metric_mesh)
        metric_extent = metric_vertices.max(axis=0) - metric_vertices.min(axis=0)
        extent_error = relative_extent_error(metric_extent, expected_metric_extent)
        if len(metric_vertices) != len(normalized_vertices):
            raise ValueError(
                "Normalized and metric CARI4D meshes have different vertex counts; cannot prove scale provenance "
                f"({len(normalized_vertices)} vs {len(metric_vertices)})"
            )
        expected_metric_vertices = normalized_vertices * scale
        vertex_scale_error = np.abs(metric_vertices - expected_metric_vertices)
        vertex_error_limit = max(
            1e-6,
            args.scale_relative_tolerance * float(np.max(np.abs(expected_metric_vertices))),
        )
        if float(vertex_scale_error.max()) > vertex_error_limit:
            raise ValueError(
                "CARI4D metric mesh vertices are not normalized vertices * best_scale. "
                f"Max vertex error={vertex_scale_error.max()} m, allowed={vertex_error_limit} m. "
                "Refusing a rotated, translated, or double-scaled mesh."
            )
        if float(extent_error.max()) > args.scale_relative_tolerance:
            raise ValueError(
                "CARI4D metric mesh does not equal normalized extent * best_scale. Refusing a possible "
                f"double/wrong scale: original={original_extent}, scale={scale}, metric={metric_extent}, "
                f"relative_error={extent_error}"
            )
        metric_source = source_metric_mesh
        scale_applied_by_exporter = False
        source_metric_already_scaled = True
        center = metric_vertices.mean(axis=0)
        sanitized_faces = transform_obj_vertices(
            metric_source, metric_output, offset=-center, sanitize_invalid_face_normals=True
        )
    else:
        metric_vertices = normalized_vertices * scale
        vertex_scale_error = np.zeros_like(metric_vertices)
        metric_extent = metric_vertices.max(axis=0) - metric_vertices.min(axis=0)
        extent_error = relative_extent_error(metric_extent, expected_metric_extent)
        metric_source = normalized_mesh
        scale_applied_by_exporter = True
        source_metric_already_scaled = False
        center = metric_vertices.mean(axis=0)
        sanitized_faces = transform_obj_vertices(
            metric_source,
            metric_output,
            scale=scale,
            offset=-center,
            sanitize_invalid_face_normals=True,
        )

    _, _, canonical_extent = obj_extent(metric_output)
    print(f"Original extent:       {original_extent.tolist()} (normalized object units)")
    print(f"Estimated scale:       {scale.tolist()} m / normalized unit")
    print(f"Raw metric extent:     {metric_extent.tolist()} m")
    print(f"Canonical metric extent: {canonical_extent.tolist()} m")
    print("Double-scale check:    PASS (metric extent == original extent * best_scale)")

    return {
        "mesh_unit": "meter",
        "scale_baked_into_mesh": True,
        "cari4d_metric_scale": float(scale[0]) if np.allclose(scale, scale[0]) else scale,
        "cari4d_scale_is_scalar": bool(np.allclose(scale, scale[0])),
        "cari4d_scale_provenance": scale_provenance,
        "derived_scale_max_vertex_error_m": derived_scale_max_vertex_error,
        "source_mesh": str(normalized_mesh),
        "source_metric_mesh": str(source_metric_mesh) if source_metric_mesh is not None else None,
        "metric_mesh": str(metric_output.resolve()),
        "source_metric_mesh_already_scaled": source_metric_already_scaled,
        "scale_applied_by_exporter": scale_applied_by_exporter,
        "double_scale_check": "PASS",
        "double_scale_relative_extent_error_xyz": extent_error,
        "double_scale_max_vertex_error_m": float(vertex_scale_error.max()),
        "original_extent_xyz_normalized": original_extent,
        "raw_metric_extent_xyz_m": metric_extent,
        "extent_xyz_m": canonical_extent,
        "canonical_object_frame": "cari4d_centered_metric_mesh",
        "canonicalization": "vertices_metric - arithmetic_mean(vertices_metric)",
        "canonical_center_in_source_metric_mesh_m": center,
        "T_canonical_from_source_metric_mesh": np.block(
            [[np.eye(3), -center[:, None]], [np.zeros((1, 3)), np.ones((1, 1))]]
        ),
        "pose_semantics": "T_cari4d_camera_from_object_canonical",
        "num_vertices": len(metric_vertices),
        "num_faces": count_obj_faces(metric_output),
        "obj_invalid_zero_normal_face_indices_repaired": sanitized_faces,
        "obj_repair_semantics": "drop invalid face-normal references only; preserve vertices, UVs, and faces",
        "source_scale_json": str(args.scale_json) if args.scale_json is not None else None,
    }


def main() -> None:
    args = parse_args()
    if not np.isfinite(args.scale_relative_tolerance) or args.scale_relative_tolerance <= 0.0:
        raise ValueError(f"--scale-relative-tolerance must be finite and positive, got {args.scale_relative_tolerance}")
    _resolve_inputs(args)
    output = args.output.expanduser().resolve()
    human_dir = output / "human"
    object_dir = output / "object"
    meta_dir = output / "meta"
    for directory in (human_dir, object_dir, meta_dir, output / "validation"):
        directory.mkdir(parents=True, exist_ok=True)

    fps, video_frame_count = _fps_and_video_frames(args)
    upstream_provenance = []
    for provenance_path in args.upstream_provenance_json:
        record = json.loads(provenance_path.read_text(encoding="utf-8"))
        if not isinstance(record, dict):
            raise TypeError(f"Upstream provenance must contain a JSON object: {provenance_path}")
        upstream_provenance.append({"file": str(provenance_path), "record": record})
    prediction = _load_torch_result(args.cari4d_result, args.cari4d_root.expanduser().resolve())
    raw_pose = _numpy(prediction["smpl_pose"], name="smpl_pose")
    human_trans = _numpy(prediction["smpl_t"], name="smpl_t").astype(np.float32)
    betas = _numpy(prediction["betas"], name="betas").astype(np.float32)
    object_transform = _numpy(prediction["pose_abs"], name="pose_abs").astype(np.float64)
    frame_ids, source_frame_labels = parse_cari4d_frame_ids(list(prediction["frames"]))
    poses, pose_conversion = _standardize_smplh_pose(raw_pose, args.cari4d_root.expanduser().resolve())

    frame_count = len(frame_ids)
    if frame_count == 0:
        raise ValueError("CARI4D result is empty")
    expected_leading = {
        "pr.smpl_pose": len(raw_pose), "pr.smpl_t": len(human_trans),
        "pr.pose_abs": len(object_transform), "pr.frames": frame_count,
    }
    if betas.ndim == 2:
        expected_leading["pr.betas"] = len(betas)
    if len(set(expected_leading.values())) != 1:
        raise ValueError(f"CARI4D result fields have mismatched frame counts: {expected_leading}; refusing to truncate")
    if human_trans.shape != (frame_count, 3):
        raise ValueError(f"CARI4D pr.smpl_t must have shape [T,3], got {human_trans.shape}")
    if betas.shape not in ((frame_count, 10), (10,)):
        raise ValueError(f"CARI4D pr.betas must have shape [T,10] or [10], got {betas.shape}")
    if object_transform.shape != (frame_count, 4, 4):
        raise ValueError(f"CARI4D pr.pose_abs must have shape [T,4,4], got {object_transform.shape}")
    expected_bottom = np.broadcast_to(np.array([0.0, 0.0, 0.0, 1.0]), object_transform[:, 3].shape)
    if not np.allclose(object_transform[:, 3], expected_bottom, atol=1e-6):
        raise ValueError("CARI4D pr.pose_abs contains non-rigid homogeneous bottom rows")

    rotations = object_transform[:, :3, :3].astype(np.float32)
    orthogonality_error = np.linalg.norm(np.swapaxes(rotations, 1, 2) @ rotations - np.eye(3)[None], axis=(1, 2))
    determinant_error = np.abs(np.linalg.det(rotations) - 1.0)
    if float(orthogonality_error.max()) > 1e-4 or float(determinant_error.max()) > 1e-4:
        raise ValueError(
            "CARI4D pr.pose_abs contains invalid rotations: "
            f"max orthogonality error={orthogonality_error.max()}, max determinant error={determinant_error.max()}"
        )
    translations = object_transform[:, :3, 3].astype(np.float32)
    quaternions = matrix_to_quaternion_wxyz(rotations)
    object_transform = object_transform.astype(np.float32)

    np.savez_compressed(
        human_dir / "smplh.npz",
        poses=poses,
        trans=human_trans,
        betas=betas,
        fps=np.asarray(fps, dtype=np.float32),
        frame_ids=frame_ids,
        poses_cari4d_raw=raw_pose.astype(np.float32),
    )
    np.savez_compressed(
        object_dir / "trajectory.npz",
        translation=translations,
        rotation_matrix=rotations,
        quaternion_wxyz=quaternions,
        transform=object_transform,
        fps=np.asarray(fps, dtype=np.float32),
        frame_ids=frame_ids,
    )
    np.save(object_dir / "initial_pose.npy", object_transform[0])
    object_metadata = _export_metric_mesh(args, object_dir)

    coordinate_metadata = {
        "coordinate_frame": "cari4d_metric_camera",
        "coordinate_system": "OpenCV camera: +X right, +Y down, +Z forward",
        "axis_convention": "opencv_camera_x_right_y_down_z_forward",
        "coordinate_evidence": "CARI4D Utils.depth2xyzmap and pinhole projection code",
        "T_isaac_from_cari4d": None,
        "isaac_frame_conversion_applied": False,
    }
    human_metadata = {
        "body_model": "SMPL-H",
        "gender": args.gender,
        "pose_representation": "axis_angle_radians_52_joints_flat_156",
        "raw_pose_shape": list(raw_pose.shape),
        "raw_pose_preserved_as": "poses_cari4d_raw",
        "pose_conversion": pose_conversion,
        "num_frames": frame_count,
        "fps": fps,
        "translation_unit": "meter",
        "source_result_file": str(args.cari4d_result),
        "source_result_fields": ["pr.smpl_pose", "pr.smpl_t", "pr.betas", "pr.frames"],
        "source_frame_labels": source_frame_labels,
        **coordinate_metadata,
    }
    object_metadata.update(
        {
            "num_frames": frame_count,
            "fps": fps,
            "translation_unit": "meter",
            "rotation_representation": ["rotation_matrix", "quaternion_wxyz"],
            "quaternion_convention": "wxyz_scalar_first",
            "source_result_file": str(args.cari4d_result),
            "source_result_field": "pr.pose_abs",
            "source_frame_labels": source_frame_labels,
            "initial_pose_file": str((object_dir / "initial_pose.npy").resolve()),
            **coordinate_metadata,
        }
    )
    sequence_name = args.sequence_name or args.cari4d_result.stem
    label_usage_values = [
        item["record"].get(
            "uses_motion_or_object_pose_labels",
            item["record"].get("uses_omomo_motion_or_object_pose_labels"),
        )
        for item in upstream_provenance
    ]
    uses_motion_or_pose_labels = (
        any(bool(value) for value in label_usage_values)
        if label_usage_values and all(value is not None for value in label_usage_values)
        else None
    )
    sequence_metadata = {
        "schema": "holosoma.cari4d_canonical_sequence.v1",
        "sequence_name": sequence_name,
        "num_frames": frame_count,
        "fps": fps,
        "source_video": str(args.video) if args.video is not None else None,
        "source_video_num_frames": video_frame_count,
        "source_result_file": str(args.cari4d_result),
        "trajectory_source": "CARI4D final pr fields reconstructed from source RGB video",
        "upstream_provenance": upstream_provenance,
        "uses_motion_or_object_pose_labels": uses_motion_or_pose_labels,
        "human_file": str((human_dir / "smplh.npz").resolve()),
        "object_trajectory_file": str((object_dir / "trajectory.npz").resolve()),
        "object_metric_mesh": str((object_dir / "object_metric.obj").resolve()),
        **coordinate_metadata,
    }
    write_json(human_dir / "metadata.json", human_metadata)
    write_json(object_dir / "metadata.json", object_metadata)
    write_json(meta_dir / "sequence.json", sequence_metadata)

    report = validate_and_write(output, video_frame_count=video_frame_count)
    print(f"Canonical sequence:    {output}")
    print(f"SMPL-H frames:         {frame_count}")
    print(f"Object frames:         {len(translations)}")
    print(f"Coordinate frame:      {coordinate_metadata['coordinate_frame']} (no Isaac conversion)")
    print(f"Export validation:     {report['status']}")
    if report["status"] != "PASS":
        for error in report["errors"]:
            print(f"  - {error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
