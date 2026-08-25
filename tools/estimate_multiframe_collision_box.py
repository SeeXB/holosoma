#!/usr/bin/env python3
"""Estimate an object-frame collision proxy from CARI4D masks and metric depth.

This is deliberately a diagnostic/export tool, not a replacement for CARI4D.  It
supports two complementary estimates:

* motion-compensated metric-depth fusion followed by a robust OBB; and
* a silhouette visual hull, treating human-occluded pixels as unknown.

Both estimates use the supplied camera-from-object trajectory.  A reference mesh
may be supplied for *post-hoc evaluation*; it is never used by reconstruction.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class FrameCloud:
    frame_id: int
    pose_index: int
    camera_points: np.ndarray
    object_points: np.ndarray
    depth_median_m: float
    depth_mad_m: float
    depth_band_m: float
    mask_pixels: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video-prefix",
        required=True,
        type=Path,
        help="Path before '.<camera>.color.mp4', for example videos-aligned/MySequence.",
    )
    parser.add_argument("--masks-h5", required=True, type=Path, help="CARI4D SAM mask HDF5 file.")
    parser.add_argument("--pose-file", required=True, type=Path)
    parser.add_argument(
        "--pose-format",
        required=True,
        choices=("foundationpose-pkl", "cari4d-result", "canonical-npz"),
        help="Explicit native pose serialization; it is not inferred from the filename.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sequence-name", help="HDF5 group name; defaults to video-prefix basename.")
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--foundationpose-index", type=int, default=0)
    parser.add_argument(
        "--cari4d-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "third_party" / "CARI4D",
        help="Needed only to unpickle a final CARI4D result containing TrainState.",
    )
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--max-points-per-frame", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--mask-erosion-px", type=int, default=2)
    parser.add_argument("--min-depth-m", type=float, default=0.1)
    parser.add_argument("--max-depth-m", type=float, default=20.0)
    parser.add_argument("--depth-mad-multiplier", type=float, default=6.0)
    parser.add_argument("--min-depth-band-m", type=float, default=0.25)
    parser.add_argument("--max-depth-band-m", type=float, default=0.75)
    parser.add_argument("--voxel-size-m", type=float, default=0.01)
    parser.add_argument("--min-voxel-frame-support", type=int, default=2)
    parser.add_argument("--obb-tail-quantile", type=float, default=0.005)
    parser.add_argument(
        "--object-mask-key",
        default="{sequence}/{frame:06d}-k{camera}.obj_rend_mask.png",
        help="HDF5 key template.",
    )
    parser.add_argument(
        "--human-mask-key",
        default="{sequence}/{frame:06d}-k{camera}.person_mask.png",
        help="HDF5 key template. Human pixels are ignored during silhouette carving.",
    )
    parser.add_argument("--skip-visual-hull", action="store_true")
    parser.add_argument("--visual-hull-resolution-m", type=float, default=0.02)
    parser.add_argument(
        "--visual-hull-half-extent-m",
        type=float,
        help="Half-width of the cubic search region. Defaults to 2x the median visible span.",
    )
    parser.add_argument("--visual-hull-positive-ratio", type=float, default=0.90)
    parser.add_argument("--visual-hull-min-evidence", type=int, default=20)
    parser.add_argument("--visual-hull-mask-dilation-px", type=int, default=2)
    parser.add_argument("--visual-hull-front-tolerance-m", type=float, default=0.10)
    parser.add_argument("--max-visual-hull-voxels", type=int, default=2_000_000)
    parser.add_argument(
        "--evaluation-mesh",
        type=Path,
        help="Optional ground-truth/evaluation mesh. Loaded only after all estimates are complete.",
    )
    return parser.parse_args()


def _absolute(path: Path) -> Path:
    return path.expanduser().resolve()


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _frame_id(value: Any) -> int:
    if isinstance(value, (int, np.integer)):
        return int(value)
    match = re.search(r"(\d+)$", str(value))
    if match is None:
        raise ValueError(f"Cannot parse a numeric frame ID from {value!r}")
    return int(match.group(1))


def load_poses(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    path = _absolute(args.pose_file)
    if not path.is_file():
        raise FileNotFoundError(path)
    if args.pose_format == "foundationpose-pkl":
        import joblib

        data = joblib.load(path)
        if not isinstance(data, dict) or "fp_poses" not in data or "frames" not in data:
            raise ValueError("FoundationPose input must contain the real fp_poses and frames fields")
        poses_all = _as_numpy(data["fp_poses"])
        if poses_all.ndim != 4 or poses_all.shape[-2:] != (4, 4):
            raise ValueError(f"fp_poses must be [T,K,4,4], got {poses_all.shape}")
        if not 0 <= args.foundationpose_index < poses_all.shape[1]:
            raise IndexError(
                f"--foundationpose-index {args.foundationpose_index} is outside K={poses_all.shape[1]}"
            )
        poses = poses_all[:, args.foundationpose_index]
        raw_frames = data["frames"]
        provenance = {"field": "fp_poses", "candidate_index": args.foundationpose_index}
    elif args.pose_format == "cari4d-result":
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is required to read CARI4D's final .pth result") from exc
        cari4d_root = str(_absolute(args.cari4d_root))
        inserted = cari4d_root not in sys.path
        if inserted:
            sys.path.insert(0, cari4d_root)
        try:
            try:
                data = torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:
                data = torch.load(path, map_location="cpu")
        finally:
            if inserted:
                sys.path.remove(cari4d_root)
        if not isinstance(data, dict) or not isinstance(data.get("pr"), dict):
            raise ValueError("CARI4D result must use the native {'gt','pr','in'} serialization")
        prediction = data["pr"]
        if "pose_abs" not in prediction or "frames" not in prediction:
            raise KeyError("CARI4D pr must contain pose_abs and frames")
        poses = _as_numpy(prediction["pose_abs"])
        raw_frames = prediction["frames"]
        provenance = {"field": "pr.pose_abs"}
    else:
        with np.load(path) as data:
            required = {"transform", "frame_ids"}
            if not required.issubset(data.files):
                raise KeyError(f"Canonical trajectory is missing {sorted(required - set(data.files))}")
            poses = np.asarray(data["transform"])
            raw_frames = np.asarray(data["frame_ids"])
        provenance = {"field": "transform"}

    poses = np.asarray(poses, dtype=np.float64)
    frame_ids = np.asarray([_frame_id(value) for value in raw_frames], dtype=np.int64)
    if poses.shape != (len(frame_ids), 4, 4):
        raise ValueError(f"Pose/frame shape mismatch: poses={poses.shape}, frames={frame_ids.shape}")
    if len(np.unique(frame_ids)) != len(frame_ids):
        raise ValueError("Pose file contains duplicate frame IDs")
    if not np.isfinite(poses).all():
        raise ValueError("Pose trajectory contains NaN or Inf")
    identity_error = np.max(np.abs(poses[:, 3] - np.asarray([0.0, 0.0, 0.0, 1.0])))
    rotations = poses[:, :3, :3]
    orthogonality_error = np.max(np.abs(rotations.transpose(0, 2, 1) @ rotations - np.eye(3)))
    determinant_error = np.max(np.abs(np.linalg.det(rotations) - 1.0))
    if identity_error > 1e-5 or orthogonality_error > 2e-3 or determinant_error > 2e-3:
        raise ValueError(
            "Invalid rigid transforms: "
            f"last-row error={identity_error:.3g}, R orthogonality={orthogonality_error:.3g}, "
            f"det(R) error={determinant_error:.3g}"
        )
    provenance.update(
        {
            "pose_file": str(path),
            "pose_format": args.pose_format,
            "transform_convention": "T_camera_from_object",
            "frame_count": int(len(frame_ids)),
            "max_rotation_orthogonality_error": float(orthogonality_error),
            "max_rotation_determinant_error": float(determinant_error),
        }
    )
    return poses, frame_ids, provenance


def load_camera(video_prefix: Path, camera_id: int) -> tuple[np.ndarray, dict[str, Any]]:
    import joblib

    path = Path(f"{video_prefix}.{camera_id}.color.pkl")
    if not path.is_file():
        raise FileNotFoundError(f"Missing CARI4D/UniDepth camera file: {path}")
    data = joblib.load(path)
    required = ("fx", "fy", "cx", "cy", "H", "W")
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"Camera file {path} is missing {missing}")
    fx, fy, cx, cy = (float(data[key]) for key in ("fx", "fy", "cx", "cy"))
    intrinsic = np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    report = {
        "camera_file": str(path.resolve()),
        "intrinsic_matrix": intrinsic.tolist(),
        "image_height": int(data["H"]),
        "image_width": int(data["W"]),
        "intrinsics_policy": "fixed fx/fy/cx/cy saved by CARI4D (same policy used by its wild-video pipeline)",
    }
    return intrinsic, report


def _mask_key(template: str, sequence: str, frame_id: int, camera_id: int) -> str:
    try:
        return template.format(sequence=sequence, frame=frame_id, camera=camera_id)
    except (KeyError, ValueError) as exc:
        raise ValueError(f"Invalid mask key template {template!r}: {exc}") from exc


def load_frame_clouds(
    args: argparse.Namespace,
    poses: np.ndarray,
    pose_frame_ids: np.ndarray,
    intrinsic: np.ndarray,
) -> tuple[list[FrameCloud], dict[str, Any]]:
    try:
        import cv2
        import h5py
        from videoio import Uint16Reader
    except ImportError as exc:
        raise RuntimeError(
            "Depth extraction requires the CARI4D Stage-A environment (cv2, h5py, videoio)."
        ) from exc

    depth_path = Path(f"{args.video_prefix}.{args.camera_id}.depth-reg.mp4")
    if not depth_path.is_file():
        raise FileNotFoundError(depth_path)
    if not args.masks_h5.is_file():
        raise FileNotFoundError(args.masks_h5)
    selected_indices = np.arange(0, len(pose_frame_ids), args.frame_step, dtype=np.int64)
    frame_to_pose = {int(pose_frame_ids[index]): int(index) for index in selected_indices}
    wanted = set(frame_to_pose)
    rng = np.random.default_rng(args.random_seed)
    fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
    kernel_size = args.mask_erosion_px * 2 + 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8) if args.mask_erosion_px else None
    clouds: list[FrameCloud] = []
    seen_video_frames = 0

    reader = Uint16Reader(str(depth_path))
    try:
        with h5py.File(args.masks_h5, "r") as masks:
            for frame_id, raw_depth in enumerate(reader):
                seen_video_frames += 1
                if frame_id not in wanted:
                    continue
                pose_index = frame_to_pose[frame_id]
                key = _mask_key(args.object_mask_key, args.sequence_name, frame_id, args.camera_id)
                if key not in masks:
                    raise KeyError(f"Missing object mask {key!r} in {args.masks_h5}")
                mask = np.asarray(masks[key]).astype(np.uint8)
                if kernel is not None:
                    mask = cv2.erode(mask, kernel, iterations=1)
                mask = mask.astype(bool)
                depth_m = np.asarray(raw_depth).squeeze().astype(np.float64) / 1000.0
                if depth_m.shape != mask.shape:
                    raise ValueError(
                        f"Depth/mask shape mismatch at frame {frame_id}: {depth_m.shape} vs {mask.shape}"
                    )
                valid = mask & (depth_m >= args.min_depth_m) & (depth_m <= args.max_depth_m)
                rows, cols = np.nonzero(valid)
                depths = depth_m[rows, cols]
                if len(depths) < 50:
                    raise ValueError(f"Frame {frame_id} has only {len(depths)} valid eroded object pixels")
                median = float(np.median(depths))
                mad = float(np.median(np.abs(depths - median)))
                robust_sigma = 1.4826 * mad
                band = float(
                    np.clip(
                        args.depth_mad_multiplier * robust_sigma,
                        args.min_depth_band_m,
                        args.max_depth_band_m,
                    )
                )
                keep = np.abs(depths - median) <= band
                rows, cols, depths = rows[keep], cols[keep], depths[keep]
                if len(depths) < 50:
                    raise ValueError(f"Frame {frame_id} has only {len(depths)} pixels after depth filtering")
                if len(depths) > args.max_points_per_frame:
                    sample = rng.choice(len(depths), args.max_points_per_frame, replace=False)
                    rows, cols, depths = rows[sample], cols[sample], depths[sample]
                camera_points = np.column_stack(
                    ((cols - cx) * depths / fx, (rows - cy) * depths / fy, depths)
                )
                pose = poses[pose_index]
                # Row-vector form of p_obj = R^T (p_cam - t).
                object_points = (camera_points - pose[:3, 3]) @ pose[:3, :3]
                clouds.append(
                    FrameCloud(
                        frame_id=frame_id,
                        pose_index=pose_index,
                        camera_points=camera_points,
                        object_points=object_points,
                        depth_median_m=median,
                        depth_mad_m=mad,
                        depth_band_m=band,
                        mask_pixels=int(mask.sum()),
                    )
                )
    finally:
        reader.close()

    missing_video_frames = sorted(wanted - {cloud.frame_id for cloud in clouds})
    if missing_video_frames:
        raise ValueError(
            f"Pose frames are absent from the depth video ({seen_video_frames} frames): {missing_video_frames[:20]}"
        )
    if len(clouds) < 3:
        raise ValueError(f"Multi-frame estimation requires at least 3 frames, got {len(clouds)}")
    diagnostics = {
        "depth_video": str(depth_path.resolve()),
        "depth_storage_unit": "uint16 millimeter",
        "backprojected_unit": "meter",
        "frames_used": [cloud.frame_id for cloud in clouds],
        "num_frames_used": len(clouds),
        "frame_step": args.frame_step,
        "mask_erosion_px": args.mask_erosion_px,
        "sampled_points_per_frame_min_median_max": [
            int(min(len(cloud.object_points) for cloud in clouds)),
            float(np.median([len(cloud.object_points) for cloud in clouds])),
            int(max(len(cloud.object_points) for cloud in clouds)),
        ],
        "mask_pixels_min_median_max": [
            int(min(cloud.mask_pixels for cloud in clouds)),
            float(np.median([cloud.mask_pixels for cloud in clouds])),
            int(max(cloud.mask_pixels for cloud in clouds)),
        ],
        "depth_band_m_min_median_max": [
            float(min(cloud.depth_band_m for cloud in clouds)),
            float(np.median([cloud.depth_band_m for cloud in clouds])),
            float(max(cloud.depth_band_m for cloud in clouds)),
        ],
    }
    return clouds, diagnostics


def voxel_frame_support_filter(
    points: np.ndarray,
    point_frames: np.ndarray,
    voxel_size: float,
    min_frame_support: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    voxel_indices = np.floor(points / voxel_size).astype(np.int64)
    unique_voxels, inverse = np.unique(voxel_indices, axis=0, return_inverse=True)
    voxel_frame_pairs = np.unique(np.column_stack((inverse, point_frames)), axis=0)
    support = np.bincount(voxel_frame_pairs[:, 0], minlength=len(unique_voxels))
    keep = support[inverse] >= min_frame_support
    filtered = points[keep]
    filtered_inverse = inverse[keep]
    if len(filtered) < 100:
        raise ValueError(
            f"Voxel support retained only {len(filtered)} points; lower --min-voxel-frame-support "
            "or improve pose/depth alignment"
        )
    # Use one observed point per occupied voxel for hull/orientation fitting.
    _, first = np.unique(filtered_inverse, return_index=True)
    downsampled = filtered[first]
    report = {
        "voxel_size_m": voxel_size,
        "minimum_distinct_frame_support": min_frame_support,
        "input_points": int(len(points)),
        "retained_points": int(len(filtered)),
        "retained_fraction": float(len(filtered) / len(points)),
        "retained_voxels": int(len(downsampled)),
        "voxel_frame_support_min_median_max": [
            int(support.min()), float(np.median(support)), int(support.max())
        ],
    }
    return filtered, downsampled, report


def fit_robust_obb(
    points: np.ndarray,
    orientation_points: np.ndarray | None,
    tail_quantile: float,
) -> tuple[Any, dict[str, Any]]:
    import trimesh

    orientation_points = points if orientation_points is None else orientation_points
    if len(orientation_points) < 4:
        raise ValueError("At least four non-coplanar points are required for an OBB")
    to_box, raw_extents = trimesh.bounds.oriented_bounds(
        orientation_points, angle_digits=1, ordered=False
    )
    box_points = trimesh.transform_points(points, to_box)
    lower = np.quantile(box_points, tail_quantile, axis=0)
    upper = np.quantile(box_points, 1.0 - tail_quantile, axis=0)
    extents = upper - lower
    if np.any(extents <= 0):
        raise ValueError(f"Degenerate OBB extents: {extents}")
    center_in_box = (lower + upper) * 0.5
    center_shift = np.eye(4)
    center_shift[:3, 3] = center_in_box
    box_to_object = np.linalg.inv(to_box) @ center_shift
    box = trimesh.creation.box(extents=extents, transform=box_to_object)
    report = {
        "tail_quantile": tail_quantile,
        "extent_xyz_box_m": extents.tolist(),
        "extent_sorted_m": np.sort(extents).tolist(),
        "raw_orientation_fit_extent_m": np.asarray(raw_extents).tolist(),
        "T_object_from_box": box_to_object.tolist(),
        "center_object_m": box_to_object[:3, 3].tolist(),
    }
    return box, report


def estimate_depth_fusion(
    clouds: list[FrameCloud], args: argparse.Namespace, output: Path
) -> tuple[np.ndarray, Any, dict[str, Any]]:
    import trimesh

    points = np.concatenate([cloud.object_points for cloud in clouds], axis=0)
    point_frames = np.concatenate(
        [np.full(len(cloud.object_points), cloud.frame_id, dtype=np.int64) for cloud in clouds]
    )
    filtered, downsampled, support_report = voxel_frame_support_filter(
        points, point_frames, args.voxel_size_m, args.min_voxel_frame_support
    )
    box, obb_report = fit_robust_obb(filtered, downsampled, args.obb_tail_quantile)
    point_cloud_path = output / "depth_fusion_points.ply"
    box_path = output / "depth_fusion_obb.obj"
    hull_path = output / "depth_fusion_convex_hull.obj"
    trimesh.points.PointCloud(downsampled).export(point_cloud_path)
    box.export(box_path)
    hull = trimesh.points.PointCloud(downsampled).convex_hull
    hull.export(hull_path)

    per_frame_extent = []
    per_frame_center = []
    for cloud in clouds:
        lower = np.quantile(cloud.object_points, args.obb_tail_quantile, axis=0)
        upper = np.quantile(cloud.object_points, 1.0 - args.obb_tail_quantile, axis=0)
        per_frame_extent.append(upper - lower)
        per_frame_center.append((upper + lower) * 0.5)
    per_frame_extent = np.asarray(per_frame_extent)
    per_frame_center = np.asarray(per_frame_center)
    visible_span = np.median(per_frame_extent, axis=0)
    visible_sorted = np.sort(visible_span)
    fusion_sorted = np.asarray(obb_report["extent_sorted_m"])
    inflation = fusion_sorted / np.maximum(visible_sorted, 1e-8)
    report = {
        **support_report,
        **obb_report,
        "coordinate_frame": "pose_source_object_canonical_frame",
        "point_cloud_file": str(point_cloud_path),
        "obb_file": str(box_path),
        "convex_hull_file": str(hull_path),
        "median_per_frame_visible_span_canonical_axes_m": visible_span.tolist(),
        "median_per_frame_visible_span_sorted_m": visible_sorted.tolist(),
        "per_frame_visible_span_q10_q50_q90_m": np.quantile(
            per_frame_extent, [0.1, 0.5, 0.9], axis=0
        ).tolist(),
        "median_per_frame_visible_center_object_m": np.median(per_frame_center, axis=0).tolist(),
        "fusion_to_visible_span_ratio_sorted": inflation.tolist(),
        "internally_unstable": bool(np.max(inflation) > 1.35),
        "warning": (
            "The median visible span is a diagnostic lower-bound-like statistic, not an enclosing collider."
        ),
    }
    return downsampled, box, report


def estimate_visual_hull(
    args: argparse.Namespace,
    clouds: list[FrameCloud],
    poses: np.ndarray,
    intrinsic: np.ndarray,
    output: Path,
) -> tuple[np.ndarray, Any, dict[str, Any]]:
    try:
        import cv2
        import h5py
        import trimesh
        from videoio import Uint16Reader
    except ImportError as exc:
        raise RuntimeError("Visual-hull estimation requires the CARI4D Stage-A environment") from exc

    per_frame_centers = []
    per_frame_extents = []
    for cloud in clouds:
        lower = np.quantile(cloud.object_points, args.obb_tail_quantile, axis=0)
        upper = np.quantile(cloud.object_points, 1.0 - args.obb_tail_quantile, axis=0)
        per_frame_centers.append((lower + upper) * 0.5)
        per_frame_extents.append(upper - lower)
    center = np.median(per_frame_centers, axis=0)
    median_span = np.median(per_frame_extents, axis=0)
    half_extent = args.visual_hull_half_extent_m
    if half_extent is None:
        half_extent = float(np.clip(2.0 * np.max(median_span), 0.50, 1.50))
    resolution = args.visual_hull_resolution_m
    axis_count = int(np.floor((2.0 * half_extent) / resolution)) + 1
    voxel_count = axis_count**3
    if voxel_count > args.max_visual_hull_voxels:
        raise ValueError(
            f"Visual-hull grid would contain {voxel_count:,} voxels; increase resolution or "
            f"raise --max-visual-hull-voxels explicitly (current {args.max_visual_hull_voxels:,})"
        )
    axis = (np.arange(axis_count) - (axis_count - 1) * 0.5) * resolution
    xx, yy, zz = np.meshgrid(axis, axis, axis, indexing="ij")
    grid = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel())) + center
    positive = np.zeros(len(grid), dtype=np.uint16)
    negative = np.zeros(len(grid), dtype=np.uint16)
    fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
    dilation_size = args.visual_hull_mask_dilation_px * 2 + 1
    kernel = np.ones((dilation_size, dilation_size), np.uint8)
    cloud_by_frame = {cloud.frame_id: cloud for cloud in clouds}
    wanted = set(cloud_by_frame)
    missing_human_masks: list[int] = []
    depth_path = Path(f"{args.video_prefix}.{args.camera_id}.depth-reg.mp4")

    reader = Uint16Reader(str(depth_path))
    try:
        with h5py.File(args.masks_h5, "r") as masks:
            for frame_id, raw_depth in enumerate(reader):
                if frame_id not in wanted:
                    continue
                cloud = cloud_by_frame[frame_id]
                pose = poses[cloud.pose_index]
                object_key = _mask_key(args.object_mask_key, args.sequence_name, frame_id, args.camera_id)
                human_key = _mask_key(args.human_mask_key, args.sequence_name, frame_id, args.camera_id)
                object_mask = np.asarray(masks[object_key]).astype(np.uint8)
                object_mask = cv2.dilate(object_mask, kernel, iterations=1).astype(bool)
                if human_key in masks:
                    human_mask = np.asarray(masks[human_key]).astype(bool)
                else:
                    missing_human_masks.append(frame_id)
                    human_mask = np.zeros_like(object_mask)
                depth_m = np.asarray(raw_depth).squeeze().astype(np.float64) / 1000.0

                camera_points = grid @ pose[:3, :3].T + pose[:3, 3]
                z = camera_points[:, 2]
                with np.errstate(divide="ignore", invalid="ignore"):
                    cols = np.rint(fx * camera_points[:, 0] / z + cx).astype(np.int32)
                    rows = np.rint(fy * camera_points[:, 1] / z + cy).astype(np.int32)
                valid = (
                    (z > 0.0)
                    & (cols >= 0)
                    & (cols < object_mask.shape[1])
                    & (rows >= 0)
                    & (rows < object_mask.shape[0])
                )
                indices = np.flatnonzero(valid)
                in_object = object_mask[rows[indices], cols[indices]]
                in_human = human_mask[rows[indices], cols[indices]]
                observed_depth = depth_m[rows[indices], cols[indices]]
                in_front = (
                    in_object
                    & (observed_depth > args.min_depth_m)
                    & (z[indices] < observed_depth - args.visual_hull_front_tolerance_m)
                )
                accepted = in_object & ~in_front
                rejected = (~in_object & ~in_human) | in_front
                positive[indices[accepted]] += 1
                negative[indices[rejected]] += 1
    finally:
        reader.close()

    evidence = positive.astype(np.int32) + negative.astype(np.int32)
    ratio = np.divide(
        positive,
        evidence,
        out=np.zeros_like(positive, dtype=np.float64),
        where=evidence > 0,
    )
    occupied = (evidence >= args.visual_hull_min_evidence) & (
        ratio >= args.visual_hull_positive_ratio
    )
    occupied_points = grid[occupied]
    if len(occupied_points) < 100:
        raise ValueError(
            f"Visual hull retained only {len(occupied_points)} voxels; inspect masks/poses or lower its thresholds"
        )
    box, box_report = fit_robust_obb(occupied_points, None, 0.0)
    points_path = output / "visual_hull_voxels.ply"
    box_path = output / "visual_hull_obb.obj"
    hull_path = output / "visual_hull_convex_hull.obj"
    trimesh.points.PointCloud(occupied_points).export(points_path)
    box.export(box_path)
    trimesh.points.PointCloud(occupied_points).convex_hull.export(hull_path)
    report = {
        **box_report,
        "coordinate_frame": "pose_source_object_canonical_frame",
        "grid_center_object_m": center.tolist(),
        "grid_half_extent_m": half_extent,
        "grid_resolution_m": resolution,
        "grid_voxels": int(len(grid)),
        "occupied_voxels": int(len(occupied_points)),
        "positive_ratio_threshold": args.visual_hull_positive_ratio,
        "minimum_evidence_frames": args.visual_hull_min_evidence,
        "front_surface_tolerance_m": args.visual_hull_front_tolerance_m,
        "missing_human_mask_frames": missing_human_masks,
        "point_cloud_file": str(points_path),
        "obb_file": str(box_path),
        "convex_hull_file": str(hull_path),
        "warning": (
            "A single fixed camera cannot constrain the visual hull along directions not exposed by object rotation."
        ),
    }
    return occupied_points, box, report


def evaluate_mesh(path: Path, estimates: dict[str, dict[str, Any]]) -> dict[str, Any]:
    import trimesh

    path = _absolute(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    loaded = trimesh.load(path, force="mesh", process=False)
    if not isinstance(loaded, trimesh.Trimesh) or len(loaded.vertices) == 0:
        raise ValueError(f"Evaluation mesh is empty or not a mesh: {path}")
    _, reference_extent = trimesh.bounds.oriented_bounds(loaded, angle_digits=1, ordered=False)
    reference_sorted = np.sort(reference_extent)
    comparisons: dict[str, Any] = {}
    for name, estimate in estimates.items():
        estimated = np.asarray(estimate["extent_sorted_m"], dtype=np.float64)
        comparisons[name] = {
            "estimated_extent_sorted_m": estimated.tolist(),
            "relative_error_sorted": ((estimated - reference_sorted) / reference_sorted).tolist(),
            "max_absolute_relative_error": float(
                np.max(np.abs((estimated - reference_sorted) / reference_sorted))
            ),
        }
    return {
        "evaluation_only": True,
        "mesh_was_not_used_by_any_estimator": True,
        "mesh_file": str(path),
        "reference_obb_extent_sorted_m": reference_sorted.tolist(),
        "reference_aabb_extent_m": np.ptp(np.asarray(loaded.vertices), axis=0).tolist(),
        "comparisons": comparisons,
    }


def save_visualization(
    output: Path,
    estimates: list[tuple[str, np.ndarray, Any]],
) -> Path:
    import os

    cache = output / ".cache"
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(7 * len(estimates), 6))
    rng = np.random.default_rng(0)
    for column, (title, points, box) in enumerate(estimates, start=1):
        axis = figure.add_subplot(1, len(estimates), column, projection="3d")
        if len(points) > 15_000:
            points = points[rng.choice(len(points), 15_000, replace=False)]
        axis.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.35, alpha=0.25, color="#0072B2")
        for edge in box.edges_unique:
            segment = box.vertices[edge]
            axis.plot(segment[:, 0], segment[:, 1], segment[:, 2], color="#D55E00", linewidth=1.5)
        combined = np.vstack((points, box.vertices))
        center = (combined.min(axis=0) + combined.max(axis=0)) * 0.5
        radius = max(float(np.ptp(combined, axis=0).max()) * 0.55, 0.05)
        axis.set_xlim(center[0] - radius, center[0] + radius)
        axis.set_ylim(center[1] - radius, center[1] + radius)
        axis.set_zlim(center[2] - radius, center[2] + radius)
        axis.set_xlabel("object X (m)")
        axis.set_ylabel("object Y (m)")
        axis.set_zlabel("object Z (m)")
        axis.set_title(title)
    figure.tight_layout()
    path = output / "multiframe_collision_estimates.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def validate_args(args: argparse.Namespace) -> None:
    positive_ints = {
        "--frame-step": args.frame_step,
        "--max-points-per-frame": args.max_points_per_frame,
        "--min-voxel-frame-support": args.min_voxel_frame_support,
        "--visual-hull-min-evidence": args.visual_hull_min_evidence,
    }
    invalid = {name: value for name, value in positive_ints.items() if value <= 0}
    if invalid:
        raise ValueError(f"These arguments must be positive: {invalid}")
    if not 0.0 <= args.obb_tail_quantile < 0.5:
        raise ValueError("--obb-tail-quantile must be in [0, 0.5)")
    if not 0.0 < args.visual_hull_positive_ratio <= 1.0:
        raise ValueError("--visual-hull-positive-ratio must be in (0, 1]")
    if args.min_depth_band_m > args.max_depth_band_m:
        raise ValueError("--min-depth-band-m cannot exceed --max-depth-band-m")
    for name in ("voxel_size_m", "visual_hull_resolution_m"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")


def main() -> None:
    args = parse_args()
    args.video_prefix = _absolute(args.video_prefix)
    args.masks_h5 = _absolute(args.masks_h5)
    args.output = _absolute(args.output)
    args.sequence_name = args.sequence_name or args.video_prefix.name
    validate_args(args)
    args.output.mkdir(parents=True, exist_ok=True)

    poses, frame_ids, pose_report = load_poses(args)
    intrinsic, camera_report = load_camera(args.video_prefix, args.camera_id)
    clouds, depth_report = load_frame_clouds(args, poses, frame_ids, intrinsic)
    depth_points, depth_box, depth_fusion_report = estimate_depth_fusion(clouds, args, args.output)

    estimates_for_visualization = [("Metric-depth fusion OBB", depth_points, depth_box)]
    estimate_reports = {"depth_fusion_obb": depth_fusion_report}
    visual_hull_report: dict[str, Any] | None = None
    if not args.skip_visual_hull:
        hull_points, hull_box, visual_hull_report = estimate_visual_hull(
            args, clouds, poses, intrinsic, args.output
        )
        estimates_for_visualization.append(("Silhouette visual-hull OBB", hull_points, hull_box))
        estimate_reports["visual_hull_obb"] = visual_hull_report

    visualization = save_visualization(args.output, estimates_for_visualization)
    evaluation = evaluate_mesh(args.evaluation_mesh, estimate_reports) if args.evaluation_mesh else None
    internal_unstable = bool(depth_fusion_report["internally_unstable"])
    if visual_hull_report is not None:
        depth_max = max(depth_fusion_report["median_per_frame_visible_span_sorted_m"])
        hull_max = max(visual_hull_report["extent_sorted_m"])
        visual_hull_report["max_extent_to_visible_span_ratio"] = float(hull_max / depth_max)
        visual_hull_report["internally_unconstrained"] = bool(hull_max / depth_max > 1.5)
        internal_unstable |= visual_hull_report["internally_unconstrained"]

    report = {
        "status": "UNSTABLE" if internal_unstable else "PASS",
        "safe_to_use_as_simulation_collider_without_review": not internal_unstable,
        "method": "multi-frame metric-depth fusion plus optional occlusion-aware silhouette visual hull",
        "sequence_name": args.sequence_name,
        "coordinate_convention": "T_camera_from_object; outputs are in pose-source object canonical frame",
        "length_unit": "meter",
        "pose": pose_report,
        "camera": camera_report,
        "depth": depth_report,
        "depth_fusion": depth_fusion_report,
        "visual_hull": visual_hull_report,
        "evaluation": evaluation,
        "visualization": str(visualization),
        "interpretation": (
            "UNSTABLE means temporal pose/depth disagreement or insufficient silhouette view diversity made "
            "the enclosing box much larger than the typical per-frame visible object span."
        ),
    }
    report_path = args.output / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Frames used: {len(clouds)}")
    print(
        "Median per-frame visible span (sorted): "
        + " x ".join(f"{value:.4f}" for value in depth_fusion_report["median_per_frame_visible_span_sorted_m"])
        + " m"
    )
    print(
        "Depth-fusion OBB (sorted): "
        + " x ".join(f"{value:.4f}" for value in depth_fusion_report["extent_sorted_m"])
        + " m"
    )
    if visual_hull_report is not None:
        print(
            "Visual-hull OBB (sorted): "
            + " x ".join(f"{value:.4f}" for value in visual_hull_report["extent_sorted_m"])
            + " m"
        )
    if evaluation is not None:
        print(
            "Evaluation mesh OBB (sorted; not used for estimation): "
            + " x ".join(f"{value:.4f}" for value in evaluation["reference_obb_extent_sorted_m"])
            + " m"
        )
    print(f"Collider readiness: {report['status']}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
