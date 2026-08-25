#!/usr/bin/env python3
"""Render final canonical SMPL-H and an accepted video-derived cuboid over RGB."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np
import torch


CUBOID_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", required=True, type=Path, help="Canonical CARI4D sequence directory.")
    parser.add_argument("--video", required=True, type=Path, help="Original/aligned RGB video.")
    parser.add_argument("--intrinsics", required=True, type=Path, help="CARI4D color.pkl camera parameters.")
    parser.add_argument(
        "--collision-report",
        required=True,
        type=Path,
        help="Report from estimate_multiframe_collision_box.py.",
    )
    parser.add_argument("--output", required=True, type=Path, help="Side-by-side H.264 visualization.")
    parser.add_argument(
        "--collision-box-output",
        type=Path,
        help="OBJ path for the recovered cuboid; defaults under sequence/object.",
    )
    parser.add_argument(
        "--accept-visible-span-cuboid",
        action="store_true",
        help=(
            "Required acknowledgement: median visible span is used as a semantic cuboid size prior even "
            "though it is not a guaranteed enclosing reconstruction."
        ),
    )
    parser.add_argument("--box-margin-m", type=float, default=0.0, help="Total size added to each box axis.")
    parser.add_argument(
        "--cari4d-root", type=Path, default=Path(__file__).resolve().parents[1] / "third_party" / "CARI4D"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--human-alpha", type=float, default=0.58)
    return parser.parse_args()


def absolute_file(path: Path) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def project(points: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    projected = points @ intrinsic.T
    return projected[..., :2] / np.maximum(projected[..., 2:3], 1e-8)


def render_body(
    bgr: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    intrinsic: np.ndarray,
    alpha: float,
) -> np.ndarray:
    height, width = bgr.shape[:2]
    xy_float = project(vertices, intrinsic)
    xy = np.rint(xy_float).astype(np.int32)
    triangles_3d = vertices[faces]
    triangles_2d = xy[faces]
    face_depth = triangles_3d[:, :, 2].mean(axis=1)
    valid = np.all(triangles_3d[:, :, 2] > 0.05, axis=1)
    lower = triangles_2d.min(axis=1)
    upper = triangles_2d.max(axis=1)
    valid &= (upper[:, 0] >= 0) & (upper[:, 1] >= 0) & (lower[:, 0] < width) & (lower[:, 1] < height)
    signed_area = (
        (triangles_2d[:, 1, 0] - triangles_2d[:, 0, 0])
        * (triangles_2d[:, 2, 1] - triangles_2d[:, 0, 1])
        - (triangles_2d[:, 1, 1] - triangles_2d[:, 0, 1])
        * (triangles_2d[:, 2, 0] - triangles_2d[:, 0, 0])
    )
    valid &= np.abs(signed_area) >= 0.5
    face_indices = np.flatnonzero(valid)
    face_indices = face_indices[np.argsort(face_depth[face_indices])[::-1]]

    layer = bgr.copy()
    mask = np.zeros((height, width), dtype=np.uint8)
    edges_a = triangles_3d[:, 1] - triangles_3d[:, 0]
    edges_b = triangles_3d[:, 2] - triangles_3d[:, 0]
    normals = np.cross(edges_a, edges_b)
    normal_length = np.linalg.norm(normals, axis=1)
    nz = np.divide(np.abs(normals[:, 2]), normal_length, out=np.zeros_like(normal_length), where=normal_length > 0)
    for face_index in face_indices:
        shade = 0.50 + 0.50 * float(nz[face_index])
        color = tuple(int(value * shade) for value in (245, 150, 45))  # BGR cyan/blue
        polygon = triangles_2d[face_index]
        cv2.fillConvexPoly(layer, polygon, color, lineType=cv2.LINE_AA)
        cv2.fillConvexPoly(mask, polygon, 255, lineType=cv2.LINE_AA)
    result = bgr.copy()
    blend = cv2.addWeighted(layer, alpha, bgr, 1.0 - alpha, 0.0)
    result[mask > 0] = blend[mask > 0]
    return result


def cuboid_corners(extents: np.ndarray) -> np.ndarray:
    half = extents * 0.5
    return np.asarray(
        [
            [-half[0], -half[1], -half[2]],
            [half[0], -half[1], -half[2]],
            [half[0], half[1], -half[2]],
            [-half[0], half[1], -half[2]],
            [-half[0], -half[1], half[2]],
            [half[0], -half[1], half[2]],
            [half[0], half[1], half[2]],
            [-half[0], half[1], half[2]],
        ],
        dtype=np.float64,
    )


def draw_cuboid(
    bgr: np.ndarray,
    corners_camera: np.ndarray,
    intrinsic: np.ndarray,
) -> np.ndarray:
    height, width = bgr.shape[:2]
    xy = np.rint(project(corners_camera, intrinsic)).astype(np.int32)
    valid = (
        (corners_camera[:, 2] > 0.05)
        & (xy[:, 0] >= 0)
        & (xy[:, 0] < width)
        & (xy[:, 1] >= 0)
        & (xy[:, 1] < height)
    )
    for start, end in CUBOID_EDGES:
        if valid[start] and valid[end]:
            cv2.line(bgr, tuple(xy[start]), tuple(xy[end]), (20, 20, 255), 3, cv2.LINE_AA)
            cv2.line(bgr, tuple(xy[start]), tuple(xy[end]), (40, 230, 255), 1, cv2.LINE_AA)
    for index, point in enumerate(xy):
        if valid[index]:
            cv2.circle(bgr, tuple(point), 3, (20, 20, 255), -1, cv2.LINE_AA)
    return bgr


def load_smplh_vertices(
    human_path: Path,
    gender: str,
    cari4d_root: Path,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    root_text = str(cari4d_root)
    inserted = root_text not in sys.path
    if inserted:
        sys.path.insert(0, root_text)
    try:
        from lib_smpl import get_smpl
    finally:
        if inserted:
            sys.path.remove(root_text)

    with np.load(human_path) as human:
        required = {"poses", "trans", "betas", "fps", "frame_ids"}
        if not required.issubset(human.files):
            raise KeyError(f"SMPL-H file is missing {sorted(required - set(human.files))}")
        poses = np.asarray(human["poses"], dtype=np.float32)
        translations = np.asarray(human["trans"], dtype=np.float32)
        betas = np.asarray(human["betas"], dtype=np.float32)
        fps = float(human["fps"])
        frame_ids = np.asarray(human["frame_ids"], dtype=np.int64)
        raw_shape = list(human["poses_cari4d_raw"].shape) if "poses_cari4d_raw" in human.files else None
    if poses.ndim != 2 or poses.shape[1] != 156:
        raise ValueError(f"Expected final SMPL-H [T,156], got {poses.shape}")
    if betas.ndim == 1:
        betas = np.repeat(betas[None], len(poses), axis=0)
    if not (len(poses) == len(translations) == len(betas) == len(frame_ids)):
        raise ValueError("SMPL-H parameter arrays have different frame counts")
    model = get_smpl(gender, hands=True, model_root=str(cari4d_root / "data" / "smpl")).eval()
    batches = []
    with torch.no_grad():
        for start in range(0, len(poses), batch_size):
            end = min(start + batch_size, len(poses))
            vertices, _, _, _ = model(
                torch.from_numpy(poses[start:end]),
                torch.from_numpy(betas[start:end]),
                torch.from_numpy(translations[start:end]),
            )
            batches.append(vertices.cpu().numpy())
    vertices = np.concatenate(batches, axis=0)
    report = {
        "smplh_file": str(human_path),
        "body_model": "SMPL-H",
        "gender": gender,
        "pose_representation": "156D axis-angle",
        "pose_shape": list(poses.shape),
        "raw_cari4d_pose_shape": raw_shape,
        "translation_shape": list(translations.shape),
        "betas_shape": list(betas.shape),
        "vertices_shape": list(vertices.shape),
        "faces": int(len(model.faces)),
        "fps": fps,
        "frame_ids_first_last": [int(frame_ids[0]), int(frame_ids[-1])],
    }
    return vertices, np.asarray(model.faces, dtype=np.int32), {**report, "frame_ids": frame_ids}


def build_recovered_cuboid(
    collision_report_path: Path,
    canonical_mesh_path: Path,
    margin_m: float,
    output_path: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    import trimesh

    report = json.loads(collision_report_path.read_text(encoding="utf-8"))
    recovered_sorted = np.asarray(
        report["depth_fusion"]["median_per_frame_visible_span_sorted_m"], dtype=np.float64
    )
    if recovered_sorted.shape != (3,) or np.any(recovered_sorted <= 0):
        raise ValueError(f"Invalid recovered visible-span dimensions: {recovered_sorted}")
    recovered_sorted = recovered_sorted + margin_m
    canonical_mesh = trimesh.load(canonical_mesh_path, force="mesh", process=False)
    if not isinstance(canonical_mesh, trimesh.Trimesh) or len(canonical_mesh.vertices) == 0:
        raise ValueError(f"Invalid canonical mesh: {canonical_mesh_path}")
    transform_box_from_object, canonical_extents = trimesh.bounds.oriented_bounds(
        canonical_mesh, angle_digits=1, ordered=False
    )
    # Preserve the canonical asset's axis rank while replacing every metric dimension with the video estimate.
    recovered_axis_order = np.empty(3, dtype=np.float64)
    recovered_axis_order[np.argsort(canonical_extents)] = recovered_sorted
    transform_object_from_box = np.linalg.inv(transform_box_from_object)
    box = trimesh.creation.box(extents=recovered_axis_order, transform=transform_object_from_box)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    box.export(output_path)
    corners_object = trimesh.transform_points(cuboid_corners(recovered_axis_order), transform_object_from_box)
    box_report = {
        "collision_box_file": str(output_path),
        "dimension_source": "median per-frame visible span from metric depth; no fused OBB dimensions",
        "source_collision_report": str(collision_report_path),
        "recovered_extent_sorted_m": np.sort(recovered_axis_order).tolist(),
        "recovered_extent_in_box_axes_m": recovered_axis_order.tolist(),
        "box_margin_m_total_per_axis": margin_m,
        "canonical_center_and_orientation_source": str(canonical_mesh_path),
        "canonical_mesh_dimensions_used": False,
        "T_object_from_box": transform_object_from_box.tolist(),
        "semantic_prior": "cuboid",
    }
    metadata_path = output_path.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(box_report, indent=2) + "\n", encoding="utf-8")
    return corners_object, recovered_axis_order, box_report


def main() -> None:
    args = parse_args()
    if not args.accept_visible_span_cuboid:
        raise ValueError(
            "Pass --accept-visible-span-cuboid to explicitly choose the multi-frame median span as a cuboid prior"
        )
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if not 0.0 <= args.human_alpha <= 1.0:
        raise ValueError("--human-alpha must be in [0,1]")
    if args.box_margin_m < 0:
        raise ValueError("--box-margin-m cannot be negative")

    sequence = args.sequence.expanduser().resolve()
    video = absolute_file(args.video)
    intrinsics_path = absolute_file(args.intrinsics)
    collision_report = absolute_file(args.collision_report)
    cari4d_root = args.cari4d_root.expanduser().resolve()
    human_path = absolute_file(sequence / "human" / "smplh.npz")
    human_metadata_path = absolute_file(sequence / "human" / "metadata.json")
    trajectory_path = absolute_file(sequence / "object" / "trajectory.npz")
    canonical_mesh_path = absolute_file(sequence / "object" / "object_metric.obj")
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    collision_box_output = (
        args.collision_box_output.expanduser().resolve()
        if args.collision_box_output
        else sequence / "object" / "collision_box_multiframe.obj"
    )

    metadata = json.loads(human_metadata_path.read_text(encoding="utf-8"))
    gender = metadata.get("gender")
    if gender not in ("male", "female"):
        raise ValueError(f"Unsupported/missing SMPL-H gender in {human_metadata_path}: {gender!r}")
    vertices, faces, human_report = load_smplh_vertices(
        human_path, gender, cari4d_root, args.batch_size
    )
    frame_ids = human_report.pop("frame_ids")
    with np.load(trajectory_path) as trajectory:
        transforms = np.asarray(trajectory["transform"], dtype=np.float64)
        object_frame_ids = np.asarray(trajectory["frame_ids"], dtype=np.int64)
    if transforms.shape != (len(frame_ids), 4, 4) or not np.array_equal(object_frame_ids, frame_ids):
        raise ValueError("Final SMPL-H and object trajectory frame IDs are not exactly aligned")

    camera = joblib.load(intrinsics_path)
    intrinsic = np.asarray(
        [
            [float(camera["fx"]), 0.0, float(camera["cx"])],
            [0.0, float(camera["fy"]), float(camera["cy"])],
            [0.0, 0.0, 1.0],
        ]
    )
    corners_object, recovered_extents, box_report = build_recovered_cuboid(
        collision_report, canonical_mesh_path, args.box_margin_m, collision_box_output
    )

    capture = cv2.VideoCapture(str(video))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if width <= 0 or height <= 0 or fps <= 0:
        raise RuntimeError(f"Could not read video properties from {video}")
    temporary = output.with_suffix(".encoding.avi")
    writer = cv2.VideoWriter(
        str(temporary), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width * 2, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create temporary video {temporary}")
    frame_to_result = {int(frame_id): index for index, frame_id in enumerate(frame_ids)}
    preview_frame = int(frame_ids[len(frame_ids) // 2])
    preview_path = output.with_suffix(".preview.png")
    rendered_count = 0
    last_requested = int(frame_ids[-1])
    source_frame = -1
    try:
        while source_frame < last_requested:
            ok, bgr = capture.read()
            if not ok:
                raise RuntimeError(f"Video ended at frame {source_frame}, before requested frame {last_requested}")
            source_frame += 1
            if source_frame not in frame_to_result:
                continue
            result_index = frame_to_result[source_frame]
            overlay = render_body(bgr, vertices[result_index], faces, intrinsic, args.human_alpha)
            transform = transforms[result_index]
            corners_camera = corners_object @ transform[:3, :3].T + transform[:3, 3]
            overlay = draw_cuboid(overlay, corners_camera, intrinsic)
            cv2.putText(bgr, "Source RGB", (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 3, cv2.LINE_AA)
            cv2.putText(bgr, "Source RGB", (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
            label = "Final SMPL-H + video-derived cuboid"
            cv2.putText(overlay, label, (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 20, 20), 3, cv2.LINE_AA)
            cv2.putText(overlay, label, (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 1, cv2.LINE_AA)
            dims = "box " + " x ".join(f"{value:.3f}" for value in np.sort(recovered_extents)) + " m"
            cv2.putText(overlay, dims, (15, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 3, cv2.LINE_AA)
            cv2.putText(overlay, dims, (15, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 230, 255), 1, cv2.LINE_AA)
            combined = np.concatenate((bgr, overlay), axis=1)
            writer.write(combined)
            rendered_count += 1
            if source_frame == preview_frame:
                cv2.imwrite(str(preview_path), combined)
    finally:
        capture.release()
        writer.release()

    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error", "-i", str(temporary),
            "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
        ],
        check=True,
    )
    temporary.unlink()
    if rendered_count != len(frame_ids):
        raise RuntimeError(f"Rendered {rendered_count} frames, expected {len(frame_ids)}")

    report = {
        "status": "PASS",
        "video": str(video),
        "video_frame_count": video_frames,
        "output_video": str(output),
        "preview": str(preview_path),
        "rendered_frames": rendered_count,
        "output_resolution": [width * 2, height],
        "camera_intrinsics": intrinsic.tolist(),
        "human": human_report,
        "object_trajectory": str(trajectory_path),
        "collision_box": box_report,
        "alignment": "SMPL-H and object use the same CARI4D metric camera frame and exact frame IDs",
        "important_limitation": (
            "CARI4D's final 72D pose was expanded to 156D with its GRAB mean-hand prior; the fingers are not "
            "independent per-frame hand articulation recovered from RGB."
        ),
    }
    report_path = output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
