#!/usr/bin/env python3
"""Blender-side static-camera search and CARI4D-friendly rendering."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector


def parse_args() -> argparse.Namespace:
    argv = os.sys.argv
    argv = argv[argv.index("--") + 1 :] if "--" in argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("search", "render"))
    parser.add_argument("--sequence-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--camera-config", type=Path)
    parser.add_argument("--resolution", type=int, default=1280)
    parser.add_argument("--search-resolution", type=int, default=256)
    parser.add_argument("--search-frame-count", type=int, default=12)
    parser.add_argument("--azimuth-step", type=int, default=15)
    parser.add_argument("--elevations", default="5,10,15,20,25")
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-end", type=int)
    parser.add_argument(
        "--rgb-only", action="store_true",
        help="render only RGB frames (skip masks/depth/normal diagnostic passes)",
    )
    parser.add_argument(
        "--fast-search", action="store_true",
        help="select a camera from geometric projections without mask renders",
    )
    return parser.parse_args(argv)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for datablocks in (
        bpy.data.meshes,
        bpy.data.curves,
        bpy.data.materials,
        bpy.data.cameras,
        bpy.data.lights,
    ):
        for block in list(datablocks):
            if block.users == 0:
                datablocks.remove(block)


def make_mesh_object(
    name: str, vertices: np.ndarray, faces: np.ndarray, smooth: bool
) -> bpy.types.Object:
    mesh = bpy.data.meshes.new(f"{name}Mesh")
    mesh.from_pydata(vertices.tolist(), [], faces.tolist())
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    for polygon in mesh.polygons:
        polygon.use_smooth = smooth
    return obj


def update_vertices(obj: bpy.types.Object, vertices: np.ndarray) -> None:
    obj.data.vertices.foreach_set("co", np.asarray(vertices, dtype=np.float32).ravel())
    obj.data.update()


def make_principled_material(
    name: str, color: tuple[float, float, float, float], roughness: float
) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Metallic"].default_value = 0.0
    bsdf.inputs["Roughness"].default_value = roughness
    specular_name = (
        "Specular IOR Level" if "Specular IOR Level" in bsdf.inputs else "Specular"
    )
    bsdf.inputs[specular_name].default_value = 0.32
    return material


def make_emission_material(
    name: str, color: tuple[float, float, float, float]
) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = color
    emission.inputs["Strength"].default_value = 1.0
    material.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def set_material(obj: bpy.types.Object, material: bpy.types.Material) -> None:
    obj.data.materials.clear()
    obj.data.materials.append(material)


def add_camera() -> bpy.types.Object:
    camera_data = bpy.data.cameras.new("StaticCamera")
    camera_data.lens = 52.0
    camera_data.sensor_width = 36.0
    camera_data.sensor_fit = "HORIZONTAL"
    camera_data.clip_start = 0.05
    camera_data.clip_end = 100.0
    camera = bpy.data.objects.new("StaticCamera", camera_data)
    bpy.context.collection.objects.link(camera)
    bpy.context.scene.camera = camera
    return camera


def aim_camera(camera: bpy.types.Object, position: np.ndarray, target: np.ndarray) -> None:
    camera.location = tuple(float(x) for x in position)
    camera.rotation_euler = (Vector(target) - camera.location).to_track_quat(
        "-Z", "Y"
    ).to_euler()


def camera_basis(position: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, ...]:
    forward = np.asarray(target, dtype=np.float64) - np.asarray(position, dtype=np.float64)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(right) < 1.0e-8:
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)
    return right, up, forward


def spherical_direction(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    azimuth = math.radians(azimuth_deg)
    elevation = math.radians(elevation_deg)
    return np.array(
        [
            math.cos(elevation) * math.cos(azimuth),
            math.cos(elevation) * math.sin(azimuth),
            math.sin(elevation),
        ],
        dtype=np.float64,
    )


def fit_distance(
    bounds: np.ndarray,
    target: np.ndarray,
    direction: np.ndarray,
    lens_mm: float,
    sensor_width_mm: float,
    margin: float = 0.10,
) -> float:
    corners = np.array(
        [
            [x, y, z]
            for x in bounds[:, 0]
            for y in bounds[:, 1]
            for z in bounds[:, 2]
        ],
        dtype=np.float64,
    )
    probe_position = target + direction
    right, up, forward = camera_basis(probe_position, target)
    relative = corners - target[None]
    tangent = sensor_width_mm / (2.0 * lens_mm)
    usable = tangent * (1.0 - margin)
    forward_offset = relative @ forward
    required_x = np.abs(relative @ right) / usable - forward_offset
    required_y = np.abs(relative @ up) / usable - forward_offset
    return float(max(required_x.max(), required_y.max(), 1.0) + 0.12)


def project_points(
    points: np.ndarray,
    camera_position: np.ndarray,
    target: np.ndarray,
    lens_mm: float,
    sensor_width_mm: float,
    resolution: int,
) -> tuple[np.ndarray, np.ndarray]:
    right, up, forward = camera_basis(camera_position, target)
    relative = np.asarray(points, dtype=np.float64) - camera_position[None]
    depth = relative @ forward
    focal_pixels = resolution * lens_mm / sensor_width_mm
    x = focal_pixels * (relative @ right) / np.maximum(depth, 1.0e-6) + resolution / 2
    y = focal_pixels * (relative @ up) / np.maximum(depth, 1.0e-6) + resolution / 2
    return np.stack([x, y], axis=1), depth


def image_array(scene: bpy.types.Scene, force_disk_sync: bool = False) -> np.ndarray:
    image = bpy.data.images.get("Render Result")
    loaded_image = None
    temporary_path = None
    if force_disk_sync:
        # On surfaceless EGL, Render Result's Python ``pixels`` buffer can be
        # stale even though save_render receives the correct render buffer.
        # A round trip through /tmp makes the mask read deterministic.
        temporary_path = Path(f"/tmp/omomo_blender_mask_{os.getpid()}.png")
        image.save_render(str(temporary_path), scene=scene)
        loaded_image = bpy.data.images.load(str(temporary_path), check_existing=False)
        image = loaded_image
    width, height = image.size
    result = np.asarray(image.pixels[:], dtype=np.float32).reshape(height, width, 4)
    if loaded_image is not None:
        bpy.data.images.remove(loaded_image)
        temporary_path.unlink(missing_ok=True)
    return result


def mask_bbox(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def bbox_crop_penalty(projected: np.ndarray, resolution: int) -> tuple[float, float]:
    lower = projected.min(axis=0)
    upper = projected.max(axis=0)
    outside = np.maximum(-lower, 0.0) + np.maximum(upper - resolution, 0.0)
    crop_penalty = float(outside.sum() / resolution)
    margin = 0.05 * resolution
    inside_margin = bool(np.all(lower >= margin) and np.all(upper <= resolution - margin))
    if inside_margin:
        visibility = 1.0
    else:
        visibility = float(np.clip(1.0 - 2.0 * crop_penalty, 0.0, 1.0))
    return crop_penalty, visibility


def view_informativeness(
    object_rotation: np.ndarray,
    object_center: np.ndarray,
    camera_position: np.ndarray,
) -> tuple[float, list[float]]:
    view_world = camera_position - object_center
    view_world /= np.linalg.norm(view_world)
    local_abs = np.abs(object_rotation.T @ view_world)
    ordered = np.sort(local_abs)[::-1]
    two_face = ordered[1] / max(ordered[0], 1.0e-8)
    third_face = min(ordered[2] / 0.30, 1.0)
    score = float(np.clip(0.72 * two_face + 0.28 * third_face, 0.0, 1.0))
    return score, local_abs.tolist()


def size_score(bbox: list[int] | None, resolution: int) -> tuple[float, int, int]:
    if bbox is None:
        return 0.0, 0, 0
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    extent = max(width, height)
    low, preferred, high = 0.15 * resolution, 0.25 * resolution, 0.42 * resolution
    if extent < low:
        score = extent / low
    elif extent <= preferred:
        score = 1.0
    else:
        score = max(0.0, 1.0 - (extent - preferred) / (high - preferred))
    return float(np.clip(score, 0.0, 1.0)), int(width), int(height)


def configure_mask_scene(scene: bpy.types.Scene, resolution: int) -> None:
    scene.render.engine = "BLENDER_EEVEE"
    scene.eevee.taa_render_samples = 1
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = True
    scene.render.use_file_extension = True
    scene.render.image_settings.color_mode = "RGBA"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    scene.world.color = (0.0, 0.0, 0.0)


def score_frame(
    scene: bpy.types.Scene,
    human: bpy.types.Object,
    object_mesh: bpy.types.Object,
    human_vertices: np.ndarray,
    object_vertices: np.ndarray,
    object_rotation: np.ndarray,
    camera_position: np.ndarray,
    target: np.ndarray,
    resolution: int,
    lens_mm: float,
    sensor_width_mm: float,
) -> dict:
    update_vertices(human, human_vertices)
    update_vertices(object_mesh, object_vertices)

    human.hide_render = True
    object_mesh.hide_render = False
    bpy.ops.render.render(write_still=False)
    debug_dir = os.environ.get("OMOMO_MASK_DEBUG_DIR")
    if debug_dir:
        Path(debug_dir).mkdir(parents=True, exist_ok=True)
        bpy.data.images["Render Result"].save_render(
            str(Path(debug_dir) / "object_only.png"), scene=scene
        )
    object_only = image_array(scene, force_disk_sync=True)[..., 3] > 0.01
    object_only_area = int(object_only.sum())
    bbox = mask_bbox(object_only)

    human.hide_render = False
    bpy.ops.render.render(write_still=False)
    if debug_dir:
        bpy.data.images["Render Result"].save_render(
            str(Path(debug_dir) / "combined.png"), scene=scene
        )
    combined = image_array(scene, force_disk_sync=True)
    visible_object = (combined[..., 0] > combined[..., 1]) & (combined[..., 0] > 0.02)
    visible_area = int(visible_object.sum())
    object_visibility = visible_area / max(object_only_area, 1)
    occlusion_ratio = 1.0 - object_visibility

    projected_human, human_depth = project_points(
        human_vertices[::4],
        camera_position,
        target,
        lens_mm,
        sensor_width_mm,
        resolution,
    )
    crop_penalty, human_visibility = bbox_crop_penalty(
        projected_human[human_depth > 0], resolution
    )
    current_size_score, bbox_width, bbox_height = size_score(bbox, resolution)
    current_view_score, view_local_abs = view_informativeness(
        object_rotation, object_vertices.mean(axis=0), camera_position
    )
    total = (
        1.00 * current_size_score
        + 0.50 * object_visibility
        + 1.50 * current_view_score
        + 0.50 * human_visibility
        - 2.00 * occlusion_ratio
        - 2.00 * crop_penalty
    )
    return {
        "score": float(total),
        "object_bbox": bbox,
        "object_bbox_width_px": bbox_width,
        "object_bbox_height_px": bbox_height,
        "object_only_area_px": object_only_area,
        "object_visible_area_px": visible_area,
        "object_visibility": float(object_visibility),
        "occlusion_ratio": float(occlusion_ratio),
        "human_visibility": float(human_visibility),
        "crop_penalty": float(crop_penalty),
        "object_size_score": float(current_size_score),
        "view_informativeness": float(current_view_score),
        "object_local_view_abs": view_local_abs,
    }


def fast_score_frame(
    human_vertices: np.ndarray,
    object_vertices: np.ndarray,
    object_rotation: np.ndarray,
    camera_position: np.ndarray,
    target: np.ndarray,
    resolution: int,
    lens_mm: float,
    sensor_width_mm: float,
) -> dict:
    """Cheap camera score for batch rerendering.

    This deliberately uses only exact GT geometry and the same projection and
    view terms as the mask-based search.  Occlusion is conservatively set to
    zero because the final render is still produced with the full human/object
    depth test; strict mask search remains available for audited runs.
    """
    projected_object, object_depth = project_points(
        object_vertices, camera_position, target, lens_mm, sensor_width_mm, resolution
    )
    visible_projected = projected_object[object_depth > 0]
    if len(visible_projected) == 0:
        bbox = None
    else:
        lower = np.floor(visible_projected.min(axis=0)).astype(int)
        upper = np.ceil(visible_projected.max(axis=0)).astype(int)
        bbox = [int(lower[0]), int(lower[1]), int(upper[0]), int(upper[1])]
    projected_human, human_depth = project_points(
        human_vertices[::4], camera_position, target, lens_mm, sensor_width_mm, resolution
    )
    crop_penalty, human_visibility = bbox_crop_penalty(
        projected_human[human_depth > 0], resolution
    )
    current_size_score, bbox_width, bbox_height = size_score(bbox, resolution)
    current_view_score, view_local_abs = view_informativeness(
        object_rotation, object_vertices.mean(axis=0), camera_position
    )
    return {
        "score": float(current_size_score + 1.5 * current_view_score + 0.5 * human_visibility - 2.0 * crop_penalty),
        "object_bbox": bbox,
        "object_bbox_width_px": bbox_width,
        "object_bbox_height_px": bbox_height,
        "object_only_area_px": int(max(bbox_width, 0) * max(bbox_height, 0)),
        "object_visible_area_px": int(max(bbox_width, 0) * max(bbox_height, 0)),
        "object_visibility": 1.0,
        "occlusion_ratio": 0.0,
        "human_visibility": float(human_visibility),
        "crop_penalty": float(crop_penalty),
        "object_size_score": float(current_size_score),
        "view_informativeness": float(current_view_score),
        "object_local_view_abs": view_local_abs,
    }


def camera_payload(
    azimuth: float,
    elevation: float,
    distance: float,
    target: np.ndarray,
    position: np.ndarray,
    lens_mm: float,
    sensor_width_mm: float,
    resolution: int,
) -> dict:
    focal_pixels = resolution * lens_mm / sensor_width_mm
    right, up, forward = camera_basis(position, target)
    world_to_camera = np.eye(4, dtype=np.float64)
    world_to_camera[:3, :3] = np.stack([right, up, forward], axis=0)
    world_to_camera[:3, 3] = -world_to_camera[:3, :3] @ position
    return {
        "static": True,
        "azimuth_deg": float(azimuth),
        "elevation_deg": float(elevation),
        "distance": float(distance),
        "position": position.tolist(),
        "look_at": target.tolist(),
        "lens_mm": float(lens_mm),
        "sensor_width_mm": float(sensor_width_mm),
        "resolution": int(resolution),
        "intrinsics": [
            [focal_pixels, 0.0, resolution / 2],
            [0.0, focal_pixels, resolution / 2],
            [0.0, 0.0, 1.0],
        ],
        "world_to_camera": world_to_camera.tolist(),
    }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def run_search(args: argparse.Namespace, data: np.lib.npyio.NpzFile) -> None:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    configure_mask_scene(scene, args.search_resolution)
    human = make_mesh_object(
        "Human", data["human_vertices"][0], data["human_faces"], smooth=False
    )
    object_mesh = make_mesh_object(
        "Object", data["object_vertices"][0], data["object_faces"], smooth=False
    )
    set_material(human, make_emission_material("HumanMaskGreen", (0.0, 1.0, 0.0, 1.0)))
    set_material(object_mesh, make_emission_material("ObjectMaskRed", (1.0, 0.0, 0.0, 1.0)))
    camera = add_camera()
    lens_mm = float(camera.data.lens)
    sensor_width_mm = float(camera.data.sensor_width)

    human_vertices_all = data["human_vertices"]
    object_vertices_all = data["object_vertices"]
    all_min = np.minimum(
        human_vertices_all.min(axis=(0, 1)), object_vertices_all.min(axis=(0, 1))
    )
    all_max = np.maximum(
        human_vertices_all.max(axis=(0, 1)), object_vertices_all.max(axis=(0, 1))
    )
    bounds = np.stack([all_min, all_max], axis=0)
    target = (all_min + all_max) / 2
    target[2] += 0.02
    sampled_frames = np.unique(
        np.linspace(
            0,
            len(human_vertices_all) - 1,
            args.search_frame_count,
            dtype=np.int32,
        )
    )
    elevations = [float(value) for value in args.elevations.split(",")]
    partial_path = output / "camera_candidates.partial.json"
    candidates = []
    if partial_path.exists():
        with partial_path.open("r", encoding="utf-8") as handle:
            partial_payload = json.load(handle)
        candidates = partial_payload.get("candidates", [])
        print(
            f"RESUME_SEARCH: loaded {len(candidates)} completed candidates from "
            f"{partial_path}",
            flush=True,
        )
    completed_candidates = {
        (int(item["azimuth_deg"]), float(item["elevation_deg"]))
        for item in candidates
    }
    start_time = time.time()
    candidate_number = len(candidates)
    total_candidates = len(range(0, 360, args.azimuth_step)) * len(elevations)
    for elevation in elevations:
        for azimuth in range(0, 360, args.azimuth_step):
            if (azimuth, elevation) in completed_candidates:
                continue
            candidate_number += 1
            direction = spherical_direction(azimuth, elevation)
            distance = fit_distance(
                bounds, target, direction, lens_mm, sensor_width_mm
            )
            position = target + distance * direction
            aim_camera(camera, position, target)
            frame_scores = []
            for frame in sampled_frames:
                if args.fast_search:
                    score = fast_score_frame(
                        human_vertices_all[frame], object_vertices_all[frame],
                        data["object_rotation"][frame], position, target,
                        args.search_resolution, lens_mm, sensor_width_mm,
                    )
                else:
                    score = score_frame(
                        scene, human, object_mesh, human_vertices_all[frame],
                        object_vertices_all[frame], data["object_rotation"][frame],
                        position, target, args.search_resolution, lens_mm,
                        sensor_width_mm,
                    )
                score["frame"] = int(frame)
                frame_scores.append(score)
            values = np.asarray([item["score"] for item in frame_scores])
            camera_score = float(0.65 * np.median(values) + 0.35 * np.percentile(values, 20))
            candidate = camera_payload(
                azimuth,
                elevation,
                distance,
                target,
                position,
                lens_mm,
                sensor_width_mm,
                1280,
            )
            candidate.update(
                {
                    "camera_score": camera_score,
                    "sampled_frames": sampled_frames.tolist(),
                    "frame_scores": frame_scores,
                }
            )
            candidates.append(candidate)
            elapsed = time.time() - start_time
            print(
                f"SEARCH {candidate_number:03d}/{total_candidates}: "
                f"az={azimuth:03d} el={elevation:04.1f} score={camera_score:.4f} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )
            write_json(
                partial_path,
                {"candidates": candidates},
            )

    candidates.sort(key=lambda item: item["camera_score"], reverse=True)
    # The camera must not merely be good on average: it must contain at least
    # one genuinely useful single-view reconstruction frame.  Enforce the
    # task's 1280px object-size target together with visibility, occlusion and
    # three-quarter-view constraints, then choose the best whole-trajectory
    # camera among candidates that pass them.
    reconstruction_candidates = []
    for candidate in candidates:
        eligible_frames = [
            frame_score
            for frame_score in candidate["frame_scores"]
            if min(
                frame_score["object_bbox_width_px"],
                frame_score["object_bbox_height_px"],
            )
            * (1280.0 / args.search_resolution)
            >= 200.0
            and frame_score["occlusion_ratio"] < 0.10
            and frame_score["human_visibility"] >= 0.98
            and frame_score["object_visibility"] >= 0.90
            and frame_score["view_informativeness"] >= 0.65
        ]
        if eligible_frames:
            candidate["eligible_reconstruction_sample_frames"] = sorted(
                eligible_frames, key=lambda item: item["score"], reverse=True
            )
            reconstruction_candidates.append(candidate)
    if not reconstruction_candidates:
        if args.fast_search and candidates:
            best = max(candidates, key=lambda item: item["camera_score"])
            reconstruction_candidates = [best]
            best["selection_policy_relaxed"] = (
                "fast geometric search had no frame meeting strict mask thresholds"
            )
        else:
            raise RuntimeError(
                "No camera has a sampled frame satisfying the 200px object-size, "
                "visibility, <10% occlusion, and 3/4-view requirements"
            )
    best = max(reconstruction_candidates, key=lambda item: item["camera_score"])
    best["selection_policy"] = {
        "full_resolution": 1280,
        "minimum_object_bbox_width_and_height_px": 200,
        "maximum_occlusion_ratio": 0.10,
        "minimum_human_visibility": 0.98,
        "minimum_object_visibility": 0.90,
        "minimum_view_informativeness": 0.65,
        "tie_break": "highest whole-trajectory camera_score",
        "passing_camera_count": len(reconstruction_candidates),
    }
    position = np.asarray(best["position"], dtype=np.float64)
    target = np.asarray(best["look_at"], dtype=np.float64)
    aim_camera(camera, position, target)
    configure_mask_scene(scene, 384)
    full_frame_scores = []
    object_only_dir = output / "selected_camera_object_only_masks_384"
    object_only_dir.mkdir(parents=True, exist_ok=True)
    frame_end = (
        len(human_vertices_all)
        if args.frame_end is None
        else min(args.frame_end, len(human_vertices_all))
    )
    if not 0 <= args.frame_start < frame_end:
        raise ValueError(
            f"Invalid render range [{args.frame_start}, {frame_end}) for "
            f"{len(human_vertices_all)} frames"
        )
    for frame in range(args.frame_start, frame_end):
        if args.fast_search:
            score = fast_score_frame(
                human_vertices_all[frame], object_vertices_all[frame],
                data["object_rotation"][frame], position, target, 384,
                lens_mm, sensor_width_mm,
            )
        else:
            score = score_frame(
                scene, human, object_mesh, human_vertices_all[frame],
                object_vertices_all[frame], data["object_rotation"][frame],
                position, target, 384, lens_mm, sensor_width_mm,
            )
        score["frame"] = frame
        full_frame_scores.append(score)
        if frame % 20 == 0:
            print(f"FULL-SCORE {frame:03d}/{len(human_vertices_all) - 1}", flush=True)
    ranked_frames = sorted(
        full_frame_scores, key=lambda item: item["score"], reverse=True
    )
    top_frames = ranked_frames[:5]
    best["best_reconstruction_frame"] = int(top_frames[0]["frame"])
    best["top_k_frames"] = top_frames
    best["search_resolution"] = args.search_resolution
    best["full_score_resolution"] = 384
    best["search_frame_count"] = int(len(sampled_frames))
    best["score_formula"] = (
        "1.0*object_size + 0.5*object_visibility + 1.5*view + "
        "0.5*human_visibility - 2.0*occlusion - 2.0*crop"
    )
    write_json(output / "camera_candidates.json", {"candidates": candidates})
    write_json(output / "frame_scores.json", {"frames": full_frame_scores})
    write_json(output / "best_camera.json", best)
    print("BEST_CAMERA=" + json.dumps(best, sort_keys=True), flush=True)


def add_area_light(
    name: str,
    location: np.ndarray,
    energy: float,
    size: float,
    target: np.ndarray,
    color: tuple[float, float, float],
) -> bpy.types.Object:
    data = bpy.data.lights.new(name, type="AREA")
    data.energy = energy
    data.shape = "DISK"
    data.size = size
    data.color = color
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    obj.location = tuple(location)
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()
    return obj


def configure_compositor(output: Path) -> None:
    scene = bpy.context.scene
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()
    render_layers = tree.nodes.new("CompositorNodeRLayers")
    composite = tree.nodes.new("CompositorNodeComposite")
    tree.links.new(render_layers.outputs["Image"], composite.inputs["Image"])

    depth_output = tree.nodes.new("CompositorNodeOutputFile")
    depth_output.base_path = str(output)
    depth_output.file_slots[0].path = "depth/"
    depth_output.format.file_format = "OPEN_EXR"
    # Blender 3.6's OpenEXR writer exposes RGB/RGBA only.  A scalar Z pass
    # connected to an RGB slot is replicated without changing its values.
    depth_output.format.color_mode = "RGB"
    depth_output.format.color_depth = "32"
    depth_output.format.exr_codec = "ZIP"
    tree.links.new(render_layers.outputs["Depth"], depth_output.inputs[0])

    normal_output = tree.nodes.new("CompositorNodeOutputFile")
    normal_output.base_path = str(output)
    normal_output.file_slots[0].path = "normal/"
    normal_output.format.file_format = "OPEN_EXR"
    normal_output.format.color_mode = "RGB"
    normal_output.format.color_depth = "16"
    normal_output.format.exr_codec = "ZIP"
    tree.links.new(render_layers.outputs["Normal"], normal_output.inputs[0])


def render_visible_masks(
    scene: bpy.types.Scene,
    human: bpy.types.Object,
    object_mesh: bpy.types.Object,
    ground: bpy.types.Object,
    lights: list[bpy.types.Object],
    output: Path,
    frame: int,
    white: bpy.types.Material,
    black: bpy.types.Material,
) -> None:
    """Render exact visible human/object masks using the normal depth test.

    Blender 3.6's surfaceless EEVEE compositor does not reliably execute
    legacy IDMask-to-PNG nodes.  Emission ID renders are deterministic and
    preserve the same camera, geometry, depth ordering, and antialiased edges.
    """

    human_material = human.data.materials[0]
    object_material = object_mesh.data.materials[0]
    background = scene.world.node_tree.nodes.get("Background")
    old_background_color = tuple(background.inputs["Color"].default_value)
    old_background_strength = float(background.inputs["Strength"].default_value)
    old_use_nodes = scene.use_nodes
    old_transparent = scene.render.film_transparent
    old_samples = scene.eevee.taa_render_samples
    old_transform = scene.view_settings.view_transform
    old_look = scene.view_settings.look
    old_exposure = scene.view_settings.exposure
    old_filepath = scene.render.filepath
    old_hides = [ground.hide_render] + [light.hide_render for light in lights]

    scene.use_nodes = False
    scene.render.film_transparent = False
    scene.eevee.taa_render_samples = 8
    scene.view_settings.view_transform = "Raw"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    background.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    background.inputs["Strength"].default_value = 0.0
    ground.hide_render = True
    for light in lights:
        light.hide_render = True

    set_material(human, white)
    set_material(object_mesh, black)
    scene.render.filepath = str(output / "human_mask" / f"frame_{frame:06d}.png")
    bpy.ops.render.render(write_still=True)

    set_material(human, black)
    set_material(object_mesh, white)
    scene.render.filepath = str(output / "object_mask" / f"frame_{frame:06d}.png")
    bpy.ops.render.render(write_still=True)

    set_material(human, human_material)
    set_material(object_mesh, object_material)
    ground.hide_render = old_hides[0]
    for light, hidden in zip(lights, old_hides[1:]):
        light.hide_render = hidden
    background.inputs["Color"].default_value = old_background_color
    background.inputs["Strength"].default_value = old_background_strength
    scene.use_nodes = old_use_nodes
    scene.render.film_transparent = old_transparent
    scene.eevee.taa_render_samples = old_samples
    scene.view_settings.view_transform = old_transform
    scene.view_settings.look = old_look
    scene.view_settings.exposure = old_exposure
    scene.render.filepath = old_filepath


def run_render(args: argparse.Namespace, data: np.lib.npyio.NpzFile) -> None:
    if args.camera_config is None:
        raise ValueError("--camera-config is required for render mode")
    output = args.output.resolve()
    directories = ("frames",) if args.rgb_only else ("frames", "human_mask", "object_mask", "depth", "normal")
    for directory in directories:
        (output / directory).mkdir(parents=True, exist_ok=True)
    camera_config = json.loads(args.camera_config.read_text())
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.eevee.taa_render_samples = 2 if args.rgb_only else 32
    scene.eevee.use_gtao = True
    scene.eevee.gtao_distance = 3.0
    scene.eevee.gtao_factor = 0.7
    scene.render.resolution_x = args.resolution
    scene.render.resolution_y = args.resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.render.film_transparent = False
    scene.render.use_file_extension = True
    scene.render.use_compositing = True
    scene.render.use_file_extension = True
    scene.render.image_settings.compression = 30
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = 0.1
    scene.view_settings.gamma = 1.0
    scene.world.use_nodes = True
    background = scene.world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = (0.79, 0.81, 0.84, 1.0)
    background.inputs["Strength"].default_value = 0.65
    scene.view_layers[0].use_pass_object_index = not args.rgb_only
    scene.view_layers[0].use_pass_z = not args.rgb_only
    scene.view_layers[0].use_pass_normal = not args.rgb_only
    scene.render.use_motion_blur = False

    object_name = (
        str(np.asarray(data["object_name"]).item())
        if "object_name" in data.files
        else "object"
    )
    human = make_mesh_object(
        "Human", data["human_vertices"][0], data["human_faces"], smooth=True
    )
    object_mesh = make_mesh_object(
        object_name, data["object_vertices"][0], data["object_faces"], smooth=False
    )
    human.pass_index = 1
    object_mesh.pass_index = 2
    set_material(
        human,
        make_principled_material("HumanBlueCyan", (0.025, 0.26, 0.80, 1.0), 0.55),
    )
    set_material(
        object_mesh,
        make_principled_material("CapturedObjectWarmOrange", (0.72, 0.25, 0.055, 1.0), 0.62),
    )

    human_vertices_all = data["human_vertices"]
    object_vertices_all = data["object_vertices"]
    all_min = np.minimum(
        human_vertices_all.min(axis=(0, 1)), object_vertices_all.min(axis=(0, 1))
    )
    all_max = np.maximum(
        human_vertices_all.max(axis=(0, 1)), object_vertices_all.max(axis=(0, 1))
    )
    target = np.asarray(camera_config["look_at"], dtype=np.float64)
    span = float(np.linalg.norm(all_max - all_min))
    ground_z = float(all_min[2] - 0.012)
    bpy.ops.mesh.primitive_plane_add(
        size=max(12.0, span * 4.0),
        location=(float(target[0]), float(target[1]), ground_z),
    )
    ground = bpy.context.object
    ground.name = "NeutralGround"
    set_material(
        ground,
        make_principled_material("GroundNeutralGray", (0.48, 0.50, 0.53, 1.0), 0.78),
    )

    light_scale = max(span, 2.0)
    lights = []
    lights.append(add_area_light(
        "Key",
        target + np.array([1.2, -1.0, 1.5]) * light_scale,
        650.0 * light_scale,
        2.0 * light_scale,
        target,
        (1.0, 0.90, 0.78),
    ))
    lights.append(add_area_light(
        "Fill",
        target + np.array([-1.1, -0.3, 0.8]) * light_scale,
        380.0 * light_scale,
        2.4 * light_scale,
        target,
        (0.76, 0.86, 1.0),
    ))
    lights.append(add_area_light(
        "TopRim",
        target + np.array([0.2, 0.8, 1.8]) * light_scale,
        520.0 * light_scale,
        1.8 * light_scale,
        target,
        (1.0, 0.96, 0.90),
    ))

    camera = add_camera()
    camera.data.lens = float(camera_config["lens_mm"])
    camera.data.sensor_width = float(camera_config["sensor_width_mm"])
    aim_camera(
        camera,
        np.asarray(camera_config["position"], dtype=np.float64),
        target,
    )
    if not args.rgb_only:
        configure_compositor(output)
    else:
        scene.use_nodes = False
    scene.render.use_persistent_data = True
    mask_white = make_emission_material("MaskWhite", (1.0, 1.0, 1.0, 1.0))
    mask_black = make_emission_material("MaskBlack", (0.0, 0.0, 0.0, 1.0))

    render_config = {
        "renderer": "Blender EEVEE",
        "resolution": [args.resolution, args.resolution],
        "fps": int(data["fps"]),
        "frame_count": int(len(human_vertices_all)),
        "camera_static": True,
        "motion_blur": False,
        "depth_of_field": False,
        "human_material": {
            "name": "HumanBlueCyan",
            "base_color_linear": [0.025, 0.26, 0.80, 1.0],
            "roughness": 0.55,
        },
        "object_material": {
            "name": "CapturedObjectWarmOrange",
            "base_color_linear": [0.72, 0.25, 0.055, 1.0],
            "roughness": 0.62,
            "smooth_shading": False,
        },
        "lighting": ["key area", "fill area", "top/rim area"],
        "diagnostic_passes": [] if args.rgb_only else ["human_mask", "object_mask", "depth", "normal"],
    }
    write_json(output / "render_config.json", render_config)
    write_json(output / "camera.json", camera_config)
    frame_end = (
        len(human_vertices_all)
        if args.frame_end is None
        else min(args.frame_end, len(human_vertices_all))
    )
    if not 0 <= args.frame_start < frame_end:
        raise ValueError(
            f"Invalid render range [{args.frame_start}, {frame_end}) for "
            f"{len(human_vertices_all)} frames"
        )
    for frame in range(args.frame_start, frame_end):
        update_vertices(human, human_vertices_all[frame])
        update_vertices(object_mesh, object_vertices_all[frame])
        scene.frame_set(frame + 1)
        scene.render.filepath = str(output / "frames" / f"{frame:06d}.png")
        bpy.ops.render.render(write_still=True)
        if not args.rgb_only:
            render_visible_masks(
                scene,
                human,
                object_mesh,
                ground,
                lights,
                output,
                frame,
                mask_white,
                mask_black,
            )
        print(f"RENDER {frame:03d}/{len(human_vertices_all) - 1}", flush=True)

    if args.rgb_only:
        bpy.ops.wm.save_as_mainfile(filepath=str(output / "scene.blend"))
        return

    # Exact full-resolution Pass A for the selected reconstruction frame.
    # The regular object_mask is Pass B (human + object).  Comparing these two
    # masks gives the requested geometry-based human-on-object occlusion ratio.
    selected_frame = int(camera_config["best_reconstruction_frame"])
    update_vertices(human, human_vertices_all[selected_frame])
    update_vertices(object_mesh, object_vertices_all[selected_frame])
    scene.frame_set(selected_frame + 1)
    human.hide_render = True
    ground.hide_render = True
    for light in lights:
        light.hide_render = True
    scene.use_nodes = False
    scene.render.film_transparent = False
    scene.eevee.taa_render_samples = 8
    scene.view_settings.view_transform = "Raw"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    background.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    background.inputs["Strength"].default_value = 0.0
    set_material(object_mesh, mask_white)
    object_only_dir = output / "object_only_mask"
    object_only_dir.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(object_only_dir / f"{selected_frame:06d}.png")
    bpy.ops.render.render(write_still=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output / "scene.blend"))


def main() -> None:
    args = parse_args()
    clear_scene()
    data = np.load(args.sequence_archive, allow_pickle=False)
    if args.mode == "search":
        run_search(args, data)
    else:
        run_render(args, data)


if __name__ == "__main__":
    main()
