#!/usr/bin/env python3
"""Render two cached retarget trajectories with a shared contact-focused camera."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from holosoma_retargeting.examples.render_mujoco_trajectory_comparison import (
    RawVideoEncoder,
)
from holosoma_retargeting.semantic_keyframes.contact_targets import (
    load_retarget_contact_targets,
)


PART_BODIES = {
    "left_hand": "left_rubber_hand_link",
    "right_hand": "right_rubber_hand_link",
    "left_elbow": "left_elbow_link",
    "right_elbow": "right_elbow_link",
    "left_foot": "left_ankle_roll_link",
    "right_foot": "right_ankle_roll_link",
}
PART_COLORS = {
    "left_hand": (255, 170, 45),
    "right_hand": (45, 220, 105),
    "left_elbow": (195, 125, 245),
    "right_elbow": (255, 145, 35),
    "left_foot": (255, 170, 45),
    "right_foot": (45, 220, 105),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    suffix = "-Bold" if bold else ""
    path = Path(f"/usr/share/fonts/truetype/dejavu/DejaVuSans{suffix}.ttf")
    if path.exists():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _load_qpos(path: Path) -> tuple[np.ndarray, int]:
    with np.load(path, allow_pickle=False) as data:
        qpos = np.asarray(data["qpos"], dtype=np.float64)
        fps = int(np.asarray(data["fps"]).item())
    if qpos.ndim != 2 or not np.isfinite(qpos).all():
        raise ValueError(f"invalid qpos trajectory: {path}")
    return qpos, fps


def _body_id(model: mujoco.MjModel, name: str) -> int:
    result = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if result < 0:
        raise ValueError(f"body not found: {name}")
    return int(result)


def _collision_geoms(model: mujoco.MjModel, body_id: int) -> tuple[int, ...]:
    result = tuple(
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) == body_id
        and (
            int(model.geom_contype[geom_id]) != 0
            or int(model.geom_conaffinity[geom_id]) != 0
        )
    )
    if not result:
        raise ValueError(f"body {body_id} has no collision geoms")
    return result


def _object_geoms(model: mujoco.MjModel) -> tuple[int, ...]:
    free_joints = np.flatnonzero(
        model.jnt_type == int(mujoco.mjtJoint.mjJNT_FREE)
    )
    if len(free_joints) < 2:
        raise ValueError("expected a dynamic object free joint")
    body_id = int(model.jnt_bodyid[int(free_joints[-1])])
    root_id = int(model.body_rootid[body_id])
    result = tuple(
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.body_rootid[int(model.geom_bodyid[geom_id])]) == root_id
        and (
            int(model.geom_contype[geom_id]) != 0
            or int(model.geom_conaffinity[geom_id]) != 0
        )
    )
    if not result:
        raise ValueError("dynamic object has no collision geoms")
    return result


def _tint_body(
    model: mujoco.MjModel,
    body_id: int,
    rgb: tuple[int, int, int],
) -> None:
    rgba = np.asarray((*rgb, 255), dtype=np.float32) / 255.0
    for geom_id in range(model.ngeom):
        if int(model.geom_bodyid[geom_id]) == body_id:
            model.geom_rgba[geom_id] = rgba


def _set_qpos(model: mujoco.MjModel, data: mujoco.MjData, qpos: np.ndarray) -> None:
    if qpos.shape != (model.nq,):
        raise ValueError(f"qpos {qpos.shape} does not match model nq={model.nq}")
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def _surface_distance(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    robot_geoms: tuple[int, ...],
    object_geoms: tuple[int, ...],
) -> float:
    best = np.inf
    for robot_geom in robot_geoms:
        for object_geom in object_geoms:
            from_to = np.zeros(6, dtype=np.float64)
            distance = float(
                mujoco.mj_geomDistance(
                    model,
                    data,
                    robot_geom,
                    object_geom,
                    10.0,
                    from_to,
                )
            )
            best = min(best, distance)
    if not np.isfinite(best):
        raise RuntimeError("failed to measure robot-object surface distance")
    return float(best)


def _camera(
    original_data: mujoco.MjData,
    final_data: mujoco.MjData,
    object_body: int,
    part_bodies: dict[str, int],
    args: argparse.Namespace,
) -> mujoco.MjvCamera:
    object_center = 0.5 * (
        original_data.xpos[object_body] + final_data.xpos[object_body]
    )
    part_center = np.mean([
        0.5 * (original_data.xpos[body] + final_data.xpos[body])
        for body in part_bodies.values()
    ], axis=0)
    lookat = 0.75 * object_center + 0.25 * part_center
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth = args.azimuth
    camera.elevation = args.elevation
    camera.distance = args.distance
    camera.lookat[:] = lookat
    return camera


def _distance_color(distance_m: float, active: bool) -> tuple[int, int, int, int]:
    if not active:
        return (170, 178, 188, 255)
    if distance_m < -0.001:
        return (255, 90, 85, 255)
    if distance_m <= 0.02:
        return (75, 225, 125, 255)
    if distance_m <= 0.03:
        return (255, 190, 60, 255)
    return (255, 90, 85, 255)


def _palm_measurements(data, object_body, part_bodies, contacts, frame_index):
    if contacts.robot_points_local is None:
        return {}
    object_rotation = data.xmat[object_body].reshape(3, 3)
    result = {}
    for index, part in enumerate(contacts.parts):
        body = part_bodies[part]
        rotation = data.xmat[body].reshape(3, 3)
        point = object_rotation.T @ (
            data.xpos[body] + rotation @ contacts.robot_points_local[index] - data.xpos[object_body]
        )
        normal = object_rotation.T @ rotation @ contacts.robot_normals_local[index]
        target = -contacts.object_normals_local[frame_index, index]
        result[part] = {
            "normal_error_deg": float(np.degrees(np.arccos(np.clip(normal @ target, -1, 1)))),
            "anchor_error_m": float(np.linalg.norm(point - contacts.object_points_local[frame_index, index])),
        }
    return result


def _annotate(
    frame: np.ndarray,
    *,
    title: str,
    frame_index: int,
    frame_count: int,
    fps: int,
    distances: dict[str, float],
    active: dict[str, bool],
    accent: tuple[int, int, int],
    palm: dict | None = None,
) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    width, height = image.size
    bottom = 68 + 29 * len(distances)
    draw.rounded_rectangle((16, 14, width - 16, bottom), radius=12, fill=(8, 12, 18, 215))
    draw.rectangle((16, 14, 27, bottom), fill=(*accent, 255))
    draw.text((43, 24), title, font=_font(25, bold=True), fill=(245, 248, 252, 255))
    labels = [(part, part.replace("_", " ").capitalize()) for part in distances]
    for row, (part, label) in enumerate(labels):
        y = 62 + 29 * row
        status = f"{1000.0 * distances[part]:5.1f} mm" if active[part] else "not active"
        if palm and active[part]:
            status += f" | palm normal {palm[part]['normal_error_deg']:.1f} deg"
        draw.ellipse((44, y + 3, 57, y + 16), fill=(*PART_COLORS[part], 255))
        draw.text((66, y), f"{label}: {status}", font=_font(19, bold=True), fill=_distance_color(distances[part], active[part]))
    footer = f"Frame {frame_index:03d}/{frame_count - 1:03d}  {frame_index / fps:4.2f}s  |  gap + / overlap - (mm)"
    draw.rounded_rectangle((16, height - 49, width - 16, height - 14), radius=9, fill=(8, 12, 18, 195))
    draw.text((28, height - 43), footer, font=_font(17), fill=(238, 242, 248, 255))
    return np.asarray(image)


def _contact_sheet(frames: list[np.ndarray], output: Path) -> None:
    if not frames:
        return
    target_width = 1200
    resized = []
    for frame in frames:
        image = Image.fromarray(frame)
        target_height = round(image.height * target_width / image.width)
        resized.append(image.resize((target_width, target_height), Image.Resampling.LANCZOS))
    sheet = Image.new("RGB", (target_width, sum(image.height for image in resized)), (15, 18, 24))
    y = 0
    for image in resized:
        sheet.paste(image, (0, y))
        y += image.height
    sheet.save(output, quality=90)


def render(args: argparse.Namespace) -> None:
    original_path = args.original.resolve()
    final_path = args.final.resolve()
    model_path = args.model.resolve()
    contacts_path = args.contacts.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if not 0.0 < args.playback_speed <= 1.0:
        raise ValueError("playback_speed must be in (0, 1]")

    original_qpos, original_fps = _load_qpos(original_path)
    final_qpos, final_fps = _load_qpos(final_path)
    if original_qpos.shape != final_qpos.shape or original_fps != final_fps:
        raise ValueError("trajectory shapes/FPS must match")
    contacts = load_retarget_contact_targets(
        contacts_path,
        expected_frames=len(original_qpos),
    )
    unknown = set(contacts.parts).difference(PART_BODIES)
    if unknown:
        raise ValueError(f"unsupported visualization parts: {sorted(unknown)}")
    part_indices = {part: index for index, part in enumerate(contacts.parts)}

    model = mujoco.MjModel.from_xml_path(str(model_path))
    if model.nq != original_qpos.shape[1]:
        raise ValueError("trajectory qpos does not match model")
    model.vis.global_.offwidth = args.width
    model.vis.global_.offheight = args.height
    part_bodies = {part: _body_id(model, PART_BODIES[part]) for part in contacts.parts}
    part_geoms = {part: _collision_geoms(model, body) for part, body in part_bodies.items()}
    object_geoms = _object_geoms(model)
    object_body = int(model.geom_bodyid[object_geoms[0]])
    for part, body in part_bodies.items():
        _tint_body(model, body, PART_COLORS[part])
    _tint_body(model, object_body, (65, 135, 235))

    original_data = mujoco.MjData(model)
    final_data = mujoco.MjData(model)
    preview_indices = set(args.preview_frames)
    preview_frames: list[np.ndarray] = []
    encoder = None if args.preview_only else RawVideoEncoder(
        output,
        width=2 * args.width,
        height=args.height,
        fps=max(1, round(original_fps * args.playback_speed)),
    )
    distance_rows: list[dict[str, object]] = []
    try:
        with mujoco.Renderer(model, height=args.height, width=args.width) as renderer:
            frames = sorted(preview_indices) if args.preview_only else range(len(original_qpos))
            for frame_index in frames:
                _set_qpos(model, original_data, original_qpos[frame_index])
                _set_qpos(model, final_data, final_qpos[frame_index])
                camera = _camera(
                    original_data,
                    final_data,
                    object_body,
                    part_bodies,
                    args,
                )
                active = {
                    part: bool(contacts.active[frame_index, index])
                    for part, index in part_indices.items()
                }
                original_distances = {
                    part: _surface_distance(model, original_data, geoms, object_geoms)
                    for part, geoms in part_geoms.items()
                }
                final_distances = {
                    part: _surface_distance(model, final_data, geoms, object_geoms)
                    for part, geoms in part_geoms.items()
                }
                original_palm = _palm_measurements(original_data, object_body, part_bodies, contacts, frame_index)
                final_palm = _palm_measurements(final_data, object_body, part_bodies, contacts, frame_index)

                renderer.update_scene(original_data, camera=camera)
                original_frame = _annotate(
                    renderer.render().copy(),
                    title=args.original_label,
                    frame_index=frame_index,
                    frame_count=len(original_qpos),
                    fps=original_fps,
                    distances=original_distances,
                    active=active,
                    accent=(230, 90, 85),
                    palm=original_palm,
                )
                renderer.update_scene(final_data, camera=camera)
                final_frame = _annotate(
                    renderer.render().copy(),
                    title=args.final_label,
                    frame_index=frame_index,
                    frame_count=len(final_qpos),
                    fps=final_fps,
                    distances=final_distances,
                    active=active,
                    accent=(65, 205, 125),
                    palm=final_palm,
                )
                comparison = np.concatenate((original_frame, final_frame), axis=1)
                if encoder is not None:
                    encoder.write(comparison)
                if frame_index in preview_indices:
                    preview_frames.append(comparison)
                distance_rows.append(
                    {
                        "frame": frame_index,
                        "active": active,
                        "original_distance_m": original_distances,
                        "new_distance_m": final_distances,
                        "original_palm": original_palm,
                        "new_palm": final_palm,
                    }
                )
    finally:
        if encoder is not None:
            encoder.close()

    preview = output.with_suffix(".contact_sheet.jpg")
    _contact_sheet(preview_frames, preview)
    metadata = {
        "playback": "direct qpos assignment + mj_forward; no simulation integration",
        "original": {"path": str(original_path), "sha256": _sha256(original_path)},
        "new": {"path": str(final_path), "sha256": _sha256(final_path)},
        "model": {"path": str(model_path), "sha256": _sha256(model_path)},
        "contacts": {"path": str(contacts_path), "sha256": _sha256(contacts_path)},
        "frames": len(original_qpos),
        "fps": original_fps,
        "video_fps": max(1, round(original_fps * args.playback_speed)),
        "playback_speed": args.playback_speed,
        "camera": {
            "azimuth": args.azimuth,
            "elevation": args.elevation,
            "distance": args.distance,
            "anchor": "shared per-frame object/selected-part center across both trajectories",
        },
        "highlight": {**{part: list(PART_COLORS[part]) for part in contacts.parts}, "object": [65, 135, 235]},
        "distance_definition": "signed MuJoCo collision-surface distance; positive gap, negative overlap",
        "contact_activation": contacts.metadata.get("activation_evidence"),
        "palm_orientation": contacts.metadata.get("orientation_policy"),
        "final_label": args.final_label,
        "original_label": args.original_label,
        "output": None if args.preview_only else str(output),
        "preview": str(preview),
        "distances": distance_rows,
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: metadata[key] for key in ("frames", "fps", "camera", "output", "preview")}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--final", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--contacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--azimuth", type=float, default=145.0)
    parser.add_argument("--elevation", type=float, default=-55.0)
    parser.add_argument("--distance", type=float, default=1.45)
    parser.add_argument("--playback-speed", type=float, default=1.0)
    parser.add_argument("--final-label", default="New B4 + Contact")
    parser.add_argument("--original-label", default="Original OmniRetarget")
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument(
        "--preview-frames",
        type=int,
        nargs="+",
        default=[60, 78, 90, 110, 130, 143],
    )
    render(parser.parse_args())


if __name__ == "__main__":
    main()
