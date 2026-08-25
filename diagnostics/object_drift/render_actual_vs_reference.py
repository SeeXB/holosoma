#!/usr/bin/env python3
"""Render an Isaac Sim policy rollout against its retargeting reference.

The policy rollout is recorded without cameras so rendering cannot alter the
physics execution.  This script then replays both qpos trajectories in the same
MuJoCo model and with exactly the same fixed world camera.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import BinaryIO

os.environ.setdefault("MUJOCO_GL", "osmesa")

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REFERENCE = (
    ROOT
    / "src/holosoma_retargeting/holosoma_retargeting"
    / "benchmark_results_full_event_transition_truncation/rl"
    / "transition_truncated_b4_mj_fps50_w_obj.npz"
)
DEFAULT_MODEL = (
    ROOT
    / "src/holosoma_retargeting/holosoma_retargeting/models/g1"
    / "g1_29dof_w_largebox.xml"
)


class RawVideoEncoder:
    def __init__(self, path: Path, width: int, height: int, fps: int) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg is required")
        path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            ffmpeg,
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(path),
        ]
        self.path = path
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE)
        if self.process.stdin is None:
            raise RuntimeError("ffmpeg stdin is unavailable")
        self.stream: BinaryIO = self.process.stdin

    def write(self, frame: np.ndarray) -> None:
        self.stream.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())

    def close(self) -> None:
        self.stream.close()
        return_code = self.process.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg failed for {self.path}: {return_code}")


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    suffix = "-Bold" if bold else ""
    path = Path(f"/usr/share/fonts/truetype/dejavu/DejaVuSans{suffix}.ttf")
    if path.exists():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def xyzw_to_wxyz(quat: np.ndarray) -> np.ndarray:
    return quat[..., [3, 0, 1, 2]]


def build_qpos(
    actual: np.lib.npyio.NpzFile, reference: np.lib.npyio.NpzFile
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    metadata = json.loads(str(actual["_metadata_json"]))
    actual_joint_names = list(metadata["dof_names"])
    reference_joint_names = list(reference["joint_names"])
    if set(actual_joint_names) != set(reference_joint_names):
        raise ValueError("Actual/reference joint-name sets differ")
    actual_order = [actual_joint_names.index(name) for name in reference_joint_names]

    motion_steps = np.asarray(actual["motion_step"], dtype=np.int64)
    # Use the first monotonic pre-wrap segment.  The recorder starts after the
    # first control step, hence this normally corresponds to motion frames 2..324.
    wrap = np.flatnonzero(np.diff(motion_steps) < 0)
    stop = int(wrap[0] + 1) if len(wrap) else len(motion_steps)
    indices = np.arange(stop)
    steps = motion_steps[indices]

    # Instrumented evaluations expose the state immediately before reset.  Use
    # it when present so the terminal frame is not replaced by the next
    # episode's freshly reset state.
    root_pos_key = "pre_root_pos" if "pre_root_pos" in actual else "root_pos"
    root_quat_key = "pre_root_quat_xyzw" if "pre_root_quat_xyzw" in actual else "root_quat_xyzw"
    dof_pos_key = "pre_dof_pos" if "pre_dof_pos" in actual else "dof_pos"

    actual_qpos = np.empty((len(indices), 43), dtype=np.float64)
    actual_qpos[:, :3] = actual[root_pos_key][indices]
    actual_qpos[:, 3:7] = xyzw_to_wxyz(actual[root_quat_key][indices])
    actual_qpos[:, 7:36] = actual[dof_pos_key][indices][:, actual_order]
    actual_qpos[:, 36:39] = actual["object_pos_w"][indices]
    actual_qpos[:, 39:43] = xyzw_to_wxyz(actual["object_quat_xyzw"][indices])

    reference_qpos = np.empty_like(actual_qpos)
    reference_qpos[:, :36] = reference["joint_pos"][steps, :36]
    reference_qpos[:, 36:39] = reference["object_pos_w"][steps]
    reference_qpos[:, 39:43] = reference["object_quat_w"][steps]
    return actual_qpos, reference_qpos, indices, metadata


def camera() -> mujoco.MjvCamera:
    result = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(result)
    result.type = mujoco.mjtCamera.mjCAMERA_FREE
    result.azimuth = 142.0
    result.elevation = -20.0
    result.distance = 3.55
    # Fixed in world coordinates; it does not follow either trajectory.
    result.lookat[:] = [0.53, -0.13, 0.55]
    return result


def render_qpos(
    renderer: mujoco.Renderer,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos: np.ndarray,
    fixed_camera: mujoco.MjvCamera,
) -> np.ndarray:
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=fixed_camera)
    return renderer.render().copy()


def annotate_panel(
    frame: np.ndarray,
    title: str,
    accent: tuple[int, int, int],
    step: int,
    seconds: float,
    delta_y: float | None = None,
) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    width, height = image.size
    draw.rounded_rectangle((14, 12, width - 14, 70), 10, fill=(7, 11, 17, 215))
    draw.rectangle((14, 12, 24, 70), fill=(*accent, 255))
    draw.text((37, 22), title, font=font(23, bold=True), fill=(248, 250, 252, 255))
    status = f"motion frame {step:03d}   t={seconds:4.2f}s"
    draw.rounded_rectangle((14, height - 49, 310, height - 13), 8, fill=(7, 11, 17, 195))
    draw.text((26, height - 44), status, font=font(17), fill=(238, 242, 247, 255))
    if delta_y is not None:
        status = f"torso actual-ref y: {delta_y * 100:+.1f} cm"
        bbox = draw.textbbox((0, 0), status, font=font(17, bold=True))
        box_width = bbox[2] - bbox[0] + 24
        color = (215, 60, 55, 220) if abs(delta_y) >= 0.15 else (26, 121, 82, 210)
        draw.rounded_rectangle(
            (width - box_width - 14, height - 49, width - 14, height - 13),
            8,
            fill=color,
        )
        draw.text(
            (width - box_width, height - 44),
            status,
            font=font(17, bold=True),
            fill=(255, 255, 255, 255),
        )
    return np.asarray(image)


def topdown_frame(
    i: int,
    steps: np.ndarray,
    ref_torso: np.ndarray,
    act_torso: np.ndarray,
    ref_object: np.ndarray,
    act_object: np.ndarray,
) -> np.ndarray:
    width, height = 960, 720
    image = Image.new("RGB", (width, height), (245, 247, 250))
    draw = ImageDraw.Draw(image, "RGBA")
    left, top, right, bottom = 108, 94, 882, 640
    x_min, x_max = 0.15, 0.85
    y_min, y_max = -1.32, 0.98

    def pixel(points: np.ndarray) -> list[tuple[int, int]]:
        px = left + (points[:, 0] - x_min) / (x_max - x_min) * (right - left)
        py = bottom - (points[:, 1] - y_min) / (y_max - y_min) * (bottom - top)
        return [(int(x), int(y)) for x, y in zip(px, py)]

    draw.text((36, 23), "World XY trajectories (fixed coordinates)", font=font(30, bold=True), fill=(25, 32, 42))
    draw.rectangle((left, top, right, bottom), fill=(255, 255, 255, 255), outline=(79, 88, 102, 255), width=2)
    for x_value in np.arange(0.2, 0.81, 0.1):
        x = pixel(np.array([[x_value, y_min]]))[0][0]
        draw.line((x, top, x, bottom), fill=(215, 220, 228, 255), width=1)
        draw.text((x - 16, bottom + 9), f"{x_value:.1f}", font=font(15), fill=(70, 78, 89))
    for y_value in np.arange(-1.2, 1.0, 0.2):
        y = pixel(np.array([[x_min, y_value]]))[0][1]
        draw.line((left, y, right, y), fill=(215, 220, 228, 255), width=1)
        draw.text((55, y - 9), f"{y_value:+.1f}", font=font(15), fill=(70, 78, 89))
    draw.text(((left + right) // 2 - 45, 677), "world x (m)", font=font(20, bold=True), fill=(45, 53, 64))
    draw.text((19, 344), "world y (m)", font=font(20, bold=True), fill=(45, 53, 64))

    colors = {
        "ref_torso": (36, 144, 88, 255),
        "act_torso": (237, 126, 33, 255),
        "ref_object": (39, 125, 196, 255),
        "act_object": (207, 57, 57, 255),
    }
    paths = [
        (ref_torso, colors["ref_torso"]),
        (act_torso, colors["act_torso"]),
        (ref_object, colors["ref_object"]),
        (act_object, colors["act_object"]),
    ]
    for points, color in paths:
        trace = pixel(points[: i + 1])
        if len(trace) > 1:
            draw.line(trace, fill=color, width=4)
        x, y = trace[-1]
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=color, outline=(255, 255, 255, 255), width=2)

    # Error connectors at the current frame show that the mismatch is a
    # world-position offset, not merely a visual difference in joint pose.
    for ref_points, act_points in ((ref_torso, act_torso), (ref_object, act_object)):
        a = pixel(act_points[i : i + 1])[0]
        b = pixel(ref_points[i : i + 1])[0]
        draw.line((*a, *b), fill=(40, 45, 52, 210), width=2)

    legends = [
        ("Reference torso", colors["ref_torso"]),
        ("Actual torso", colors["act_torso"]),
        ("Reference box", colors["ref_object"]),
        ("Actual box", colors["act_object"]),
    ]
    x = 365
    for label, color in legends:
        draw.line((x, 72, x + 24, 72), fill=color, width=5)
        draw.text((x + 31, 60), label, font=font(16), fill=(38, 45, 55))
        x += 142
    torso_dy = (act_torso[i, 1] - ref_torso[i, 1]) * 100
    box_dy = (act_object[i, 1] - ref_object[i, 1]) * 100
    status = (
        f"frame {steps[i]:03d} / 324   t={steps[i] / 50:4.2f}s    "
        f"actual-reference y: torso {torso_dy:+.1f} cm, box {box_dy:+.1f} cm"
    )
    draw.rounded_rectangle((167, 644, 902, 680), 8, fill=(24, 30, 39, 230))
    draw.text((181, 651), status, font=font(16, bold=True), fill=(255, 255, 255))
    return np.asarray(image)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actual", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "diagnostics/object_drift/videos")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    actual = np.load(args.actual.resolve(), allow_pickle=False)
    reference = np.load(args.reference.resolve(), allow_pickle=False)
    actual_qpos, reference_qpos, indices, metadata = build_qpos(actual, reference)
    steps = np.asarray(actual["motion_step"][indices], dtype=np.int64)
    fps = int(metadata["fps"])

    model = mujoco.MjModel.from_xml_path(str(args.model.resolve()))
    if model.nq != actual_qpos.shape[1]:
        raise ValueError(f"model nq={model.nq}, trajectory nq={actual_qpos.shape[1]}")
    model.vis.global_.offwidth = args.width
    model.vis.global_.offheight = args.height
    actual_data = mujoco.MjData(model)
    reference_data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    fixed_camera = camera()

    ref_torso = np.asarray(actual["reference_torso_pos_w"][indices])
    act_torso = np.asarray(actual["actual_torso_pos_w"][indices])
    ref_object = np.asarray(actual["reference_object_pos_w"][indices])
    act_object = np.asarray(actual["object_pos_w"][indices])

    reference_encoder = RawVideoEncoder(output_dir / "reference_fixed_world.mp4", args.width, args.height, fps)
    actual_encoder = RawVideoEncoder(output_dir / "actual_fixed_world.mp4", args.width, args.height, fps)
    comparison_encoder = RawVideoEncoder(
        output_dir / "actual_vs_reference_fixed_world.mp4", 2 * args.width, args.height, fps
    )
    topdown_encoder = RawVideoEncoder(output_dir / "actual_vs_reference_topdown_xy.mp4", 960, 720, fps)
    preview_steps = {2, 29, 50, 100, 150, 221, 249, 300, 324}
    previews: list[tuple[int, np.ndarray]] = []

    try:
        for i, (step, aq, rq) in enumerate(zip(steps, actual_qpos, reference_qpos)):
            ref_frame = render_qpos(renderer, model, reference_data, rq, fixed_camera)
            actual_frame = render_qpos(renderer, model, actual_data, aq, fixed_camera)
            delta_y = act_torso[i, 1] - ref_torso[i, 1]
            ref_frame = annotate_panel(
                ref_frame, "Reference retargeting", (36, 144, 88), int(step), step / fps
            )
            actual_frame = annotate_panel(
                actual_frame,
                "Actual 30k policy rollout",
                (237, 126, 33),
                int(step),
                step / fps,
                float(delta_y),
            )
            pair = np.concatenate((ref_frame, actual_frame), axis=1)
            reference_encoder.write(ref_frame)
            actual_encoder.write(actual_frame)
            comparison_encoder.write(pair)
            topdown_encoder.write(topdown_frame(i, steps, ref_torso, act_torso, ref_object, act_object))
            if int(step) in preview_steps:
                previews.append((int(step), pair.copy()))
    finally:
        renderer.close()
        reference_encoder.close()
        actual_encoder.close()
        comparison_encoder.close()
        topdown_encoder.close()

    if previews:
        tile_height, tile_width = previews[0][1].shape[:2]
        sheet = Image.new("RGB", (tile_width, tile_height * len(previews)), (20, 24, 31))
        for row, (_, preview) in enumerate(previews):
            sheet.paste(Image.fromarray(preview), (0, row * tile_height))
        sheet.save(output_dir / "actual_vs_reference_contact_sheet.jpg", quality=90)

    torso_error = act_torso - ref_torso
    object_error = act_object - ref_object
    max_torso_i = int(np.argmax(np.abs(torso_error[:, 1])))
    max_object_i = int(np.argmax(np.abs(object_error[:, 1])))
    frame29_i = int(np.flatnonzero(steps == 29)[0])
    summary = {
        "source_actual": str(args.actual.resolve()),
        "source_reference": str(args.reference.resolve()),
        "frame_range": [int(steps[0]), int(steps[-1])],
        "frame_count": int(len(steps)),
        "fps": fps,
        "frame29_actual_minus_reference_torso_xyz_m": torso_error[frame29_i].tolist(),
        "frame29_actual_minus_reference_object_xyz_m": object_error[frame29_i].tolist(),
        "max_abs_torso_y_error_m": float(torso_error[max_torso_i, 1]),
        "max_abs_torso_y_error_frame": int(steps[max_torso_i]),
        "max_abs_object_y_error_m": float(object_error[max_object_i, 1]),
        "max_abs_object_y_error_frame": int(steps[max_object_i]),
    }
    (output_dir / "trajectory_error_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
