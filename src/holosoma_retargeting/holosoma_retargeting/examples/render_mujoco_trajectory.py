"""Render one cached retargeting trajectory through MuJoCo offscreen rendering."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")

import mujoco
import numpy as np
from PIL import Image

from holosoma_retargeting.examples.render_mujoco_trajectory_comparison import (
    CameraConfig,
    RawVideoEncoder,
    _annotate,
    _camera_for_frame,
    _load_trajectory,
    _set_qpos,
    _sha256,
)


def render(args: argparse.Namespace) -> None:
    trajectory_path = args.trajectory.expanduser().resolve()
    model_path = args.model.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    metadata_path = output_path.with_suffix(".json")
    preview_path = output_path.with_suffix(".contact_sheet.jpg")

    qpos, source_fps = _load_trajectory(trajectory_path)
    fps = args.fps or source_fps
    model = mujoco.MjModel.from_xml_path(str(model_path))
    if qpos.shape[1] != model.nq:
        raise ValueError(f"Trajectory nq={qpos.shape[1]} does not match model nq={model.nq}")
    model.vis.global_.offwidth = args.width
    model.vis.global_.offheight = args.height
    data = mujoco.MjData(model)
    camera_config = CameraConfig(
        azimuth=args.azimuth,
        elevation=args.elevation,
        distance=args.distance,
    )
    preview_indices = sorted(set(int(value) for value in np.clip(args.preview_frames, 0, len(qpos) - 1)))
    preview_frames: list[np.ndarray] = []
    encoder = RawVideoEncoder(output_path, width=args.width, height=args.height, fps=fps)
    try:
        with mujoco.Renderer(model, height=args.height, width=args.width) as renderer:
            for frame_index, frame_qpos in enumerate(qpos):
                camera = _camera_for_frame(frame_qpos, frame_qpos, camera_config)
                _set_qpos(model, data, frame_qpos)
                renderer.update_scene(data, camera=camera)
                frame = _annotate(
                    renderer.render().copy(),
                    title=args.title,
                    frame_index=frame_index,
                    frame_count=len(qpos),
                    fps=fps,
                    accent=(76, 192, 135),
                )
                encoder.write(frame)
                if frame_index in preview_indices:
                    preview_frames.append(frame)
    finally:
        encoder.close()

    if preview_frames:
        sheet = Image.new("RGB", (args.width, args.height * len(preview_frames)), (15, 18, 24))
        for row, frame in enumerate(preview_frames):
            sheet.paste(Image.fromarray(frame), (0, row * args.height))
        sheet.save(preview_path, quality=92)

    metadata = {
        "playback": "direct cached qpos assignment followed by mujoco.mj_forward; no simulation integration",
        "trajectory": {"path": str(trajectory_path), "sha256": _sha256(trajectory_path)},
        "model": {"path": str(model_path), "sha256": _sha256(model_path)},
        "frames": len(qpos),
        "source_fps": source_fps,
        "output_fps": fps,
        "duration_seconds": len(qpos) / fps,
        "resolution": [args.width, args.height],
        "camera": {
            "azimuth": camera_config.azimuth,
            "elevation": camera_config.elevation,
            "distance": camera_config.distance,
            "anchor": "per-frame robot/object center; fixed orientation and distance",
        },
        "output": str(output_path),
        "preview": str(preview_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--title", default="CARI4D to G1 | Semantic B4")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--azimuth", type=float, default=CameraConfig.azimuth)
    parser.add_argument("--elevation", type=float, default=CameraConfig.elevation)
    parser.add_argument("--distance", type=float, default=CameraConfig.distance)
    parser.add_argument(
        "--preview-frames",
        type=int,
        nargs="+",
        default=[0, 26, 30, 66, 86, 117, 160, 162, 167, 195],
    )
    return parser


if __name__ == "__main__":
    render(_parser().parse_args())
