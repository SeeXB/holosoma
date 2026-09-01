"""Render Original, Uniform-2, and Semantic-B4 trajectories side by side."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

os.environ.setdefault("MUJOCO_GL", "osmesa")

import mujoco
import numpy as np
from PIL import Image

PACKAGE_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = PACKAGE_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from holosoma_retargeting.examples.render_mujoco_trajectory_comparison import (  # noqa: E402
    CameraConfig,
    RawVideoEncoder,
    _annotate,
    _load_trajectory,
    _set_qpos,
    _sha256,
)


METHODS = (
    ("original", "Original OmniRetarget", (226, 92, 88)),
    ("uniform2", "Uniform-2", (76, 120, 168)),
    ("semantic_b4", "Final Semantic B4", (67, 170, 139)),
)


def _camera_for_frame(qposes: Sequence[np.ndarray], config: CameraConfig) -> mujoco.MjvCamera:
    robot_center = np.mean(np.asarray([qpos[:3] for qpos in qposes]), axis=0)
    object_center = np.mean(np.asarray([qpos[-7:-4] for qpos in qposes]), axis=0)
    lookat = config.focus_robot * robot_center + (1.0 - config.focus_robot) * object_center
    lookat[2] += config.lookat_z_offset
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth = config.azimuth
    camera.elevation = config.elevation
    camera.distance = config.distance
    camera.lookat[:] = lookat
    return camera


def _contact_sheet(frames: Sequence[tuple[int, Sequence[np.ndarray]]], path: Path) -> None:
    if not frames:
        return
    tile_height, tile_width = frames[0][1][0].shape[:2]
    sheet = Image.new("RGB", (3 * tile_width, len(frames) * tile_height), (15, 18, 24))
    for row, (_, panels) in enumerate(frames):
        for column, panel in enumerate(panels):
            sheet.paste(Image.fromarray(panel), (column * tile_width, row * tile_height))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=92)


def render(args: argparse.Namespace) -> None:
    experiment_dir = args.experiment_dir.resolve()
    model_path = args.model.resolve()
    output_dir = experiment_dir / "videos"
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        key: experiment_dir / "runs" / key / f"{args.task_name}_original.npz"
        for key, _, _ in METHODS
    }
    loaded = {key: _load_trajectory(path) for key, path in paths.items()}
    shapes = {value[0].shape for value in loaded.values()}
    fps_values = {value[1] for value in loaded.values()}
    if len(shapes) != 1 or len(fps_values) != 1:
        raise ValueError(f"Trajectory mismatch: shapes={shapes}, fps={fps_values}")
    shape = shapes.pop()
    source_fps = fps_values.pop()
    fps = args.fps or source_fps

    model = mujoco.MjModel.from_xml_path(str(model_path))
    if shape[1] != model.nq:
        raise ValueError(f"Trajectory nq={shape[1]} does not match model nq={model.nq}")
    model.vis.global_.offwidth = args.width
    model.vis.global_.offheight = args.height
    data = {key: mujoco.MjData(model) for key, _, _ in METHODS}
    camera_config = CameraConfig(
        azimuth=args.azimuth,
        elevation=args.elevation,
        distance=args.distance,
    )
    preview_indices = sorted(set(int(value) for value in np.clip(args.preview_frames, 0, shape[0] - 1)))
    previews: list[tuple[int, Sequence[np.ndarray]]] = []
    video_path = output_dir / "omniretarget_vs_uniform2_vs_semantic_b4_mujoco.mp4"
    encoder = None if args.preview_only else RawVideoEncoder(
        video_path,
        width=3 * args.width,
        height=args.height,
        fps=fps,
    )
    try:
        with mujoco.Renderer(model, height=args.height, width=args.width) as renderer:
            indices = preview_indices if args.preview_only else range(shape[0])
            for frame_index in indices:
                qposes = [loaded[key][0][frame_index] for key, _, _ in METHODS]
                camera = _camera_for_frame(qposes, camera_config)
                panels: list[np.ndarray] = []
                for (key, title, accent), qpos in zip(METHODS, qposes):
                    _set_qpos(model, data[key], qpos)
                    renderer.update_scene(data[key], camera=camera)
                    panel = renderer.render().copy()
                    panels.append(
                        _annotate(
                            panel,
                            title=title,
                            frame_index=frame_index,
                            frame_count=shape[0],
                            fps=fps,
                            accent=accent,
                        )
                    )
                if frame_index in preview_indices:
                    previews.append((frame_index, tuple(panels)))
                if encoder is not None:
                    encoder.write(np.concatenate(panels, axis=1))
    finally:
        if encoder is not None:
            encoder.close()

    preview_path = output_dir / "semantic_keyframes_triptych.jpg"
    _contact_sheet(previews, preview_path)
    metadata = {
        "playback": "direct cached qpos assignment followed by mujoco.mj_forward; no simulation integration",
        "camera": "one shared per-frame anchor computed from the mean robot/object centers of all three methods",
        "trajectories": {
            key: {"path": str(paths[key]), "sha256": _sha256(paths[key])}
            for key, _, _ in METHODS
        },
        "model": {"path": str(model_path), "sha256": _sha256(model_path)},
        "frames": shape[0],
        "source_fps": source_fps,
        "output_fps": fps,
        "duration_seconds": shape[0] / fps,
        "panel_resolution": [args.width, args.height],
        "video_resolution": [3 * args.width, args.height],
        "video": str(video_path) if encoder is not None else None,
        "contact_sheet": str(preview_path),
        "preview_frames": preview_indices,
    }
    (output_dir / "mujoco_triptych_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--task-name", default="cari4d_sub3_largebox_003")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--azimuth", type=float, default=CameraConfig.azimuth)
    parser.add_argument("--elevation", type=float, default=CameraConfig.elevation)
    parser.add_argument("--distance", type=float, default=CameraConfig.distance)
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument(
        "--preview-frames",
        type=int,
        nargs="+",
        default=[0, 26, 30, 66, 117, 160, 162, 167, 195],
    )
    return parser


if __name__ == "__main__":
    render(_parser().parse_args())
