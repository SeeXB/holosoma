"""Render cached retargeting trajectories with the same MuJoCo camera.

The default inputs are the registered OmniRetarget baseline and the selected
Transition-Truncated-B4 semantic method for ``sub3_largebox_003``.  The script
uses direct qpos playback (no simulation integration), so every video frame is
an exact visualization of one cached trajectory frame.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Sequence

# The environment must be selected before importing mujoco.  OSMesa works on
# machines without a display or a compatible EGL/PyOpenGL combination.  A
# caller with a configured GPU stack can still opt into ``MUJOCO_GL=egl``.
os.environ.setdefault("MUJOCO_GL", "osmesa")

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


PACKAGE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ORIGINAL = (
    PACKAGE_DIR
    / "benchmark_results_semantic_final"
    / "runs"
    / "compat_original"
    / "sub3_largebox_003_original.npz"
)
DEFAULT_FINAL = (
    PACKAGE_DIR
    / "benchmark_results_full_event_transition_truncation"
    / "runs"
    / "transition_truncated_b4"
    / "sub3_largebox_003_original.npz"
)
DEFAULT_MODEL = PACKAGE_DIR / "models" / "g1" / "g1_29dof_w_largebox.xml"
DEFAULT_OUTPUT_DIR = (
    PACKAGE_DIR
    / "benchmark_results_full_event_transition_truncation"
    / "videos"
)


@dataclass(frozen=True)
class CameraConfig:
    azimuth: float = 142.0
    elevation: float = -18.0
    distance: float = 3.15
    focus_robot: float = 0.62
    lookat_z_offset: float = 0.08


class RawVideoEncoder:
    """Stream RGB frames to ffmpeg without retaining the video in memory."""

    def __init__(self, path: Path, *, width: int, height: int, fps: int) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg is required to encode MP4 output")
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
        self._process = subprocess.Popen(command, stdin=subprocess.PIPE)
        if self._process.stdin is None:
            raise RuntimeError("ffmpeg did not expose a stdin pipe")
        self._stdin: BinaryIO = self._process.stdin

    def write(self, frame: np.ndarray) -> None:
        self._stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())

    def close(self) -> None:
        self._stdin.close()
        return_code = self._process.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg failed for {self.path} ({return_code=})")


def _load_trajectory(path: Path) -> tuple[np.ndarray, int]:
    with np.load(path, allow_pickle=False) as payload:
        qpos = np.asarray(payload["qpos"], dtype=np.float64)
        fps = int(payload["fps"]) if "fps" in payload else 30
    if qpos.ndim != 2:
        raise ValueError(f"Expected a 2-D qpos trajectory, got {qpos.shape} from {path}")
    if not np.all(np.isfinite(qpos)):
        raise ValueError(f"Trajectory contains non-finite qpos values: {path}")
    return qpos, fps


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    suffix = "-Bold" if bold else ""
    path = Path(f"/usr/share/fonts/truetype/dejavu/DejaVuSans{suffix}.ttf")
    if path.exists():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _annotate(
    frame: np.ndarray,
    *,
    title: str,
    frame_index: int,
    frame_count: int,
    fps: int,
    accent: tuple[int, int, int],
) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    width, height = image.size
    draw.rounded_rectangle((18, 16, width - 18, 72), radius=12, fill=(8, 12, 18, 205))
    draw.rectangle((18, 16, 28, 72), fill=(*accent, 255))
    draw.text((44, 27), title, font=_font(26, bold=True), fill=(245, 248, 252, 255))
    status = f"Frame {frame_index:03d}/{frame_count - 1:03d}   {frame_index / fps:5.2f} s"
    status_font = _font(20)
    bbox = draw.textbbox((0, 0), status, font=status_font)
    status_width = bbox[2] - bbox[0]
    draw.rounded_rectangle(
        (width - status_width - 42, height - 52, width - 18, height - 16),
        radius=9,
        fill=(8, 12, 18, 185),
    )
    draw.text(
        (width - status_width - 30, height - 46),
        status,
        font=status_font,
        fill=(235, 240, 247, 255),
    )
    return np.asarray(image)


def _camera_for_frame(
    original_qpos: np.ndarray,
    final_qpos: np.ndarray,
    config: CameraConfig,
) -> mujoco.MjvCamera:
    # Compute a method-independent camera anchor.  Both panels therefore use
    # exactly the same camera, while the anchor follows the action through the
    # long y-axis translation of this sequence.
    robot_center = 0.5 * (original_qpos[:3] + final_qpos[:3])
    object_center = 0.5 * (original_qpos[-7:-4] + final_qpos[-7:-4])
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


def _set_qpos(model: mujoco.MjModel, data: mujoco.MjData, qpos: np.ndarray) -> None:
    if qpos.shape != (model.nq,):
        raise ValueError(f"qpos has shape {qpos.shape}, but the MuJoCo model requires {(model.nq,)}")
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def _render_pair(
    renderer: mujoco.Renderer,
    model: mujoco.MjModel,
    original_data: mujoco.MjData,
    final_data: mujoco.MjData,
    original_qpos: np.ndarray,
    final_qpos: np.ndarray,
    *,
    frame_index: int,
    frame_count: int,
    fps: int,
    camera_config: CameraConfig,
) -> tuple[np.ndarray, np.ndarray]:
    camera = _camera_for_frame(original_qpos, final_qpos, camera_config)

    _set_qpos(model, original_data, original_qpos)
    renderer.update_scene(original_data, camera=camera)
    original_frame = renderer.render().copy()

    _set_qpos(model, final_data, final_qpos)
    renderer.update_scene(final_data, camera=camera)
    final_frame = renderer.render().copy()

    return (
        _annotate(
            original_frame,
            title="Original OmniRetarget",
            frame_index=frame_index,
            frame_count=frame_count,
            fps=fps,
            accent=(226, 92, 88),
        ),
        _annotate(
            final_frame,
            title="Final: Transition-Truncated-B4",
            frame_index=frame_index,
            frame_count=frame_count,
            fps=fps,
            accent=(76, 192, 135),
        ),
    )


def _contact_sheet(frames: Sequence[tuple[int, np.ndarray, np.ndarray]], path: Path) -> None:
    if not frames:
        return
    tile_height, tile_width = frames[0][1].shape[:2]
    sheet = Image.new("RGB", (2 * tile_width, len(frames) * tile_height), (15, 18, 24))
    for row, (_, original, final) in enumerate(frames):
        sheet.paste(Image.fromarray(original), (0, row * tile_height))
        sheet.paste(Image.fromarray(final), (tile_width, row * tile_height))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=92)


def render(args: argparse.Namespace) -> None:
    original_path = args.original.resolve()
    final_path = args.final.resolve()
    model_path = args.model.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    original_qpos, original_fps = _load_trajectory(original_path)
    final_qpos, final_fps = _load_trajectory(final_path)
    if original_qpos.shape != final_qpos.shape:
        raise ValueError(
            f"Trajectory shapes differ: original={original_qpos.shape}, final={final_qpos.shape}"
        )
    if original_fps != final_fps:
        raise ValueError(f"Trajectory FPS differs: original={original_fps}, final={final_fps}")
    fps = args.fps or original_fps
    frame_count = original_qpos.shape[0]

    model = mujoco.MjModel.from_xml_path(str(model_path))
    if original_qpos.shape[1] != model.nq:
        raise ValueError(
            f"Trajectory nq={original_qpos.shape[1]} does not match model nq={model.nq}"
        )
    model.vis.global_.offwidth = args.width
    model.vis.global_.offheight = args.height
    original_data = mujoco.MjData(model)
    final_data = mujoco.MjData(model)
    camera_config = CameraConfig(
        azimuth=args.azimuth,
        elevation=args.elevation,
        distance=args.distance,
    )
    preview_indices = sorted(set(np.clip(args.preview_frames, 0, frame_count - 1)))
    previews: list[tuple[int, np.ndarray, np.ndarray]] = []

    outputs = {
        "original": output_dir / "original_omniretarget_mujoco.mp4",
        "final": output_dir / "final_transition_truncated_b4_mujoco.mp4",
        "comparison": output_dir / "original_vs_final_side_by_side_mujoco.mp4",
    }
    encoders: dict[str, RawVideoEncoder] = {}
    if not args.preview_only:
        encoders = {
            "original": RawVideoEncoder(outputs["original"], width=args.width, height=args.height, fps=fps),
            "final": RawVideoEncoder(outputs["final"], width=args.width, height=args.height, fps=fps),
            "comparison": RawVideoEncoder(
                outputs["comparison"], width=2 * args.width, height=args.height, fps=fps
            ),
        }

    try:
        with mujoco.Renderer(model, height=args.height, width=args.width) as renderer:
            frame_indices = preview_indices if args.preview_only else range(frame_count)
            for frame_index in frame_indices:
                original_frame, final_frame = _render_pair(
                    renderer,
                    model,
                    original_data,
                    final_data,
                    original_qpos[frame_index],
                    final_qpos[frame_index],
                    frame_index=frame_index,
                    frame_count=frame_count,
                    fps=fps,
                    camera_config=camera_config,
                )
                if frame_index in preview_indices:
                    previews.append((frame_index, original_frame, final_frame))
                if encoders:
                    encoders["original"].write(original_frame)
                    encoders["final"].write(final_frame)
                    encoders["comparison"].write(
                        np.concatenate((original_frame, final_frame), axis=1)
                    )
    finally:
        for encoder in encoders.values():
            encoder.close()

    preview_path = output_dir / "mujoco_comparison_contact_sheet.jpg"
    _contact_sheet(previews, preview_path)
    metadata = {
        "playback": "direct cached qpos assignment followed by mujoco.mj_forward; no simulation integration",
        "original": {"path": str(original_path), "sha256": _sha256(original_path)},
        "final": {"path": str(final_path), "sha256": _sha256(final_path)},
        "model": {"path": str(model_path), "sha256": _sha256(model_path)},
        "frames": frame_count,
        "source_fps": original_fps,
        "output_fps": fps,
        "duration_seconds": frame_count / fps,
        "resolution": [args.width, args.height],
        "comparison_resolution": [2 * args.width, args.height],
        "camera": {
            "azimuth": camera_config.azimuth,
            "elevation": camera_config.elevation,
            "distance": camera_config.distance,
            "anchor": "per-frame shared robot/object center from the mean of both methods",
        },
        "outputs": {key: str(path) for key, path in outputs.items()} if encoders else {},
        "preview": str(preview_path),
    }
    (output_dir / "mujoco_video_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, default=DEFAULT_ORIGINAL)
    parser.add_argument("--final", type=Path, default=DEFAULT_FINAL)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--azimuth", type=float, default=CameraConfig.azimuth)
    parser.add_argument("--elevation", type=float, default=CameraConfig.elevation)
    parser.add_argument("--distance", type=float, default=CameraConfig.distance)
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument(
        "--preview-frames",
        type=int,
        nargs="+",
        default=[0, 26, 30, 34, 79, 117, 128, 162, 195],
    )
    return parser


def cli() -> None:
    render(_parser().parse_args())


if __name__ == "__main__":
    cli()
