#!/usr/bin/env python3
"""Track a human and an object from explicit box prompts with official SAM2.

This is a generic interactive RGB-video segmentation adapter.  The boxes are
per-video user inputs in image pixels; no color key, dataset label, motion, or
object-pose annotation is consumed.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from sam2.build_sam import build_sam2_video_predictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True)
    parser.add_argument("--prompt-frame", type=int, default=0)
    parser.add_argument("--human-box", nargs=4, required=True, type=float, metavar=("XMIN", "YMIN", "XMAX", "YMAX"))
    parser.add_argument("--object-box", nargs=4, required=True, type=float, metavar=("XMIN", "YMIN", "XMAX", "YMAX"))
    parser.add_argument("--visualize", action="store_true")
    return parser.parse_args()


def sequence_name(video: Path) -> str:
    suffix = ".0.color"
    return video.stem[:-len(suffix)] if video.stem.endswith(suffix) else video.stem


def validate_box(name: str, box: list[float], width: int, height: int) -> np.ndarray:
    array = np.asarray(box, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN/Inf: {array.tolist()}")
    x0, y0, x1, y1 = array.tolist()
    if not (0 <= x0 < x1 < width and 0 <= y0 < y1 < height):
        raise ValueError(f"{name}={array.tolist()} is outside a {width}x{height} frame")
    return array


def extract_frames(video: Path, frames_dir: Path) -> tuple[list[np.ndarray], float]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        index = len(frames)
        path = frames_dir / f"{index:06d}.jpg"
        if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise RuntimeError(f"Could not write temporary frame: {path}")
        frames.append(frame)
    capture.release()
    if not frames or not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid or empty video: {video}")
    return frames, fps


def save_h5(path: Path, sequence: str, masks: dict[int, dict[int, np.ndarray]], num_frames: int) -> None:
    with h5py.File(path, "w") as handle:
        group = handle.create_group(sequence)
        for index in range(num_frames):
            if index not in masks or set(masks[index]) != {1, 2}:
                raise RuntimeError(f"SAM2 did not return both tracked IDs at frame {index}")
            frame_id = f"{index:06d}"
            group.create_dataset(
                f"{frame_id}-k0.person_mask.png", data=masks[index][1], compression="gzip"
            )
            group.create_dataset(
                f"{frame_id}-k0.obj_rend_mask.png", data=masks[index][2], compression="gzip"
            )


def save_visualization(
    path: Path, frames: list[np.ndarray], masks: dict[int, dict[int, np.ndarray]], fps: float
) -> None:
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width * 2, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create visualization: {path}")
    for index, frame in enumerate(frames):
        overlay = frame.copy()
        human = masks[index][1]
        obj = masks[index][2]
        overlay[human] = (0.5 * overlay[human] + 0.5 * np.asarray([255, 0, 0])).astype(np.uint8)
        overlay[obj] = (0.5 * overlay[obj] + 0.5 * np.asarray([0, 0, 255])).astype(np.uint8)
        writer.write(np.concatenate((frame, overlay), axis=1))
    writer.release()


def main() -> None:
    args = parse_args()
    args.video = args.video.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    if not args.video.is_file() or not args.checkpoint.is_file():
        raise FileNotFoundError(f"video={args.video}, checkpoint={args.checkpoint}")
    if not torch.cuda.is_available():
        raise RuntimeError("SAM2 video tracking requires CUDA")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="holosoma_sam2_frames_") as temporary:
        frames_dir = Path(temporary)
        frames, fps = extract_frames(args.video, frames_dir)
        height, width = frames[0].shape[:2]
        if not 0 <= args.prompt_frame < len(frames):
            raise ValueError(f"prompt frame {args.prompt_frame} is outside [0, {len(frames)})")
        human_box = validate_box("human box", args.human_box, width, height)
        object_box = validate_box("object box", args.object_box, width, height)

        print(f"Building SAM2 predictor: config={args.config}, checkpoint={args.checkpoint}", flush=True)
        predictor = build_sam2_video_predictor(args.config, str(args.checkpoint), device="cuda")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = predictor.init_state(
                video_path=str(frames_dir), offload_video_to_cpu=True, offload_state_to_cpu=True
            )
            predictor.add_new_points_or_box(
                inference_state=state, frame_idx=args.prompt_frame, obj_id=1, box=human_box
            )
            predictor.add_new_points_or_box(
                inference_state=state, frame_idx=args.prompt_frame, obj_id=2, box=object_box
            )
            masks: dict[int, dict[int, np.ndarray]] = {}
            for frame_index, object_ids, mask_logits in predictor.propagate_in_video(state):
                frame_masks: dict[int, np.ndarray] = {}
                for object_id, logits in zip(object_ids, mask_logits):
                    frame_masks[int(object_id)] = np.asarray((logits[0] > 0).cpu(), dtype=bool)
                masks[int(frame_index)] = frame_masks

        sequence = sequence_name(args.video)
        h5_path = args.output_dir / f"{sequence}_masks_k0.h5"
        save_h5(h5_path, sequence, masks, len(frames))
        if args.visualize:
            save_visualization(args.output_dir / f"{sequence}_sam2_vis.mp4", frames, masks, fps)
        print(f"Saved {len(frames)} frames to {h5_path}", flush=True)


if __name__ == "__main__":
    main()
