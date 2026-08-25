#!/usr/bin/env python3
"""Stage an arbitrary RGB video and prepare CARI4D's packed masks.

The default backend is CARI4D's official SAM3 text-prompted video
segmentation. ``color-key`` is an explicit adapter for synthetic or
green-screen-like inputs; it is never selected automatically.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np


GENDER_SUBJECT_ALIAS = {"male": "Sub01", "female": "Sub06"}


def parse_args() -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path, help="Input RGB MP4; its original name is unrestricted.")
    parser.add_argument("--output-root", required=True, type=Path, help="Preprocessing output root.")
    parser.add_argument(
        "--gender",
        required=True,
        choices=tuple(GENDER_SUBJECT_ALIAS),
        help="SMPL-H gender. CARI4D supports male/female models and does not infer this field.",
    )
    parser.add_argument(
        "--object-name",
        required=True,
        help="Short alphanumeric object alias used only in CARI4D's required sequence filename.",
    )
    parser.add_argument(
        "--sequence-name",
        help=(
            "Optional CARI4D-native sequence alias. If omitted, a compatible alias is generated from gender, "
            "object name, and input stem."
        ),
    )
    parser.add_argument(
        "--mask-backend",
        choices=("sam3", "sam2-box", "existing", "color-key"),
        default="sam3",
        help=(
            "Mask producer. SAM3 is the automatic general RGB-video default; sam2-box is a general "
            "interactive fallback initialized by two bounding boxes; color-key must be explicitly requested."
        ),
    )
    parser.add_argument("--human-prompt", help="SAM3 text prompt, for example 'person'.")
    parser.add_argument("--object-prompt", help="SAM3 text prompt, for example 'cardboard box'.")
    parser.add_argument(
        "--sam3-python",
        type=Path,
        help="Python executable in the separate SAM3 environment (default: current Python).",
    )
    parser.add_argument("--sam3-chunk-size", type=int, default=300)
    parser.add_argument("--sam2-python", type=Path, help="Python executable with the official SAM2 package installed.")
    parser.add_argument("--sam2-checkpoint", type=Path, help="Local official SAM2/SAM2.1 checkpoint.")
    parser.add_argument(
        "--sam2-config",
        default="configs/sam2.1/sam2.1_hiera_s.yaml",
        help="SAM2 Hydra config name, relative to the installed SAM2 package config search path.",
    )
    parser.add_argument("--sam2-prompt-frame", type=int, default=0)
    parser.add_argument("--human-box", nargs=4, type=float, metavar=("XMIN", "YMIN", "XMAX", "YMAX"))
    parser.add_argument("--object-box", nargs=4, type=float, metavar=("XMIN", "YMIN", "XMAX", "YMAX"))
    parser.add_argument("--existing-masks", type=Path, help="Existing CARI4D-format HDF5 for backend=existing.")
    parser.add_argument("--human-hue", nargs=2, type=int, metavar=("MIN", "MAX"))
    parser.add_argument("--object-hue", nargs=2, type=int, metavar=("MIN", "MAX"))
    parser.add_argument("--min-saturation", type=int, default=35)
    parser.add_argument("--min-value", type=int, default=35)
    parser.add_argument("--morph-kernel", type=int, default=1)
    parser.add_argument("--cari4d-root", type=Path, default=repository / "third_party" / "CARI4D")
    return parser.parse_args()


def _slug(value: str, *, field: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "", value)
    if not slug:
        raise ValueError(f"{field} must contain at least one alphanumeric character")
    return slug


def _sequence_name(args: argparse.Namespace) -> str:
    if args.sequence_name:
        sequence = args.sequence_name
    else:
        object_token = _slug(args.object_name, field="--object-name")
        source_token = _slug(args.video.stem, field="--video stem")[:48]
        sequence = f"Custom_{GENDER_SUBJECT_ALIAS[args.gender]}_{object_token}_{source_token}"
    tokens = sequence.split("_")
    if len(tokens) < 3:
        raise ValueError("--sequence-name must have at least three underscore-separated tokens")
    const_file = args.cari4d_root / "behave_data" / "const.py"
    if not const_file.is_file():
        raise FileNotFoundError(f"Invalid CARI4D checkout; missing {const_file}")
    subject_genders = runpy.run_path(str(const_file))["_sub_gender"]
    actual_gender = subject_genders.get(tokens[1])
    if actual_gender != args.gender:
        raise ValueError(
            f"CARI4D indexes sequence token 2 for gender; token {tokens[1]!r} maps to "
            f"{actual_gender!r}, not --gender={args.gender!r}. Use the generated name or a matching "
            "source-supported alias."
        )
    _slug(tokens[2], field="object token in --sequence-name")
    return sequence


def _video_info(path: Path) -> tuple[int, float, int, int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {path}")
    frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if frames <= 0 or not np.isfinite(fps) or fps <= 0 or width <= 0 or height <= 0:
        raise ValueError(f"Invalid video metadata: frames={frames}, fps={fps}, size={width}x{height}")
    return frames, fps, width, height


def _color_mask(
    hsv: np.ndarray,
    hue_range: tuple[int, int] | list[int],
    min_saturation: int,
    min_value: int,
    kernel: np.ndarray,
) -> np.ndarray:
    lower = np.asarray([hue_range[0], min_saturation, min_value], dtype=np.uint8)
    upper = np.asarray([hue_range[1], 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    if kernel.shape[0] > 1:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask.astype(bool)


def _run_color_key(args: argparse.Namespace, video: Path, masks: Path, sequence: str) -> None:
    if args.human_hue is None or args.object_hue is None:
        raise ValueError("backend=color-key requires explicit --human-hue MIN MAX and --object-hue MIN MAX")
    for label, value in (("human", args.human_hue), ("object", args.object_hue)):
        if not 0 <= value[0] <= value[1] <= 179:
            raise ValueError(f"Invalid {label} hue range {value}; OpenCV hue must be in [0,179]")
    if not (0 <= args.min_saturation <= 255 and 0 <= args.min_value <= 255):
        raise ValueError("--min-saturation and --min-value must be in [0,255]")
    if args.morph_kernel < 1 or args.morph_kernel % 2 == 0:
        raise ValueError("--morph-kernel must be a positive odd integer")

    capture = cv2.VideoCapture(str(video))
    kernel = np.ones((args.morph_kernel, args.morph_kernel), dtype=np.uint8)
    with h5py.File(masks, "w") as handle:
        group = handle.create_group(sequence)
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            human = _color_mask(hsv, args.human_hue, args.min_saturation, args.min_value, kernel)
            obj = _color_mask(hsv, args.object_hue, args.min_saturation, args.min_value, kernel)
            overlap = human & obj
            if overlap.any():
                hue = hsv[..., 0].astype(np.float32)
                human_center = 0.5 * sum(args.human_hue)
                object_center = 0.5 * sum(args.object_hue)
                to_human = overlap & (np.abs(hue - human_center) <= np.abs(hue - object_center))
                obj[to_human] = False
                human[overlap & ~to_human] = False
            frame_id = f"{index:06d}"
            group.create_dataset(f"{frame_id}-k0.person_mask.png", data=human, compression="gzip")
            group.create_dataset(f"{frame_id}-k0.obj_rend_mask.png", data=obj, compression="gzip")
            index += 1
    capture.release()


def _run_sam3(args: argparse.Namespace, video: Path, masks_dir: Path) -> list[str]:
    if not args.human_prompt or not args.object_prompt:
        raise ValueError("backend=sam3 requires --human-prompt and --object-prompt")
    if args.sam3_chunk_size <= 0:
        raise ValueError("--sam3-chunk-size must be positive")
    script = args.cari4d_root / "prep" / "run_sam3_masks.py"
    if not script.is_file():
        raise FileNotFoundError(f"CARI4D SAM3 mask script is missing: {script}")
    python = args.sam3_python or Path(sys.executable)
    if not Path(python).is_file():
        raise FileNotFoundError(f"SAM3 Python executable does not exist: {python}")
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError(
            "backend=sam3 requires HF_TOKEN in the SAM3 process environment after access to "
            "facebook/sam3 has been granted; credentials are not accepted on the command line or logged"
        )
    command = [
        str(python), str(script), "--video", str(video),
        "--human_prompt", args.human_prompt, "--object_prompt", args.object_prompt,
        "--output_dir", str(masks_dir), "--chunk_size", str(args.sam3_chunk_size), "--visualize",
    ]
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)
    return command


def _run_sam2_box(args: argparse.Namespace, video: Path, masks_dir: Path) -> list[str]:
    if args.sam2_python is None or args.sam2_checkpoint is None:
        raise ValueError("backend=sam2-box requires --sam2-python and --sam2-checkpoint")
    if args.human_box is None or args.object_box is None:
        raise ValueError("backend=sam2-box requires --human-box and --object-box")
    # Do not resolve this path: venv Python executables are commonly symlinks,
    # and resolving one would discard the venv's site-packages at execution.
    python = args.sam2_python.expanduser().absolute()
    checkpoint = args.sam2_checkpoint.expanduser().resolve()
    if not python.is_file():
        raise FileNotFoundError(f"SAM2 Python executable does not exist: {python}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"SAM2 checkpoint does not exist: {checkpoint}")
    script = Path(__file__).resolve().with_name("run_sam2_box_masks.py")
    if not script.is_file():
        raise FileNotFoundError(f"SAM2 mask helper is missing: {script}")
    command = [
        str(python), str(script), "--video", str(video), "--output-dir", str(masks_dir),
        "--checkpoint", str(checkpoint), "--config", args.sam2_config,
        "--prompt-frame", str(args.sam2_prompt_frame),
        "--human-box", *(str(value) for value in args.human_box),
        "--object-box", *(str(value) for value in args.object_box),
        "--visualize",
    ]
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)
    return command


def _validate_masks(
    masks: Path,
    sequence: str,
    *,
    expected_frames: int,
    expected_shape: tuple[int, int],
) -> dict[str, object]:
    human_counts: list[int] = []
    object_counts: list[int] = []
    overlaps: list[int] = []
    with h5py.File(masks, "r") as handle:
        if set(handle.keys()) != {sequence}:
            raise ValueError(f"Mask HDF5 must contain exactly group {sequence!r}, got {list(handle.keys())}")
        group = handle[sequence]
        expected_keys: set[str] = set()
        for index in range(expected_frames):
            frame_id = f"{index:06d}"
            human_key = f"{frame_id}-k0.person_mask.png"
            object_key = f"{frame_id}-k0.obj_rend_mask.png"
            expected_keys.update((human_key, object_key))
            if human_key not in group or object_key not in group:
                raise KeyError(f"Missing CARI4D mask datasets at frame {index}: {human_key}, {object_key}")
            human = np.asarray(group[human_key], dtype=bool)
            obj = np.asarray(group[object_key], dtype=bool)
            if human.shape != expected_shape or obj.shape != expected_shape:
                raise ValueError(
                    f"Mask shape mismatch at frame {index}: human={human.shape}, object={obj.shape}, "
                    f"video={expected_shape}"
                )
            human_count, object_count = int(human.sum()), int(obj.sum())
            if human_count == 0 or object_count == 0:
                raise ValueError(
                    f"Empty mask at frame {index}: human_pixels={human_count}, object_pixels={object_count}. "
                    "CARI4D is not designed for long occlusions; fix segmentation instead of filling silently."
                )
            human_counts.append(human_count)
            object_counts.append(object_count)
            overlaps.append(int(np.count_nonzero(human & obj)))
        extras = sorted(set(group.keys()) - expected_keys)
        if extras:
            raise ValueError(f"Mask HDF5 contains unexpected datasets (first 10): {extras[:10]}")
    return {
        "status": "PASS",
        "frames": expected_frames,
        "human_mask_pixels_min_max": [min(human_counts), max(human_counts)],
        "object_mask_pixels_min_max": [min(object_counts), max(object_counts)],
        "overlap_pixels_total": sum(overlaps),
        "nonempty_every_frame": True,
    }


def main() -> None:
    args = parse_args()
    args.video = args.video.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.cari4d_root = args.cari4d_root.expanduser().resolve()
    if not args.video.is_file():
        raise FileNotFoundError(args.video)
    sequence = _sequence_name(args)
    frames, fps, width, height = _video_info(args.video)

    videos_dir = args.output_root / "videos"
    masks_dir = args.output_root / "masks"
    validation_dir = args.output_root / "validation"
    for directory in (videos_dir, masks_dir, validation_dir):
        directory.mkdir(parents=True, exist_ok=True)
    staged_video = videos_dir / f"{sequence}.0.color.mp4"
    if staged_video.resolve() != args.video:
        shutil.copy2(args.video, staged_video)
    masks = masks_dir / f"{sequence}_masks_k0.h5"

    backend_command: list[str] | None = None
    if args.mask_backend == "sam3":
        backend_command = _run_sam3(args, staged_video, masks_dir)
    elif args.mask_backend == "sam2-box":
        backend_command = _run_sam2_box(args, staged_video, masks_dir)
    elif args.mask_backend == "existing":
        if args.existing_masks is None:
            raise ValueError("backend=existing requires --existing-masks")
        source_masks = args.existing_masks.expanduser().resolve()
        if not source_masks.is_file():
            raise FileNotFoundError(source_masks)
        if source_masks != masks.resolve():
            shutil.copy2(source_masks, masks)
    else:
        _run_color_key(args, staged_video, masks, sequence)

    validation = _validate_masks(masks, sequence, expected_frames=frames, expected_shape=(height, width))
    report = {
        **validation,
        "schema": "holosoma.cari4d_video_preparation.v1",
        "source_video": str(args.video),
        "staged_video": str(staged_video.resolve()),
        "sequence_name": sequence,
        "gender": args.gender,
        "object_name": args.object_name,
        "fps": fps,
        "resolution_wh": [width, height],
        "mask_h5": str(masks.resolve()),
        "mask_backend": args.mask_backend,
        "general_rgb_default_backend": "sam3",
        "sample_specific_backend": args.mask_backend == "color-key",
        "backend_command": backend_command,
        "human_prompt": args.human_prompt if args.mask_backend == "sam3" else None,
        "object_prompt": args.object_prompt if args.mask_backend == "sam3" else None,
        "sam2_box_parameters": (
            {
                "prompt_frame": args.sam2_prompt_frame,
                "human_box_xyxy_pixels": args.human_box,
                "object_box_xyxy_pixels": args.object_box,
                "checkpoint": str(args.sam2_checkpoint.expanduser().resolve()),
                "config": args.sam2_config,
            }
            if args.mask_backend == "sam2-box" else None
        ),
        "color_key_parameters": (
            {
                "human_hue_opencv": args.human_hue,
                "object_hue_opencv": args.object_hue,
                "min_saturation": args.min_saturation,
                "min_value": args.min_value,
                "morph_kernel": args.morph_kernel,
            }
            if args.mask_backend == "color-key" else None
        ),
        "uses_motion_or_object_pose_labels": False,
    }
    report_path = validation_dir / "video_preparation_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
