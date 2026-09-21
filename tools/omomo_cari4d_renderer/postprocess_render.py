#!/usr/bin/env python3
"""Package Blender frames, diagnostics, GT masks, and visual comparisons."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw

from common import read_json, write_json


def parse_args() -> argparse.Namespace:
    workspace = Path(__file__).resolve().parents[2]
    root = workspace / "exp/omomo_cari4d/sub03_largebox3"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--render-dir", type=Path, default=root / "cari4d_friendly"
    )
    parser.add_argument(
        "--camera-config", type=Path, default=root / "camera_search/best_camera.json"
    )
    parser.add_argument(
        "--original-video",
        type=Path,
        default=Path(
            "src/holosoma_retargeting/holosoma_retargeting/demo_data/external/omomo/motion_videos/sub3/"
            "sub3_largebox_003.mp4"
        ),
    )
    parser.add_argument("--baseline-dir", type=Path, default=root / "baseline")
    parser.add_argument("--fps", type=int, default=30)
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("RUN " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def sorted_images(directory: Path) -> list[Path]:
    return sorted(
        [path for path in directory.iterdir() if path.suffix.lower() in (".png", ".jpg")]
    )


def encode_video(frame_pattern: Path, output: Path, fps: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(frame_pattern),
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "17",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ]
    )


def make_contact_sheet(
    frame_paths: list[Path], selected_frame: int, output: Path
) -> None:
    indices = np.unique(np.linspace(0, len(frame_paths) - 1, 12, dtype=int)).tolist()
    if selected_frame not in indices:
        indices[-1] = selected_frame
        indices.sort()
    tile_size = 300
    caption_height = 34
    sheet = Image.new("RGB", (4 * tile_size, 3 * (tile_size + caption_height)), "white")
    draw = ImageDraw.Draw(sheet)
    for slot, frame_index in enumerate(indices[:12]):
        image = Image.open(frame_paths[frame_index]).convert("RGB")
        image.thumbnail((tile_size, tile_size), Image.Resampling.LANCZOS)
        x = (slot % 4) * tile_size + (tile_size - image.width) // 2
        y = (slot // 4) * (tile_size + caption_height)
        sheet.paste(image, (x, y))
        label = f"frame {frame_index:03d}" + (
            "  [selected]" if frame_index == selected_frame else ""
        )
        draw.text((slot % 4 * tile_size + 8, y + tile_size + 7), label, fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def mask_paths_for_frames(mask_dir: Path, frame_count: int) -> list[Path]:
    paths = sorted_images(mask_dir)
    if len(paths) != frame_count:
        raise RuntimeError(
            f"Expected {frame_count} masks in {mask_dir}, found {len(paths)}"
        )
    return paths


def pack_gt_masks(
    human_masks: list[Path], object_masks: list[Path], output: Path, sequence: str
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "w") as handle:
        group = handle.create_group(sequence)
        for frame, (human_path, object_path) in enumerate(
            zip(human_masks, object_masks)
        ):
            human = np.asarray(Image.open(human_path).convert("L")) > 127
            obj = np.asarray(Image.open(object_path).convert("L")) > 127
            group.create_dataset(
                f"{frame:06d}-k0.person_mask.png", data=human, compression="gzip"
            )
            group.create_dataset(
                f"{frame:06d}-k0.obj_rend_mask.png", data=obj, compression="gzip"
            )


def binary_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 127


def main() -> None:
    args = parse_args()
    camera = read_json(args.camera_config)
    selected = int(camera["best_reconstruction_frame"])
    frames = sorted_images(args.render_dir / "frames")
    if len(frames) != 196:
        raise RuntimeError(f"Expected 196 rendered frames, found {len(frames)}")

    friendly_video = args.render_dir / "sub03_largebox3_cari4d.mp4"
    encode_video(args.render_dir / "frames/%06d.png", friendly_video, args.fps)
    shutil.copy2(friendly_video, args.render_dir / "rgb.mp4")
    shutil.copy2(friendly_video, args.render_dir / "sub03_largebox3.0.color.mp4")
    # CARI4D's official UniDepth wrapper parses the object token from the
    # third underscore-separated filename component.  Preserve the requested
    # alias above and also emit the real OMOMO metadata identifier expected by
    # that unmodified parser.
    canonical_sequence = "sub3_largebox_003"
    shutil.copy2(
        friendly_video, args.render_dir / f"{canonical_sequence}.0.color.mp4"
    )

    args.baseline_dir.mkdir(parents=True, exist_ok=True)
    baseline_frames = args.baseline_dir / "frames"
    baseline_frames.mkdir(parents=True, exist_ok=True)
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(args.original_video),
            str(baseline_frames / "%06d.png"),
        ]
    )
    shutil.copy2(args.original_video, args.baseline_dir / "rgb.mp4")
    write_json(
        args.baseline_dir / "render_config.json",
        {
            "source": "original OMOMO visualization",
            "source_video": args.original_video,
            "rerendered": False,
            "reason": "The baseline is the exact original visualization; the friendly side is the new Blender render.",
        },
    )

    make_contact_sheet(frames, selected, args.render_dir / "preview_contact_sheet.png")
    selected_path = args.render_dir / "selected_reconstruction_frame.png"
    shutil.copy2(frames[selected], selected_path)

    human_masks = mask_paths_for_frames(args.render_dir / "human_mask", len(frames))
    object_masks = mask_paths_for_frames(args.render_dir / "object_mask", len(frames))
    visible_mask = binary_mask(object_masks[selected])
    object_only_path = args.render_dir / "object_only_mask" / f"{selected:06d}.png"
    object_only_mask = binary_mask(object_only_path)
    ys, xs = np.nonzero(object_only_mask)
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    object_only_area = int(object_only_mask.sum())
    visible_area = int(visible_mask.sum())
    exact_occlusion = 1.0 - visible_area / max(object_only_area, 1)

    rgba = np.dstack(
        [
            np.asarray(Image.open(selected_path).convert("RGB")),
            visible_mask.astype(np.uint8) * 255,
        ]
    )
    Image.fromarray(rgba, mode="RGBA").save(
        args.render_dir / "selected_reconstruction_frame_rgba.png"
    )

    masks_root = args.render_dir / "gt_masks"
    pack_gt_masks(
        human_masks,
        object_masks,
        masks_root / "sub03_largebox3_masks_k0.h5",
        "sub03_largebox3",
    )
    pack_gt_masks(
        human_masks,
        object_masks,
        masks_root / f"{canonical_sequence}_masks_k0.h5",
        canonical_sequence,
    )
    candidate_dir = args.render_dir / "candidate_frames"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    for rank, score in enumerate(camera["top_k_frames"], start=1):
        frame = int(score["frame"])
        shutil.copy2(
            frames[frame], candidate_dir / f"rank{rank:02d}_frame_{frame:03d}.png"
        )
    write_json(candidate_dir / "scores.json", {"top_k_frames": camera["top_k_frames"]})

    comparison = args.render_dir.parent / "comparison.mp4"
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(args.original_video),
            "-i",
            str(friendly_video),
            "-filter_complex",
            (
                "[0:v]scale=1280:853:force_original_aspect_ratio=decrease,"
                "pad=1280:1280:(ow-iw)/2:(oh-ih)/2:color=white[left];"
                "[1:v]scale=1280:1280[right];[left][right]hstack=inputs=2[out]"
            ),
            "-map",
            "[out]",
            "-c:v",
            "libx264",
            "-crf",
            "19",
            "-pix_fmt",
            "yuv420p",
            str(comparison),
        ]
    )
    statistics = {
        "selected_reconstruction_frame": selected,
        "object_only_bbox_px": bbox,
        "object_only_bbox_width_px": bbox[2] - bbox[0],
        "object_only_bbox_height_px": bbox[3] - bbox[1],
        "object_only_area_px": object_only_area,
        "object_visible_area_px": visible_area,
        "exact_occlusion_ratio_full_resolution": exact_occlusion,
        "baseline_video": args.baseline_dir / "rgb.mp4",
        "friendly_video": friendly_video,
        "comparison_video": comparison,
        "selected_frame_png": selected_path,
        "gt_masks_h5": masks_root / "sub03_largebox3_masks_k0.h5",
    }
    write_json(args.render_dir / "selected_frame_metrics.json", statistics)
    print(json.dumps(statistics, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
