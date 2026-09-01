#!/usr/bin/env python3
"""Make a resolution-limited CARI4D input variant without changing motion."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import h5py


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--source-masks", type=Path, required=True)
    parser.add_argument("--source-sequence", required=True)
    parser.add_argument("--alias-sequence", required=True)
    parser.add_argument("--source-mesh", type=Path, required=True)
    parser.add_argument("--frame-index", type=int, required=True)
    parser.add_argument("--resolution", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    root = args.output_root.resolve()
    for name in ("videos", "masks", "meshes", "packed", "nlf"):
        (root / name).mkdir(parents=True, exist_ok=True)

    video = root / "videos" / f"{args.alias_sequence}.0.color.mp4"
    subprocess.run(
        [
            "ffmpeg", "-loglevel", "error", "-y", "-i", str(args.source_video),
            "-vf", f"scale={args.resolution}:{args.resolution}:flags=lanczos",
            "-c:v", "libx264", "-preset", "slow", "-crf", "12",
            "-pix_fmt", "yuv420p", "-an", str(video),
        ],
        check=True,
    )

    masks = root / "masks" / f"{args.alias_sequence}_masks_k0.h5"
    with h5py.File(args.source_masks, "r") as source, h5py.File(masks, "w") as target:
        source_group = source[args.source_sequence]
        target_group = target.create_group(args.alias_sequence)
        for key in sorted(source_group.keys()):
            mask = source_group[key][:].astype("uint8")
            resized = cv2.resize(
                mask,
                (args.resolution, args.resolution),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            target_group.create_dataset(
                key, data=resized, compression="gzip", compression_opts=1
            )

    tag = f"{args.alias_sequence}_{args.frame_index:03d}"
    mesh_dir = root / "meshes" / f"{tag}_rgba"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    mesh = mesh_dir / f"{tag}_align.obj"
    if mesh.is_symlink():
        mesh.unlink()
    elif mesh.exists():
        raise FileExistsError(f"Refusing to replace non-symlink: {mesh}")
    mesh.symlink_to(args.source_mesh.resolve())

    capture = cv2.VideoCapture(str(video))
    report = {
        "schema": "holosoma.cari4d_resolution_variant.v1",
        "reason": "official 196-frame 1280-square batched rasterization exceeds 24 GiB",
        "source_video": str(args.source_video.resolve()),
        "source_masks": str(args.source_masks.resolve()),
        "alias_sequence": args.alias_sequence,
        "resolution": args.resolution,
        "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "motion_modified": False,
        "video_resampling": "Lanczos 1280-to-target; H.264 CRF 12",
        "mask_resampling": "nearest-neighbor",
        "mesh_target": str(args.source_mesh.resolve()),
    }
    capture.release()
    (root / "resolution_variant.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
