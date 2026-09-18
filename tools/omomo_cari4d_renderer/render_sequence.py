#!/usr/bin/env python3
"""Prepare, render, and encode one OMOMO sequence with a static camera.

This is the multi-sequence wrapper around the original sub3 renderer.  It
keeps every sequence in its own ``exp/omomo_cari4d/<sequence>`` directory and
does not alter the OMOMO motion or object trajectory.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from common import canonical_sequence_name


def run(command: list[str]) -> None:
    print("RUN " + " ".join(str(value) for value in command), flush=True)
    subprocess.run([str(value) for value in command], check=True)


def parse_args() -> argparse.Namespace:
    workspace = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--workspace", type=Path, default=workspace)
    parser.add_argument("--blender", type=Path, default=workspace / "third_party/blender-3.6.15-linux-x64/blender")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--resolution", type=int, default=720)
    parser.add_argument("--search-resolution", type=int, default=192)
    parser.add_argument("--search-frame-count", type=int, default=8)
    parser.add_argument("--azimuth-step", type=int, default=30)
    parser.add_argument("--elevations", default="5,15,25,35")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--strict-search", action="store_true", help="use the original mask-render camera search")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sequence = canonical_sequence_name(args.sequence)
    bundle_name = "sub03_largebox3" if sequence == "sub3_largebox_003" else sequence
    root = args.workspace / f"exp/omomo_cari4d/{bundle_name}"
    archive = root / "input/omomo_gt_sequence.npz"
    camera_dir = root / "camera_search"
    render_dir = root / "cari4d_friendly"
    video = render_dir / f"{sequence}_rerender.mp4"

    prepare = [args.python, args.workspace / "tools/omomo_cari4d_renderer/prepare_sequence.py", "--sequence", sequence]
    if args.force or not archive.exists():
        run(prepare)
    if args.force or not (camera_dir / "best_camera.json").exists():
        search_command = [
            args.blender, "-b", "-P", args.workspace / "tools/omomo_cari4d_renderer/blender_pipeline.py", "--",
            "search", "--sequence-archive", archive, "--output", camera_dir,
            "--search-resolution", args.search_resolution,
            "--search-frame-count", args.search_frame_count,
            "--azimuth-step", args.azimuth_step,
            "--elevations", args.elevations,
        ]
        if not args.strict_search:
            search_command.append("--fast-search")
        run(search_command)
    if args.force or not video.exists():
        render_command = [
            args.blender, "-b", "-P", args.workspace / "tools/omomo_cari4d_renderer/blender_pipeline.py", "--",
            "render", "--sequence-archive", archive,
            "--camera-config", camera_dir / "best_camera.json",
            "--resolution", args.resolution, "--output", render_dir,
        ]
        if not args.diagnostics:
            render_command.append("--rgb-only")
        run(render_command)
        run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-framerate", "30", "-i", render_dir / "frames/%06d.png",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", video,
        ])
        # ``rgb.mp4`` is the stable filename consumed by downstream VLM tools.
        rgb = render_dir / "rgb.mp4"
        if rgb != video:
            rgb.unlink(missing_ok=True)
            rgb.symlink_to(video.name)
    print(f"RERENDER_VIDEO={video}", flush=True)


if __name__ == "__main__":
    main()
