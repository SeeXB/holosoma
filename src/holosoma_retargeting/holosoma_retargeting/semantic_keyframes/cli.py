"""Command-line entry point for pre-retargeting semantic keyframes."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from holosoma_retargeting.semantic_keyframes.pipeline import generate_semantic_keyframes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Use a VLM to plan task-specific semantic actions/functions, then resolve them to frame windows from local motion signals."
        )
    )
    parser.add_argument("--video", type=Path, required=True, help="Source RGB video shown to the VLM.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--smplx-file", type=Path,
        help="Video-derived SMPL-X human/object trajectory in the original pipeline format.",
    )
    source.add_argument(
        "--bundle-file", type=Path,
        help="Aligned NPZ with human_joints, object_poses_wxyz_xyz and joint names.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output semantic-keyframe JSON file.")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("assets/body_models"),
        help="SMPL-X body-model directory.",
    )
    parser.add_argument("--sample-count", type=int, default=12, help="Number of video images sent to the VLM.")
    parser.add_argument("--max-repairs", type=int, default=3, help="Maximum invalid-plan repair attempts.")
    parser.add_argument(
        "--legacy-fixed-roles", action="store_true",
        help="Use the historical fixed box-carry roles (kept for compatibility).",
    )
    parser.add_argument("--baseline", type=Path, help="Optional v1 JSON used only for a post-generation diff.")
    parser.add_argument("--diff-output", type=Path, help="Optional v1/v2 diff output path.")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    generate_semantic_keyframes(
        video=args.video,
        smplx_file=args.smplx_file,
        bundle_file=args.bundle_file,
        dynamic=not args.legacy_fixed_roles,
        output=args.output,
        model_dir=args.model_dir,
        sample_count=args.sample_count,
        max_repairs=args.max_repairs,
        baseline=args.baseline,
        diff_output=args.diff_output,
    )
    print(f"Wrote VLM plan and deterministic trigger windows to {args.output}")


if __name__ == "__main__":
    main()
