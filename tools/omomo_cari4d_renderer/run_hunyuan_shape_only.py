#!/usr/bin/env python3
"""Run the official Hunyuan3D-2 shape stage without the optional paint stage.

This wrapper exists outside CARI4D and uses the exact shape pipeline call made
by official ``prep/run_hy3d_recon.py``.  It does not alter model code, weights,
sampling arguments, or the input RGBA image.  The untextured mesh is sufficient
for geometric aspect-ratio/Chamfer diagnostics and CARI4D's normalized object
mesh input.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.text2image import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgba", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="tencent/Hunyuan3D-2")
    parser.add_argument("--seed", type=int, default=600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image = Image.open(args.rgba).convert("RGBA")
    seed_everything(args.seed)
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(args.model)
    mesh = pipeline(image=image)[0]
    mesh.export(args.output)
    print(f"Saved official Hunyuan shape mesh: {args.output}")


if __name__ == "__main__":
    main()
