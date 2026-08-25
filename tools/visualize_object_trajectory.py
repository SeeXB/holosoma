#!/usr/bin/env python3
"""Render first/middle/last canonical object poses as a dependency-light PNG."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from hoi_pipeline.common import read_obj_vertices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", required=True, type=Path, help="Canonical sequence directory.")
    parser.add_argument("--output", type=Path, help="PNG path; defaults under sequence/validation.")
    parser.add_argument("--max-points", type=int, default=5000, help="Maximum vertices drawn per pose.")
    args = parser.parse_args()
    if args.max_points <= 0:
        raise ValueError(f"--max-points must be positive, got {args.max_points}")

    sequence = args.sequence.expanduser().resolve()
    mesh_path = sequence / "object" / "object_metric.obj"
    trajectory_path = sequence / "object" / "trajectory.npz"
    output = args.output or sequence / "validation" / "object_trajectory_keyframes.png"
    output = output.expanduser().resolve()
    if not mesh_path.is_file() or not trajectory_path.is_file():
        raise FileNotFoundError(f"Missing {mesh_path} or {trajectory_path}")

    cache_root = sequence / "validation" / ".cache"
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("This quick visualization requires matplotlib in the Stage-A environment") from exc

    vertices = read_obj_vertices(mesh_path)
    if len(vertices) > args.max_points:
        indices = np.linspace(0, len(vertices) - 1, args.max_points, dtype=np.int64)
        vertices = vertices[indices]
    with np.load(trajectory_path) as trajectory:
        transforms = trajectory["transform"]
        frame_ids = trajectory["frame_ids"]
    if not len(transforms):
        raise ValueError("Object trajectory is empty")
    key_indices = sorted({0, len(transforms) // 2, len(transforms) - 1})
    colors = ("#0072B2", "#E69F00", "#009E73")

    figure = plt.figure(figsize=(9, 8))
    axis = figure.add_subplot(111, projection="3d")
    all_points: list[np.ndarray] = []
    for color, index in zip(colors, key_indices, strict=False):
        transform = transforms[index]
        points = vertices @ transform[:3, :3].T + transform[:3, 3]
        all_points.append(points)
        axis.scatter(points[:, 0], points[:, 1], points[:, 2], s=1.2, alpha=0.5, color=color,
                     label=f"frame {int(frame_ids[index])}")
        origin = transform[:3, 3]
        for local_axis, axis_color in enumerate(("r", "g", "b")):
            direction = transform[:3, local_axis] * max(np.ptp(vertices, axis=0).max() * 0.35, 0.01)
            axis.quiver(*origin, *direction, color=axis_color, linewidth=1.5)

    combined = np.concatenate(all_points, axis=0)
    center = (combined.min(axis=0) + combined.max(axis=0)) / 2.0
    radius = max(float(np.ptp(combined, axis=0).max()) / 2.0, 1e-3)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_xlabel("CARI4D camera X (m)")
    axis.set_ylabel("CARI4D camera Y (m)")
    axis.set_zlabel("CARI4D camera Z (m)")
    axis.set_title("CARI4D object trajectory: first / middle / last")
    axis.legend()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)
    print(f"Saved trajectory visualization: {output}")


if __name__ == "__main__":
    main()
