#!/usr/bin/env python3
"""Convert a Hunyuan GLB into CARI4D's expected normalized OBJ layout.

The conversion is deliberately outside CARI4D.  It preserves the GLB's frame
and scale and only bounds topology size for downstream rasterization/tracking.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pymeshlab
import trimesh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-glb", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--frame-index", type=int, required=True)
    parser.add_argument("--target-faces", type=int, default=50_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tag = f"{args.sequence}_{args.frame_index:03d}"
    out_dir = args.output_root / f"{tag}_rgba"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_obj = out_dir / f"{tag}_raw.obj"
    output_obj = out_dir / f"{tag}_align.obj"

    scene = trimesh.load(args.input_glb, force="scene", process=False)
    if not scene.geometry:
        raise RuntimeError(f"GLB has no geometry: {args.input_glb}")
    mesh = scene.to_geometry()
    before_bounds = np.asarray(mesh.bounds, dtype=np.float64)
    before_faces = len(mesh.faces)
    mesh.export(raw_obj)

    if args.target_faces > 0 and before_faces > args.target_faces:
        mesh_set = pymeshlab.MeshSet()
        mesh_set.load_new_mesh(str(raw_obj))
        mesh_set.meshing_decimation_quadric_edge_collapse(
            targetfacenum=args.target_faces,
            preservenormal=True,
            preservetopology=True,
            planarquadric=True,
        )
        mesh_set.save_current_mesh(str(output_obj), save_vertex_normal=True)
    else:
        raw_obj.replace(output_obj)

    staged = trimesh.load(output_obj, force="mesh", process=False)
    after_bounds = np.asarray(staged.bounds, dtype=np.float64)
    extent_error = np.abs(np.ptp(after_bounds, axis=0) - np.ptp(before_bounds, axis=0))
    extent_error /= np.maximum(np.ptp(before_bounds, axis=0), 1e-12)
    if not np.isfinite(staged.vertices).all() or len(staged.faces) == 0:
        raise RuntimeError("Converted mesh is invalid")
    if float(extent_error.max()) > 2e-3:
        raise RuntimeError(f"Decimation changed mesh extents: {extent_error.tolist()}")
    raw_obj.unlink(missing_ok=True)

    metadata = {
        "schema": "holosoma.cari4d_hunyuan_mesh_stage.v1",
        "source_glb": str(args.input_glb.resolve()),
        "output_obj": str(output_obj.resolve()),
        "sequence": args.sequence,
        "frame_index": args.frame_index,
        "conversion": "trimesh GLB scene flatten + MeshLab quadric decimation",
        "coordinate_or_scale_transform_applied": False,
        "faces_before": before_faces,
        "faces_after": int(len(staged.faces)),
        "bounds_before": before_bounds.tolist(),
        "bounds_after": after_bounds.tolist(),
        "max_relative_extent_error": float(extent_error.max()),
    }
    metadata_path = out_dir / f"{tag}_stage.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
