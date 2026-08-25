#!/usr/bin/env python3
"""Stage a known canonical OBJ as a textured CARI4D tracking template.

This utility is intended for experiments where object shape is known but its
motion is not.  It never reads pose or trajectory labels.  CARI4D still
estimates every object pose from video/depth observations.  For normalized
meshes CARI4D must also estimate metric scale; known metric assets bypass that
scale-estimation stage explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--sequence-name", required=True)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--texture-rgb", nargs=3, type=int, default=(224, 126, 231))
    parser.add_argument(
        "--input-unit",
        choices=("arbitrary", "meter"),
        default="arbitrary",
        help=(
            "Use 'meter' only for an asset whose vertices are already metric. "
            "It is emitted directly as CARI4D's _align.obj and is never rescaled."
        ),
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _face_with_uv(token: str) -> str:
    fields = token.split("/")
    if len(fields) == 1:
        return f"{fields[0]}/1"
    if len(fields) == 2:
        return token if fields[1] else f"{fields[0]}/1"
    if len(fields) == 3:
        return token if fields[1] else f"{fields[0]}/1/{fields[2]}"
    raise ValueError(f"Unsupported OBJ face token: {token!r}")


def main() -> None:
    args = parse_args()
    source = args.input.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not source.is_file() or source.suffix.lower() != ".obj":
        raise FileNotFoundError(f"Input OBJ does not exist: {source}")
    if args.frame_index < 0:
        raise ValueError("--frame-index must be non-negative")
    if any(channel < 0 or channel > 255 for channel in args.texture_rgb):
        raise ValueError("--texture-rgb values must be in [0,255]")

    frame_token = f"{args.frame_index:03d}"
    stem = f"{args.sequence_name}_{frame_token}"
    output_dir = output_root / f"{stem}_rgba"
    output_dir.mkdir(parents=True, exist_ok=True)
    # estimate_scale.py consumes ``_rgba.obj`` and produces ``_align.obj``.
    # A known metric asset must skip that stage, so emit the latter directly.
    artifact_suffix = "align" if args.input_unit == "meter" else "rgba"
    output_obj = output_dir / f"{stem}_{artifact_suffix}.obj"
    output_mtl = output_dir / f"{stem}_{artifact_suffix}.mtl"
    output_texture = output_dir / f"{stem}_{artifact_suffix}.png"

    vertices: list[list[float]] = []
    face_count = 0
    source_has_uv = False
    output_lines = [f"mtllib {output_mtl.name}\n"]
    material_written = False
    for line in source.read_text(encoding="utf-8", errors="strict").splitlines(keepends=True):
        if line.startswith("mtllib ") or line.startswith("usemtl "):
            continue
        if line.startswith("v "):
            vertices.append([float(value) for value in line.split()[1:4]])
        elif line.startswith("vt "):
            source_has_uv = True
        elif line.startswith("f "):
            if not material_written:
                if not source_has_uv:
                    output_lines.append("vt 0.5 0.5\n")
                output_lines.append("usemtl tracking_template_material\n")
                material_written = True
            fields = line.strip().split()
            if not source_has_uv:
                fields = [fields[0], *(_face_with_uv(token) for token in fields[1:])]
            line = " ".join(fields) + "\n"
            face_count += 1
        output_lines.append(line)

    if not vertices or not face_count:
        raise ValueError(f"OBJ contains no usable geometry: {source}")
    output_obj.write_text("".join(output_lines), encoding="utf-8")
    output_mtl.write_text(
        "newmtl tracking_template_material\n"
        "Ka 0.0 0.0 0.0\n"
        "Kd 1.0 1.0 1.0\n"
        "Ks 0.0 0.0 0.0\n"
        "d 1.0\n"
        "illum 1\n"
        f"map_Kd {output_texture.name}\n",
        encoding="utf-8",
    )
    rgb = np.asarray(args.texture_rgb, dtype=np.uint8)
    texture_bgr = np.broadcast_to(rgb[::-1], (16, 16, 3)).copy()
    if not cv2.imwrite(str(output_texture), texture_bgr):
        raise RuntimeError(f"Could not write {output_texture}")

    vertex_array = np.asarray(vertices, dtype=np.float64)
    extent = vertex_array.max(axis=0) - vertex_array.min(axis=0)
    report = {
        "status": "PASS",
        "source_obj": str(source),
        "source_sha256": _sha256(source),
        "tracking_template_obj": str(output_obj),
        "sequence_name": args.sequence_name,
        "frame_index": args.frame_index,
        "mesh_scale_applied": 1.0,
        "input_unit": args.input_unit,
        "source_extent_xyz": extent.tolist(),
        "source_extent_unit": "meter" if args.input_unit == "meter" else "arbitrary_object_unit",
        "num_vertices": len(vertices),
        "num_faces": face_count,
        "shape_source": "user_supplied_known_canonical_obj",
        "texture_source": "constant user-specified RGB tracking texture",
        "uses_known_cad_shape": True,
        "uses_motion_or_object_pose_labels": False,
        "metric_scale_is_known": args.input_unit == "meter",
        "scale_estimation_required": args.input_unit != "meter",
        "next_step": (
            "Run CARI4D FoundationPose directly; do not run estimate_scale_video.py"
            if args.input_unit == "meter"
            else "CARI4D estimate_scale_video.py must estimate metric scale from video depth"
        ),
    }
    report_path = output_dir / "known_cad_provenance.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
