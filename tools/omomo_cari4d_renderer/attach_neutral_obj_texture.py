#!/usr/bin/env python3
"""Attach a neutral UV texture to a geometry-only OBJ without changing geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import trimesh


def add_uv(token: str) -> str:
    parts = token.split("/")
    if len(parts) == 1:
        return f"{parts[0]}/1"
    if len(parts) == 2:
        return f"{parts[0]}/1"
    if len(parts) == 3 and not parts[1]:
        return f"{parts[0]}/1/{parts[2]}"
    return token


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-obj", type=Path, required=True)
    parser.add_argument("--output-obj", type=Path, required=True)
    args = parser.parse_args()

    args.output_obj.parent.mkdir(parents=True, exist_ok=True)
    source_mesh = trimesh.load(args.input_obj, force="mesh", process=False)
    mtl = args.output_obj.with_suffix(".mtl")
    texture = args.output_obj.with_name(args.output_obj.stem + "_neutral.png")

    output_lines = [f"mtllib {mtl.name}\n"]
    inserted_uv = False
    inserted_material = False
    with args.input_obj.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if line.startswith("mtllib ") or line.startswith("usemtl "):
                continue
            if line.startswith("f "):
                if not inserted_uv:
                    output_lines.append("vt 0.5 0.5\n")
                    inserted_uv = True
                if not inserted_material:
                    output_lines.append("usemtl neutral_geometry_only\n")
                    inserted_material = True
                fields = line.strip().split()
                output_lines.append("f " + " ".join(add_uv(x) for x in fields[1:]) + "\n")
            else:
                output_lines.append(line)
    args.output_obj.write_text("".join(output_lines), encoding="utf-8")
    mtl.write_text(
        "newmtl neutral_geometry_only\n"
        "Ka 0.4 0.4 0.4\n"
        "Kd 0.4 0.4 0.4\n"
        "Ks 0.0 0.0 0.0\n"
        "d 1.0\n"
        "illum 1\n"
        f"map_Kd {texture.name}\n",
        encoding="utf-8",
    )
    cv2.imwrite(str(texture), np.full((2, 2, 3), 102, dtype=np.uint8))

    output_mesh = trimesh.load(args.output_obj, force="mesh", process=False)
    same_vertices = np.array_equal(source_mesh.vertices, output_mesh.vertices)
    same_faces = np.array_equal(source_mesh.faces, output_mesh.faces)
    if not same_vertices or not same_faces:
        raise RuntimeError("Neutral texture attachment changed OBJ geometry")
    report = {
        "schema": "holosoma.neutral_obj_texture.v1",
        "input_obj": str(args.input_obj.resolve()),
        "output_obj": str(args.output_obj.resolve()),
        "material": str(mtl.resolve()),
        "texture": str(texture.resolve()),
        "purpose": "satisfy upstream CARI4D TexturesUV assumption for shape-only Hunyuan output",
        "geometry_or_scale_modified": False,
        "vertices_identical": same_vertices,
        "faces_identical": same_faces,
        "texture_rgb": [102, 102, 102],
    }
    report_path = args.output_obj.with_name(args.output_obj.stem + "_texture_report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
