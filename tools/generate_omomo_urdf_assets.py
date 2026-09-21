#!/usr/bin/env python3
"""Build single-rigid-body OMOMO object assets from the official OBJ meshes.

The OMOMO object meshes in ``captured_objects`` are in the canonical object
coordinate system used by the sequence records.  Each record supplies a
nearly-constant ``obj_scale``; this tool uses the median scale of the selected
target sequences, recenters the mesh at its geometric vertex centroid (the
convention used by the existing largebox asset), and writes a mesh-backed URDF.

The generated URDF is intentionally conservative: one free rigid body, a
visual mesh, the same mesh as a collision fallback, and a box-approximation
inertia for the nominal 0.1 kg object used by the current object-WBT recipe.
The manifest records every assumption so that collision proxies and physical
parameters can be refined later without losing provenance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import trimesh


DEFAULT_SOURCE_ROOT = Path("src/holosoma_retargeting/holosoma_retargeting/demo_data/external/omomo")
DEFAULT_OUTPUT_ROOT = Path(
    "src/holosoma_retargeting/holosoma_retargeting/demo_data/models"
)

# The selected OMOMO tasks.  The sequence records are split between the train
# and test pickle files (the latter contains subjects 16 and 17).
TARGET_SEQUENCES = (
    "sub9_clothesstand_058",
    "sub17_floorlamp_026",
    "sub10_largebox_089",
    "sub16_largebox_007",
    "sub1_largetable_028",
    "sub16_largetable_013",
    "sub9_monitor_025",
    "sub15_plasticbox_013",
    "sub7_smallbox_023",
    "sub3_smalltable_013",
    "sub12_smalltable_032",
    "sub15_suitcase_053",
    "sub4_suitcase_031",
    "sub11_trashcan_024",
    "sub12_tripod_041",
    "sub9_tripod_015",
    "sub10_whitechair_118",
    "sub16_whitechair_002",
    "sub15_woodchair_020",
    "sub14_woodchair_004",
)

OBJECT_NAMES = (
    "clothesstand",
    "floorlamp",
    "largebox",
    "largetable",
    "monitor",
    "plasticbox",
    "smallbox",
    "smalltable",
    "suitcase",
    "trashcan",
    "tripod",
    "whitechair",
    "woodchair",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace generated mesh/URDF files (never modifies source OBJ files).",
    )
    parser.add_argument(
        "--include-largebox",
        action="store_true",
        help="Also write a generated largebox asset instead of preserving the validated one.",
    )
    return parser.parse_args()


def _load_records(source_root: Path) -> dict[str, dict[str, Any]]:
    data_root = source_root / "data"
    records: dict[str, dict[str, Any]] = {}
    for filename in (
        "train_diffusion_manip_seq_joints24.p",
        "test_diffusion_manip_seq_joints24.p",
    ):
        path = data_root / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        for record in joblib.load(path).values():
            name = record.get("seq_name")
            if isinstance(name, str):
                records[name] = record
    return records


def _object_name(sequence: str) -> str:
    # Sequence format is sub<subject>_<object>_<index>; object names in this
    # target set contain no underscores.
    parts = sequence.split("_")
    if len(parts) != 3:
        raise ValueError(f"Unexpected OMOMO sequence name: {sequence}")
    return parts[1]


def _source_mesh_path(source_root: Path, object_name: str) -> Path:
    return source_root / "data" / "captured_objects" / f"{object_name}_cleaned_simplified.obj"


def _box_inertia(mass: float, dimensions: np.ndarray) -> tuple[float, float, float]:
    dx, dy, dz = (float(value) for value in dimensions)
    # Uniform-box approximation, with a small floor for degenerate thin meshes.
    ixx = max(mass * (dy * dy + dz * dz) / 12.0, 1.0e-6)
    iyy = max(mass * (dx * dx + dz * dz) / 12.0, 1.0e-6)
    izz = max(mass * (dx * dx + dy * dy) / 12.0, 1.0e-6)
    return ixx, iyy, izz


def _urdf_text(
    object_name: str,
    mesh_filename: str,
    dimensions: np.ndarray,
    mass: float = 0.1,
) -> str:
    ixx, iyy, izz = _box_inertia(mass, dimensions)
    return f'''<?xml version="1.0" ?>
<!-- Generated from OMOMO captured_objects; see omomo_asset_manifest.json. -->
<robot name="{object_name}">
  <dynamics damping="0.5" friction="0.9"/>
  <link name="{object_name}_link">
    <inertial>
      <mass value="{mass:.9g}"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="{ixx:.9g}" ixy="0" ixz="0" iyy="{iyy:.9g}" iyz="0" izz="{izz:.9g}"/>
    </inertial>
    <contact>
      <lateral_friction value="0.9"/>
      <rolling_friction value="0.5"/>
      <stiffness value="30000"/>
      <damping value="1000"/>
    </contact>
    <visual>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry><mesh filename="{mesh_filename}" scale="1 1 1"/></geometry>
      <material name="omomo_{object_name}_material">
        <color rgba="0.7 0.8 0.9 0.7"/>
      </material>
    </visual>
    <collision name="{object_name}">
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry><mesh filename="{mesh_filename}" scale="1 1 1"/></geometry>
    </collision>
  </link>
</robot>
'''


def _write_text(path: Path, content: str, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    records = _load_records(source_root)

    missing_sequences = [name for name in TARGET_SEQUENCES if name not in records]
    if missing_sequences:
        raise RuntimeError(f"Target sequences missing from OMOMO records: {missing_sequences}")

    manifest: dict[str, Any] = {
        "source_root": str(source_root),
        "record_files": [
            str(source_root / "data" / "train_diffusion_manip_seq_joints24.p"),
            str(source_root / "data" / "test_diffusion_manip_seq_joints24.p"),
        ],
        "coordinate_convention": "mesh centered at vertex centroid; metric scale is median OMOMO obj_scale",
        "nominal_mass_kg": 0.1,
        "inertia_model": "uniform box approximation from scaled mesh AABB",
        "collision_model": "same cleaned simplified mesh as visual fallback",
        "objects": {},
    }

    for object_name in OBJECT_NAMES:
        if object_name == "largebox" and not args.include_largebox:
            existing_dir = output_root / "largebox"
            existing_mesh = existing_dir / "largebox.obj"
            existing_urdf = existing_dir / "largebox.urdf"
            if not existing_mesh.is_file() or not existing_urdf.is_file():
                raise FileNotFoundError(
                    "Validated largebox asset is missing; rerun with --include-largebox"
                )
            mesh = trimesh.load_mesh(existing_mesh, force="mesh", process=False)
            dims = np.ptp(np.asarray(mesh.vertices), axis=0)
            manifest["objects"][object_name] = {
                "status": "preserved_existing_validated_asset",
                "mesh": str(existing_mesh),
                "urdf": str(existing_urdf),
                "dimensions_m": dims.tolist(),
                "source_mesh": str(_source_mesh_path(source_root, object_name)),
                "selected_sequences": [
                    name for name in TARGET_SEQUENCES if _object_name(name) == object_name
                ],
            }
            continue

        selected = [
            name for name in TARGET_SEQUENCES if _object_name(name) == object_name
        ]
        if not selected:
            raise RuntimeError(f"No selected target sequence for object {object_name}")
        source_mesh_path = _source_mesh_path(source_root, object_name)
        if not source_mesh_path.is_file():
            raise FileNotFoundError(source_mesh_path)

        scales = np.concatenate(
            [np.asarray(records[name]["obj_scale"], dtype=np.float64).reshape(-1) for name in selected]
        )
        median_scale = float(np.median(scales))
        source_mesh = trimesh.load_mesh(source_mesh_path, force="mesh", process=False)
        vertices = np.asarray(source_mesh.vertices, dtype=np.float64)
        faces = np.asarray(source_mesh.faces, dtype=np.int64)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
            raise ValueError(f"Invalid vertices in {source_mesh_path}: {vertices.shape}")
        if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
            raise ValueError(f"Invalid faces in {source_mesh_path}: {faces.shape}")
        if not np.isfinite(vertices).all() or not np.isfinite(scales).all():
            raise ValueError(f"Non-finite mesh/scale values in {source_mesh_path}")

        # The existing validated largebox mesh is source OBJ * obj_scale,
        # recentered at the source vertex centroid.  Keep the same convention.
        centered_scaled = (vertices - vertices.mean(axis=0)) * median_scale
        generated_mesh = trimesh.Trimesh(
            vertices=centered_scaled,
            faces=faces,
            process=False,
        )
        dimensions = np.ptp(centered_scaled, axis=0)
        if not np.isfinite(dimensions).all() or np.any(dimensions <= 0):
            raise ValueError(f"Degenerate scaled mesh for {object_name}: {dimensions}")

        object_dir = output_root / object_name
        object_dir.mkdir(parents=True, exist_ok=True)
        mesh_path = object_dir / f"{object_name}.obj"
        urdf_path = object_dir / f"{object_name}.urdf"
        if not mesh_path.exists() or args.overwrite:
            generated_mesh.export(mesh_path)
        urdf = _urdf_text(object_name, f"{object_name}.obj", dimensions)
        _write_text(urdf_path, urdf, args.overwrite)

        # Keep a template beside the existing largebox template convention.
        template_path = output_root / "templates" / f"{object_name}.urdf.jinja"
        _write_text(template_path, urdf, args.overwrite)

        manifest["objects"][object_name] = {
            "status": "generated_from_official_obj",
            "source_mesh": str(source_mesh_path),
            "mesh": str(mesh_path),
            "urdf": str(urdf_path),
            "template": str(template_path),
            "selected_sequences": selected,
            "obj_scale_min": float(scales.min()),
            "obj_scale_median": median_scale,
            "obj_scale_max": float(scales.max()),
            "source_vertex_centroid": vertices.mean(axis=0).tolist(),
            "dimensions_m": dimensions.tolist(),
            "vertex_count": int(len(vertices)),
            "face_count": int(len(faces)),
        }

    manifest_path = output_root / "omomo_asset_manifest.json"
    _write_text(
        manifest_path,
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        overwrite=True,
    )
    print(json.dumps({"manifest": str(manifest_path), "objects": manifest["objects"]}, indent=2))


if __name__ == "__main__":
    main()
