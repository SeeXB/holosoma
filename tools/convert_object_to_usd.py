#!/usr/bin/env python3
"""Convert a meter-scale canonical object OBJ to a dynamic IsaacLab-ready USD."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from isaaclab.app import AppLauncher


COLLISION_CHOICES = ("convexDecomposition", "convexHull", "none")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input", required=True, type=Path, help="Meter-scale object_metric.obj.")
parser.add_argument("--output", required=True, type=Path, help="Destination object.usd.")
parser.add_argument(
    "--collision-approximation",
    choices=COLLISION_CHOICES,
    default="convexDecomposition",
    help="Manipulation collision approximation (default: convexDecomposition).",
)
parser.add_argument("--mass", type=float, default=1.0, help="Dynamic rigid-body mass in kg (default: 1.0).")
parser.add_argument(
    "--mass-source",
    choices=("placeholder", "measured"),
    default="placeholder",
    help="Provenance label only; RGB reconstruction does not estimate mass.",
)
parser.add_argument("--bbox-relative-tolerance", type=float, default=1e-2)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import json
import platform
import subprocess

import isaaclab
import omni.kit.app

from isaaclab.sim.converters import MeshConverter, MeshConverterCfg
from isaaclab.sim.schemas import schemas_cfg

from hoi_pipeline.common import write_json
from hoi_pipeline.usd_validation import validate_object_usd, write_usd_report


def _print_versions() -> dict[str, str]:
    kit_version = str(omni.kit.app.get_app().get_app_version())
    isaac_lab_path = Path(isaaclab.__file__).resolve()
    isaac_lab_root = isaac_lab_path.parents[3]
    try:
        isaac_lab_git = subprocess.run(
            ["git", "-C", str(isaac_lab_root), "describe", "--tags", "--always", "--dirty"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        isaac_lab_git = "unavailable"
    versions = {
        "python": platform.python_version(),
        "isaac_sim": kit_version,
        "isaac_lab": str(getattr(isaaclab, "__version__", "unknown")),
        "isaac_lab_git": isaac_lab_git,
        "isaac_lab_path": str(isaac_lab_path),
    }
    print(f"Isaac Sim version: {versions['isaac_sim']}", flush=True)
    print(f"Isaac Lab version: {versions['isaac_lab']} ({versions['isaac_lab_path']})", flush=True)
    print(f"Isaac Lab git:     {versions['isaac_lab_git']}", flush=True)
    print(f"Python version:    {versions['python']}", flush=True)
    return versions


def _report_path(output: Path) -> Path:
    if output.parent.name == "object":
        return output.parent.parent / "validation" / "usd_report.json"
    return output.with_suffix(".validation.json")


def _update_object_metadata(output: Path, report: dict, versions: dict[str, str]) -> None:
    metadata_path = output.parent / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    metadata.update(
        {
            "isaaclab_usd": str(output),
            "isaaclab_mesh_scale": [1.0, 1.0, 1.0],
            "collision_approximation": args.collision_approximation,
            "rigid_body": True,
            "mass_kg": args.mass,
            "mass_is_placeholder": args.mass_source == "placeholder",
            "mass_source": args.mass_source,
            "usd_validation": report["status"],
            "isaac_versions": versions,
        }
    )
    write_json(metadata_path, metadata)


def main() -> None:
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input mesh does not exist: {input_path}")
    if input_path.suffix.lower() != ".obj":
        raise ValueError(f"The canonical object input must be OBJ, got {input_path.suffix}")
    if output_path.suffix.lower() not in (".usd", ".usda", ".usdc"):
        raise ValueError(f"Output must be a USD file, got {output_path}")
    if not math.isfinite(args.mass) or args.mass <= 0.0:
        raise ValueError(f"Mass must be finite and positive, got {args.mass}")
    if not math.isfinite(args.bbox_relative_tolerance) or args.bbox_relative_tolerance <= 0.0:
        raise ValueError(
            f"--bbox-relative-tolerance must be finite and positive, got {args.bbox_relative_tolerance}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    versions = _print_versions()

    collision_cfg_classes = {
        "convexDecomposition": schemas_cfg.ConvexDecompositionPropertiesCfg,
        "convexHull": schemas_cfg.ConvexHullPropertiesCfg,
        "none": None,
    }
    collision_class = collision_cfg_classes[args.collision_approximation]
    converter_cfg = MeshConverterCfg(
        asset_path=str(input_path),
        usd_dir=str(output_path.parent),
        usd_file_name=output_path.name,
        force_usd_conversion=True,
        make_instanceable=False,
        scale=(1.0, 1.0, 1.0),
        mass_props=schemas_cfg.MassPropertiesCfg(mass=args.mass),
        rigid_props=schemas_cfg.RigidBodyPropertiesCfg(),
        collision_props=schemas_cfg.CollisionPropertiesCfg(
            collision_enabled=args.collision_approximation != "none"
        ),
        mesh_collision_props=collision_class() if collision_class is not None else None,
    )
    print(f"Input metric OBJ:      {input_path}", flush=True)
    print(f"Output USD:            {output_path}", flush=True)
    print("MeshConverter scale:   (1.0, 1.0, 1.0); metric scale is already baked", flush=True)
    print(f"Collision:             {args.collision_approximation}", flush=True)
    print(f"Mass:                  {args.mass} kg ({args.mass_source})", flush=True)
    converter = MeshConverter(converter_cfg)
    generated_path = Path(converter.usd_path).resolve()
    if generated_path != output_path:
        raise RuntimeError(f"MeshConverter wrote an unexpected path: {generated_path} != {output_path}")

    report = validate_object_usd(
        output_path,
        input_path,
        expected_collision=args.collision_approximation,
        expected_mass=args.mass,
        bbox_relative_tolerance=args.bbox_relative_tolerance,
    )
    report["mass_is_placeholder"] = args.mass_source == "placeholder"
    report["versions"] = versions
    report_path = _report_path(output_path)
    write_usd_report(report_path, report)
    _update_object_metadata(output_path, report, versions)
    print(f"OBJ bbox:              {report['obj_bbox_extent_xyz_m']} m", flush=True)
    print(f"USD bbox:              {report['usd_bbox_extent_xyz_m']} m", flush=True)
    print(
        f"Scale check:           {'PASS' if not report['obvious_scale_error'] and not report['errors'] else 'FAIL'}",
        flush=True,
    )
    print(f"RigidBody:             {'PASS' if report['rigid_body_prims'] else 'FAIL'}", flush=True)
    collision_pass = (
        not report["collision_prims"]
        if args.collision_approximation == "none"
        else bool(report["collision_prims"])
    )
    print(f"Collision:             {'PASS' if collision_pass else 'FAIL'}", flush=True)
    print(f"Mass:                  {args.mass} kg ({args.mass_source})", flush=True)
    print(f"USD validation:        {report['status']} ({report_path})", flush=True)
    if report["status"] != "PASS":
        for error in report["errors"]:
            print(f"  - {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
