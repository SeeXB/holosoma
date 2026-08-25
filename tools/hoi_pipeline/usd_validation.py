"""USD validation helpers; import only after Isaac Sim has been launched."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics

from hoi_pipeline.common import obj_extent, relative_extent_error, write_json


def validate_object_usd(
    usd_path: Path,
    source_obj: Path,
    *,
    expected_collision: str = "convexDecomposition",
    expected_mass: float | None = 1.0,
    bbox_relative_tolerance: float = 1e-2,
) -> dict[str, Any]:
    errors: list[str] = []
    if not np.isfinite(bbox_relative_tolerance) or bbox_relative_tolerance <= 0.0:
        raise ValueError(f"bbox_relative_tolerance must be finite and positive, got {bbox_relative_tolerance}")
    if not usd_path.is_file():
        return {"status": "FAIL", "errors": [f"USD file does not exist: {usd_path}"]}
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        return {"status": "FAIL", "errors": [f"Could not open USD stage: {usd_path}"]}
    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    kilograms_per_unit = float(UsdPhysics.GetStageKilogramsPerUnit(stage))
    if not np.isclose(meters_per_unit, 1.0, rtol=0.0, atol=1e-9):
        errors.append(f"USD stage metersPerUnit must be 1.0, found {meters_per_unit}")
    if not np.isclose(kilograms_per_unit, 1.0, rtol=0.0, atol=1e-9):
        errors.append(f"USD stage kilogramsPerUnit must be 1.0, found {kilograms_per_unit}")
    default_prim = stage.GetDefaultPrim()
    if not default_prim or not default_prim.IsValid():
        errors.append("USD stage has no valid default prim")

    mesh_prims = [prim for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh)]
    if not mesh_prims:
        errors.append("USD contains no visual Mesh prim")
    collision_prims = [prim for prim in mesh_prims if prim.HasAPI(UsdPhysics.CollisionAPI)]
    enabled_collision_prims = []
    approximations: list[str] = []
    for prim in collision_prims:
        enabled = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
        if enabled is not False:
            enabled_collision_prims.append(prim)
        if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            approximation = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
            if approximation:
                approximations.append(str(approximation))
    if expected_collision == "none":
        if enabled_collision_prims:
            errors.append("Collision was requested as none, but an enabled CollisionAPI was found")
    else:
        if not enabled_collision_prims:
            errors.append("No enabled collision Mesh prim found")
        if expected_collision not in approximations:
            errors.append(
                f"Expected collision approximation {expected_collision!r}, found {sorted(set(approximations))}"
            )

    rigid_prims = [prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    mass_prims = [prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.MassAPI)]
    if not rigid_prims:
        errors.append("USD contains no RigidBodyAPI")
    elif not any(UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get() is not False for prim in rigid_prims):
        errors.append("USD contains RigidBodyAPI, but all rigid bodies are disabled")
    if not mass_prims:
        errors.append("USD contains no MassAPI")
    masses = [UsdPhysics.MassAPI(prim).GetMassAttr().Get() for prim in mass_prims]
    masses = [float(value) for value in masses if value is not None]
    if expected_mass is not None:
        if not masses:
            errors.append(f"Expected mass {expected_mass} kg, but no authored mass value was found")
        elif not any(np.isclose(value, expected_mass, rtol=1e-6, atol=1e-7) for value in masses):
            errors.append(f"Expected mass {expected_mass} kg, found {masses}")

    _, _, obj_bbox = obj_extent(source_obj)
    usd_bbox = np.full(3, np.nan)
    bbox_error = np.full(3, np.inf)
    if default_prim and default_prim.IsValid():
        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
            useExtentsHint=False,
        )
        world_range = bbox_cache.ComputeWorldBound(default_prim).ComputeAlignedRange()
        usd_bbox = np.asarray(world_range.GetMax(), dtype=np.float64) - np.asarray(
            world_range.GetMin(), dtype=np.float64
        )
        bbox_error = relative_extent_error(usd_bbox, obj_bbox)
        if not np.all(np.isfinite(usd_bbox)) or np.any(usd_bbox <= 0.0):
            errors.append(f"USD has invalid bounding-box extent {usd_bbox}")
        elif float(bbox_error.max()) > bbox_relative_tolerance:
            ratios = usd_bbox / np.maximum(obj_bbox, 1e-12)
            errors.append(
                f"USD/OBJ bbox mismatch: OBJ={obj_bbox}, USD={usd_bbox}, ratios={ratios}. "
                "Possible coordinate conversion or 100x/1000x scale error."
            )

    ratios = usd_bbox / np.maximum(obj_bbox, 1e-12)
    obvious_scale_error = bool(
        np.any((ratios > 50.0) | (ratios < 0.02)) if np.all(np.isfinite(ratios)) else True
    )
    if obvious_scale_error and not any("bbox mismatch" in error for error in errors):
        errors.append(f"Obvious 100x/1000x scale discrepancy detected; USD/OBJ ratios={ratios}")

    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "usd_file": str(usd_path),
        "source_obj": str(source_obj),
        "default_prim": str(default_prim.GetPath()) if default_prim and default_prim.IsValid() else None,
        "meters_per_unit": meters_per_unit,
        "kilograms_per_unit": kilograms_per_unit,
        "visual_mesh_prims": [str(prim.GetPath()) for prim in mesh_prims],
        "collision_prims": [str(prim.GetPath()) for prim in enabled_collision_prims],
        "collision_approximations": sorted(set(approximations)),
        "expected_collision_approximation": expected_collision,
        "rigid_body_prims": [str(prim.GetPath()) for prim in rigid_prims],
        "mass_prims": [str(prim.GetPath()) for prim in mass_prims],
        "mass_values_kg": masses,
        "expected_mass_kg": expected_mass,
        "obj_bbox_extent_xyz_m": obj_bbox,
        "usd_bbox_extent_xyz_m": usd_bbox,
        "usd_to_obj_bbox_ratio_xyz": ratios,
        "bbox_relative_error_xyz": bbox_error,
        "bbox_relative_tolerance": bbox_relative_tolerance,
        "obvious_scale_error": obvious_scale_error,
    }


def write_usd_report(report_path: Path, report: dict[str, Any]) -> None:
    write_json(report_path, report)
