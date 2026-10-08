#!/usr/bin/env python3
"""Build a compound-object scene while preserving the source robot exactly.

The source MJCF supplies the robot, including its original rubber-hand meshes.
This tool only replaces the object's single non-convex collision mesh with
precomputed compound-convex parts. It cannot add or replace robot hands.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "src/holosoma_retargeting/holosoma_retargeting/demo_data"
DEFAULT_SCENE = DATA / "omomo/official_inputs/scenes/g1_29dof_w_largetable.xml"
DEFAULT_PARTS = DATA / "retarget_inputs/sub1_largetable_028_officialpt_compound_20260917/assets"
ORIGINAL_HAND_BODIES = ("left_rubber_hand_link", "right_rubber_hand_link")
FORBIDDEN_HAND_BODIES = ("left_sphere_hand_link", "right_sphere_hand_link")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _resolve_mesh_paths(root: ET.Element, source_scene: Path) -> None:
    compiler = root.find("compiler")
    mesh_dir = source_scene.parent
    if compiler is not None and compiler.get("meshdir"):
        mesh_dir = source_scene.parent / str(compiler.get("meshdir"))
        compiler.attrib.pop("meshdir", None)
    for mesh in root.findall("asset/mesh"):
        raw = mesh.get("file")
        if not raw:
            continue
        path = Path(raw)
        if not path.is_absolute():
            path = (mesh_dir / path).resolve()
        if not path.exists() and "src" in path.parts:
            # Historical OMOMO scenes were generated under another checkout.
            # Preserve the asset suffix while rebasing it to this repository.
            path = ROOT.joinpath(*path.parts[path.parts.index("src") :]).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"scene mesh asset is missing: {raw} -> {path}")
        mesh.set("file", str(path))


def _collision_geoms(model: mujoco.MjModel, body_name: str) -> tuple[int, ...]:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"body missing from generated scene: {body_name}")
    return tuple(
        index
        for index in range(model.ngeom)
        if int(model.geom_bodyid[index]) == body_id
        and (int(model.geom_contype[index]) != 0 or int(model.geom_conaffinity[index]) != 0)
    )


def build_scene(
    source_scene: Path,
    object_name: str,
    object_parts_dir: Path,
    output: Path,
) -> dict[str, object]:
    source_scene = source_scene.resolve()
    object_parts_dir = object_parts_dir.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    part_paths = sorted(object_parts_dir.glob("part_*.obj"))
    if not part_paths:
        raise ValueError(f"no part_*.obj collision meshes in {object_parts_dir}")

    tree = ET.parse(source_scene)
    root = tree.getroot()
    _resolve_mesh_paths(root, source_scene)
    asset = root.find("asset")
    object_body = root.find(f".//body[@name='{object_name}_link']")
    if asset is None or object_body is None:
        raise ValueError(f"source scene is missing {object_name!r} assets/body")

    body_names = {body.get("name") for body in root.iter("body")}
    missing_hands = set(ORIGINAL_HAND_BODIES).difference(body_names)
    if missing_hands:
        raise ValueError(f"source scene is missing original hands: {sorted(missing_hands)}")
    forbidden = set(FORBIDDEN_HAND_BODIES).intersection(body_names)
    if forbidden:
        raise ValueError(f"source scene replaces the robot hands: {sorted(forbidden)}")

    original = object_body.find(f"geom[@name='{object_name}']")
    if original is None:
        raise ValueError(f"source scene is missing object geom {object_name!r}")
    original.set("contype", "0")
    original.set("conaffinity", "0")
    original.set("group", "1")
    original.set("density", "0")

    existing_meshes = {node.get("name") for node in asset.findall("mesh")}
    existing_geoms = {node.get("name") for node in object_body.findall("geom")}
    if any(name and name.startswith(f"{object_name}_part_") for name in existing_geoms):
        raise ValueError("source scene already contains compound object collision geoms")
    inherited = {
        key: value for key, value in original.attrib.items() if key in {"friction", "solref", "solimp", "pos", "quat"}
    }
    for index, part_path in enumerate(part_paths):
        name = f"{object_name}_part_{index:03d}"
        if name in existing_meshes:
            raise ValueError(f"duplicate mesh asset: {name}")
        ET.SubElement(asset, "mesh", name=name, file=str(part_path))
        ET.SubElement(
            object_body,
            "geom",
            name=name,
            type="mesh",
            mesh=name,
            contype="1",
            conaffinity="1",
            group="3",
            density="0",
            rgba="0.7 0.8 0.9 0.0",
            **inherited,
        )

    keyframe = root.find("keyframe")
    if keyframe is not None:
        root.remove(keyframe)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    tree.write(temporary, encoding="utf-8", xml_declaration=True)

    # Validate the temporary scene before publishing it as a completed artifact.
    model = mujoco.MjModel.from_xml_path(str(temporary))
    if model.nq != 43:
        raise RuntimeError(f"generated scene changed robot/object DOFs: nq={model.nq}")
    hand_geoms = {name: _collision_geoms(model, name) for name in ORIGINAL_HAND_BODIES}
    if any(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) >= 0 for name in FORBIDDEN_HAND_BODIES):
        raise RuntimeError("generated scene unexpectedly contains replacement sphere hands")
    object_geoms = _collision_geoms(model, f"{object_name}_link")
    object_geom_names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index) for index in object_geoms}
    expected_object_geoms = {f"{object_name}_part_{index:03d}" for index in range(len(part_paths))}
    if object_geom_names != expected_object_geoms:
        raise RuntimeError(
            "generated object collision set differs from compound parts: "
            f"actual={sorted(object_geom_names)}, expected={sorted(expected_object_geoms)}"
        )

    temporary.replace(output)
    report = {
        "schema": "holosoma.force_affordance_scene.v2",
        "source_scene": str(source_scene),
        "source_scene_sha256": _sha256(source_scene),
        "output": str(output),
        "output_sha256": _sha256(output),
        "object_name": object_name,
        "collision_representation": "compound_convex",
        "object_collision_parts_dir": str(object_parts_dir),
        "object_collision_parts": [{"path": str(path), "sha256": _sha256(path)} for path in part_paths],
        "model_nq": int(model.nq),
        "robot_hand_policy": "preserve source scene; no replacement bodies",
        "physical_hand_bodies": list(ORIGINAL_HAND_BODIES),
        "physical_hand_collision_geom_ids": {name: list(indices) for name, indices in hand_geoms.items()},
        "forbidden_replacement_bodies_absent": True,
    }
    _write_json(output.with_suffix(".json"), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--object-name", default="largetable")
    parser.add_argument("--object-parts-dir", type=Path, default=DEFAULT_PARTS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build_scene(
                args.source_scene,
                args.object_name,
                args.object_parts_dir,
                args.output,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
