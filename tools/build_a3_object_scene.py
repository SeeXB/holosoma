#!/usr/bin/env python3
"""Add a free object mesh to an adapted A3 MuJoCo model.

The generated scene keeps the object free joint last, matching the retargeter
layout ``[robot floating base, robot joints, object pose]``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET


def build_scene(robot_xml: Path, object_mesh: Path, object_name: str, output: Path) -> None:
    tree = ET.parse(robot_xml)
    root = tree.getroot()
    asset = root.find("asset")
    worldbody = root.find("worldbody")
    if asset is None or worldbody is None:
        raise ValueError(f"Expected <asset> and <worldbody> in {robot_xml}")
    if root.find(f".//body[@name='{object_name}_link']") is not None:
        raise ValueError(f"Scene already contains {object_name}_link")

    ET.SubElement(
        asset,
        "mesh",
        name=f"{object_name}_mesh",
        file=str(object_mesh.resolve()),
    )
    body = ET.SubElement(worldbody, "body", name=f"{object_name}_link")
    ET.SubElement(body, "freejoint")
    ET.SubElement(
        body,
        "inertial",
        pos="0 0 0",
        mass="0.1",
        diaginertia="0.002 0.002 0.002",
    )
    ET.SubElement(
        body,
        "geom",
        name=object_name,
        type="mesh",
        mesh=f"{object_name}_mesh",
        contype="1",
        conaffinity="1",
        rgba="0.7 0.8 0.9 0.7",
        friction="0.9 0.5 0.5",
        solref="0.02 1",
        solimp="0.9 0.95 0.001",
    )
    ET.indent(tree, space="  ")
    output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output, encoding="unicode")
    output.write_text(output.read_text(encoding="utf-8") + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-xml", required=True, type=Path)
    parser.add_argument("--object-mesh", required=True, type=Path)
    parser.add_argument("--object-name", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    build_scene(args.robot_xml, args.object_mesh, args.object_name, args.output)


if __name__ == "__main__":
    main()
