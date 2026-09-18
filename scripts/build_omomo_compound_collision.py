"""Build shared MJCF/URDF convex parts without changing source geometry or poses."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import mujoco
import trimesh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--object", required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-hulls", type=int, default=32)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = root / "src/holosoma_retargeting/holosoma_retargeting/models" / args.object / f"{args.object}.obj"
    out = args.output.resolve()
    if (out / "collision_manifest.json").exists():
        raise FileExistsError("Use a new output directory to preserve collision provenance")
    assets = out / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.load(source, force="mesh")
    settings = dict(maxConvexHulls=args.max_hulls, resolution=1000000,
                    minimumVolumePercentErrorAllowed=0.5, maxNumVerticesPerCH=64, shrinkWrap=True)
    parts = mesh.convex_decomposition(**settings)
    paths = []
    for index, part in enumerate(parts):
        path = assets / f"part_{index:03d}.obj"
        part.export(path)
        paths.append(path)
    urdf = ET.parse(source.with_suffix(".urdf"))
    link = urdf.getroot().find("link")
    for collision in list(link.findall("collision")):
        link.remove(collision)
    for visual in link.findall("visual/geometry/mesh"):
        visual.set("filename", str(source))
    for index, path in enumerate(paths):
        collision = ET.SubElement(link, "collision", name=f"{args.object}_part_{index:03d}")
        ET.SubElement(collision, "origin", xyz="0 0 0", rpy="0 0 0")
        ET.SubElement(ET.SubElement(collision, "geometry"), "mesh", filename=str(path), scale="1 1 1")
    urdf_path = assets / f"{args.object}.urdf"
    urdf.write(urdf_path, encoding="utf-8", xml_declaration=True)

    source_scene = args.input_root.resolve() / f"scenes/g1_29dof_w_{args.object}.xml"
    scene = ET.parse(source_scene)
    compiler = scene.getroot().find("compiler")
    meshdir = source_scene.parent / compiler.get("meshdir", ".")
    for node in scene.getroot().findall("asset/mesh"):
        p = Path(node.get("file"))
        node.set("file", str(p if p.is_absolute() else (meshdir / p).resolve()))
    asset = scene.getroot().find("asset")
    body = scene.getroot().find(f".//body[@name='{args.object}_link']")
    original = body.find(f"geom[@name='{args.object}']")
    original.set("contype", "0")
    original.set("conaffinity", "0")
    original.set("group", "1")
    original.set("density", "0")
    for index, path in enumerate(paths):
        name = f"{args.object}_part_{index:03d}"
        ET.SubElement(asset, "mesh", name=name, file=str(path))
        attrs = {k: v for k,v in original.attrib.items() if k in ("friction", "solref", "solimp", "pos", "quat")}
        ET.SubElement(body, "geom", name=name, type="mesh", mesh=name, contype="1", conaffinity="1",
                      group="3", density="0", rgba="0.7 0.8 0.9 0.0", **attrs)
    scene_path = out / f"input/scenes/g1_29dof_w_{args.object}.xml"
    scene_path.parent.mkdir(parents=True, exist_ok=True)
    scene.write(scene_path, encoding="utf-8", xml_declaration=True)
    input_pt = args.input_root / f"{args.task}.pt"
    shutil.copy2(input_pt, out / "input" / input_pt.name)
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    report = dict(task=args.task, object=args.object, source=str(source),
                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  input_sha256=hashlib.sha256(input_pt.read_bytes()).hexdigest(),
                  settings=settings, parts=len(parts), source_watertight=bool(mesh.is_watertight),
                  source_volume_m3=float(mesh.volume), single_hull_volume_m3=float(mesh.convex_hull.volume),
                  sum_part_volumes_m3=float(sum(p.volume for p in parts)),
                  urdf=str(urdf_path), scene=str(scene_path), nq=model.nq,
                  note="Non-watertight source volume is indicative only. Same convex parts are used for retargeting and RL.")
    (out / "collision_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
