"""Build matching MJCF/URDF convex-part colliders in an isolated experiment directory."""
import argparse
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import trimesh

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--input-root", type=Path, required=True)
args = parser.parse_args()
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=True)
asset_dir = out / "assets"
asset_dir.mkdir(exist_ok=True)
source = ROOT / "src/holosoma_retargeting/holosoma_retargeting/demo_data/models/clothesstand/clothesstand.obj"
mesh = trimesh.load(source, force="mesh")
settings = dict(maxConvexHulls=64, resolution=1000000, minimumVolumePercentErrorAllowed=0.5,
                maxNumVerticesPerCH=64, shrinkWrap=True)
parts = mesh.convex_decomposition(**settings)
paths = []
for i, part in enumerate(parts):
    path = asset_dir / f"part_{i:03d}.obj"
    part.export(path)
    paths.append(path)

urdf = ET.parse(source.with_suffix(".urdf"))
link = urdf.getroot().find("link")
for collision in list(link.findall("collision")):
    link.remove(collision)
for visual_mesh in link.findall("visual/geometry/mesh"):
    visual_mesh.set("filename", str(source))
for i, path in enumerate(paths):
    collision = ET.SubElement(link, "collision", name=f"clothesstand_part_{i:03d}")
    ET.SubElement(collision, "origin", xyz="0 0 0", rpy="0 0 0")
    ET.SubElement(ET.SubElement(collision, "geometry"), "mesh", filename=str(path), scale="1 1 1")
urdf.write(asset_dir / "clothesstand.urdf", encoding="utf-8", xml_declaration=True)

scene = ET.parse(args.input_root / "scenes/g1_29dof_w_clothesstand.xml")
asset = scene.getroot().find("asset")
body = scene.getroot().find(".//body[@name='clothesstand_link']")
original = body.find("geom[@name='clothesstand']")
original.set("contype", "0")
original.set("conaffinity", "0")
original.set("group", "1")
for i, path in enumerate(paths):
    name = f"clothesstand_part_{i:03d}"
    ET.SubElement(asset, "mesh", name=name, file=str(path))
    ET.SubElement(body, "geom", name=name, type="mesh", mesh=name, contype="1", conaffinity="1",
                  group="3", rgba="0.7 0.8 0.9 0.0", friction="0.9 0.5 0.5", solref="0.02 1")
scene_path = out / "input/scenes/g1_29dof_w_clothesstand.xml"
scene_path.parent.mkdir(parents=True, exist_ok=True)
scene.write(scene_path, encoding="utf-8", xml_declaration=True)
shutil.copy2(args.input_root / "sub9_clothesstand_058.pt", out / "input/sub9_clothesstand_058.pt")
report = {"source": str(source), "settings": settings, "parts": len(parts),
          "source_volume_m3": float(mesh.volume), "single_hull_volume_m3": float(mesh.convex_hull.volume),
          "sum_part_volumes_m3": float(sum(p.volume for p in parts)),
          "urdf": str(asset_dir / "clothesstand.urdf"), "scene": str(scene_path)}
(out / "collision_manifest.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2), flush=True)
