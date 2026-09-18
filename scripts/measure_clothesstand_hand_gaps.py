"""Reference-pose gap between RL half-sphere hands and shared convex object parts.

Uses reference wrist FK (known to differ from training URDF by ~3.3 mm),
actual URDF hand mounting transforms, and MuJoCo convex geomDistance.
This is geometric clearance, not force closure or success certification.
"""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--reference", type=Path, required=True)
objects = parser.add_mutually_exclusive_group(required=True)
objects.add_argument("--parts-dir", type=Path)
objects.add_argument("--object-mesh", type=Path)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
scene = ET.Element("mujoco")
ET.SubElement(scene, "compiler", angle="radian", fusestatic="false")
assets, world = ET.SubElement(scene, "asset"), ET.SubElement(scene, "worldbody")
ET.SubElement(assets, "mesh", name="hand", file=str(ROOT / "src/holosoma/holosoma/data/robots/g1/meshes/half_sphere.obj"))
for side in ("left", "right"):
    ET.SubElement(ET.SubElement(world, "body", name=side), "geom", name=side, type="mesh", mesh="hand")
obj = ET.SubElement(world, "body", name="object")
part_names = []
mesh_paths = sorted(args.parts_dir.glob("part_*.obj")) if args.parts_dir else [args.object_mesh]
for i, path in enumerate(mesh_paths):
    name = f"part{i}"
    part_names.append(name)
    ET.SubElement(assets, "mesh", name=name, file=str(path.resolve()))
    ET.SubElement(obj, "geom", name=name, type="mesh", mesh=name)
m = mujoco.MjModel.from_xml_string(ET.tostring(scene, encoding="unicode"))
d = mujoco.MjData(m)
ref = np.load(args.reference, allow_pickle=False)
rows = []
sample_frames = np.linspace(0, len(ref["object_pos_w"]) - 1, 7).round().astype(int)
for f in sample_frames:
    f = int(f)
    m.body("object").pos[:] = ref["object_pos_w"][f]
    m.body("object").quat[:] = ref["object_quat_w"][f]
    for side in ("left", "right"):
        wrist = list(ref["body_names"]).index(f"{side}_wrist_yaw_link")
        rot = Rotation.from_quat(ref["body_quat_w"][f, wrist][[1, 2, 3, 0]])
        m.body(side).pos[:] = ref["body_pos_w"][f, wrist] + rot.apply([.029, -.003 if side == "left" else .003, 0])
        m.body(side).quat[:] = (rot * Rotation.from_euler("y", 1.57)).as_quat()[[3, 0, 1, 2]]
    mujoco.mj_forward(m, d)
    row = {"frame": f, "time_s": f / float(ref["fps"].item())}
    for side in ("left", "right"):
        row[f"{side}_gap_m"] = min(float(mujoco.mj_geomDistance(m, d, m.geom(side).id, m.geom(name).id,
                                                               5.0, np.zeros(6))) for name in part_names)
    rows.append(row)
report = {"reference": str(args.reference), "parts": len(part_names), "rows": rows,
          "note": "Reference wrist FK plus actual RL hand geometry/mount; ~3.3mm wrist FK asset discrepancy remains."}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
