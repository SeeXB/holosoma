"""Extract reference hand/object surface anchors, adapted to the training hand.

Run offline in hsretargeting. Contacts are inferred from reference convex-mesh
distance, not measured human contact annotations or force labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def origin(element):
    if element is None:
        return np.eye(4)
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler("xyz", np.fromstring(element.get("rpy", "0 0 0"), sep=" ")).as_matrix()
    transform[:3, 3] = np.fromstring(element.get("xyz", "0 0 0"), sep=" ")
    return transform


def training_hand(urdf, side):
    root = ET.parse(urdf).getroot()
    joint = root.find(f"joint[@name='{side}_hand_palm_joint']")
    if joint is None or joint.get("type") != "fixed":
        raise ValueError("Expected a fixed hand attachment")
    body = joint.find("parent").get("link")
    link = joint.find("child").get("link")
    collision = root.find(f"link[@name='{link}']/collision")
    mesh = collision.find("geometry/mesh")
    mesh_file = (urdf.parent / mesh.get("filename")).resolve()
    shape = trimesh.load(mesh_file, force="mesh", process=False)
    shape.vertices *= np.fromstring(mesh.get("scale", "1 1 1"), sep=" ")
    shape.apply_transform(origin(joint.find("origin")) @ origin(collision.find("origin")))
    return body, shape.convex_hull, mesh_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--robot-urdf", type=Path, required=True)
    parser.add_argument("--object-urdf", type=Path, required=True)
    parser.add_argument("--object-geom", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contact-distance", type=float, default=0.03)
    args = parser.parse_args()
    if not 0 < args.contact_distance <= 0.05:
        raise ValueError("Expected reference proximity threshold in (0, 0.05] m")
    model = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    data = mujoco.MjData(model)
    object_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, args.object_geom)
    if object_geom < 0:
        raise ValueError("Object geometry missing")
    object_body = model.geom_bodyid[object_geom]
    hands = [training_hand(args.robot_urdf, side) for side in ("left", "right")]
    dependencies = {args.scene.resolve(), args.robot_urdf.resolve(), args.object_urdf.resolve()}
    dependencies.update(hand[2] for hand in hands)
    for mesh in ET.parse(args.object_urdf).iter("mesh"):
        dependencies.add((args.object_urdf.parent / mesh.get("filename")).resolve())
    # Scene mesh paths in these task scenes are authored absolute.
    scene_root = ET.parse(args.scene).getroot()
    compiler = scene_root.find("compiler")
    meshdir = compiler.get("meshdir", "") if compiler is not None else ""
    for mesh in scene_root.findall("asset/mesh"):
        dependencies.add((args.scene.parent / meshdir / mesh.get("file")).resolve())
    with np.load(args.motion, allow_pickle=False) as ref:
        frames = len(ref["joint_pos"])
        fps = float(ref["fps"].item())
        hand_local = np.zeros((frames, 2, 3))
        object_local = np.zeros_like(hand_local)
        distance = np.zeros((frames, 2))
        residual = np.zeros_like(distance)
        names = ref["body_names"].tolist()
        for t in range(frames):
            data.qpos[:] = np.r_[ref["joint_pos"][t], ref["object_pos_w"][t], ref["object_quat_w"][t]]
            mujoco.mj_forward(model, data)
            obj_pos, obj_rot = data.xpos[object_body], data.xmat[object_body].reshape(3, 3)
            np.testing.assert_allclose(obj_pos, ref["object_pos_w"][t], atol=1e-7)
            for h, (side, (body, hull, _)) in enumerate(zip(("left", "right"), hands)):
                wrist = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
                geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_rubber_hand_link")
                if wrist < 0 or geom < 0:
                    raise ValueError("Reference wrist/hand geometry missing")
                np.testing.assert_allclose(data.xpos[wrist], ref["body_pos_w"][t, names.index(body)], atol=1e-6)
                pair = np.zeros(6)
                distance[t, h] = mujoco.mj_geomDistance(model, data, geom, object_geom, 2.0, pair)
                if distance[t, h] >= 2.0:
                    raise ValueError("Reference hand outside distance query range")
                # Keep the reference point on the OBJECT, then map it onto the
                # nearest point of the actual half-sphere hull in wrist coordinates.
                target = pair[3:]
                object_local[t, h] = obj_rot.T @ (target - obj_pos)
                wrist_local_target = data.xmat[wrist].reshape(3, 3).T @ (target - data.xpos[wrist])
                points, gap, _ = trimesh.proximity.closest_point_naive(hull, wrist_local_target[None])
                hand_local[t, h] = points[0]
                residual[t, h] = gap[0]
    active = distance <= args.contact_distance
    if not active.any(axis=0).all():
        raise ValueError("No reference contact detected for one or both hands")
    metadata = {
        "schema": "holosoma.reference_surface_contacts.v1", "fps": fps,
        "motion_sha256": sha(args.motion), "motion_file": str(args.motion.resolve()),
        "source": "retargeted reference convex-mesh closest object surface points; inferred proximity, not human annotations",
        "reference_contact_distance_m": args.contact_distance,
        "hand_mapping": "nearest training hand convex-hull surface point in wrist frame",
        "geometry_sha256": {str(p): sha(p) for p in sorted(dependencies)},
        "active_frames_per_hand": active.sum(axis=0).tolist(),
        "active_spans_inclusive": [[int(np.flatnonzero(a)[0]), int(np.flatnonzero(a)[-1])] for a in active.T],
        "reference_pose_training_hand_gap_mean_m": float(residual[active].mean()),
        "reference_pose_training_hand_gap_max_m": float(residual[active].max()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    np.savez_compressed(args.output, hand_points_local=hand_local.astype(np.float32),
                        object_points_local=object_local.astype(np.float32), active=active,
                        body_names=np.array([h[0] for h in hands]), reference_distance_m=distance,
                        reference_pose_training_hand_gap_m=residual,
                        metadata_json=np.array(json.dumps(metadata)))
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({**metadata, "artifact_sha256": sha(args.output)}, indent=2))


if __name__ == "__main__":
    main()
