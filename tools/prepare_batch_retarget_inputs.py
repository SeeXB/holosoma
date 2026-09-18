#!/usr/bin/env python3
"""Prepare reproducible batch inputs for the 20 selected OMOMO sequences.

The retargeting entry point historically consumes the 591-column InterMimic
``.pt`` container.  The renderer exports a clearer OMOMO ``.npz`` bundle, so
this adapter keeps the exact GT joints/object poses while packaging only the
fields consumed by the legacy loader.  It also materializes one MuJoCo scene
per reconstructed object category from the validated G1 largebox template.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation

from omomo_cari4d_renderer.smplh_joint_order import (
    SMPLH_RETARGET_JOINT_NAMES,
    SMPLH_RETARGET_LAYOUT,
    validate_retarget_smplh_geometry,
)


TASK_OBJECTS = {
    "sub3_largebox_003": "largebox",
    "sub10_largebox_089": "largebox",
    "sub1_largetable_028": "largetable",
    "sub16_largetable_013": "largetable",
    "sub9_monitor_025": "monitor",
    "sub11_monitor_127": "monitor",
    "sub15_plasticbox_013": "plasticbox",
    "sub1_plasticbox_077": "plasticbox",
    "sub7_smallbox_023": "smallbox",
    "sub3_smalltable_013": "smalltable",
    "sub12_smalltable_032": "smalltable",
    "sub15_suitcase_053": "suitcase",
    "sub4_suitcase_031": "suitcase",
    "sub11_trashcan_024": "trashcan",
    "sub12_tripod_041": "tripod",
    "sub9_tripod_015": "tripod",
    "sub10_whitechair_118": "whitechair",
    "sub16_whitechair_002": "whitechair",
    "sub15_woodchair_020": "woodchair",
    "sub14_woodchair_004": "woodchair",
}


def _object_poses(bundle: dict[str, np.ndarray]) -> np.ndarray:
    if "object_poses_wxyz_xyz" in bundle:
        return np.asarray(bundle["object_poses_wxyz_xyz"], dtype=np.float32)
    rotation = np.asarray(bundle["object_rotation"], dtype=np.float64)
    translation = np.asarray(bundle["object_translation"], dtype=np.float32)
    xyzw = Rotation.from_matrix(rotation).as_quat()
    wxyz = xyzw[:, [3, 0, 1, 2]].astype(np.float32)
    return np.concatenate([wxyz, translation], axis=1)


def _write_pt(path: Path, human_joints: np.ndarray, object_poses: np.ndarray) -> None:
    if human_joints.ndim != 3 or human_joints.shape[1:] != (52, 3):
        raise ValueError(f"expected human_joints [T,52,3], got {human_joints.shape}")
    if object_poses.shape != (human_joints.shape[0], 7):
        raise ValueError(f"expected object poses [T,7], got {object_poses.shape}")
    if not np.isfinite(human_joints).all() or not np.isfinite(object_poses).all():
        raise ValueError("human joints and object poses must be finite")
    if not np.allclose(np.linalg.norm(object_poses[:, :4], axis=1), 1.0, atol=1e-4, rtol=0):
        raise ValueError("object poses must contain unit wxyz quaternions before packing")
    # Actual InterMimic reader consumes xyz + xyzw (NOT xyzw + xyz).
    # Its [6,3,4,5,0,1,2] permutation returns canonical wxyz + xyz.
    packed = torch.zeros((len(human_joints), 591), dtype=torch.float32)
    packed[:, 162:318] = torch.from_numpy(human_joints.astype(np.float32).reshape(len(human_joints), -1))
    wxyz_xyz = object_poses.astype(np.float32)
    packed[:, 318:325] = torch.from_numpy(wxyz_xyz[:, [4, 5, 6, 1, 2, 3, 0]])
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(packed, path)
    # Exercise the real consumer, including serialization, before declaring an
    # input ready. This would have caught the previous position/quaternion swap.
    from holosoma_retargeting.src.utils import load_intermimic_data
    restored_human, restored_object = load_intermimic_data(str(path))
    np.testing.assert_array_equal(restored_human, human_joints.astype(np.float32))
    np.testing.assert_array_equal(restored_object, wxyz_xyz)


def canonicalize_object_poses(object_poses, source_vertices, object_scales, asset_vertices):
    """Move poses to the generated asset origin, preserving world geometry.

    OMOMO uses s R v + t; assets use v_asset = s_asset (v-c_source)
    (possibly with a nonzero asset centroid). Moving the local origin requires
    t_asset = t + R(s*c_source - c_asset). Per-frame scale jitter remains an
    explicitly measured approximation when using one rigid asset.
    """
    source = np.asarray(source_vertices, dtype=np.float64)
    asset = np.asarray(asset_vertices, dtype=np.float64)
    scales = np.asarray(object_scales, dtype=np.float64).reshape(-1)
    pose = np.asarray(object_poses, dtype=np.float64).copy()
    if source.shape != asset.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("Canonical mesh must preserve source vertex correspondence")
    cs, ca = source.mean(0), asset.mean(0)
    centered = source - cs
    scale = float(np.sum(centered * (asset - ca)) / np.sum(centered * centered))
    residual = np.linalg.norm(asset - ca - scale * centered, axis=1)
    if scale <= 0 or residual.max() > 1e-4:
        raise ValueError("Canonical asset is not a centered uniformly scaled source mesh")
    rotation = Rotation.from_quat(pose[:, [1, 2, 3, 0]])
    shifts = rotation.apply(scales[:, None] * cs - ca)
    pose[:, 4:] += shifts
    max_radius = np.linalg.norm(centered, axis=1).max()
    return pose.astype(np.float32), {
        "source_centroid": cs.tolist(), "asset_centroid": ca.tolist(),
        "asset_scale": scale, "mesh_fit_max_error_m": float(residual.max()),
        "origin_shift_first_m": shifts[0].tolist(),
        "origin_shift_norm_max_m": float(np.linalg.norm(shifts, axis=1).max()),
        "rigid_scale_jitter_max_vertex_error_bound_m": float(np.max(np.abs(scales-scale))*max_radius + residual.max()),
    }


def _write_scene(template: Path, destination: Path, object_name: str, object_mesh: Path) -> None:
    text = template.read_text(encoding="utf-8")
    if 'name="largebox_mesh"' not in text:
        tree = ET.parse(template)
        root = tree.getroot()
        compiler = root.find("compiler")
        if compiler is not None and compiler.get("meshdir"):
            compiler.set("meshdir", str((template.parent / compiler.get("meshdir")).resolve()))
        asset = root.find("asset")
        worldbody = root.find("worldbody")
        if asset is None or worldbody is None:
            raise ValueError(f"scene template lacks asset/worldbody: {template}")
        ET.SubElement(asset, "mesh", name=f"{object_name}_mesh", file=str(object_mesh.resolve()))
        body = ET.SubElement(worldbody, "body", name=f"{object_name}_link")
        ET.SubElement(body, "freejoint")
        ET.SubElement(body, "inertial", pos="0 0 0", mass="0.1", diaginertia="0.002 0.002 0.002")
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
        key = root.find("keyframe/key")
        if key is not None and key.get("qpos"):
            key.set("qpos", key.get("qpos") + " 0 0 0 1 0 0 0")
        destination.parent.mkdir(parents=True, exist_ok=True)
        tree.write(destination, encoding="unicode")
        return
    # The template uses a path relative to the repository's models/g1 folder;
    # batch scenes live under exp/, so make the robot asset directory explicit.
    text = text.replace('meshdir="assets/"', f'meshdir="{(template.parent / "assets").resolve()}/"', 1)
    text = text.replace('name="largebox_mesh"', f'name="{object_name}_mesh"')
    text = text.replace('file="../../largebox/largebox.obj"', f'file="{object_mesh.resolve()}"')
    text = text.replace('name="largebox_link"', f'name="{object_name}_link"')
    text = text.replace('name="largebox" type="mesh" mesh="largebox_mesh"',
                        f'name="{object_name}" type="mesh" mesh="{object_name}_mesh"')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, default=Path("exp/omomo_cari4d"))
    parser.add_argument("--output-root", type=Path, default=Path("exp/retargeting/omomo_batch/input"))
    parser.add_argument("--models-root", type=Path,
                        default=Path("src/holosoma_retargeting/holosoma_retargeting/models"))
    parser.add_argument("--template", type=Path,
                        default=Path("src/holosoma_retargeting/holosoma_retargeting/models/g1/g1_29dof_w_largebox.xml"))
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=list(TASK_OBJECTS),
        default=None,
        help="Optional subset; the manifest contains exactly the selected tasks.",
    )
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    scene_root = args.output_root / "scenes"
    written = []
    selected_tasks = args.tasks or list(TASK_OBJECTS)
    for task_name in selected_tasks:
        object_name = TASK_OBJECTS[task_name]
        sequence_dir = args.bundle_root / task_name
        if task_name == "sub3_largebox_003":
            sequence_dir = args.bundle_root / "sub03_largebox3"
        bundle_path = sequence_dir / "input" / "omomo_gt_sequence.npz"
        if not bundle_path.is_file():
            raise FileNotFoundError(bundle_path)
        with np.load(bundle_path, allow_pickle=False) as source:
            # Avoid decompressing unused full human/object vertex sequences.
            needed = ("human_joints", "human_joint_layout", "smplh_joint_names",
                      "object_poses_wxyz_xyz", "object_rotation", "object_translation",
                      "object_vertices_local", "object_scale")
            bundle = {key: source[key] for key in needed if key in source}
        human_joints = np.asarray(bundle["human_joints"], dtype=np.float32)
        layout = str(np.asarray(bundle.get("human_joint_layout", "")).item())
        names = tuple(str(value) for value in np.asarray(bundle.get("smplh_joint_names", [])).tolist())
        if layout != SMPLH_RETARGET_LAYOUT or names != SMPLH_RETARGET_JOINT_NAMES:
            raise ValueError(
                f"{bundle_path} lacks verified {SMPLH_RETARGET_LAYOUT} joint ordering; "
                "regenerate it with tools/omomo_cari4d_renderer/prepare_sequence.py"
            )
        validate_retarget_smplh_geometry(human_joints)
        object_poses = _object_poses(bundle)

        mesh = args.models_root / object_name / f"{object_name}.obj"
        urdf = args.models_root / object_name / f"{object_name}.urdf"
        if not mesh.is_file() or not urdf.is_file():
            raise FileNotFoundError(f"missing object assets for {object_name}: {mesh}, {urdf}")
        asset_vertices = trimesh.load_mesh(mesh, process=False).vertices
        object_poses, alignment = canonicalize_object_poses(
            object_poses, bundle["object_vertices_local"], bundle["object_scale"], asset_vertices)
        _write_pt(args.output_root / f"{task_name}.pt", human_joints, object_poses)
        scene = scene_root / f"g1_29dof_w_{object_name}.xml"
        _write_scene(args.template, scene, object_name, mesh)
        written.append({
            "task_name": task_name,
            "object_name": object_name,
            "frames": int(len(human_joints)),
            "input_pt": str((args.output_root / f"{task_name}.pt").resolve()),
            "mesh": str(mesh.resolve()),
            "urdf": str(urdf.resolve()),
            "scene": str(scene.resolve()),
            "canonical_alignment": alignment,
            "human_joint_layout": SMPLH_RETARGET_LAYOUT,
        })
    (args.output_root / "manifest.json").write_text(
        json.dumps({"schema": "holosoma.omomo_batch_inputs.v4", "packed_object_order": "xyz_xyzw",
                    "human_joint_layout": SMPLH_RETARGET_LAYOUT,
                    "reader_roundtrip_verified": True, "tasks": written}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"count": len(written), "output_root": str(args.output_root.resolve())}, indent=2))


if __name__ == "__main__":
    main()
