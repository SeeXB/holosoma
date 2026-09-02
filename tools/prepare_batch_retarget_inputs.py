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

import numpy as np
import torch
from scipy.spatial.transform import Rotation


TASK_OBJECTS = {
    "sub9_clothesstand_058": "clothesstand",
    "sub17_floorlamp_026": "floorlamp",
    "sub10_largebox_089": "largebox",
    "sub16_largebox_007": "largebox",
    "sub1_largetable_028": "largetable",
    "sub16_largetable_013": "largetable",
    "sub9_monitor_025": "monitor",
    "sub15_plasticbox_013": "plasticbox",
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
    # InterMimic loader reads human joints from [162:318] and object poses from
    # [318:325] in [qx,qy,qz,qw,x,y,z] order.
    packed = torch.zeros((len(human_joints), 591), dtype=torch.float32)
    packed[:, 162:318] = torch.from_numpy(human_joints.astype(np.float32).reshape(len(human_joints), -1))
    wxyz_xyz = object_poses.astype(np.float32)
    packed[:, 318:325] = torch.from_numpy(wxyz_xyz[:, [1, 2, 3, 4, 5, 6, 0]])
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(packed, path)


def _write_scene(template: Path, destination: Path, object_name: str, object_mesh: Path) -> None:
    text = template.read_text(encoding="utf-8")
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
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    scene_root = args.output_root / "scenes"
    written = []
    for task_name, object_name in TASK_OBJECTS.items():
        sequence_dir = args.bundle_root / task_name
        if task_name == "sub3_largebox_003":
            sequence_dir = args.bundle_root / "sub03_largebox3"
        bundle_path = sequence_dir / "input" / "omomo_gt_sequence.npz"
        if not bundle_path.is_file():
            raise FileNotFoundError(bundle_path)
        with np.load(bundle_path, allow_pickle=False) as source:
            bundle = {key: source[key] for key in source.files}
        human_joints = np.asarray(bundle["human_joints"], dtype=np.float32)
        object_poses = _object_poses(bundle)
        _write_pt(args.output_root / f"{task_name}.pt", human_joints, object_poses)

        mesh = args.models_root / object_name / f"{object_name}.obj"
        urdf = args.models_root / object_name / f"{object_name}.urdf"
        if not mesh.is_file() or not urdf.is_file():
            raise FileNotFoundError(f"missing object assets for {object_name}: {mesh}, {urdf}")
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
        })
    (args.output_root / "manifest.json").write_text(
        json.dumps({"schema": "holosoma.omomo_batch_inputs.v1", "tasks": written}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"count": len(written), "output_root": str(args.output_root.resolve())}, indent=2))


if __name__ == "__main__":
    main()
