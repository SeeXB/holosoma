#!/usr/bin/env python3
"""Extract the real OMOMO GT sequence into a compact Blender-ready archive.

The motion invocation follows OMOMO's official ``run_smplx_model`` path: SMPL-H
root translation, root axis-angle, 21 body joints and zero hand pose are passed
to ``human_body_prior.body_model.BodyModel``.  The locally available licensed
SMPL-H resource contains ten shape directions, so only the first ten of the
sixteen stored OMOMO betas can be evaluated; no pose or trajectory is changed.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import scipy.sparse
from scipy.spatial.transform import Rotation
import torch
import trimesh

from common import canonical_sequence_name, normalized_dimensions, write_json
from smplh_joint_order import (
    SMPLH_NATIVE_TO_RETARGET,
    SMPLH_RETARGET_JOINT_NAMES,
    SMPLH_RETARGET_LAYOUT,
    reorder_smplh_native_to_retarget,
    validate_retarget_smplh_geometry,
)


def parse_args() -> argparse.Namespace:
    workspace = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", default="sub03_largebox3")
    parser.add_argument(
        "--omomo-data",
        type=Path,
        default=Path("/mnt/sdadrive/shixiongbo/omomo/data"),
    )
    parser.add_argument("--motion-video", type=Path, default=None)
    parser.add_argument("--object-mesh", type=Path, default=None)
    parser.add_argument(
        "--record-file", type=Path, action="append", default=None,
        help="OMOMO sequence pickle; may be repeated (defaults to train and test)",
    )
    parser.add_argument(
        "--smplh-model",
        type=Path,
        default=workspace / "third_party/CARI4D/data/smpl/smplh/SMPLH_male.pkl",
    )
    parser.add_argument(
        "--human-body-prior-root",
        type=Path,
        default=workspace / "third_party/human_body_prior",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def to_numpy(value: object) -> np.ndarray:
    if scipy.sparse.issparse(value):
        return value.toarray()
    if hasattr(value, "r"):
        value = value.r
    return np.asarray(value)


def convert_smplh_pkl(source: Path, destination: Path) -> int:
    """Convert the licensed SMPL-H pickle to BodyModel's documented NPZ form."""
    with source.open("rb") as handle:
        model = pickle.load(handle, encoding="latin1")
    required = (
        "v_template",
        "f",
        "shapedirs",
        "posedirs",
        "J_regressor",
        "kintree_table",
        "weights",
    )
    arrays = {key: to_numpy(model[key]) for key in required}
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)
    return int(arrays["shapedirs"].shape[-1])


def find_sequence(records: dict, sequence_name: str) -> tuple[int, dict]:
    matches = [(int(key), value) for key, value in records.items() if value.get("seq_name") == sequence_name]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one record for {sequence_name!r}, found {len(matches)}"
        )
    return matches[0]


def sequence_object_name(sequence_name: str) -> str:
    match = re.fullmatch(r"sub\d+_([a-z0-9]+)_\d{3}", sequence_name)
    if not match:
        raise ValueError(f"Cannot infer captured object from {sequence_name!r}")
    return match.group(1)


def load_sequence_record(
    omomo_data: Path, sequence_name: str, record_files: list[Path] | None
) -> tuple[int, dict, Path]:
    paths = record_files or [
        omomo_data / "train_diffusion_manip_seq_joints24.p",
        omomo_data / "test_diffusion_manip_seq_joints24.p",
    ]
    matches: list[tuple[int, dict, Path]] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        records = joblib.load(path)
        found = [
            (int(key), value)
            for key, value in records.items()
            if value.get("seq_name") == sequence_name
        ]
        matches.extend((key, value, path) for key, value in found)
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one record for {sequence_name!r}, found {len(matches)}"
        )
    return matches[0]


def main() -> None:
    args = parse_args()
    sequence_name = canonical_sequence_name(args.sequence)
    object_name = sequence_object_name(sequence_name)
    workspace = Path(__file__).resolve().parents[2]
    if args.output is None:
        output = (
            workspace / "exp/omomo_cari4d/sub03_largebox3/input/omomo_gt_sequence.npz"
            if sequence_name == "sub3_largebox_003"
            else workspace / f"exp/omomo_cari4d/{sequence_name}/input/omomo_gt_sequence.npz"
        )
    else:
        output = args.output
    subject = int(sequence_name.split("_", 1)[0][3:])
    motion_video = args.motion_video or (
        args.omomo_data.parent / "motion_videos" / f"sub{subject}" / f"{sequence_name}.mp4"
    )
    object_path = args.object_mesh or (
        args.omomo_data / f"captured_objects/{object_name}_cleaned_simplified.obj"
    )
    record_key, record, records_path = load_sequence_record(
        args.omomo_data, sequence_name, args.record_file
    )
    for required_path in (
        records_path,
        object_path,
        motion_video,
        args.smplh_model,
        args.human_body_prior_root,
    ):
        if not required_path.exists():
            raise FileNotFoundError(required_path)

    print(f"Loading {records_path}", flush=True)
    frame_count = int(np.asarray(record["trans"]).shape[0])

    converted_model = output.with_name("SMPLH_male_bodymodel_10betas.npz")
    shape_direction_count = convert_smplh_pkl(args.smplh_model, converted_model)
    if shape_direction_count != 10:
        raise RuntimeError(
            f"Expected local SMPL-H model to have 10 shape directions, got "
            f"{shape_direction_count}"
        )

    sys.path.insert(0, str(args.human_body_prior_root))
    from human_body_prior.body_model.body_model import BodyModel

    body_model = BodyModel(
        bm_fname=str(converted_model), num_betas=shape_direction_count
    ).eval()
    root_orient = torch.as_tensor(record["root_orient"], dtype=torch.float32)
    pose_body = torch.as_tensor(record["pose_body"], dtype=torch.float32)
    pose_hand = torch.zeros((frame_count, 90), dtype=torch.float32)
    betas_all = np.asarray(record["betas"], dtype=np.float32).reshape(-1)
    betas = torch.as_tensor(
        np.repeat(betas_all[None, :shape_direction_count], frame_count, axis=0),
        dtype=torch.float32,
    )
    translations = torch.as_tensor(record["trans"], dtype=torch.float32)
    with torch.no_grad():
        body = body_model(
            root_orient=root_orient,
            pose_body=pose_body,
            pose_hand=pose_hand,
            betas=betas,
            trans=translations,
        )
    human_vertices = body.v.detach().cpu().numpy().astype(np.float32)
    human_joints_native = body.Jtr.detach().cpu().numpy().astype(np.float32)
    human_joints = reorder_smplh_native_to_retarget(human_joints_native)
    validate_retarget_smplh_geometry(human_joints)
    human_faces = body_model.f.detach().cpu().numpy().astype(np.int32)

    object_mesh = trimesh.load_mesh(object_path, process=False)
    object_vertices_local = np.asarray(object_mesh.vertices, dtype=np.float32)
    object_faces = np.asarray(object_mesh.faces, dtype=np.int32)
    object_scale = np.asarray(record["obj_scale"], dtype=np.float32)
    object_rotation = np.asarray(record["obj_rot"], dtype=np.float32)
    object_translation = np.asarray(record["obj_trans"], dtype=np.float32).reshape(
        frame_count, 3
    )
    # This is exactly HandFootManipDataset.apply_transformation_to_obj_geometry.
    object_vertices = (
        object_scale[:, None, None]
        * np.einsum("tij,vj->tvi", object_rotation, object_vertices_local)
        + object_translation[:, None, :]
    ).astype(np.float32)

    # The semantic-keyframe resolver consumes an explicit metric trajectory.
    # Store the OMOMO rotation as the resolver's wxyz quaternion convention.
    object_quaternion_xyzw = Rotation.from_matrix(object_rotation).as_quat()
    object_quaternion_wxyz = object_quaternion_xyzw[:, [3, 0, 1, 2]].astype(np.float32)
    smplh_joint_names = np.asarray(SMPLH_RETARGET_JOINT_NAMES)

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        sequence_name=np.asarray(sequence_name),
        object_name=np.asarray(object_name),
        fps=np.asarray(30, dtype=np.int32),
        human_vertices=human_vertices,
        human_faces=human_faces,
        human_joints=human_joints,
        smplh_joint_names=smplh_joint_names,
        human_joint_layout=np.asarray(SMPLH_RETARGET_LAYOUT),
        human_joint_native_to_retarget=SMPLH_NATIVE_TO_RETARGET,
        frame_ids=np.arange(frame_count, dtype=np.int32),
        object_poses_wxyz_xyz=np.concatenate(
            [object_quaternion_wxyz, object_translation], axis=1
        ),
        object_vertices=object_vertices,
        object_faces=object_faces,
        object_vertices_local=object_vertices_local,
        object_rotation=object_rotation,
        object_translation=object_translation,
        object_scale=object_scale,
        betas=betas_all,
        root_orient=np.asarray(record["root_orient"], dtype=np.float32),
        pose_body=np.asarray(record["pose_body"], dtype=np.float32),
        human_translation=np.asarray(record["trans"], dtype=np.float32),
    )

    local_dimensions = np.ptp(object_vertices_local, axis=0)
    metric_dimensions_per_frame = object_scale[:, None] * local_dimensions[None]
    all_vertices = np.concatenate(
        [human_vertices.reshape(-1, 3), object_vertices.reshape(-1, 3)], axis=0
    )
    metadata = {
        "requested_sequence": args.sequence,
        "sequence_identifier": sequence_name,
        "record_key": record_key,
        "source_records": records_path,
        "source_video": motion_video,
        "source_object_mesh": object_path,
        "source_smplh_model": args.smplh_model,
        "human_body_prior_commit": _git_head(args.human_body_prior_root),
        "human_model_type": "SMPL-H (6890 vertices, 52 joints)",
        "human_joint_layout": SMPLH_RETARGET_LAYOUT,
        "human_joint_generation": "BodyModel.Jtr followed by explicit native-to-retarget permutation",
        "human_joint_native_to_retarget": SMPLH_NATIVE_TO_RETARGET,
        "frame_count": frame_count,
        "fps": 30,
        "coordinate_convention": "right-handed, Z-up; values in metres",
        "motion_modified": False,
        "object_transform_formula": "V_world[t] = scale[t] * R[t] @ V_local + t[t]",
        "object_local_aabb_dimensions": local_dimensions,
        "object_local_aabb_ratio": normalized_dimensions(local_dimensions),
        "object_metric_aabb_dimensions_median": np.median(
            metric_dimensions_per_frame, axis=0
        ),
        "object_scale_min_median_max": [
            float(object_scale.min()),
            float(np.median(object_scale)),
            float(object_scale.max()),
        ],
        "sequence_world_bounds": np.stack(
            [all_vertices.min(axis=0), all_vertices.max(axis=0)]
        ),
        "smplh_shape_directions_available": shape_direction_count,
        "omomo_betas_stored": int(betas_all.shape[0]),
        "shape_resource_note": (
            "The local licensed SMPL-H model has 10 shape directions. The first "
            "10 OMOMO betas are evaluated; the remaining 6 cannot affect this "
            "model. Poses, translations, object geometry and object transforms "
            "are exact OMOMO GT."
        ),
        "archive": output,
        "object_name": object_name,
    }
    write_json(output.with_suffix(".metadata.json"), metadata)
    print(json.dumps(metadata, indent=2, default=str), flush=True)


def _git_head(repository: Path) -> str | None:
    head_path = repository / ".git/HEAD"
    if not head_path.exists():
        return None
    head = head_path.read_text(encoding="utf-8").strip()
    if head.startswith("ref: "):
        ref = repository / ".git" / head[5:]
        if ref.exists():
            return ref.read_text(encoding="utf-8").strip()
    return head


if __name__ == "__main__":
    main()
