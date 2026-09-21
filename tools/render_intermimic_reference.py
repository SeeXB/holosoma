#!/usr/bin/env python3
"""Render the exact body positions and object pose stored in an InterMimic PT."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from holosoma_retargeting.config_types.data_type import SMPLH_DEMO_JOINTS
from holosoma_retargeting.examples.render_mujoco_trajectory_comparison import RawVideoEncoder
from holosoma_retargeting.src.utils import load_intermimic_data


BONES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12), (12, 13),
    (11, 14), (14, 15), (15, 16), (16, 17),
    (11, 33), (33, 34), (34, 35), (35, 36),
    (17, 18), (18, 19), (19, 20),
    (17, 21), (21, 22), (22, 23),
    (17, 24), (24, 25), (25, 26),
    (17, 27), (27, 28), (28, 29),
    (17, 30), (30, 31), (31, 32),
    (36, 37), (37, 38), (38, 39),
    (36, 40), (40, 41), (41, 42),
    (36, 43), (43, 44), (44, 45),
    (36, 46), (46, 47), (47, 48),
    (36, 49), (49, 50), (50, 51),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    suffix = "-Bold" if bold else ""
    path = Path(f"/usr/share/fonts/truetype/dejavu/DejaVuSans{suffix}.ttf")
    return ImageFont.truetype(str(path), size=size) if path.exists() else ImageFont.load_default()


def _annotate(frame: np.ndarray, frame_index: int, frame_count: int, fps: int) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    width, height = image.size
    draw.rounded_rectangle((18, 16, width - 18, 76), radius=12, fill=(8, 12, 18, 215))
    draw.rectangle((18, 16, 28, 76), fill=(66, 184, 224, 255))
    draw.text((44, 25), "Official InterMimic PT reference", font=_font(25, True), fill=(245, 248, 252, 255))
    draw.text((44, 51), "exact body_pos[162:318] + object pose", font=_font(15), fill=(194, 212, 226, 255))
    status = f"Frame {frame_index:03d}/{frame_count - 1:03d}   {frame_index / fps:5.2f} s"
    status_font = _font(19)
    bbox = draw.textbbox((0, 0), status, font=status_font)
    status_width = bbox[2] - bbox[0]
    draw.rounded_rectangle(
        (width - status_width - 42, height - 52, width - 18, height - 16),
        radius=9,
        fill=(8, 12, 18, 195),
    )
    draw.text((width - status_width - 30, height - 46), status, font=status_font, fill=(235, 240, 247, 255))
    return np.asarray(image)


def _scene_xml(object_mesh: Path) -> str:
    mesh = str(object_mesh.resolve()).replace("&", "&amp;").replace('"', "&quot;")
    return f"""
<mujoco model="intermimic_reference">
  <compiler angle="radian"/>
  <option gravity="0 0 -9.81"/>
  <asset>
    <mesh name="object_mesh" file="{mesh}"/>
    <texture name="floor_tex" type="2d" builtin="checker" width="512" height="512"
             rgb1=".18 .27 .36" rgb2=".08 .12 .18"/>
    <material name="floor_mat" texture="floor_tex" texrepeat="2 2" texuniform="true"
              reflectance=".25"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="20 20 .1" material="floor_mat"/>
    <body name="object" mocap="true">
      <geom name="object_visual" type="mesh" mesh="object_mesh" rgba=".76 .82 .92 .62"
            contype="0" conaffinity="0"/>
    </body>
    <light name="key" pos="0 -3 5" dir="0 .4 -1" directional="true"
           diffuse="1 1 1" ambient=".3 .3 .3" specular=".2 .2 .2"/>
  </worldbody>
  <visual>
    <global offwidth="1920" offheight="1080"/>
    <headlight diffuse=".6 .6 .6" ambient=".15 .15 .15" specular=".5 .5 .5"/>
    <rgba haze=".08 .12 .18 1"/>
  </visual>
</mujoco>
"""


def _append_geom(scene: mujoco.MjvScene, geom_type: mujoco.mjtGeom, size, pos, mat, rgba) -> None:
    if scene.ngeom >= scene.maxgeom:
        raise RuntimeError("MuJoCo visualization scene ran out of geometry slots")
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        type=geom_type,
        size=np.asarray(size, dtype=np.float64),
        pos=np.asarray(pos, dtype=np.float64),
        mat=np.asarray(mat, dtype=np.float64).reshape(9),
        rgba=np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def _bone_matrix(start: np.ndarray, end: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length < 1e-8:
        return 0.5 * (start + end), np.eye(3), length
    direction /= length
    z_axis = np.asarray([0.0, 0.0, 1.0])
    dot = float(np.clip(np.dot(z_axis, direction), -1.0, 1.0))
    if dot > 0.999999:
        matrix = np.eye(3)
    elif dot < -0.999999:
        matrix = np.diag([-1.0, 1.0, -1.0])
    else:
        axis = np.cross(z_axis, direction)
        axis /= np.linalg.norm(axis)
        cross = np.asarray(
            [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
        )
        matrix = np.eye(3) + np.sqrt(1.0 - dot * dot) * cross + (1.0 - dot) * (cross @ cross)
    return 0.5 * (start + end), matrix, length


def _add_skeleton(scene: mujoco.MjvScene, joints: np.ndarray) -> None:
    body_color = (0.98, 0.72, 0.22, 1.0)
    left_color = (0.20, 0.62, 1.0, 1.0)
    right_color = (0.23, 0.92, 0.58, 1.0)
    for first, second in BONES:
        midpoint, matrix, length = _bone_matrix(joints[first], joints[second])
        if length < 1e-8:
            continue
        color = left_color if first in range(17, 33) else right_color if first in range(36, 52) else body_color
        _append_geom(scene, mujoco.mjtGeom.mjGEOM_CAPSULE, (0.009, length / 2.0, 0.0), midpoint, matrix, color)
    for index, point in enumerate(joints):
        color = left_color if 17 <= index <= 32 else right_color if 36 <= index <= 51 else body_color
        radius = 0.021 if index in {0, 13, 17, 36} else 0.014
        _append_geom(scene, mujoco.mjtGeom.mjGEOM_SPHERE, (radius, 0.0, 0.0), point, np.eye(3), color)


def render(args: argparse.Namespace) -> None:
    reference = args.reference_pt.resolve()
    object_mesh = args.object_mesh.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    joints, object_poses = load_intermimic_data(str(reference))
    joints = np.asarray(joints, dtype=np.float64)
    object_poses = np.asarray(object_poses, dtype=np.float64)
    if joints.shape[1:] != (52, 3) or object_poses.shape != (len(joints), 7):
        raise ValueError(f"Unexpected InterMimic shapes: joints={joints.shape}, object={object_poses.shape}")

    if args.bundle_output is not None:
        bundle_output = args.bundle_output.resolve()
        bundle_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            bundle_output,
            object_name=np.asarray(args.object_name),
            fps=np.asarray(args.fps, dtype=np.float64),
            human_joints=joints.astype(np.float32),
            smplh_joint_names=np.asarray(SMPLH_DEMO_JOINTS),
            human_joint_layout=np.asarray("smplh_retarget_v1"),
            frame_ids=np.arange(len(joints), dtype=np.int32),
            object_poses_wxyz_xyz=object_poses.astype(np.float32),
            source_kind=np.asarray("official_intermimic_pt"),
            source_reference_pt=np.asarray(str(reference)),
            source_reference_sha256=np.asarray(_sha256(reference)),
        )

    model = mujoco.MjModel.from_xml_string(_scene_xml(object_mesh))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width, max_geom=500)
    encoder = RawVideoEncoder(output, width=args.width, height=args.height, fps=args.fps)
    previews: list[tuple[int, np.ndarray]] = []
    preview_set = set(args.preview_frames)
    try:
        for frame_index, (frame_joints, object_pose) in enumerate(zip(joints, object_poses, strict=True)):
            data.mocap_pos[0] = object_pose[4:]
            data.mocap_quat[0] = object_pose[:4]
            mujoco.mj_forward(model, data)
            camera = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(camera)
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            camera.azimuth = args.azimuth
            camera.elevation = args.elevation
            camera.distance = args.distance
            camera.lookat[:] = 0.65 * frame_joints[0] + 0.35 * object_pose[4:]
            camera.lookat[2] += 0.12
            renderer.update_scene(data, camera=camera)
            _add_skeleton(renderer.scene, frame_joints)
            image = _annotate(renderer.render().copy(), frame_index, len(joints), args.fps)
            encoder.write(image)
            if frame_index in preview_set:
                previews.append((frame_index, image))
    finally:
        encoder.close()
        renderer.close()

    if previews:
        sheet = Image.new("RGB", (args.width, args.height * len(previews)), (12, 15, 20))
        for row, (_, image) in enumerate(previews):
            sheet.paste(Image.fromarray(image), (0, row * args.height))
        sheet.save(output.with_suffix(".contact_sheet.jpg"), quality=92)

    metadata = {
        "schema": "holosoma.intermimic_reference_render.v1",
        "reference_pt": str(reference),
        "reference_pt_sha256": _sha256(reference),
        "source_fields": {
            "human": "body_pos columns 162:318 reshaped to [T,52,3]",
            "object": "columns 318:325 stored as xyz+xyzw and decoded as wxyz+xyz",
        },
        "rendered_human_representation": "exact stored joint positions rendered as a skeleton; not an inferred surface mesh",
        "frames": len(joints),
        "fps": args.fps,
        "output": str(output),
        "object_mesh": str(object_mesh),
        "object_mesh_sha256": _sha256(object_mesh),
        "semantic_bundle": str(args.bundle_output.resolve()) if args.bundle_output is not None else None,
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-pt", type=Path, required=True)
    parser.add_argument("--object-mesh", type=Path, required=True)
    parser.add_argument("--object-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--bundle-output",
        type=Path,
        help="Also export the exact PT joints/object poses as a semantic-signal bundle.",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--azimuth", type=float, default=142.0)
    parser.add_argument("--elevation", type=float, default=-16.0)
    parser.add_argument("--distance", type=float, default=3.2)
    parser.add_argument("--preview-frames", type=int, nargs="+", default=(0, 42, 96, 102, 103, 104, 140))
    render(parser.parse_args())


if __name__ == "__main__":
    main()
