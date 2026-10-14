from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np

from scripts.adapt_retarget_contacts_for_rl import build


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_transfers_object_anchors_and_only_adapts_sphere_geometry(tmp_path):
    semantic = tmp_path / "semantic.json"
    semantic.write_text('{"events": []}')
    motion = tmp_path / "motion.npz"
    np.savez(motion, joint_pos=np.zeros((5, 1)), fps=np.array([2.0]))

    mesh = tmp_path / "sphere.obj"
    mesh.write_text("v 0 -0.04 0.02\nv 0 0.04 0.02\nv 0 0 0.06\nf 1 2 3\n")
    urdf = tmp_path / "robot.urdf"
    urdf.write_text(
        f"""<robot name="test">
        <link name="left_wrist_yaw_link"/><link name="right_wrist_yaw_link"/>
        <link name="left_sphere_hand_link"><collision><geometry><mesh filename="{mesh.name}"/></geometry></collision></link>
        <link name="right_sphere_hand_link"><collision><geometry><mesh filename="{mesh.name}"/></geometry></collision></link>
        <joint name="left_fixed" type="fixed"><origin xyz="0.03 -0.003 0" rpy="0 1.57079632679 0"/><parent link="left_wrist_yaw_link"/><child link="left_sphere_hand_link"/></joint>
        <joint name="right_fixed" type="fixed"><origin xyz="0.03 -0.003 0" rpy="0 1.57079632679 0"/><parent link="right_wrist_yaw_link"/><child link="right_sphere_hand_link"/></joint>
        </robot>"""
    )
    source = tmp_path / "retarget.npz"
    object_points = np.zeros((3, 2, 3), dtype=np.float32)
    object_points[:, 0, 0] = [0.0, 1.0, 2.0]
    object_points[:, 1, 0] = [10.0, 11.0, 12.0]
    activation = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]], dtype=np.float32)
    source_metadata = {
        "schema": "holosoma.retarget_surface_contacts.v2",
        "frames": 3,
        "fps": 1.0,
        "no_retargeted_robot_trajectory_input": True,
        "inputs": {"semantic_plan_sha256": _sha(semantic)},
    }
    np.savez(
        source,
        metadata_json=json.dumps(source_metadata),
        semantic_parts=np.array(["left_hand", "right_hand"]),
        object_points_local=object_points,
        object_normals_local=np.broadcast_to(np.array([[1, 0, 0], [-1, 0, 0]]), (3, 2, 3)).copy(),
        active=activation > 0,
        activation=activation,
        # Deliberately absurd rubber points: the adapter must ignore these.
        robot_points_local=np.full((2, 3), 999.0),
    )
    output = tmp_path / "rl_contacts.npz"
    metadata = build(
        argparse.Namespace(
            retarget_targets=source,
            motion=motion,
            semantic_file=semantic,
            robot_urdf=urdf,
            output=output,
            support_band_mm=0.001,
        )
    )
    with np.load(output, allow_pickle=False) as result:
        assert result["body_names"].tolist() == [
            "left_wrist_yaw_link",
            "right_wrist_yaw_link",
        ]
        assert result["source_geom_names"].tolist() == [
            "left_sphere_hand_link",
            "right_sphere_hand_link",
        ]
        np.testing.assert_allclose(result["object_points_local"][:, 0, 0], [0, 0.5, 1, 1.5, 2])
        np.testing.assert_allclose(result["activation"][:, 0], [0, 0.25, 0.5, 0.75, 1])
        np.testing.assert_allclose(result["object_normals_local"][:, 0], np.tile([1, 0, 0], (5, 1)))
        np.testing.assert_allclose(result["object_normals_local"][:, 1], np.tile([-1, 0, 0], (5, 1)))
        np.testing.assert_allclose(result["body_points_local"][0, 0], [0.05, -0.043, 0], atol=1e-8)
        np.testing.assert_allclose(result["body_points_local"][0, 1], [0.05, 0.037, 0], atol=1e-8)
        assert np.max(result["body_points_local"]) < 1.0
    assert metadata["source_robot_points_ignored"] is True
    assert metadata["contact_definition"].endswith("no RL trajectory contact search")
