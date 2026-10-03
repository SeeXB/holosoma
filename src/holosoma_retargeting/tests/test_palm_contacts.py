"""Palm contact constrains a point and normal, not finger heading."""
import mujoco
import numpy as np
import pytest
from holosoma_retargeting.config_types.semantic import SemanticRetargetingConfig
from holosoma_retargeting.semantic_keyframes.contact_targets import RetargetContactTargets
from holosoma_retargeting.src.interaction_mesh_retargeter import InteractionMeshRetargeter


def palm_retargeter():
    model = mujoco.MjModel.from_xml_string('''
        <mujoco><asset><mesh name="patch_mesh" vertex="-.04 -.04 -.04 -.04 -.04 .04 -.04 .04 -.04 -.04 .04 .04 .04 -.04 -.04 .04 -.04 .04 .04 .04 -.04 .04 .04 .04"/></asset><worldbody>
          <body name="root" pos="0 0 1"><freejoint/><geom type="sphere" size=".1"/>
            <body name="palm"><joint name="roll" type="hinge" axis="1 0 0"/>
              <geom name="hand" type="sphere" size=".05"/>
              <body name="twist"><joint name="twist" type="hinge" axis="0 1 0"/>
                <geom name="patch" type="mesh" mesh="patch_mesh"/>
              </body>
            </body>
          </body>
          <body name="object" pos="1 0 1"><freejoint/><geom name="object_mesh" type="mesh" mesh="patch_mesh"/></body>
        </worldbody></mujoco>''')
    ret = InteractionMeshRetargeter.__new__(InteractionMeshRetargeter)
    ret.robot_model = model
    ret.robot_data = mujoco.MjData(model)
    ret.has_dynamic_object = True
    ret.q_a_indices = np.arange(model.nq - 7)
    ret.semantic_config = SemanticRetargetingConfig(contact_target_mode="palm_patch")
    ret._contact_geometry_by_part = {"left_hand": (model.geom("patch").id,)}
    ret._contact_object_geometries = (model.geom("object_mesh").id,)
    ret._retarget_contact_targets = RetargetContactTargets(
        parts=("left_hand",), active=np.ones((1, 1), bool),
        object_points_local=np.zeros((1, 1, 3)), weights=np.ones(1),
        source_distance_m=np.zeros((1, 1)), metadata={},
        robot_points_local=np.array([[.2, 0, 0]]),
        robot_normals_local=np.array([[0, 1, 0]]),
        object_normals_local=np.array([[[0, -1, 0]]]),
    )
    return ret


def test_palm_normal_jacobian_matches_finite_difference():
    ret = palm_retargeter()
    q = ret.robot_data.qpos.copy()
    q[ret.robot_model.joint("roll").qposadr[0]] = .4
    jac, normal, target, _, _ = ret._linearized_palm_normals(q, 0)[0]
    eps = 1e-6
    for idx in (0, 4, int(ret.robot_model.joint("roll").qposadr[0])):
        plus, minus = q.copy(), q.copy()
        plus[idx] += eps
        minus[idx] -= eps
        n_plus = ret._linearized_palm_normals(plus, 0)[0][1]
        n_minus = ret._linearized_palm_normals(minus, 0)[0][1]
        np.testing.assert_allclose(jac[:, idx], (n_plus - n_minus) / (2 * eps), atol=1e-7)
    np.testing.assert_allclose(target, [0, 1, 0])


def test_twist_about_palm_normal_has_no_heading_penalty():
    ret = palm_retargeter()
    q = ret.robot_data.qpos.copy()
    idx = int(ret.robot_model.joint("twist").qposadr[0])
    q[idx] = 1.2
    jac, normal, target, _, _ = ret._linearized_palm_normals(q, 0)[0]
    np.testing.assert_allclose(normal, target, atol=1e-12)
    np.testing.assert_allclose(jac[:, idx], 0, atol=1e-12)


def test_palm_contact_point_is_fixed_in_link_coordinates():
    ret = palm_retargeter()
    q = ret.robot_data.qpos.copy()
    _, point, anchor, _, _ = ret._linearized_surface_contacts(q, 0)[0]
    np.testing.assert_allclose(point, [-.8, 0, 0])
    np.testing.assert_allclose(anchor, 0)


def test_inactive_palm_frames_do_not_run_extra_forward(monkeypatch):
    ret = palm_retargeter()
    ret._retarget_contact_targets.active[0, 0] = False
    def unexpected_forward(*args):
        raise AssertionError("inactive palm normal must not invoke mj_forward")
    monkeypatch.setattr(mujoco, "mj_forward", unexpected_forward)
    assert ret._linearized_palm_normals(ret.robot_data.qpos.copy(), 0) == ()


def test_precontact_ramp_uses_onset_target_without_enabling_hard_support():
    ret = palm_retargeter()
    ret.semantic_config = SemanticRetargetingConfig(
        contact_target_mode="palm_patch",
        contact_approach_frames=5,
    )
    active = np.zeros((7, 1), dtype=bool)
    active[5:, 0] = True
    points = np.zeros((7, 1, 3))
    points[5, 0] = [0.1, 0.2, 0.3]
    old = ret._retarget_contact_targets
    targets = RetargetContactTargets(
        parts=old.parts,
        active=active,
        object_points_local=points,
        weights=old.weights,
        source_distance_m=np.zeros((7, 1)),
        metadata=old.metadata,
        robot_points_local=old.robot_points_local,
        robot_normals_local=old.robot_normals_local,
        object_normals_local=np.tile([[[0.0, -1.0, 0.0]]], (7, 1, 1)),
    )
    ret._retarget_contact_targets = targets

    q = ret.robot_data.qpos.copy()
    _, point, anchor, first_weight, _ = ret._linearized_surface_contacts(q, 0)[0]
    _, _, normal_target, normal_strength, _ = ret._linearized_palm_normals(q, 0)[0]

    np.testing.assert_allclose(
        anchor,
        point + (targets.object_points_local[5, 0] - point) / 6.0,
    )
    np.testing.assert_allclose(normal_target, -targets.object_normals_local[5, 0])
    assert first_weight == 1.0
    assert normal_strength == first_weight
    assert ret._linearized_surface_contacts(q, 6) == ()
    assert ret._linearized_palm_normals(q, 6) == ()


def test_palm_nearest_surface_ramps_the_current_palm_gap_not_fixed_anchor():
    ret = palm_retargeter()
    ret.semantic_config = SemanticRetargetingConfig(
        contact_target_mode="palm_nearest_surface",
        contact_approach_frames=5,
    )
    old = ret._retarget_contact_targets
    active = np.zeros((7, 1), dtype=bool)
    active[5:, 0] = True
    ret._retarget_contact_targets = RetargetContactTargets(
        parts=old.parts,
        active=active,
        object_points_local=np.full((7, 1, 3), 0.3),
        weights=old.weights,
        source_distance_m=np.zeros((7, 1)),
        metadata=old.metadata,
        robot_points_local=old.robot_points_local,
        robot_normals_local=old.robot_normals_local,
        object_normals_local=np.tile([[[0.0, -1.0, 0.0]]], (7, 1, 1)),
    )

    _, point, anchor, weight, _ = ret._linearized_surface_contacts(
        ret.robot_data.qpos.copy(), 0
    )[0]

    # The test object's nearest face is 4 cm from its center, whereas the
    # deliberately unrelated artifact anchor is [0.3, 0.3, 0.3].
    np.testing.assert_allclose(anchor, [-0.04, 0.0, 0.0], atol=1e-12)
    assert np.linalg.norm(point - anchor) < np.linalg.norm(point - np.full(3, 0.3))
    assert weight == pytest.approx(1.0 / 6.0)


def test_palm_face_plane_constrains_only_the_anchor_triangle_normal() -> None:
    ret = palm_retargeter()
    ret.semantic_config = SemanticRetargetingConfig(
        contact_target_mode="palm_face_plane",
        contact_temporal_schedule="smooth_window",
        contact_approach_frames=5,
    )
    old = ret._retarget_contact_targets
    active = np.zeros((7, 1), dtype=bool)
    active[5:, 0] = True
    points = np.zeros((7, 1, 3))
    points[5, 0] = [-0.04, 0.0, 0.0]
    ret._retarget_contact_targets = RetargetContactTargets(
        parts=old.parts,
        active=active,
        object_points_local=points,
        weights=old.weights,
        source_distance_m=np.zeros((7, 1)),
        metadata=old.metadata,
        robot_points_local=old.robot_points_local,
        robot_normals_local=old.robot_normals_local,
        object_normals_local=old.object_normals_local.repeat(7, axis=0),
    )

    q = ret.robot_data.qpos.copy()
    jacobian, point, target, weight, _ = ret._linearized_surface_contacts(q, 0)[0]
    assert jacobian.shape == (1, len(ret.q_a_indices))
    np.testing.assert_allclose(point, target, atol=1e-12)
    assert abs(jacobian[0, 0]) == pytest.approx(1.0)
    assert jacobian[0, 1] == pytest.approx(0.0, abs=1e-12)
    assert weight == 1.0

    _, onset_point, onset_target, _, _ = ret._linearized_surface_contacts(q, 5)[0]
    assert abs(onset_point[0]) == pytest.approx(0.76)
    np.testing.assert_allclose(onset_target, 0.0)


def test_palm_artifact_rejects_incomplete_or_nonunit_normals():
    ret = palm_retargeter()
    targets = ret._retarget_contact_targets
    targets.validate(1)
    bad = {**targets.__dict__, "object_normals_local": np.zeros((1, 1, 3))}
    with pytest.raises(ValueError, match="unit normals"):
        RetargetContactTargets(**bad).validate(1)
    bad["object_normals_local"] = None
    with pytest.raises(ValueError, match="both normal"):
        RetargetContactTargets(**bad).validate(1)
