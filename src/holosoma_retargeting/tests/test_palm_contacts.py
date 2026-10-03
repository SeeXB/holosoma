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


def test_palm_support_clearance_jacobian_matches_finite_difference():
    ret = palm_retargeter()
    q = ret.robot_data.qpos.copy()
    idx = int(ret.robot_model.joint("roll").qposadr[0])
    q[idx] = .2
    rows = ret._linearized_palm_supports(q, 0)
    eps = 1e-6
    plus, minus = q.copy(), q.copy()
    plus[1] += eps
    minus[1] -= eps
    plus_dist = ret._linearized_palm_supports(plus, 0)[0][1]
    minus_dist = ret._linearized_palm_supports(minus, 0)[0][1]
    np.testing.assert_allclose(rows[0][0][1], (plus_dist - minus_dist) / (2 * eps), atol=1e-7)
    assert rows[0][1] < 0


def test_palm_support_retains_exposed_vertices_across_sqp_steps():
    ret = palm_retargeter()
    q = ret.robot_data.qpos.copy()
    idx = int(ret.robot_model.joint("roll").qposadr[0])
    geom = ret.robot_model.geom("patch").id
    q[idx] = .4
    ret._linearized_palm_supports(q, 0)
    first = ret._palm_support_vertices[(0, geom)].copy()
    q[idx] = -.4
    ret._linearized_palm_supports(q, 0)
    second = ret._palm_support_vertices[(0, geom)]
    assert first.issubset(second)
    assert len(second) > len(first)
