"""Verify that scalar joints participate in retargeting under the installed bindings."""

import mujoco
import numpy as np
import pytest

from holosoma_retargeting.src.interaction_mesh_retargeter import InteractionMeshRetargeter


@pytest.mark.parametrize("joint_name", ["hinge", "slide"])
def test_scalar_joint_point_jacobian_matches_finite_difference(joint_name):
    model = mujoco.MjModel.from_xml_string("""
        <mujoco>
          <worldbody>
            <body name="robot" pos="0 0 1">
              <freejoint/>
              <geom type="sphere" size="0.1"/>
              <body name="arm">
                <joint name="hinge" type="hinge" axis="0 0 1"/>
                <geom type="sphere" size="0.05"/>
                <body name="tip" pos="0.3 0.2 0">
                  <joint name="slide" type="slide" axis="1 0 0"/>
                  <geom type="sphere" size="0.05"/>
                </body>
              </body>
            </body>
            <body name="object" pos="2 0 1">
              <freejoint/>
              <geom type="sphere" size="0.1"/>
            </body>
          </worldbody>
        </mujoco>
    """)
    retargeter = InteractionMeshRetargeter.__new__(InteractionMeshRetargeter)
    retargeter.robot_model = model
    retargeter.robot_data = mujoco.MjData(model)
    retargeter.has_dynamic_object = True
    data = retargeter.robot_data
    data.qpos[model.joint("hinge").qposadr[0]] = 0.4
    data.qpos[model.joint("slide").qposadr[0]] = 0.1
    q = data.qpos.copy()
    body_id = model.body("tip").id
    qadr = int(model.joint(joint_name).qposadr[0])
    jacobian = retargeter._calc_contact_jacobian_from_point(body_id, np.zeros(3))

    epsilon = 1e-6
    positions = []
    for delta in (epsilon, -epsilon):
        data.qpos[:] = q
        data.qpos[qadr] += delta
        mujoco.mj_forward(model, data)
        positions.append(data.xpos[body_id].copy())
    finite_difference = (positions[0] - positions[1]) / (2 * epsilon)
    assert np.linalg.norm(finite_difference) > 0.1
    np.testing.assert_allclose(jacobian[:, qadr], finite_difference, atol=1e-8, rtol=1e-6)
