"""A3 action scales use each joint’s own effort limit and stiffness."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from holosoma.config_types.action import ActionTermCfg
from holosoma.config_values.a3_robot import a3_31dof
from holosoma.managers.action.terms.joint_control import JointPositionActionTerm

pytestmark = pytest.mark.no_sim


def _scales(robot):
    env = SimpleNamespace(
        num_dof=len(robot.dof_names),
        num_envs=1,
        device="cpu",
        dof_names=robot.dof_names,
        robot_config=robot,
    )
    return JointPositionActionTerm(ActionTermCfg(func="test"), env).action_scales.tolist()


def test_a3_every_joint_uses_its_own_effort_and_kp():
    control = a3_31dof.control
    assert control.action_scale == 0.25
    assert control.action_scales_by_effort_limit_over_p_gain
    assert control.action_scale_overrides == {}

    scales = _scales(a3_31dof)
    env = SimpleNamespace(
        num_dof=len(a3_31dof.dof_names),
        num_envs=1,
        device="cpu",
        dof_names=a3_31dof.dof_names,
        robot_config=a3_31dof,
    )
    gains = JointPositionActionTerm(ActionTermCfg(func="test"), env).p_gains.tolist()
    for name, scale, effort, kp in zip(
        a3_31dof.dof_names, scales, a3_31dof.dof_effort_limit_list, gains, strict=True
    ):
        assert scale == pytest.approx(0.25 * effort / kp, abs=1e-6), name

    by_name = dict(zip(a3_31dof.dof_names, scales, strict=True))
    assert by_name["left_ankle_roll_joint"] == pytest.approx(0.045625, abs=1e-6)
    assert by_name["right_ankle_roll_joint"] == pytest.approx(0.045625, abs=1e-6)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"left_elbow_typo": 0.4}, "unknown joints"),
        ({"left_elbow_joint": 0.0}, "finite and positive"),
        ({"left_elbow_joint": float("nan")}, "finite and positive"),
    ],
)
def test_invalid_per_joint_override_is_rejected(overrides, message):
    control = replace(a3_31dof.control, action_scale_overrides=overrides)
    robot = replace(a3_31dof, control=control)
    with pytest.raises(ValueError, match=message):
        _scales(robot)
