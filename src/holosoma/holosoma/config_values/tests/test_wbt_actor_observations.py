"""Keep training-only presets compatible with the original WBT actor input."""

from __future__ import annotations

import pytest

from holosoma.config_values.wbt.g1 import paper_dr
from holosoma.config_values.wbt.g1.observation import (
    actor_obs_shared,
    g1_29dof_wbt_observation_w_object,
)

pytestmark = pytest.mark.no_sim

ACTOR_TERM_DIMS = {
    "actions": 29,
    "base_ang_vel": 3,
    "dof_pos": 29,
    "dof_vel": 29,
    "motion_command": 58,
    "motion_ref_ori_b": 6,
}


def test_actor_keeps_original_154_dimensional_channels() -> None:
    actor = paper_dr.paper_observation.groups["actor_obs"]
    assert actor == actor_obs_shared
    assert set(actor.terms) == set(ACTOR_TERM_DIMS)
    assert sum(ACTOR_TERM_DIMS[name] for name in actor.terms) == 154
    assert actor.history_length == 1
    assert actor.concatenate


def test_privileged_observations_remain_critic_only() -> None:
    actor = paper_dr.paper_observation.groups["actor_obs"]
    critic = paper_dr.paper_observation.groups["critic_obs"]
    assert critic == g1_29dof_wbt_observation_w_object.groups["critic_obs"]
    privileged = {
        "base_lin_vel", "motion_ref_pos_b", "robot_body_pos_b",
        "robot_body_ori_b", "obj_pos_b", "obj_ori_b", "obj_lin_vel_b",
    }
    assert privileged.isdisjoint(actor.terms)
    assert privileged <= set(critic.terms)


@pytest.mark.parametrize("name", paper_dr.__all__)
def test_every_paper_dr_preset_preserves_original_actor(name: str) -> None:
    preset = getattr(paper_dr, name)
    assert preset.observation.groups["actor_obs"] == actor_obs_shared
