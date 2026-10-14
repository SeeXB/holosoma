"""Approach reward must stay small; force amplification is per active part."""
from dataclasses import replace
import hashlib

import pytest
import torch

from holosoma.config_values.wbt.g1.reward import (
    g1_29dof_wbt_reward_w_object,
    g1_29dof_wbt_reward_w_object_contact_force_shaped,
)
from holosoma.managers.reward.terms.reference_contact import (
    ReferenceContactForceShaped, contact_force_shaped_reward,
)
from holosoma.managers.reward.tests.test_reference_contact_force import force_case

pytestmark = pytest.mark.no_sim


def test_no_force_has_small_dense_reward_and_distance_dependence():
    body = torch.zeros(3, 2, 3)
    body[1] = .05 / 3**.5
    body[2] = .1 / 3**.5
    reward, approach, bonus, _ = contact_force_shaped_reward(
        body, torch.zeros_like(body), torch.ones(3, 2), torch.zeros(3, 2), .05, .1, 4,
    )
    assert .1 == pytest.approx(reward[0].item())
    assert reward[0] > reward[1] > reward[2] > 0
    torch.testing.assert_close(reward, approach)
    assert not bonus.any()


def test_force_is_continuous_bounded_and_cannot_be_borrowed():
    body = torch.zeros(4, 2, 3)
    pressure = torch.tensor([[0., 0.], [1., 0.], [4., 0.], [4000., 0.]])
    reward, _, bonus, _ = contact_force_shaped_reward(
        body, body, torch.ones(4, 2), pressure, .05, .1, 4,
    )
    torch.testing.assert_close(reward, torch.tensor([.1, .2125, .55, .55]))
    torch.testing.assert_close(bonus, torch.tensor([0., .1125, .45, .45]))
    # Far-away loaded part cannot amplify a close but unloaded part.
    body[2:, 0, 0] = 1
    reward, *_ = contact_force_shaped_reward(
        body, torch.zeros_like(body), torch.ones(4, 2), pressure, .05, .1, 4,
    )
    assert reward[2].item() == pytest.approx(.05)


def test_inactive_parts_and_nan_force_cannot_gain_bonus():
    points = torch.zeros(2, 3, 3)
    pressure = torch.tensor([[float("nan"), 100., -4.], [100., 100., 100.]])
    activation = torch.tensor([[1., 0., .5], [0., 0., 0.]])
    reward, _, bonus, _ = contact_force_shaped_reward(points, points, activation, pressure, .05, .1, 4)
    torch.testing.assert_close(reward, torch.tensor([.075, 0.]))
    assert not bonus.any()


def test_runtime_uses_object_force_and_keeps_legacy_separate(force_case):
    env, cfg, command, force, _, artifact, _ = force_case
    term = ReferenceContactForceShaped(cfg, env)
    # 2 N on each part gives halfway force amplification, not a hard gate.
    torch.testing.assert_close(term(env), torch.tensor([.55]))
    artifact.unlink()  # No file I/O in the control loop.
    force[:] = 0
    torch.testing.assert_close(term(env), torch.tensor([.1]))
    assert command.metrics["contact/force_bonus_raw"].item() == 0
    force[0, 0, 1] = 4
    torch.testing.assert_close(term(env), torch.tensor([.55]))
    force[0, 0, 1] = -4  # Reversed normal, not compressive pressure.
    torch.testing.assert_close(term(env), torch.tensor([.1]))
    term.activation[:] = 0
    torch.testing.assert_close(term(env), torch.zeros(1))


@pytest.mark.parametrize("key,value", [("approach_fraction", 0), ("approach_fraction", 1),
                                      ("approach_fraction", float("nan")),
                                      ("force_scale_n", 0), ("force_scale_n", -1),
                                      ("force_scale_n", float("nan"))])
def test_invalid_parameters_rejected(force_case, key, value):
    env, cfg, *_ = force_case
    with pytest.raises(ValueError):
        ReferenceContactForceShaped(replace(cfg, params={**cfg.params, key: value}), env)


def test_launcher_preserves_actor_sampler_original_terms_and_explicit_scale(force_case):
    from scripts.run_three_group_experiment import train_command
    from holosoma.config_values.experiment import get_annotated_experiment_config
    from holosoma.utils.config_registry import parse_config

    _, _, _, _, _, artifact, _ = force_case
    preset = g1_29dof_wbt_reward_w_object_contact_force_shaped
    assert all(preset.terms[k] == v for k, v in g1_29dof_wbt_reward_w_object.terms.items())
    cfg = dict(task="test", tag="shaped", seed=42, num_envs=32, iterations=2,
               original_motion="original", b4_motion="b4", semantic_file="semantic",
               object_urdf="urdf", contact_file=str(artifact),
               contact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
               semantic_reward_mode="contact_force_shaped", wandb=False,
               contact_approach_fraction=.15, contact_force_scale_n=3.91)
    command = train_command(cfg, 3, "train", "run")
    parsed = parse_config(get_annotated_experiment_config, args=list(map(str, command[2:])))
    contact = parsed.reward.terms["reference_contact_position"]
    assert contact.func.endswith(":ReferenceContactForceShaped")
    assert contact.params["approach_fraction"] == .15
    assert contact.params["force_scale_n"] == 3.91
    assert contact.weight == 2
    assert parsed.simulator.config.object_contact_body_names == ["left", "right"]
    actor = parsed.observation.groups["actor_obs"]
    assert "base_lin_vel" not in actor.terms and "motion_ref_pos_b" not in actor.terms
    assert parsed.command.setup_terms["motion_command"].params["motion_config"].sampling_mode == "semantic_adaptive"
