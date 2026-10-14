"""Object-filtered force gating must never reward a non-contacting grasp."""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from holosoma.config_values.wbt.g1.reward import (
    g1_29dof_wbt_reward_w_object,
    g1_29dof_wbt_reward_w_object_contact_force,
    g1_29dof_wbt_reward_w_object_contact_position,
)
from holosoma.managers.reward.terms.reference_contact import (
    ReferenceContactForce,
    contact_force_gate,
    contact_position_reward,
)
from holosoma.utils.rotations import quat_apply

pytestmark = pytest.mark.no_sim


def test_all_active_parts_need_current_compressive_force():
    normals = torch.tensor([[[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]]).expand(8, 2, 3)
    forces = normals.clone() * 2.0
    forces[0] = 0  # Perfectly aligned points, but no physical contact.
    forces[1, 1] = 0  # One-sided contact cannot earn a grasp bonus.
    forces[2] = -forces[2]  # Force points in the wrong direction.
    forces[3] = torch.tensor([5.0, 0.0, 0.0])  # Only tangential to anchor normals.
    forces[4, 1] = 0  # An inactive part need not be loaded.
    forces[6] = 0.5 * normals[6]  # Below the 1 N noise floor.
    forces[7, 0, 0] = float("nan")
    activation = torch.ones(8, 2)
    activation[4, 1] = 0
    activation[5] = 0  # Strong forces outside the contact interval give zero.
    gate, normal_force, loaded = contact_force_gate(forces, normals, activation, 1.0)
    assert gate.tolist() == [False, False, False, False, True, False, False, False]
    assert torch.isfinite(normal_force).all()
    assert not loaded[5].any()


def test_force_growth_never_increases_position_reward():
    normals = torch.tensor([[[1.0, 0, 0], [-1.0, 0, 0]]]).expand(2, 2, 3)
    forces = normals * torch.tensor([2.0, 2000.0]).reshape(2, 1, 1)
    activation = torch.full((2, 2), 0.5)
    points = torch.zeros(2, 2, 3)
    position, _ = contact_position_reward(points, points, activation, 0.05)
    gate, _, _ = contact_force_gate(forces, normals, activation, 1.0)
    torch.testing.assert_close(position * gate, torch.full((2,), 0.5))


@pytest.fixture
def force_case(tmp_path):
    motion = tmp_path / "motion.npz"
    motion.write_bytes(b"motion identity")
    semantic = tmp_path / "plan.json"
    semantic.write_text("{}")
    artifact = tmp_path / "contacts.npz"
    metadata = {
        "schema": "holosoma.reference_surface_contacts.v2", "fps": 50.0,
        "motion_sha256": hashlib.sha256(motion.read_bytes()).hexdigest(),
        "semantic_sha256": hashlib.sha256(semantic.read_bytes()).hexdigest(),
        "geometry_sha256": {},
    }
    normals = np.tile([[0, 1, 0], [0, -1, 0]], (2, 1, 1)).astype(np.float32)
    arrays = dict(
        metadata_json=json.dumps(metadata), body_names=["left", "right"],
        body_points_local=np.zeros((2, 2, 3)), object_points_local=np.zeros((2, 2, 3)),
        object_normals_local=normals, active=np.ones((2, 2), dtype=np.bool_),
    )
    np.savez(artifact, **arrays)
    preset = g1_29dof_wbt_reward_w_object_contact_force.terms["reference_contact_position"]
    cfg = replace(preset, params={
        **preset.params, "contact_file": str(artifact),
        "contact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
    })
    object_state = torch.tensor([[0, 0, 0, 0, 0, 0, 1]], dtype=torch.float32)
    force = torch.tensor([[[0, 2.0, 0], [0, -2.0, 0]]])

    def filtered(body_names, object_name):
        assert body_names == ["left", "right"]
        assert object_name == "object"
        return force

    simulator = SimpleNamespace(
        body_names=["right", "left"], _rigid_body_pos=torch.zeros(1, 2, 3),
        _rigid_body_rot=torch.tensor([0, 0, 0, 1]).float().expand(1, 2, 4).clone(),
        get_object_contact_forces=filtered,
        # Huge ground/self forces and old history must NOT open the gate.
        contact_forces=torch.full((1, 2, 3), 999.0),
        contact_forces_history=torch.full((1, 3, 2, 3), 999.0),
    )
    command = SimpleNamespace(
        object_name="object", metrics={}, time_steps=torch.tensor([0]),
        motion_cfg=SimpleNamespace(sampling_mode="semantic_adaptive", motion_file=str(motion),
                                   semantic_file=str(semantic)),
        motion=SimpleNamespace(num_motions=1, time_step_total=2, fps=np.array([50.0])),
        _simulator_object_states=lambda: object_state,
    )
    env = SimpleNamespace(device="cpu", simulator=simulator,
                          command_manager=SimpleNamespace(get_state=lambda _: command))
    return env, cfg, command, force, object_state, artifact, arrays


def test_runtime_uses_object_filtered_force_not_net_force_or_history(force_case):
    env, cfg, command, force, _, artifact, _ = force_case
    term = ReferenceContactForce(cfg, env)
    torch.testing.assert_close(term(env), torch.ones(1))
    artifact.unlink()  # No per-step filesystem access.
    force[0, 1] = 0
    torch.testing.assert_close(term(env), torch.zeros(1))
    assert command.metrics["contact/position_reward_raw"].item() == 1.0
    assert command.metrics["contact/force_reward_raw"].item() == 0.0
    force[:] = 0  # Reset/no physical contact must immediately close the gate.
    command.time_steps[:] = 1
    torch.testing.assert_close(term(env), torch.zeros(1))


def test_force_normal_projection_rotates_with_actual_object(force_case):
    env, cfg, _, force, obj, _, _ = force_case
    rotation = torch.tensor([0.0, 0.0, np.sin(0.7), np.cos(0.7)], dtype=torch.float32)
    obj[:, 3:7] = rotation
    env.simulator._rigid_body_rot[:] = rotation
    force[:] = quat_apply(rotation.expand(1, 2, 4), force, True)
    torch.testing.assert_close(ReferenceContactForce(cfg, env)(env), torch.ones(1))


@pytest.mark.parametrize("failure", ["normals", "sensor", "object"])
def test_force_mode_fails_closed_without_required_inputs(force_case, failure):
    env, cfg, command, _, _, artifact, arrays = force_case
    if failure == "normals":
        arrays.pop("object_normals_local")
        np.savez(artifact, **arrays)
        cfg = replace(cfg, params={**cfg.params, "contact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()})
    elif failure == "sensor":
        del env.simulator.get_object_contact_forces
    else:
        command.object_name = "different_object"
    with pytest.raises(ValueError):
        ReferenceContactForce(cfg, env)(env)


@pytest.mark.parametrize("threshold", [0.0, -1.0, float("nan")])
def test_zero_force_cannot_be_enabled_by_an_invalid_threshold(force_case, threshold):
    env, cfg, *_ = force_case
    cfg = replace(cfg, params={**cfg.params, "minimum_normal_force_n": threshold})
    with pytest.raises(ValueError):
        ReferenceContactForce(cfg, env)


def test_new_preset_and_launcher_preserve_original_reward_and_actor(force_case):
    from scripts.run_three_group_experiment import train_command
    from holosoma.config_values.experiment import get_annotated_experiment_config
    from holosoma.utils.config_registry import parse_config

    _, _, _, _, _, artifact, _ = force_case
    new = g1_29dof_wbt_reward_w_object_contact_force
    old = g1_29dof_wbt_reward_w_object_contact_position
    assert all(new.terms[key] == value for key, value in g1_29dof_wbt_reward_w_object.terms.items())
    assert new.terms["reference_contact_position"].weight == old.terms["reference_contact_position"].weight == 2.0
    cfg = dict(task="test", tag="force", seed=42, num_envs=32, iterations=2,
               original_motion="original", b4_motion="b4", semantic_file="semantic",
               object_urdf="urdf", contact_file=str(artifact),
               contact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
               semantic_reward_mode="contact_force", wandb=False)
    command = train_command(cfg, 3, "train", "run")
    parsed = parse_config(get_annotated_experiment_config, args=list(map(str, command[2:])))
    assert parsed.reward == replace(new, terms={
        **new.terms,
        "reference_contact_position": replace(new.terms["reference_contact_position"], params={
            **new.terms["reference_contact_position"].params,
            "contact_file": str(artifact), "contact_sha256": cfg["contact_sha256"],
        }),
    })
    assert parsed.simulator.config.object_contact_body_names == ["left", "right"]
    actor = parsed.observation.groups["actor_obs"]
    assert "base_lin_vel" not in actor.terms and "motion_ref_pos_b" not in actor.terms
    assert parsed.command.setup_terms["motion_command"].params["motion_config"].sampling_mode == "semantic_adaptive"
