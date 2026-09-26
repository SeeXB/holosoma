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
    g1_29dof_wbt_reward_w_object_contact_position,
)
from holosoma.managers.reward.terms.reference_contact import ReferenceContactPosition, contact_position_reward
from holosoma.utils.rotations import quat_apply

pytestmark = pytest.mark.no_sim


def test_contact_gate_height_error_and_both_hands():
    target = torch.zeros(4, 2, 3)
    hand = target.clone()
    hand[1, :, 2] = 0.05
    hand[2, 0, 2] = 0.10
    active = torch.tensor([[True, True], [True, True], [True, True], [False, False]])
    reward, error = contact_position_reward(hand, target, active, 0.05)
    torch.testing.assert_close(reward, torch.tensor([1., np.exp(-1), np.exp(-2), 0.], dtype=torch.float32))
    assert error[3] == 0
    # An inactive hand must neither dilute the error nor earn a constant bonus.
    active[2, 1] = False
    assert contact_position_reward(hand, target, active, .05)[0][2] == pytest.approx(np.exp(-4))


@pytest.fixture
def contact_case(tmp_path):
    motion = tmp_path / "motion.npz"
    motion.write_bytes(b"test motion identity")
    artifact = tmp_path / "contact.npz"
    metadata = {"schema": "holosoma.reference_surface_contacts.v1", "fps": 50.,
                "motion_sha256": hashlib.sha256(motion.read_bytes()).hexdigest(), "geometry_sha256": {}}
    np.savez(artifact, metadata_json=json.dumps(metadata), body_names=["left", "right"],
             hand_points_local=np.array([[[.1, 0, 0], [.1, 0, 0]]] * 3),
             object_points_local=np.array([[[0, .2, 0], [0, -.2, 0]]] * 3),
             active=np.array([[False, False], [True, True], [True, True]]))
    cfg = g1_29dof_wbt_reward_w_object_contact_position.terms["reference_contact_position"]
    cfg = replace(cfg, params={**cfg.params, "contact_file": str(artifact),
                              "contact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()})
    env = SimpleNamespace(device="cpu", simulator=SimpleNamespace(body_names=["right", "left"]))
    term = ReferenceContactPosition(cfg, env)  # Command manager does not exist yet.
    obj = torch.tensor([[3., 4., 5., 0., 0., 0., 1.], [6., 7., 8., 0., 0., 0., 1.]])
    command = SimpleNamespace(motion_cfg=SimpleNamespace(sampling_mode="semantic_adaptive", motion_file=str(motion)),
                              motion=SimpleNamespace(num_motions=1, time_step_total=3, fps=np.array([50.])),
                              time_steps=torch.tensor([1, 0]), metrics={}, _simulator_object_states=lambda: obj)
    env.command_manager = SimpleNamespace(get_state=lambda _: command)
    env.simulator._rigid_body_pos = obj[:, None, :3] + torch.tensor([[[-.1, -.2, 0], [-.1, .2, 0]]])
    env.simulator._rigid_body_rot = torch.tensor([0., 0., 0., 1.]).expand(2, 2, 4).clone()
    return env, term, command, obj, motion, artifact


def test_runtime_tracks_actual_object_frame_and_respects_body_order(contact_case):
    env, term, command, obj, _, artifact = contact_case
    torch.testing.assert_close(term(env), torch.tensor([1., 0.]))
    # Apply the same world rotation/translation to object and robot (including
    # distinct environment origins). The object-local task must be invariant.
    rotation = torch.tensor([0., 0., np.sin(.6), np.cos(.6)], dtype=torch.float32)
    offset = torch.tensor([8., -2., 3.])
    env.simulator._rigid_body_pos = quat_apply(rotation.expand(2, 2, 4), env.simulator._rigid_body_pos, True) + offset
    env.simulator._rigid_body_rot[:] = rotation
    obj[:, :3] = quat_apply(rotation.expand(2, 4), obj[:, :3], True) + offset
    obj[:, 3:7] = rotation
    artifact.unlink()  # No file I/O once initialized.
    torch.testing.assert_close(term(env), torch.tensor([1., 0.]))
    command.time_steps[:] = 2  # Per-env reference frame and resets, not wall time.
    obj[:, 2] -= .1
    reward = term(env)
    torch.testing.assert_close(reward, torch.full((2,), float(np.exp(-4))), rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("invalid", ["sampling", "motion", "artifact", "frames"])
def test_rejects_incompatible_inputs(contact_case, invalid):
    env, term, command, _, motion, artifact = contact_case
    if invalid == "sampling":
        command.motion_cfg.sampling_mode = "original_adaptive"
    elif invalid == "motion":
        motion.write_bytes(b"changed")
    elif invalid == "artifact":
        artifact.write_bytes(b"changed")
    else:
        command.motion.time_step_total = 4
    with pytest.raises(ValueError):
        term(env)


def test_only_opt_in_reward_and_group3_command_change():
    from scripts.run_three_group_experiment import train_command
    from holosoma.config_values.wbt.g1.paper_dr import (
        g1_29dof_wbt_w_object_b4_s0_original_adaptive_paper_dr as g1,
        g1_29dof_wbt_w_object_b4_s2_semantic_adaptive_paper_dr as old_g3,
    )
    assert g1.reward == old_g3.reward == g1_29dof_wbt_reward_w_object
    new = g1_29dof_wbt_reward_w_object_contact_position
    assert len(new.terms) == len(g1.reward.terms) + 1
    assert all(new.terms[key] == value for key, value in g1.reward.terms.items())
    assert new.semantic_keyframe is None
    cfg = dict(task="test", tag="contact", seed=42, num_envs=32, iterations=2,
               original_motion="original", b4_motion="b4", semantic_file="semantic", object_urdf="urdf")
    contact = {**cfg, "contact_file": "contact", "contact_sha256": "sha"}
    for g in [1, 2]:
        assert train_command(contact, g, "train", "run") == train_command(cfg, g, "train", "run")
    command = train_command(contact, 3, "train", "run")
    assert "reward:g1-29dof-wbt-w-object-contact-position" in command
    assert command[command.index("--reward.terms.reference-contact-position.params.contact-file") + 1] == "contact"
