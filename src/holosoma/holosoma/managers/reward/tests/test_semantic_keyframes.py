from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from holosoma.config_types.reward import (
    RewardManagerCfg,
    RewardTermCfg,
    SemanticKeyframeRewardCfg,
)
from holosoma.config_values.wbt.g1.reward import (
    g1_29dof_wbt_reward,
    g1_29dof_wbt_reward_w_object,
    g1_29dof_wbt_w_object_semantic_e1_part_reward,
    g1_29dof_wbt_w_object_semantic_e2_part_rel_reward,
    g1_29dof_wbt_w_object_semantic_keyframe_reward,
)
from holosoma.managers.reward.manager import (
    RewardManager,
    allocate_fixed_positive_budget,
    compute_base_positive_reward_budget,
)
from holosoma.managers.reward.semantic_keyframes import (
    SemanticKeyframeRuntime,
    combine_valid_objectives,
    infer_omni_tracking_sigmas,
)

pytestmark = pytest.mark.no_sim

TRACKED_BODIES = [
    "pelvis",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "torso_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
]


def _unit_positive_term(env):
    return env.positive_raw


def _unit_penalty_term(env):
    return env.penalty_raw


class _CommandManager:
    def __init__(self, command):
        self.command = command

    def get_state(self, name):
        assert name == "motion_command"
        return self.command


def _identity_quaternions(num_envs: int, num_bodies: int) -> torch.Tensor:
    quaternions = torch.zeros((num_envs, num_bodies, 4), dtype=torch.float32)
    quaternions[..., 3] = 1.0
    return quaternions


def _fake_env(*, motion_fps: float = 50.0, has_object: bool = True):
    num_envs = 1
    num_bodies = len(TRACKED_BODIES)
    body_pos = torch.tensor(
        [
            [
                [0.0, 0.0, 0.8],
                [0.4, 0.25, 0.8],
                [0.4, -0.25, 0.8],
                [0.0, 0.0, 1.1],
                [0.0, 0.1, 0.05],
                [0.0, -0.1, 0.05],
            ]
        ],
        dtype=torch.float32,
    )
    body_quat = _identity_quaternions(num_envs, num_bodies)
    zero_vel = torch.zeros((num_envs, num_bodies, 3), dtype=torch.float32)
    object_pos = torch.tensor([[0.55, 0.0, 0.55]], dtype=torch.float32)
    object_quat = _identity_quaternions(num_envs, 1)[:, 0]
    motion = SimpleNamespace(
        fps=np.asarray(motion_fps),
        has_object=has_object,
        motion_start_idx=torch.tensor([0], dtype=torch.long),
    )
    command = SimpleNamespace(
        motion=motion,
        motion_cfg=SimpleNamespace(body_names_to_track=list(TRACKED_BODIES)),
        time_steps=torch.tensor([10], dtype=torch.long),
        motion_ids=torch.tensor([0], dtype=torch.long),
        body_pos_relative_w=body_pos.clone(),
        body_quat_relative_w=body_quat.clone(),
        body_pos_w=body_pos.clone(),
        body_quat_w=body_quat.clone(),
        body_lin_vel_w=zero_vel.clone(),
        body_ang_vel_w=zero_vel.clone(),
        robot_body_pos_w=body_pos.clone(),
        robot_body_quat_w=body_quat.clone(),
        robot_body_lin_vel_w=zero_vel.clone(),
        robot_body_ang_vel_w=zero_vel.clone(),
        object_pos_w=object_pos.clone(),
        object_quat_w=object_quat.clone(),
        simulator_object_pos_w=object_pos.clone(),
        simulator_object_quat_w=object_quat.clone(),
    )
    reward_cfg = g1_29dof_wbt_reward_w_object if has_object else g1_29dof_wbt_reward
    env = SimpleNamespace(
        command_manager=_CommandManager(command),
        reward_manager=SimpleNamespace(cfg=reward_cfg),
    )
    return env, command


def _payload(*, first_name: str = "phase_alpha", second_name: str = "phase_beta", fps=20):
    payload = {
        "events": [
            {
                "event": first_name,
                "body_parts": ["left_hand"],
                "windows": [{"start_frame": 0, "trigger_frame": 4, "end_frame": 20}],
            },
            {
                "event": second_name,
                "body_parts": ["right_hand"],
                "windows": [{"start_frame": 10, "trigger_frame": 10, "end_frame": 30}],
            },
        ]
    }
    if fps is not None:
        payload["fps"] = fps
    return payload


def _write_payload(path: Path, payload) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _config(path: Path, **overrides) -> SemanticKeyframeRewardCfg:
    values = {
        "enabled": True,
        "semantic_file": str(path),
        "semantic_fps": 20.0,
        "sigma_time": 0.2,
    }
    values.update(overrides)
    return SemanticKeyframeRewardCfg(**values)


def test_disabled_runtime_returns_three_zero_rewards_without_reading_json():
    env, _ = _fake_env()
    output = SemanticKeyframeRuntime(SemanticKeyframeRewardCfg(enabled=False), env).evaluate()
    assert output.part_reward.item() == 0.0
    assert output.rel_reward.item() == 0.0
    assert output.dyn_reward.item() == 0.0
    assert output.active_gate.item() == 0.0


def test_baseline_is_unchanged_and_semantic_presets_reuse_exact_term_set():
    baseline_terms = g1_29dof_wbt_reward_w_object.terms
    assert g1_29dof_wbt_reward_w_object.semantic_keyframe is None

    variants = (
        (
            g1_29dof_wbt_w_object_semantic_e1_part_reward,
            (True, False, False),
        ),
        (
            g1_29dof_wbt_w_object_semantic_e2_part_rel_reward,
            (True, True, False),
        ),
        (g1_29dof_wbt_w_object_semantic_keyframe_reward, (True, True, True)),
    )
    for variant, enabled in variants:
        semantic_cfg = variant.semantic_keyframe
        assert semantic_cfg is not None
        assert variant.terms == baseline_terms
        assert (semantic_cfg.enable_part, semantic_cfg.enable_rel, semantic_cfg.enable_dyn) == enabled


def test_no_external_entity_relative_reward_is_zero(tmp_path):
    path = _write_payload(tmp_path / "semantic.json", _payload())
    env, _ = _fake_env(has_object=False)
    runtime = SemanticKeyframeRuntime(_config(path), env)
    output = runtime.evaluate()
    assert output.rel_reward.item() == 0.0
    assert not output.rel_valid.item()
    assert output.valid_objective_count.item() == 2.0


def test_configurable_body_mapping_supports_one_to_many(tmp_path):
    payload = _payload()
    payload["events"] = [
        {
            "event": "arbitrary_label",
            "body_parts": ["upper_effectors"],
            "windows": [{"start_frame": 0, "trigger_frame": 4, "end_frame": 8}],
        }
    ]
    path = _write_payload(tmp_path / "mapping.json", payload)
    env, _ = _fake_env()
    config = _config(
        path,
        body_part_mapping={
            "upper_effectors": ["left_wrist_yaw_link", "right_wrist_yaw_link"],
        },
    )
    runtime = SemanticKeyframeRuntime(config, env)
    assert runtime.events[0].body_names == ("left_wrist_yaw_link", "right_wrist_yaw_link")
    assert runtime.events[0].body_indices == (1, 2)


def test_fps_mapping_uses_seconds_not_equal_frame_indices(tmp_path):
    path = _write_payload(tmp_path / "fps.json", _payload())
    env, _ = _fake_env(motion_fps=50.0)
    runtime = SemanticKeyframeRuntime(_config(path), env)
    # semantic trigger frame 4 at 20 Hz is t=0.2 s, hence RL frame 10 at 50 Hz.
    assert runtime.events[0].trigger_time_s == pytest.approx(0.2)
    assert runtime.trigger_rl_frame(0) == 10


def test_missing_semantic_fps_requires_explicit_config(tmp_path):
    path = _write_payload(tmp_path / "missing_fps.json", _payload(fps=None))
    env, _ = _fake_env()
    with pytest.raises(ValueError, match="semantic_fps must be explicitly configured"):
        SemanticKeyframeRuntime(
            SemanticKeyframeRewardCfg(enabled=True, semantic_file=str(path), semantic_fps=None),
            env,
        )


def test_transition_truncation_zeros_previous_gate_at_next_trigger(tmp_path):
    path = _write_payload(tmp_path / "truncate.json", _payload())
    env, _ = _fake_env()
    runtime = SemanticKeyframeRuntime(_config(path), env)
    times = torch.tensor([0.49, 0.50, 0.51], dtype=torch.float32)
    gates = runtime.gates_at_times(times)
    assert gates[0, 0] > 0.0
    assert gates[1, 0] == 0.0
    assert gates[2, 0] == 0.0


def test_reference_equal_simulated_state_gives_maximum_active_rewards(tmp_path):
    path = _write_payload(tmp_path / "perfect.json", _payload())
    env, _ = _fake_env()
    runtime = SemanticKeyframeRuntime(_config(path), env)
    output = runtime.evaluate()
    assert output.active_gate.item() == pytest.approx(1.0)
    assert output.part_reward.item() == pytest.approx(1.0)
    assert output.rel_reward.item() == pytest.approx(1.0)
    assert output.dyn_reward.item() == pytest.approx(1.0)


def test_perturbing_only_semantic_body_lowers_part_reward(tmp_path):
    path = _write_payload(tmp_path / "part.json", _payload())
    env, command = _fake_env()
    perfect = SemanticKeyframeRuntime(_config(path), env).evaluate().part_reward.item()
    command.robot_body_pos_w[:, 1, 0] += 0.15
    perturbed = SemanticKeyframeRuntime(_config(path), env).evaluate().part_reward.item()
    assert perturbed < perfect


def test_perturbing_only_relative_geometry_lowers_rel_reward(tmp_path):
    path = _write_payload(tmp_path / "rel.json", _payload())
    env, command = _fake_env()
    perfect = SemanticKeyframeRuntime(_config(path), env).evaluate()
    command.simulator_object_pos_w[:, 0] += 0.15
    perturbed = SemanticKeyframeRuntime(_config(path), env).evaluate()
    assert perturbed.part_reward.item() == pytest.approx(perfect.part_reward.item())
    assert perturbed.rel_reward.item() < perfect.rel_reward.item()


def test_perturbing_only_velocity_lowers_dynamics_reward(tmp_path):
    path = _write_payload(tmp_path / "dyn.json", _payload())
    env, command = _fake_env()
    perfect = SemanticKeyframeRuntime(_config(path), env).evaluate()
    command.robot_body_lin_vel_w[:, 1, 0] += 1.0
    perturbed = SemanticKeyframeRuntime(_config(path), env).evaluate()
    assert perturbed.part_reward.item() == pytest.approx(perfect.part_reward.item())
    assert perturbed.dyn_reward.item() < perfect.dyn_reward.item()


def test_event_names_are_diagnostic_only_and_cannot_change_rewards(tmp_path):
    first_payload = _payload(first_name="name_a", second_name="name_b")
    renamed_payload = deepcopy(first_payload)
    renamed_payload["events"][0]["event"] = "completely_different_1"
    renamed_payload["events"][1]["event"] = "completely_different_2"
    first_path = _write_payload(tmp_path / "first.json", first_payload)
    renamed_path = _write_payload(tmp_path / "renamed.json", renamed_payload)
    first_env, _ = _fake_env()
    renamed_env, _ = _fake_env()
    first = SemanticKeyframeRuntime(_config(first_path), first_env).evaluate()
    renamed = SemanticKeyframeRuntime(_config(renamed_path), renamed_env).evaluate()
    assert torch.equal(first.part_reward, renamed.part_reward)
    assert torch.equal(first.rel_reward, renamed.rel_reward)
    assert torch.equal(first.dyn_reward, renamed.dyn_reward)
    assert torch.equal(first.active_gate, renamed.active_gate)
    first_final = allocate_fixed_positive_budget(
        torch.tensor([6.0]),
        torch.tensor([-0.2]),
        first.active_gate,
        first.combined_reward,
        7.0,
    ).total
    renamed_final = allocate_fixed_positive_budget(
        torch.tensor([6.0]),
        torch.tensor([-0.2]),
        renamed.active_gate,
        renamed.combined_reward,
        7.0,
    ).total
    assert torch.equal(first_final, renamed_final)


def test_semantic_sigmas_are_inherited_from_omni_baseline():
    sigmas = infer_omni_tracking_sigmas(g1_29dof_wbt_reward)
    assert sigmas.position == pytest.approx(0.3)
    assert sigmas.orientation == pytest.approx(0.4)
    assert sigmas.linear_velocity == pytest.approx(1.0)
    assert sigmas.angular_velocity == pytest.approx(3.14)
    fields = SemanticKeyframeRewardCfg.__dataclass_fields__
    assert not any(name.startswith(("lambda_", "beta_", "sigma_part", "sigma_rel", "sigma_dyn")) for name in fields)


def test_positive_budgets_are_automatically_inherited():
    assert compute_base_positive_reward_budget(g1_29dof_wbt_reward) == pytest.approx(5.0)
    assert compute_base_positive_reward_budget(g1_29dof_wbt_reward_w_object) == pytest.approx(7.0)
    assert compute_base_positive_reward_budget(g1_29dof_wbt_w_object_semantic_keyframe_reward) == pytest.approx(
        7.0
    )


def test_zero_activity_exactly_recovers_baseline_and_penalty():
    base_positive = torch.tensor([0.0, 2.5, 5.0])
    penalty = torch.tensor([-0.1, -2.0, 0.0])
    allocation = allocate_fixed_positive_budget(
        base_positive,
        penalty,
        torch.zeros(3),
        torch.tensor([0.2, 0.7, 1.0]),
        5.0,
    )
    assert torch.equal(allocation.positive_total, base_positive)
    assert torch.equal(allocation.penalty, penalty)
    assert torch.equal(allocation.total, base_positive + penalty)
    assert torch.equal(allocation.alpha, torch.zeros(3))


def test_perfect_base_and_semantic_quality_preserve_max_budget_for_any_activity():
    activity = torch.linspace(0.0, 1.0, 101)
    allocation = allocate_fixed_positive_budget(
        torch.full_like(activity, 7.0),
        torch.zeros_like(activity),
        activity,
        torch.ones_like(activity),
        7.0,
    )
    assert torch.allclose(allocation.positive_total, torch.full_like(activity, 7.0), atol=1e-6)
    assert torch.all(allocation.alpha <= 1.0 / 8.0)


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_equal_valid_objective_aggregation_preserves_common_extremes(value):
    component = torch.tensor([value])
    combined, count = combine_valid_objectives(
        (component, component, component),
        (torch.tensor([True]), torch.tensor([True]), torch.tensor([True])),
    )
    assert combined.item() == pytest.approx(value)
    assert count.item() == 3.0


def test_invalid_relation_is_excluded_instead_of_counted_as_zero():
    combined, count = combine_valid_objectives(
        (torch.tensor([0.8]), torch.tensor([0.0]), torch.tensor([0.6])),
        (torch.tensor([True]), torch.tensor([False]), torch.tensor([True])),
    )
    assert combined.item() == pytest.approx(0.7)
    assert count.item() == 2.0


def test_empty_semantic_event_list_recovers_baseline_for_entire_episode(tmp_path):
    path = _write_payload(tmp_path / "empty.json", {"fps": 20, "events": []})
    env, _ = _fake_env()
    output = SemanticKeyframeRuntime(_config(path), env).evaluate()
    assert output.active_gate.item() == 0.0
    assert output.valid_objective_count.item() == 0.0
    allocation = allocate_fixed_positive_budget(
        torch.tensor([4.2]),
        torch.tensor([-0.3]),
        output.active_gate,
        output.combined_reward,
        7.0,
    )
    assert allocation.total.item() == pytest.approx(3.9)


def test_reward_manager_a0_matches_baseline_and_keeps_penalty_unchanged(monkeypatch):
    terms = {
        "bounded_tracking": RewardTermCfg(func=f"{__name__}:_unit_positive_term", weight=2.0),
        "penalty": RewardTermCfg(func=f"{__name__}:_unit_penalty_term", weight=-0.5),
    }
    env = SimpleNamespace(
        num_envs=2,
        max_episode_length_s=10.0,
        positive_raw=torch.tensor([0.25, 0.75]),
        penalty_raw=torch.tensor([1.0, 3.0]),
    )
    baseline = RewardManager(RewardManagerCfg(terms=terms), env, "cpu").compute(0.02)

    semantic_cfg = SemanticKeyframeRewardCfg(enabled=True, semantic_file="unused.json")
    fixed_manager = RewardManager(
        RewardManagerCfg(terms=terms, semantic_keyframe=semantic_cfg),
        env,
        "cpu",
    )
    zeros = torch.zeros(2)
    fake_output = SimpleNamespace(
        active_gate=zeros,
        combined_reward=torch.tensor([0.2, 0.9]),
        valid_objective_count=torch.full((2,), 3.0),
        part_reward=zeros,
        rel_reward=zeros,
        dyn_reward=zeros,
    )
    monkeypatch.setattr(
        "holosoma.managers.reward.manager.get_semantic_keyframe_runtime",
        lambda _env, _cfg: SimpleNamespace(evaluate=lambda: fake_output),
    )
    fixed = fixed_manager.compute(0.02)
    assert torch.equal(fixed, baseline)
    assert torch.equal(fixed_manager.latest_metrics["reward/penalty"], env.penalty_raw * -0.5)
