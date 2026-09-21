"""Training-only Omni/Semantic presets aligned to OmniRetarget's published DR.

The paper specifies the following runtime randomization for object interaction:

* final object mass 0.1--2.0 kg;
* object COM offset +/-0.08 m on x/y/z;
* object inertia 50--150%;
* object shape +/-10%;
* torso COM x/y/z +/-0.025/0.05/0.075 m;
* joint default-position bias +/-0.01 rad;
* pushes up to 0.3 m/s and 0.78 rad/s every 1--3 s;
* observation noise of 0.05 Rot6D, 0.5/0.2 m/s or rad/s base
  linear/angular velocity, and 0.01/0.5 rad or rad/s joint position/velocity.

This preset implements every item supported safely by the current replicated
IsaacSim scene.  Per-environment shape scaling is deliberately NOT claimed:
IsaacLab requires it before simulation startup with ``replicate_physics=False``;
the current Holosoma manager runs setup terms after the replicated scene starts.
Hot-scaling collision geometry would be physically undefined.  Shape +/-10%
therefore remains the one documented paper-DR gap in these runs.

The paper does not list robot/object material DR, reset-pose noise, actuator
gain/RFI DR, or action delay.  Material terms are absent, reset-pose noise is
zeroed, and the disabled actuator/action-delay state terms remain only because
the action and manager paths require their bookkeeping objects.
"""

from __future__ import annotations

import math
from dataclasses import replace

from holosoma.config_types.command import CommandTermCfg
from holosoma.config_types.observation import ObservationManagerCfg, ObsGroupCfg, ObsTermCfg
from holosoma.config_types.randomization import (
    BaseComRange,
    InertiaScale,
    RandomizationManagerCfg,
    RandomizationTermCfg,
)
from holosoma.config_values.wbt.g1.command import motion_config_w_object_transition_truncated_b4
from holosoma.config_values.wbt.g1.experiment import (
    g1_29dof_wbt_w_object_semantic_r1_b4_omni,
)
from holosoma.config_values.wbt.g1.reward import g1_29dof_wbt_reward_w_object
from holosoma.config_values.wbt.g1.observation import (
    actor_obs_shared,
    g1_29dof_wbt_observation_w_object,
)
from holosoma.config_values.wbt.g1.randomization import (
    base_reset_terms,
    base_setup_terms,
    base_step_terms,
)
from holosoma.config_values.wbt.g1.termination import g1_29dof_wbt_termination


_paper_setup_terms = {
    # Required state holders; their perturbations remain disabled.
    "actuator_randomizer_state": base_setup_terms["actuator_randomizer_state"],
    "setup_action_delay_buffers": base_setup_terms["setup_action_delay_buffers"],
    # Four published robot DR terms (observation noise is configured below).
    "push_randomizer_state": RandomizationTermCfg(
        func=base_setup_terms["push_randomizer_state"].func,
        params={
            "push_interval_s": [1.0, 3.0],
            "max_push_vel": [0.3, 0.3, 0.3, 0.78, 0.78, 0.78],
            "enabled": True,
        },
    ),
    "randomize_base_com_startup": RandomizationTermCfg(
        func=base_setup_terms["randomize_base_com_startup"].func,
        params={
            "base_com_range": BaseComRange(
                x=[-0.025, 0.025],
                y=[-0.05, 0.05],
                z=[-0.075, 0.075],
            ),
            "enabled": True,
        },
    ),
    "setup_dof_pos_bias": RandomizationTermCfg(
        func=base_setup_terms["setup_dof_pos_bias"].func,
        params={"dof_pos_bias_range": [-0.01, 0.01], "enabled": True},
    ),
    # Published object-property DR.  large_box.urdf is authored at 0.1 kg, so
    # additive [0, 1.9] produces the requested final [0.1, 2.0] kg range.
    "randomize_object_rigid_body_mass_startup": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.objects:randomize_object_rigid_body_mass_startup",
        # Mass and inertia are separate paper DR channels.  Do not let the
        # generic mass writer pre-scale inertia by the sampled mass ratio;
        # the independent term below must leave final inertia at 50--150% of
        # the authored nominal value, not 50--150% of a mass-scaled value.
        params={"mass_distribution_params": [0.0, 1.9], "recompute_inertia": False},
    ),
    "randomize_object_rigid_body_com_startup": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.objects:randomize_object_rigid_body_com_startup",
        params={
            "com_distribution_params": {
                "x": [-0.08, 0.08],
                "y": [-0.08, 0.08],
                "z": [-0.08, 0.08],
            }
        },
    ),
    "randomize_object_rigid_body_inertia_startup": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.objects:randomize_object_rigid_body_inertia_startup",
        params={
            "inertia_distribution_params_dict": InertiaScale(
                Ixx=[0.5, 1.5],
                Iyy=[0.5, 1.5],
                Izz=[0.5, 1.5],
            )
        },
    ),
}

paper_randomization = RandomizationManagerCfg(
    setup_terms=_paper_setup_terms,
    reset_terms=dict(base_reset_terms),
    step_terms=dict(base_step_terms),
)


# The actor previously omitted pelvis linear velocity and pelvis position error.
# The paper's full proprioceptive object policy contains both.  Noise magnitudes
# below are literal uniform half-widths, matching ObservationManager semantics.
_paper_actor_terms = dict(actor_obs_shared.terms)
_paper_actor_terms.update(
    {
        "motion_ref_pos_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:motion_ref_pos_b",
            scale=1.0,
            noise=0.0,
        ),
        "base_lin_vel": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:base_lin_vel",
            scale=1.0,
            noise=0.5,
        ),
    }
)
# Assert the already-present channels really match the paper instead of merely
# inheriting them by accident.
_paper_actor_terms["motion_ref_ori_b"] = replace(_paper_actor_terms["motion_ref_ori_b"], noise=0.05)
_paper_actor_terms["base_ang_vel"] = replace(_paper_actor_terms["base_ang_vel"], noise=0.2)
_paper_actor_terms["dof_pos"] = replace(_paper_actor_terms["dof_pos"], noise=0.01)
_paper_actor_terms["dof_vel"] = replace(_paper_actor_terms["dof_vel"], noise=0.5)

paper_observation = ObservationManagerCfg(
    groups={
        "actor_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=True,
            history_length=1,
            terms=_paper_actor_terms,
        ),
        "critic_obs": g1_29dof_wbt_observation_w_object.groups["critic_obs"],
    }
)


# Remove undocumented reset-pose perturbations while retaining the shared-XY
# implementation in the base config for compatibility.
_paper_motion = replace(
    motion_config_w_object_transition_truncated_b4,
    motion_file=(
        "src/holosoma/holosoma/data/motions/benchmarks/benchmark_results_full_event_transition_truncation/"
        "transition_truncated_b4_mj_fps50_w_obj.npz"
    ),
    use_adaptive_timesteps_sampler=True,
    sampling_mode="original_adaptive",
    semantic_file=(
        "src/holosoma_retargeting/holosoma_retargeting/demo_data/semantic_keyframes/"
        "sub3_largebox_003_semantic_v2.json"
    ),
    semantic_fps=None,
    noise_to_initial_pose=replace(
        motion_config_w_object_transition_truncated_b4.noise_to_initial_pose,
        overall_noise_scale=0.0,
    ),
)


def _paper_command_for_sampling_mode(sampling_mode: str):
    motion = replace(_paper_motion, sampling_mode=sampling_mode)
    return replace(
        g1_29dof_wbt_w_object_semantic_r1_b4_omni.command,
        setup_terms={
            "motion_command": CommandTermCfg(
                func="holosoma.managers.command.terms.wbt:MotionCommand",
                params={"motion_config": motion},
            )
        },
    )


paper_command = _paper_command_for_sampling_mode("original_adaptive")
paper_command_semantic_uniform = _paper_command_for_sampling_mode("semantic_uniform")
paper_command_semantic_adaptive = _paper_command_for_sampling_mode("semantic_adaptive")


# Published object thresholds.  The paper does not publish the numerical body
# thresholds or the exact switch point for enabling object termination, so the
# repository's body thresholds stay unchanged and object thresholds apply from
# the start of these runs.
_termination_terms = dict(g1_29dof_wbt_termination.terms)
_bad_tracking = _termination_terms["bad_tracking"]
_termination_terms["bad_tracking"] = replace(
    _bad_tracking,
    params={
        **_bad_tracking.params,
        "bad_object_pos_threshold": 1.0,
        "bad_object_ori_threshold": math.pi / 4.0,
    },
)
paper_termination = replace(g1_29dof_wbt_termination, terms=_termination_terms)


def _paper_experiment(base, *, command_cfg=paper_command):
    return replace(
        base,
        randomization=paper_randomization,
        observation=paper_observation,
        command=command_cfg,
        termination=paper_termination,
        reward=g1_29dof_wbt_reward_w_object,
    )


g1_29dof_wbt_w_object_b4_omni_paper_dr = _paper_experiment(
    g1_29dof_wbt_w_object_semantic_r1_b4_omni
)
g1_29dof_wbt_w_object_b4_semantic_paper_dr = _paper_experiment(
    g1_29dof_wbt_w_object_semantic_r1_b4_omni,
    command_cfg=paper_command_semantic_adaptive,
)

# First-round sampling ablations.  All three presets share the original Omni
# reward, PPO, DR, observations, and termination; only reset-time sampling mode
# differs.
g1_29dof_wbt_w_object_b4_s0_original_adaptive_paper_dr = _paper_experiment(
    g1_29dof_wbt_w_object_semantic_r1_b4_omni,
    command_cfg=paper_command,
)
g1_29dof_wbt_w_object_b4_s1_semantic_uniform_paper_dr = _paper_experiment(
    g1_29dof_wbt_w_object_semantic_r1_b4_omni,
    command_cfg=paper_command_semantic_uniform,
)
g1_29dof_wbt_w_object_b4_s2_semantic_adaptive_paper_dr = _paper_experiment(
    g1_29dof_wbt_w_object_semantic_r1_b4_omni,
    command_cfg=paper_command_semantic_adaptive,
)

# Anatomy-v2 experiment. Keep the first-round sampling-only S2 preset frozen
# so loading/evaluating its existing checkpoints does not change reward rules.
_semantic_contact_terms = dict(g1_29dof_wbt_reward_w_object.terms)
_semantic_contact_terms["undesired_contacts"] = replace(
    _semantic_contact_terms["undesired_contacts"],
    func="holosoma.managers.reward.terms.wbt:SemanticPlanUndesiredContacts",
)
_semantic_contact_motion = replace(
    _paper_motion,
    sampling_mode="semantic_adaptive",
    semantic_file="",  # Require the task-specific reviewed plan at launch.
)
_semantic_contact_command = replace(
    paper_command_semantic_adaptive,
    setup_terms={"motion_command": CommandTermCfg(
        func="holosoma.managers.command.terms.wbt:MotionCommand",
        params={"motion_config": _semantic_contact_motion},
    )},
)
g1_29dof_wbt_w_object_b4_s2_semantic_contacts_paper_dr = replace(
    g1_29dof_wbt_w_object_b4_s2_semantic_adaptive_paper_dr,
    command=_semantic_contact_command,
    reward=replace(g1_29dof_wbt_reward_w_object, terms=_semantic_contact_terms),
)

__all__ = [
    "g1_29dof_wbt_w_object_b4_s2_semantic_contacts_paper_dr",
    "g1_29dof_wbt_w_object_b4_omni_paper_dr",
    "g1_29dof_wbt_w_object_b4_semantic_paper_dr",
    "g1_29dof_wbt_w_object_b4_s0_original_adaptive_paper_dr",
    "g1_29dof_wbt_w_object_b4_s1_semantic_uniform_paper_dr",
    "g1_29dof_wbt_w_object_b4_s2_semantic_adaptive_paper_dr",
]
