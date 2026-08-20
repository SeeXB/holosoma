"""Whole Body Tracking reward presets for the G1 robot."""

from dataclasses import replace

from holosoma.config_types.reward import RewardManagerCfg, RewardTermCfg, SemanticKeyframeRewardCfg

g1_29dof_wbt_reward = RewardManagerCfg(
    terms={
        # Motion tracking rewards - global reference frame
        "motion_global_ref_position_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_ref_position_error_exp",
            params={"sigma": 0.3},
            weight=0.5,
        ),
        "motion_global_ref_orientation_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_ref_orientation_error_exp",
            params={"sigma": 0.4},
            weight=0.5,
        ),
        # Motion tracking rewards - relative body frame
        "motion_relative_body_position_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_relative_body_position_error_exp",
            params={"sigma": 0.3},
            weight=1.0,
        ),
        "motion_relative_body_orientation_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_relative_body_orientation_error_exp",
            params={"sigma": 0.4},
            weight=1.0,
        ),
        # Motion tracking rewards - body velocities
        "motion_global_body_lin_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_body_lin_vel",
            params={"sigma": 1.0},
            weight=1.0,
        ),
        "motion_global_body_ang_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_body_ang_vel",
            params={"sigma": 3.14},
            weight=1.0,
        ),
        # Regularization rewards
        "action_rate_l2": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:penalty_action_rate",
            weight=-0.1,
        ),
        "limits_dof_pos": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:limits_dof_pos",
            params={"soft_dof_pos_limit": 0.9},
            weight=-10.0,
        ),
        "undesired_contacts": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:UndesiredContacts",
            params={
                "threshold": 1.0,
                "undesired_contacts_body_names": (
                    "^(?!left_foot_contact_point$)(?!right_foot_contact_point$)"
                    "(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$)"
                    "(?!left_ankle_roll_link$)(?!right_ankle_roll_link$).+$"
                ),
            },
            weight=-0.1,
        ),
    }
)

g1_29dof_wbt_fast_sac_reward = RewardManagerCfg(
    terms={
        **g1_29dof_wbt_reward.terms,
        "action_rate_l2": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:penalty_action_rate",
            weight=-1.0,
        ),
        "motion_global_ref_position_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_ref_position_error_exp",
            params={"sigma": 0.3},
            weight=1.0,
        ),
        "motion_global_ref_orientation_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_ref_orientation_error_exp",
            params={"sigma": 0.4},
            weight=0.5,
        ),
        # Motion tracking rewards - relative body frame
        "motion_relative_body_position_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_relative_body_position_error_exp",
            params={"sigma": 0.3},
            weight=2.0,
        ),
        "motion_relative_body_orientation_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_relative_body_orientation_error_exp",
            params={"sigma": 0.4},
            weight=1.0,
        ),
    }
)

g1_29dof_wbt_reward_w_object = RewardManagerCfg(
    terms={
        **g1_29dof_wbt_reward.terms,
        # Motion tracking rewards - global reference frame
        "object_global_ref_position_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:object_global_ref_position_error_exp",
            params={"sigma": 0.3},
            weight=1.0,
        ),
        "object_global_ref_orientation_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:object_global_ref_orientation_error_exp",
            params={"sigma": 0.4},
            weight=1.0,
        ),
    }
)


# Fixed semantic timing/mapping configuration shared by all three normalized
# objectives; it contains no event-name-dependent or reward-magnitude parameter.
semantic_keyframe_config = SemanticKeyframeRewardCfg(
    enabled=True,
    semantic_file=(
        "src/holosoma_retargeting/holosoma_retargeting/demo_data/"
        "semantic_keyframes/sub3_largebox_003_semantic_v2.json"
    ),
    semantic_fps=30.0,
    sigma_time=0.12,
)

semantic_part_only_config = replace(semantic_keyframe_config, enable_rel=False, enable_dyn=False)
semantic_part_rel_config = replace(semantic_keyframe_config, enable_dyn=False)


# Ablations retain the exact baseline term set and differ only in which valid
# normalized semantic objectives participate in the fixed-budget average. The
# original object preset above remains untouched for reproducibility.
g1_29dof_wbt_w_object_semantic_e1_part_reward = RewardManagerCfg(
    terms={**g1_29dof_wbt_reward_w_object.terms},
    semantic_keyframe=semantic_part_only_config,
)

g1_29dof_wbt_w_object_semantic_e2_part_rel_reward = RewardManagerCfg(
    terms={**g1_29dof_wbt_reward_w_object.terms},
    semantic_keyframe=semantic_part_rel_config,
)

g1_29dof_wbt_w_object_semantic_keyframe_reward = RewardManagerCfg(
    terms={**g1_29dof_wbt_reward_w_object.terms},
    semantic_keyframe=semantic_keyframe_config,
)

# The robot-only form uses exactly the same runtime. With no tracked external
# entity its relative-geometry objective is INVALID and excluded from averaging.
g1_29dof_wbt_semantic_keyframe_reward = RewardManagerCfg(
    terms={**g1_29dof_wbt_reward.terms},
    semantic_keyframe=semantic_keyframe_config,
)

# Final paper-facing aliases. Legacy names above remain registered so existing
# launch commands continue to work, but both resolve to the fixed-budget method.
g1_29dof_wbt_semantic_reward = g1_29dof_wbt_semantic_keyframe_reward
g1_29dof_wbt_w_object_semantic_reward = g1_29dof_wbt_w_object_semantic_keyframe_reward

__all__ = [
    "g1_29dof_wbt_fast_sac_reward",
    "g1_29dof_wbt_reward",
    "g1_29dof_wbt_reward_w_object",
    "g1_29dof_wbt_semantic_keyframe_reward",
    "g1_29dof_wbt_semantic_reward",
    "g1_29dof_wbt_w_object_semantic_e1_part_reward",
    "g1_29dof_wbt_w_object_semantic_e2_part_rel_reward",
    "g1_29dof_wbt_w_object_semantic_keyframe_reward",
    "g1_29dof_wbt_w_object_semantic_reward",
    "semantic_keyframe_config",
    "semantic_part_only_config",
    "semantic_part_rel_config",
]
