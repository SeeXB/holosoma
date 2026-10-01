"""A3 Semantic-B4 trajectory sampling ablations in HSSim."""

from dataclasses import replace

from holosoma.config_types.command import CommandTermCfg
from holosoma.config_types.scene import IsaacSimPhysicsConfig, PhysicsConfig, RigidObjectConfig, SceneConfig
from holosoma.config_values.a3_robot import a3_31dof_w_object
from holosoma.config_values.wbt.g1.paper_dr import (
    g1_29dof_wbt_w_object_b4_s2_semantic_adaptive_paper_dr,
)


A3_TRACKED_BODIES = [
    "pelvis_link",
    "left_hip_roll_Link", "left_knee_Link", "left_ankle_roll_Link",
    "right_hip_roll_Link", "right_knee_Link", "right_ankle_roll_Link",
    "torso_Link",
    "left_shoulder_roll_Link", "left_elbow_Link", "left_wrist_yaw_Link",
    "right_shoulder_roll_Link", "right_elbow_Link", "right_wrist_yaw_Link",
]

_base = g1_29dof_wbt_w_object_b4_s2_semantic_adaptive_paper_dr
_base_motion = _base.command.setup_terms["motion_command"].params["motion_config"]
_a3_motion = replace(
    _base_motion,
    motion_file=(
        "src/holosoma/holosoma/data/motions/tasks/sub3_largebox_003/"
        "sub3_largebox_003_a3_semantic_b4_aligned_mj_w_obj.npz"
    ),
    body_names_to_track=A3_TRACKED_BODIES,
    body_name_ref=["torso_Link"],
    sampling_mode="semantic_adaptive",
    semantic_file=(
        "exp/semantic_plans/dynamic_json_intermimic_v6/retarget_plan_root/omomo/"
        "sub3_largebox_003/semantic_plan.json"
    ),
    semantic_fps=None,
    enable_default_pose_prepend=False,
    enable_default_pose_append=False,
)


def _a3_command_for_motion(motion_config):
    return replace(
        _base.command,
        setup_terms={
            "motion_command": CommandTermCfg(
                func="holosoma.managers.command.terms.wbt:MotionCommand",
                params={"motion_config": motion_config},
            )
        },
    )


_a3_command = _a3_command_for_motion(_a3_motion)
_a3_original_adaptive_command = _a3_command_for_motion(
    replace(_a3_motion, sampling_mode="original_adaptive")
)

_termination_terms = dict(_base.termination.terms)
_bad_tracking = _termination_terms["bad_tracking"]
_termination_terms["bad_tracking"] = replace(
    _bad_tracking,
    params={
        **_bad_tracking.params,
        "body_names_to_track": A3_TRACKED_BODIES,
        "bad_motion_body_pos_body_names": [
            "left_ankle_roll_Link", "right_ankle_roll_Link",
            "left_wrist_yaw_Link", "right_wrist_yaw_Link",
        ],
    },
)
_a3_termination = replace(_base.termination, terms=_termination_terms)

_a3_object_scene = SceneConfig(
    env_spacing=0.0,
    rigid_objects={
        "object": RigidObjectConfig(
            urdf_file="holosoma/data/scene_objects/boxes/large_box_a3_030_034_034.urdf",
            physics=PhysicsConfig(
                isaacsim=IsaacSimPhysicsConfig(
                    static_friction=1.2,
                    dynamic_friction=1.2,
                    restitution=0.0,
                    friction_combine_mode="multiply",
                )
            ),
        )
    },
)

a3_31dof_wbt_w_object_b4_s2_semantic_adaptive_paper_dr = replace(
    _base,
    training=replace(
        _base.training,
        name="a3_sub3_largebox_003_semanticb4_semanticadaptive",
    ),
    robot=a3_31dof_w_object,
    command=_a3_command,
    termination=_a3_termination,
    scene=_a3_object_scene,
)

a3_31dof_wbt_w_object_b4_s0_original_adaptive_paper_dr = replace(
    a3_31dof_wbt_w_object_b4_s2_semantic_adaptive_paper_dr,
    training=replace(
        a3_31dof_wbt_w_object_b4_s2_semantic_adaptive_paper_dr.training,
        name="a3_sub3_largebox_003_semanticb4_originalrl",
    ),
    command=_a3_original_adaptive_command,
)

__all__ = [
    "a3_31dof_wbt_w_object_b4_s0_original_adaptive_paper_dr",
    "a3_31dof_wbt_w_object_b4_s2_semantic_adaptive_paper_dr",
]
