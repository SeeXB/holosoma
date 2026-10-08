"""A3 31-DoF robot configuration for HSSim/IsaacSim training."""

from pathlib import Path

from holosoma.config_types.robot import (
    RobotAssetConfig,
    RobotBridgeConfig,
    RobotConfig,
    RobotControlConfig,
    RobotInitState,
)
from holosoma.config_types.scene import IsaacSimPhysicsConfig, PhysicsConfig, PhysXPhysicsConfig


_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_A3_ASSET_ROOT = _PROJECT_ROOT / "src/holosoma_retargeting/holosoma_retargeting/demo_data/models"

A3_JOINT_NAMES = [
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "head_yaw_joint", "head_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
]

A3_BODY_NAMES = [
    "pelvis_link", "waist_yaw_Link", "waist_roll_Link", "torso_Link",
    "head_yaw_Link", "head_pitch_Link",
    "left_shoulder_pitch_Link", "left_shoulder_roll_Link", "left_shoulder_yaw_Link",
    "left_elbow_Link", "left_wrist_roll_Link", "left_wrist_pitch_Link", "left_wrist_yaw_Link",
    "right_shoulder_pitch_Link", "right_shoulder_roll_Link", "right_shoulder_yaw_Link",
    "right_elbow_Link", "right_wrist_roll_Link", "right_wrist_pitch_Link", "right_wrist_yaw_Link",
    "left_hip_pitch_Link", "left_hip_roll_Link", "left_hip_yaw_Link", "left_knee_Link",
    "left_ankle_pitch_Link", "left_ankle_roll_Link",
    "right_hip_pitch_Link", "right_hip_roll_Link", "right_hip_yaw_Link", "right_knee_Link",
    "right_ankle_pitch_Link", "right_ankle_roll_Link",
]

_LEFT_ARM = A3_JOINT_NAMES[5:12]
_RIGHT_ARM = A3_JOINT_NAMES[12:19]
_LEFT_LEG = A3_JOINT_NAMES[19:25]
_RIGHT_LEG = A3_JOINT_NAMES[25:31]

_DEFAULT_ANGLES = dict(
    zip(
        A3_JOINT_NAMES,
        [
            -0.102833337647, -0.013908875794, 0.418878999967, 0.0, 0.0,
            -0.203177615045, 0.215229948948, -0.019749876704, 0.507039851986,
            -0.000621841690, 0.094388786112, 0.0,
            -0.169786949816, -0.240327164168, 0.162140049005, 0.430922724784,
            0.003176605041, 0.082325753100, 0.0,
            -0.630740218417, 0.020790004027, 0.236128138755, 0.642632143649,
            -0.255090542926, 0.110568696520,
            -0.647141883229, -0.146646794466, -0.094787425768, 0.517665846146,
            -0.210058990113, -0.073540410396,
        ],
        strict=True,
    )
)

a3_31dof = RobotConfig(
    num_bodies=32,
    dof_obs_size=31,
    actions_dim=31,
    policy_obs_dim=-1,
    critic_obs_dim=-1,
    algo_obs_dim_dict={},
    key_bodies=["left_ankle_roll_Link", "right_ankle_roll_Link"],
    num_feet=2,
    foot_body_name="ankle_roll_Link",
    foot_height_name="ankle_roll_Link",
    knee_name="knee_Link",
    torso_name="torso_Link",
    dof_names=A3_JOINT_NAMES,
    upper_dof_names=A3_JOINT_NAMES[:19],
    upper_left_arm_dof_names=_LEFT_ARM,
    upper_right_arm_dof_names=_RIGHT_ARM,
    lower_dof_names=_LEFT_LEG + _RIGHT_LEG,
    has_torso=True,
    has_upper_body_dof=True,
    left_ankle_dof_names=_LEFT_LEG[-2:],
    right_ankle_dof_names=_RIGHT_LEG[-2:],
    knee_dof_names=["left_knee_joint", "right_knee_joint"],
    hips_dof_names=_LEFT_LEG[:3] + _RIGHT_LEG[:3],
    dof_pos_lower_limit_list=[
        -2.617993878, -0.349065850, -0.488692191, -1.047197551, -0.436332313,
        -2.879793266, -0.087266463, -2.792526803, -0.959931089, -2.792526803,
        -1.623156204, -1.623156204, -2.879793266, -2.617993878, -2.792526803,
        -0.959931089, -2.792526803, -1.623156204, -1.623156204,
        -2.513274123, -0.523598776, -2.722713633, -0.122173048, -0.907571211,
        -0.349065850, -2.513274123, -1.605702912, -2.722713633, -0.122173048,
        -0.907571211, -0.349065850,
    ],
    dof_pos_upper_limit_list=[
        2.617993878, 0.349065850, 0.418879020, 1.047197551, 0.261799388,
        2.879793266, 2.617993878, 2.792526803, 1.745329252, 2.792526803,
        1.623156204, 1.623156204, 2.879793266, 0.087266463, 2.792526803,
        1.745329252, 2.792526803, 1.623156204, 1.623156204,
        2.932153143, 1.605702912, 2.722713633, 2.495820830, 0.523598776,
        0.349065850, 2.932153143, 0.523598776, 2.722713633, 2.495820830,
        0.523598776, 0.349065850,
    ],
    dof_vel_limit_list=[
        12.042771839, 22.7, 9.24785, 12.775810125, 12.775810125,
        13.613568166, 13.613568166, 15.707963268, 15.707963268, 15.707963268,
        12.775810125, 12.775810125, 13.613568166, 13.613568166, 15.707963268,
        15.707963268, 15.707963268, 12.775810125, 12.775810125,
        12.042771839, 12.042771839, 12.042771839, 14.660765717, 10.8, 19.37,
        12.042771839, 12.042771839, 12.042771839, 14.660765717, 10.8, 19.37,
    ],
    dof_effort_limit_list=[
        220, 46, 118, 6, 6, 60, 60, 24, 24, 24, 6, 6,
        60, 60, 24, 24, 24, 6, 6,
        220, 220, 220, 320, 118.2, 54.75, 220, 220, 220, 320, 118.2, 54.75,
    ],
    dof_armature_list=[0.001] * 31,
    dof_joint_friction_list=[
        1.1971, 0.69223, 1.7, 0.1, 0.1,
        0.6293, 0.6293, 0.41197, 0.41197, 0.41197, 0.1, 0.1,
        0.6293, 0.6293, 0.41197, 0.41197, 0.41197, 0.1, 0.1,
        1.1971, 1.1971, 1.1971, 2.4276, 1.4, 0.778,
        1.1971, 1.1971, 1.1971, 2.4276, 1.4, 0.778,
    ],
    body_names=A3_BODY_NAMES,
    terminate_after_contacts_on=["pelvis", "shoulder", "hip"],
    penalize_contacts_on=["pelvis", "shoulder", "hip"],
    init_state=RobotInitState(
        pos=[0.0, 0.0, 1.0],
        rot=[0.0, 0.0, 0.0, 1.0],
        lin_vel=[0.0, 0.0, 0.0],
        ang_vel=[0.0, 0.0, 0.0],
        default_joint_angles=_DEFAULT_ANGLES,
    ),
    randomize_link_body_names=[
        "pelvis_link", "torso_Link", "left_hip_pitch_Link", "left_hip_roll_Link",
        "left_hip_yaw_Link", "left_knee_Link", "right_hip_pitch_Link", "right_hip_roll_Link",
        "right_hip_yaw_Link", "right_knee_Link",
    ],
    waist_dof_names=A3_JOINT_NAMES[:3],
    waist_yaw_dof_name="waist_yaw_joint",
    waist_roll_dof_name="waist_roll_joint",
    waist_pitch_dof_name="waist_pitch_joint",
    arm_dof_names=_LEFT_ARM + _RIGHT_ARM,
    left_arm_dof_names=_LEFT_ARM,
    right_arm_dof_names=_RIGHT_ARM,
    symmetry_joint_names={
        **{name: name.replace("left_", "right_") for name in _LEFT_ARM + _LEFT_LEG},
        **{name: name.replace("right_", "left_") for name in _RIGHT_ARM + _RIGHT_LEG},
        **{name: name for name in A3_JOINT_NAMES[:5]},
    },
    flip_sign_joint_names=[
        "waist_yaw_joint", "waist_roll_joint", "head_yaw_joint",
        "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_wrist_roll_joint",
        "left_wrist_yaw_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
        "right_wrist_roll_joint", "right_wrist_yaw_joint",
        "left_hip_roll_joint", "left_hip_yaw_joint", "left_ankle_roll_joint",
        "right_hip_roll_joint", "right_hip_yaw_joint", "right_ankle_roll_joint",
    ],
    apply_dof_armature_in_isaacgym=True,
    contact_pairs_multiplier=16,
    control=RobotControlConfig(
        control_type="P",
        stiffness={
            "waist": 420.0, "head": 70.0, "shoulder": 300.0, "elbow": 300.0,
            "wrist": 120.0, "hip": 500.0, "knee": 500.0, "ankle": 300.0,
        },
        damping={
            "waist": 20.0, "head": 5.0, "shoulder": 14.0, "elbow": 14.0,
            "wrist": 7.0, "hip": 24.0, "knee": 24.0, "ankle": 16.0,
        },
        action_scale=0.25,
        action_clip_value=100.0,
        clip_actions=True,
        clip_torques=True,
        action_scales_by_effort_limit_over_p_gain=True,
    ),
    asset=RobotAssetConfig(
        asset_root=str(_A3_ASSET_ROOT),
        collapse_fixed_joints=True,
        replace_cylinder_with_capsule=True,
        flip_visual_attachments=False,
        armature=0.001,
        thickness=0.01,
        urdf_file="a3/a3_31dof.urdf",
        usd_file=None,
        xml_file="a3/a3_31dof.xml",
        robot_type="a3_31dof",
        enable_self_collisions=True,
        default_dof_drive_mode=3,
        fix_base_link=False,
        link_physics=PhysicsConfig(
            physx=PhysXPhysicsConfig(
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
            ),
            isaacsim=IsaacSimPhysicsConfig(
                static_friction=1.2,
                dynamic_friction=1.2,
                restitution=0.0,
                friction_combine_mode="multiply",
            ),
        ),
    ),
    bridge=RobotBridgeConfig(sdk_type="ros2", motor_type="serial"),
)

a3_31dof_w_object = a3_31dof

__all__ = ["A3_BODY_NAMES", "A3_JOINT_NAMES", "a3_31dof", "a3_31dof_w_object"]
