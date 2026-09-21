from holosoma_retargeting.config_types.data_conversion import DataConversionConfig
from holosoma_retargeting.config_types.data_type import MotionDataConfig
from holosoma_retargeting.config_types.robot import RobotConfig


def test_a3_robot_config_and_smplh_mapping() -> None:
    robot = RobotConfig(robot_type="a3")
    motion = MotionDataConfig(data_format="smplh", robot_type="a3")

    assert robot.ROBOT_DOF == 31
    assert robot.ROBOT_HEIGHT == 1.73
    assert robot.ROBOT_NAME == "a3_31dof"
    assert len(robot.FOOT_STICKING_LINKS) == 8
    assert motion.resolved_joints_mapping["L_Wrist"] == "left_wrist_yaw_Link"
    assert motion.resolved_joints_mapping["R_Toe"] == "right_foot_front_link"


def test_a3_conversion_joint_order_has_all_actuated_joints() -> None:
    config = DataConversionConfig(input_file="motion.npz", robot="a3")
    assert len(config.JOINT_NAMES) == 31
    assert config.JOINT_NAMES[:5] == [
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        "head_yaw_joint",
        "head_pitch_joint",
    ]
