"""
Test that the robot motion planner's joint angles yield the same end effector pose as the expected end effector pose.
Currently restricted to testing panda-sim-env. Should test again for real robot env.
"""

from typing import List

import numpy as np
import pytest

from demo_aug.configs.env_configs import all_env_configs
from demo_aug.demo import Demo
from demo_aug.envs.base_env import MotionPlannerType
from demo_aug.envs.motion_planners.base_motion_planner import RobotEnvConfig
from demo_aug.envs.nerf_robomimic_env import NeRFRobomimicEnv
from demo_aug.utils.data_collection_utils import load_demos
from demo_aug.utils.mathutils import compute_quaternion_distances


@pytest.fixture
def demo_path() -> str:
    # return "../diffusion_policy/data/robomimic/datasets/lift/ph/1_demo.hdf5"
    return "/juno/u/thankyou/autom/consistency_policy/data/robomimic/datasets/augmented/demo_025trials/2023-11-12/aefnb5l1/lift_trials_25_scaleaug-scalefactor0.67-1.33_se3aug-dx-0.22-0.22-dz-0.02-0.02-dthetaz-0.4-1.1_startaug-eeposbound0.15-eerotbound0.85_envpad0_envinfldist0.08_edgestepsize0.005_envdistfact0.3_PRM_mujocorendereronly.hdf5"


def test_generated_end_effector_pose_matches_desired_end_effector_pose(demo_path):
    """
    Test that the generated end effector pose matches the observed end effector pose when
    we set the observed end effector pose as the target.

    Also, sanity check that the FK result from the observed joint angles matches the observed end effector pose.
    """
    ANGLE_TOL = 0.2  # degrees
    max_demos = 5
    # Load the demo from the specified demo path
    demos: List[Demo] = load_demos(demo_path)[:max_demos]
    for loaded_demo in demos:
        env_cfg = all_env_configs["panda-sim-env"]
        env_cfg.motion_planner_cfg.motion_planner_type = MotionPlannerType.PRM
        env = NeRFRobomimicEnv(env_cfg)

        obs = env.robot_obj.env.reset()
        start_eef_pos = obs["robot0_eef_pos"]
        start_eef_quat_xyzw = obs["robot0_eef_quat"]
        start_eef_quat_wxyz = start_eef_quat_xyzw[[3, 0, 1, 2]]

        # lets get some of the observed joint angles and their corresponding eef pos and quat values. then compare with what e.g. drake is giving us when doing IK
        # because we use drake's IK for the camera obs
        start_eef_quat_xyzw = loaded_demo.get_obs_for_range(0, 0 + 1)[0][
            "robot0_eef_quat"
        ]
        start_eef_quat_wxyz = start_eef_quat_xyzw[[3, 0, 1, 2]]
        start_eef_pos = loaded_demo.get_obs_for_range(0, 0 + 1)[0]["robot0_eef_pos"]
        start_joint_angles = (
            loaded_demo.get_obs_for_range(0, 0 + 1)[0]["robot0_joint_pos"]
            if "robot0_joint_pos" in loaded_demo.get_obs_for_range(0, 0 + 1)[0]
            else loaded_demo.get_obs_for_range(0, 0 + 1)[0]["robot0_joint_qpos"]
        )

        max_timestep = len(loaded_demo.timestep_data) - 1
        desired_eef_quat_xyzw = loaded_demo.get_obs_for_range(
            max_timestep, max_timestep + 1
        )[0]["robot0_eef_quat"]
        desired_eef_quat_wxyz = desired_eef_quat_xyzw[[3, 0, 1, 2]]
        desired_eef_pos = loaded_demo.get_obs_for_range(max_timestep, max_timestep + 1)[
            0
        ]["robot0_eef_pos"]
        desired_joint_angles = (
            loaded_demo.get_obs_for_range(max_timestep, max_timestep + 1)[0][
                "robot0_joint_pos"
            ]
            if "robot0_joint_pos"
            in loaded_demo.get_obs_for_range(max_timestep, max_timestep + 1)[0]
            else loaded_demo.get_obs_for_range(max_timestep, max_timestep + 1)[0][
                "robot0_joint_qpos"
            ]
        )

        desired_eef_pos_from_joint, desired_eef_quat_from_joint = (
            env.robot_obj.forward_kinematics(desired_joint_angles)
        )
        # sanity check desired joint angles yields desired eef pos and quat
        desired_eef_pos_from_joint, desired_eef_quat_from_joint = (
            env.robot_obj.forward_kinematics(desired_joint_angles)
        )
        assert np.allclose(
            desired_eef_pos, desired_eef_pos_from_joint, rtol=0, atol=5e-4
        ), "desired eef pos does not match desired eef pos from joint angles"
        assert (
            compute_quaternion_distances(
                [desired_eef_quat_wxyz, desired_eef_quat_from_joint]
            )[0]
            * 180
            / np.pi
            < ANGLE_TOL
        ), "desired eef quat does not match desired eef quat from joint angles"
        start_eef_pos_from_joint, start_eef_quat_from_joint = (
            env.robot_obj.forward_kinematics(start_joint_angles)
        )
        assert np.allclose(
            start_eef_pos, start_eef_pos_from_joint, rtol=0, atol=5e-4
        ), "start eef pos does not match start eef pos from joint angles"
        assert (
            compute_quaternion_distances(
                [start_eef_quat_wxyz, start_eef_quat_from_joint]
            )[0]
            * 180
            / np.pi
            < ANGLE_TOL
        ), "start eef quat does not match start eef quat from joint angles"
        # compute difference between desired_eef_quat_wxyz and desired_eef_quat_from_joint in angles

        # these start and end values all refer to observation space values
        start_cfg = RobotEnvConfig(
            robot_gripper_qpos=[0.02, 0.02],
            robot_ee_pos=start_eef_pos,
            robot_ee_quat_wxyz=start_eef_quat_wxyz,
            robot_base_pos=env.robot_obj.base_pos,
            robot_base_quat_wxyz=env.robot_obj.base_quat_wxyz,
            task_relev_obj_pos=np.array([0.0, 0.0, 0.0]),
            task_relev_obj_rot=np.eye(3),
            task_relev_obj_pos_nerf=np.array([0.0, 0.0, 0.0]),
            task_relev_obj_rot_nerf=np.eye(3),
        )

        goal_cfg = RobotEnvConfig(
            robot_gripper_qpos=[0.02, 0.02],
            robot_ee_pos=desired_eef_pos,
            robot_ee_quat_wxyz=desired_eef_quat_wxyz,
            robot_base_pos=env.robot_obj.base_pos,
            robot_base_quat_wxyz=env.robot_obj.base_quat_wxyz,
            task_relev_obj_pos=np.array([0.0, 0.0, 0.0]),
            task_relev_obj_rot=np.eye(3),
            task_relev_obj_pos_nerf=np.array([0.0, 0.0, 0.0]),
            task_relev_obj_rot_nerf=np.eye(3),
        )

        reach_traj = env.get_optimal_trajectory(
            start_cfg,
            goal_cfg,
        )

        # get the generated pose by doing FK on the relevant joint angles on the env's robot
        generated_joint_angle_final = reach_traj["robot_joint_qpos"][-1]
        generated_eef_pos_final = (
            reach_traj["robot_ee_pos_eef_site_world"][-1].cpu().numpy()
        )
        generated_eef_quat_final = (
            reach_traj["robot_ee_quat_wxyz_eef_site_world"][-1].cpu().numpy()
        )

        generated_joint_angle_start = reach_traj["robot_joint_qpos"][0]
        generated_eef_pos_start = (
            reach_traj["robot_ee_pos_eef_site_world"][0].cpu().numpy()
        )
        generated_eef_quat_start = (
            reach_traj["robot_ee_quat_wxyz_eef_site_world"][0].cpu().numpy()
        )

        generated_eef_pos_from_joint, generated_eef_quat_from_joint = (
            env.robot_obj.forward_kinematics(generated_joint_angle_final)
        )
        generated_eef_pos_from_joint_start, generated_eef_quat_from_joint_start = (
            env.robot_obj.forward_kinematics(generated_joint_angle_start)
        )

        # tolerate ANGLE_TOLmm error
        assert np.allclose(
            generated_eef_pos_final, generated_eef_pos_from_joint, rtol=0, atol=5e-4
        ), "generated eef pos from joint angles does not match generated eef pos from traj"
        # assert np.allclose(generated_eef_quat_final, generated_eef_quat_from_joint, rtol=0, atol=5e-4), "generated eef quat from joint angles does not match generated eef quat from traj"
        assert (
            compute_quaternion_distances(
                [generated_eef_quat_final, generated_eef_quat_from_joint]
            )[0]
            * 180
            / np.pi
            < ANGLE_TOL
        ), "generated eef quat from joint angles does not match generated eef quat from traj"
        assert np.allclose(
            desired_eef_pos, generated_eef_pos_final, rtol=0, atol=5e-4
        ), "desired eef pos does not match generated eef pos"
        # assert np.allclose(desired_eef_quat_wxyz, generated_eef_quat_final, rtol=0, atol=5e-4), "desired eef quat does not match generated eef quat"
        assert (
            compute_quaternion_distances(
                [desired_eef_quat_wxyz, generated_eef_quat_final]
            )[0]
            * 180
            / np.pi
            < ANGLE_TOL
        ), "desired eef quat does not match generated eef quat"

        assert np.allclose(
            start_eef_pos, generated_eef_pos_from_joint_start, rtol=0, atol=5e-4
        ), "start eef pos does not match generated eef pos from joint angles"
        # assert np.allclose(start_eef_quat_wxyz, generated_eef_quat_from_joint_start, rtol=0, atol=5e-4), "start eef quat does not match generated eef quat from joint angles"
        assert (
            compute_quaternion_distances(
                [start_eef_quat_wxyz, generated_eef_quat_from_joint_start]
            )[0]
            * 180
            / np.pi
            < ANGLE_TOL
        ), "start eef quat does not match generated eef quat from joint angles"

        assert np.allclose(
            start_eef_pos, generated_eef_pos_start, rtol=0, atol=5e-4
        ), "start eef pos does not match generated eef pos"
        # assert np.allclose(start_eef_quat_wxyz, generated_eef_quat_start, rtol=0, atol=5e-4), "start eef quat does not match generated eef quat"
        assert (
            compute_quaternion_distances(
                [start_eef_quat_wxyz, generated_eef_quat_start]
            )[0]
            * 180
            / np.pi
            < ANGLE_TOL
        ), "start eef quat does not match generated eef quat"
