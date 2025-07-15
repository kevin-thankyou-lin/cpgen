import logging
import pathlib
from typing import Dict, List

import cv2
import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from demo_aug.envs.base_env import CameraConfig, EnvConfig, MultiCameraConfig
from demo_aug.envs.motion_planners.base_motion_planner import RobotEnvConfig
from demo_aug.envs.nerf_robomimic_env import NeRFRobomimicEnv
from demo_aug.objects.nerf_object import NeRFObject

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    # set seeds
    np.random.seed(1)
    torch.manual_seed(1)

    env_cfg = EnvConfig(
        multi_camera_cfg=MultiCameraConfig(
            [
                CameraConfig(
                    name="agentview",
                    height=512,
                    width=512,
                    fx=618.0386719675123,
                    fy=618.0386719675123,
                    cx=256.0,
                    cy=256.0,
                ),
                CameraConfig(
                    name="robot0_eye_in_hand",
                    height=512,
                    width=512,
                    fx=333.62569545,
                    fy=333.62569545,
                    cx=256.0,
                    cy=256.0,
                ),
            ]
        )
    )
    nerf_robomimic_env = NeRFRobomimicEnv(env_cfg)
    robot_joint_qpos = np.array(
        [
            -0.04141039,
            0.21736869,
            0.00753974,
            -2.5898454,
            -0.00784382,
            2.95545758,
            0.77382831,
        ]
    )
    start_idx = 5
    # goal_idx = 41
    goal_idx = 48
    max_future_steps = 14
    hardcode_open_gripper = True

    f = h5py.File(
        pathlib.Path(
            "~/autom/diffusion_policy/data/robomimic/datasets/lift/ph/image_abs.hdf5"
        ).expanduser(),
        "r",
    )
    task_relev_obj_nerf_path = pathlib.Path(
        "../nerfstudio/outputs/robomimic_lift_2023-05-25/tensorf/2023-05-25_160044/config.yml"
    )
    task_relev_obj_mesh_path: str = (
        "package://models/assets/task_relevant/mesh-outputs/mesh.sdf"
    )

    env = nerf_robomimic_env.robot_obj.env
    demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    ep = demos[0]
    states = f["data/{}/states".format(ep)][()]

    # set start state constraints for optimization
    start_state = dict(states=states[start_idx])
    env.reset()
    env_reset_start_state = env.reset_to(start_state)

    base_pos = np.array(
        [-0.56, 0.0, 0.912]
    )  # nerf_robomimic_env.robot_obj.base_pos  querying env causes segfault
    base_quat_wxyz = np.array(
        [1, 0, 0, 0]
    )  # nerf_robomimic_env.robot_obj.base_quat_wxyz
    ee_start_pos = env_reset_start_state["robot0_eef_pos"]
    ee_start_quat_xyzw = env_reset_start_state["robot0_eef_quat"]

    joint_qpos_idxs = (
        np.array(env.env.robots[0]._ref_joint_pos_indexes, dtype=np.uint8) + 1
    )
    gripper_qpos_idxs = np.array(
        [joint_qpos_idxs[-1] + 1, joint_qpos_idxs[-1] + 2], dtype=np.uint8
    )
    robot_joint_qpos_start = start_state["states"][joint_qpos_idxs]
    robot_gripper_qpos_start = start_state["states"][gripper_qpos_idxs]
    robot_gripper_qpos_start[-1] = -robot_gripper_qpos_start[-1]
    robot_gripper_qpos_start[-2] = 0.04  # start_state["states"][gripper_qpos_idxs]
    robot_gripper_qpos_start[-1] = 0.04  # -robot_gripper_qpos_start[-1]

    goal_state = dict(states=states[goal_idx])
    env.reset()
    env_reset_goal_state = env.reset_to(goal_state)
    ee_goal_pos = env_reset_goal_state["robot0_eef_pos"]
    ee_goal_quat_xyzw = env_reset_goal_state["robot0_eef_quat"]
    ee_goal_quat_wxyz = [
        ee_goal_quat_xyzw[3],
        ee_goal_quat_xyzw[0],
        ee_goal_quat_xyzw[1],
        ee_goal_quat_xyzw[2],
    ]
    task_relev_obj_pos = env.env.sim.data.body_xpos[env.env.cube_body_id].copy()
    task_relev_obj_quat_wxyz = env.env.sim.data.body_xquat[env.env.cube_body_id].copy()

    ee_goal_pos_data = env.env.sim.data.get_body_xpos("robot0_right_hand").copy()
    ee_goal_quat_wxyz_data = env.env.sim.data.get_body_xquat("robot0_right_hand").copy()

    robot_joint_qpos_goal = goal_state["states"][joint_qpos_idxs]
    robot_gripper_qpos_goal = goal_state["states"][gripper_qpos_idxs]
    robot_gripper_qpos_goal[-1] = -robot_gripper_qpos_goal[-1]
    if hardcode_open_gripper:
        robot_gripper_qpos_goal[-1] = 0.04
        robot_gripper_qpos_goal[-2] = 0.04
        robot_gripper_qpos_start[-1] = 0.04
        robot_gripper_qpos_start[-2] = 0.04

    orig_start_cfg = RobotEnvConfig(
        robot_joint_qpos_start.copy(),
        robot_gripper_qpos_start.copy(),
        robot_ee_pos=None,
        robot_ee_quat_wxyz=None,
        robot_base_pos=base_pos.copy(),
        robot_base_quat_wxyz=base_quat_wxyz.copy(),
        task_relev_obj_pos=task_relev_obj_pos.copy(),
        task_relev_obj_quat_wxyz=task_relev_obj_quat_wxyz.copy(),
        task_irrelev_obj_pos=None,
        task_irrelev_obj_quat_wxyz=None,
    )
    orig_goal_cfg = RobotEnvConfig(
        robot_joint_qpos=robot_joint_qpos_goal.copy(),
        robot_gripper_qpos=robot_gripper_qpos_goal.copy(),
        robot_ee_pos=ee_goal_pos.copy(),
        robot_ee_quat_wxyz=ee_goal_quat_wxyz.copy(),
        robot_base_pos=base_pos.copy(),
        robot_base_quat_wxyz=base_quat_wxyz.copy(),
        task_relev_obj_pos=task_relev_obj_pos.copy(),
        task_relev_obj_quat_wxyz=task_relev_obj_quat_wxyz.copy(),
        task_irrelev_obj_pos=None,
        task_irrelev_obj_quat_wxyz=None,
    )

    future_task_relev_objs = []
    for i in range(max_future_steps):
        if goal_idx + i >= len(states):
            break
        future_task_relev_objs.append(
            NeRFObject(
                task_relev_obj_nerf_path,
                bounding_box_min=torch.FloatTensor([-0.07, -0.07, 0.8]),
                bounding_box_max=torch.FloatTensor([0.07, 0.07, 0.86]),
            )
        )

    # R.from_quat(orig_goal_cfg.task_relev_obj_quat_xyzw)
    orig_future_cfg_list: List[RobotEnvConfig] = []
    for i in range(max_future_steps):
        if goal_idx + i >= len(states):
            break
        future_state = dict(states=states[goal_idx + i])
        env.reset()
        env_reset_future_state = env.reset_to(future_state)
        ee_future_pos = env_reset_future_state["robot0_eef_pos"]
        ee_future_quat_xyzw = env_reset_future_state["robot0_eef_quat"]
        ee_future_quat_wxyz = [
            ee_future_quat_xyzw[3],
            ee_future_quat_xyzw[0],
            ee_future_quat_xyzw[1],
            ee_future_quat_xyzw[2],
        ]
        task_relev_obj_pos_future = env.env.sim.data.body_xpos[
            env.env.cube_body_id
        ].copy()
        task_relev_obj_quat_wxyz_future = env.env.sim.data.body_xquat[
            env.env.cube_body_id
        ].copy()
        # technically, the task_relev pos/quat future are always eye(4)
        # because I'd be using a new nerf anyways ... esp. the case for non rigid objects
        ee_future_pos_data = env.env.sim.data.get_body_xpos("robot0_right_hand")
        ee_future_quat_wxyz_data = env.env.sim.data.get_body_xquat("robot0_right_hand")

        robot_joint_qpos_future = future_state["states"][joint_qpos_idxs]
        robot_gripper_qpos_future = future_state["states"][gripper_qpos_idxs]
        robot_gripper_qpos_future[-1] = -robot_gripper_qpos_future[-1]
        orig_future_cfg_list.append(
            RobotEnvConfig(
                robot_joint_qpos=robot_joint_qpos_future,
                robot_gripper_qpos=robot_gripper_qpos_future,
                robot_ee_pos=ee_future_pos,
                robot_ee_quat_wxyz=ee_future_quat_wxyz,
                robot_base_pos=base_pos,
                robot_base_quat_wxyz=base_quat_wxyz,
                task_relev_obj_pos=None,  # task_relev_obj_pos_future,
                task_relev_obj_quat_wxyz=None,  # task_relev_obj_quat_wxyz_future,
                task_irrelev_obj_pos=None,
                task_irrelev_obj_quat_wxyz=None,
            )
        )

    test_obs = False
    test_motion_planner = False
    test_task_relev_obj_se3_randomization = False
    test_transf_future_cfgs = True
    if test_obs:
        obs = nerf_robomimic_env.get_observation(
            robot_joint_qpos, robot_gripper_qpos_start, torch.eye(4)
        )
        # Define the observation keys to handle
        obs_keys = [
            "agentview_rgb",
            "agentview_acc",
            "robot0_eye_in_hand_rgb",
            "robot0_eye_in_hand_acc",
        ]
        # save_dir = f"test-nerf-robomimic-obs-goal-idx{goal_idx}"
        save_dir = "a_nerf_goal"

        for key in obs_keys:
            # TODO(klin): unclear why the images are reversed
            file_path = f"images/frame_{key}.png"
            full_file_path = pathlib.Path(save_dir) / file_path

            # create directory if it doesn't exist
            full_file_path.parent.mkdir(parents=True, exist_ok=True)

            if "rgb" in key:
                success = cv2.imwrite(
                    str(full_file_path), cv2.cvtColor(obs[key], cv2.COLOR_BGR2RGB)
                )
            elif "acc" in key:
                vis_acc = cv2.normalize(obs[key], None, 0, 255, cv2.NORM_MINMAX).astype(
                    "uint8"
                )
                success = cv2.imwrite(str(full_file_path), vis_acc)
            else:
                success = False

            if not success:
                logging.warning(f"Failed to save {key} to {full_file_path}")

    if test_motion_planner:
        optimal_trajectory = nerf_robomimic_env.get_optimal_trajectory(
            orig_start_cfg,
            orig_goal_cfg,
            "package://models/assets/task_relevant/mesh-outputs/mesh.sdf",
        )
        obs_lst: List[Dict[str, np.ndarray]] = nerf_robomimic_env.get_observations(
            optimal_trajectory["robot_joint_qpos"],
            optimal_trajectory["robot_gripper_qpos"],
            task_relev_obj_pose=None,
            task_irrelev_obj_pose=None,
        )
        save_dir = pathlib.Path("test_motion_planning_again_again")
        for i, obs in enumerate(obs_lst):
            for cam in env.env.camera_names:
                rgb = obs[f"{cam}_rgb"]
                # create directory if it doesn't exist
                full_file_path = save_dir / f"{cam}_rgb{i}.png"
                full_file_path.parent.mkdir(parents=True, exist_ok=True)

                # save normal image
                success = cv2.imwrite(
                    str(full_file_path), cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                )
                if not success:
                    logging.warning(f"Failed to save image to {full_file_path}")
        logging.info(f"Saved images to {save_dir}")

    if test_task_relev_obj_se3_randomization:
        for idx in range(10):
            logging.info(f"idx: {idx}")
            new_robot_qpos_start: Dict[str, torch.Tensor] = (
                nerf_robomimic_env.sample_robot_qpos()
            )
            X_se3 = nerf_robomimic_env.sample_task_relev_obj_se3_transform()
            pos_transf, rot_transf = X_se3[:3, 3], X_se3[:3, :3]

            # apply rot transform to task relevant object and end effector
            new_task_relev_obj_rot_goal = (
                rot_transf
                @ R.from_quat(orig_goal_cfg.task_relev_obj_quat_xyzw).as_matrix()
            )
            new_ee_rot_goal = (
                rot_transf @ R.from_quat(orig_goal_cfg.robot_ee_quat_xyzw).as_matrix()
            )

            # apply pos transform to task relevant object and end effector
            new_task_relev_obj_pos_goal = (
                rot_transf @ orig_goal_cfg.task_relev_obj_pos + pos_transf
            )
            new_ee_pos_goal = rot_transf @ orig_goal_cfg.robot_ee_pos + pos_transf

            new_task_relev_obj_quat_xyzw = R.from_matrix(
                new_task_relev_obj_rot_goal
            ).as_quat()
            new_task_relev_obj_quat_wxyz = np.array(
                [
                    new_task_relev_obj_quat_xyzw[3],
                    new_task_relev_obj_quat_xyzw[0],
                    new_task_relev_obj_quat_xyzw[1],
                    new_task_relev_obj_quat_xyzw[2],
                ]
            )
            new_start_cfg = RobotEnvConfig(
                robot_joint_qpos=new_robot_qpos_start["robot_joint_qpos"].cpu().numpy(),
                robot_gripper_qpos=np.array(
                    [0.04, 0.04]
                ),  # new_robot_qpos_start["robot_gripper_qpos"],
                robot_base_pos=base_pos.copy(),
                robot_base_quat_wxyz=base_quat_wxyz.copy(),
                task_relev_obj_pos=new_task_relev_obj_pos_goal,
                task_relev_obj_quat_wxyz=new_task_relev_obj_quat_wxyz,
                task_relev_obj_rot=new_task_relev_obj_rot_goal,
                task_relev_obj_pos_transf=pos_transf,
                task_relev_obj_rot_transf=rot_transf,
            )
            # should probably also set the task_relev_obj parameters to None

            # the main thing to store was just the position and rotation transforms right?
            # in that case, an option is to store the transform and lazily compute the new values?
            new_goal_cfg = RobotEnvConfig(
                robot_joint_qpos=None,
                robot_gripper_qpos=orig_goal_cfg.robot_gripper_qpos,
                robot_ee_pos=new_ee_pos_goal,
                robot_ee_quat_wxyz=None,
                robot_ee_rot=new_ee_rot_goal,
                task_relev_obj_pos=pos_transf,  # new_task_relev_obj_pos_goal,
                task_relev_obj_quat_wxyz=None,
                task_relev_obj_rot=rot_transf,  # new_task_relev_obj_rot_goal,
                task_irrelev_obj_pos=None,
                task_irrelev_obj_quat_wxyz=None,
                task_irrelev_obj_rot=None,
            )
            optimal_trajectory = nerf_robomimic_env.get_optimal_trajectory(
                new_start_cfg,
                new_goal_cfg,
                "package://models/assets/task_relevant/mesh-outputs/mesh.sdf",
            )

            if optimal_trajectory is None:
                logging.info(
                    "Did not find an optimal_trajectory, skipping this iteration"
                )
                continue

            obs_lst: List[Dict[str, torch.Tensor]] = (
                nerf_robomimic_env.get_observations(
                    optimal_trajectory["robot_joint_qpos"],
                    optimal_trajectory["robot_gripper_qpos"],
                    task_relev_obj_pose=optimal_trajectory["task_relev_obj_pose"],
                )
            )

            save_dir = pathlib.Path(f"test_se3_randomization_newest_1_testv3/{idx}")
            for i, obs in enumerate(obs_lst):
                for cam in env_cfg.multi_camera_cfg.camera_names:
                    rgb = obs[f"{cam}_rgb"]
                    # create directory if it doesn't exist
                    full_file_path = save_dir / f"{cam}_rgb{i}.png"
                    full_file_path.parent.mkdir(parents=True, exist_ok=True)

                    # save normal image
                    success = cv2.imwrite(
                        str(full_file_path), cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                    )
                    if not success:
                        logging.warning(f"Failed to save image to {full_file_path}")
            logging.info(f"Saved images to {save_dir}")

    if test_transf_future_cfgs:
        for idx in range(10):
            new_robot_qpos_start: Dict[str, torch.Tensor] = (
                nerf_robomimic_env.sample_robot_qpos()
            )
            X_se3 = nerf_robomimic_env.sample_task_relev_obj_se3_transform()
            pos_transf, rot_transf = X_se3[:3, 3], X_se3[:3, :3]

            # apply rot transform to task relevant object and end effector
            new_task_relev_obj_rot_goal = (
                rot_transf
                @ R.from_quat(orig_goal_cfg.task_relev_obj_quat_xyzw).as_matrix()
            )
            new_ee_rot_goal = (
                rot_transf @ R.from_quat(orig_goal_cfg.robot_ee_quat_xyzw).as_matrix()
            )

            # apply pos transform to task relevant object and end effector
            new_task_relev_obj_pos_goal = (
                rot_transf @ orig_goal_cfg.task_relev_obj_pos + pos_transf
            )
            new_ee_pos_goal = rot_transf @ orig_goal_cfg.robot_ee_pos + pos_transf

            new_task_relev_obj_quat_xyzw = R.from_matrix(
                new_task_relev_obj_rot_goal
            ).as_quat()
            new_task_relev_obj_quat_wxyz = np.array(
                [
                    new_task_relev_obj_quat_xyzw[3],
                    new_task_relev_obj_quat_xyzw[0],
                    new_task_relev_obj_quat_xyzw[1],
                    new_task_relev_obj_quat_xyzw[2],
                ]
            )
            new_start_cfg = RobotEnvConfig(
                robot_joint_qpos=new_robot_qpos_start["robot_joint_qpos"].cpu().numpy(),
                robot_gripper_qpos=np.array(
                    [0.04, 0.04]
                ),  # new_robot_qpos_start["robot_gripper_qpos"],
                robot_base_pos=base_pos.copy(),
                robot_base_quat_wxyz=base_quat_wxyz.copy(),
                task_relev_obj_pos=new_task_relev_obj_pos_goal,
                task_relev_obj_quat_wxyz=new_task_relev_obj_quat_wxyz,
                task_relev_obj_rot=new_task_relev_obj_rot_goal,
                task_relev_obj_pos_transf=pos_transf,
                task_relev_obj_rot_transf=rot_transf,
                task_relev_obj_pos_nerf=pos_transf,
                task_relev_obj_rot_nerf=rot_transf,
            )

            # should probably also set the task_relev_obj parameters to None

            # the main thing to store was just the position and rotation transforms right?
            # in that case, an option is to store the transform and lazily compute the new values?
            new_goal_cfg = RobotEnvConfig(
                robot_joint_qpos=None,
                robot_gripper_qpos=orig_goal_cfg.robot_gripper_qpos,
                robot_ee_pos=new_ee_pos_goal,
                robot_ee_quat_wxyz=None,
                robot_ee_rot=new_ee_rot_goal,
                task_relev_obj_pos=pos_transf,  # new_task_relev_obj_pos_goal,
                robot_base_pos=orig_goal_cfg.robot_base_pos,
                robot_base_quat_wxyz=orig_goal_cfg.robot_base_quat_wxyz,
                task_relev_obj_quat_wxyz=None,
                task_relev_obj_rot=rot_transf,  # new_task_relev_obj_rot_goal,
                task_relev_obj_pos_transf=pos_transf,
                task_relev_obj_rot_transf=rot_transf,
                task_relev_obj_pos_nerf=pos_transf,
                task_relev_obj_rot_nerf=rot_transf,
                task_irrelev_obj_pos=None,
                task_irrelev_obj_quat_wxyz=None,
                task_irrelev_obj_rot=None,
            )

            new_future_cfg_list: List[RobotEnvConfig] = []

            for future_cfg in orig_future_cfg_list:
                # apply pos transform to task relevant object and end effector
                new_ee_rot = (
                    rot_transf @ R.from_quat(future_cfg.robot_ee_quat_xyzw).as_matrix()
                )
                new_ee_pos = rot_transf @ future_cfg.robot_ee_pos + pos_transf
                new_ee_quat_xyzw = R.from_matrix(new_ee_rot).as_quat()
                new_ee_quat_wxyz = np.array(
                    [
                        new_ee_quat_xyzw[3],
                        new_ee_quat_xyzw[0],
                        new_ee_quat_xyzw[1],
                        new_ee_quat_xyzw[2],
                    ]
                )

                # optimization: have step at which things are welded
                new_future_cfg_list.append(
                    RobotEnvConfig(
                        robot_gripper_qpos=future_cfg.robot_gripper_qpos,
                        robot_ee_pos=new_ee_pos,
                        robot_ee_quat_wxyz=new_ee_quat_wxyz,
                        robot_ee_rot=new_ee_rot,
                        robot_base_pos=future_cfg.robot_base_pos,
                        robot_base_quat_wxyz=future_cfg.robot_base_quat_wxyz,
                        task_relev_obj_pos=pos_transf,
                        task_relev_obj_rot=rot_transf,  # new_task_relev_obj_rot_goal,
                        task_relev_obj_pos_nerf=pos_transf,
                        task_relev_obj_rot_nerf=rot_transf,
                    )
                )
                logging.warning(
                    "Need to clarify what task_relev_obj_pos/task_relev_obj_pos_nerf actually refer to"
                )

            reach_traj = nerf_robomimic_env.get_optimal_trajectory(
                new_start_cfg,
                new_goal_cfg,
                "package://models/assets/task_relevant/mesh-outputs/mesh.sdf",
            )

            if reach_traj is None:
                logging.info(
                    "Did not find an optimal_trajectory from start to goal, skipping this iteration"
                )
                continue

            reach_traj_task_relev_obj = NeRFObject(
                task_relev_obj_nerf_path,
                bounding_box_min=torch.tensor([-0.07, -0.07, 0.8]),
                bounding_box_max=torch.tensor([0.07, 0.07, 0.86]),
            )

            reach_traj_task_relev_objs = []
            for i in range(len(reach_traj["robot_joint_qpos"])):
                reach_traj_task_relev_objs.append(reach_traj_task_relev_obj)

            future_trajectory = nerf_robomimic_env.get_robot_and_obj_trajectory(
                new_goal_cfg,
                new_future_cfg_list,
                task_relev_obj_paths=[task_relev_obj_mesh_path]
                * len(new_future_cfg_list),
                check_collisions=False,
            )
            # includes the original states

            if future_trajectory is None:
                logging.info(
                    "Did not find an optimal_trajectory from goal to future, skipping this iteration"
                )
                continue

            overall_trajectory: Dict[str, List[torch.Tensor]] = {}
            overall_trajectory["robot_joint_qpos"] = (
                reach_traj["robot_joint_qpos"]
                + future_trajectory["robot_joint_qpos"][1:]
            )
            overall_trajectory["robot_gripper_qpos"] = (
                reach_traj["robot_gripper_qpos"]
                + future_trajectory["robot_gripper_qpos"][1:]
            )
            overall_trajectory["task_relev_obj_pose"] = (
                reach_traj["task_relev_obj_pose"]
                + future_trajectory["task_relev_obj_pose"][1:]
            )
            overall_trajectory["task_relev_obj"] = (
                reach_traj_task_relev_objs + future_task_relev_objs
            )
            overall_trajectory["robot_ee_pos_gripper_site_world"] = (
                reach_traj["robot_ee_pos_gripper_site_world"]
                + future_trajectory["robot_ee_pos_gripper_site_world"][1:]
            )
            overall_trajectory["robot_ee_rot_gripper_site_world"] = (
                reach_traj["robot_ee_rot_gripper_site_world"]
                + future_trajectory["robot_ee_rot_gripper_site_world"][1:]
            )

            # compute idx at which gripper qpos "closes" i.e. both grippers are below e.g. 0.02
            # gripper_qpos_close_idx = np.where(np.array(overall_trajectory["robot_gripper_qpos"])[:, -1] < 0.02)[0]

            # hardcoded welding idx
            task_relev_obj_weld_idx = gripper_qpos_close_idx = (
                len(overall_trajectory["robot_ee_pos_gripper_site_world"]) - 8
            )

            # compute X_{EE, NERF} = X_{EE}^{-1} X_{NERF} at task_relev_obj_weld_idx
            p_W_EE = overall_trajectory["robot_ee_pos_gripper_site_world"][
                task_relev_obj_weld_idx
            ]
            r_W_EE = overall_trajectory["robot_ee_rot_gripper_site_world"][
                task_relev_obj_weld_idx
            ]
            X_W_EE = torch.eye(4)
            X_W_EE[:3, :3] = r_W_EE
            X_W_EE[:3, 3] = p_W_EE
            X_W_NERF = overall_trajectory["task_relev_obj_pose"][
                task_relev_obj_weld_idx
            ]
            X_EE_NERF = torch.inverse(X_W_EE) @ X_W_NERF
            # get updated task_relev_obj_pose X_{W, NERF} = X_{W, EE} X_{EE, NERF}

            # apply to get X_W_NERF as sanity check
            # test_W_NERF = X_W_EE @ X_EE_NERF
            # print("test_W_NERF, should be I", test_W_NERF)
            # TODO(klin): below should be computed once user decides on the object-EE welding timestep
            for i in range(
                task_relev_obj_weld_idx + 1,
                len(overall_trajectory["task_relev_obj_pose"]),
            ):
                p_W_EE = overall_trajectory["robot_ee_pos_gripper_site_world"][i]
                r_W_EE = overall_trajectory["robot_ee_rot_gripper_site_world"][i]
                X_W_EE = torch.eye(4)
                X_W_EE[:3, :3] = r_W_EE
                X_W_EE[:3, 3] = p_W_EE

                overall_trajectory["task_relev_obj_pose"][i] = X_W_EE @ X_EE_NERF

            obs_lst: List[Dict[str, torch.Tensor]] = (
                nerf_robomimic_env.get_observations(
                    overall_trajectory["robot_joint_qpos"],
                    overall_trajectory["robot_gripper_qpos"],
                    task_relev_obj_pose=overall_trajectory["task_relev_obj_pose"],
                    task_relev_obj=overall_trajectory["task_relev_obj"],
                )
            )
            save_dir = pathlib.Path(
                f"test_transf_future_actions_w_loading_nerfs_welded_nose3/{idx}"
            )
            for i, obs in enumerate(obs_lst):
                for cam in env_cfg.multi_camera_cfg.camera_names:
                    rgb = obs[f"{cam}_image"]
                    # create directory if it doesn't exist
                    full_file_path = save_dir / f"{cam}_image{i}.png"
                    full_file_path.parent.mkdir(parents=True, exist_ok=True)

                    # save normal image
                    success = cv2.imwrite(
                        str(full_file_path), cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                    )
                    if not success:
                        logging.warning(f"Failed to save image to {full_file_path}")
            logging.info(f"Saved images to {save_dir}")
