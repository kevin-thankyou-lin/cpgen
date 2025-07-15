"""
Quick access visualization of robot configuration distribution
"""

import pathlib
from typing import Dict

import imageio
import numpy as np
import torch
import tyro
from scipy.spatial.transform import Rotation as R

import demo_aug
from demo_aug.configs.base_config import DemoAugConfig
from demo_aug.envs.motion_planners.motion_planning_space import (
    DrakeMotionPlanningSpace,
    IKType,
)
from demo_aug.objects.robot_object import RobotObject


def main(cfg: DemoAugConfig):
    robot_obj: RobotObject = RobotObject(
        cfg.env_cfg.robot_cfg, multi_camera_cfg=cfg.env_cfg.multi_camera_cfg
    )

    center_eef_pos_sampling = np.array([0.46825551, 0.33483847, 0.15920483])
    center_eef_quat_xyzw_sampling = np.array(
        [0.59590929, 0.44576579, -0.51371556, 0.42694413]
    )
    orig_ee_goal_pos = np.array([0.43016782, 0.3153621, 0.15920483])

    rotx = cfg.aug_cfg.start_aug_cfg.cartesian_space_aug_configs.eef_rot_angle_x_bound
    rotz = cfg.aug_cfg.start_aug_cfg.cartesian_space_aug_configs.eef_rot_angle_z_bound
    roty = cfg.aug_cfg.start_aug_cfg.cartesian_space_aug_configs.eef_rot_angle_y_bound
    save_dir = pathlib.Path(
        f"robot_configuration_distribution/rot_bnd_{rotx}_{roty}_{rotz}"
    )
    # remove spaces, "(" and ")" from the directory name, replace commas with underscores
    save_dir = save_dir.with_name(
        save_dir.name.replace(" ", "").replace("(", "").replace(")", "")
    )
    save_dir = save_dir.with_name(save_dir.name.replace(",", "_"))
    save_dir.mkdir(exist_ok=True, parents=True)

    for i in range(cfg.trials_per_constraint):
        robot_pose_start: Dict[str, torch.Tensor] = robot_obj.sample_valid_joint_qpos(
            sample_near_default_qpos=cfg.aug_cfg.start_aug_cfg.space == "joint",
            near_qpos_scaling=cfg.aug_cfg.start_aug_cfg.joint_space_aug_configs.joint_qpos_noise_magnitude,
            sample_near_eef_pose=cfg.aug_cfg.start_aug_cfg.space == "cartesian",
            center_eef_pos=(
                center_eef_pos_sampling
                if cfg.aug_cfg.start_aug_cfg.cartesian_space_aug_configs.ee_pos is None
                else cfg.aug_cfg.start_aug_cfg.cartesian_space_aug_configs.ee_pos
            ),
            center_eef_quat_xyzw=center_eef_quat_xyzw_sampling,
            sample_pos_x_bound=cfg.aug_cfg.start_aug_cfg.cartesian_space_aug_configs.eef_pos_x_bound,
            sample_pos_y_bound=cfg.aug_cfg.start_aug_cfg.cartesian_space_aug_configs.eef_pos_y_bound,
            sample_pos_z_bound=cfg.aug_cfg.start_aug_cfg.cartesian_space_aug_configs.eef_pos_z_bound,
            sample_rot_angle_z_bound=cfg.aug_cfg.start_aug_cfg.cartesian_space_aug_configs.eef_rot_angle_z_bound,
            sample_rot_angle_y_bound=cfg.aug_cfg.start_aug_cfg.cartesian_space_aug_configs.eef_rot_angle_y_bound,
            sample_rot_angle_x_bound=cfg.aug_cfg.start_aug_cfg.cartesian_space_aug_configs.eef_rot_angle_x_bound,
            # TODO(klin): this min height doesn't make sense if e.g. goal is at insertion point
            sample_min_height=orig_ee_goal_pos[2] - 0.2,
        )
        env_background_xml = str("package://models/assets/arenas/table_arena.xml")
        arm_motion_planning_space: DrakeMotionPlanningSpace = DrakeMotionPlanningSpace(
            drake_package_path=str(
                pathlib.Path(demo_aug.__file__).parent / "models/package.xml"
            ),
            task_irrelev_obj_url=env_background_xml,
            obj_to_init_info=None,
            name_to_frame_info=cfg.env_cfg.robot_cfg.frame_name_to_frame_info.default().as_dict(),
            robot_base_pos=robot_obj.base_pos,
            gripper_type=(
                "panda_hand"
                if cfg.env_cfg.robot_cfg.robot_model_type == "sim_panda_arm_hand"
                else "robotiq85"
            ),
            robot_base_quat_wxyz=robot_obj.base_quat_wxyz,
            view_meshcat=cfg.env_cfg.motion_planner_cfg.view_meshcat,
            env_dist_factor=cfg.env_cfg.motion_planner_cfg.sampling_based_motion_planner_cfg.env_dist_factor,
            edge_step_size=cfg.env_cfg.motion_planner_cfg.sampling_based_motion_planner_cfg.edge_step_size,
            env_collision_padding=cfg.env_cfg.motion_planner_cfg.sampling_based_motion_planner_cfg.env_collision_padding,
            input_config_frame="ee_obs_frame",
        )

        # inverse kinematics to get joint_qpos
        # do some inverse kinematics to get the robot configuration ... okay fine!
        # TODO(klin)

        # TODO: try using drake for this IK
        X_ee_goal = np.eye(4)
        X_ee_goal[:3, :3] = R.from_quat(
            robot_pose_start["robot_ee_quat_xyzw"]
        ).as_matrix()
        X_ee_goal[:3, 3] = robot_pose_start["robot_ee_pos"]
        panda_retract_config = np.array(
            [
                0,
                np.pi / 16.0,
                0.00,
                -np.pi / 2.0 - np.pi / 3.0,
                0.00,
                np.pi - 0.2,
                np.pi / 4,
                0,
                0,
                0,
                0,
                0,
                0,
            ]
        )
        robot_joint_qpos = arm_motion_planning_space.inverse_kinematics(
            X_ee_goal,
            q_grip=[0, 0, 0, 0, 0, 0],
            q_init=panda_retract_config,
            ik_type=IKType.X_EE_TO_Q_ROBOT,
            n_trials=3,
            min_dist_thresh=0.000,
        )[0][:-1]
        robot_joint_qpos = torch.tensor(robot_joint_qpos)

        obs = robot_obj.get_observation(
            robot_joint_qpos,
            robot_pose_start["robot_gripper_qpos"],
            camera_names=cfg.env_cfg.multi_camera_cfg.camera_names,
        )

        for camera_name, cam_obs in obs.items():
            img = (cam_obs["rgb"].detach().cpu().numpy() * 255).astype(np.uint8)
            # convert black background to white
            img[img == 0] = 255
            save_path = save_dir / f"{camera_name}_{i}.png"
            imageio.imwrite(save_path, img)
            print(f"Saved {save_path}")


if __name__ == "__main__":
    # load demonstrations file
    tyro.cli(main)


"""
PYTHONPATH=. python scripts/dataset/robot_configuration_distrbution.py
--cfg.aug-cfg.start-aug-cfg.space cartesian --cfg.aug-cfg.start-aug-cfg.cartesian-space-aug-configs.eef-pos-bound 0.06
--cfg.aug-cfg.start-aug-cfg.cartesian-space-aug-configs.eef-pos-x-bound -0.06 0.06
--cfg.aug-cfg.start-aug-cfg.cartesian-space-aug-configs.eef-pos-y-bound -0.06 0.06
--cfg.aug-cfg.start-aug-cfg.cartesian-space-aug-configs.eef-pos-z-bound -0.01 0.1
--cfg.aug-cfg.start-aug-cfg.cartesian-space-aug-configs.eef-rot-angle-x-bound 40 40
--cfg.aug-cfg.start-aug-cfg.cartesian-space-aug-configs.eef-rot-angle-y-bound -10 10
--cfg.aug-cfg.start-aug-cfg.cartesian-space-aug-configs.eef-rot-angle-z-bound -15 10 fr3-real-env
"""
