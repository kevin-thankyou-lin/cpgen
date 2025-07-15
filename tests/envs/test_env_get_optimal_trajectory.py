import logging

import numpy as np
import torch

from demo_aug.envs.base_env import EnvConfig, MotionPlannerConfig, MotionPlannerType
from demo_aug.envs.motion_planners.base_motion_planner import RobotEnvConfig
from demo_aug.envs.motion_planners.motion_planning_space import ObjectInitializationInfo
from demo_aug.envs.nerf_robomimic_env import NeRFRobomimicEnv
from demo_aug.objects.nerf_object import MeshPaths

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    # set seeds
    np.random.seed(1)
    torch.manual_seed(1)

    env_cfg = EnvConfig(
        motion_planner_cfg=MotionPlannerConfig(
            view_meshcat=True, motion_planner_type=MotionPlannerType.CUROBO
        )
    )
    nerf_robomimic_env = NeRFRobomimicEnv(env_cfg)

    task = "square"
    if task == "lift":
        start_cfg = RobotEnvConfig(
            robot_joint_qpos=None,
            robot_gripper_qpos=np.array([0.028, 0.028]),
            robot_ee_pos=np.array([-0.19563106, -0.09605819, 0.8298026]),
            robot_ee_quat_wxyz=np.array(
                [0.12672813, 0.94468251, -0.26862037, 0.13913314]
            ),
            robot_ee_rot=None,
            robot_base_pos=np.array([-0.56, 0.0, 0.912]),
            robot_base_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            robot_base_rot=None,
            task_relev_obj_pos=np.array([0.0, 0.0, 0.0]),
            task_relev_obj_quat_wxyz=None,
            task_relev_obj_rot=np.array(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            ),
            task_relev_obj_pos_transf=None,
            task_relev_obj_rot_transf=None,
            task_irrelev_obj_pos=None,
            task_irrelev_obj_quat_wxyz=None,
            task_irrelev_obj_rot=None,
            task_relev_obj_pos_nerf=np.array([0.0, 0.0, 0.0]),
            task_relev_obj_rot_nerf=np.array(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            ),
        )
        goal_cfg = RobotEnvConfig(
            robot_joint_qpos=np.array(
                [0.1790, 0.9376, -0.1150, -1.9610, 0.1135, 3.0227, 1.0880]
            ),
            robot_gripper_qpos=np.array([0.02050286, 0.02484637]),
            robot_ee_pos=np.array([0.0310601, 0.01926124, 0.82532003]),
            robot_ee_quat_wxyz=np.array(
                [0.0215382, 0.9861418, -0.14792169, 0.07196961]
            ),
            robot_ee_rot=None,
            robot_base_pos=np.array([-0.56, 0.0, 0.912]),
            robot_base_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            robot_base_rot=None,
            task_relev_obj_pos=np.array([0, 0, 0]),
            task_relev_obj_quat_wxyz=None,
            task_relev_obj_rot=np.array(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            ),
            task_relev_obj_pos_transf=None,
            task_relev_obj_rot_transf=None,
            task_irrelev_obj_pos=None,
            task_irrelev_obj_quat_wxyz=None,
            task_irrelev_obj_rot=None,
            task_relev_obj_pos_nerf=np.array([0, 0, 0]),
            task_relev_obj_rot_nerf=np.array(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            ),
        )

        obj_to_init_info = {
            "red_cube": ObjectInitializationInfo(
                # mesh_path="demo_aug/models/assets/task_relevant/2023-10-31-01-07-29-463673/red_cube_35_2023-10-31-01-07-29-463673red_cube-t50-watertight-mesh-convex-decomp0_transformed-multi.sdf",
                mesh_path="demo_aug/models/assets/task_relevant/2023-10-31-01-07-29-463673/red_cube-t50-watertight-mesh-convex-decomp0_transformed.obj",
                X_parentframe_obj=np.array(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ),
                weld_to_ee=False,
            )
        }

    elif task == "square":
        start_cfg = RobotEnvConfig(
            robot_joint_qpos=None,
            robot_gripper_qpos=np.array(
                [0.013941492885351181, 0.014292004048626633], dtype=np.float32
            ),
            robot_ee_pos=np.array(
                [0.0001158677911791, 0.057435649336980524, 0.8796491848557545],
                dtype=np.float64,
            ),
            # robot_ee_pos=np.array([0.09001158677911791, 0.057435649336980524, 0.8596491848557545], dtype=np.float64),
            robot_ee_quat_wxyz=np.array(
                [
                    0.11801443501081922,
                    0.9804166707126559,
                    -0.06697509715724492,
                    0.14272379366653085,
                ],
                dtype=np.float64,
            ),
            robot_ee_rot=None,
            robot_base_pos=np.array([-0.56, 0.0, 0.912], dtype=np.float64),
            robot_base_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            robot_base_rot=None,
            task_relev_obj_pos=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            task_relev_obj_quat_wxyz=None,
            task_relev_obj_rot=np.array(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
            ),
            task_relev_obj_pos_transf=None,
            task_relev_obj_rot_transf=None,
            task_irrelev_obj_pos=None,
            task_irrelev_obj_quat_wxyz=None,
            task_irrelev_obj_rot=None,
            task_relev_obj_pos_nerf=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            task_relev_obj_rot_nerf=np.array(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
            ),
        )
        goal_cfg = RobotEnvConfig(
            robot_joint_qpos=None,
            robot_gripper_qpos=np.array(
                [0.013941492432275767, 0.014292004048626633], dtype=np.float64
            ),
            robot_ee_pos=np.array(
                [0.14537927008295026, 0.10602959302051795, 0.9794255789249494],
                dtype=np.float64,
            ),
            robot_ee_quat_wxyz=np.array(
                [
                    -0.011502067724405292,
                    0.9985313665378789,
                    -0.010705075558817221,
                    0.05184798776557493,
                ],
                dtype=np.float64,
            ),
            robot_ee_rot=None,
            robot_base_pos=np.array([-0.56, 0.0, 0.912], dtype=np.float64),
            robot_base_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            robot_base_rot=None,
            task_relev_obj_pos=np.array([0, 0, 0], dtype=np.int64),
            task_relev_obj_quat_wxyz=None,
            task_relev_obj_rot=np.array(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
            ),
            task_relev_obj_pos_transf=None,
            task_relev_obj_rot_transf=None,
            task_irrelev_obj_pos=None,
            task_irrelev_obj_quat_wxyz=None,
            task_irrelev_obj_rot=None,
            task_relev_obj_pos_nerf=np.array([0, 0, 0], dtype=np.int64),
            task_relev_obj_rot_nerf=np.array(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
            ),
        )

        obj_to_init_info = {
            "square_peg": ObjectInitializationInfo(
                X_parentframe_obj=np.array(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ),
                mesh_paths=MeshPaths(
                    sdf_path="demo_aug/models/assets/task_relevant/nerf_outputs/nerfacto-data/robomimic/square/newest/5/nerfacto/2024-04-07_192154/meshes/2024-04-08-00-29-13-137721/square_peg_5_2024-04-08-00-29-13-137721square_peg(5,)-watertight-mesh-convex-decomp0_transformed-multi.sdf",
                    obj_path="demo_aug/models/assets/task_relevant/nerf_outputs/nerfacto-data/robomimic/square/newest/5/nerfacto/2024-04-07_192154/meshes/2024-04-08-00-29-13-608838/watertight-mesh_transformed.obj",
                ),
                weld_to_ee=False,
                parent_frame_name=None,
            ),
            "square_nut": ObjectInitializationInfo(
                X_parentframe_obj=np.array(
                    [
                        [0.99987062, -0.00925074, 0.01315916, -0.1571541],
                        [-0.00832725, -0.99761085, -0.06858026, 0.17663628],
                        [0.01376214, 0.0684618, -0.99755881, 0.96291732],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ),
                mesh_paths=MeshPaths(
                    sdf_path="demo_aug/models/assets/task_relevant/nerf_outputs/nerfacto-data/robomimic/square/newest/85/nerfacto/2023-11-07_115443/meshes/2024-04-08-00-29-13-795831/square_nut_85_2024-04-08-00-29-13-795831square_nut(85,)-watertight-mesh-convex-decomp0_transformed-multi.sdf",
                    obj_path="demo_aug/models/assets/task_relevant/nerf_outputs/nerfacto-data/robomimic/square/newest/85/nerfacto/2023-11-07_115443/meshes/2024-04-08-00-29-13-803873/watertight-mesh_transformed.obj",
                ),
                weld_to_ee=True,
                parent_frame_name="ee_obs_frame",
            ),
        }

    optimal_trajectory = nerf_robomimic_env.get_optimal_trajectory(
        start_cfg,
        goal_cfg,
        obj_to_init_info=obj_to_init_info,
        visualize_trajectory=True,  # set this to False to not visualize the trajectory; currently loops forever see the printed port at localhost:<port> for the meshcat server viz
    )
