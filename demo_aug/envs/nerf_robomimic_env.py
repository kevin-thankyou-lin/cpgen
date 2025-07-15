import datetime
import logging
import pathlib
import random
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import gym
import numpy as np
import torch
from gym import spaces
from gym.utils import seeding
from PIL import Image
from scipy.spatial.transform import Rotation as R

import demo_aug
from demo_aug.configs.robot_configs import (
    ROBOT_BASE_FRAME_NAME,
)
from demo_aug.envs.base_env import ActionSpaceType, EnvConfig, MotionPlannerType
from demo_aug.envs.motion_planners.base_motion_planner import RobotEnvConfig

# use gym to be compatible with diffusion_policy
from demo_aug.envs.motion_planners.curobo_planner import CuroboMotionPlanner
from demo_aug.envs.motion_planners.drake_motion_planner import DrakeMotionPlanner
from demo_aug.envs.motion_planners.motion_planning_space import (
    DrakeMotionPlanningSpace,
    FixedGripperWrapper,
    HandMotionPlanningSpace,
    IKType,
    ObjectInitializationInfo,
)
from demo_aug.envs.motion_planners.sampling_based_motion_planner import (
    NearPoseUniformSampler,
    SamplingBasedMotionPlanner,
)
from demo_aug.envs.motion_planners.trajopt_based_motion_planner import (
    TrajoptBasedMotionPlanner,
)
from demo_aug.objects.nerf_object import (
    GSplatObject,
    MeshObject,
    NeRFObject,
    TransformationWrapper,
    TransformType,
)
from demo_aug.objects.robot_object import RobotObject
from demo_aug.utils.composite import alpha_composite
from demo_aug.utils.mathutils import (
    biased_sampling,
    multiply_with_X_transf,
    random_rotation_matrix,
    random_z_rotation,
)
from demo_aug.utils.nerf_obj import SphereNeRF


class NeRFRobomimicEnv(gym.Env):
    """Hybrid NeRF + Robomimic environment.

    Args:
        robot_nerf_weights_path (str): Path to the weights of the first NeRF model.
        task_relev_obj_weights_path (str): Path to the weights of the task relevant object NeRF model.
    """

    def __init__(
        self,
        env_cfg: EnvConfig,
        renderable_objs: Optional[
            Dict[str, Union[NeRFObject, GSplatObject, TransformationWrapper]]
        ] = None,
        aug_cfg: Optional[Any] = None,
    ):
        self.multi_camera_cfg = env_cfg.multi_camera_cfg
        self.env_cfg = env_cfg

        # Initialize NeRF models
        # self.task_irrelev_obj = RobomimicObject()
        self.robot_obj: RobotObject = RobotObject(
            env_cfg.robot_cfg, multi_camera_cfg=self.multi_camera_cfg
        )  # needs env

        if self.env_cfg.motion_planner_cfg.use_drake:
            self.motion_planner = DrakeMotionPlanner(
                self.robot_obj,
                drake_package_path=str(
                    pathlib.Path(demo_aug.__file__).parent / "models/package.xml"
                ),
                input_ee_pose_frame_convention="mujoco",
                view_meshcat=env_cfg.motion_planner_cfg.view_meshcat,
            )
            self.arm_motion_planning_space = None

        if (
            self.env_cfg.motion_planner_cfg.motion_planner_type
            == MotionPlannerType.CUROBO
        ):
            # using two different curobo motion planners because has a bug where world reset doesn't work
            # if first world config doesn't include the same number of meshes as the next world config
            self.curobo_motion_planner = None
            self.robot_traj_curobo_motion_planner = self.curobo_motion_planner
        elif (
            self.env_cfg.motion_planner_cfg.motion_planner_type == MotionPlannerType.PRM
        ):
            self.robot_traj_curobo_motion_planner = None

        self.task_relev_obj = NeRFObject(
            env_cfg.task_relev_obj_cfg.config_path,
            env_cfg.task_relev_obj_cfg.bounding_box_min,
            env_cfg.task_relev_obj_cfg.bounding_box_max,
        )

        self.renderable_objs = renderable_objs
        self._curr_renderable_objs = None
        self.aug_cfg = aug_cfg

        # TODO(klin): more general way to handle the different nerfs at different timesteps
        # alternatively create more envs with different nerfs

        # Define action and observation spaces
        if self.env_cfg.action_space_type == ActionSpaceType.POS_DELTA:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,))
        elif (
            self.env_cfg.action_space_type
            == ActionSpaceType.POS_DELTA_AND_AXIS_ANGLE_DELTA
        ):
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(6,)
            )  # [xyz, scaled axis-angle] delta
        else:
            raise NotImplementedError

        self.observation_space = spaces.Dict()
        self.observation_space["rgb"] = spaces.Box(low=0, high=255, shape=(3, 84, 84))

        # Initialize pose and centroid variables
        self.robot_pose: torch.Tensor
        self.target_pose: torch.Tensor

        # representing robot state at this level of abstraction since overwriting underlying kinematics obj.
        self._joint_qpos: torch.Tensor  # need to be set at initialization
        self._gripper_qpos: torch.Tensor  # need to be set at initialization
        self._eef_pos: torch.Tensor  # need to be set at initialization
        self._eef_quat_wxyz: torch.Tensor  # need to be set at initialization

        self.rendered_output_names = ["rgb", "depth", "accumulation"]

        self.camera_names = self.multi_camera_cfg.camera_names
        self.camera_intrinsics: Dict[str, torch.Tensor] = {}
        self.camera_extrinsics: Dict[str, torch.Tensor] = {}

        if self.robot_obj.env.name == "PickPlaceCan":
            self.env_background_xml = "package://models/assets/arenas/bins_arena.xml"
        elif self.robot_obj.env.name in ["NutAssemblySquare", "Lift"]:
            self.env_background_xml = "package://models/assets/arenas/table_arena.xml"
        else:
            raise NotImplementedError(
                f"Task irrelevant obj url not implemented for environment name: {self.robot_obj.env.name}"
            )

        if self.env_cfg.robot_cfg.robot_model_type == "real_FR3_robotiq":
            self.env_background_xml = "package://models/assets/arenas/empty_arena.xml"

        # self.seed()

    def update_task_relev_obj_nerf(
        self,
        task_relev_obj_config_path: str,
        task_relev_obj_cfg_bounding_box_min: torch.Tensor,
        task_relev_obj_cfg_bounding_box_max: torch.Tensor,
    ) -> None:
        self.task_relev_obj = NeRFObject(
            config_path=task_relev_obj_config_path,
            bounding_box_min=task_relev_obj_cfg_bounding_box_min,
            bounding_box_max=task_relev_obj_cfg_bounding_box_max,
        )

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, float, bool, Dict]:
        robot_joint_qpos, robot_gripper_qpos, cartesian_pose = self._apply_action(
            action
        )
        cam_obs = self.get_observation(
            robot_joint_qpos,
            robot_gripper_qpos,
            task_relev_obj=self._curr_renderable_objs,
            task_relev_obj_pose=torch.eye(4),
        )
        # TODO: figure out if these values are actually needed for eval policy
        cam_extrinsics = {
            "27432424_left": [
                0.10374486633397735,
                -0.27059578645833954,
                0.3968124441689313,
                -2.253711031156438,
                -0.03190250154154195,
                -0.9417288730855438,
            ],
            "27432424_right": [
                0.17468779009665633,
                -0.3647690648833747,
                0.40367867772328875,
                -2.2673846494397165,
                -0.025963438904754277,
                -0.9536602167727584,
            ],
            "12391924_left": np.array(
                [0.28603076, -0.03267323, 0.481997, 2.78326823, 0.00344975, 1.56497219]
            ),
            "12391924_right": np.array(
                [0.28593346, 0.02840582, 0.48135208, 2.78294618, 0.00756925, 1.56855078]
            ),
            "12391924_left_gripper_offset": [
                -0.07496436728692035,
                0.03375652826239056,
                0.01352048433972477,
                -0.34256567561092544,
                0.021615754419753275,
                -1.5874660014129223,
            ],
            "12391924_right_gripper_offset": [
                -0.07644216443928634,
                -0.02730158707694055,
                0.012627235466440005,
                -0.3429787912920421,
                0.0175529883724852,
                -1.5911095137360665,
            ],
        }

        cam_intrinsics = {
            "27432424_left": np.array(
                [
                    [523.86346436, 0.0, 645.31665039],
                    [0.0, 523.86346436, 365.47906494],
                    [0.0, 0.0, 1.0],
                ]
            ),
            "27432424_right": np.array(
                [
                    [523.86346436, 0.0, 645.31665039],
                    [0.0, 523.86346436, 365.47906494],
                    [0.0, 0.0, 1.0],
                ]
            ),
            "12391924_left": np.array(
                [
                    [731.47088623, 0.0, 646.26635742],
                    [0.0, 731.47088623, 355.99679565],
                    [0.0, 0.0, 1.0],
                ]
            ),
            "12391924_right": np.array(
                [
                    [731.47088623, 0.0, 646.26635742],
                    [0.0, 731.47088623, 355.99679565],
                    [0.0, 0.0, 1.0],
                ]
            ),
        }

        # remove cam_obs with keys containing "acc" or "c2w"
        cam_obs = {
            k: v for k, v in cam_obs.items() if "acc" not in k and "c2w" not in k
        }

        obs = {
            "image": cam_obs,
            "camera_type": {"12391924": 0, "27432424": 1},
            "robot_state": {
                "cartesian_position": cartesian_pose,
                "gripper_position": robot_gripper_qpos.cpu().numpy(),
                "joint_positions": robot_joint_qpos.cpu().numpy(),
            },
            "camera_extrinsics": cam_extrinsics,
            "camera_intrinsics": cam_intrinsics,
        }
        reward = self._get_reward()
        done = self._is_done()
        info = self._get_info()
        return obs, reward, done, info

    def _get_reward(self):
        return 0.0

    def _is_done(self):
        return False

    def _get_info(self):
        return {}

    def reset(
        self,
        seed: int = 0,
        robot_state: Optional[Dict[str, torch.Tensor]] = None,
        objs_transf_params_seq: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
        use_ik_for_qpos_update: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], Optional[Dict]]:
        self.seed(seed)

        # HACKY for kinematics only env policy eval'ing
        if robot_state is None:
            robot_state: Dict[str, torch.Tensor] = self.sample_robot_qpos(
                sample_near_default_qpos=True,
                near_qpos_scaling=0.06,
                sample_near_eef_pose=False,
            )

            robot_state["robot_gripper_qpos"][:] = (
                0.0  # hardcode to open to be in dist. state
            )
            self._joint_qpos = robot_state["robot_joint_qpos"]
            self._gripper_qpos = robot_state["robot_gripper_qpos"][
                0
            ]  # 1D; hardcoded for robotiq gripper for now
            self._eef_pos = robot_state["robot_ee_pos"]
            self._eef_quat_wxyz = robot_state["robot_ee_quat_wxyz"]
            self._eef_quat_xyzw = np.roll(self._eef_quat_wxyz, -1)
            self._eef_quat_euler_xyz = R.from_quat(self._eef_quat_xyzw).as_euler("xyz")

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

            if self.arm_motion_planning_space is None:
                self.arm_motion_planning_space: DrakeMotionPlanningSpace = DrakeMotionPlanningSpace(
                    drake_package_path=str(
                        pathlib.Path(demo_aug.__file__).parent / "models/package.xml"
                    ),
                    task_irrelev_obj_url=self.env_background_xml,
                    obj_to_init_info=None,
                    name_to_frame_info=self.env_cfg.robot_cfg.frame_name_to_frame_info.default().as_dict(),
                    robot_base_pos=self.robot_obj.base_pos,
                    gripper_type=(
                        "panda_hand"
                        if self.env_cfg.robot_cfg.robot_model_type
                        == "sim_panda_arm_hand"
                        else "robotiq85"
                    ),
                    robot_base_quat_wxyz=self.robot_obj.base_quat_wxyz,
                    view_meshcat=self.env_cfg.motion_planner_cfg.view_meshcat,
                    env_dist_factor=self.env_cfg.motion_planner_cfg.sampling_based_motion_planner_cfg.env_dist_factor,
                    edge_step_size=self.env_cfg.motion_planner_cfg.sampling_based_motion_planner_cfg.edge_step_size,
                    env_collision_padding=self.env_cfg.motion_planner_cfg.sampling_based_motion_planner_cfg.env_collision_padding,
                    input_config_frame="ee_obs_frame",
                )

                # self.curobo_motion_planner = CuroboMotionPlanner(
                #     self.arm_motion_planning_space,
                #     self.env_cfg.robot_cfg.curobo_robot_file_path,
                #     self.env_cfg.robot_cfg.curobo_robot_ee_link,
                # )

            # robot_joint_qpos = self.curobo_motion_planner.get_robot_trajectory(
            #     eef_quat_wxyz_pos, regularization_joint_config=panda_retract_config
            # )[0]
            # hardcode gripper to open
            robot_state["robot_gripper_qpos"][:] = 0
            self._joint_qpos = robot_state["robot_joint_qpos"]
            self._gripper_qpos = robot_state["robot_gripper_qpos"][
                0
            ]  # 1D; hardcoded for robotiq gripper for now
            self._eef_pos = robot_state["robot_ee_pos"]
            self._eef_quat_wxyz = robot_state["robot_ee_quat_wxyz"]
            self._eef_quat_xyzw = np.roll(self._eef_quat_wxyz, -1)
            self._eef_quat_euler_xyz = R.from_quat(self._eef_quat_xyzw).as_euler("xyz")

            # TODO: try using drake for this IK
            X_ee_goal = np.eye(4)
            X_ee_goal[:3, :3] = R.from_quat(self._eef_quat_xyzw).as_matrix()
            X_ee_goal[:3, 3] = self._eef_pos

            robot_joint_qpos = self.arm_motion_planning_space.inverse_kinematics(
                X_ee_goal,
                q_grip=[0, 0, 0, 0, 0, 0],
                q_init=panda_retract_config,
                ik_type=IKType.X_EE_TO_Q_ROBOT,
                n_trials=3,
                min_dist_thresh=0.000,
            )[0][:-1]
            robot_joint_qpos = torch.tensor(robot_joint_qpos)

        else:
            robot_joint_qpos = robot_state["robot_joint_qpos"]
            self._joint_qpos = robot_state["robot_joint_qpos"]
            self._gripper_qpos = robot_state["robot_gripper_qpos"][0]
            self._eef_pos = robot_state["robot_ee_pos"]
            self._eef_quat_wxyz = robot_state["robot_ee_quat_wxyz"]
            self._eef_quat_xyzw = np.roll(self._eef_quat_wxyz, -1)
            self._eef_quat_euler_xyz = R.from_quat(self._eef_quat_xyzw).as_euler("xyz")

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

            if self.arm_motion_planning_space is None:
                self.arm_motion_planning_space: DrakeMotionPlanningSpace = DrakeMotionPlanningSpace(
                    drake_package_path=str(
                        pathlib.Path(demo_aug.__file__).parent / "models/package.xml"
                    ),
                    task_irrelev_obj_url=self.env_background_xml,
                    obj_to_init_info=None,
                    name_to_frame_info=self.env_cfg.robot_cfg.frame_name_to_frame_info.default().as_dict(),
                    robot_base_pos=self.robot_obj.base_pos,
                    gripper_type=(
                        "panda_hand"
                        if self.env_cfg.robot_cfg.robot_model_type
                        == "sim_panda_arm_hand"
                        else "robotiq85"
                    ),
                    robot_base_quat_wxyz=self.robot_obj.base_quat_wxyz,
                    view_meshcat=self.env_cfg.motion_planner_cfg.view_meshcat,
                    env_dist_factor=self.env_cfg.motion_planner_cfg.sampling_based_motion_planner_cfg.env_dist_factor,
                    edge_step_size=self.env_cfg.motion_planner_cfg.sampling_based_motion_planner_cfg.edge_step_size,
                    env_collision_padding=self.env_cfg.motion_planner_cfg.sampling_based_motion_planner_cfg.env_collision_padding,
                    input_config_frame="ee_obs_frame",
                )

            if use_ik_for_qpos_update:
                # may have updated eef pose s.t. it's not consistent with qpos
                X_ee_goal = np.eye(4)
                X_ee_goal[:3, :3] = R.from_quat(self._eef_quat_xyzw).as_matrix()
                X_ee_goal[:3, 3] = self._eef_pos
                qpos = self.arm_motion_planning_space.inverse_kinematics(
                    X_ee_goal,
                    q_grip=[0, 0, 0, 0, 0, 0],
                    q_init=panda_retract_config,
                    ik_type=IKType.X_EE_TO_Q_ROBOT,
                    n_trials=3,
                    min_dist_thresh=0.000,
                )[0]

                # self.robot_obj.forward_kinematics(qpos[:-1], body_name="camera_mount")
                self._joint_qpos = torch.tensor(qpos[:-1])
                robot_state["robot_joint_qpos"] = self._joint_qpos
                robot_joint_qpos = self._joint_qpos

        # define a closure function
        def transf_closure(
            transf_mat: torch.Tensor,
        ) -> Callable[[torch.Tensor], torch.Tensor]:
            def transf(x: torch.Tensor) -> torch.Tensor:
                return multiply_with_X_transf(torch.linalg.inv(transf_mat), x)

            return transf

        self._curr_renderable_objs = {}
        for name, c_obj in self.renderable_objs.items():
            transform_param = objs_transf_params_seq[name].get("1:X_SE3")
            if transform_param is None:
                transform_param = objs_transf_params_seq[name]["0:X_SE3"]

            transform_param_t = torch.tensor(
                transform_param, dtype=torch.float32, device=torch.device("cuda")
            )

            self._curr_renderable_objs[name] = TransformationWrapper(
                c_obj,
                transf_closure(transform_param_t),
                transform_type=TransformType.SE3,
                transform_params={
                    "X_SE3": transform_param.copy(),
                    "X_SE3_origin": np.eye(4),
                },
                transform_name="obj_transf",
            )

        # TODO: figure out why there's the discrepancy between the joint qpos at reset versus step ...
        cartesian_pose = np.concatenate([self._eef_pos, self._eef_quat_euler_xyz])
        cam_obs = self.get_observation(
            # robot_state["robot_joint_qpos"],
            robot_joint_qpos,
            robot_state["robot_gripper_qpos"],
            task_relev_obj=self._curr_renderable_objs,
            task_relev_obj_pose=torch.eye(4),
        )

        # TODO: figure out if these values are actually needed for eval policy
        cam_extrinsics = {
            "27432424_left": [
                0.10374486633397735,
                -0.27059578645833954,
                0.3968124441689313,
                -2.253711031156438,
                -0.03190250154154195,
                -0.9417288730855438,
            ],
            "27432424_right": [
                0.17468779009665633,
                -0.3647690648833747,
                0.40367867772328875,
                -2.2673846494397165,
                -0.025963438904754277,
                -0.9536602167727584,
            ],
            "12391924_left": np.array(
                [0.28603076, -0.03267323, 0.481997, 2.78326823, 0.00344975, 1.56497219]
            ),
            "12391924_right": np.array(
                [0.28593346, 0.02840582, 0.48135208, 2.78294618, 0.00756925, 1.56855078]
            ),
            "12391924_left_gripper_offset": [
                -0.07496436728692035,
                0.03375652826239056,
                0.01352048433972477,
                -0.34256567561092544,
                0.021615754419753275,
                -1.5874660014129223,
            ],
            "12391924_right_gripper_offset": [
                -0.07644216443928634,
                -0.02730158707694055,
                0.012627235466440005,
                -0.3429787912920421,
                0.0175529883724852,
                -1.5911095137360665,
            ],
        }

        cam_intrinsics = {
            "27432424_left": np.array(
                [
                    [523.86346436, 0.0, 645.31665039],
                    [0.0, 523.86346436, 365.47906494],
                    [0.0, 0.0, 1.0],
                ]
            ),
            "27432424_right": np.array(
                [
                    [523.86346436, 0.0, 645.31665039],
                    [0.0, 523.86346436, 365.47906494],
                    [0.0, 0.0, 1.0],
                ]
            ),
            "12391924_left": np.array(
                [
                    [731.47088623, 0.0, 646.26635742],
                    [0.0, 731.47088623, 355.99679565],
                    [0.0, 0.0, 1.0],
                ]
            ),
            "12391924_right": np.array(
                [
                    [731.47088623, 0.0, 646.26635742],
                    [0.0, 731.47088623, 355.99679565],
                    [0.0, 0.0, 1.0],
                ]
            ),
        }

        # remove cam_obs with keys containing "acc" or "c2w"
        cam_obs = {
            k: v for k, v in cam_obs.items() if "acc" not in k and "c2w" not in k
        }

        obs = {
            "image": cam_obs,
            "camera_type": {"12391924": 0, "27432424": 1},
            "robot_state": {
                "cartesian_position": cartesian_pose,
                "gripper_position": self._gripper_qpos.cpu().numpy(),
                "joint_positions": self._joint_qpos.cpu().numpy(),
            },
            "camera_extrinsics": cam_extrinsics,
            "camera_intrinsics": cam_intrinsics,
        }

        # get robot joint qpos and gripper qpos
        return obs, None

    def reset_to(self, reset_dict: Dict[str, torch.Tensor]) -> None:
        """Reset the environment to a given state.

        For a more complete reset_dict, ideally also include CameraConfig and EnvConfig.
        """
        raise NotImplementedError

    def render(self, mode: str = "rgb_array") -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def _update_camera_extrinsics(self, robot_joint_qpos: torch.Tensor) -> None:
        # handling nerf_object rendering differently from robot_object
        # since camera extrinsics implicitly encoded by setting robot to correct
        # joint configuration by mujoco at the moment
        for cam_name in self.camera_names:
            self.camera_extrinsics[cam_name] = self.robot_obj.get_camera_extrinsics(
                cam_name, robot_joint_qpos
            )

    def get_observation(
        self,
        robot_joint_qpos: torch.Tensor,
        robot_gripper_qpos: torch.Tensor,
        task_relev_obj_pose: Optional[torch.Tensor],
        task_irrelev_obj_pose: Optional[torch.Tensor] = None,
        task_relev_obj: Optional[Dict[str, NeRFObject]] = None,
        task_relev_obj_mesh: Optional[Dict[str, MeshObject]] = None,
        save_obs: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        TODO(klin): Note that these poses are more pose transforms for non-nerf (mujoco frames).

        Update camera poses based on robot_joint_qpos.
        """
        result: Dict[str, torch.Tensor] = {}
        # camera poses also depend on robot pose!
        self._update_camera_extrinsics(robot_joint_qpos)

        fx = [
            self.multi_camera_cfg.get_cam_cfg(cam_name).fx
            for cam_name in self.camera_names
        ]
        fy = [
            self.multi_camera_cfg.get_cam_cfg(cam_name).fy
            for cam_name in self.camera_names
        ]
        cx = [
            self.multi_camera_cfg.get_cam_cfg(cam_name).cx
            for cam_name in self.camera_names
        ]
        cy = [
            self.multi_camera_cfg.get_cam_cfg(cam_name).cy
            for cam_name in self.camera_names
        ]
        height = [
            self.multi_camera_cfg.get_cam_cfg(cam_name).height
            for cam_name in self.camera_names
        ]
        width = [
            self.multi_camera_cfg.get_cam_cfg(cam_name).width
            for cam_name in self.camera_names
        ]
        camera_names = self.camera_names
        camera_type = [
            self.multi_camera_cfg.get_cam_cfg(cam_name).camera_type
            for cam_name in self.camera_names
        ]
        cam_extrinsics = [
            self.camera_extrinsics[cam_name] for cam_name in self.camera_names
        ]

        obj_name_to_cam_name_to_task_relevant_obs: Dict[str, Dict[str, NeRFObject]] = {}
        for obj_name, obj in task_relev_obj.items():
            if not self.env_cfg.obs_cfg.use_mujoco_renderer_only:
                obj_name_to_cam_name_to_task_relevant_obs[obj_name] = obj.render(
                    [np.eye(4) for _ in range(len(self.camera_names))],
                    cam_extrinsics,
                    fx,
                    fy,
                    cx,
                    cy,
                    height,
                    width,
                    camera_type,
                    camera_names,
                )
                if False:
                    import cv2
                    import matplotlib.pyplot as plt

                    output_dir = pathlib.Path(
                        f"aaa_from_transform_wrapper/{obj.obj_type}/"
                    )
                    output_dir.mkdir(parents=True, exist_ok=True)
                    # Save the rendered images
                    for key, image in obj_name_to_cam_name_to_task_relevant_obs[
                        obj_name
                    ]["agentview"].items():
                        # Convert the image to PIL format
                        image = image.clone().detach()
                        if key == "rgb":
                            image *= 255
                            image = cv2.cvtColor(
                                image.cpu().numpy().astype(np.uint8), cv2.COLOR_RGB2BGR
                            )
                        elif key == "depth":
                            # print image min max
                            print(
                                f"{obj.obj_type} depth min: {image.min()}, depth max: {image.max()}"
                            )
                            image = image.cpu().numpy()[..., 0]
                            # render depth image with actual values
                            if True:
                                min_val = image.min()
                                max_val = image.max()
                                min_val = 0.74
                                max_val = 0.86
                                from matplotlib import colors as mcolors

                                # Set up normalization based on the actual min and max values of the image
                                norm = mcolors.Normalize(vmin=min_val, vmax=max_val)

                                # Choose a colormap
                                cmap = plt.get_cmap(
                                    "viridis"
                                )  # This can be changed to any other colormap

                                # Apply the colormap with the actual value normalization
                                color_coded_image = cmap(norm(image))

                                # Save the color-coded image
                                plt.imsave(
                                    f"{output_dir}/{obj.obj_type}_color_coded_image_actual_values.png",
                                    color_coded_image,
                                    cmap=cmap,
                                )

                                # Optionally display the image with a color bar that reflects the actual values
                                fig, ax = plt.subplots()
                                cax = ax.imshow(color_coded_image, cmap=cmap, norm=norm)
                                fig.colorbar(
                                    cax, ax=ax
                                )  # To see the color bar with actual values
                                plt.savefig(
                                    f"{output_dir}/{obj.obj_type}_color_coded_image_actual_values_with_colorbar.png"
                                )

                            image = cv2.normalize(
                                image, None, 0, 255, cv2.NORM_MINMAX
                            ).astype("uint8")
                        else:
                            image = image.cpu().numpy()[..., 0]
                            image = cv2.normalize(
                                image, None, 0, 255, cv2.NORM_MINMAX
                            ).astype("uint8")
                        time = f"{datetime.datetime.now()}"
                        file_path = output_dir / f"{key}_{time}.png"
                        cv2.imwrite(str(file_path), image)
                        print(f"saved to {file_path}")

                if False:
                    import cv2

                    # height = 512
                    # width = 512
                    # fx = 618.0386719675123
                    # fy = 618.0386719675123
                    # cx = 256.0
                    # cy = 256.0
                    cam_name = "agentview"
                    single_height = self.multi_camera_cfg.get_cam_cfg(cam_name).height
                    single_width = self.multi_camera_cfg.get_cam_cfg(cam_name).width
                    fx = self.multi_camera_cfg.get_cam_cfg(cam_name).fx
                    fy = self.multi_camera_cfg.get_cam_cfg(cam_name).fy
                    cx = self.multi_camera_cfg.get_cam_cfg(cam_name).cx
                    cy = self.multi_camera_cfg.get_cam_cfg(cam_name).cy
                    c2w = self.camera_extrinsics[cam_name]
                    # c2w = torch.tensor(
                    #     [
                    #         [-0.9932743906974792, -0.09902417659759521, 0.060000598430633545, 0.03497029095888138],
                    #         [0.1157836765050888, -0.8494995832443237, 0.5147275328636169, 0.30000001192092896],
                    #         [1.4901161193847656e-08, 0.5182128548622131, 0.855251669883728, 1.3284684419631958],
                    #         [0.0, 0.0, 0.0, 1.0],
                    #     ]
                    # )
                    from nerfstudio.cameras.cameras import CameraType

                    nerf_obj = obj._obj._obj._obj
                    outputs = nerf_obj.render(
                        torch.eye(4),
                        c2w=c2w[:3],
                        fx=fx,
                        fy=fy,
                        cx=cx,
                        cy=cy,
                        height=single_height,
                        width=single_width,
                        camera_type=CameraType.PERSPECTIVE,
                        upsample_then_downsample=False,
                    )
                    output_dir = pathlib.Path("aaa_images_0.2_in_env_from_nerfobject")
                    output_dir.mkdir(parents=True, exist_ok=True)
                    # Save the rendered images
                    for key, image in outputs.items():
                        # Convert the image to PIL format
                        if key == "rgb":
                            image *= 255
                            image = cv2.cvtColor(
                                image.cpu().numpy().astype(np.uint8), cv2.COLOR_RGB2BGR
                            )
                        elif key == "depth":
                            image = image.cpu().numpy()[..., 0]
                            image = cv2.normalize(
                                image, None, 0, 255, cv2.NORM_MINMAX
                            ).astype("uint8")
                        else:
                            image = image.cpu().numpy()[..., 0]
                            image = cv2.normalize(
                                image, None, 0, 255, cv2.NORM_MINMAX
                            ).astype("uint8")

                        file_path = output_dir / f"{key}.png"
                        cv2.imwrite(str(file_path), image)
                        print(f"saved to {file_path}")

        cam_name_to_robot_obs = self.robot_obj.get_observation(
            robot_joint_qpos,
            robot_gripper_qpos,
            cube_pos_transf=task_relev_obj_pose[:3, 3],
            cube_rot_transf=task_relev_obj_pose[:3, :3],
            camera_names=camera_names,
            device=torch.device("cuda"),  # list(task_relevant_obs.values())[0].device,
            heights=height,
            widths=width,
            task_relev_obj=task_relev_obj,
            task_relev_obj_mesh=task_relev_obj_mesh,
            render_task_relev_obj=self.env_cfg.obs_cfg.use_mujoco_renderer_only,
        )

        if save_obs:
            time = f"{datetime.datetime.now()}"
            time = time.replace(" ", "-")

        for cam_name in self.camera_names:
            # if task_relev_obj_mesh is not None and self.env_cfg.obs_cfg.use_mujoco_renderer_only:
            if self.env_cfg.obs_cfg.use_mujoco_renderer_only:
                # TODO(klin): need correctly update env camera_names to actually render images
                cam_result = {
                    f"{cam_name}_image": (
                        (
                            cam_name_to_robot_obs[cam_name]["rgb"]
                            .detach()
                            .cpu()
                            .numpy()
                            * 255
                        )
                        .astype(np.uint8)
                        .copy()
                    ),
                    f"{cam_name}_c2w": self.camera_extrinsics[cam_name]
                    .detach()
                    .cpu()
                    .numpy()
                    .copy(),
                }
            else:
                # TODO(klin): plot robot/task_relevant obs individually
                rgb_task_relev_obs_lst = [
                    cam_name_to_task_relevant_obs[cam_name]["rgb"]
                    for obj_name, cam_name_to_task_relevant_obs in obj_name_to_cam_name_to_task_relevant_obs.items()
                ]
                disp_task_relev_obs_lst = [
                    1 / cam_name_to_task_relevant_obs[cam_name]["depth"].squeeze()
                    for obj_name, cam_name_to_task_relevant_obs in obj_name_to_cam_name_to_task_relevant_obs.items()
                ]
                acc_task_relev_obs_lst = [
                    cam_name_to_task_relevant_obs[cam_name]["accumulation"].squeeze()
                    for obj_name, cam_name_to_task_relevant_obs in obj_name_to_cam_name_to_task_relevant_obs.items()
                ]
                rgb_list = torch.stack(
                    [cam_name_to_robot_obs[cam_name]["rgb"], *rgb_task_relev_obs_lst]
                )  # , task_irrelev_obs["rgb"].squeeze()])
                disp_list = torch.stack(
                    [
                        1 / cam_name_to_robot_obs[cam_name]["depth"].squeeze(),
                        *disp_task_relev_obs_lst,
                    ]
                )
                acc_list = torch.stack(
                    [
                        cam_name_to_robot_obs[cam_name]["accumulation"].squeeze(),
                        *acc_task_relev_obs_lst,
                    ]
                )

                # white bg = False is hardcoded for the particular rendering configs for nerfacto
                # see default arguments in nerf_object.py's eval_setup()
                obs, alph = alpha_composite(
                    rgb_list, disp_list, acc_list, white_background=False
                )

                # TODO(klin): get the correct camera_cfg's fx/y values
                cam_result = {
                    f"{cam_name}_image": (obs.detach().cpu().numpy() * 255)
                    .astype(np.uint8)
                    .copy(),
                    f"{cam_name}_acc": alph.detach().cpu().numpy().copy(),
                    f"{cam_name}_c2w": self.camera_extrinsics[cam_name]
                    .detach()
                    .cpu()
                    .numpy()
                    .copy(),
                    # f"{cam_name}_cam_fl_x": self.multi_camera_cfg.get_cam_cfg(cam_name).fx,
                    # f"{cam_name}_cam_fl_y": self.multi_camera_cfg.get_cam_cfg(cam_name).fy,
                    # f"{cam_name}_cam_cx": self.multi_camera_cfg.get_cam_cfg(cam_name).cx,
                    # f"{cam_name}_cam_cy": self.multi_camera_cfg.get_cam_cfg(cam_name).cy
                }
            # TODO(klin) might be camera params being invalid?
            if save_obs:
                time = f"{datetime.datetime.now()}"
                output_dir = "debug_im_square_peg_1"
                pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
                output_dir = f"{output_dir}/{time}"
                pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

                for (
                    obj_name,
                    cam_name_to_task_relevant_obs,
                ) in obj_name_to_cam_name_to_task_relevant_obs.items():
                    # Convert the RGB image
                    img_rgb = Image.fromarray(
                        (
                            cam_name_to_task_relevant_obs[cam_name]["rgb"]
                            .detach()
                            .cpu()
                            .numpy()
                            * 255
                        ).astype(np.uint8)
                    )

                    # Assuming the distance and acc observations are grayscale images. Convert them accordingly.
                    # If not, you need to modify this part according to the format of these observations.
                    # For depth, invert values and normalize to 0-255, then clip to ensure they are within the range
                    # distance_array = \
                    # (1 / cam_name_to_task_relevant_obs[cam_name]["depth"].detach().cpu().numpy() * 255)
                    distance_array = (
                        cam_name_to_task_relevant_obs[cam_name]["depth"]
                        .detach()
                        .cpu()
                        .numpy()
                        * 255
                    )
                    distance_array_clipped = np.clip(distance_array, 0, 255).astype(
                        np.uint8
                    )

                    img_distance = Image.fromarray(distance_array_clipped.squeeze(-1))

                    # For acc, normalize to 0-255, then clip to ensure they are within the range
                    acc_array = (
                        cam_name_to_task_relevant_obs[cam_name]["accumulation"]
                        .detach()
                        .cpu()
                        .numpy()
                        * 255
                    )
                    acc_array_clipped = np.clip(acc_array, 0, 255).astype(np.uint8)
                    img_acc = Image.fromarray(acc_array_clipped.squeeze(-1))

                    # Concatenate images side by side
                    concatenated_img = Image.new(
                        "RGB",
                        (
                            img_rgb.width + img_distance.width + img_acc.width,
                            img_rgb.height,
                        ),
                    )
                    concatenated_img.paste(img_rgb, (0, 0))
                    concatenated_img.paste(img_distance, (img_rgb.width, 0))
                    concatenated_img.paste(
                        img_acc, (img_rgb.width + img_distance.width, 0)
                    )

                    # Save the concatenated image
                    concatenated_img.save(
                        f"{output_dir}/{obj_name}_{cam_name}_concatenated_image.png"
                    )

                # create folder if not exist using pathlib
                pathlib.Path(f"{output_dir}/rgb_imgs").mkdir(
                    parents=True, exist_ok=True
                )

                all_images = []  # to store all images for concatenation

                for (
                    obj_name,
                    cam_name_to_task_relevant_obs,
                ) in obj_name_to_cam_name_to_task_relevant_obs.items():
                    # save cam_name_to_task_relevant_obs[cam_name]["rgb"] images to "rgb_imgs" folder
                    img = Image.fromarray(
                        (
                            cam_name_to_task_relevant_obs[cam_name]["rgb"]
                            .detach()
                            .cpu()
                            .numpy()
                            * 255
                        ).astype(np.uint8)
                    )
                    img.save(f"{output_dir}/rgb_imgs/{obj_name}_{cam_name}_image.png")
                    all_images.append(img)
                    print("saved to rgb_imgs")

                # save acc_imgs
                acc_img = Image.fromarray(cam_result[f"{cam_name}_image"])
                acc_img.save(f"{output_dir}/rgb_imgs/2img_merged_{cam_name}_image.png")
                all_images.append(acc_img)

                # Create a blank image with the total width of all images and height of the tallest image
                total_width = sum(img.width for img in all_images)
                max_height = max(img.height for img in all_images)
                merged_img = Image.new("RGB", (total_width, max_height))

                # Paste each image into merged_img
                x_offset = 0
                for img in all_images:
                    merged_img.paste(img, (x_offset, 0))
                    x_offset += img.width

                merged_img.save(f"{output_dir}/rgb_imgs/merged_image.png")
                print("saved merged image to rgb_imgs")

            result.update(cam_result)
        return result

    def seed(self, seed: int = None) -> List[int]:
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _apply_action(self, action: torch.Tensor) -> None:
        # action = array of end effector + gripper command and do IK to get joint angle
        eef_pos, eef_euler_xyz = action[:3], action[3:6]
        eef_quat_xyzw = R.from_euler("xyz", eef_euler_xyz).as_quat()

        gripper_action = action[6:]

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

        if self.arm_motion_planning_space is None:
            self.arm_motion_planning_space: DrakeMotionPlanningSpace = DrakeMotionPlanningSpace(
                drake_package_path=str(
                    pathlib.Path(demo_aug.__file__).parent / "models/package.xml"
                ),
                task_irrelev_obj_url=self.env_background_xml,
                obj_to_init_info=None,
                name_to_frame_info=self.env_cfg.robot_cfg.frame_name_to_frame_info.default().as_dict(),
                robot_base_pos=self.robot_obj.base_pos,
                gripper_type=(
                    "panda_hand"
                    if self.env_cfg.robot_cfg.robot_model_type == "sim_panda_arm_hand"
                    else "robotiq85"
                ),
                robot_base_quat_wxyz=self.robot_obj.base_quat_wxyz,
                view_meshcat=self.env_cfg.motion_planner_cfg.view_meshcat,
                env_dist_factor=self.env_cfg.motion_planner_cfg.sampling_based_motion_planner_cfg.env_dist_factor,
                edge_step_size=self.env_cfg.motion_planner_cfg.sampling_based_motion_planner_cfg.edge_step_size,
                env_collision_padding=self.env_cfg.motion_planner_cfg.sampling_based_motion_planner_cfg.env_collision_padding,
                input_config_frame="ee_obs_frame",
            )

            # self.curobo_motion_planner = CuroboMotionPlanner(
            #     self.arm_motion_planning_space,
            #     self.env_cfg.robot_cfg.curobo_robot_file_path,
            #     self.env_cfg.robot_cfg.curobo_robot_ee_link,
            # )

        # robot_joint_qpos = self.curobo_motion_planner.get_robot_trajectory(
        #     eef_quat_wxyz_pos, regularization_joint_config=panda_retract_config
        # )[0]

        # TODO: try using drake for this IK
        X_ee_goal = np.eye(4)
        X_ee_goal[:3, :3] = R.from_quat(eef_quat_xyzw).as_matrix()
        X_ee_goal[:3, 3] = eef_pos
        robot_joint_qpos = self.arm_motion_planning_space.inverse_kinematics(
            X_ee_goal,
            q_grip=[0, 0, 0, 0, 0, 0],
            q_init=panda_retract_config,
            ik_type=IKType.X_EE_TO_Q_ROBOT,
            n_trials=3,
            min_dist_thresh=0.000,
        )[0][:-1]
        robot_joint_qpos = torch.tensor(robot_joint_qpos)

        # hardcode gripper action
        current_qpos = self._gripper_qpos
        self._gripper_qpos = robot_gripper_qpos = current_qpos + gripper_action * 0.1
        torch.clip(robot_gripper_qpos, 0.0, 0.8, out=robot_gripper_qpos)

        cartesian_pose = action[:6]
        # not object level dynamics for now
        return robot_joint_qpos, robot_gripper_qpos, cartesian_pose

    def _load_nerf_model(
        self, weights_path: pathlib.Path, rgb: Optional[torch.Tensor] = None
    ) -> SphereNeRF:
        # Load NeRF model weights
        if weights_path is None:
            return SphereNeRF(rgb=rgb)
        else:
            return NeRFObject(weights_path, rgb=rgb)

    def optimal_action(self) -> torch.Tensor:
        """Compute optimal action by moving in the direction of the target"""
        raise NotImplementedError

    def get_observations(
        self,
        robot_joint_qpos: List[torch.Tensor],
        robot_gripper_qpos: List[torch.Tensor],
        task_relev_obj_pose: List[torch.Tensor],
        task_relev_obj: Optional[List[Dict[str, NeRFObject]]] = None,
        task_relev_obj_mesh: Optional[List[Dict[str, MeshObject]]] = None,
        task_irrelev_obj_pose: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Update camera poses based on robot_joint_qpos.
        Get observations based on robot, task relevant and irrelevant poses.

        render_task_relev_obj_mesh: bool. If true (and in sim, which is True for now),
            render from robot_obj
        """
        if self.env_cfg.obs_cfg.use_mujoco_renderer_only:
            assert task_relev_obj_mesh is not None, (
                "Must provide task_relev_obj_mesh if using "
                "mujoco renderer only and wanting to render transformed task relev obj"
            )
        obs_lst: List[Dict[str, np.ndarray]] = []
        for i in range(len(robot_joint_qpos)):
            print(f"Observation {i}")
            obs = self.get_observation(
                robot_joint_qpos[i],
                robot_gripper_qpos[i],
                task_relev_obj_pose[i]
                if task_relev_obj_pose is not None
                else torch.eye(4),
                task_irrelev_obj_pose[i]
                if task_irrelev_obj_pose is not None
                else torch.eye(4),
                task_relev_obj[i] if task_relev_obj is not None else None,
                task_relev_obj_mesh[i] if task_relev_obj_mesh is not None else None,
            )
            obs_lst.append(obs)
        return obs_lst

    def get_optimal_trajectory(
        self,
        start_cfg: RobotEnvConfig,
        goal_cfg: RobotEnvConfig,
        default_robot_joint_qpos: Optional[List] = None,
        default_robot_goal_joint_pos: Optional[List] = None,
        enforce_reach_traj_gripper_non_close: bool = False,
        use_collision_free_waypoint_heuristic: bool = False,
        collision_free_waypoint_threshold: float = 0.06,
        collision_free_waypoint_rotation_angle_bound: float = np.pi / 4,
        collision_free_waypoint_sampling_radius: float = 0.015,
        start_at_collision_free_waypoint: bool = False,
        obj_to_init_info: Optional[Dict[str, ObjectInitializationInfo]] = None,
        original_gripper_command: Optional[Literal[-1, 1]] = -1,
        visualize_trajectory: bool = False,
        truncate_last_n_steps: Optional[int] = None,
    ) -> Dict[str, List]:
        """
        Get optimal trajectory in generalized coordinates from start to goal.

        Re gripper action:
            If there is gripper action, as determined by the original start and end gripper actions,
            we do a pre-motion step and a post-motion step involving only the gripper.

        start_cfg: generalized coordinates for robot (joint and gripper angles for a robot manipulator)
        goal_cfg: can a constraint for some frames e.g. end effector pose or even positions of gripper tips.
        """
        # -1 means  open gripper, 1 means close gripper
        if original_gripper_command == -1:
            # TODO(klin): investigate this if statement --- seems never used
            # take a pre-motion step to open the gripper by fixing all else except gripper
            # q1_arm = np.concatenate([start_cfg.robot_joint_qpos, start_cfg.robot_gripper_qpos])
            # path = [q1_arm]
            # q1_arm_planning = np.concatenate([start_cfg.robot_joint_qpos, q_gripper_planning])
            if self.env_cfg.robot_cfg.robot_model_type == "sim_panda_arm_hand":
                q_gripper_planning = np.array([0.04, 0.04])
            elif self.env_cfg.robot_cfg.robot_model_type == "real_FR3_robotiq":
                q_gripper_planning = np.array([0.0])
        else:
            raise ValueError("original_gripper_command must be -1 or 1")

        # create the relevant motion planning spaces
        arm_motion_planning_space: DrakeMotionPlanningSpace = DrakeMotionPlanningSpace(
            drake_package_path=str(
                pathlib.Path(demo_aug.__file__).parent / "models/package.xml"
            ),
            task_irrelev_obj_url=self.env_background_xml,
            obj_to_init_info=obj_to_init_info,
            name_to_frame_info=self.env_cfg.robot_cfg.frame_name_to_frame_info.default().as_dict(),
            robot_base_pos=self.robot_obj.base_pos,
            gripper_type=(
                "panda_hand"
                if self.env_cfg.robot_cfg.robot_model_type == "sim_panda_arm_hand"
                else "robotiq85"
            ),
            robot_base_quat_wxyz=self.robot_obj.base_quat_wxyz,
            view_meshcat=self.env_cfg.motion_planner_cfg.view_meshcat,
            env_dist_factor=self.env_cfg.motion_planner_cfg.sampling_based_motion_planner_cfg.env_dist_factor,
            edge_step_size=self.env_cfg.motion_planner_cfg.sampling_based_motion_planner_cfg.edge_step_size,
            env_collision_padding=self.env_cfg.motion_planner_cfg.sampling_based_motion_planner_cfg.env_collision_padding,
        )
        arm_motion_planning_space: FixedGripperWrapper = FixedGripperWrapper(
            arm_motion_planning_space, q_gripper_planning
        )

        hand_bounds = self.env_cfg.motion_planner_cfg.sampling_based_motion_planner_cfg.hand_motion_planning_space_bounds
        hand_bounds = np.array(hand_bounds[0]), np.array(hand_bounds[1])
        hand_free_gripper_motion_planning_space: HandMotionPlanningSpace = HandMotionPlanningSpace(
            name_to_frame_info=self.env_cfg.robot_cfg.frame_name_to_frame_info.default().as_dict(),
            drake_package_path=str(
                pathlib.Path(demo_aug.__file__).parent / "models/package.xml"
            ),
            task_irrelev_obj_url=self.env_background_xml,
            obj_to_init_info=obj_to_init_info,
            robot_base_pos=None,
            robot_base_quat_wxyz=None,
            gripper_type=(
                "panda_hand"
                if self.env_cfg.robot_cfg.robot_model_type == "sim_panda_arm_hand"
                else "robotiq85"
            ),
            # TODO: better passing of parameters here ... should directly specify hand type in robot_cfg
            view_meshcat=self.env_cfg.motion_planner_cfg.view_meshcat,
            bounds=(hand_bounds[0], hand_bounds[1]),
            input_config_frame="ee_obs_frame",
            env_dist_factor=self.env_cfg.motion_planner_cfg.sampling_based_motion_planner_cfg.env_dist_factor,
            edge_step_size=self.env_cfg.motion_planner_cfg.sampling_based_motion_planner_cfg.edge_step_size,
            env_collision_padding=self.env_cfg.motion_planner_cfg.sampling_based_motion_planner_cfg.env_collision_padding,
        )
        hand_motion_planning_space: FixedGripperWrapper = FixedGripperWrapper(
            hand_free_gripper_motion_planning_space, q_gripper_planning
        )

        # check start/goal hand-gripper positions are collision free
        q_hand_gripper_1 = np.concatenate(
            [start_cfg.robot_ee_quat_wxyz, start_cfg.robot_ee_pos, q_gripper_planning]
        )
        q_hand_gripper_2 = np.concatenate(
            [goal_cfg.robot_ee_quat_wxyz, goal_cfg.robot_ee_pos, q_gripper_planning]
        )

        test_fixed_start: bool = False
        if test_fixed_start:
            # TODO(remove) zk: testing case where start will collide if no padding!
            q_hand_gripper_1 = np.concatenate(
                [
                    goal_cfg.robot_ee_quat_wxyz,
                    np.array(
                        [
                            goal_cfg.robot_ee_pos[0] - 0.08,
                            goal_cfg.robot_ee_pos[1] - 0.08,
                            goal_cfg.robot_ee_pos[2],
                        ]
                    ),
                    q_gripper_planning,
                ]
            )

        # for now hardcode pose of gripper2 = gripper1
        # q_hand_gripper_2 = q_hand_gripper_1.copy()

        # check q_hand_grippers are collision free
        for i, q_hand_gripper in enumerate([q_hand_gripper_1, q_hand_gripper_2]):
            cur_pose_name: str = "start" if i == 0 else "end"
            # check if q_hand_gripper is in the bounds of the hand_motion_planning_space
            if not hand_motion_planning_space.is_in_bounds(q_hand_gripper):
                # print bounds and q_hand_gripper
                if cur_pose_name == "start":
                    q_hand_gripper[4:7] = np.clip(
                        q_hand_gripper[4:7],
                        hand_motion_planning_space.bounds[0][4:7],
                        hand_motion_planning_space.bounds[1][4:7],
                    )
                    logging.info(
                        f"{cur_pose_name} hand + gripper pose is not in bounds of hand_motion_planning_space;could be"
                        " better to sample from within bounds, would need more work since transforms are w.r.t. object"
                    )
                    print(
                        "manually adjusted position of hand + gripper pose to be in bounds"
                    )
                    if not hand_motion_planning_space.is_in_bounds(q_hand_gripper):
                        print(
                            "manually adjusted position of hand + gripper pose is still not in bounds"
                        )
                        return None
                else:
                    print(f"{cur_pose_name} hand + gripper pose: {q_hand_gripper}")
                    logging.info(
                        f"{cur_pose_name} hand + gripper pose is not in bounds of hand_motion_planning_space"
                    )
                    print(
                        f"{cur_pose_name} hand + gripper pose is not in bounds of hand_motion_planning_space"
                    )
                    return None

            is_collision = hand_motion_planning_space.is_collision(
                q_hand_gripper, input_config_frame="ee_obs_frame", viz=True
            )
            if is_collision:
                print(
                    f"{cur_pose_name} hand + gripper pose causes collision with environment; impossible to reach"
                    " without collisions"
                )
                return None
            # TODO: think the hand's orieintation here is correct?
            # just incorrect when it's going to get optimal traj and then we viz in drake?
            # OK so the end ose frame is pointing down -- weird; end pose frame should be similar to the start pose frame?? seems like a frame wrong
            X_ee = np.eye(4)
            X_ee[:3, 3] = q_hand_gripper[4:7]
            X_ee[:3, :3] = R.from_quat(
                np.roll(q_hand_gripper[0:4], shift=-1)
            ).as_matrix()

        is_full_arm_trajectory: bool = False
        path = q_arm_gripper = []
        if self.env_cfg.motion_planner_cfg.motion_planner_type in [
            MotionPlannerType.KINEMATIC_TRAJECTORY_OPTIMIZATION,
            MotionPlannerType.LINEAR_INTERPOLATION,
        ]:
            motion_planner = TrajoptBasedMotionPlanner()
            q_hand_gripper = motion_planner.get_optimal_trajectory(
                q_hand_gripper_1,
                q_hand_gripper_2,
                motion_planning_space=hand_motion_planning_space,
                debug=True,
                input_config_frame="panda_hand_obs_frame_robomimic",
                num_trajopt_guesses=10,
            )

            if q_hand_gripper is None:
                return None

        elif self.env_cfg.motion_planner_cfg.motion_planner_type in [
            MotionPlannerType.KINEMATIC_TRAJECTORY_OPTIMIZATION,
            MotionPlannerType.LINEAR_INTERPOLATION,
            # TODO(klin): add params to use original drake traj opt
        ]:
            # prev
            return self.motion_planner.get_optimal_trajectory(
                start_cfg,
                goal_cfg,
                motion_planner_type=self.env_cfg.motion_planner_cfg.motion_planner_type,
                obj_to_init_info=obj_to_init_info,
                default_robot_joint_qpos=default_robot_joint_qpos,
                default_robot_goal_joint_pos=default_robot_goal_joint_pos,
                enforce_reach_traj_gripper_non_close=enforce_reach_traj_gripper_non_close,
                use_collision_free_waypoint_heuristic=use_collision_free_waypoint_heuristic,
                collision_free_waypoint_threshold=collision_free_waypoint_threshold,
                collision_free_waypoint_rotation_angle_bound=collision_free_waypoint_rotation_angle_bound,
                collision_free_waypoint_sampling_radius=collision_free_waypoint_sampling_radius,
                start_at_collision_free_waypoint=start_at_collision_free_waypoint,
            )
            # additionally figure out task relevant object poses???
            # return results
        elif (
            self.env_cfg.motion_planner_cfg.motion_planner_type == MotionPlannerType.PRM
        ):
            motion_planner = SamplingBasedMotionPlanner(arm_motion_planning_space)

            numpy_random = np.random.RandomState(42)
            init_samples = motion_planner.get_init_samples(
                q_hand_gripper_1,
                q_hand_gripper_2,
                hand_motion_planning_space,
                n_samples=100,
                sample_along_eef_interp=True,
                check_collisions=False,
                sample_along_sampled_eef_interp=True,
                sample_near_goal=False,
                sample_top_down=True,
            )

            Q_near_goal: List[np.ndarray] = motion_planner.get_init_samples(
                q_hand_gripper_1,
                q_hand_gripper_2,
                hand_motion_planning_space,
                n_samples=100,
                check_collisions=False,
                sample_near_goal=True,
            )

            # toc = time.time()

            init_samples.extend(Q_near_goal)
            init_samples = np.array(init_samples)

            # original samples
            n_original_samples = init_samples.shape[0]
            lb, up = hand_motion_planning_space.bounds
            in_bounds = np.all((init_samples >= lb) & (init_samples <= up), axis=1)
            init_samples = init_samples[in_bounds]
            n_post_filtering_samples = init_samples.shape[0]
            print(
                f"Filtered out {n_original_samples - n_post_filtering_samples} / {n_original_samples} samples"
            )
            init_samples = init_samples.tolist()

            # randomly shuffle samples
            random.shuffle(init_samples)

            min_values = hand_motion_planning_space.bounds[0]
            max_values = hand_motion_planning_space.bounds[1]

            # replace any np.infs with 1
            min_values[min_values == -np.inf] = -1
            max_values[max_values == np.inf] = 1

            sampler = NearPoseUniformSampler(
                bias=0.2,
                normalize_quat=True,
                quat_start_idx=3,
                quat_end_idx=7,
                start_conf=q_hand_gripper_1,
                goal_conf=q_hand_gripper_2,
                numpy_random=numpy_random,
                min_values=hand_motion_planning_space.bounds[0],
                max_values=hand_motion_planning_space.bounds[1],
                init_samples=init_samples,
                use_1D_gripper=False,
            )

            q_hand_gripper, q_hand_gripper_cost = motion_planner.get_optimal_trajectory(
                q_hand_gripper_1,
                q_hand_gripper_2,
                motion_planning_space=hand_motion_planning_space,
                sampler=sampler,
                debug=False,
                Q_near_goal=Q_near_goal,
                timeout=self.env_cfg.motion_planner_cfg.sampling_based_motion_planner_cfg.timeout,
            )

            if q_hand_gripper is None:
                return None

        if (
            self.env_cfg.motion_planner_cfg.motion_planner_type
            == MotionPlannerType.CUROBO
        ):
            if self.curobo_motion_planner is None:
                self.curobo_motion_planner = CuroboMotionPlanner(
                    arm_motion_planning_space,
                    self.env_cfg.robot_cfg.curobo_robot_file_path,
                    self.env_cfg.robot_cfg.curobo_robot_ee_link,
                )
            else:
                self.curobo_motion_planner.update_env(arm_motion_planning_space)

            is_full_arm_trajectory = True
            # TODO: Check if there already exists a full arm start/goal config
            # Set up full arm start config - redundant if the problem passed one in
            X_ee_start = np.eye(4)
            X_ee_start[:3, :3] = R.from_quat(
                np.roll(q_hand_gripper_1[0:4], shift=-1)
            ).as_matrix()
            X_ee_start[:3, 3] = q_hand_gripper_1[4:7]

            eef_quat_wxyz_pos = np.concatenate(
                [q_hand_gripper_1[0:4], q_hand_gripper_1[4:7]]
            )[np.newaxis, :]

            panda_retract_config = np.array(
                [
                    0,
                    np.pi / 16.0,
                    0.00,
                    -np.pi / 2.0 - np.pi / 3.0,
                    0.00,
                    np.pi - 0.2,
                    np.pi / 4,
                    0.04,
                    0.04,
                ]
            )
            q_start = self.curobo_motion_planner.get_robot_trajectory(
                eef_quat_wxyz_pos, regularization_joint_config=panda_retract_config
            )
            if q_start is None:
                print("Failed to get IK solution for start")
                return None
            q_start = np.array(q_start)[0]
            q_start = np.concatenate([q_start, q_hand_gripper_1[7:]])

            # Set up full arm goal config - redundant if the problem passed one in
            X_ee_goal = np.eye(4)
            X_ee_goal[:3, :3] = R.from_quat(
                np.roll(q_hand_gripper_2[0:4], shift=-1)
            ).as_matrix()
            X_ee_goal[:3, 3] = q_hand_gripper_2[4:7]

            goal_type: Literal["pose", "joint"] = (
                "pose"  # current curobo pull doesn't succeed with js planning
            )
            if goal_type == "pose":
                q_goal = q_hand_gripper_2
            elif goal_type == "joint":
                q_goal = (
                    arm_motion_planning_space.inverse_kinematics(
                        X_ee_goal,
                        q_grip=q_hand_gripper_2[7:],
                        q_init=None,
                        ik_type=IKType.X_EE_TO_Q_ROBOT,
                        n_trials=3,
                        min_dist_thresh=0.000,
                    )[0]
                    if goal_cfg.robot_joint_qpos is None
                    else np.concatenate(
                        [goal_cfg.robot_joint_qpos, goal_cfg.robot_gripper_qpos]
                    )
                )

            q_arm_gripper, q_hand_gripper_cost = (
                self.curobo_motion_planner.get_optimal_trajectory(
                    q_start,
                    q_goal,
                    goal_type=goal_type,
                    visualize=visualize_trajectory,
                    gripper_dim=len(start_cfg.robot_gripper_qpos),
                )
            )

            # manually set q_hand_gripper_cost to be angular distance between start and goal
            def angular_distance(R1: np.ndarray, R2: np.ndarray):
                # Compute the relative rotation matrix
                R_relative = np.dot(np.linalg.inv(R1), R2)
                angle = np.arccos((np.trace(R_relative) - 1) / 2)
                return angle

            q_hand_gripper_cost = angular_distance(
                X_ee_start[:3, :3], X_ee_goal[:3, :3]
            )

            if q_arm_gripper is None:
                print("Failed to find trajectory between start and goal")
                logging.info("Failed to find trajectory between start and goal")
                # import ipdb

                # ipdb.set_trace()
                return None

            gripper_dim = len(start_cfg.robot_gripper_qpos)

            # assuming q_hand_gripper_1 and q_hand_gripper_2 have the same gripper and that we fix the gripper
            q_arm_gripper = np.concatenate(
                [
                    q_arm_gripper,
                    np.tile(
                        q_hand_gripper_1[-gripper_dim:][np.newaxis, :],
                        (len(q_arm_gripper), 1),
                    ),
                ],
                axis=-1,
            )

            if self.env_cfg.robot_cfg.robot_model_type == "sim_panda_arm_hand":
                # Negate the last element for each entry of the q_arm_gripper list
                # discrepancy between drake and robomimic gripper directions
                for i in range(len(q_arm_gripper)):
                    q_arm_gripper[i][-1] = -q_arm_gripper[i][-1]

        # TODO(klin): refactor all motion planners' get_optimal_trajectory
        # to return things in joint space as the API abstraction
        if not is_full_arm_trajectory:
            if q_hand_gripper is None:
                return None

            path = q_arm_gripper = []
            for i, q_val in enumerate(q_hand_gripper):
                if i == 0:
                    q_init = None
                else:
                    q_init = q_arm_gripper[-1]
                X_ee = np.eye(4)
                X_ee[:3, :3] = R.from_quat(np.roll(q_val[0:4], shift=-1)).as_matrix()
                X_ee[:3, 3] = q_val[4:7]

                # get the corresponding robot joint qpos
                q_ik = arm_motion_planning_space.inverse_kinematics(
                    X_ee,
                    q_grip=q_val[7:],
                    q_init=q_init,
                    ik_type=IKType.X_EE_TO_Q_ROBOT,
                    n_trials=3,
                    min_dist_thresh=0.000,
                )[0]

                q_arm_gripper.append(q_ik)
        else:
            path = list(q_arm_gripper)

        if visualize_trajectory:
            import ipdb

            ipdb.set_trace()
            arm_motion_planning_space.visualize_trajectory(q_arm_gripper, sleep=0.2)

        results = {}

        gripper_dim = len(start_cfg.robot_gripper_qpos)

        # TODO(klin) handle gripper command only above: currently manually duplicate first path element except just have it have the starting gripper qpos
        # currently not using realistic gripper dynamics
        path = [
            np.concatenate([path[0][:-gripper_dim], start_cfg.robot_gripper_qpos])
        ] + path

        if np.all(start_cfg.robot_gripper_qpos == goal_cfg.robot_gripper_qpos):
            # set all gripper qpos as goal gripper qpos: assumes equality from only_use_tracking_start_gripper_qpos=T
            for i in range(len(path)):
                path[i][-gripper_dim:] = goal_cfg.robot_gripper_qpos

        use_better_gripper_dynamics = True
        if use_better_gripper_dynamics:
            last_gripper_qpos = path[-1][-gripper_dim:]
            if np.all(start_cfg.robot_gripper_qpos != goal_cfg.robot_gripper_qpos):
                # interpolate between the last gripper qpos in path and the goal gripper qpos
                interpolated_gripper_qpos = np.linspace(
                    last_gripper_qpos, goal_cfg.robot_gripper_qpos, num=3 + 1
                )[1:]
                for gripper_qpos in interpolated_gripper_qpos:
                    path.extend(
                        [np.concatenate([path[-1][:-gripper_dim], gripper_qpos])]
                    )
            elif np.all(start_cfg.robot_gripper_qpos == goal_cfg.robot_gripper_qpos):
                # set all gripper qpos to be the goal gripper qpos
                for i in range(len(path)):
                    path[i][-gripper_dim:] = goal_cfg.robot_gripper_qpos
                # append the goal gripper qpos to the last joint qpos in path
                path.extend(
                    [
                        np.concatenate(
                            [path[-1][:-gripper_dim], goal_cfg.robot_gripper_qpos]
                        )
                    ]
                )
            else:
                print(
                    "Bug: weird case where only one of start and goal gripper qpos are different"
                )
                import ipdb
        else:
            path += [
                np.concatenate([path[-1][:-gripper_dim], goal_cfg.robot_gripper_qpos])
            ]

        if truncate_last_n_steps is not None:
            last_idx = max(0, len(path) - truncate_last_n_steps)
            path = path[:last_idx]

        # TODO(klin): make a method to get this dict
        results["robot_joint_qpos"] = [q[:-gripper_dim] for q in path]
        results["robot_gripper_qpos"] = [q[-gripper_dim:] for q in path]

        results["robot_ee_pos_action_frame_world"] = [
            arm_motion_planning_space.get_end_effector_pose(
                q, frame_name="ee_action_frame"
            )[:3, 3]
            for q in path
        ]
        results["robot_ee_quat_wxyz_action_frame_world"] = [
            np.roll(
                R.from_matrix(
                    arm_motion_planning_space.get_end_effector_pose(
                        q, frame_name="ee_action_frame"
                    )[:3, :3]
                ).as_quat(),
                shift=1,
            )
            for q in path
        ]
        results["robot_ee_rot_action_frame_world"] = [
            arm_motion_planning_space.get_end_effector_pose(
                q, frame_name="ee_action_frame"
            )[:3, :3]
            for q in path
        ]
        results["robot_ee_pos_obs_frame_world"] = [
            arm_motion_planning_space.get_end_effector_pose(
                q, frame_name="ee_obs_frame"
            )[:3, 3]
            for q in path
        ]
        results["robot_ee_quat_wxyz_obs_frame_world"] = [
            np.roll(
                R.from_matrix(
                    arm_motion_planning_space.get_end_effector_pose(
                        q, frame_name="ee_obs_frame"
                    )[:3, :3]
                ).as_quat(),
                shift=1,
            )
            for q in path
        ]
        results["robot_ee_rot_obs_frame_world"] = [
            arm_motion_planning_space.get_end_effector_pose(
                q, frame_name="ee_obs_frame"
            )[:3, :3]
            for q in path
        ]

        results["robot_ee_pos_action_frame_base"] = [
            arm_motion_planning_space.get_poses(
                q,
                frame_names=["ee_action_frame"],
                base_frame_name=ROBOT_BASE_FRAME_NAME,
            )[0][:3, 3]
            for q in path
        ]
        results["robot_ee_quat_wxyz_action_frame_base"] = [
            np.roll(
                R.from_matrix(
                    arm_motion_planning_space.get_poses(
                        q,
                        frame_names=["ee_action_frame"],
                        base_frame_name=ROBOT_BASE_FRAME_NAME,
                    )[0][:3, :3]
                ).as_quat(),
                shift=1,
            )
            for q in path
        ]
        results["robot_ee_rot_action_frame_base"] = [
            arm_motion_planning_space.get_poses(
                q,
                frame_names=["ee_action_frame"],
                base_frame_name=ROBOT_BASE_FRAME_NAME,
            )[0][:3, :3]
            for q in path
        ]
        results["robot_ee_pos_obs_frame_base"] = [
            arm_motion_planning_space.get_poses(
                q, frame_names=["ee_obs_frame"], base_frame_name=ROBOT_BASE_FRAME_NAME
            )[0][:3, 3]
            for q in path
        ]
        results["robot_ee_quat_wxyz_obs_frame_base"] = [
            np.roll(
                R.from_matrix(
                    arm_motion_planning_space.get_poses(
                        q,
                        frame_names=["ee_obs_frame"],
                        base_frame_name=ROBOT_BASE_FRAME_NAME,
                    )[0][:3, :3]
                ).as_quat(),
                shift=1,
            )
            for q in path
        ]
        results["robot_ee_rot_obs_frame_base"] = [
            arm_motion_planning_space.get_poses(
                q, frame_names=["ee_obs_frame"], base_frame_name=ROBOT_BASE_FRAME_NAME
            )[0][:3, :3]
            for q in path
        ]

        # TODO(klin) compute task relevant object poses correctly by passing in the frame for which to get pose
        results["task_relev_obj_pos"] = [torch.tensor([0, 0, 0]) for _ in path]
        results["task_relev_obj_rot"] = [torch.eye(3) for _ in path]
        results["task_relev_obj_pose"] = [torch.eye(4) for _ in path]

        # convert everything to tensors if not already
        for k, v in results.items():
            if not isinstance(v[0], torch.Tensor):
                results[k] = [torch.tensor(v_i) for v_i in v]

        results["overall_cost"] = q_hand_gripper_cost

        return results

    def get_robot_and_obj_trajectory(
        self,
        start_cfg: RobotEnvConfig,
        future_cfg_list: List[RobotEnvConfig],
        check_collisions: bool = True,
        obj_to_init_info: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, List]:
        # Q: is it the best idea to do traj tracking w/ OC?
        # hmm maybe just do IK control? not sure
        if False and self.env_cfg.motion_planner_cfg.use_drake:
            return self.motion_planner.get_robot_and_obj_trajectory(
                self.robot_obj.default_joint_qpos,
                start_cfg,
                future_cfg_list,
                check_collisions,
                obj_to_init_info=obj_to_init_info,
            )
        elif (
            True
            or self.env_cfg.motion_planner_cfg.motion_planner_type
            == MotionPlannerType.CUROBO
        ):
            arm_motion_planning_space: DrakeMotionPlanningSpace = DrakeMotionPlanningSpace(
                drake_package_path=str(
                    pathlib.Path(demo_aug.__file__).parent / "models/package.xml"
                ),
                task_irrelev_obj_url=self.env_background_xml,
                obj_to_init_info=obj_to_init_info,
                name_to_frame_info=self.env_cfg.robot_cfg.frame_name_to_frame_info.default().as_dict(),
                gripper_type=(
                    "panda_hand"
                    if self.env_cfg.robot_cfg.robot_model_type == "sim_panda_arm_hand"
                    else "robotiq85"
                ),
                robot_base_pos=self.robot_obj.base_pos,
                robot_base_quat_wxyz=self.robot_obj.base_quat_wxyz,
                view_meshcat=self.env_cfg.motion_planner_cfg.view_meshcat,
                env_dist_factor=self.env_cfg.motion_planner_cfg.sampling_based_motion_planner_cfg.env_dist_factor,
                edge_step_size=self.env_cfg.motion_planner_cfg.sampling_based_motion_planner_cfg.edge_step_size,
                env_collision_padding=self.env_cfg.motion_planner_cfg.sampling_based_motion_planner_cfg.env_collision_padding,
            )
            if self.robot_traj_curobo_motion_planner is None:
                self.robot_traj_curobo_motion_planner = CuroboMotionPlanner(
                    arm_motion_planning_space,
                    self.env_cfg.robot_cfg.curobo_robot_file_path,
                    self.env_cfg.robot_cfg.curobo_robot_ee_link,
                )
            else:
                self.robot_traj_curobo_motion_planner.update_env(
                    arm_motion_planning_space
                )
            motion_planner = self.robot_traj_curobo_motion_planner
            eef_quat_wxyz_pos = np.array(
                [np.concatenate([start_cfg.robot_ee_quat_wxyz, start_cfg.robot_ee_pos])]
                + [
                    np.concatenate([cfg.robot_ee_quat_wxyz, cfg.robot_ee_pos])
                    for cfg in future_cfg_list
                ]
            )
            q_arm_gripper: List[np.ndarray] = motion_planner.get_robot_trajectory(
                eef_quat_wxyz_pos
            )
            if q_arm_gripper is None:
                print("Failed to find trajectory between start and goal")
                logging.info("Failed to find trajectory between start and goal")
                return None
            # take the gripper qposes from the original start and end cfgs then append to corresponding q_arm_gripper entry
            for i in range(len(q_arm_gripper)):
                if i == 0:
                    q_arm_gripper[i] = np.concatenate(
                        [q_arm_gripper[i], start_cfg.robot_gripper_qpos]
                    )
                else:
                    q_arm_gripper[i] = np.concatenate(
                        [q_arm_gripper[i], future_cfg_list[i - 1].robot_gripper_qpos]
                    )

            path = q_arm_gripper
            gripper_dim = len(start_cfg.robot_gripper_qpos)

            results = {}
            results["robot_joint_qpos"] = [q[:-gripper_dim] for q in path]
            results["robot_gripper_qpos"] = [q[-gripper_dim:] for q in path]
            results["robot_ee_pos_action_frame_world"] = [
                arm_motion_planning_space.get_end_effector_pose(
                    q, frame_name="ee_action_frame"
                )[:3, 3]
                for q in path
            ]
            results["robot_ee_quat_wxyz_action_frame_world"] = [
                np.roll(
                    R.from_matrix(
                        arm_motion_planning_space.get_end_effector_pose(
                            q, frame_name="ee_action_frame"
                        )[:3, :3]
                    ).as_quat(),
                    shift=1,
                )
                for q in path
            ]
            results["robot_ee_rot_action_frame_world"] = [
                arm_motion_planning_space.get_end_effector_pose(
                    q, frame_name="ee_action_frame"
                )[:3, :3]
                for q in path
            ]
            results["robot_ee_pos_obs_frame_world"] = [
                arm_motion_planning_space.get_end_effector_pose(
                    q, frame_name="ee_obs_frame"
                )[:3, 3]
                for q in path
            ]
            results["robot_ee_quat_wxyz_obs_frame_world"] = [
                np.roll(
                    R.from_matrix(
                        arm_motion_planning_space.get_end_effector_pose(
                            q, frame_name="ee_obs_frame"
                        )[:3, :3]
                    ).as_quat(),
                    shift=1,
                )
                for q in path
            ]
            results["robot_ee_rot_obs_frame_world"] = [
                arm_motion_planning_space.get_end_effector_pose(
                    q, frame_name="ee_obs_frame"
                )[:3, :3]
                for q in path
            ]

            results["robot_ee_pos_action_frame_base"] = [
                arm_motion_planning_space.get_poses(
                    q,
                    frame_names=["ee_action_frame"],
                    base_frame_name=ROBOT_BASE_FRAME_NAME,
                )[0][:3, 3]
                for q in path
            ]
            results["robot_ee_quat_wxyz_action_frame_base"] = [
                np.roll(
                    R.from_matrix(
                        arm_motion_planning_space.get_poses(
                            q,
                            frame_names=["ee_action_frame"],
                            base_frame_name=ROBOT_BASE_FRAME_NAME,
                        )[0][:3, :3]
                    ).as_quat(),
                    shift=1,
                )
                for q in path
            ]
            results["robot_ee_rot_action_frame_base"] = [
                arm_motion_planning_space.get_poses(
                    q,
                    frame_names=["ee_action_frame"],
                    base_frame_name=ROBOT_BASE_FRAME_NAME,
                )[0][:3, :3]
                for q in path
            ]
            results["robot_ee_pos_obs_frame_base"] = [
                arm_motion_planning_space.get_poses(
                    q,
                    frame_names=["ee_obs_frame"],
                    base_frame_name=ROBOT_BASE_FRAME_NAME,
                )[0][:3, 3]
                for q in path
            ]
            results["robot_ee_quat_wxyz_obs_frame_base"] = [
                np.roll(
                    R.from_matrix(
                        arm_motion_planning_space.get_poses(
                            q,
                            frame_names=["ee_obs_frame"],
                            base_frame_name=ROBOT_BASE_FRAME_NAME,
                        )[0][:3, :3]
                    ).as_quat(),
                    shift=1,
                )
                for q in path
            ]
            results["robot_ee_rot_obs_frame_base"] = [
                arm_motion_planning_space.get_poses(
                    q,
                    frame_names=["ee_obs_frame"],
                    base_frame_name=ROBOT_BASE_FRAME_NAME,
                )[0][:3, :3]
                for q in path
            ]

            # TODO(klin) compute task relevant object poses correctly by passing in the frame for which to get pose
            results["task_relev_obj_pos"] = [torch.tensor([0, 0, 0]) for _ in path]
            results["task_relev_obj_rot"] = [torch.eye(3) for _ in path]
            results["task_relev_obj_pose"] = [torch.eye(4) for _ in path]

            # convert everything to tensors if not already
            for k, v in results.items():
                if not isinstance(v[0], torch.Tensor):
                    results[k] = [torch.tensor(v_i) for v_i in v]
            return results
        else:
            raise NotImplementedError("Only implemented for drake")

    def get_env_state(self) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def sample_robot_qpos(
        self,
        output_convention: str = "drake",
        sample_near_default_qpos: bool = False,
        near_qpos_scaling: float = 0.1,
        sample_near_eef_pose: bool = False,
        center_eef_pos: Optional[np.ndarray] = None,
        center_eef_quat_xyzw: Optional[np.ndarray] = None,
        sample_pos_x_bound: Optional[Tuple[float, float]] = None,
        sample_pos_y_bound: Optional[Tuple[float, float]] = None,
        sample_pos_z_bound: Optional[Tuple[float, float]] = None,
        sample_min_height: Optional[float] = None,
        sample_rot_angle_z_bound: Optional[Tuple[float, float]] = None,
        sample_rot_angle_y_bound: Optional[Tuple[float, float]] = None,
        sample_rot_angle_x_bound: Optional[Tuple[float, float]] = None,
        forward_kinematics_body_name: Optional[
            str
        ] = None,  # used for returning eef pos too
    ) -> Dict[str, torch.Tensor]:
        """
        Sample a valid configuration for robot.
        Don't consider collisions here.
        """
        return self.robot_obj.sample_valid_joint_qpos(
            output_convention=output_convention,
            sample_near_default_qpos=sample_near_default_qpos,
            near_qpos_scaling=near_qpos_scaling,
            sample_near_eef_pose=sample_near_eef_pose,
            center_eef_pos=center_eef_pos,
            center_eef_quat_xyzw=center_eef_quat_xyzw,
            sample_pos_x_bound=sample_pos_x_bound,
            sample_pos_y_bound=sample_pos_y_bound,
            sample_pos_z_bound=sample_pos_z_bound,
            sample_min_height=sample_min_height,
            sample_rot_angle_z_bound=sample_rot_angle_z_bound,
            sample_rot_angle_y_bound=sample_rot_angle_y_bound,
            sample_rot_angle_x_bound=sample_rot_angle_x_bound,
            forward_kinematics_body_name=forward_kinematics_body_name,
        )

    @staticmethod
    def sample_task_relev_obj_se3_transform(
        dx_range: List[float],
        dy_range: List[float],
        dz_range: List[float],
        dthetaz_range: List[float],
        use_biased_sampling_z_rot: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Only sample from a TODO(klin) user specified or heuristically derived set.

        More generally, user can specify a set of task agnostic constraints and the
        code just samples an element (that's also valid) within that set.

        Currently, code samples a transformation randomly from a box, rather than
        having some target location and computing the transformation to get there.
        """
        X_se3 = np.eye(4)
        dx = biased_sampling(dx_range, alpha=0.75, beta=0.75)
        dy = biased_sampling(dy_range, alpha=0.75, beta=0.75)
        dz = biased_sampling(dz_range, alpha=0.75, beta=0.75)

        pos_transform = np.array([dx, dy, dz])

        # compute/sample new rotation transform
        rotation_transform = random_z_rotation(
            theta_min=dthetaz_range[0],
            theta_max=dthetaz_range[1],
            use_biased_sampling=use_biased_sampling_z_rot,
            alpha=0.75,
            beta=0.75,
        )
        X_se3[:3, :3] = rotation_transform
        X_se3[:3, 3] = pos_transform
        return X_se3

    def sample_robosuite_placement_initializer(
        self,
        x_range: Optional[Tuple[float, float]] = None,
        y_range: Optional[Tuple[float, float]] = None,
        z_range: Optional[Tuple[float, float]] = None,
        obj_name: Optional[str] = "square_nut",
    ) -> np.ndarray:
        """
        Sample a valid transformation for the task relevant object.
        """
        obj_poses = self.robot_obj.sample_object_initial_state()
        if obj_name == "square_nut":
            obj_pos, obj_quat_wxyz, _ = obj_poses["SquareNut"]
            obj_quat_xyzw = np.roll(obj_quat_wxyz, shift=-1)
        elif obj_name == "red_cube":
            obj_pos, obj_quat_wxyz, _ = obj_poses["cube"]
            obj_quat_xyzw = np.roll(obj_quat_wxyz, shift=-1)
        elif obj_name == "can":
            obj_pos, obj_quat_wxyz, _ = obj_poses["Can"]
            obj_quat_xyzw = np.roll(obj_quat_wxyz, shift=-1)
        else:
            raise NotImplementedError(
                f"Object {obj_name} not supported yet for sample robosuite placement initializer"
            )

        X_se3 = np.eye(4)
        X_se3[:3, 3] = obj_pos
        X_se3[:3, :3] = R.from_quat(obj_quat_xyzw).as_matrix()
        # clip the position
        if x_range is not None:
            X_se3[0, 3] = np.clip(X_se3[0, 3], x_range[0], x_range[1])
        if y_range is not None:
            X_se3[1, 3] = np.clip(X_se3[1, 3], y_range[0], y_range[1])
        if z_range is not None:
            X_se3[2, 3] = np.clip(X_se3[2, 3], z_range[0], z_range[1])

        return X_se3

    @staticmethod
    def sample_scale_transform(
        scale_range: List[float],
        X_scale_origin: np.ndarray = np.eye(4),
        apply_non_uniform_scaling: bool = False,
    ) -> np.ndarray:
        scale_val = biased_sampling(scale_range, alpha=0.7, beta=0.7)
        X_scale = np.eye(4)
        if apply_non_uniform_scaling:
            X_scale[0, 0] = scale_val
            X_scale[1, 1] = biased_sampling(scale_range, alpha=0.7, beta=0.7)
            X_scale[2, 2] = biased_sampling(scale_range, alpha=0.7, beta=0.7)
        else:
            X_scale[:3, :3] *= scale_val

        return X_scale, X_scale_origin

    @staticmethod
    def sample_shear_transform(
        shear_range: List[float], X_shear_origin: np.ndarray = np.eye(4)
    ) -> np.ndarray:
        """
        Shear matrix is of the form:
        [[1, a, b, 0],
        [c, 1, d, 0],
        [e, f, 1, 0],
        [0, 0, 0, 1]]

        For some reason, using multiple 'directions' at once causes the rendering of the
        nut at a certain rendering angle (co-inciding w/ a dark cloud 'passing' for the
        region outside the bounding box) to be jagged.

        Using one direction seems to help a bit.
        """
        # Create shear matrix
        X_shear = np.eye(4)

        # Select a random direction for shearing
        random_direction = np.random.choice(["xy", "xz", "yx", "yz", "zx", "zy"])

        if random_direction == "xy":
            X_shear[0, 1] = biased_sampling(shear_range, alpha=0.8, beta=0.8)
        elif random_direction == "xz":
            X_shear[0, 2] = biased_sampling(shear_range, alpha=0.8, beta=0.8)
        elif random_direction == "yx":
            X_shear[1, 0] = biased_sampling(shear_range, alpha=0.8, beta=0.8)
        elif random_direction == "yz":
            X_shear[1, 2] = biased_sampling(shear_range, alpha=0.8, beta=0.8)
        elif random_direction == "zx":
            X_shear[2, 0] = biased_sampling(shear_range, alpha=0.8, beta=0.8)
        elif random_direction == "zy":
            X_shear[2, 1] = biased_sampling(shear_range, alpha=0.8, beta=0.8)
        # currently hardcoding z direction shear to be going 'upwards' (thus assumes a table top setting)
        return X_shear, X_shear_origin

    @staticmethod
    def sample_warp_transform(warp_range: List[float]) -> np.ndarray:
        # return np.random.uniform(warp_range[0], warp_range[1])
        return np.eye(4)

    @staticmethod
    def sample_ee_noise_transform(
        pos_x_min: float = 0,
        pos_x_max: float = 0,
        pos_y_min: float = 0,
        pos_y_max: float = 0,
        pos_z_min: float = 0,
        pos_z_max: float = 0,
        rot_angle_max: float = 0,
    ) -> np.ndarray:
        transf = np.eye(4)
        # sample uniformly from the range
        transf[0, 3] = np.random.uniform(pos_x_min, pos_x_max)
        transf[1, 3] = np.random.uniform(pos_y_min, pos_y_max)
        transf[2, 3] = np.random.uniform(pos_z_min, pos_z_max)
        # sample uniformly from the range
        transf[:3, :3] = random_rotation_matrix(rot_angle_max)
        return transf

    def randomize_eef_task_relev_obj_se3(
        self,
        orig_start_cfg: RobotEnvConfig,
        orig_goal_cfg: RobotEnvConfig,
        orig_future_robot_env_cfgs: List[RobotEnvConfig],
        ignore_future_task_relev_obj: bool = True,
    ) -> RobotEnvConfig:
        """
        Randomize task relevant object and end effector pose in a valid (i.e. collision free) way.

        Returns a new goal RobotEnvConfig that is compatible with original start, goal and future cfgs.

        Must include the end effector pose and task relevant object pose.
        """
        from scipy.spatial.transform import Rotation as R

        x_range = [-0.18, 0.18]
        y_range = [-0.18, 0.18]
        z_range = [0, 0]
        # sample new absolute position for task relevant object
        x = np.random.uniform(x_range[0], x_range[1])
        y = np.random.uniform(y_range[0], y_range[1])
        z = np.random.uniform(z_range[0], z_range[1])

        # compute transform
        delta_x = x - orig_goal_cfg.task_relev_obj_pos[0]
        delta_y = y - orig_goal_cfg.task_relev_obj_pos[1]
        delta_z = z - orig_goal_cfg.task_relev_obj_pos[2]
        pos_transform = np.array([delta_x, delta_y, delta_z])

        # compute/sample new rotation transform
        rotation_transform = random_z_rotation()

        # apply rot transform to task relevant object and end effector
        new_task_relev_obj_rot_goal = rotation_transform @ R.from_quat(
            orig_goal_cfg.task_relev_obj_quat_xyzw
        )
        new_ee_rot_goal = rotation_transform @ R.from_quat(
            orig_goal_cfg.robot_ee_quat_xyzw
        )

        # apply pos transform to task relevant object and end effector
        new_task_relev_obj_pos_goal = (
            rotation_transform @ orig_goal_cfg.task_relev_obj_pos + pos_transform
        )
        new_ee_pos_goal = (
            rotation_transform @ orig_goal_cfg.robot_ee_pos + pos_transform
        )

        # the main thing to store was just the position and rotation transforms right?
        # in that case, an option is to store the transform and lazily compute the new values?
        new_goal_cfg = RobotEnvConfig(
            robot_joint_qpos=None,
            robot_gripper_qpos=orig_goal_cfg.robot_gripper_qpos,
            robot_ee_pos=new_ee_pos_goal,
            robot_ee_quat_wxyz=None,
            robot_ee_rot=new_ee_rot_goal,
            task_relev_obj_pos=new_task_relev_obj_pos_goal,
            task_relev_obj_quat_wxyz=None,
            task_relev_obj_rot=new_task_relev_obj_rot_goal,
            task_irrelev_obj_pos=None,
            task_irrelev_obj_quat_wxyz=None,
            task_irrelev_obj_rot=None,
        )

        trajectories, result = self.motion_planner.is_start_goal_cfgs_valid(
            orig_start_cfg, new_goal_cfg
        )
        if result:
            return new_goal_cfg

        return new_goal_cfg

    def get_eef_ik(
        self,
        X_W_ee_init: np.ndarray,
        q_gripper_init: np.ndarray,
        name_to_frame_info: Dict[str, Dict[str, Any]],
        kp_to_P_goal: Dict[str, np.ndarray],
        check_collisions: bool = True,
    ) -> torch.Tensor:
        """Get the end effector pose using inverse kinematics.

        Args:
            robot_joint_qpos (torch.Tensor): Robot joint qpos.
            robot_gripper_qpos (torch.Tensor): Robot gripper qpos.
            task_relev_obj_pose (torch.Tensor): Task relevant object pose.
            task_irrelev_obj_pose (Optional[torch.Tensor], optional): Task irrelevant object pose. Defaults to None.

        Returns:
            torch.Tensor: End effector pose.
        """
        return self.motion_planner.get_eef_ik(
            X_W_ee_init=X_W_ee_init,
            q_gripper_init=q_gripper_init,
            name_to_frame_info=name_to_frame_info,
            kp_to_P_goal=kp_to_P_goal,
            check_collisions=check_collisions,
        )

    def get_kpts_from_ee_gripper_objs_and_params(
        self, X_ee: np.ndarray, q_gripper: np.ndarray, name_to_kp_params: Dict
    ) -> Dict[str, np.ndarray]:
        """Get keypoints from end effector pose and keypoint parameters.

        TODO(klin): if this function is only for gripper (and not task relevant objects),
        then can take in a list of X_ee and q_gripper and return a list of kpts.

        If function is also for objects, needs more thought but one way to take in a list and have a single
        env is to load all objects in together.

        Args:
            X_ee_lst (List[np.ndarray]): List of end effector poses.
            name_to_kp_params (Dict): Dictionary mapping keypoint name to keypoint parameters.

        Returns:
            List[Dict[str, np.ndarray]]: List of keypoint to position dicts.
        """
        return self.motion_planner.get_kpts_from_ee_gripper_objs_and_params(
            X_ee, q_gripper, name_to_kp_params
        )
