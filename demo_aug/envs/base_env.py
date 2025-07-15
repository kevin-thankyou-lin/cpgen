import pathlib
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Literal, Optional, Tuple

import gym
import numpy as np
import torch
from gym import spaces
from gym.utils import seeding
from nerfstudio.cameras.camera_utils import viewmatrix

from demo_aug.configs.robot_configs import CameraConfig, MultiCameraConfig, RobotConfig

# use gym to be compatible with diffusion_policy
from demo_aug.utils.nerf_obj import SphereNeRF


@dataclass
class ExpertActionConfig:
    max_dist: float = 0.3  # max distance to move in one step


class NeRFModelType(Enum):
    SPHERE = 0
    CYLINDER = 1
    CYLINDER_WITH_SPHERES = 2
    TRAINED_NERF = 3


@dataclass
class EnvResetConfig:
    randomize_robot_pose: bool = True
    randomize_target_pose: bool = False
    fix_robot_x_plane: bool = True
    fix_target_x_plane: bool = True

    def __post_init__(self):
        if self.fix_robot_x_plane and not self.fix_target_x_plane:
            raise ValueError(
                "If robot x plane is fixed, then target x plane must be fixed as well for data generation done"
                " condition."
            )


class ActionSpaceType(Enum):
    POS_DELTA = 0
    POS_DELTA_AND_AXIS_ANGLE_DELTA = 1
    EE_POS_AND_AXIS_ANGLE_DELTA_WORLD_GRIPPER_QPOS = 2
    EE_POS_AND_AXIS_ANGLE_DELTA_EE_FRAME_GRIPPER_QPOS = 3


class EnvType(Enum):
    SPHERE = auto()
    NERF_3D_TRACE = auto()
    NERF_ROBOMIMIC = (
        auto()
    )  # nerf for task relevant object and mujoco for robot + irrelevant


@dataclass
class ObsConfig:
    render_target: bool = True
    use_mujoco_renderer_only: bool = False
    # if true, use mujoco renderer for all objects, else use NeRF for target and mujoco for robot


class MotionPlannerType(Enum):
    """
    KINEMATIC_TRAJECTORY_OPTIMIZATION: Drake's KinematicTrajectoryOptimization
    https://drake.mit.edu/doxygen_cxx/classdrake_1_1planning_1_1trajectory__optimization_1_1_kinematic_trajectory_optimization.html

    LINEAR_INTERPOLATION: linearly interpolate between start and goal: hardcoded to move a maximum of 0.05m in pos/quat
        - hyperparameters: pos velocities, rot velocities

    PRM: Probabilistic Roadmap
    """

    KINEMATIC_TRAJECTORY_OPTIMIZATION = auto()
    LINEAR_INTERPOLATION = auto()
    PRM = auto()
    CUROBO = auto()
    VAMP = auto()


@dataclass
class SamplingBasedMotionPlannerConfig:
    planning_process: Literal["EEF_then_JOINT"] = "EEF_then_JOINT"  # unused for now
    # should be in general motion planner config I think
    hand_motion_planning_space_bounds: Tuple[Tuple[float, ...], Tuple[float, ...]] = (
        field(
            default_factory=lambda: (
                (-1, -1, -1, -1, -np.inf, -np.inf, -np.inf, -0.04, -0.04),
                (1, 1, 1, 1, np.inf, np.inf, np.inf, 0.04, 0.04),
            )
        )
    )  # assumes ordering of 'q' is [quat_wxyz, pos_xyz, gripper_qpos]
    timeout: float = 30

    # warm starting configs
    warm_start: bool = True
    # pad object
    env_collision_padding: float = 0
    """
    Padding to add to the 'environment' when checking for collisions between robot and environment.
    If object is welded to robot, the object is part of the robot (and should not be padded).
    """
    edge_step_size: float = 0.005
    """Step size when performing edge collision checking"""
    env_dist_factor: float = 0.3
    """Factor to multiply env_dist by when computing cost"""
    env_influence_distance: float = 0.08
    """Distance from object to consider for env influence"""


@dataclass
class MotionPlannerConfig:
    use_drake: bool = True
    view_meshcat: bool = False
    motion_planner_type: MotionPlannerType = (
        MotionPlannerType.KINEMATIC_TRAJECTORY_OPTIMIZATION
    )
    sampling_based_motion_planner_cfg: SamplingBasedMotionPlannerConfig = field(
        default_factory=lambda: SamplingBasedMotionPlannerConfig()
    )
    truncate_last_n_steps: int = 0


@dataclass
class NeRFObjectConfig:
    config_path: pathlib.Path = field(
        default_factory=lambda: pathlib.Path(
            "../nerfstudio/outputs/robomimic_lift_2023-05-25/tensorf/2023-05-25_160044/config.yml"
        )
    )
    bounding_box_min: Tuple[float, float, float] = field(
        default_factory=lambda: (-0.07, -0.07, 0.8)
    )
    bounding_box_max: Tuple[float, float, float] = field(
        default_factory=lambda: (0.07, 0.07, 0.86)
    )


@dataclass
class EnvConfig:
    env_type: EnvType = (
        EnvType.NERF_ROBOMIMIC
    )  # TODO(klin): eventually won't be robomimic
    expert_action_cfg: ExpertActionConfig = field(
        default_factory=lambda: ExpertActionConfig()
    )
    env_reset_cfg: EnvResetConfig = field(default_factory=lambda: EnvResetConfig())
    obs_cfg: ObsConfig = field(default_factory=lambda: ObsConfig())
    # these are the camera *rendering* configs!
    camera_cfg: CameraConfig = field(default_factory=lambda: CameraConfig())
    multi_camera_cfg: MultiCameraConfig = field(
        default_factory=lambda: MultiCameraConfig()
    )
    # do post_init check for robot_cfg to assert if robomimic then have dataset else inherit dataset from DemoConfig path
    robot_cfg: RobotConfig = field(default_factory=lambda: RobotConfig())
    # task_relev_obj_cfg:
    robot_nerf_weights_path: Optional[pathlib.Path] = None
    target_nerf_weights_path: Optional[pathlib.Path] = None
    done_distance_threshold: float = 0.05
    robot_reset_pose_range: Tuple[float, float] = (0, 5.5)
    target_reset_pose_range: Tuple[float, float] = (0, 5.5)
    action_space_type: ActionSpaceType = ActionSpaceType.POS_DELTA
    motion_planner_cfg: MotionPlannerConfig = field(
        default_factory=lambda: MotionPlannerConfig()
    )
    task_relev_obj_cfg: NeRFObjectConfig = field(
        default_factory=lambda: NeRFObjectConfig()
    )

    def __post_init__(self):
        if self.env_type == EnvType.NERF_3D_TRACE:
            if self.env_reset_cfg.randomize_target_pose:
                assert not self.env_reset_cfg.fix_robot_x_plane, "Cannot fix robot x plane for NERF_3D_TRACE if randomize_target_pose is set"

        if self.robot_cfg.robot_model_type == "real_FR3_robotiq":
            self.motion_planner_cfg.sampling_based_motion_planner_cfg.hand_motion_planning_space_bounds = (
                (-1, -1, -1, -1, -np.inf, -np.inf, -np.inf, 0, 0, 0, 0, 0, 0),
                (1, 1, 1, 1, np.inf, np.inf, np.inf, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8),
            )  # assumes ordering of 'q' is [quat_wxyz, pos_xyz, gripper_qpos]
            # figured out via pydrake: plant.GetPositionLowerLimits() or plant.get_joint(JointIndex(i))


class BaseEnv(gym.Env):
    """NeRF environment.

    Args:
        robot_nerf_weights_path (str): Path to the weights of the first NeRF model.
        target_nerf_weights_path (str): Path to the weights of the second NeRF model.
    """

    def __init__(
        self,
        env_cfg: EnvConfig = EnvConfig(),
    ):
        self.camera_cfg = env_cfg.camera_cfg
        self.env_cfg = env_cfg

        # Initialize NeRF models
        self.robot_nerf = self._load_nerf_model(
            env_cfg.robot_nerf_weights_path,
            rgb=torch.tensor([0, 0, 1], dtype=torch.float32),
        )
        self.target_nerf = self._load_nerf_model(
            env_cfg.target_nerf_weights_path,
            rgb=torch.tensor([0, 1, 0], dtype=torch.float32),
        )

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
        self.robot_pose: np.ndarray
        self.target_pose: np.ndarray

        lookat = self.camera_cfg.center - self.camera_cfg.target
        self.c2w = viewmatrix(lookat, self.camera_cfg.up, self.camera_cfg.center)
        self.rendered_output_names = ["rgb", "depth", "accumulation"]

        self.seed()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        self._apply_action(action)
        obs = self.get_observation()
        reward = self._get_reward()
        done = self._is_done()
        info = self._get_info()
        return obs, reward, done, info

    def _get_reward(self):
        raise NotImplementedError

    def _is_done(self):
        raise NotImplementedError

    def _get_info(self):
        return {}

    def reset(self) -> Tuple[np.ndarray, float, bool, Dict]:
        raise NotImplementedError

    def reset_to(self, reset_dict: Dict[str, np.ndarray]) -> None:
        """Reset the environment to a given state.

        For a more complete reset_dict, ideally also include CameraConfig and EnvConfig.
        """
        raise NotImplementedError

    def render(self, mode: str = "rgb_array") -> Dict[str, np.ndarray]:
        raise NotImplementedError

    def get_observation(self) -> Dict[str, np.ndarray]:
        return self.render()

    def seed(self, seed: int = None) -> List[int]:
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _apply_action(self, action: np.ndarray) -> None:
        raise NotImplementedError

    def _load_nerf_model(
        self, weights_path: pathlib.Path, rgb: Optional[torch.Tensor] = None
    ) -> SphereNeRF:
        # Load NeRF model weights
        if weights_path is None:
            return SphereNeRF(rgb=rgb)

    def optimal_action(self) -> np.ndarray:
        """Compute optimal action by moving in the direction of the target"""
        raise NotImplementedError

    def optimal_trajectory(self) -> np.ndarray:
        """Compute optimal trajectory by moving in the direction of the target"""
        raise NotImplementedError

    def get_env_state(self) -> Dict[str, np.ndarray]:
        raise NotImplementedError
