import pathlib
from typing import Dict, List, Optional, Tuple, Union

# use gym to be compatible with diffusion_policy
import gym
import numpy as np
import torch
from gym import spaces
from gym.utils import seeding
from nerfstudio.cameras.camera_utils import viewmatrix

# ideally remove as dependency. However, looks necessary for camera rendering. Hack setup.py file instead.
from demo_aug.envs.base_env import ActionSpaceType, EnvConfig
from demo_aug.utils.composite import alpha_composite
from demo_aug.utils.mathutils import axisangle2quat, quat2mat
from demo_aug.utils.nerf_obj import SphereNeRF


class SphereNeRFEnv(gym.Env):
    """NeRF environment for testing.

    The environment consists of two NeRF models, each representing a single object.
    The goal is to move the first object so that it is as close as possible to the
    second object. The observation is the rendered image of both objects from a
    fixed camera pose. The reward is the negative distance between the two objects.

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
        self.observation_space["rgb"] = spaces.Box(low=0, high=255, shape=(3, 96, 96))

        # Initialize NeRF models
        self.robot_nerf = self._load_nerf_model(
            env_cfg.robot_nerf_weights_path,
            rgb=torch.tensor([0, 0, 1], dtype=torch.float32),
        )
        self.target_nerf = self._load_nerf_model(
            env_cfg.target_nerf_weights_path,
            rgb=torch.tensor([0, 1, 0], dtype=torch.float32),
        )

        # Initialize pose and centroid variables
        self.robot_pose: np.ndarray
        self.target_pose: np.ndarray

        lookat = self.camera_cfg.center - self.camera_cfg.target
        self.c2w = viewmatrix(lookat, self.camera_cfg.up, self.camera_cfg.center)
        self.rendered_output_names = ["rgb", "depth", "accumulation"]

        self.seed()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        self._apply_action(action)

        # Compute new centroid positions
        robot_centroid = self._compute_centroid(self.robot_pose)
        target_centroid = self._compute_centroid(self.target_pose)

        # Compute distance between centroids
        distance = np.linalg.norm(robot_centroid - target_centroid)

        # Render observations from fixed pose
        obs = self.render()

        done = False
        if distance < self.env_cfg.done_distance_threshold:
            done = True

        return (
            obs,
            -distance,
            done,
            {
                "distance": distance,
                "robot_centroid": robot_centroid,
                "target_centroid": target_centroid,
            },
        )

    def reset(self) -> Tuple[np.ndarray, float, bool, Dict]:
        self.robot_pose = torch.eye(4)[:3]
        self.target_pose = torch.eye(4)[:3]

        if self.env_cfg.env_reset_cfg.randomize_target_pose:
            self.target_pose[:, 3] = (
                torch.rand(3, dtype=torch.float32) - 0.5
            ) * self.env_cfg.target_reset_pose_range[1]
            # ideally check if target_pose is visible from camera, hack for now
            self.target_pose[0, 3] *= 3 / 4
            if self.env_cfg.env_reset_cfg.fix_target_x_plane:
                self.target_pose[0, 3] = 0

        if self.env_cfg.env_reset_cfg.randomize_robot_pose:
            self.robot_pose[:, 3] = (
                torch.rand(3, dtype=torch.float32) - 0.5
            ) * self.env_cfg.robot_reset_pose_range[1]
            # ideally check if robot_pose is visible from camera, hack for now
            self.robot_pose[0, 3] *= 3 / 4
            if self.env_cfg.env_reset_cfg.fix_robot_x_plane:
                self.robot_pose[0, 3] = 0

        # Compute new centroid positions
        robot_centroid = self._compute_centroid(self.robot_pose)
        target_centroid = self._compute_centroid(self.target_pose)

        # Compute distance between centroids
        distance = np.linalg.norm(robot_centroid - target_centroid)

        # Render observations from fixed pose
        obs = self.render()

        return (
            obs,
            -distance,
            False,
            {
                "distance": distance,
                "robot_centroid": robot_centroid,
                "target_centroid": target_centroid,
            },
        )

    def get_valid_start_robot_pose(self, goal_pose: np.ndarray) -> np.ndarray:
        """Currently, this function is unhinged since it doesn't check if the pose is physically feasible.

        Ideally, we would check if the robot is in collision with the target object. More generally,
        we would check:
            i) if the robot is in collision with any object in the scene
            ii) if it's possible to reach the goal pose from the current pose without collisions; note the optimization
                might get weird if the goal_pose requires a contact or gets close to a contact.

        Args:
            goal_pose (np.ndarray): The desired pose of the robot. In the sphere env, assume there are no collisions.
        """
        robot_pose = torch.eye(4)[:3]
        robot_pose[:, 3] = (
            torch.rand(3, dtype=torch.float32) - 0.5
        ) * self.env_cfg.robot_reset_pose_range[1]

        # ideally check if robot_pose is visible from camera, hack for now
        robot_pose[0, 3] *= 3 / 4
        if self.env_cfg.env_reset_cfg.fix_robot_x_plane:
            robot_pose[0, 3] = 0

        # TODO(klin): check if robot_pose is in collision with any object in the scene
        # TODO(klin): check if it's possible to reach the goal pose from the current pose without collisions
        return robot_pose

    def reset_to(self, reset_dict: Dict[str, np.ndarray]) -> None:
        """Reset the environment to a given state.

        For a more complete reset_dict, ideally also include CameraConfig and EnvConfig.
        """
        if reset_dict is None or len(reset_dict) == 0:
            print("reset_dict is empty, calling reset()")
            self.reset()
            return

        self.robot_pose = reset_dict["robot_pose"].copy()
        self.target_pose = reset_dict["target_pose"].copy()

    def render(
        self,
        robot_pose: Optional[np.ndarray] = None,
        target_pose: Optional[np.ndarray] = None,
        mode: str = "rgb_array",
    ) -> Dict[str, np.ndarray]:
        obs_robot = self.robot_nerf.render(
            robot_pose if robot_pose is not None else self.robot_pose,
            self.c2w,
            self.camera_cfg.cx,
            self.camera_cfg.cy,
            self.camera_cfg.fx,
            self.camera_cfg.fy,
            self.camera_cfg.height,
            self.camera_cfg.width,
            self.camera_cfg.camera_type,
            self.rendered_output_names,
        )
        obs_target = self.target_nerf.render(
            target_pose if target_pose is not None else self.target_pose,
            self.c2w,
            self.camera_cfg.cx,
            self.camera_cfg.cy,
            self.camera_cfg.fx,
            self.camera_cfg.fy,
            self.camera_cfg.height,
            self.camera_cfg.width,
            self.camera_cfg.camera_type,
            self.rendered_output_names,
        )

        if self.env_cfg.obs_cfg.render_target:
            rgb_list = torch.stack([obs_robot["rgb"], obs_target["rgb"]])
            # disp_list is not fully correct but enough for alpha compositing
            disp_list = torch.stack([1 / obs_robot["depth"], 1 / obs_target["depth"]])
            acc_list = torch.stack(
                [obs_robot["accumulation"], obs_target["accumulation"]]
            )
            obs, alph = alpha_composite(rgb_list, disp_list, acc_list)
        else:
            obs = obs_robot["rgb"]
            alph = 1 / obs_robot["depth"]

        # match robomimic's output format of (C x H x W) and type float32
        return {
            "agentview_image": obs.permute(2, 0, 1).numpy().copy(),
            "agentview_alph": alph.numpy().copy(),
            "agentview_c2w": self.c2w.numpy().copy(),
            "agentview_cam_fl_x": self.camera_cfg.fx,
            "agentview_cam_fl_y": self.camera_cfg.fy,
            "agentview_cam_cx": self.camera_cfg.cx,
            "agentview_cam_cy": self.camera_cfg.cy,
        }

    def get_observation(self) -> Dict[str, np.ndarray]:
        return self.render()

    def seed(self, seed: int = None) -> List[int]:
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _apply_action(self, action: np.ndarray) -> None:
        if not self.action_space.contains(action):
            raise ValueError(
                f"Action {action} must be within action space {self.action_space}"
            )

        if self.env_cfg.action_space_type == ActionSpaceType.POS_DELTA:
            self.robot_pose[:3, 3] = (
                torch.as_tensor(action, dtype=torch.float32) + self.robot_pose[:3, 3]
            )
            if self.env_cfg.env_reset_cfg.fix_robot_x_plane:
                self.robot_pose[0, 3] = 0
            return

        pos, axisangle = action[:3], action[3:]
        quat = axisangle2quat(axisangle)
        rot = quat2mat(quat)

        self.robot_pose[:3, :3] = (
            torch.as_tensor(rot, dtype=torch.float32) @ self.robot_pose[:3, :3]
        )
        self.robot_pose[:3, 3] = (
            torch.as_tensor(pos, dtype=torch.float32) + self.robot_pose[:3, 3]
        )

        if self.env_cfg.env_reset_cfg.fix_robot_x_plane:
            self.robot_pose[0, 3] = 0

    def _load_nerf_model(
        self, weights_path: pathlib.Path, rgb: Optional[torch.Tensor] = None
    ) -> SphereNeRF:
        # Load NeRF model weights
        if weights_path is None:
            return SphereNeRF(rgb=rgb)

    def _compute_centroid(self, pose: Union[np.ndarray, torch.Tensor]):
        if isinstance(pose, torch.Tensor):
            pose = pose.numpy()
        centroid = pose[..., 3].copy()
        return centroid

    def optimal_action(self) -> np.ndarray:
        """Compute optimal action by moving in the direction of the target"""
        robot_centroid = self._compute_centroid(self.robot_pose)
        target_centroid = self._compute_centroid(self.target_pose)
        distance = np.linalg.norm(robot_centroid - target_centroid)
        direction = target_centroid - robot_centroid
        direction = direction / np.linalg.norm(direction)

        action = self.action_space.sample() * 0.0
        if distance != 0:
            action[:3] = direction * min(
                distance, self.env_cfg.expert_action_cfg.max_dist
            )
        return action

    def generate_goal_reaching_actions_observations_poses(
        self,
        robot_pose: Union[np.ndarray, torch.Tensor],
        target_pose: Union[np.ndarray, torch.Tensor],
    ) -> Tuple[List[np.ndarray], List[Dict[str, np.ndarray]], List[np.ndarray]]:
        """
        TODO(klin): very non-standard function. Find better name.

        Generate a sequence of actions, observations, and robot poses recorded when moving the robot from
        robot_pose to target_pose. The sequence is generated by moving the robot in the direction of
        the target until the robot is within a certain distance of the target.
        """
        actions: List[np.ndarray] = []
        observations: List[Dict[str, np.ndarray]] = []
        robot_poses: List[np.ndarray] = []

        if isinstance(robot_pose, torch.Tensor):
            robot_pose = robot_pose.numpy()
        if isinstance(target_pose, torch.Tensor):
            target_pose = target_pose.numpy()

        curr_robot_pose = robot_pose.copy()
        curr_target_pose = target_pose.copy()

        robot_centroid = self._compute_centroid(robot_pose)
        target_centroid = self._compute_centroid(target_pose)
        distance = np.linalg.norm(robot_centroid - target_centroid)
        while distance > self.env_cfg.expert_action_cfg.max_dist:
            direction = target_centroid - robot_centroid
            direction = direction / np.linalg.norm(direction)
            observation = self.render(
                robot_pose=curr_robot_pose, target_pose=curr_target_pose
            )
            action = self.action_space.sample() * 0.0
            if distance != 0:
                action[:3] = direction * min(
                    distance, self.env_cfg.expert_action_cfg.max_dist
                )

            curr_robot_pose[:3, 3] = curr_robot_pose[:3, 3] + action[:3]
            robot_centroid = self._compute_centroid(curr_robot_pose)

            actions.append(action)
            observations.append(observation)
            robot_poses.append(curr_robot_pose)

            distance = np.linalg.norm(robot_centroid - target_centroid)
        return actions, observations, robot_poses

    def get_env_state(self) -> Dict[str, np.ndarray]:
        return {
            "robot_pose": self.robot_pose.numpy().copy(),
            "target_pose": self.target_pose.numpy().copy(),
        }
