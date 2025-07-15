import pathlib
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

# use gym to be compatible with diffusion_policy
from gym import spaces
from gym.utils import seeding

# ideally remove as dependency. However, looks necessary for camera rendering. Hack setup.py file instead.
from nerfstudio.cameras.camera_utils import viewmatrix

from demo_aug.envs.base_env import ActionSpaceType, BaseEnv, EnvConfig, NeRFModelType
from demo_aug.utils.composite import alpha_composite
from demo_aug.utils.mathutils import axisangle2quat, quat2mat, random_rotation_matrix
from demo_aug.utils.nerf_obj import CylinderNeRF, CylinderWithTwoSpheresNeRF, SphereNeRF


class TaskProgressName(Enum):
    REACHING = auto()
    TRACING = auto()
    DONE = auto()


class TaskProgress:
    def __init__(self, stage: TaskProgressName):
        self.stage = stage
        self.reach_stage_max_dist: float
        self.trace_stage_max_dist: float

    def reset_task_progress(
        self, reach_stage_max_dist: float, trace_stage_max_dist: float
    ):
        """TODO(klin): hard coded task progress arguments isn't great"""
        self.reach_stage_max_dist = reach_stage_max_dist
        self.trace_stage_max_dist = trace_stage_max_dist
        self.stage = TaskProgressName.REACHING

    def update_stage(self, stage: TaskProgressName):
        self.stage = stage


class NeRF3DTraceEnv(BaseEnv):
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

        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = "cpu"
        self.dtype = torch.float32

        # Initialize NeRF models
        self.robot_nerf = self._load_nerf_model(
            env_cfg.robot_nerf_weights_path,
            rgb=torch.tensor([0, 0, 1], dtype=self.dtype),
            nerf_model_type=NeRFModelType.SPHERE,
        )
        self.target_nerf = CylinderWithTwoSpheresNeRF()

        # Target nerf can be an entire structure too
        self.initial_robot_pose: torch.Tensor = torch.eye(
            4, dtype=self.dtype, device=self.device
        )[:3]
        self.non_transformed_entry_sphere_pose: torch.Tensor = torch.eye(
            4, dtype=self.dtype, device=self.device
        )[:3]
        self.non_transformed_exit_sphere_pose: torch.Tensor = torch.eye(
            4, dtype=self.dtype, device=self.device
        )[:3]

        self.non_transformed_entry_sphere_pose[:, 3] = (
            self.target_nerf.sphere1.sphere_center
        )
        self.non_transformed_exit_sphere_pose[:, 3] = (
            self.target_nerf.sphere2.sphere_center
        )

        # TODO(klin) keep consistent between np and torch?
        # TBD upon reset()
        self.entry_sphere_centroid: np.ndarray
        self.exit_sphere_centroid: np.ndarray

        lookat = self.camera_cfg.center - self.camera_cfg.target
        self.c2w = viewmatrix(lookat, self.camera_cfg.up, self.camera_cfg.center)
        self.rendered_output_names = self.camera_cfg.rendered_output_names

        self.task_progress = TaskProgress(TaskProgressName.REACHING)

        self.seed()

    def _get_reward(self) -> float:
        """Calculates the reward based on the current task progress stage and the robot's position.
        Maximum reward = dist(robot_init, sphere_entry) + dist(sphere_entry, sphere_exit).

        Also, updates the task progress stage.
        TODO(klin): unclear if _get_reward is the same as "_update_task_progress"
        having a separate _update_task_progress may need repeated computations which is also awkward ...?

        """
        robot_centroid = NeRF3DTraceEnv.compute_centroid(self.robot_pose)

        if self.task_progress.stage.name == TaskProgressName.REACHING.name:
            dist_robot_entry_sphere = np.linalg.norm(
                self.entry_sphere_centroid - robot_centroid
            )
            reward = self.task_progress.reach_stage_max_dist - dist_robot_entry_sphere
        elif self.task_progress.stage.name == TaskProgressName.TRACING.name:
            dist_robot_exit_sphere = np.linalg.norm(
                self.exit_sphere_centroid - robot_centroid
            )
            reward = (
                self.task_progress.trace_stage_max_dist - dist_robot_exit_sphere
            ) + self.task_progress.reach_stage_max_dist
        elif self.task_progress.stage.name == TaskProgressName.DONE.name:
            reward = (
                self.task_progress.reach_stage_max_dist
                + self.task_progress.trace_stage_max_dist
            )
        else:
            raise NotImplementedError(
                f"Unknown task progress stage {self.task_progress.stage.name}"
            )

        # update task progress
        if self.task_progress.stage.name == TaskProgressName.REACHING.name:
            dist_robot_entry_sphere = np.linalg.norm(
                self.entry_sphere_centroid - robot_centroid
            )
            if dist_robot_entry_sphere < self.env_cfg.done_distance_threshold:
                self.task_progress.update_stage(TaskProgressName.TRACING)
        elif self.task_progress.stage.name == TaskProgressName.TRACING.name:
            dist_robot_exit_sphere = np.linalg.norm(
                self.exit_sphere_centroid - robot_centroid
            )
            if dist_robot_exit_sphere < self.env_cfg.done_distance_threshold:
                self.task_progress.update_stage(TaskProgressName.DONE)

        return reward

    def reset(self) -> Tuple[np.ndarray, float, bool, Dict]:
        self.robot_pose = torch.eye(4, dtype=self.dtype, device=self.device)[:3]
        self.target_pose = torch.eye(4, dtype=self.dtype, device=self.device)[:3]

        if self.env_cfg.env_reset_cfg.randomize_target_pose:
            self.target_pose[:, 3] = (
                torch.rand(3, dtype=self.dtype, device=self.device) - 0.5
            ) * self.env_cfg.target_reset_pose_range[1]
            # ideally check if target_pose is visible from camera, hardcode for now
            self.target_pose[0, 3] *= 3 / 5
            if self.env_cfg.env_reset_cfg.fix_target_x_plane:
                self.target_pose[0, 3] = 0

            self.target_pose[:3, :3] = random_rotation_matrix()

        if self.env_cfg.env_reset_cfg.randomize_robot_pose:
            self.robot_pose[:, 3] = (
                torch.rand(3, dtype=self.dtype, device=self.device) - 0.5
            ) * self.env_cfg.robot_reset_pose_range[1]
            # ideally check if robot_pose is visible from camera, hack for now
            self.robot_pose[0, 3] *= 3 / 4
            if self.env_cfg.env_reset_cfg.fix_robot_x_plane:
                self.robot_pose[0, 3] = 0

        # reach_stage_max_dist = distance between current entry sphere and current robot centroids
        current_robot_centroid = NeRF3DTraceEnv.compute_centroid(self.robot_pose)
        target_pose_homog = torch.eye(4, dtype=self.dtype, device=self.device)
        target_pose_homog[:3, :4] = self.target_pose

        # target pose and hence entry/exit sphere poses are fixed for an episode;
        # bite the computation bullet and recompute centroids based on current target pose
        reach_stage_max_dist: float = np.linalg.norm(
            self.entry_sphere_centroid - current_robot_centroid
        )

        # trace_stage_max_dist = distance between current exit sphere and current entry sphere centroids
        trace_stage_max_dist: float = np.linalg.norm(
            self.exit_sphere_centroid - self.entry_sphere_centroid
        )

        self.task_progress.reset_task_progress(
            reach_stage_max_dist, trace_stage_max_dist
        )
        obs = self.get_observation()
        reward = self._get_reward()

        # Compute new centroid positions
        robot_centroid = NeRF3DTraceEnv.compute_centroid(self.robot_pose)
        target_centroid = NeRF3DTraceEnv.compute_centroid(self.target_pose)

        return (
            obs,
            reward,
            False,
            {
                "robot_centroid": robot_centroid,
                "target_centroid": target_centroid,
            },
        )

    def get_valid_robot_pose(self, goal_pose: np.ndarray) -> np.ndarray:
        """Currently, this function doesn't ensure the sampled pose is:

        kinematically feasible, collision-free and can reach the goal without colliding with the background.

        # perhaps this check can occur later on?
        Ideally, we would check if the robot is in collision with the target object. More generally,
        we would check:
            i) if the robot is in collision with any object in the scene
            ii) if it's possible to reach the goal pose from the current pose without collisions; note the optimization
                might get weird if the goal_pose requires a contact or gets close to a contact.

        Args:
            goal_pose (np.ndarray): The desired pose of the robot. In the sphere env, assume there are no collisions.
        """
        robot_pose = torch.eye(4, dtype=self.dtype, device=self.device)[:3]
        robot_pose[:, 3] = (
            torch.rand(3, dtype=self.dtype) - 0.5
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

    # TODO(klin): make this function able to take in an robot pose
    # argh there may be bugs if using this function that doesn't explicitly require robot pose and "falls" back
    # to the default!
    def render(
        self,
        robot_pose: Optional[np.ndarray] = None,
        target_pose: Optional[np.ndarray] = None,
        robot_obj: Optional[Any] = None,
        target_obj: Optional[Any] = None,
        mode: str = "rgb_array",
    ) -> Dict[str, np.ndarray]:
        robot_obj = robot_obj if robot_obj is not None else self.robot_nerf
        target_obj = target_obj if target_obj is not None else self.target_nerf
        robot_pose = robot_pose if robot_pose is not None else self.robot_pose
        target_pose = target_pose if target_pose is not None else self.target_pose

        obs_robot = robot_obj.render(
            robot_pose,
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
        obs_target = target_obj.render(
            target_pose,
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

    def get_observation(
        self,
        robot_pose: Optional[np.ndarray] = None,
        target_pose: Optional[np.ndarray] = None,
        robot_obj: Optional[Any] = None,
        target_obj: Optional[Any] = None,
    ) -> Dict[str, np.ndarray]:
        return self.render(robot_pose, target_pose, robot_obj, target_obj)

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
                torch.as_tensor(action, dtype=self.dtype) + self.robot_pose[:3, 3]
            )
            if self.env_cfg.env_reset_cfg.fix_robot_x_plane:
                self.robot_pose[0, 3] = 0
            return

        pos, axisangle = action[:3], action[3:]
        quat = axisangle2quat(axisangle)
        rot = quat2mat(quat)

        self.robot_pose[:3, :3] = (
            torch.as_tensor(rot, dtype=self.dtype) @ self.robot_pose[:3, :3]
        )
        self.robot_pose[:3, 3] = (
            torch.as_tensor(pos, dtype=self.dtype) + self.robot_pose[:3, 3]
        )

        if self.env_cfg.env_reset_cfg.fix_robot_x_plane:
            self.robot_pose[0, 3] = 0

    def _load_nerf_model(
        self,
        weights_path: pathlib.Path,
        rgb: Optional[torch.Tensor] = None,
        nerf_model_type: NeRFModelType = None,
    ) -> SphereNeRF:
        if nerf_model_type is None:
            raise ValueError("nerf_model_type must be specified")

        if nerf_model_type == NeRFModelType.SPHERE:
            return SphereNeRF(rgb=rgb, sphere_radius=0.5)
        elif nerf_model_type == NeRFModelType.CYLINDER:
            return CylinderNeRF(rgb=rgb)
        elif nerf_model_type == NeRFModelType.CYLINDER_WITH_SPHERES:
            return CylinderWithTwoSpheresNeRF(rgb=rgb)
        elif nerf_model_type == NeRFModelType.TRAINED_NERF:
            raise NotImplementedError("Trained NeRF model is not implemented yet")

    @staticmethod
    def compute_centroid(pose: Union[np.ndarray, torch.Tensor]):
        if isinstance(pose, torch.Tensor):
            pose = pose.numpy()
        centroid = pose[..., :3, 3].copy()
        return centroid

    @property
    def entry_sphere_centroid(self) -> np.ndarray:
        # convert self.target_pose to tensor if not already
        if not isinstance(self.target_pose, torch.Tensor):
            self.target_pose = torch.tensor(
                self.target_pose, dtype=self.dtype, device=self.device
            )
        target_pose_homog = torch.eye(4, dtype=self.dtype, device=self.device)
        target_pose_homog[:3, :4] = self.target_pose

        non_transformed_entry_sphere_pose_homog = torch.eye(
            4, dtype=self.dtype, device=self.device
        )
        non_transformed_entry_sphere_pose_homog[:3, :4] = (
            self.non_transformed_entry_sphere_pose
        )
        entry_sphere_pose_homog = (
            target_pose_homog @ non_transformed_entry_sphere_pose_homog
        )
        return NeRF3DTraceEnv.compute_centroid(entry_sphere_pose_homog)

    @property
    def exit_sphere_centroid(self) -> np.ndarray:
        if not isinstance(self.target_pose, torch.Tensor):
            self.target_pose = torch.tensor(
                self.target_pose, dtype=self.dtype, device=self.device
            )
        target_pose_homog = torch.eye(4, dtype=self.dtype, device=self.device)
        target_pose_homog[:3, :4] = self.target_pose

        non_transformed_exit_sphere_pose_homog = torch.eye(
            4, dtype=self.dtype, device=self.device
        )
        non_transformed_exit_sphere_pose_homog[:3, :4] = (
            self.non_transformed_exit_sphere_pose
        )
        exit_sphere_pose_homog = (
            target_pose_homog @ non_transformed_exit_sphere_pose_homog
        )
        return NeRF3DTraceEnv.compute_centroid(exit_sphere_pose_homog)

    def optimal_action(self) -> np.ndarray:
        """Compute optimal action by:

        If in reaching stage, reach the entry sphere centroid
        If in tracing stage, reach the exit sphere centroid
        """
        robot_centroid = NeRF3DTraceEnv.compute_centroid(self.robot_pose)
        if self.task_progress.stage == TaskProgressName.REACHING:
            target_centroid = self.entry_sphere_centroid
        elif self.task_progress.stage == TaskProgressName.TRACING:
            target_centroid = self.exit_sphere_centroid
        elif self.task_progress.stage == TaskProgressName.DONE:
            return np.zeros(3, dtype=np.float32)
        else:
            raise NotImplementedError(f"Unknown stage {self.task_progress.stage}")

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
        TODO(klin): non-standard function. Find better name.

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

        robot_centroid = NeRF3DTraceEnv.compute_centroid(robot_pose)
        target_centroid = NeRF3DTraceEnv.compute_centroid(target_pose)
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
            robot_centroid = NeRF3DTraceEnv.compute_centroid(curr_robot_pose)

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

    def _is_done(self):
        return self.task_progress.stage == TaskProgressName.DONE

    # also, the NeRFs themselves aren't renderable collidable right?
    def set_demo_relevant_objs(self, objs):
        self.target_nerf = objs

    def get_demo_relevant_objs_pose(self) -> np.ndarray:
        return self.target_pose

    def set_demo_relevant_objs_pose(self, pose: np.ndarray) -> None:
        self.target_pose = pose

    def set_robot_pose(self, pose: np.ndarray) -> None:
        self.robot_pose = pose

    def set_robot_pose_goal(self, pose: np.ndarray) -> None:
        self.target_pose = pose

    def sample_valid_q_values(
        self, use_noise_reversal: bool = False
    ) -> Tuple[np.ndarray, Dict]:
        """Sample a valid joint configuration, q, for the robot.

        Currently hardcoded for sphere robot.

        Args:
            use_noise_reversal: Whether to use noise reversal to start from current (s, a, s')
            and generate a new transition (s_bar, a_bar, s')
        """
        if use_noise_reversal:
            raise NotImplementedError(
                "Noise reversal for generating augmented actions is not implemented yet"
            )

        robot_pose = torch.eye(4, dtype=self.dtype, device=self.device)[:3]
        robot_pose[:, 3] = (
            torch.rand(3, dtype=self.dtype) - 0.5
        ) * self.env_cfg.robot_reset_pose_range[1]
        return robot_pose, {}

    def generate_expert_trajectory(
        self, q_goal: np.ndarray, q_start: np.ndarray, sampling_info: Optional[Dict]
    ) -> List[np.ndarray]:
        """Generate a trajectory starting from q_goal and ending on q_start.
        Currently has no collision checking, etc.

        sampling_info: Optional[Dict]
            May contain info to warm start trajectory generation.

        Returns:
            actions: List[np.ndarray] length N - 1
            robot_poses: List[np.ndarray] length N
        """
        # convert to numpy if necessary
        if isinstance(q_goal, torch.Tensor):
            q_goal = q_goal.detach().numpy()
        if isinstance(q_start, torch.Tensor):
            q_start = q_start.detach().numpy()

        curr_robot_pose = q_start.copy()
        curr_target_pose = q_goal.copy()

        actions = []
        robot_poses = [curr_robot_pose]

        robot_centroid = NeRF3DTraceEnv.compute_centroid(curr_robot_pose)
        target_centroid = NeRF3DTraceEnv.compute_centroid(curr_target_pose)
        distance = np.linalg.norm(robot_centroid - target_centroid)
        while distance > 0:
            direction = target_centroid - robot_centroid
            direction = direction / np.linalg.norm(direction)

            action = self.action_space.sample() * 0.0
            if distance != 0:
                action[:3] = direction * min(
                    distance, self.env_cfg.expert_action_cfg.max_dist
                )
                # action[:3] = [0, -0.3, 0]
            # expand out action to have same shape as current robot pose
            pose_action = np.zeros_like(curr_robot_pose)
            pose_action[:3, 3] = action[:3]
            curr_robot_pose = curr_robot_pose + pose_action
            robot_centroid = NeRF3DTraceEnv.compute_centroid(curr_robot_pose)

            actions.append(action)
            robot_poses.append(curr_robot_pose)

            distance = np.linalg.norm(robot_centroid - target_centroid)

        return actions, robot_poses

    def apply_se3_transform_to_robot_and_demo_relev(
        self,
        orig_robot_pose: np.ndarray,
        orig_current_and_future_actions: np.ndarray,
        orig_target_pose: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply a SE3 transformation to robot (end effector), current and future actions and demo relevant object.

        Implementation only works for a sphere robot. For robot with full pose delta actions,
        would need to modify the orientation of actions.
        """
        # convert to tensor if necessary
        if not isinstance(orig_robot_pose, torch.Tensor):
            orig_robot_pose = np.array(orig_robot_pose)
            orig_robot_pose = torch.tensor(
                orig_robot_pose, dtype=self.dtype, device=self.device
            )
        if not isinstance(orig_current_and_future_actions, torch.Tensor):
            orig_current_and_future_actions = np.array(orig_current_and_future_actions)
            orig_current_and_future_actions = torch.tensor(
                orig_current_and_future_actions, dtype=self.dtype, device=self.device
            )
        if not isinstance(orig_target_pose, torch.Tensor):
            orig_target_pose = np.array(orig_target_pose)
            orig_target_pose = torch.tensor(
                orig_target_pose, dtype=self.dtype, device=self.device
            )

        # Determine the shape of the input tensors
        input_shape = orig_robot_pose.shape

        # Reshape the input tensors to match [N, 4, 3] shape if needed
        if len(input_shape) == 2:
            orig_robot_pose = orig_robot_pose.unsqueeze(0)
            orig_current_and_future_actions = orig_current_and_future_actions.unsqueeze(
                0
            )
            orig_target_pose = orig_target_pose.unsqueeze(0)

        R = random_rotation_matrix(self.dtype)

        # sample a random translation; hack to keep target in view
        t_abs = (
            torch.rand(3, dtype=self.dtype, device=self.device) - 0.5
        ) * self.env_cfg.robot_reset_pose_range[1]
        t = t_abs - orig_robot_pose[0, :3, 3]

        transf_homog = torch.eye(4, dtype=self.dtype, device=self.device)
        transf_homog[..., :3, :3] = R
        transf_homog[..., :3, 3] = t

        # apply the transformation to the robot pose
        transf_robot_pose = (
            torch.eye(4, dtype=self.dtype, device=self.device)
            .unsqueeze(0)
            .repeat(orig_robot_pose.shape[0], 1, 1)
        )
        transf_robot_pose[..., :3, :4] = orig_robot_pose.clone()
        transf_robot_pose = torch.matmul(transf_homog, transf_robot_pose)

        # currently hardcoded to not be general
        # action is a vector so need to update current action to [action, 0] then hit with trans_homog
        transf_actions = orig_current_and_future_actions.clone()
        transf_actions = torch.cat(
            (transf_actions, torch.zeros_like(transf_actions[..., :1])), dim=-1
        )

        transf_actions[..., :4] = torch.matmul(
            transf_homog, transf_actions[..., :4].transpose(-2, -1)
        ).transpose(-2, -1)

        # convert R, t to homogeneous matrix
        transf_target_pose = (
            torch.eye(4, dtype=self.dtype, device=self.device)
            .unsqueeze(0)
            .repeat(orig_target_pose.shape[0], 1, 1)
        )
        transf_target_pose[..., :3, :4] = orig_target_pose.clone()
        transf_target_pose = torch.matmul(transf_homog, transf_target_pose)

        return (
            transf_robot_pose[..., :3, :4],
            transf_actions[..., :3],
            transf_target_pose[..., :3, :4],
        )
        # return transf_robot_pose[:3, :4], transf_actions[..., :3], transf_target_pose[:3, :4]

    def apply_scaling_transform_to_demo_relevant(
        self,
        orig_robot_pose: np.ndarray,
        orig_current_and_future_actions: np.ndarray,
        orig_target_pose: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ """
        raise NotImplementedError(
            "apply_scaling_transform_to_demo_relevant: Not implemented yet"
        )

    def get_observations_from_robot_demo_relevant_obj_trajectory(
        self,
        robot_trajectory: List[np.ndarray],
        demo_relevant_objs_trajectory: List[np.ndarray],
        robot_objs_list: Optional[List[Any]] = None,
        demo_relevant_objs_list: Optional[List[Any]] = None,
    ) -> Dict[str, np.ndarray]:
        obs_lst: List = []
        assert (
            len(robot_trajectory) == len(demo_relevant_objs_trajectory)
        ), "Mismatched lengths: len(robot_trajectory) != len(demo_relevant_objs_trajectory)"
        if robot_objs_list is not None:
            assert len(robot_objs_list) == len(robot_trajectory), "mismatched lengths"
        if demo_relevant_objs_list is not None:
            assert len(demo_relevant_objs_list) == len(
                demo_relevant_objs_trajectory
            ), "mismatched lengths"
        for i in range(len(robot_trajectory)):
            obs = self.get_observation(
                robot_pose=robot_trajectory[i],
                target_pose=demo_relevant_objs_trajectory[i],
                robot_obj=robot_objs_list[i] if robot_objs_list is not None else None,
                target_obj=demo_relevant_objs_list[i]
                if demo_relevant_objs_list is not None
                else None,
            )
            obs_lst.append(obs)
        # images= []
        # from PIL import Image
        # import imageio
        # for j in range(len(obs_lst)):
        #     img = Image.fromarray((np.transpose(obs_lst[j]["agentview_image"], (1, 2, 0)) * 255).astype(np.uint8))
        #     img.show()
        #     images.append(img)

        # gif_path = f"demo_{9999}.gif"
        # print(f"Saving gif to {gif_path}")
        # imageio.mimsave(gif_path, images, duration=0.1)

        return obs_lst
