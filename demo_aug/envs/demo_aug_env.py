import pathlib
from typing import Dict, List, Optional, Tuple

# use gym to be compatible with diffusion_policy
import numpy as np
import torch
from gym import spaces
from gym.utils import seeding

# ideally remove as dependency. However, looks necessary for camera rendering. Hack setup.py file instead.
from nerfstudio.cameras.camera_utils import viewmatrix

from demo_aug.envs.base_env import ActionSpaceType, BaseEnv, EnvConfig
from demo_aug.utils.nerf_obj import SphereNeRF


# currently unused
class DemoAugEnv(BaseEnv):
    """DemoAugEnv environment.

    Env that takes in a constraint pose etc and returns a rendered image.
    Then, we check how close we end up with the constraint pose?

    Annoying thing is that the env is static so we can only get up to the constraint pose ...

    I think a one step thing is probably better since the supervised loss tell us the same
    information without us needing to rendering a bunch of stuff.
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

    def get_observation(
        self,
        robot_pose: Optional[np.ndarray] = None,
        demo_relevant_objs_pose: Optional[np.ndarray] = None,
        demo_irrelevant_objs_pose: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        raise NotImplementedError

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

    def is_colliding(
        self,
        robot_pose: Optional[np.ndarray] = None,
        demo_relevant_objs_pose: Optional[np.ndarray] = None,
        demo_irrelevant_objs_pose: Optional[np.ndarray] = None,
    ) -> bool:
        """Check if the robot is colliding with any of the objects"""
        raise NotImplementedError

    @staticmethod
    def train_nerf(
        nerf_data_path: pathlib.Path, save_nerf_path: pathlib.Path
    ) -> callable:
        raise NotImplementedError

    def add_demo_relevant_objs(self) -> None:
        raise NotImplementedError

    def add_demo_irrelevant_objs(self) -> None:
        raise NotImplementedError

    def add_robot(self) -> None:
        """Add a robot to the environment.

        Key requirement is that the object must have collision geometry and be renderable.
        Representation: nerf representation
        """
        raise NotImplementedError
