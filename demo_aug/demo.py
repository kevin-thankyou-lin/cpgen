"""
Function that takes in a video where each frame contains proprioception of
the robot (since we have that for free), images and corresonding camera info.

At each time step, we've access to a trained nerf from which we can render
new scenes from.

Key functions:

0. identify constraint pose
1. sample (robot) pose
2. reset to pose
3. generate optimal action
4. collect trajectory
"""

import logging
import pathlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np

from demo_aug.aug_datastructures import (
    TimestepAnnotationData,
    TimestepData,
)
from demo_aug.configs.base_config import (
    ConstraintInfo,
)
from demo_aug.envs.motion_planners.base_motion_planner import RobotEnvConfig
from demo_aug.objects.nerf_object import (
    GSplatObject,
    MeshObject,
    NeRFObject,
    NeRFObjectManager,
)
from demo_aug.objects.reconstructor import ReconstructionManager, ReconstructionType
from demo_aug.utils.viz_utils import create_gifs_from_h5py_file

logging.basicConfig(level=logging.INFO)


class DemoType(Enum):
    GRASP: int = 0


@dataclass
class Demo:
    """A single demonstration is a list of single timestep data."""

    name: str
    demo_path: pathlib.Path
    timestep_data: List[TimestepData]
    # constraint timesteps are the timesteps where the robot reaches a constraint pose ...
    # can contain a saved nerf model, more importantly, contains paths to saved files
    timestep_annotation_data_dict: Dict[int, TimestepAnnotationData] = field(
        default_factory=dict
    )
    # the above is a centralized place to store data related to the individual timestep data for performing augs ... seems refactorable
    # can we put stuff directly in timestep_data? No, because it's a frozen datatype hence immutable. However,
    # TODO rename the above to:
    demo_annotations: Dict[int, TimestepAnnotationData] = field(default_factory=dict)
    demo_annotations_path: Optional[pathlib.Path] = None

    # I guess user can specify a single constraint info for each call ...
    # at some point, queryied user should be able to specify a list of constraint infos
    constraint_infos: Optional[List[ConstraintInfo]] = None

    # placeholder for proper instantiation
    env: Optional[Any] = None
    # this nerf object manager, in some ways, is related to timestep_data or timestep annotation data dict.
    # at the end of the day, it's a means to store information for each timestep; I think there should be another abstraction
    # where we have the i) original demo ii) constraint info + annotations + 3d reconstruction. Then, we can query the demo
    # at each timestep for any relevant information we want to then do augmentations.
    # the interface between a Demo object and the augmentation pipeline can be more clearly defined.
    _nerf_object_manager: NeRFObjectManager = NeRFObjectManager()

    # central place to store all 3D reconstructions: includes meshes, nerfs, 3d segmentation masks, etc.
    _reconstructions_manager: Optional[ReconstructionManager] = None

    def visualize(self):
        """Create a gif of the demo."""
        gif_dir = self.demo_path.parent / "gifs"
        gif_dir.mkdir(parents=True, exist_ok=True)
        create_gifs_from_h5py_file(self.demo_path, gif_dir / f"{self.name}.gif")

    def add_reconstruction_manager(self, rec_manager: ReconstructionManager):
        self._reconstructions_manager = rec_manager

    def add_constraint_infos(self, constraint_infos: List[ConstraintInfo]):
        self.constraint_infos = constraint_infos

    def generate_new_demo(self):
        return

    def save(self):
        """
        Save the demo.
        """
        pass

    def get_robot_env_cfg(self, t_start: int, t_end: int) -> List[RobotEnvConfig]:
        """Returns a list of robot env configs for the given timestep range (doesn't include t_end)."""
        return [self.timestep_data[t].robot_env_cfg for t in range(t_start, t_end)]

    def get_robot_poses_for_range(self, t_start: int, t_end: int) -> List[np.ndarray]:
        """
        @param t_start: start of the timestep range
        @param t_end: end of the timestep range
        @return: list of poses
        """
        return [self.timestep_data[t].robot_pose for t in range(t_start, t_end)]

    def get_obs_for_range(self, t_start: int, t_end: int) -> List[np.ndarray]:
        """
        @param t_start: start of the timestep range
        @param t_end: end of the timestep range
        @return: list of poses
        """
        return [
            {k: np.copy(v) for k, v in self.timestep_data[t].obs.items()}
            for t in range(t_start, t_end)
        ]

    def get_task_relev_obj_poses_for_range(
        self, t_start: int, t_end: int
    ) -> List[List[np.ndarray]]:
        """
        @param t_start: start of the timestep range
        @param t_end: end of the timestep range
        @return: list of poses
        """
        return [
            self.timestep_annotation_data_dict[t].task_relev_obj_poses
            for t in range(t_start, t_end)
        ]

    # TODO: we never really need a range that's not the entire constraint_info I think?
    # maybe keep this structure regardless; alt. let the method be inside constraint info?
    # seems weird though.

    # TODO(04/25/24): update to return the unified scene_representation object that contains both mesh and nerf and/or gsplat
    # constraint info should always be applied since it contains info about
    # which objects are task relevant at each timestep; alternative is to move
    # this info into the Demo.
    def get_task_relev_objs_for_range(
        self,
        t_start: int,
        t_end: int,
        constraint_info: ConstraintInfo,
        scene_representation_type: ReconstructionType = ReconstructionType.NeRF,
    ) -> List[Dict[str, Union[NeRFObject, GSplatObject, MeshObject]]]:
        """
        Returns the nerf objects corresponding to the original nerf (without transformations) applied.
        We apply transformations and get TransformationWrappers that wrap these NeRFObjects.

        @param t_start: start of the timestep range
        @param t_end: end of the timestep range
        """
        obj_repr_dict_list: List[Dict[str, NeRFObject]] = []

        assert (
            constraint_info.time_range
            == (
                t_start,
                t_end,
            )
        ), "constraint_info time range must match t_start and t_end (otherwise using wrong constraint info)"
        # constraint_info contains the timesteps to use for task_relev_objs
        for abs_t in range(t_start, t_end):
            obj_repr_dict: Dict[str, Union[NeRFObject, GSplatObject]] = {}
            for obj_name in constraint_info.get_task_relevant_obj_names(abs_t):
                image_ts = constraint_info.get_reconstruction_ts_for_t_and_obj(
                    abs_t, obj_name
                )
                assert (
                    self._reconstructions_manager is not None
                ), "reconstructions manager must be set"
                obj_repr = self._reconstructions_manager.get_reconstruction(
                    image_ts, obj_name, scene_representation_type
                )
                obj_repr_dict[obj_name] = obj_repr
            obj_repr_dict_list.append(obj_repr_dict)
        return obj_repr_dict_list

    def get_task_irrelev_objs_for_range(
        self, t_start: int, t_end: int
    ) -> List[List[np.ndarray]]:
        """
        @param t_start: start of the timestep range
        @param t_end: end of the timestep range
        @return: list of poses
        """
        return [
            self.timestep_annotation_data_dict[t].task_irrelev_objs
            for t in range(t_start, t_end)
        ]

    def get_task_irrelev_objs_poses_for_range(
        self, t_start: int, t_end: int
    ) -> List[np.ndarray]:
        """
        @param t_start: start of the timestep range
        @param t_end: end of the timestep range
        @return: list of poses
        """
        return [
            self.timestep_annotation_data_dict[t].task_irrelev_objs_pose
            for t in range(t_start, t_end)
        ]

    def get_curr_and_future_actions_for_range(
        self, t_start: int, t_end: int, num_future_actions: int = 0
    ) -> List[np.ndarray]:
        """
        @param t_start: start of the timestep range
        @param t_end: end of the timestep range
        @return: list of poses

        # t_end means don't include this timestep, maybe we want to include it though?
        """
        assert (
            num_future_actions + t_end <= len(self.timestep_data)
        ), f"num_future_actions={num_future_actions} + t_end={t_end} > len(self.timestep_data)={len(self.timestep_data)}"
        return [
            self.timestep_data[t].action
            for t in range(t_start, t_end + num_future_actions)
        ]
