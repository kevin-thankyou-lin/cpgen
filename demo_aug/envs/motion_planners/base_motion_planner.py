from dataclasses import dataclass
from functools import cached_property
from typing import Dict, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

from demo_aug.envs.base_env import MotionPlannerConfig
from demo_aug.envs.motion_planners.motion_planning_space import MotionPlanningSpace
from demo_aug.objects.robot_object import RobotObject


@dataclass(frozen=True)
class RobotEnvConfig:
    robot_joint_qpos: Optional[np.ndarray] = None
    robot_gripper_qpos: Optional[np.ndarray] = None
    robot_ee_pos: Optional[np.ndarray] = None
    robot_ee_quat_wxyz: Optional[np.ndarray] = None
    robot_ee_rot: Optional[np.ndarray] = None
    robot_base_pos: Optional[np.ndarray] = None
    robot_base_quat_wxyz: Optional[np.ndarray] = None
    robot_base_rot: Optional[np.ndarray] = None
    task_relev_obj_pos: Optional[np.ndarray] = None
    task_relev_obj_quat_wxyz: Optional[np.ndarray] = None
    task_relev_obj_rot: Optional[np.ndarray] = None
    task_relev_obj_pos_transf: Optional[np.ndarray] = None
    task_relev_obj_rot_transf: Optional[np.ndarray] = None
    task_irrelev_obj_pos: Optional[np.ndarray] = None
    task_irrelev_obj_quat_wxyz: Optional[np.ndarray] = None
    task_irrelev_obj_rot: Optional[np.ndarray] = None
    task_relev_obj_pos_nerf: Optional[np.ndarray] = None
    task_relev_obj_rot_nerf: Optional[np.ndarray] = None

    @cached_property
    def task_relev_obj_quat_xyzw(self) -> np.ndarray:
        assert (
            self.task_relev_obj_quat_wxyz is not None
            or self.task_relev_obj_rot is not None
        ), "task_relev_obj_quat_wxyz or task_relev_obj_rot must be specified"
        if self.robot_base_rot is None:
            return np.array(
                [
                    self.task_relev_obj_quat_wxyz[1],
                    self.task_relev_obj_quat_wxyz[2],
                    self.task_relev_obj_quat_wxyz[3],
                    self.task_relev_obj_quat_wxyz[0],
                ]
            )
        else:
            return R.from_matrix(self.robot_base_rot).as_quat()

    @property
    def robot_ee_quat_xyzw(self) -> np.ndarray:
        assert (
            self.robot_ee_quat_wxyz is not None or self.robot_ee_rot is not None
        ), "robot_ee_quat_wxyz or robot_ee_rot must be specified"
        if self.robot_ee_rot is None:
            return np.array(
                [
                    self.robot_ee_quat_wxyz[1],
                    self.robot_ee_quat_wxyz[2],
                    self.robot_ee_quat_wxyz[3],
                    self.robot_ee_quat_wxyz[0],
                ]
            )
        else:
            return R.from_matrix(self.robot_ee_rot).as_quat()

    @property
    def robot_base_quat_xyzw(self) -> np.ndarray:
        assert (
            self.robot_base_quat_wxyz is not None or self.robot_base_rot is not None
        ), "robot_base_quat_wxyz or robot_base_rot must be specified"
        if self.robot_base_rot is None:
            return np.array(
                [
                    self.robot_base_quat_wxyz[1],
                    self.robot_base_quat_wxyz[2],
                    self.robot_base_quat_wxyz[3],
                    self.robot_base_quat_wxyz[0],
                ]
            )
        else:
            return R.from_matrix(self.robot_base_rot).as_quat()

    @property
    def task_irrelev_obj_quat_xyzw(self) -> np.ndarray:
        assert (
            self.task_irrelev_obj_quat_wxyz is not None
            or self.task_irrelev_obj_rot is not None
        ), "task_irrelev_obj_quat_wxyz or task_irrelev_obj_rot must be specified"
        if self.task_irrelev_obj_rot is None:
            return np.array(
                [
                    self.task_irrelev_obj_quat_wxyz[1],
                    self.task_irrelev_obj_quat_wxyz[2],
                    self.task_irrelev_obj_quat_wxyz[3],
                    self.task_irrelev_obj_quat_wxyz[0],
                ]
            )
        else:
            return R.from_matrix(self.task_irrelev_obj_rot).as_quat()


@dataclass
class Trajectory:
    robot_joint_qpos: np.ndarray
    robot_gripper_qpos: np.ndarray
    robot_ee_pos: np.ndarray
    robot_ee_quat_wxyz: np.ndarray
    robot_base_pos: np.ndarray
    robot_ee_quat_wxyz: np.ndarray
    task_relev_obj_pos: np.ndarray
    task_relev_obj_quat_wxyz: np.ndarray
    task_irrelev_obj_pos: np.ndarray
    task_irrelev_obj_quat_wxyz: np.ndarray


class BaseMotionPlanner:
    def __init__(self, robot_obj: RobotObject, motion_planner_cfg: MotionPlannerConfig):
        self.robot_obj = robot_obj
        self.motion_planner_cfg = motion_planner_cfg

    def get_optimal_trajectory(
        self,
        start_cfg: RobotEnvConfig,
        end_cfg: RobotEnvConfig,
        motion_planning_space: MotionPlanningSpace,
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Compute the optimal trajectory from the start configuration to the end configuration
        based on the given RobotEnvConfigs.

        Note that end_cfg could be a set!

        Return dictionary containing the optimal trajectory. If the function cannot
        calculate the optimal trajectory, return None while logging the error.

        Reasons for failure include:
        - goal cfg being infeasible (i.e. not valid IK solutions). This failure is reasonable.
            That said, a learning system should be able to somehow internalize that this
            goal is infeasible and not try to reach it, perhaps via a learnt Q function.
        - start cfg being infeasible (i.e. not valid IK solutions). This failure is also reasonable
            assuming we're randomly sampling start cfgs.
        - other failures that are not reasonable but unideal, such as having a collision-free motion planner
            PRM timing out. Policy isn't supervised in these examples, so we can't expect it to learn
            reasonable behavior in these cases.

            We should either (1) figure out what to do in a case by case basis, or (2) try to
            make the motion planner run for more iterations.

            For the toy tasks we're considering, we can assume that if start and goal are both feasible,
            then it's possible to get a collision free motion plan. When is it the case, in robot manipulation,
            that it's not possible to reach two feasible configurations without collision?

            Sometimes, the task is achievable if we first
            move things out of the way. Or, if we're allowed to collide / push some objects out of the way.

        # TODO(KLIN): remove these since code will change ...
        Implementation thoughts:
            (1) motion planning space returns a few queries including:
                - getting EEF poses for a given joint configuration
                - getting joint configurations for a given EEF pose i.e. IK
                - minimum distance between EEF and environment
                - whether a given joint configuration is in collision

        Assumption:
            Starting configuration fully specifies the robot state i.e. all joint angles and gripper state
            the goal, however, could be specified in terms of the end effector pose in that case, we can
            flexibly choose how to do motion planning.
        """
        raise NotImplementedError
