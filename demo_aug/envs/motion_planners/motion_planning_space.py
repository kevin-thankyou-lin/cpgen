import datetime
import logging
import pathlib
import time
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pydrake.planning as planning
from pydrake.common.eigen_geometry import Quaternion
from pydrake.geometry import (
    CollisionFilterDeclaration,
    GeometrySet,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    Rgba,
    Role,
    Sphere,
    StartMeshcat,
)
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.multibody.inverse_kinematics import InverseKinematics
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.tree import (
    BodyIndex,
    FixedOffsetFrame,
    ModelInstanceIndex,
    RevoluteJoint,
)
from pydrake.solvers import SnoptSolver, SolutionResult
from scipy.spatial.transform import Rotation as R

from demo_aug.configs.robot_configs import PandaGripperNameToFrameInfo
from demo_aug.objects.nerf_object import MeshPaths
from demo_aug.utils.drake_utils import (
    get_collision_geometry_ids_by_model_instance_index,
)
from demo_aug.utils.mathutils import interpolate_poses, rotation_matrix_distance
from demo_aug.utils.run_script_utils import retry_on_exception
from demo_aug.utils.snopt_utils import (
    SNOPT_SOLVER_MAX_OPTIMALITY_VALUE,
    extract_feasiblity_optimality_values_from_snopt_log,
)


@dataclass
class ObjectInitializationInfo:
    X_parentframe_obj: np.ndarray
    mesh_paths: MeshPaths
    weld_to_ee: bool
    parent_frame_name: Optional[str] = (
        None  # Optional: specify the parent frame; if None, use world frame
    )

    def __post_init__(self):
        if self.weld_to_ee:
            assert (
                self.parent_frame_name is not None
            ), "parent_frame_name must be specified if weld_to_ee is True"


class IKType(Enum):
    X_EE_TO_Q_ROBOT = 0
    P_KPTS_TO_X_EE = 1


class MotionPlanningSpace:
    """
    Sets up environment and robot; can be queried for collisions and distance to collisions.

    Subclasses might involve special effects on the robot, e.g. robot arm with gripper where
    gripper joint position is fixed.

    Note: fix gripper wrapper should always implement any method that calls .bound() because,
    otherwise, the .bounds() property will be called on the wrapped object. Eventually, add tests,
    or don't use this fixed gripper wrapper?
    """

    @property
    def bounds(self) -> Tuple[np.ndarray]:
        """
        Returns the bounds of the motion planning space.
        For example, for a robot arm, the bounds would be the robot's joint limits.
        """
        raise NotImplementedError

    def get_min_dist_to_env(self, q: np.ndarray) -> float:
        """
        Return the minimum distance between the end effector and the environment.
        """
        raise NotImplementedError

    def is_collision(self, q: np.ndarray) -> bool:
        """
        Return whether the robot is in collision given the joint configuration.
        """
        raise NotImplementedError

    def sample(self) -> np.ndarray:
        """
        Sample a configuration from the motion planning space.
        """
        raise NotImplementedError

    def is_visible(self, q1: np.ndarray, q2: np.ndarray) -> bool:
        """
        Return whether the 'edge' from q1 to q2 is collision free.
        """
        raise NotImplementedError

    def interpolate(self, q1: np.ndarray, q2: np.ndarray, u: float) -> np.ndarray:
        """
        Interpolate between q1 and q2.
        """
        raise NotImplementedError

    def distance(self, q1: np.ndarray, q2: np.ndarray) -> float:
        """
        Return the distance between the two configurations.
        """
        raise NotImplementedError


class CuroboMotionPlanningSpace(MotionPlanningSpace):
    def __init__(
        self,
        drake_package_path: str,
        task_irrelev_obj_url: str = "package://models/assets/arenas/table_arena.xml",
        robot_base_pos: Optional[np.ndarray] = None,
        robot_base_quat_wxyz: Optional[np.ndarray] = None,
        obj_to_init_info: Optional[Dict[str, ObjectInitializationInfo]] = None,
    ):
        # Save some information on the planning space that VAMP (and maybe others) need later
        self.environment_xml_url = task_irrelev_obj_url
        self.robot_base_pos = robot_base_pos
        self.robot_base_quat_wxyz = robot_base_quat_wxyz
        self.package_root_path = drake_package_path
        self.obj_to_init_info = obj_to_init_info


class DrakeMotionPlanningSpace(MotionPlanningSpace):
    """For drake, involves: plant, builder, scene_graph. For viz, involves meshcat?

    # I should be using RobotDiagramBuilder it seems:

    https://drake.mit.edu/pydrake/pydrake.planning.html?highlight=robot%20diagram#pydrake.planning.RobotDiagram

    For collision checking docs, see:
    https://drake.mit.edu/pydrake/pydrake.planning.html?highlight=collision%20checker#pydrake.planning.CollisionChecker

    mutable_plant_context
    """

    def __init__(
        self,
        drake_package_path: str,
        task_irrelev_obj_url: str = "package://models/assets/arenas/table_arena.xml",
        add_robot: bool = True,
        add_robot_hand: bool = False,
        gripper_type: Literal["panda_hand", "robotiq85"] = "panda_hand",
        name_to_frame_info: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
        robot_base_pos: Optional[np.ndarray] = None,
        robot_base_quat_wxyz: Optional[np.ndarray] = None,
        task_relev_obj_pos_nerf: Optional[List[np.ndarray]] = None,
        task_relev_obj_rot_nerf: Optional[List[np.ndarray]] = None,
        obj_to_init_info: Optional[Dict[str, ObjectInitializationInfo]] = None,
        set_gripper_dim_to_one: bool = False,
        edge_step_size: float = 0.002,
        env_collision_padding: float = 0.0075,
        env_dist_factor: float = 0.0,
        env_influence_distance: float = 0.0,
        view_meshcat: bool = False,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        input_config_frame: Optional[Literal["EE_SITE"]] = None,
    ):
        """
        Args:
            drake_package_path: str
                https://drake.mit.edu/doxygen_cxx/classdrake_1_1multibody_1_1_package_map.html
                Path to drake package file. Used to map ROS paths to their full paths.
            task_irrelev_obj_url: drake 'url' path task irrelevant object meshes
            set_gripper_dim_to_one: bool
                Relevant only for franka robot xml using a parallel jaw gripper that has 2DOF for gripper
        """
        if env_collision_padding > 0:
            import ipdb

            ipdb.set_trace()
        if name_to_frame_info is None:
            print(
                f"Using default name_to_frame_info {PandaGripperNameToFrameInfo.default()}"
            )
            name_to_frame_info = (
                PandaGripperNameToFrameInfo.default().name_to_frame_info
            )

        if bounds is None:
            self._bounds = None
        else:
            assert len(bounds) == 2, "bounds must be a tuple of length 2"
            assert (
                bounds[0].shape == bounds[1].shape
            ), "bounds must be a tuple of arrays of the same shape"
            self._bounds = bounds

        self.ee_obs_frame_name = self.input_config_frame = input_config_frame

        assert (
            sum([add_robot, add_robot_hand]) == 1
        ), "Can only add exactly one of robot or robot hand.If robot, assume hand is part of robot model if desired."

        # Save some information on the planning space that VAMP (and maybe others) need later
        self.environment_xml_url = task_irrelev_obj_url
        self.robot_base_pos = robot_base_pos
        self.robot_base_quat_wxyz = robot_base_quat_wxyz
        self.package_root_path = drake_package_path

        (
            robot_diagram,
            robot_model_instances,
            meshcat,
            visualizer,
            collision_visualizer,
        ) = DrakeMotionPlanningSpace.setup_env(
            drake_package_path=drake_package_path,
            task_irrelev_obj_url=task_irrelev_obj_url,
            add_robot=add_robot,
            add_robot_hand=add_robot_hand,
            gripper_type=gripper_type,
            name_to_frame_info=name_to_frame_info,
            robot_base_pos=robot_base_pos,
            robot_base_quat_wxyz=robot_base_quat_wxyz,
            task_relev_obj_pos_nerf=task_relev_obj_pos_nerf,
            task_relev_obj_rot_nerf=task_relev_obj_rot_nerf,
            obj_to_init_info=obj_to_init_info,
            view_meshcat=view_meshcat,
        )

        plant = robot_diagram.plant()
        scene_graph = robot_diagram.scene_graph()

        if plant is None or scene_graph is None:
            # TODO(klin): raise a more specific error + pass the error message from setup_env to here
            raise ValueError("Failed to set up environment.")

        self.plant = plant
        self.scene_graph = scene_graph

        self.context = robot_diagram.CreateDefaultContext()
        self.plant_context = robot_diagram.mutable_plant_context(self.context)

        self.add_robot = add_robot
        self.add_robot_hand = add_robot_hand
        self.name_to_frame_info = name_to_frame_info
        self.obj_to_init_info = obj_to_init_info
        self.set_gripper_dim_to_one = set_gripper_dim_to_one

        self.robot_diagram = robot_diagram
        self.robot_model_instances = robot_model_instances

        self.collision_checker = self._make_scene_graph_collision_checker(
            robot_diagram,
            robot_model_instances,
            edge_step_size=edge_step_size,
            env_collision_padding=env_collision_padding,
        )
        self.collision_checker_context = (
            self.collision_checker.MakeStandaloneModelContext()
        )

        if add_robot_hand:
            robot_eef_geometry_ids = get_collision_geometry_ids_by_model_instance_index(
                plant, robot_model_instances[0]
            )
            robot_hand_geometry_set = GeometrySet(robot_eef_geometry_ids)
            collision_filter_manager = scene_graph.collision_filter_manager(
                self.collision_checker_context.scene_graph_context()
            )
            collision_filter_manager.Apply(
                CollisionFilterDeclaration().ExcludeWithin(robot_hand_geometry_set)
            )

        self.edge_step_size = edge_step_size
        self.env_dist_factor = env_dist_factor
        self.env_influence_distance = env_influence_distance

        self.meshcat = meshcat
        self.visualizer = visualizer
        self.collision_visualizer = collision_visualizer

        # visualize
        if visualizer is not None:
            visualizer_context = visualizer.GetMyContextFromRoot(self.context)
            visualizer.ForcedPublish(visualizer_context)
            collision_context = collision_visualizer.GetMyContextFromRoot(self.context)
            collision_visualizer.ForcedPublish(collision_context)

        self.robot_type = "arm_hand" if add_robot else "hand"
        if self.robot_type == "arm_hand":
            self.non_gripper_dim = self.plant_arm_dim = 7
        else:
            self.non_gripper_dim = 7  # pos + quat
        self.plant_gripper_dim = (
            len(plant.GetPositionUpperLimits()) - self.non_gripper_dim
        )

        if gripper_type == "panda_hand":
            self.gripper_dim = 2  # input and output of class's gripper dim
        elif gripper_type == "robotiq85":
            self.gripper_dim = 1
        else:
            raise ValueError(f"Unknown hand type {gripper_type}")

    def set_edge_step_size(self, edge_step_size: float) -> None:
        self.edge_step_size = edge_step_size
        self.collision_checker.set_edge_step_size(edge_step_size)

    def set_start_cfg(self, q: np.ndarray) -> None:
        self._start_cfg = q

    def set_goal_cfg(self, q: np.ndarray) -> None:
        self._goal_cfg = q

    def distance(
        self,
        q1: np.ndarray,
        q2: np.ndarray,
        dist_type: Literal["end_effector_l2", "configuration_l2"] = "configuration_l2",
    ) -> float:
        """
        Return the distance between the two configurations.
        """
        if dist_type == "end_effector_l2":
            return self._end_effector_distance(q1, q2)
        elif dist_type == "configuration_l2":
            return self._configuration_distance(q1, q2)
        else:
            raise ValueError(f"Unknown distance type {dist_type}")

    @cached_property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        For now, sampling in 9DOF ...

        Speed up later by sampling only in 1DOF for gripper instead of 2DOF
        """
        if self._bounds is None:
            lower = self.plant.GetPositionLowerLimits()  # [:8]
            upper = self.plant.GetPositionUpperLimits()  # [:8]
            if self.set_gripper_dim_to_one:
                # Note: not sure how to check if the gripper is a 2DOF gripper (when it really should be 1DOF)
                lower = np.delete(lower, -2)
                upper = np.delete(upper, -2)
        else:
            lower, upper = self._bounds
        return lower, upper

    def is_collision(
        self,
        q_val: np.ndarray,
        input_config_frame: Optional[Literal["EE_SITE"]] = None,
        viz: bool = False,
    ) -> bool:
        """
        Return whether the robot is in collision given the joint configuration.
        """
        # enable parallelism if it makes sense
        # self.collision_checker.CheckConfigsCollisionFree([q, q], parallelize=True)

        # TODO: handle the frames correctly ...
        input_frame = (
            input_config_frame
            if input_config_frame is not None
            else self.input_config_frame
        )
        if input_frame is not None:
            # TODO(klin): make separate MotionPlanningSpace class; this function only makes sense for end effector motion planning space
            # assert "panda_hand" in input_frame, "input_frame must be a panda hand frame"
            # test this @KL: visually check if collision or not
            X_W_input = RigidTransform(
                Quaternion(q_val[0:4] / np.linalg.norm(q_val[0:4])), q_val[4:7]
            )
            X_base_input_info = self.name_to_frame_info[input_frame]
            X_base_input = RigidTransform(
                Quaternion(X_base_input_info.offset_quat_wxyz),
                X_base_input_info.offset_pos,
            )
            X_W_base = X_W_input.multiply(X_base_input.inverse())

            # q_hand ordering [quat_wxyz, pos, grip]; unclear how to check though
            q_val = np.concatenate(
                [
                    X_W_base.rotation().ToQuaternion().wxyz(),
                    X_W_base.translation(),
                    q_val[7:],
                ]
            )

            if self.visualizer is not None and (
                viz
                or not self.collision_checker.CheckContextConfigCollisionFree(
                    model_context=self.collision_checker_context, q=q_val
                )
            ):
                self.plant.SetPositions(self.plant_context, q_val)
                self.visualizer.ForcedPublish(
                    self.visualizer.GetMyContextFromRoot(self.context)
                )
                self.collision_visualizer.ForcedPublish(
                    self.collision_visualizer.GetMyContextFromRoot(self.context)
                )

        return not self.collision_checker.CheckContextConfigCollisionFree(
            model_context=self.collision_checker_context, q=q_val
        )

    def is_visible(
        self, q1: np.ndarray, q2: np.ndarray, input_config_frame: Optional[str] = None
    ) -> bool:
        """
        Return whether the 'edge' from q1 to q2 is collision free.
        """
        # TODO(klin): decide whether to use adaptive edge check sizes after using hand motion planning
        # if self._end_effector_distance(q1, self._goal_cfg) < 0.03 or self._end_effector_distance(q2, self._goal_cfg) < 0.03:
        #     self.set_edge_step_size(0.0025)
        # is_edge_collision_free = self.collision_checker.CheckContextEdgeCollisionFree(self.collision_checker_context, q1, q2)
        # self.set_edge_step_size(0.04)
        # # I see what the issue is w.r.t. edge collision checks: for checking the whole arm; the edge resolution is wild
        raise NotImplementedError(
            "Need to implement is_visible for DrakeMotionPlanningSpace"
        )
        # TODO(klin): the base drake motion planning space isn't necessarily using poses!
        input_frame = (
            input_config_frame
            if input_config_frame is not None
            else self.input_config_frame
        )
        res = self._convert_pose_frame([q1, q2], input_frame)
        return self.collision_checker.CheckContextEdgeCollisionFree(
            self.collision_checker_context, res[0], res[1]
        )

    def interpolate(
        self,
        q1: np.ndarray,
        q2: np.ndarray,
        u: float,
        interp_type: str = "configuration",
    ) -> np.ndarray:
        """
        Interpolate between q1 and q2.
        """
        if isinstance(q1, list):
            q1 = np.array(q1)
        if isinstance(q2, list):
            q2 = np.array(q2)

        if interp_type == "configuration":
            return q1 + u * (q2 - q1)
        elif interp_type == "end_effector":
            raise NotImplementedError("Need to implement interpolate for end effector")
        elif interp_type == "end_effector_dist":
            raise NotImplementedError(
                "Need to implement interpolate for end effector dist"
            )
        else:
            raise ValueError(f"Unknown interp_type {interp_type}")

    def extend_configs(
        self,
        q1: np.ndarray,
        q2: np.ndarray,
        resolution: float = 0.004,
        extend_type: Literal[
            "configuration", "end_effector", "end_effector_dist"
        ] = "end_effector",
    ) -> List[np.ndarray]:
        """
        TODO(klin): this function belongs in motion_planning_space.py

        Given two configurations, q1 and q2, return a list of configurations that are at most resolution apart.
        Always include q1 in the list; include q2 if it is non-zero distance from q2 under the extend_type distance.

        extend_type:
            - configuration: interpolate between q1 and q2 directly in configuration space
            - end_effector: interpolate between q1 and q2 in end effector space,
                convert to configuration space using IK
            - end_effector_dist: interpolate between q1 and q2 in configuration space,
                but use the distance between end effector positions to determine number of intermediate configurations
        """
        assert resolution > 0, "resolution used in extend_configs must be positive"

        if extend_type == "end_effector_dist":
            if (dist := self.distance(q1, q2, "end_effector_l2")) == 0:
                return [q1]
            step = resolution / dist
            qs = [
                self.interpolate(q1, q2, u, interp_type=extend_type)
                for u in np.arange(0, 1 + 1e-4, step)
            ]
        elif extend_type == "end_effector":
            if (dist := self.distance(q1, q2, "end_effector_l2")) == 0:
                return [q1]  # redundant qs: remove q2
            elif dist <= resolution:
                return [q1, q2]

            X_1 = self.get_end_effector_pose(q1)
            X_2 = self.get_end_effector_pose(q2)

            num_poses = int(np.ceil(dist / resolution)) + 1
            assert (
                num_poses > 2
            ), "num_poses must be greater than 2 to return more than just q1 and q2"

            Xs = interpolate_poses(X_1, X_2, num_poses)
            qs = [
                self.inverse_kinematics(X, q2[7:], IKType.X_EE_TO_Q_ROBOT, q2)[0]
                for X in Xs
            ]
        else:
            raise NotImplementedError(f"extend_type {extend_type} not implemented")

        qs = self._convert_qs_to_valid_output(qs)
        return qs

    def _convert_qs_to_valid_output(self, qs: List[np.ndarray]) -> List[np.ndarray]:
        """
        Convert the qs generated using computation with the drake motion planning space to qs with the expected output dim.
        """
        res = []
        for q in qs:
            if self.plant_gripper_dim != self.gripper_dim:
                if self.plant_gripper_dim == 6 and self.gripper_dim == 1:
                    # take the last element of q and use it as the gripper value
                    q = np.concatenate([q[: self.non_gripper_dim], np.array([q[-1]])])

            res.append(q)
        return res

    def get_end_effector_pose(
        self,
        q: np.ndarray,
        frame_name: Optional[str] = None,
        base_frame_name: Optional[str] = None,
    ) -> np.ndarray:
        """
        Return the pose of the end effector given the joint configuration.
        """
        if frame_name is None:
            frame_name = self.ee_obs_frame_name

        q = self._construct_valid_input_q_value(q)
        return self.get_poses(q, [frame_name], base_frame_name)[0]

    def _construct_valid_input_q_value(self, q: np.ndarray) -> np.ndarray:
        """
        Construct a valid q value by assuming the gripper dim is incorrect then setting gripper values to be zeros.
        """
        # if gripper q dim is too small, randomly set the values to be queryable
        if len(q) != len(self.plant.GetPositionUpperLimits()):
            q = np.concatenate(
                [q[: self.non_gripper_dim], np.zeros(shape=self.plant_gripper_dim)]
            )
            # ideally does more than randomly setting the value as zero ...
        return q

    def get_poses(
        self,
        q: np.ndarray,
        frame_names: List[str],
        base_frame_name: Optional[str] = None,
    ) -> List[np.ndarray]:
        """
        Return poses of frames at joint configuration q.
        """
        q = self._construct_valid_input_q_value(q)

        if base_frame_name is None:
            base_frame = self.plant.world_frame()
        else:
            base_frame = self.plant.GetFrameByName(base_frame_name)

        frames = [self.plant.GetFrameByName(frame) for frame in frame_names]
        self.plant.SetPositions(self.plant_context, q)
        return [
            frame.CalcPose(self.plant_context, base_frame).GetAsMatrix4()
            for frame in frames
        ]

    def _end_effector_distance(
        self, q1: np.ndarray, q2: np.ndarray, rot_factor: float = 0.1
    ) -> float:
        """
        Return the distance between the two end effectors given the joint configurations.

        TODO(klin): speed up by factor of 2-3x by doing l2 on N pre-selected keypoints on end effector.
        Note: unclear if ^'s actually 2-3x because would need to read out N CalcPosesInWorld values
        """
        ee_site_frame = self.plant.GetFrameByName(self.ee_obs_frame_name)

        self.plant.SetPositions(self.plant_context, q1)
        X_ee_1 = ee_site_frame.CalcPoseInWorld(self.plant_context).GetAsMatrix4()
        self.plant.SetPositions(self.plant_context, q2)
        X_ee_2 = ee_site_frame.CalcPoseInWorld(self.plant_context).GetAsMatrix4()

        pos_dist = np.linalg.norm(X_ee_1[:3, 3] - X_ee_2[:3, 3])
        rot_dist = rotation_matrix_distance(X_ee_1[:3, :3], X_ee_2[:3, :3])
        return pos_dist + rot_factor * rot_dist

    def _robot_clearance(
        self, q_val: np.ndarray, influence_distance: float = 0.1
    ) -> float:
        """
        Return the distance between the robot and the environment.

        Intended usage: called in distance() to add robot-env distance as a factor. Unfortunately,
        distance() gets called 60k times so per motion plan (for hand only motion planning) so this function is too slow.
        """
        if not hasattr(self, "clearance_cache"):
            self.clearance_cache = {}

        if q_val.tobytes() in self.clearance_cache:
            return self.clearance_cache[q_val.tobytes()]

        robot_clearance = self.collision_checker.CalcRobotClearance(
            q=q_val, influence_distance=influence_distance
        )
        min_clearance_distance = (
            robot_clearance.distances().min()
            if robot_clearance.distances().size > 0
            else influence_distance
        )

        min_clearance_distance = np.clip(min_clearance_distance, 0, influence_distance)
        self.clearance_cache[q_val.tobytes()] = min_clearance_distance
        return min_clearance_distance

    def inverse_kinematics(
        self,
        X_ee: np.ndarray,
        q_grip: np.ndarray,
        ik_type: IKType,
        q_init: Optional[np.ndarray] = None,
        constrain_orientation: bool = True,
        min_dist: float = 0.00,
        min_dist_thresh: float = 0.001,
        debug: bool = True,
        viz_all: bool = False,
        n_trials: int = 1,
        save_to_file: bool = True,
    ) -> np.ndarray:
        """
        Given constraints, get generalized coordinates (i.e. set of parameters used to
        represent the state of a system in a configuration space).

        Usually, the above translates to: given end effector position (or pose), get joint angles.
        This function handles the general case of IK and retrieves generalized coordinates.

        How to take in the constraints corresponding to gripper points? See drake motion planner.
        """
        plant, plant_context = self.plant, self.plant_context
        if q_init is None:
            q_init = self.plant.GetPositions(plant_context)

        X_ee = RigidTransform(X_ee)
        ik = InverseKinematics(plant, plant_context)
        prog = ik.get_mutable_prog()

        solver_results_dir = pathlib.Path(
            "ik_results"
        ) / datetime.datetime.now().strftime("%Y_%m_%d-%H")
        solver_results_dir.mkdir(parents=True, exist_ok=True)
        if ik_type == IKType.X_EE_TO_Q_ROBOT:
            gripper_frame = plant.GetFrameByName(self.ee_obs_frame_name)
            ik.AddPositionConstraint(
                gripper_frame,
                [0, 0, 0],
                plant.world_frame(),
                X_ee.translation(),
                X_ee.translation(),
            )
            if constrain_orientation:
                ik.AddOrientationConstraint(
                    gripper_frame,
                    RotationMatrix(),
                    plant.world_frame(),
                    X_ee.rotation(),
                    0.0,
                )

            # avoid q because annoying when debugging w/ ipdb (ipdb> q leads to quitting debug session)
            q_var = ik.q()
            # prog.AddQuadraticErrorCost(np.identity(len(q_var)), q_init, q_var)
        elif ik_type == IKType.P_KPTS_TO_X_EE:
            raise NotImplementedError("Need to implement IKType.P_KPTS_TO_X_EE")

        if min_dist > 0 or min_dist_thresh > 0:
            ik.AddMinimumDistanceConstraint(min_dist, min_dist_thresh)

        # TODO: update the hardcoding!
        non_gripper_len = 7
        # prog.AddBoundingBoxConstraint(
        #     np.abs(q_grip[non_gripper_len:]),
        #     np.abs(q_grip[non_gripper_len:]),
        #     q_var[non_gripper_len:],
        # )
        prog.AddBoundingBoxConstraint(
            np.abs(q_grip),
            np.abs(q_grip),
            q_var[non_gripper_len:],
        )
        snopt_solver = SnoptSolver()

        if save_to_file:
            current_time = datetime.datetime.now().strftime("%b_%d_%H_%M_%S_%f")
            file_path: pathlib.Path = solver_results_dir / f"ik_{current_time}.snopt"
            prog.SetSolverOption(snopt_solver.solver_id(), "Print file", str(file_path))

        prog.SetSolverOption(snopt_solver.solver_id(), "Major iterations limit", 500)
        prog.SetSolverOption(snopt_solver.solver_id(), "Iterations limit", 500)
        prog.SetSolverOption(snopt_solver.solver_id(), "Time limit", 4)
        prog.SetSolverOption(snopt_solver.solver_id(), "Timing level", 2)

        for trial_idx in range(n_trials):
            if trial_idx == 0:
                q_guess = q_init
            else:
                q_guess = q_init + np.random.uniform(-1, 1, size=q_init.shape)
                q_guess = np.clip(
                    q_guess,
                    self.plant.GetPositionLowerLimits(),
                    self.plant.GetPositionUpperLimits(),
                )

            prog.SetInitialGuess(q_var, q_guess)
            result = snopt_solver.Solve(prog)

            if result.is_success():
                break
            else:
                if save_to_file:
                    feasibility_value, optimality_value, is_acceptably_feasible = (
                        extract_feasiblity_optimality_values_from_snopt_log(file_path)
                    )
                    file_path.unlink()
                    logging.info(
                        f"Feasibility: {feasibility_value}, Optimality: {optimality_value}"
                    )
                    if (
                        is_acceptably_feasible
                        and optimality_value < SNOPT_SOLVER_MAX_OPTIMALITY_VALUE
                    ):
                        logging.info(
                            "Feasibility and optimal values are acceptable. Using current result."
                        )
                        result.set_solution_result(SolutionResult.kSolutionFound)
                        break

                if self.visualizer is not None and (viz_all or debug):
                    print(f"meshcat port is: {self.meshcat.port()}")
                    q_res = result.GetSolution()
                    self.plant.SetPositions(self.plant_context, q_res)
                    self.visualizer.ForcedPublish(
                        self.visualizer.GetMyContextFromRoot(self.context)
                    )
                    self.collision_visualizer.ForcedPublish(
                        self.collision_visualizer.GetMyContextFromRoot(self.context)
                    )
                    # /scr/thankyou/autom/demo-aug/demo_aug/models/assets/task_relevant/nerf_outputs/nerfacto-/scr/thankyou/autom/demo-aug/data/real_world_recordings/2024-06-13-00:51:14-glass-holder-start-near-agentview-good/nerfacto/2024-06-14_152057/meshes/

                    X_ee_res = self.get_end_effector_pose(q_res)
                    X_ee_init = self.get_end_effector_pose(q_init)
                    for X, X_name in [
                        (X_ee, "X_ee_goal"),
                        (X_ee_init, "X_ee_init"),
                        (X_ee_res, "X_ee_res"),
                    ]:
                        # view the position of X_ee and X_ee_init in meshcat
                        self.meshcat.SetObject(
                            X_name, Sphere(0.02), rgba=Rgba(0.9, 0.1, 0.1, 1)
                        )
                        self.meshcat.SetTransform(
                            X_name,
                            RigidTransform(
                                X[:3, 3]
                                if isinstance(X, np.ndarray)
                                else X.translation()
                            ),
                        )

                    print(
                        f"Failed to solve IK with min_dist {min_dist} and min_dist_thresh {min_dist_thresh} on trial"
                        f" {trial_idx}."
                    )

        q_res = result.GetSolution()

        if self.visualizer is not None and viz_all:
            print(f"meshcat port is: {self.meshcat.port()}")
            q_res = result.GetSolution()
            self.plant.SetPositions(self.plant_context, q_res)
            self.visualizer.ForcedPublish(
                self.visualizer.GetMyContextFromRoot(self.context)
            )
            self.collision_visualizer.ForcedPublish(
                self.collision_visualizer.GetMyContextFromRoot(self.context)
            )

            X_ee_res = self.get_end_effector_pose(q_res)
            X_ee_init = self.get_end_effector_pose(q_init)
            for X, X_name in [
                (X_ee, "X_ee"),
                (X_ee_init, "X_ee_init"),
                (X_ee_res, "X_ee_res"),
            ]:
                # view the position of X_ee and X_ee_init in meshcat
                self.meshcat.SetObject(
                    X_name, Sphere(0.02), rgba=Rgba(0.9, 0.1, 0.1, 1)
                )
                self.meshcat.SetTransform(
                    X_name,
                    RigidTransform(
                        X[:3, 3] if isinstance(X, np.ndarray) else X.translation()
                    ),
                )

        [q_res] = self._convert_qs_to_valid_output([q_res])
        return q_res, result.is_success()

    def set_q(self, q: np.ndarray):
        """
        Set the joint configuration of the robot.
        """
        if self.visualizer is not None:
            self.plant.SetPositions(self.plant_context, q)
            self.visualizer.ForcedPublish(
                self.visualizer.GetMyContextFromRoot(self.context)
            )
            self.collision_visualizer.ForcedPublish(
                self.collision_visualizer.GetMyContextFromRoot(self.context)
            )

    def sample_neighborhood(self, q: np.ndarray, r: float) -> np.ndarray:
        """
        TODO(klin): not implemented properly given r. E.g. the bounds have different scales and some bounds e.g. gripper would be saturated first.
        # awkward because also used in get_init_samples()
        Sample a configuration from the neighborhood of q.
        2. have the gripper 1D or just fix it's value so only sample gripper
        """
        # raise NotImplementedError(
        #     "sample_neighborhood not implemented properly given r. E.g. the bounds have different scales and some bounds"
        #     " e.g. gripper would be saturated first."
        # )
        sample = q + np.random.uniform(-r, r, size=q.shape)
        return np.clip(sample, self.bounds[0], self.bounds[1])

    @staticmethod
    def _configuration_distance(q1: np.ndarray, q2: np.ndarray):
        """Handles joint angles that wrap around."""

        # Compute the absolute difference between the joint angles
        diff = np.abs(q1 - q2)

        # Handle joint angles that wrap around (e.g., angles in radians)
        wrapped_diff = np.minimum(diff, 2 * np.pi - diff)

        # Compute the Euclidean distance between the wrapped joint angles
        distance = np.linalg.norm(wrapped_diff)

        return distance

    @staticmethod
    @retry_on_exception(max_retries=5, retry_delay=1, exceptions=(RuntimeError,))
    def AddTaskRelevObj(
        parser: Parser,
        plant: MultibodyPlant,
        X_parentframe_obj: RigidTransform,  # reduces to X_EE_NERF or X_EE_MESH in this case
        obj_url: str,
        parent_frame_name: Optional[str] = None,
    ) -> ModelInstanceIndex:
        """
        Add a task relevant object to the plant and weld it to some parent frame.
        """
        if "package://" in obj_url:
            obj_model_index = parser.AddModelsFromUrl(str(obj_url))[0]
        else:
            obj_model_index = parser.AddModels(str(obj_url))[0]

        if parent_frame_name is None:
            parent_frame = plant.world_frame()
        else:
            parent_frame = plant.GetFrameByName(parent_frame_name)
        plant.WeldFrames(
            parent_frame,
            plant.GetFrameByName(plant.GetModelInstanceName(obj_model_index)),
            X_parentframe_obj,
        )

        return obj_model_index

    @staticmethod
    def setup_env(
        drake_package_path: str,
        task_irrelev_obj_url: str = "package://models/assets/arenas/table_arena.xml",
        add_robot: bool = True,
        add_robot_hand: bool = False,
        gripper_type: Literal["panda_hand", "robotiq85"] = "panda_hand",
        name_to_frame_info: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
        robot_base_pos: Optional[np.ndarray] = None,
        robot_base_quat_wxyz: Optional[np.ndarray] = None,
        task_relev_obj_pos_nerf: Optional[List[np.ndarray]] = None,
        task_relev_obj_rot_nerf: Optional[List[np.ndarray]] = None,
        obj_to_init_info: Optional[Dict[str, ObjectInitializationInfo]] = None,
        view_meshcat: bool = False,
        edge_step_size: float = 0.01,
    ) -> Tuple:
        """
        Initializes the environment for the Drake Motion Planner.

        Args:
            robot_base_pos (numpy.ndarray): The position of the robot's base.
            robot_base_quat_wxyz (numpy.ndarray): The orientation of the robot's base.
            task_relev_obj_pos_nerf (numpy.ndarray): The position of the relevant object in the task.
            task_relev_obj_rot_nerf (numpy.ndarray): The orientation of the relevant object in the task.
            task_relev_obj_path (pathlib.Path): The path to the object relevant to the task. Either have
                absolute (or maybe relative path), or use package://... syntax for drake model loading

        Returns:
            Tuple: A tuple containing the initialized plant, builder, and scene graph.

        """
        builder = planning.RobotDiagramBuilder()
        # builder = planning.RobotDiagramBuilder(time_step=1)

        parser = builder.parser()
        plant = builder.plant()

        # TODO: klin: add option for discrete solver by default for mimic joint
        # https://chatgpt.com/share/e2afcf21-78c0-423b-a2f3-dec51539da7f
        # plant.set_discrete_contact_solver(DiscreteContactSolver.kSap)

        parser.package_map().AddPackageXml(filename=drake_package_path)
        parser.AddModelsFromUrl(task_irrelev_obj_url)

        robot_model_instances: List[ModelInstanceIndex] = []

        if add_robot:
            assert robot_base_pos is not None and robot_base_quat_wxyz is not None, (
                "robot_base_pos and robot_base_quat_wxyz must be specified if add_robot is True"
                "because we assume robot base is fixed to the world frame."
            )
            X_WRobotBase = RigidTransform(
                Quaternion(robot_base_quat_wxyz), robot_base_pos
            )
            model_instance_index: ModelInstanceIndex = (
                DrakeMotionPlanningSpace.AddFranka(
                    parser, plant, body_world_pose=X_WRobotBase, eef_name=gripper_type
                )
            )
            robot_model_instances.append(model_instance_index)
        if add_robot_hand:
            if gripper_type == "panda_hand":
                model_instance_index: ModelInstanceIndex = (
                    DrakeMotionPlanningSpace.AddFrankaHand(parser)
                )
            elif gripper_type == "robotiq85":
                model_instance_index: ModelInstanceIndex = (
                    DrakeMotionPlanningSpace.AddRobotiq85(parser)
                )
            else:
                raise ValueError(f"Unknown hand type {gripper_type}")
            robot_model_instances.append(model_instance_index)
        if name_to_frame_info is not None:
            for name, frame_info in name_to_frame_info.items():
                X_W_FRAME = RigidTransform(
                    Quaternion(frame_info.offset_quat_wxyz), frame_info.offset_pos
                )
                plant.AddFrame(
                    FixedOffsetFrame(
                        name, plant.GetFrameByName(frame_info.src_frame), X_W_FRAME
                    )
                )

        if obj_to_init_info is not None:
            for obj_name, obj_init_info in obj_to_init_info.items():
                X_parentframe_obj = RigidTransform(obj_init_info.X_parentframe_obj)
                if not obj_init_info.weld_to_ee:
                    parent_frame_name = None
                else:
                    parent_frame_name = "ee_obs_frame"

                obj_model_index = DrakeMotionPlanningSpace.AddTaskRelevObj(
                    parser,
                    plant,
                    X_parentframe_obj,
                    obj_init_info.mesh_paths.sdf_path,
                    parent_frame_name,
                )
                if obj_init_info.weld_to_ee:
                    # assuming the object is part of the robot (for collision checks) if it's welded to the end effector
                    robot_model_instances.append(obj_model_index)

        meshcat, visualizer, collision_visualizer = None, None, None
        if view_meshcat:
            meshcat = StartMeshcat()
            visualizer = MeshcatVisualizer.AddToBuilder(
                builder.builder(),
                builder.scene_graph(),
                meshcat,
                MeshcatVisualizerParams(role=Role.kIllustration),
            )
            collision_visualizer = MeshcatVisualizer.AddToBuilder(
                builder.builder(),
                builder.scene_graph(),
                meshcat,
                MeshcatVisualizerParams(prefix="collision", role=Role.kProximity),
            )

        robot_diagram = builder.Build()  # plant.finalize() occurs here
        return (
            robot_diagram,
            robot_model_instances,
            meshcat,
            visualizer,
            collision_visualizer,
        )

    def _make_scene_graph_collision_checker(
        self,
        robot_diagram,
        robot_model_instances,
        edge_step_size: float = 0.01,
        env_collision_padding: float = 0,
        remove_halfspace: bool = True,
    ) -> planning.SceneGraphCollisionChecker:
        """
        Make a scene graph collision checker for the robot.

        Args:
            robot_diagram (Diagram): The robot diagram.
            # robot_model_index (int): The model index of the robot.
            robot_model_instances: List[ModelInstanceIndex]
            edge_step_size (float): The step size for checking edges.
            remove_halfspace (bool): Whether to remove halfspace geometries from the collision checker. Not sure why halfspaces are even needed?

        """
        collision_checker = planning.SceneGraphCollisionChecker(
            model=robot_diagram,
            robot_model_instances=robot_model_instances,
            configuration_distance_function=self.distance,
            edge_step_size=edge_step_size,
            env_collision_padding=env_collision_padding,
        )
        collision_checker.SetConfigurationInterpolationFunction(self.interpolate)
        if remove_halfspace:
            for body_idx in range(self.plant.num_bodies()):
                if body_idx == 0:
                    continue
                name = self.plant.get_body(BodyIndex(body_idx)).name()
                if "panda" not in name:
                    continue
                collision_checker.SetCollisionFilteredBetween(
                    self.plant.get_body(BodyIndex(body_idx)),
                    self.plant.world_body(),
                    True,
                )

        return collision_checker

    @staticmethod
    def AddFrankaHand(parser: Parser) -> ModelInstanceIndex:
        return parser.AddModelsFromUrl(
            "package://models/franka_description/urdf/hand.urdf"
        )[0]

    @staticmethod
    def AddRobotiq85(parser: Parser) -> ModelInstanceIndex:
        return parser.AddModelsFromUrl(
            "package://models/robotiq_85_description/urdf/robotiq_85_gripper.urdf"
        )[0]

    @staticmethod
    def AddFranka(
        parser: Parser,
        plant: MultibodyPlant,
        body_world_pose: Optional[RigidTransform] = None,
        eef_name: Literal["panda_hand", "robotiq85"] = "panda_hand",
    ) -> ModelInstanceIndex:
        print("")
        if eef_name == "panda_hand":
            franka = parser.AddModelsFromUrl(
                "package://models/franka_description/urdf/panda_arm_hand.urdf"
            )[0]
            # Set default positions:
            q0 = [
                0.0229177,
                0.19946329,
                -0.01342641,
                -2.63559645,
                0.02568405,
                2.93396808,
                0.79548173,
            ]
            print("Added panda_arm_hand to plant")
        elif eef_name == "robotiq85":
            franka = parser.AddModelsFromUrl(
                "package://models/franka_description/urdf/panda_arm_robotiq_85.urdf"
            )[0]
            # Set default positions:
            q0 = [
                0.0229177,
                0.19946329,
                -0.01342641,
                -2.63559645,
                0.02568405,
                2.93396808,
                0.79548173,
                0,
                0,
                0,
                0,
                0,
                0,
            ]
            print("Added panda_arm_robotiq_85 to plant")
        else:
            raise ValueError(
                f"Unknown eef_name {eef_name}. Can only be 'panda_hand' or 'robotiq85'"
            )

        if body_world_pose is None:
            body_world_pose = RigidTransform()
        plant.WeldFrames(
            plant.world_frame(), plant.GetFrameByName("panda_link0"), body_world_pose
        )

        index = 0
        # TODO(klin): check this revolute setting hardcoded value ...
        for joint_index in plant.GetJointIndices(franka):
            joint = plant.get_mutable_joint(joint_index)
            if isinstance(joint, RevoluteJoint):
                joint.set_default_angle(q0[index])
                index += 1

        return franka

    @staticmethod
    def convert_mujoco_q_to_drake(mujoco_q: np.ndarray) -> np.ndarray:
        """
        Convert mujoco joint angles to drake joint angles.
        """
        drake_q = mujoco_q.copy()
        drake_q = np.delete(drake_q, -2)
        return drake_q

    def visualize_trajectory(
        self,
        traj: List[np.ndarray],
        sleep: float = 0.1,
        input_config_frame: Optional[str] = None,
    ) -> None:
        """
        Visualize the trajectory.
        """
        assert (
            self.visualizer is not None and self.collision_visualizer is not None
        ), "visualizer and collision_visualizer must be initialized to visualize trajectory"
        print(f"visualizing trajectory of length {len(traj)} at {self.meshcat.port()}")
        while True:
            for q_val in traj:
                input_frame = (
                    input_config_frame
                    if input_config_frame is not None
                    else self.input_config_frame
                )
                if input_frame is not None:
                    # TODO(klin): make separate MotionPlanningSpace class; this function only makes sense for end effector motion planning space
                    assert (
                        "panda_hand" in input_frame
                    ), "input_frame must be a panda hand frame"
                    X_W_input = RigidTransform(
                        Quaternion(q_val[0:4] / np.linalg.norm(q_val[0:4])), q_val[4:7]
                    )
                    X_base_input_info = (
                        PandaGripperNameToFrameInfo.default().get_frame_info(
                            input_frame
                        )
                    )
                    X_base_input = RigidTransform(
                        Quaternion(X_base_input_info.offset_quat_wxyz),
                        X_base_input_info.offset_pos,
                    )
                    X_W_base = X_W_input.multiply(X_base_input.inverse())

                    # q_hand ordering [quat_wxyz, pos, grip]; unclear how to check though
                    q_val = np.concatenate(
                        [
                            X_W_base.rotation().ToQuaternion().wxyz(),
                            X_W_base.translation(),
                            q_val[7:],
                        ]
                    )

                self.plant.SetPositions(self.plant_context, q_val)
                self.visualizer.ForcedPublish(
                    self.visualizer.GetMyContextFromRoot(self.context)
                )
                self.collision_visualizer.ForcedPublish(
                    self.collision_visualizer.GetMyContextFromRoot(self.context)
                )
                time.sleep(sleep)

    def is_in_bounds(self, q: np.ndarray) -> bool:
        """
        Check if the joint configuration is in bounds.
        """
        return np.all(q >= self.bounds[0]) and np.all(q <= self.bounds[1])


class HandMotionPlanningSpace(DrakeMotionPlanningSpace):
    def __init__(
        self,
        drake_package_path: str,
        task_irrelev_obj_url: str = "package://models/assets/arenas/table_arena.xml",
        name_to_frame_info: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
        robot_base_pos: Optional[np.ndarray] = None,
        robot_base_quat_wxyz: Optional[np.ndarray] = None,
        task_relev_obj_pos_nerf: Optional[List[np.ndarray]] = None,
        task_relev_obj_rot_nerf: Optional[List[np.ndarray]] = None,
        gripper_type: Literal[
            "panda_hand", "robotiq85"
        ] = "panda_hand",  # todo: update 'hand' to 'gripper'
        obj_to_init_info: Optional[Dict[str, ObjectInitializationInfo]] = None,
        set_gripper_dim_to_one: bool = False,
        edge_step_size: float = 0.005,
        env_collision_padding: float = 0.0,
        view_meshcat: bool = False,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        input_config_frame: Optional[Literal["EE_SITE"]] = None,
        env_dist_factor: float = 0.0,
        env_influence_distance: float = 0.0,
    ):
        """
        Assumption: q = [quat_wxyz, pos, grip]

        Args:
            set_gripper_dim_to_one (bool): Whether to set the gripper dimension to 1.0. This is useful for
                motion planning in end effector space.
        """
        super().__init__(
            drake_package_path=drake_package_path,
            task_irrelev_obj_url=task_irrelev_obj_url,
            add_robot=False,
            add_robot_hand=True,
            gripper_type=gripper_type,
            name_to_frame_info=name_to_frame_info,
            robot_base_pos=robot_base_pos,
            robot_base_quat_wxyz=robot_base_quat_wxyz,
            task_relev_obj_pos_nerf=task_relev_obj_pos_nerf,
            task_relev_obj_rot_nerf=task_relev_obj_rot_nerf,
            obj_to_init_info=obj_to_init_info,
            set_gripper_dim_to_one=set_gripper_dim_to_one,
            edge_step_size=edge_step_size,
            view_meshcat=view_meshcat,
            bounds=bounds,
            input_config_frame=input_config_frame,
            env_collision_padding=env_collision_padding,
            env_dist_factor=env_dist_factor,
            env_influence_distance=env_influence_distance,
        )

    def set_q(
        self,
        q: np.ndarray,
        input_config_frame: Optional[str] = None,
    ):
        """Sets joint configuration of the robot."""
        if input_config_frame is None:
            input_config_frame = self.input_config_frame

        if input_config_frame is not None:
            # q_hand ordering [quat_wxyz, pos, grip]; unclear how to check though
            X_W_input = RigidTransform(
                Quaternion(q[0:4] / np.linalg.norm(q[0:4])), q[4:7]
            )
            X_base_input_info = PandaGripperNameToFrameInfo.default().get_frame_info(
                input_config_frame
            )
            X_base_input = RigidTransform(
                Quaternion(X_base_input_info.offset_quat_wxyz),
                X_base_input_info.offset_pos,
            )
            X_W_base = X_W_input.multiply(X_base_input.inverse())

            # q_hand ordering [quat_wxyz, pos, grip]; unclear how to check though
            q = np.concatenate(
                [
                    X_W_base.rotation().ToQuaternion().wxyz(),
                    X_W_base.translation(),
                    q[7:],
                ]
            )

        super().set_q(q)

    def distance(
        self,
        q1: np.ndarray,
        q2: np.ndarray,
        rot_factor: float = 0.2,
        env_dist_factor: float = 0,
        env_influence_distance: float = 0.08,
    ) -> float:
        """
        Return the distance between the two configurations.

        Hm, current implementation will favor shorter number of connections huh ...
        # want dist(a, c) = dist(a, b) + dist(b, c) if a b c is a 'straight line'
        Args:
            rot_factor (float): The weight for the rotation distance.
            env_dist_factor (float): The weight for the environment distance.
                Unfortunately, using robot clearance (i.e. env_dist_factor > 0) is too slow
                because distance() is called ~60k times => ~300s for pure hand to object motion planning.
            env_influence_distance (float): The distance at which the environment starts to influence the distance.
                only used if env_dist_factor > 0
        """
        if isinstance(q1, list):
            q1 = np.array(q1)
        if isinstance(q2, list):
            q2 = np.array(q2)

        if np.all(q1 == 0) and np.all(q2 == 0) or np.all(q1 == q2):
            return 0.0

        dist = np.linalg.norm(
            q1[4:7] - q2[4:7]
        ) + rot_factor * rotation_matrix_distance(
            R.from_quat(np.roll(q1[:4], shift=-1)).as_matrix(),
            R.from_quat(np.roll(q2[:4], shift=-1)).as_matrix(),
        )

        dist += 0.005  # HACK prefer less connections overall

        if env_dist_factor > 0:
            q2_base_frame = self._convert_pose_frame(
                [q2], self.input_config_frame, "panda_hand"
            )[0]
            dist += (
                (
                    (
                        env_influence_distance
                        - self._robot_clearance(q2_base_frame, env_influence_distance)
                    )
                    ** 2
                )
                * env_dist_factor
                * dist
            )
            # multiply by dist to account for algo prefering fewer connections to decrease the env distance component cost
        return dist

    def _convert_pose_frame(
        self, qs: List[np.ndarray], input_frame: str, output_frame: Optional[str] = None
    ) -> List[np.ndarray]:
        res = []
        for q in qs:
            # q_hand ordering [quat_wxyz, pos, grip]; unclear how to check though
            X_W_input = RigidTransform(
                Quaternion(q[0:4] / np.linalg.norm(q[0:4])), q[4:7]
            )
            X_base_input_info = PandaGripperNameToFrameInfo.default().get_frame_info(
                input_frame
            )
            # check that output_frame is src_frame; other way is to explicitly set positions of frame then querying poses ...
            if output_frame is not None:
                assert (
                    X_base_input_info.src_frame == output_frame
                ), f"output_frame {output_frame} must be {X_base_input_info.src_frame} for the following conversion"
            X_base_input = RigidTransform(
                Quaternion(X_base_input_info.offset_quat_wxyz),
                X_base_input_info.offset_pos,
            )
            X_W_base = X_W_input.multiply(X_base_input.inverse())

            # q_hand ordering [quat_wxyz, pos, grip]; unclear how to check though
            q = np.concatenate(
                [
                    X_W_base.rotation().ToQuaternion().wxyz(),
                    X_W_base.translation(),
                    q[7:],
                ]
            )
            res.append(q)
        return res

    def is_visible(
        self, q1: np.ndarray, q2: np.ndarray, input_config_frame: Optional[str] = None
    ) -> bool:
        input_frame = (
            input_config_frame
            if input_config_frame is not None
            else self.input_config_frame
        )
        res = self._convert_pose_frame([q1, q2], input_frame)
        # if for edge step is less than 'distance', step with the smaller of the two clearances
        # 'correct' way is to get smaller of two distances and then take that much of a step for q1 in the direction of q2
        # and vice versa for q2
        # for now, take the smaller of the two clearances and set as step size
        clearance_1 = self.collision_checker.CalcRobotClearance(
            q=res[0], influence_distance=0.1
        )
        clearance_2 = self.collision_checker.CalcRobotClearance(
            q=res[0], influence_distance=0.1
        )
        min_robot_env_dist = np.minimum(
            clearance_1.distances().min(), clearance_2.distances().min()
        )
        original_edge_step_size = self.edge_step_size
        if min_robot_env_dist > 0.05:
            self.set_edge_step_size(0.04)
        elif min_robot_env_dist > 0.02:
            self.set_edge_step_size(0.015)

        is_visible: bool = self.collision_checker.CheckContextEdgeCollisionFree(
            self.collision_checker_context, res[0], res[1]
        )
        self.set_edge_step_size(original_edge_step_size)
        return is_visible

    def extend_configs(
        self,
        q1: np.ndarray,
        q2: np.ndarray,
        resolution: float = 0.004,
        extend_type: Literal[
            "configuration", "end_effector", "end_effector_dist"
        ] = "end_effector",
    ) -> List:
        if (dist := self.distance(q1, q2)) < resolution:
            return [q2]

        X_1, X_2 = np.eye(4), np.eye(4)
        X_1[:3, 3], X_2[:3, 3] = q1[4:7], q2[4:7]
        X_1[:3, :3], X_2[:3, :3] = (
            R.from_quat(np.roll(q1[:4], shift=-1)).as_matrix(),
            R.from_quat(np.roll(q2[:4], shift=-1)).as_matrix(),
        )

        num_poses = int(np.ceil(dist / resolution)) + 1

        Xs = interpolate_poses(X_1, X_2, num_poses)
        return [
            np.concatenate(
                [np.roll(R.from_matrix(X[:3, :3]).as_quat(), shift=1), X[:3, 3], q1[7:]]
            )
            for X in Xs
        ]

    def interpolate(
        self,
        q1: np.ndarray,
        q2: np.ndarray,
        u: float,
        input_config_frame: Optional[str] = None,
    ) -> np.ndarray:
        """
        Interpolate between two configurations.
        """
        # print(f"interpolate from handmp called")
        from scipy.spatial.transform import Rotation as R
        from scipy.spatial.transform import Slerp

        if isinstance(q1, list):
            q1 = np.array(q1)
        if isinstance(q2, list):
            q2 = np.array(q2)

        if np.all(q1 == 0) and np.all(q2 == 0) or np.all(q1 == q2):
            # handle case where drake tests the function
            return q1

        R1 = R.from_quat(np.roll(q1[:4], shift=-1)).as_matrix()
        R2 = R.from_quat(np.roll(q2[:4], shift=-1)).as_matrix()
        T1 = q1[4:7]
        T2 = q2[4:7]

        # Interpolate translation using linear interpolation
        pos = (1 - u) * T1 + u * T2

        # Interpolate rotation using spherical linear interpolation (slerp)
        rotation_slerp = Slerp([0, 1], R.from_matrix([R1, R2]))
        quat_xyzw = rotation_slerp(u).as_quat()

        return np.concatenate([np.roll(quat_xyzw, shift=1), pos, q1[7:]])

    def visualize_poses(self, poses: List[np.ndarray], save_path: Optional[str] = None):
        # plot a list of poses in 3D using matplotlib (plot their positions only)
        import matplotlib.pyplot as plt

        # Create a new figure for the 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # Subsample every 100th pose
        subsampled_poses = poses[::100]

        # Extract x, y, z coordinates from the subsampled poses
        x = [pose[4] for pose in subsampled_poses]
        y = [pose[5] for pose in subsampled_poses]
        z = [pose[6] for pose in subsampled_poses]

        # Scatter plot
        ax.scatter(x, y, z, marker="o")

        # Adjust the axes to ensure that min/max values are included
        ax.set_xlim(min(x), max(x))
        ax.set_ylim(min(y), max(y))
        ax.set_zlim(min(z), max(z))

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # Show or save the plot
        if save_path:
            plt.savefig(save_path)
            print(f"Saved visualization to {save_path}")
        else:
            plt.show()

    def sample_neighborhood(self, q: np.ndarray, radius: float) -> np.ndarray:
        """
        Sample a configuration in the neighborhood of q.

        Args:
            q (numpy.ndarray): The configuration to sample around.
            radius (float): The radius of the neighborhood to sample from.

        Returns:
            numpy.ndarray: A configuration in the neighborhood of q.
        """
        sample = super().sample_neighborhood(q, radius)
        # normalize quaternion
        sample[:4] /= np.linalg.norm(sample[:4])
        return np.clip(sample, self.bounds[0], self.bounds[1])


class FixedGripperWrapper:
    # Note: if any method is not implemented, it will be passed to the wrapped object
    # if that wrapped object's method calls .bounds(), the values mon't be the correct values ...
    def __init__(
        self,
        motion_planning_space: DrakeMotionPlanningSpace,
        q_gripper_fixed: Optional[np.ndarray] = None,
    ):
        self._motion_planning_space = motion_planning_space
        self.q_gripper_fixed = q_gripper_fixed

    @cached_property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        # Note: hardcoding gripper removal; robust way = check idxs of gripper joints and delete those
        lower, upper = (
            self._motion_planning_space.bounds[0].copy(),
            self._motion_planning_space.bounds[1].copy(),
        )
        lower[-len(self.q_gripper_fixed) :] = self.q_gripper_fixed[
            -len(self.q_gripper_fixed) :
        ]
        upper[-len(self.q_gripper_fixed) :] = self.q_gripper_fixed[
            -len(self.q_gripper_fixed) :
        ]
        return lower, upper

    def sample_neighborhood(self, q: np.ndarray, radius: float) -> np.ndarray:
        sampled = self._motion_planning_space.sample_neighborhood(q, radius)
        return np.clip(sampled, self.bounds[0], self.bounds[1])

    def set_q(self, q: np.ndarray):
        """
        Set the joint configuration of the robot. Used for visualization.
        """
        assert (
            q.shape[-1] == self.bounds[0].shape[-1]
        ), "q must have the same shape as the bounds"
        # q = np.concatenate([q, self.q_gripper_fixed])
        q = np.concatenate([q[: -len(self.q_gripper_fixed)], self.q_gripper_fixed])
        self.plant.SetPositions(self.plant_context, q)
        if self.visualizer is not None:
            self.visualizer.ForcedPublish(
                self.visualizer.GetMyContextFromRoot(self.context)
            )
            self.collision_visualizer.ForcedPublish(
                self.collision_visualizer.GetMyContextFromRoot(self.context)
            )

    def is_collision(
        self,
        q_val: np.ndarray,
        input_config_frame: Optional[Literal["EE_SITE"]] = None,
        viz: bool = False,
    ) -> bool:
        q_val = np.concatenate(
            [q_val[: -len(self.q_gripper_fixed)], self.q_gripper_fixed]
        )
        q_val = self._construct_valid_input_q_value(q_val)
        return self._motion_planning_space.is_collision(q_val, input_config_frame, viz)

    def is_visible(self, q1: np.ndarray, q2: np.ndarray) -> bool:
        """
        Return whether the 'edge' from q1 to q2 is collision free.
        """
        q1, q2 = np.array(q1), np.array(q2)
        assert (
            q1.shape[-1] == self.bounds[0].shape[-1]
            and q2.shape[-1] == self.bounds[0].shape[-1]
        ), "q must have the same shape as the bounds"
        if not np.all(self.q_gripper_fixed == 0.04):
            import ipdb

            ipdb.set_trace()
        q1 = np.concatenate([q1[: -len(self.q_gripper_fixed)], self.q_gripper_fixed])
        q2 = np.concatenate([q2[: -len(self.q_gripper_fixed)], self.q_gripper_fixed])
        return self._motion_planning_space.is_visible(q1, q2)

    @staticmethod
    def convert_config_frame(
        q_in: np.ndarray,
        input_frame: Literal["ee_obs_frame", "ee_action_frame"],
    ) -> np.ndarray:
        # TODO(klin): make separate MotionPlanningSpace class; this function only makes sense for end effector motion planning space
        assert "panda_hand" in input_frame, "input_frame must be a panda hand frame"
        X_W_input = RigidTransform(
            Quaternion(q_in[0:4] / np.linalg.norm(q_in[0:4])), q_in[4:7]
        )
        X_base_input_info = PandaGripperNameToFrameInfo.default().get_frame_info(
            input_frame
        )
        X_base_input = RigidTransform(
            Quaternion(X_base_input_info.offset_quat_wxyz), X_base_input_info.offset_pos
        )
        X_W_base = X_W_input.multiply(X_base_input.inverse())

        # q_hand ordering [quat_wxyz, pos, grip]; unclear how to check though
        q_out = np.concatenate(
            [
                X_W_base.rotation().ToQuaternion().wxyz(),
                X_W_base.translation(),
                q_in[7:],
            ]
        )

        return q_out

    def is_in_bounds(self, q: np.ndarray) -> bool:
        """
        Check if the joint configuration is in bounds.
        """
        q = self._construct_valid_input_q_value(q)
        return np.all(q >= self.bounds[0]) and np.all(q <= self.bounds[1])

    # Optionally, if you want to be able to access any method of the original object
    # without explicitly defining a wrapper method, you can use Python's __getattr__ magic method
    def __getattr__(self, name):
        # This will only get called for attributes/methods not explicitly defined in the wrapper
        return getattr(self._motion_planning_space, name)
