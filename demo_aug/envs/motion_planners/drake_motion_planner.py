import datetime
import logging
import pathlib
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import scipy.interpolate as si
import scipy.spatial.transform as st
import torch
from pydrake.common.eigen_geometry import Quaternion
from pydrake.geometry import (
    CollisionFilterDeclaration,
    GeometrySet,
    Mesh,
    Meshcat,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    Rgba,
    Role,
    Sphere,
    StartMeshcat,
)
from pydrake.math import BsplineBasis, RigidTransform, RotationMatrix
from pydrake.multibody.inverse_kinematics import (
    InverseKinematics,
    MinimumDistanceConstraint,
)
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import (
    AddMultibodyPlantSceneGraph,
    CoulombFriction,
    MultibodyPlant,
)
from pydrake.multibody.tree import FixedOffsetFrame, ModelInstanceIndex, RevoluteJoint
from pydrake.planning import KinematicTrajectoryOptimization
from pydrake.solvers import MathematicalProgramResult, SnoptSolver, SolutionResult
from pydrake.systems.framework import Context, DiagramBuilder
from pydrake.trajectories import BsplineTrajectory
from scipy.spatial.transform import Rotation as R

import demo_aug
from demo_aug.configs.robot_configs import (
    DRAKE_TO_ROBOMIMIC_GRIPPER_OFFSET_POS,
    ROBOMIMIC_EE_SITE_FRAME_NAME,
    ROBOMIMIC_GRIPPER_SITE_FRAME_NAME,
    ROBOMIMIC_HAND_GRIPPER_OFFSET_QUAT_WXYZ,
    ROBOT_BASE_FRAME_NAME,
)
from demo_aug.envs.base_env import MotionPlannerType
from demo_aug.envs.motion_planners.base_motion_planner import RobotEnvConfig
from demo_aug.envs.motion_planners.motion_planning_space import ObjectInitializationInfo
from demo_aug.objects.robot_object import RobotObject
from demo_aug.utils.drake_utils import PublishPositionTrajectory
from demo_aug.utils.mathutils import (
    get_timesteps,
    random_rotation_matrix,
    random_z_rotation,
)
from demo_aug.utils.obj_utils import extract_model_name_from_sdf
from demo_aug.utils.run_script_utils import retry_on_exception
from demo_aug.utils.snopt_utils import (
    SNOPT_SOLVER_MAX_OPTIMALITY_VALUE,
    extract_feasiblity_optimality_values_from_snopt_log,
)


def rotation_difference_angle(
    mat1: Union[np.ndarray, R], mat2: Union[np.ndarray, R]
) -> float:
    """Compute the (axis) angle (in radians) between two rotation matrices."""

    # Convert the rotation matrices into Rotation objects if they are not already
    r1 = R.from_matrix(mat1) if not isinstance(mat1, R) else mat1
    r2 = R.from_matrix(mat2) if not isinstance(mat2, R) else mat2

    # Compute the relative rotation
    relative_rotation = r1.inv() * r2

    # Convert to rotation vector representation
    rotvec = relative_rotation.as_rotvec()

    # Extract the angle by computing the norm of the rotation vector
    angle = np.linalg.norm(rotvec)

    return angle


def get_start_to_goal_time(
    start_pos: np.ndarray,
    end_pos: np.ndarray,
    start_rot: np.ndarray,
    end_rot: np.ndarray,
    pos_vel: float = 0.25,
    rot_vel: float = 0.7,
) -> float:
    """
    Compute the time to reach a goal pose from a start pose.

    Parameters:
    - pos_vel: m/s
    - rot_vel: rad/s

    Returns:
    - Float representing the time to reach the goal.
    """
    # Compute positional difference
    pos_diff = np.linalg.norm(start_pos - end_pos)

    # Compute rotational difference in radians
    rot_diff = rotation_difference_angle(start_rot, end_rot)

    # Compute the time for each transition
    pos_time = pos_diff / pos_vel
    rot_time = rot_diff / rot_vel
    print(f"pos_diff: {pos_diff}")
    print(f"rot_diff: {rot_diff}")
    print(f"pos_time: {pos_time}")
    print(f"rot_time: {rot_time}")
    # Return the maximum time
    return max(pos_time, rot_time)


class IKType(Enum):
    X_EE_TO_Q_ROBOT = 0
    P_KPTS_TO_X_EE = 1


def get_obj_file_paths_from_sdf(
    sdf_file_path, parent_tag: Literal["visual", "collision"] = "collision"
) -> List[pathlib.Path]:
    # Convert the input path to a Path object for easier manipulation
    sdf_file_path = pathlib.Path(sdf_file_path).resolve()

    # Get the directory of the .sdf file
    sdf_directory = sdf_file_path.parent

    # Load and parse the .sdf file
    tree = ET.parse(sdf_file_path)
    root = tree.getroot()

    # Initialize a list to hold full paths of .obj files
    obj_file_paths = []

    # Search for all 'uri' elements within 'mesh' elements
    for mesh in root.findall(f".//{parent_tag}/geometry/mesh/uri"):
        # Extract the text content of the uri element
        uri = mesh.text

        # Check if the uri references an .obj file
        if uri and uri.endswith(".obj"):
            # Construct the full path using pathlib's / operator for path concatenation
            full_path = sdf_directory / uri
            obj_file_paths.append(full_path.resolve())

    return obj_file_paths


# TODO(klin): figure out how to scale object correctly
class DrakeMotionPlanner:
    def __init__(
        self,
        robot_obj: RobotObject,
        drake_package_path: str = str(
            pathlib.Path(demo_aug.__file__).parent / "models/package.xml"
        ),
        input_ee_pose_frame_convention: str = "mujoco",
        task_irrelev_obj_url: str = "package://models/assets/arenas/table_arena.xml",
        action_freq: float = 10,  # TODO(klin): adjust based on actual robot; use robomimic's soon
        view_meshcat: bool = False,
    ):
        """TODO(klin): once dealing w/ non mujoco urdfs, update handling
        of frame position offsets (esp. in get_IK())
        """
        self.robot_obj = robot_obj
        self.drake_package_path = str(drake_package_path)
        self.input_ee_pose_frame_convention = input_ee_pose_frame_convention
        self.task_irrelev_obj_url = task_irrelev_obj_url
        self.action_freq = action_freq
        self.view_meshcat = view_meshcat
        if view_meshcat:
            self.meshcat = StartMeshcat()
        else:
            self.meshcat = None

    @staticmethod
    def AddFrankaHand(
        parser: Parser, plant: MultibodyPlant, body_world_pose: RigidTransform = None
    ) -> ModelInstanceIndex:
        franka_hand = parser.AddModelFromFile(
            "models/franka_description/urdf/hand.urdf"
        )
        if body_world_pose is None:
            body_world_pose = RigidTransform()

        return franka_hand

    @staticmethod
    def AddFranka(
        parser: Parser, plant: MultibodyPlant, body_world_pose: RigidTransform = None
    ) -> ModelInstanceIndex:
        franka = parser.AddModelsFromUrl(
            "package://models/franka_description/urdf/panda_arm_hand.urdf"
        )[0]
        if body_world_pose is None:
            body_world_pose = RigidTransform()
        plant.WeldFrames(
            plant.world_frame(), plant.GetFrameByName("panda_link0"), body_world_pose
        )

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
        index = 0
        for joint_index in plant.GetJointIndices(franka):
            joint = plant.get_mutable_joint(joint_index)
            if isinstance(joint, RevoluteJoint):
                joint.set_default_angle(q0[index])
                index += 1

        return franka

    @staticmethod
    @retry_on_exception(max_retries=5, retry_delay=1, exceptions=(RuntimeError,))
    def AddTaskRelevObj(
        parser: Parser,
        plant: MultibodyPlant,
        X_parentframe_obj: RigidTransform,  # reduces to X_EE_NERF or X_EE_MESH in this case
        obj_url: str,
        parent_frame_name: Optional[str] = None,
        add_to_parent_geometry: bool = False,
    ) -> ModelInstanceIndex:
        """Add a task relevant object to the plant and weld it to the parent frame.

        Args:
            add_to_parent_geometry: Whether to add the object geometry to a parent geometry. Defaults to False.
                We do so to use drake's custom RobotClearance class for collision checking.
                https://drake.mit.edu/pydrake/pydrake.planning.html?highlight=robotclearance#pydrake.planning.RobotClearance

                Doing the below method of adding object to be safe when doing collision-free IK;
                in the future, likely use e.g. FastIK or something else in collision checking.
                TODO(klin): unclear if doing so make collision checking difficult?
        """
        # This is used when using drake for IK collision checking
        if add_to_parent_geometry:
            if "nut" in obj_url:
                task_relev_obj_urls = get_obj_file_paths_from_sdf(obj_url, "collision")
                hand = plant.GetBodyByName("panda_hand")
                X_Hand_Grippersite = RigidTransform(
                    DRAKE_TO_ROBOMIMIC_GRIPPER_OFFSET_POS
                )
                hand_to_object = (
                    X_Hand_Grippersite.GetAsMatrix4() @ X_parentframe_obj.GetAsMatrix4()
                )

                for i, task_relev_obj_url in enumerate(task_relev_obj_urls):
                    visual_mesh = Mesh(str(task_relev_obj_url), 1.0)
                    plant.RegisterVisualGeometry(
                        hand,
                        RigidTransform(hand_to_object),
                        visual_mesh,
                        f"new_object_visual{i}",
                        np.array([0.5, 0.5, 0.5, 1.0]),
                    )
                    material = CoulombFriction(
                        static_friction=0.5, dynamic_friction=0.3
                    )  # Adjust friction properties as needed

                    collision_mesh = Mesh(str(task_relev_obj_url), 1.0)
                    plant.RegisterCollisionGeometry(
                        hand,
                        RigidTransform(hand_to_object),
                        collision_mesh,
                        f"new_object_collision{i}",
                        material,
                    )
            else:
                import ipdb

                ipdb.set_trace()
                print(
                    f'add_to_parent_geometry is True but obj_url is not "nut": {obj_url}'
                )

        else:
            # if there's package:// in front of the url use the below
            if "package://" in obj_url:
                obj = parser.AddModelsFromUrl(str(obj_url))[0]
            else:
                # obj = parser.AddModelFromFile(str(obj_url))
                obj = parser.AddModels(str(obj_url))[0]

            if parent_frame_name is None:
                # Default to the world frame if no parent_frame_name is provided
                parent_frame = plant.world_frame()
            else:
                parent_frame = plant.GetFrameByName(parent_frame_name)
            plant.WeldFrames(
                parent_frame,
                plant.GetFrameByName(plant.GetModelInstanceName(obj)),
                X_parentframe_obj,
            )

        return True

    @staticmethod
    def get_IK(
        plant: MultibodyPlant,
        context: Context,
        ee_pose: Optional[RigidTransform] = None,
        visualizer: Optional[MeshcatVisualizer] = None,
        collision_visualizer: Optional[MeshcatVisualizer] = None,
        meshcat: Optional[Meshcat] = None,
        ik_type: IKType = IKType.X_EE_TO_Q_ROBOT,
        ee_frame_name: str = "panda_hand_gripper_site_robomimic",
        input_ee_pose_frame_convention: str = "mujoco",
        constraint_orientation: bool = True,
        initial_robot_joint_qpos: Optional[np.ndarray] = None,
        initial_robot_gripper_qpos: Optional[np.ndarray] = None,
        X_W_ee_init: Optional[np.ndarray] = None,
        q_gripper_init: Optional[np.ndarray] = None,
        kp_to_P_goal: Optional[Dict[str, np.ndarray]] = None,
        goal_robot_gripper_qpos: Optional[np.ndarray] = None,
        check_collisions: bool = True,
        min_dist: float = 0.01,
        min_dist_thresh: float = 0.03,
        n_trials: int = 5,
    ):
        """
        Computes the inverse kinematics for a given end effector pose (in  world frame) and gripper_qpos,
        subject to minimum distance constraints. Note that the ee_pose_frame_convention
        mightn't be aligned with drake's frame_name.

        Args:
            plant: The MultibodyPlant representing the robot.
            context: The Context of the MultibodyPlant.
            ee_pose: The desired end effector pose as a RigidTransform. Defaults to None.
            frame_name: The name of the frame whose position is used as the end effector. Defaults to "panda_hand".
            input_ee_pose_frame_convention: The frame convention used in the provided ee_pose.
                Defaults to "mujoco".
            check_collisions: Whether to check for collisions. Defaults to True. Set to False for easier solving.

        Returns:
            A numpy array of joint angles representing the solution to the inverse kinematics problem.
            These output joint angles conform to drake convention by default



        https://chatgpt.com/share/86da943c-59e2-497f-9e81-9854c095ca99
        How to write the optimization problem that keeps joints having similar values.
        """
        plant_context = plant.GetMyContextFromRoot(context)
        q0 = plant.GetPositions(plant_context)

        ik = InverseKinematics(plant, plant_context)
        prog = ik.get_mutable_prog()

        # create folder for trajopt results using pathlib
        solver_results_dir = pathlib.Path(
            "ik_results"
        ) / datetime.datetime.now().strftime("%Y_%m_%d-%H")
        solver_results_dir.mkdir(parents=True, exist_ok=True)

        if ik_type == IKType.X_EE_TO_Q_ROBOT:
            gripper_frame = plant.GetFrameByName(ee_frame_name)

            if initial_robot_joint_qpos is not None:
                if (
                    initial_robot_gripper_qpos is not None
                    and initial_robot_gripper_qpos[-1] < 0
                ):
                    initial_robot_gripper_qpos[-1] = -initial_robot_gripper_qpos[-1]
                plant.SetPositions(
                    plant_context,
                    np.concatenate(
                        (initial_robot_joint_qpos, initial_robot_gripper_qpos)
                    ),
                )
                q0 = plant.GetPositions(plant_context)
                assert np.all(
                    initial_robot_gripper_qpos >= 0.0
                ), "gripper qpos must be non-negative if input_gripper_qpos_convention is drake"

            ik.AddPositionConstraint(
                gripper_frame,
                [0, 0, 0],
                plant.world_frame(),
                ee_pose.translation(),
                ee_pose.translation(),
            )
            if constraint_orientation:
                ik.AddOrientationConstraint(
                    gripper_frame,
                    RotationMatrix(),
                    plant.world_frame(),
                    ee_pose.rotation(),
                    0.0,
                )
            # when debugging, don't stright up use q
            q_var = ik.q()
            prog.AddQuadraticErrorCost(np.identity(len(q_var)), q0, q_var)

        elif ik_type == IKType.P_KPTS_TO_X_EE:
            assert (
                kp_to_P_goal is not None
            ), "kp_to_P_goal must be provided if ik_type is P_KPTS_TO_X_EE"

            # set reference values
            q0 = np.zeros(plant.num_positions())
            # convert to quaternion
            eef_quat_xyzw_init = st.Rotation.from_matrix(X_W_ee_init[:3, :3]).as_quat()
            eef_quat_wxyz_init = np.array(
                [eef_quat_xyzw_init[-1], *eef_quat_xyzw_init[:-1]]
            )
            q0[:4] = eef_quat_wxyz_init
            q0[4:7] = X_W_ee_init[:3, 3]
            q0[7:] = np.random.uniform(0, 0.04, 1) * np.ones(2)

            q_var = ik.q()
            for i, (k, v) in enumerate(kp_to_P_goal.items()):
                if i >= 1:
                    break
                ik.AddPositionConstraint(
                    plant.GetFrameByName(k),
                    [0, 0, 0],
                    plant.world_frame(),
                    v,
                    v,
                )
            # effectively enforcing single joint centered at 0 for gripper qpos
            gripper_tip_deviation = 5 * (q_var[-1] - q_var[-2]) ** 2
            prog.AddCost(gripper_tip_deviation)
            # prog.AddQuadraticErrorCost(np.identity(len(q_var)), q0, q_var)  # cost of gripper messes up optimization
            prog.AddQuadraticErrorCost(np.identity(len(q_var[:7])), q0[:7], q_var[:7])

            prog.SetInitialGuess(q_var, q0)
            snopt_solver = SnoptSolver()

            current_time = datetime.datetime.now().strftime("%b_%d_%H_%M_%S")
            file_path: pathlib.Path = (
                solver_results_dir
                / f"ik_kpts_to_eef_grip_no_collision_{current_time}.snopt"
            )

            prog.SetSolverOption(
                snopt_solver.solver_id(), "Major iterations limit", 5000
            )
            prog.SetSolverOption(snopt_solver.solver_id(), "Iterations limit", 5000)
            # prog.SetSolverOption(snopt_solver.solver_id(), "Print file", str(file_path))
            prog.SetSolverOption(snopt_solver.solver_id(), "Time limit", 4)
            prog.SetSolverOption(snopt_solver.solver_id(), "Timing level", 2)

            result = snopt_solver.Solve(prog)

            q_res = result.GetSolution(q_var)
            if not result.is_success():
                logging.info(
                    "IK failure on kpts to eef + gripper on first round; even without collision constraints!"
                )
                logging.info(result.get_solver_id().name())
                logging.info(result.GetInfeasibleConstraints(prog))
                logging.info(result.GetInfeasibleConstraintNames(prog))
                logging.info(result.get_x_val())
                # use prints to ensure terminal outputs because Drake's logger seems to override my logger
                print(result.get_solver_id().name())
                print(result.GetInfeasibleConstraints(prog))
                print(result.GetInfeasibleConstraintNames(prog))
                print(result.get_x_val())
                if meshcat is not None:
                    meshcat.SetObject(
                        "start", Sphere(0.02), rgba=Rgba(0.9, 0.1, 0.1, 1)
                    )
                    meshcat.SetTransform("start", RigidTransform(q_res[4:7]))

            for i, (k, v) in enumerate(kp_to_P_goal.items()):
                if i < 1:
                    continue
                ik.AddPositionConstraint(
                    plant.GetFrameByName(k),
                    [0, 0, 0],
                    plant.world_frame(),
                    v,
                    v,
                )

            for _ in range(n_trials):
                randomly_rotate_q_init = False
                if randomly_rotate_q_init:
                    # Generate a random rotation matrix
                    rotation = R.random(random_state=42)

                    # Convert the random rotation matrix to a quaternion (xyzw)
                    random_rotation_quaternion = rotation.as_quat()

                    # Get quat_xyzw version of q_res (from wxyz)
                    q_res_xyzw = np.roll(q_res[:4], shift=-1)

                    # Multiply q_res_xyzw with the random rotation quaternion
                    q_res_mat = R.from_quat(q_res_xyzw).as_matrix() @ (
                        R.from_quat(random_rotation_quaternion).as_matrix()
                    )
                    q_res_xyzw = R.from_matrix(q_res_mat).as_quat()

                    # Convert result back to wxyz and store in q_res
                    q_res[:4] = np.roll(q_res_xyzw, shift=1)

                q_res[7:] = np.random.uniform(0, 0.04, 1) * np.ones(2)
                prog.SetInitialGuess(q_var, q_res)

                current_time = datetime.datetime.now().strftime("%b_%d_%H_%M_%S_%f")
                file_path: pathlib.Path = (
                    solver_results_dir
                    / f"ik_kpts_to_eef_grip_no_collision_round2_{current_time}.snopt"
                )

                prog.SetSolverOption(
                    snopt_solver.solver_id(), "Major iterations limit", 5000
                )
                prog.SetSolverOption(snopt_solver.solver_id(), "Iterations limit", 5000)
                prog.SetSolverOption(
                    snopt_solver.solver_id(), "Print file", str(file_path)
                )
                prog.SetSolverOption(snopt_solver.solver_id(), "Time limit", 4)
                prog.SetSolverOption(snopt_solver.solver_id(), "Timing level", 2)
                result = snopt_solver.Solve(prog)

                q_res = result.GetSolution()
                if not result.is_success():
                    logging.info("IK failure on second pass for matching keypoints!")
                    logging.info(result.get_solver_id().name())
                    logging.info(result.GetInfeasibleConstraints(prog))
                    logging.info(result.GetInfeasibleConstraintNames(prog))
                    feasibility_value, optimality_value, is_acceptably_feasible = (
                        extract_feasiblity_optimality_values_from_snopt_log(file_path)
                    )
                    file_path.unlink(missing_ok=True)
                    logging.info(f"feasibility_value: {feasibility_value}")
                    logging.info(f"optimality_value: {optimality_value}")
                    logging.info(
                        f"solver return value: {result.get_solver_details().info}"
                    )
                    logging.info(f"Optimal cost: {result.get_optimal_cost()}")

                    if meshcat is not None:
                        meshcat.SetObject(
                            "start", Sphere(0.02), rgba=Rgba(0.9, 0.1, 0.1, 1)
                        )
                        meshcat.SetTransform("start", RigidTransform(q_res[4:7]))

                    if meshcat is not None:
                        for i, (k, v) in enumerate(kp_to_P_goal.items()):
                            meshcat.SetObject(
                                k,
                                Sphere(0.005),
                                rgba=Rgba(0.1 * i, 0.9 * i, 0.1 * i, 1),
                            )
                            meshcat.SetTransform(k, RigidTransform(v))
                else:
                    file_path.unlink(missing_ok=True)
                    if visualizer is not None:
                        # set position to q_res
                        plant.SetPositions(plant_context, q_res)
                        visualizer_context = visualizer.GetMyContextFromRoot(context)
                        visualizer.ForcedPublish(visualizer_context)

                        if meshcat is not None:
                            meshcat.SetObject(
                                "start", Sphere(0.02), rgba=Rgba(0.9, 0.1, 0.1, 1)
                            )
                            meshcat.SetTransform("start", RigidTransform(q_res[4:7]))

                        if meshcat is not None:
                            for i, (k, v) in enumerate(kp_to_P_goal.items()):
                                meshcat.SetObject(
                                    k,
                                    Sphere(0.005),
                                    rgba=Rgba(0.1 * i, 0.9 * i, 0.1 * i, 1),
                                )
                                meshcat.SetTransform(k, RigidTransform(v))
                    break

        if goal_robot_gripper_qpos is not None:
            assert isinstance(
                goal_robot_gripper_qpos, np.ndarray
            ) and goal_robot_gripper_qpos.shape == (
                2,
            ), "goal_robot_gripper_qpos must be a numpy array of shape (2,)"
            if ik_type == IKType.P_KPTS_TO_X_EE:
                print(
                    "If directly setting goal_robot_gripper_qpos, ik_type must be X_EE_TO_Q_ROBOT"
                )
            # force gripper to match goal_robot_gripper_qpos
            prog.AddBoundingBoxConstraint(
                np.abs(goal_robot_gripper_qpos[-1]),
                np.abs(goal_robot_gripper_qpos[-1]),
                [q_var[-1]],
            )
            prog.AddBoundingBoxConstraint(
                np.abs(goal_robot_gripper_qpos[-2]),
                np.abs(goal_robot_gripper_qpos[-2]),
                [q_var[-2]],
            )

            prog.SetInitialGuess(q_var, q0)
            snopt_solver = SnoptSolver()

            current_time = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
            file_path: pathlib.Path = (
                solver_results_dir
                / f"ik_{ik_type.name}_no_collision_{current_time}.snopt"
            )

            prog.SetSolverOption(
                snopt_solver.solver_id(), "Major iterations limit", 5000
            )
            prog.SetSolverOption(snopt_solver.solver_id(), "Iterations limit", 5000)
            prog.SetSolverOption(snopt_solver.solver_id(), "Print file", str(file_path))
            prog.SetSolverOption(snopt_solver.solver_id(), "Time limit", 4)
            prog.SetSolverOption(snopt_solver.solver_id(), "Timing level", 2)

            result = snopt_solver.Solve(prog)

            q_res = result.GetSolution(q_var)
            if not result.is_success():
                logging.info(
                    "IK failure even without collision constraints! May have reached some kinematic limit!"
                )
                logging.info(result.get_solver_id().name())
                logging.info(result.GetInfeasibleConstraints(prog))
                logging.info(result.GetInfeasibleConstraintNames(prog))
                try:
                    feasibility_value, optimality_value, is_acceptably_feasible = (
                        extract_feasiblity_optimality_values_from_snopt_log(file_path)
                    )
                except Exception as e:
                    print(e)
                    import ipdb

                    ipdb.set_trace()
                    feasibility_value, optimality_value, is_acceptably_feasible = (
                        extract_feasiblity_optimality_values_from_snopt_log(file_path)
                    )
                file_path.unlink(missing_ok=True)
                logging.info(f"feasibility_value: {feasibility_value}")
                logging.info(f"optimality_value: {optimality_value}")
                logging.info(f"solver return value: {result.get_solver_details().info}")
                logging.info(f"Optimal cost: {result.get_optimal_cost()}")
                if meshcat is not None and ee_pose is not None:
                    meshcat.SetObject(
                        "start", Sphere(0.02), rgba=Rgba(0.9, 0.1, 0.1, 1)
                    )
                    meshcat.SetTransform("start", RigidTransform(ee_pose.translation()))

                if visualizer is not None:
                    plant_context = plant.GetMyContextFromRoot(context)
                    plant.SetPositions(plant_context, q_res)

                    visualizer_context = visualizer.GetMyContextFromRoot(context)
                    visualizer.ForcedPublish(visualizer_context)
                if collision_visualizer is not None:
                    collision_context = collision_visualizer.GetMyContextFromRoot(
                        context
                    )
                    collision_visualizer.ForcedPublish(collision_context)

                return q_res, result.is_success()
            else:
                file_path.unlink(missing_ok=True)

        if not check_collisions:
            if visualizer is not None:
                visualizer_context = visualizer.GetMyContextFromRoot(context)
                visualizer.ForcedPublish(visualizer_context)

                if collision_visualizer is not None:
                    collision_context = collision_visualizer.GetMyContextFromRoot(
                        context
                    )
                    collision_visualizer.ForcedPublish(collision_context)

            return q_res, result.is_success()

        prog.SetInitialGuess(q_var, q_res)
        ik.AddMinimumDistanceConstraint(min_dist, min_dist_thresh)

        current_time = datetime.datetime.now().strftime("%b_%d_%H_%M_%S")
        file_path: pathlib.Path = (
            solver_results_dir / f"ik_shelves_demo_w_collision_{current_time}.snopt"
        )

        prog.SetSolverOption(snopt_solver.solver_id(), "Major iterations limit", 5000)
        prog.SetSolverOption(snopt_solver.solver_id(), "Iterations limit", 5000)
        prog.SetSolverOption(snopt_solver.solver_id(), "Print file", str(file_path))
        prog.SetSolverOption(snopt_solver.solver_id(), "Time limit", 4)
        prog.SetSolverOption(snopt_solver.solver_id(), "Timing level", 2)

        result = snopt_solver.Solve(prog)
        q_res = result.GetSolution(q_var)
        if not result.is_success():
            logging.info("IK failure with minimum distance constraints added")
            logging.info(result.get_solver_id().name())
            logging.info(result.GetInfeasibleConstraints(prog))
            logging.info(result.GetInfeasibleConstraintNames(prog))
            feasibility_value, optimality_value, is_acceptably_feasible = (
                extract_feasiblity_optimality_values_from_snopt_log(file_path)
            )
            logging.info(f"feasibility_value: {feasibility_value}")
            logging.info(f"optimality_value: {optimality_value}")
            logging.info(f"solver return value: {result.get_solver_details().info}")
            logging.info(f"Optimal cost: {result.get_optimal_cost()}")
            file_path.unlink(missing_ok=True)

        if visualizer is not None:
            logging.info("Visualizing minimum distance constraint EE pose")

            plant_context = plant.GetMyContextFromRoot(context)
            plant.SetPositions(plant_context, q_res)

            visualizer_context = visualizer.GetMyContextFromRoot(context)
            visualizer.ForcedPublish(visualizer_context)

            if collision_visualizer is not None:
                collision_context = collision_visualizer.GetMyContextFromRoot(context)
                collision_visualizer.ForcedPublish(collision_context)

            if kp_to_P_goal is not None:
                for i, (frame_name, _) in enumerate(kp_to_P_goal.items()):
                    X_W_tip = plant.GetFrameByName(frame_name).CalcPoseInWorld(
                        plant_context
                    )
                    meshcat.SetObject(
                        frame_name,
                        Sphere(0.005),
                        rgba=Rgba(0.1 * i, 0.9 * i, 0.1 * i, 1),
                    )
                    meshcat.SetTransform(frame_name, X_W_tip)

        q_res = result.GetSolution()
        return q_res, result.is_success()

    @staticmethod
    def generate_trajopt_init_guess(
        robot_joint_qpos_start: np.ndarray,
        robot_joint_qpos_goal: np.ndarray,
        robot_gripper_qpos_start: np.ndarray,
        robot_gripper_qpos_goal: np.ndarray,
        robot_kinematic_ranges: Dict[str, np.ndarray],
        trajopt_num_control_points: int,
        trajopt_basis: BsplineBasis,
    ) -> BsplineTrajectory:
        """
        Maybe pass in the joint ranges here?
        """
        # Generate random waypoint configuration
        robot_joint_ranges = robot_kinematic_ranges["robot_joint_qpos"]
        robot_gipper_ranges = robot_kinematic_ranges["robot_gripper_qpos"]
        robot_joint_waypoint = np.random.uniform(
            robot_joint_ranges[:, 0], robot_joint_ranges[:, 1]
        )
        robot_gripper_waypoint = np.random.uniform(
            robot_gipper_ranges[:, 0], robot_gipper_ranges[:, 1]
        )

        # Initialize trajectory
        q_guess = np.zeros((9, trajopt_num_control_points))

        # Calculate the number of control points for each segment
        num_control_points_per_segment = trajopt_num_control_points // 2

        # Interpolate between start, waypoint, and goal for each joint and gripper parameter
        for i in range(9):
            if i < 7:
                q_guess[i, :] = np.concatenate(
                    [
                        np.linspace(
                            robot_joint_qpos_start[i],
                            robot_joint_waypoint[i],
                            num_control_points_per_segment,
                        ),
                        np.linspace(
                            robot_joint_waypoint[i],
                            robot_joint_qpos_goal[i],
                            trajopt_num_control_points - num_control_points_per_segment,
                        ),
                    ]
                )
            else:
                q_guess[i, :] = np.concatenate(
                    [
                        np.linspace(
                            robot_gripper_qpos_start[i - 7],
                            robot_gripper_waypoint[i - 7],
                            num_control_points_per_segment,
                        ),
                        np.linspace(
                            robot_gripper_waypoint[i - 7],
                            robot_gripper_qpos_goal[i - 7],
                            trajopt_num_control_points - num_control_points_per_segment,
                        ),
                    ]
                )

        # Create the initial guess for the trajectory
        return BsplineTrajectory(trajopt_basis, q_guess)

    @staticmethod
    def update_results(
        results: Dict[str, List[torch.Tensor]],
        joint_qpos: List[float],
        gripper_qpos: List[float],
        X_W_GripperSite_robomimic: RigidTransform,
        X_W_EEFSite_robomimic: RigidTransform,
        X_RobotBase_GripperSite_robomimic: RigidTransform,
        X_RobotBase_EEFSite_robomimic: RigidTransform,
        task_relev_obj_pos: np.ndarray,
        task_relev_obj_rot: np.ndarray,
    ) -> None:
        """
        Update the results dictionary with the current robot joint positions, EE poses and task relevant object poses
        """
        results["robot_joint_qpos"].append(torch.FloatTensor(joint_qpos))
        results["robot_gripper_qpos"].append(torch.FloatTensor(gripper_qpos))
        # gripper site w.r.t world
        results["robot_ee_pos_gripper_site_world"].append(
            torch.FloatTensor(X_W_GripperSite_robomimic.translation().copy())
        )
        results["robot_ee_quat_wxyz_gripper_site_world"].append(
            torch.FloatTensor(
                X_W_GripperSite_robomimic.rotation().ToQuaternion().wxyz().copy()
            )
        )
        results["robot_ee_rot_gripper_site_world"].append(
            torch.FloatTensor(X_W_GripperSite_robomimic.rotation().matrix().copy())
        )
        # EEF site w.r.t world
        results["robot_ee_pos_eef_site_world"].append(
            torch.FloatTensor(X_W_EEFSite_robomimic.translation().copy())
        )
        results["robot_ee_quat_wxyz_eef_site_world"].append(
            torch.FloatTensor(
                X_W_EEFSite_robomimic.rotation().ToQuaternion().wxyz().copy()
            )
        )
        results["robot_ee_rot_eef_site_world"].append(
            torch.FloatTensor(X_W_EEFSite_robomimic.rotation().matrix().copy())
        )

        results["robot_ee_pos_gripper_site_base"].append(
            torch.FloatTensor(X_RobotBase_GripperSite_robomimic.translation().copy())
        )

        # gripper site w.r.t robot base
        results["robot_ee_quat_wxyz_gripper_site_base"].append(
            torch.FloatTensor(
                X_RobotBase_GripperSite_robomimic.rotation()
                .ToQuaternion()
                .wxyz()
                .copy()
            )
        )
        results["robot_ee_rot_gripper_site_base"].append(
            torch.FloatTensor(
                X_RobotBase_GripperSite_robomimic.rotation().matrix().copy()
            )
        )
        # EEF site w.r.t robot base
        results["robot_ee_pos_eef_site_base"].append(
            torch.FloatTensor(X_RobotBase_EEFSite_robomimic.translation().copy())
        )
        results["robot_ee_quat_wxyz_eef_site_base"].append(
            torch.FloatTensor(
                X_RobotBase_EEFSite_robomimic.rotation().ToQuaternion().wxyz().copy()
            )
        )
        results["robot_ee_rot_eef_site_base"].append(
            torch.FloatTensor(X_RobotBase_EEFSite_robomimic.rotation().matrix().copy())
        )

        results["task_relev_obj_pos"].append(
            torch.FloatTensor(task_relev_obj_pos.copy())
        )
        results["task_relev_obj_rot"].append(
            torch.FloatTensor(task_relev_obj_rot.copy())
        )
        task_relev_obj_pose = np.eye(4)
        task_relev_obj_pose[:3, :3] = task_relev_obj_rot
        task_relev_obj_pose[:3, 3] = task_relev_obj_pos
        results["task_relev_obj_pose"].append(
            torch.FloatTensor(task_relev_obj_pose.copy())
        )

    @staticmethod
    def setup_and_solve_trajopt_prog(
        plant: MultibodyPlant,
        context: Context,
        robot_joint_qpos_start: np.ndarray,
        robot_gripper_qpos_start: np.ndarray,
        robot_joint_qpos_goal: np.ndarray,
        robot_gripper_qpos_goal: np.ndarray,
        meshcat: Optional[Meshcat] = None,
        visualizer: Optional[MeshcatVisualizer] = None,
        collision_visualizer: Optional[MeshcatVisualizer] = None,
        min_duration: float = 1.0,
        max_duration: float = 1.5,
        use_collision_constraint: bool = True,
        min_dist: float = 0.01,
        min_dist_thresh: float = 0.03,
        robot_kinematic_ranges: Optional[Dict[str, np.ndarray]] = None,
        num_init_guesses: int = 10,
        viz_trajopt_process: bool = False,
    ) -> Tuple[MathematicalProgramResult, KinematicTrajectoryOptimization]:
        """
        Args:
            min_dist: minimum allowed value of the signed distance between any candidate pair of geometries.
        """
        num_q = plant.num_positions()
        plant_context = plant.GetMyContextFromRoot(context)
        q_start = np.concatenate([robot_joint_qpos_start, robot_gripper_qpos_start])
        q_goal = np.concatenate([robot_joint_qpos_goal, robot_gripper_qpos_goal])

        trajopt = KinematicTrajectoryOptimization(plant.num_positions(), 15)
        prog = trajopt.get_mutable_prog()
        trajopt.AddDurationCost(1.0)
        trajopt.AddPathLengthCost(1.0)
        trajopt.AddPositionBounds(
            plant.GetPositionLowerLimits(), plant.GetPositionUpperLimits()
        )
        # trajopt.AddVelocityBounds(plant.GetVelocityLowerLimits(), plant.GetVelocityUpperLimits())
        # manually update velocity bounds to expand the orientation velocity bounds by 1 dimension
        # trajopt.AddVelocityBounds(np.concatenate([[-np.inf], plant.GetVelocityLowerLimits()]), np.concatenate([[-np.inf], plant.GetVelocityUpperLimits()]))

        # import ipdb; ipdb.set_trace()
        acc_lower_lims = -np.array([15, 7.5, 10, 12.5, 15, 20, 20, np.inf, np.inf])
        trajopt.AddAccelerationBounds(acc_lower_lims / 4, -acc_lower_lims / 4)
        trajopt.AddDurationConstraint(min_duration, max_duration)

        trajopt.AddPathPositionConstraint(q_start, q_start, 0)
        trajopt.AddPathPositionConstraint(q_goal, q_goal, 1)

        prog.AddQuadraticErrorCost(
            np.eye(num_q), q_start, trajopt.control_points()[:, 0]
        )
        prog.AddQuadraticErrorCost(
            np.eye(num_q), q_start, trajopt.control_points()[:, -1]
        )

        # start and end with zero velocity
        trajopt.AddPathVelocityConstraint(np.zeros((num_q, 1)), np.zeros((num_q, 1)), 0)
        trajopt.AddPathVelocityConstraint(np.zeros((num_q, 1)), np.zeros((num_q, 1)), 1)

        # Solve once without the collisions and set that as the initial guess for
        # the version with collisions.
        snopt_solver = SnoptSolver()

        # create folder for trajopt results using pathlib
        solver_results_dir = pathlib.Path("trajopt_results")
        solver_results_dir.mkdir(parents=True, exist_ok=True)

        current_time = datetime.datetime.now().strftime("%b_%d_%H_%M_%S")
        file_path: pathlib.Path = (
            solver_results_dir
            / f"trajopt_shelves_demo_no_collision_{current_time}.snopt"
        )

        prog.SetSolverOption(snopt_solver.solver_id(), "Major iterations limit", 5000)
        prog.SetSolverOption(snopt_solver.solver_id(), "Iterations limit", 5000)
        prog.SetSolverOption(snopt_solver.solver_id(), "Print file", str(file_path))
        prog.SetSolverOption(snopt_solver.solver_id(), "Time limit", 4)
        prog.SetSolverOption(snopt_solver.solver_id(), "Timing level", 2)
        # try the below
        # prog.SetSolverOption(snopt_solver.solver_id(), "verbose", (0, 1))

        def PlotPath(control_points):
            traj = BsplineTrajectory(
                trajopt.basis(), control_points.reshape((num_q, -1))
            )

            hand_robomimic_gripper_site_frame = plant.GetFrameByName(
                ROBOMIMIC_GRIPPER_SITE_FRAME_NAME
            )
            hand_positions = []

            times = np.linspace(0, 1, 50)
            for t in times:
                plant.SetPositions(plant_context, traj.vector_values([t]))
                X_W_GripperSite_robomimic = (
                    hand_robomimic_gripper_site_frame.CalcPoseInWorld(plant_context)
                )
                hand_positions.append(X_W_GripperSite_robomimic.translation())

            hand_positions = np.array(hand_positions).T
            meshcat.SetLine("hand_positions", hand_positions)

        if meshcat is not None and viz_trajopt_process:
            prog.AddVisualizationCallback(
                PlotPath, trajopt.control_points().reshape((-1,))
            )

        tic = time.time()
        result = snopt_solver.Solve(prog)
        toc = time.time()

        print("Time to solve [trajopt w/o collisions + time limit of 2s]: ", toc - tic)
        if not result.is_success():
            logging.info("Trajectory optimization failed, even without collisions!")
            logging.info(result.get_solver_id().name())
            logging.info(result.GetInfeasibleConstraints(prog))
            logging.info(result.GetInfeasibleConstraintNames(prog))
            feasibility_value, optimality_value, is_acceptably_feasible = (
                extract_feasiblity_optimality_values_from_snopt_log(file_path)
            )
            logging.info(f"feasibility_value: {feasibility_value}")
            logging.info(f"optimality_value: {optimality_value}")
            logging.info(f"solver return value: {result.get_solver_details().info}")
            logging.info(f"Optimal cost: {result.get_optimal_cost()}")

        file_path.unlink(missing_ok=True)

        if visualizer is not None:
            PublishPositionTrajectory(
                trajopt.ReconstructTrajectory(result), context, plant, visualizer
            )
        if collision_visualizer is not None:
            collision_visualizer.ForcedPublish(
                collision_visualizer.GetMyContextFromRoot(context)
            )

        if not use_collision_constraint:
            return result, trajopt

        trajopt.SetInitialGuess(trajopt.ReconstructTrajectory(result))
        # TODO(klin): update the minimum distance constraint to be the minimum
        # distance between EE and objects at the final goal?
        # other way is to have gripper be extra wide rather than trying to be optimal?
        collision_constraint = MinimumDistanceConstraint(
            plant, min_dist, plant_context, None, min_dist_thresh
        )
        evaluate_at_s = np.linspace(0, 1, 50)

        # add more constraints near the end and append to evaluate_at_s
        evaluate_at_s = np.append(evaluate_at_s, np.linspace(0.81, 0.99, 19))
        for s in evaluate_at_s:
            trajopt.AddPathPositionConstraint(collision_constraint, s)

        # create folder for trajopt results using pathlib
        solver_results_dir = pathlib.Path("trajopt_results")
        solver_results_dir.mkdir(parents=True, exist_ok=True)

        current_time = datetime.datetime.now().strftime("%b_%d_%H_%M_%S")
        file_path: pathlib.Path = (
            solver_results_dir / f"trajopt_shelves_demo_collision_{current_time}.snopt"
        )
        prog.SetSolverOption(snopt_solver.solver_id(), "Major iterations limit", 1000)
        prog.SetSolverOption(snopt_solver.solver_id(), "Iterations limit", 1000)
        prog.SetSolverOption(snopt_solver.solver_id(), "Print file", str(file_path))
        prog.SetSolverOption(snopt_solver.solver_id(), "Time limit", 15)
        prog.SetSolverOption(snopt_solver.solver_id(), "Timing level", 2)

        for i in range(num_init_guesses):
            if i > 1:
                init_guess_traj = DrakeMotionPlanner.generate_trajopt_init_guess(
                    robot_joint_qpos_start,
                    robot_joint_qpos_goal,
                    robot_gripper_qpos_start,
                    robot_gripper_qpos_goal,
                    robot_kinematic_ranges,
                    trajopt.num_control_points(),
                    trajopt.basis(),
                )

                trajopt.SetInitialGuess(init_guess_traj)

            current_time = datetime.datetime.now().strftime("%b_%d_%H_%M_%S")
            file_path: pathlib.Path = (
                solver_results_dir
                / f"trajopt_shelves_demo_collision_{current_time}.snopt"
            )
            prog.SetSolverOption(snopt_solver.solver_id(), "Print file", str(file_path))

            tic = time.perf_counter()
            result = snopt_solver.Solve(prog)
            toc = time.perf_counter()
            print(
                "Time to solve [trajopt w/ collisions + time limit of 15s]: ", toc - tic
            )

            if not result.is_success():
                logging.info(
                    "Trajectory optimization with collisions has result.is_success() == False"
                    " Checking if feasibility and optimality values are acceptable ..."
                )
                tic = time.perf_counter()
                feasibility_value, optimality_value, is_acceptably_feasible = (
                    extract_feasiblity_optimality_values_from_snopt_log(file_path)
                )
                toc = time.perf_counter()
                file_path.unlink(missing_ok=True)

                print("Time to extract feasibility and optimality values: ", toc - tic)
                logging.info(f"Optimal cost: {result.get_optimal_cost()}")
                logging.info(f"feasibility_value: {feasibility_value}")
                logging.info(f"optimality_value: {optimality_value}")
                logging.info(f"solver return value: {result.get_solver_details().info}")
                if (
                    is_acceptably_feasible
                    and optimality_value < SNOPT_SOLVER_MAX_OPTIMALITY_VALUE
                ):
                    logging.info(
                        "Feasible and optimal values are acceptable. Using current result."
                    )
                    result.set_solution_result(SolutionResult.kSolutionFound)
                    return result, trajopt
                else:
                    logging.info(
                        "Feasible and optimal values are not acceptable. Trying again ..."
                    )
                logging.info(result.get_solver_id().name())
                logging.info(result.GetInfeasibleConstraints(prog))
                logging.info(result.GetInfeasibleConstraintNames(prog))
            else:
                file_path.unlink(missing_ok=True)

                print("Trajectory optimization with collision constraints succeeded!")
                logging.info(
                    "Trajectory optimization with collision constraints succeeded!"
                )
                return result, trajopt

            if visualizer is not None:
                PublishPositionTrajectory(
                    trajopt.ReconstructTrajectory(result), context, plant, visualizer
                )
            if collision_visualizer is not None:
                collision_visualizer.ForcedPublish(
                    collision_visualizer.GetMyContextFromRoot(context)
                )

        return result, trajopt

    def setup_env(
        self,
        add_robot: bool = True,
        add_robot_hand: bool = False,
        name_to_frame_info: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
        robot_base_pos: Optional[np.ndarray] = None,
        robot_base_quat_wxyz: Optional[np.ndarray] = None,
        task_relev_obj_pos_nerf: Optional[List[np.ndarray]] = None,
        task_relev_obj_rot_nerf: Optional[List[np.ndarray]] = None,
        obj_to_init_info: Optional[Dict[str, ObjectInitializationInfo]] = None,
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
        builder = DiagramBuilder()

        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
        parser = Parser(plant)
        parser.package_map().AddPackageXml(filename=self.drake_package_path)

        if add_robot:
            X_WRobotBase = RigidTransform(
                Quaternion(robot_base_quat_wxyz), robot_base_pos
            )
            DrakeMotionPlanner.AddFranka(parser, plant, body_world_pose=X_WRobotBase)
            parser.AddModelsFromUrl(self.task_irrelev_obj_url)

        if add_robot_hand:
            DrakeMotionPlanner.AddFrankaHand(parser, plant)

        if name_to_frame_info is not None:
            for name, frame_info in name_to_frame_info.items():
                X_W_FRAME = RigidTransform(
                    Quaternion(np.array(frame_info["offset_quat_wxyz"])),
                    np.array(frame_info["offset_pos"]),
                )
                plant.AddFrame(
                    FixedOffsetFrame(
                        name,
                        plant.GetFrameByName(frame_info["src_frame_of_offset_frame"]),
                        X_W_FRAME,
                    )
                )

        # Previous hardcoded stuff TODO(klin): integrate to the name_to_frame_info dict
        # Add a frame to the panda hand body
        hand = plant.GetBodyByName("panda_hand")
        X_Hand_Grippersite = RigidTransform(DRAKE_TO_ROBOMIMIC_GRIPPER_OFFSET_POS)
        # only need pos offset due to difference between drake and robomimic franka frame definitions
        plant.AddFrame(
            FixedOffsetFrame(
                ROBOMIMIC_EE_SITE_FRAME_NAME, hand.body_frame(), X_Hand_Grippersite
            )
        )
        X_Hand_Grippersite_robomimic = RigidTransform(
            Quaternion(ROBOMIMIC_HAND_GRIPPER_OFFSET_QUAT_WXYZ),
            DRAKE_TO_ROBOMIMIC_GRIPPER_OFFSET_POS,
        )
        # Frame to use for extracting robot EEF poses
        plant.AddFrame(
            FixedOffsetFrame(
                ROBOMIMIC_GRIPPER_SITE_FRAME_NAME,
                hand.body_frame(),
                X_Hand_Grippersite_robomimic,
            )
        )

        if obj_to_init_info is not None:
            for obj_name, obj_init_info in obj_to_init_info.items():
                X_parentframe_obj = RigidTransform(obj_init_info.X_parentframe_obj)
                if not obj_init_info.weld_to_ee:
                    parent_frame_name = None
                else:
                    parent_frame_name = ROBOMIMIC_EE_SITE_FRAME_NAME

                success = DrakeMotionPlanner.AddTaskRelevObj(
                    parser,
                    plant,
                    X_parentframe_obj,
                    obj_init_info.mesh_paths.sdf_path,
                    parent_frame_name,
                    add_to_parent_geometry=obj_init_info.weld_to_ee,
                )

                if not success:
                    return None, None, None

        plant.Finalize()

        return plant, builder, scene_graph

    def get_optimal_trajectory(
        self,
        start_cfg: RobotEnvConfig,
        goal_cfg: RobotEnvConfig,
        motion_planner_type: MotionPlannerType = MotionPlannerType.KINEMATIC_TRAJECTORY_OPTIMIZATION,
        obj_to_init_info: Optional[Dict[str, ObjectInitializationInfo]] = None,
        # task_relev_obj_path: str,
        default_robot_joint_qpos: Optional[List] = None,
        default_robot_goal_joint_pos: Optional[List] = None,
        enforce_reach_traj_gripper_non_close: bool = False,
        use_collision_free_waypoint_heuristic: bool = False,
        collision_free_waypoint_threshold: float = 0.06,
        collision_free_waypoint_rotation_angle_bound: float = np.pi / 4,
        open_gripper_at_waypoint: bool = True,
        collision_free_waypoint_sampling_radius: float = 0.015,
        interp_wp_to_goal_traj_eef: bool = True,
        interp_wp_to_goal_traj_joint: bool = False,
        num_trajopt_guesses: int = 10,
        convert_last_gripper_qpos_entry_negative: bool = True,
        start_at_collision_free_waypoint: bool = False,
    ) -> Optional[Dict[str, List]]:
        """
        Compute the optimal trajectory from the start configuration to the goal configuration
        based on the given RobotEnvConfigs.

        Return dictionary containing the optimal trajectory. If the function cannot
        calculate the optimal trajectory, it raises a NotImplementedError.

        default_robot_joint_qpos: optional list of default robot joint qpos
            used to encourage initial IK solution to be close to the goal
            however, usually start cfg's robot joint qpos corresponds to
            a randomly sampled joint qpos that's already close to the default joint qpos

        note that the ee_pose in start_cfg assumes a certain frame.
        Currently, assumes gripper_site frame. May need to adjust e.g. for different end effector.

        Args:
            start_cfg (RobotEnvConfig): The start configuration.
            enforce_reach_traj_gripper_non_close (bool): Whether to have the gripper always be open
                throughout the trajectory.
            use_collision_free_waypoint_heuristic (bool): Whether to use a collision free waypoint in
                the trajectory. Heuristic to
                help motion planner not have to plan *towards* very near-collision states. Specifically,
                for a given goal eef pose, choose this waypoint to be collision_free_waypoint_threshold
                in the negative z-axis (in the eef pose frame, not world frame) direction. Assumes z axis
                of eef pose frame is the direction 'out' of the gripper.
            collision_free_waypoint_threshold (float): The distance to move the gripper (in the goal gripper frame)
                in the negative z-axis direction
            collision_free_waypoint_rotation_angle_bound (float):
                The angle bound (in radians) to rotate the gripper at the waypoint.
            start_at_collision_free_waypoint (bool): Whether to start the trajectory at the collision free
                waypoint (and skip the provided start_cfg).

            interp_wp_to_goal_traj_joint (bool): Whether to interpolate the waypoint to goal trajectory in joint space.
            interp_wp_to_goal_traj_eef (bool): Whether to interpolate the waypoint to goal trajectory in eef space.

        TODO(klin): implement collision_free_waypoint_rotation; would be good to start trajectories from random rotations
        instead of setting the goal to be some rotation.

        Returns:
            results: Dict[str, Union[np.ndarray]]
                keys include: robot_ee_pos_eef_site_world, robot_joint_angles
        """
        overall_cost: float = 0

        if use_collision_free_waypoint_heuristic:
            assert (
                (interp_wp_to_goal_traj_joint and not interp_wp_to_goal_traj_eef)
                or (interp_wp_to_goal_traj_eef and not interp_wp_to_goal_traj_joint)
            ), "Can only interpolate waypoint to goal trajectory in one of either joint or eef space."
        plant, builder, scene_graph = self.setup_env(
            add_robot=True,
            add_robot_hand=False,
            robot_base_pos=start_cfg.robot_base_pos,
            robot_base_quat_wxyz=start_cfg.robot_base_quat_wxyz,
            task_relev_obj_pos_nerf=[start_cfg.task_relev_obj_pos_nerf],
            task_relev_obj_rot_nerf=[start_cfg.task_relev_obj_rot_nerf],
            obj_to_init_info=obj_to_init_info,
        )

        if plant is None:
            logging.info("Failed to setup environment")
            return None

        # may need to define more frames here
        # (for EEs can predefine things; for other objects, may need user to add frames)
        if self.view_meshcat:
            visualizer = MeshcatVisualizer.AddToBuilder(
                builder,
                scene_graph,
                self.meshcat,
                MeshcatVisualizerParams(role=Role.kIllustration),
            )
            collision_visualizer = MeshcatVisualizer.AddToBuilder(
                builder,
                scene_graph,
                self.meshcat,
                MeshcatVisualizerParams(prefix="collision", role=Role.kProximity),
            )
            self.meshcat.SetProperty("collision", "visible", False)
        else:
            visualizer = None
            collision_visualizer = None

        diagram = builder.Build()
        context = diagram.CreateDefaultContext()

        if obj_to_init_info is not None:
            # Exclude collisions between welded object and end effector
            plant_context = plant.GetMyContextFromRoot(context)
            scene_graph_context = scene_graph.GetMyContextFromRoot(context)

            robot_eef_body_names: List[str] = [
                "panda_hand",
                "panda_leftfinger",
                "panda_rightfinger",
                "panda_link5",
                "panda_link6",
                "panda_link7",
                "panda_link8",
            ]
            robot_eef_geometry_ids: List[int] = []
            for body_name in robot_eef_body_names:
                robot_eef_geometry_ids.extend(
                    plant.GetCollisionGeometriesForBody(plant.GetBodyByName(body_name))
                )

            for obj, init_info in obj_to_init_info.items():
                if init_info.weld_to_ee:
                    # parse out the object name from init_info['mesh_path']
                    assert pathlib.Path(init_info.mesh_path).suffix == ".sdf", (
                        "Assumes that the mesh that's welded to the EEF is in an .sdf file"
                        " (so that we can parse out the object name from the file name)"
                        "If there is no model name in the sdf file or if mesh_path is not a .obj file,"
                        "figure out how to get the model name another way."
                    )
                    body_name = extract_model_name_from_sdf(init_info.mesh_path)
                    object_geometry_ids = plant.GetCollisionGeometriesForBody(
                        plant.GetBodyByName(body_name)
                    )

                    robot_hand_geometry_set = GeometrySet(robot_eef_geometry_ids)
                    object_geometry_set = GeometrySet(object_geometry_ids)

                    collision_filter_manager = scene_graph.collision_filter_manager(
                        scene_graph_context
                    )
                    collision_filter_manager.Apply(
                        CollisionFilterDeclaration().ExcludeBetween(
                            robot_hand_geometry_set, object_geometry_set
                        )
                    )

        # force update the visualizer
        if self.view_meshcat:
            visualizer_context = visualizer.GetMyContextFromRoot(context)
            visualizer.ForcedPublish(visualizer_context)
            collision_context = collision_visualizer.GetMyContextFromRoot(context)
            collision_visualizer.ForcedPublish(collision_context)

        if start_cfg.robot_joint_qpos is None:
            assert start_cfg.robot_ee_pos is not None and (
                start_cfg.robot_ee_rot is not None
                or start_cfg.robot_ee_quat_wxyz is not None
            ), (
                "if start_cfg.robot_joint_qpos is None, need to provide robot_ee_pos and (robot_ee_rot or"
                " robot_ee_quat_wxyz)"
            )
            # perform IK on start_cfg.robot

            robot_qpos, is_success = DrakeMotionPlanner.get_IK(
                plant,
                context,
                RigidTransform(
                    Quaternion(
                        start_cfg.robot_ee_rot
                        if start_cfg.robot_ee_rot is not None
                        else start_cfg.robot_ee_quat_wxyz
                    ),
                    start_cfg.robot_ee_pos,
                ),
                ee_frame_name=ROBOMIMIC_EE_SITE_FRAME_NAME,
                constraint_orientation=False,  # don't constrain orientation for starting pose?
                initial_robot_joint_qpos=default_robot_joint_qpos,
                initial_robot_gripper_qpos=start_cfg.robot_gripper_qpos,
                goal_robot_gripper_qpos=start_cfg.robot_gripper_qpos,
                visualizer=visualizer,
                collision_visualizer=collision_visualizer,
                meshcat=self.meshcat,
                # min_dist=0.01,
                min_dist=0.005,
                check_collisions=use_collision_free_waypoint_heuristic,  # check collision at start pose
                # check_collisions=(
                #     False if use_collision_free_waypoint_heuristic else True
                # ),  # if start_at_collision_free_waypoint, only check collisions at the waypoint
            )

            if not is_success:
                # Todo(klin): debug weird IK failure for lift
                # import ipdb

                # ipdb.set_trace()
                return None

            start_cfg_robot_joint_qpos = robot_qpos[:-2]
        else:
            start_cfg_robot_joint_qpos = start_cfg.robot_joint_qpos

        # get a goal joint  cfg for a given robot eef pose
        # can have extra IK results for the EE goal orientation constraint
        # could try different orientations (more than symmetry about fingers) but riskier

        # self.meshcat.SetObject("start", Sphere(0.02), rgba=Rgba(0.9, 0.1, 0.1, 1))
        # self.meshcat.SetTransform("start", RigidTransform(start_cfg.robot_ee_pos))
        # self.meshcat.SetObject("goal", Sphere(0.02), rgba=Rgba(0.1, 0.9, 0.1, 1))
        # self.meshcat.SetTransform("goal", RigidTransform(goal_cfg.robot_ee_pos))
        robot_qpos_goal, is_success = DrakeMotionPlanner.get_IK(
            plant,
            context,
            RigidTransform(
                Quaternion(
                    goal_cfg.robot_ee_rot
                    if goal_cfg.robot_ee_rot is not None
                    else goal_cfg.robot_ee_quat_wxyz
                ),
                goal_cfg.robot_ee_pos,
            ),
            visualizer=visualizer,
            collision_visualizer=collision_visualizer,
            ee_frame_name=ROBOMIMIC_EE_SITE_FRAME_NAME,
            initial_robot_joint_qpos=default_robot_goal_joint_pos,
            initial_robot_gripper_qpos=start_cfg.robot_gripper_qpos,
            goal_robot_gripper_qpos=goal_cfg.robot_gripper_qpos,
            check_collisions=False,  # should still check collisions for non-gripper?
            min_dist=0.025,
        )

        assert np.all(robot_qpos_goal[-2:] >= 0), "gripper qpos should be non-negative"

        # hmmm think I still need to check collisions somehow / for some objects esp.
        # if doing the rotating grippper thing

        if not is_success:
            # import ipdb; ipdb.set_trace()
            logging.info("IK failed ... goal_cfg is likely invalid")
            return None

        # use_collision_free_waypoint_heuristic = False
        if use_collision_free_waypoint_heuristic:
            ee_rot_mat = Quaternion(
                goal_cfg.robot_ee_rot
                if goal_cfg.robot_ee_rot is not None
                else goal_cfg.robot_ee_quat_wxyz
            ).rotation()
            z_displacement = -np.array([0, 0, 1]) * collision_free_waypoint_threshold
            X_W_displacement = ee_rot_mat.dot(z_displacement)
            collision_free_waypoint = goal_cfg.robot_ee_pos + X_W_displacement
            collision_free_waypoint[:2] += np.random.uniform(
                -collision_free_waypoint_sampling_radius,
                collision_free_waypoint_sampling_radius,
                size=2,
            )
            # Randomly sample a rotation
            rand_rot = R.from_matrix(
                random_z_rotation(
                    collision_free_waypoint_rotation_angle_bound,
                    collision_free_waypoint_rotation_angle_bound,
                )
            )
            orig_goal_rot = (
                R.from_matrix(goal_cfg.robot_ee_rot)
                if goal_cfg.robot_ee_rot is not None
                else R.from_quat(
                    np.concatenate(
                        [
                            goal_cfg.robot_ee_quat_wxyz[1:],
                            [goal_cfg.robot_ee_quat_wxyz[0]],
                        ]
                    )
                )
            )

            collision_free_wp_rot = (rand_rot * orig_goal_rot).as_matrix()

            robot_qpos_waypoint, is_success = DrakeMotionPlanner.get_IK(
                plant,
                context,
                RigidTransform(
                    Quaternion(collision_free_wp_rot),
                    collision_free_waypoint,
                ),
                meshcat=self.meshcat,
                visualizer=visualizer,
                collision_visualizer=collision_visualizer,
                ee_frame_name=ROBOMIMIC_EE_SITE_FRAME_NAME,
                initial_robot_joint_qpos=robot_qpos_goal[
                    :-2
                ],  # hardcoded assumption that gripper is last 2 joints
                # initial_robot_joint_qpos=start_cfg_robot_joint_qpos,
                initial_robot_gripper_qpos=start_cfg.robot_gripper_qpos,
                # goal_robot_gripper_qpos=(
                #     np.array([0.04, 0.04]) if open_gripper_at_waypoint else start_cfg.robot_gripper_qpos
                # ),
                constraint_orientation=True,  # don't constrain orientation for waypoint
                # (for now, for the door opening task at least)
                goal_robot_gripper_qpos=start_cfg.robot_gripper_qpos,
                min_dist=0.0,  # should be empirically determined based on the min distance between gripper and object?
            )

            assert np.all(
                robot_qpos_waypoint[-2:] >= 0
            ), "gripper qpos should be non-negative"

            if not is_success:
                logging.info(
                    "IK failed ... collision-free waypoint calculation failed. Is the problem impossible? "
                    "Probably not. Try increasing collision_free_waypoint_threshold?"
                    "Issue could be: gripper too close to the object as object wasn't 3D segmented properly."
                )
                plant_context = plant.GetMyContextFromRoot(context)
                plant.SetPositions(plant_context, robot_qpos_waypoint)

                if self.view_meshcat:
                    visualizer_context = visualizer.GetMyContextFromRoot(context)
                    visualizer.ForcedPublish(visualizer_context)
                    self.meshcat.SetObject(
                        "start", Sphere(0.02), rgba=Rgba(0.9, 0.1, 0.1, 1)
                    )
                    self.meshcat.SetTransform(
                        "start", RigidTransform(collision_free_waypoint)
                    )

                return None

            # TODO(klin): check what minimum distance between robot and environment at start actually is!
            # little hack include manually setting the gripper back to robot_qpos --- kind of voids the min_dist check
            # one way is to check the min distance for the config being passed in to setup_and_solve_trajopt_prog
            # probably more robust to do so? also should fix the gripper qpos hack ... how?

            if (
                motion_planner_type
                == MotionPlannerType.KINEMATIC_TRAJECTORY_OPTIMIZATION
            ):
                # start debugging from here - think this trajopt should succeed??
                result, trajopt = DrakeMotionPlanner.setup_and_solve_trajopt_prog(
                    plant,
                    context,
                    start_cfg_robot_joint_qpos,
                    start_cfg.robot_gripper_qpos,
                    robot_qpos_waypoint[:-2],
                    robot_qpos_waypoint[-2:],
                    meshcat=self.meshcat,
                    visualizer=visualizer,
                    collision_visualizer=collision_visualizer,
                    min_dist=0.01,
                    min_dist_thresh=0.05,
                    min_duration=0.01,
                    max_duration=2,
                    use_collision_constraint=True,  # TODO(klin) convert back to True after test go down
                    robot_kinematic_ranges=self.robot_obj.kinematic_ranges,
                    num_init_guesses=num_trajopt_guesses,
                    viz_trajopt_process=True,
                )

                overall_cost += result.get_optimal_cost()

                start_to_waypoint_traj = trajopt.ReconstructTrajectory(result)
                if self.view_meshcat:
                    PublishPositionTrajectory(
                        start_to_waypoint_traj, context, plant, visualizer
                    )
                    collision_visualizer.ForcedPublish(
                        collision_visualizer.GetMyContextFromRoot(context)
                    )

                if not result.is_success():
                    logging.info(
                        "start_to_waypoint_traj Trajopt failed ... view meshcat?"
                    )
                    # Perhaps have option to allow saving of the data or updating of parameters?
                    return None

                plant_context = plant.GetMyContextFromRoot(context)

                timesteps = get_timesteps(
                    start_to_waypoint_traj.end_time(), 1 / self.action_freq
                )
                results: Dict[str, List] = defaultdict(list)

                if start_at_collision_free_waypoint:
                    timesteps = [start_to_waypoint_traj.end_time()]

                for idx, t in enumerate(timesteps):
                    plant.SetPositions(
                        plant_context, start_to_waypoint_traj.vector_values([t])
                    )
                    joint_qpos = start_to_waypoint_traj.vector_values([t])[:7, 0]
                    gripper_qpos = start_to_waypoint_traj.vector_values([t])[7:, 0]
                    if convert_last_gripper_qpos_entry_negative:
                        gripper_qpos[-1] *= -1  # correction for mujoco
                    if enforce_reach_traj_gripper_non_close:
                        # check that curren gripper qpos is not closing w.r.t previous gripper qpos
                        if idx > 0:
                            gripper_qpos[-1] = min(
                                gripper_qpos[-1],
                                start_to_waypoint_traj.vector_values(
                                    [timesteps[idx - 1]]
                                )[7:, 0][-1]
                                * -1,
                            )
                            gripper_qpos[-2] = max(
                                gripper_qpos[-2],
                                start_to_waypoint_traj.vector_values(
                                    [timesteps[idx - 1]]
                                )[7:, 0][-2],
                            )

                    hand_robomimic_gripper_site_frame = plant.GetFrameByName(
                        ROBOMIMIC_GRIPPER_SITE_FRAME_NAME
                    )
                    X_W_GripperSite_robomimic = (
                        hand_robomimic_gripper_site_frame.CalcPoseInWorld(plant_context)
                    )
                    base_frame = plant.GetFrameByName(ROBOT_BASE_FRAME_NAME)
                    X_RobotBase_GripperSite_robomimic = (
                        hand_robomimic_gripper_site_frame.CalcPose(
                            plant_context, base_frame
                        )
                    )

                    hand_robomimic_eef_site_frame = plant.GetFrameByName(
                        ROBOMIMIC_EE_SITE_FRAME_NAME
                    )
                    X_W_EEFSite_robomimic = (
                        hand_robomimic_eef_site_frame.CalcPoseInWorld(plant_context)
                    )
                    X_RobotBase_EEFSite_robomimic = (
                        hand_robomimic_eef_site_frame.CalcPose(
                            plant_context, base_frame
                        )
                    )
                    DrakeMotionPlanner.update_results(
                        results,
                        joint_qpos,
                        gripper_qpos,
                        X_W_GripperSite_robomimic,
                        X_W_EEFSite_robomimic,
                        X_RobotBase_GripperSite_robomimic,
                        X_RobotBase_EEFSite_robomimic,
                        goal_cfg.task_relev_obj_pos,
                        goal_cfg.task_relev_obj_rot,
                    )
            elif motion_planner_type == MotionPlannerType.LINEAR_INTERPOLATION:
                results: Dict[str, List] = defaultdict(list)
                # Set initial robot positions
                plant.SetPositions(
                    plant_context,
                    np.concatenate(
                        [start_cfg_robot_joint_qpos, start_cfg.robot_gripper_qpos]
                    ),
                )
                # Get start position and rotation for end effector
                hand_robomimic_gripper_site_frame = plant.GetFrameByName(
                    ROBOMIMIC_GRIPPER_SITE_FRAME_NAME
                )
                X_W_GripperSite_start = (
                    hand_robomimic_gripper_site_frame.CalcPoseInWorld(plant_context)
                )
                hand_robomimic_eef_site_frame = plant.GetFrameByName(
                    ROBOMIMIC_EE_SITE_FRAME_NAME
                )
                X_W_EEFSite_start = hand_robomimic_eef_site_frame.CalcPoseInWorld(
                    plant_context
                )
                start_eef_pos = X_W_GripperSite_start.translation()
                start_eef_rot = X_W_GripperSite_start.rotation().matrix()
                start_eef_pos_eef_site = X_W_EEFSite_start.translation()
                start_eef_rot_eef_site = X_W_EEFSite_start.rotation().matrix()

                # Set waypoint robot positions
                plant.SetPositions(plant_context, robot_qpos_waypoint)
                # Get end effector position and rotation at waypoint
                hand_robomimic_gripper_site_frame = plant.GetFrameByName(
                    ROBOMIMIC_GRIPPER_SITE_FRAME_NAME
                )
                hand_robomimic_eef_site_frame = plant.GetFrameByName(
                    ROBOMIMIC_EE_SITE_FRAME_NAME
                )
                base_frame = plant.GetFrameByName(ROBOT_BASE_FRAME_NAME)

                X_W_GripperSite_wp = hand_robomimic_gripper_site_frame.CalcPoseInWorld(
                    plant_context
                )
                X_W_EEFSite_wp = hand_robomimic_eef_site_frame.CalcPoseInWorld(
                    plant_context
                )
                wp_eef_pos = X_W_GripperSite_wp.translation()
                wp_eef_rot = X_W_GripperSite_wp.rotation().matrix()
                wp_eef_pos_eef_site = X_W_EEFSite_wp.translation()
                wp_eef_rot_eef_site = X_W_EEFSite_wp.rotation().matrix()

                if not start_at_collision_free_waypoint:
                    # Add code to linear interpolate from start to waypoint
                    # Interpolation setup
                    overall_cost += rotation_difference_angle(start_eef_rot, wp_eef_rot)

                    interp_total_time = get_start_to_goal_time(
                        start_eef_pos, wp_eef_pos, start_eef_rot, wp_eef_rot
                    )
                    interp_start_end_times = np.array([0, interp_total_time])

                    positions = np.array([start_eef_pos, wp_eef_pos])
                    rotations = st.Rotation.from_matrix(
                        np.array([start_eef_rot, wp_eef_rot])
                    )
                    positions_eef_site = np.array(
                        [start_eef_pos_eef_site, wp_eef_pos_eef_site]
                    )
                    rotations_eef_site = st.Rotation.from_matrix(
                        np.array([start_eef_rot_eef_site, wp_eef_rot_eef_site])
                    )
                    grip_qposs = np.array([start_cfg.robot_gripper_qpos, [0.04, 0.04]])

                    rot_interp = st.Slerp(interp_start_end_times, rotations)
                    pos_interp = si.interp1d(
                        interp_start_end_times, positions, axis=0, assume_sorted=True
                    )
                    rot_interp_eef_site = st.Slerp(
                        interp_start_end_times, rotations_eef_site
                    )
                    pos_interp_eef_site = si.interp1d(
                        interp_start_end_times,
                        positions_eef_site,
                        axis=0,
                        assume_sorted=True,
                    )
                    grip_qpos_interp = si.interp1d(
                        interp_start_end_times, grip_qposs, axis=0, assume_sorted=True
                    )

                    interp_times = get_timesteps(
                        interp_total_time, 1 / self.action_freq
                    )

                    for i, interp_time in enumerate(interp_times):
                        if i == 0:
                            prev_robot_joint_qpos = start_cfg_robot_joint_qpos
                            prev_robot_gripper_qpos = start_cfg.robot_gripper_qpos

                        target_eef_rot = rot_interp(interp_time).as_matrix()
                        target_eef_pos = pos_interp(interp_time)
                        target_gripper_qpos = grip_qpos_interp(interp_time)

                        robot_qpos_ik, is_success = DrakeMotionPlanner.get_IK(
                            plant,
                            context,
                            RigidTransform(
                                RotationMatrix(target_eef_rot), target_eef_pos
                            ),
                            ee_frame_name=ROBOMIMIC_GRIPPER_SITE_FRAME_NAME,
                            initial_robot_joint_qpos=prev_robot_joint_qpos,
                            initial_robot_gripper_qpos=prev_robot_gripper_qpos,
                            goal_robot_gripper_qpos=target_gripper_qpos,
                            check_collisions=True,
                            min_dist=0.002,
                            min_dist_thresh=0.01,
                            visualizer=visualizer,
                            collision_visualizer=collision_visualizer,
                            meshcat=self.meshcat,
                        )
                        # maybe the min dist for gripper is 0.002 and min dist for other links could be larger?

                        if not is_success:
                            logging.info(
                                "IK failed in linear interpolation 'motion planning' start to wp: view meshcat?"
                            )
                            # Perhaps have option to allow saving of the data or updating of parameters?
                            return None

                        X_W_GripperSite_robomimic = RigidTransform(
                            RotationMatrix(rot_interp(interp_time).as_matrix()),
                            pos_interp(interp_time),
                        )
                        X_W_EEFSite_robomimic = RigidTransform(
                            RotationMatrix(
                                rot_interp_eef_site(interp_time).as_matrix()
                            ),
                            pos_interp_eef_site(interp_time),
                        )

                        X_RobotBase_GripperSite_robomimic = (
                            hand_robomimic_gripper_site_frame.CalcPose(
                                plant_context, base_frame
                            )
                        )
                        X_RobotBase_EEFSite_robomimic = (
                            hand_robomimic_eef_site_frame.CalcPose(
                                plant_context, base_frame
                            )
                        )

                        task_relev_obj_pos = goal_cfg.task_relev_obj_pos
                        task_relev_obj_rot = goal_cfg.task_relev_obj_rot

                        prev_robot_joint_qpos = np.copy(robot_qpos_ik[:7])
                        prev_robot_gripper_qpos = np.copy(robot_qpos_ik[7:])

                        # ensure second element is negative
                        if convert_last_gripper_qpos_entry_negative:
                            robot_qpos_ik[-1] *= -1

                        # should check gripper isn't closing?
                        DrakeMotionPlanner.update_results(
                            results,
                            robot_qpos_ik[:7],  # joint_qpos
                            robot_qpos_ik[7:],  # gripper_qpos
                            X_W_GripperSite_robomimic,
                            X_W_EEFSite_robomimic,
                            X_RobotBase_GripperSite_robomimic,
                            X_RobotBase_EEFSite_robomimic,
                            task_relev_obj_pos,
                            task_relev_obj_rot,
                        )

            # here and avoid singularities online
            if interp_wp_to_goal_traj_eef:
                interp_times = [
                    0,
                    0.25,
                    0.5,
                    0.75,
                    0.99,
                ]  # trying keep gripper open here
                goal_eef_pos = torch.FloatTensor(
                    goal_cfg.robot_ee_pos
                )  # eef site and gripper site pos are the same
                torch.FloatTensor(
                    Quaternion(
                        goal_cfg.robot_ee_rot
                        if goal_cfg.robot_ee_rot is not None
                        else goal_cfg.robot_ee_quat_wxyz
                    ).rotation()
                )
                torch.FloatTensor(
                    goal_cfg.robot_ee_rot
                    if goal_cfg.robot_ee_rot is not None
                    else R.from_quat(
                        np.concatenate(
                            [
                                goal_cfg.robot_ee_quat_wxyz[1:],
                                [goal_cfg.robot_ee_quat_wxyz[0]],
                            ]
                        )
                    ).as_matrix()
                )

                # set plant position to robot_qpos_goal then read out the eef pos and
                #  rot in both gripper and eef site frames
                plant.SetPositions(plant_context, robot_qpos_goal)
                # get end effector position and rotation at goal
                hand_robomimic_gripper_site_frame = plant.GetFrameByName(
                    ROBOMIMIC_GRIPPER_SITE_FRAME_NAME
                )
                X_W_GripperSite_goal = (
                    hand_robomimic_gripper_site_frame.CalcPoseInWorld(plant_context)
                )
                hand_robomimic_eef_site_frame = plant.GetFrameByName(
                    ROBOMIMIC_EE_SITE_FRAME_NAME
                )
                X_W_EEFSite_goal = hand_robomimic_eef_site_frame.CalcPoseInWorld(
                    plant_context
                )
                goal_eef_pos = X_W_GripperSite_goal.translation()
                goal_eef_rot = X_W_GripperSite_goal.rotation().matrix()
                X_W_EEFSite_goal.translation()
                goal_eef_rot_eef_site = X_W_EEFSite_goal.rotation().matrix()

                if "robot_ee_pos_gripper_site_world" in results:
                    wp_eef_pos = (
                        results["robot_ee_pos_gripper_site_world"][-1]
                        .clone()
                        .cpu()
                        .numpy()
                    )
                    # goal_eef_rot = torch.FloatTensor(goal_cfg.robot_ee_rot)
                    wp_eef_rot = (
                        results["robot_ee_rot_gripper_site_world"][-1]
                        .clone()
                        .cpu()
                        .numpy()
                    )  # goal_cfg.robot_ee_rot is not in terms of gripper site frame
                    wp_eef_pos_eef_site = (
                        results["robot_ee_pos_eef_site_world"][-1].clone().cpu().numpy()
                    )
                    wp_eef_rot_eef_site = (
                        results["robot_ee_rot_eef_site_world"][-1].clone().cpu().numpy()
                    )

                positions = np.array([wp_eef_pos, goal_eef_pos])
                rotations = st.Rotation.from_matrix(
                    np.array([wp_eef_rot, goal_eef_rot])
                )
                positions_eef_site = np.array([wp_eef_pos_eef_site, goal_eef_pos])
                rotations_eef_site = st.Rotation.from_matrix(
                    np.array([wp_eef_rot_eef_site, goal_eef_rot_eef_site])
                )

                interp_start_end_times = np.array([0, 1])
                rot_interp = st.Slerp(interp_start_end_times, rotations)
                pos_interp = si.interp1d(
                    interp_start_end_times, positions, axis=0, assume_sorted=True
                )

                rot_interp_eef_site = st.Slerp(
                    interp_start_end_times, rotations_eef_site
                )
                pos_interp_eef_site = si.interp1d(
                    interp_start_end_times,
                    positions_eef_site,
                    axis=0,
                    assume_sorted=True,
                )

                # compute robot_joint_qpos for each interp time using IK
                for i, interp_time in enumerate(interp_times):
                    # get robot joint qpos at this time
                    if i == 0:
                        prev_robot_joint_qpos = robot_qpos_waypoint[:-2]
                        prev_robot_gripper_qpos = robot_qpos_waypoint[-2:]
                    else:
                        prev_robot_joint_qpos = (
                            results["robot_joint_qpos"][-1]
                            .clone()
                            .detach()
                            .cpu()
                            .numpy()
                        )
                        prev_robot_gripper_qpos = (
                            results["robot_gripper_qpos"][-1]
                            .clone()
                            .detach()
                            .cpu()
                            .numpy()
                        )

                    target_eef_rot = rot_interp(interp_time).as_matrix()
                    target_eef_pos = pos_interp(interp_time)

                    robot_qpos_ik, is_success = DrakeMotionPlanner.get_IK(
                        plant,
                        context,
                        RigidTransform(RotationMatrix(target_eef_rot), target_eef_pos),
                        ee_frame_name=ROBOMIMIC_GRIPPER_SITE_FRAME_NAME,
                        initial_robot_joint_qpos=prev_robot_joint_qpos,
                        initial_robot_gripper_qpos=prev_robot_gripper_qpos,
                        goal_robot_gripper_qpos=start_cfg.robot_gripper_qpos,
                        check_collisions=False,
                    )
                    if not is_success:
                        raise RuntimeError(
                            "IK failed for waypoint to goal interpolation EEF pose"
                        )

                    # TODO(klin) here check
                    if interp_time == interp_times[-1]:
                        print(
                            f"IK rotation xyzw: {R.from_matrix(target_eef_rot).as_quat()}"
                        )
                        print(f"IK position: {target_eef_pos}")
                        # IK rotation: [0.80190345 0.59271192 0.03566211 0.06611838]
                        # IK position: [0.03099233 0.01929313 0.82581438]
                        # force visaulizer to publish at the end
                        if self.view_meshcat:
                            plant.SetPositions(plant_context, robot_qpos_ik)
                            visualizer.ForcedPublish(
                                visualizer.GetMyContextFromRoot(context)
                            )
                            collision_visualizer.ForcedPublish(
                                collision_visualizer.GetMyContextFromRoot(context)
                            )

                    X_W_GripperSite_robomimic = RigidTransform(
                        RotationMatrix(rot_interp(interp_time).as_matrix()),
                        pos_interp(interp_time),
                    )
                    X_W_EEFSite_robomimic = RigidTransform(
                        RotationMatrix(rot_interp_eef_site(interp_time).as_matrix()),
                        pos_interp_eef_site(interp_time),
                    )

                    X_RobotBase_GripperSite_robomimic = (
                        hand_robomimic_gripper_site_frame.CalcPose(
                            plant_context, base_frame
                        )
                    )
                    X_RobotBase_EEFSite_robomimic = (
                        hand_robomimic_eef_site_frame.CalcPose(
                            plant_context, base_frame
                        )
                    )

                    task_relev_obj_pos = goal_cfg.task_relev_obj_pos
                    task_relev_obj_rot = goal_cfg.task_relev_obj_rot

                    gripper_qpos = (
                        [0.04, -0.04]
                        if interp_time != 1
                        else results["robot_gripper_qpos"][-1]
                        .clone()
                        .detach()
                        .cpu()
                        .numpy()
                    )

                    DrakeMotionPlanner.update_results(
                        results,
                        robot_qpos_ik[:7],  # joint_qpos
                        gripper_qpos,  # gripper_qpos
                        X_W_GripperSite_robomimic,
                        X_W_EEFSite_robomimic,
                        X_RobotBase_GripperSite_robomimic,
                        X_RobotBase_EEFSite_robomimic,
                        task_relev_obj_pos,
                        task_relev_obj_rot,
                    )

            # don't interpolate: motion plan to goal; not sure what the point of doing so is ... I guess it's a
            # hardcoded waypoint will we ever need this though? # maybe if doing PRM w/ extra safety?
            else:
                result, trajopt = DrakeMotionPlanner.setup_and_solve_trajopt_prog(
                    plant,
                    context,
                    robot_qpos_waypoint[:-2],
                    robot_qpos_waypoint[-2:],
                    goal_cfg.robot_joint_qpos
                    if goal_cfg.robot_joint_qpos is not None
                    else robot_qpos_goal[:-2],
                    robot_qpos_goal[-2:],
                    meshcat=self.meshcat,
                    visualizer=visualizer,
                    collision_visualizer=collision_visualizer,
                    min_duration=0.01,  # hardcoded for now
                    use_collision_constraint=False,
                    viz_trajopt_process=True,
                )
                waypoint_to_goal_traj = trajopt.ReconstructTrajectory(result)
                if self.view_meshcat:
                    PublishPositionTrajectory(
                        waypoint_to_goal_traj, context, plant, visualizer
                    )
                    collision_visualizer.ForcedPublish(
                        collision_visualizer.GetMyContextFromRoot(context)
                    )

                timesteps = get_timesteps(
                    waypoint_to_goal_traj.end_time() + np.finfo(float).eps,
                    1 / self.action_freq,
                )
                timesteps = timesteps[
                    1:
                ]  # skip the first timestep since it's the same as the last timestep of start_to_waypoint_traj

                if not result.is_success():
                    logging.info(
                        "waypoint_to_goal_traj Trajopt failed ... view meshcat?"
                    )
                    # Perhaps have option to allow saving of the data or updating of parameters?
                    import ipdb

                    ipdb.set_trace()
                    return None

                for idx, t in enumerate(timesteps):
                    plant.SetPositions(
                        plant_context, waypoint_to_goal_traj.vector_values([t])
                    )
                    joint_qpos = waypoint_to_goal_traj.vector_values([t])[:7, 0]
                    gripper_qpos = waypoint_to_goal_traj.vector_values([t])[7:, 0]
                    if convert_last_gripper_qpos_entry_negative:
                        gripper_qpos[-1] *= -1  # correction for mujoco
                    if enforce_reach_traj_gripper_non_close:
                        # check that curren gripper qpos is not closing w.r.t previous gripper qpos
                        if idx > 0:
                            gripper_qpos[-1] = min(
                                gripper_qpos[-1],
                                waypoint_to_goal_traj.vector_values(
                                    [timesteps[idx - 1]]
                                )[7:, 0][-1]
                                * -1,
                            )
                            gripper_qpos[-2] = max(
                                gripper_qpos[-2],
                                waypoint_to_goal_traj.vector_values(
                                    [timesteps[idx - 1]]
                                )[7:, 0][-2],
                            )
                    hand_robomimic_gripper_site_frame = plant.GetFrameByName(
                        ROBOMIMIC_GRIPPER_SITE_FRAME_NAME
                    )
                    X_W_GripperSite_robomimic = (
                        hand_robomimic_gripper_site_frame.CalcPoseInWorld(plant_context)
                    )
                    base_frame = plant.GetFrameByName(ROBOT_BASE_FRAME_NAME)
                    X_RobotBase_GripperSite_robomimic = (
                        hand_robomimic_gripper_site_frame.CalcPose(
                            plant_context, base_frame
                        )
                    )

                    hand_robomimic_eef_site_frame = plant.GetFrameByName(
                        ROBOMIMIC_EE_SITE_FRAME_NAME
                    )
                    X_W_EEFSite_robomimic = (
                        hand_robomimic_eef_site_frame.CalcPoseInWorld(plant_context)
                    )
                    X_RobotBase_EEFSite_robomimic = (
                        hand_robomimic_eef_site_frame.CalcPose(
                            plant_context, base_frame
                        )
                    )

                    DrakeMotionPlanner.update_results(
                        results,
                        joint_qpos,
                        gripper_qpos,
                        X_W_GripperSite_robomimic,
                        X_W_EEFSite_robomimic,
                        X_RobotBase_GripperSite_robomimic,
                        X_RobotBase_EEFSite_robomimic,
                        goal_cfg.task_relev_obj_pos,
                        goal_cfg.task_relev_obj_rot,
                    )
        else:
            result, trajopt = DrakeMotionPlanner.setup_and_solve_trajopt_prog(
                plant,
                context,
                start_cfg_robot_joint_qpos,
                start_cfg.robot_gripper_qpos,
                robot_qpos_goal[:-2],
                robot_qpos_goal[-2:],
                meshcat=self.meshcat,
                visualizer=visualizer,
                collision_visualizer=collision_visualizer,
                viz_trajopt_process=True,
            )
            traj = trajopt.ReconstructTrajectory(result)
            if self.view_meshcat:
                PublishPositionTrajectory(traj, context, plant, visualizer)
                collision_visualizer.ForcedPublish(
                    collision_visualizer.GetMyContextFromRoot(context)
                )

            if not result.is_success():
                logging.info("Trajopt failed ... view meshcat?")
                # Perhaps have option to allow saving of the data or updating of parameters?
                return None

            plant_context = plant.GetMyContextFromRoot(context)

            timesteps = get_timesteps(traj.end_time(), 1 / self.action_freq)
            results: Dict[str, List] = defaultdict(list)

            for idx, t in enumerate(timesteps):
                plant.SetPositions(plant_context, traj.vector_values([t]))
                joint_qpos = traj.vector_values([t])[:7, 0]
                gripper_qpos = traj.vector_values([t])[7:, 0]
                if convert_last_gripper_qpos_entry_negative:
                    gripper_qpos[-1] *= -1  # correction for mujoco
                if enforce_reach_traj_gripper_non_close:
                    # check that curren gripper qpos is not closing w.r.t previous gripper qpos
                    if idx > 0:
                        gripper_qpos[-1] = min(
                            gripper_qpos[-1],
                            traj.vector_values([timesteps[idx - 1]])[7:, 0][-1] * -1,
                        )
                        gripper_qpos[-2] = max(
                            gripper_qpos[-2],
                            traj.vector_values([timesteps[idx - 1]])[7:, 0][-2],
                        )

                hand_robomimic_gripper_site_frame = plant.GetFrameByName(
                    ROBOMIMIC_GRIPPER_SITE_FRAME_NAME
                )
                X_W_GripperSite_robomimic = (
                    hand_robomimic_gripper_site_frame.CalcPoseInWorld(plant_context)
                )
                base_frame = plant.GetFrameByName(ROBOT_BASE_FRAME_NAME)
                X_RobotBase_GripperSite_robomimic = (
                    hand_robomimic_gripper_site_frame.CalcPose(
                        plant_context, base_frame
                    )
                )

                hand_robomimic_eef_site_frame = plant.GetFrameByName(
                    ROBOMIMIC_EE_SITE_FRAME_NAME
                )
                X_W_EEFSite_robomimic = hand_robomimic_eef_site_frame.CalcPoseInWorld(
                    plant_context
                )
                X_RobotBase_EEFSite_robomimic = hand_robomimic_eef_site_frame.CalcPose(
                    plant_context, base_frame
                )

                DrakeMotionPlanner.update_results(
                    results,
                    joint_qpos,
                    gripper_qpos,
                    X_W_GripperSite_robomimic,
                    X_W_EEFSite_robomimic,
                    X_RobotBase_GripperSite_robomimic,
                    X_RobotBase_EEFSite_robomimic,
                    goal_cfg.task_relev_obj_pos,
                    goal_cfg.task_relev_obj_rot,
                )

        results["overall_cost"] = overall_cost
        assert (
            start_at_collision_free_waypoint or overall_cost > 0
        ), "overall_cost should be positive"
        return results

    def get_robot_and_obj_trajectory(
        self,
        default_robot_joint_qpos: np.ndarray,
        start_cfg: RobotEnvConfig,
        future_cfg_list: List[RobotEnvConfig],
        check_collisions: bool = True,
        add_noise_to_task_relev_obj_pose: bool = True,
        task_relev_obj_pos_noise_bound: float = 0.001,
        task_relev_obj_rot_angle_noise_bound: float = 0.2,  # TODO(klin): rotations w.r.t world frame is hacky for now
        convert_last_gripper_qpos_entry_negative: bool = True,
        obj_to_init_info: Dict[str, ObjectInitializationInfo] = None,
    ) -> Optional[Dict[str, Union[np.ndarray, torch.Tensor]]]:
        """
        Args:
            add_noise_to_task_relev_obj_pose: whether to add noise to task relev obj pose
                extra data augmentation because control mightn't always be perfect.
            convert_last_gripper_qpos_entry_negative: whether to convert last gripper qpos entry to negative
                (for mujoco gripper convention)

        Converts robot EEF poses to robot joint qposes and returns task relev object poses in a results dict.
        This function doesn't really deal w/ object poses; it's more just IK for robot EEF poses.
        However, it does return the task relev object poses in the results dict for convenience.

        Assumes that gripper qpos are input with drake compatibility i.e. all positive.
        """
        results: Dict[str, List] = defaultdict(list)
        assert (
            start_cfg.robot_gripper_qpos is not None
        ), "start_cfg.robot_gripper_qpos is None"
        assert np.all(
            start_cfg.robot_gripper_qpos >= 0
        ), "start_cfg.robot_gripper_qpos has negative values"

        robot_gripper_qpos_lst = map(lambda x: x.robot_gripper_qpos, future_cfg_list)
        robot_gripper_qpos_lst = filter(lambda x: x is not None, robot_gripper_qpos_lst)
        assert all(
            map(lambda x: np.all(x >= 0), robot_gripper_qpos_lst)
        ), "future_cfg_list.robot_gripper_qpos has negative values"

        # Q: is it the best idea to do traj tracking w/ OC?
        # hmm maybe just do IK control? not sure

        # TODO(klin): let setup_env task a *dict* of poses, paths, etc.
        try:
            plant, builder, scene_graph = self.setup_env(
                add_robot=True,
                add_robot_hand=False,
                robot_base_pos=start_cfg.robot_base_pos,
                robot_base_quat_wxyz=start_cfg.robot_base_quat_wxyz,
                task_relev_obj_pos_nerf=[start_cfg.task_relev_obj_pos_nerf],
                task_relev_obj_rot_nerf=[start_cfg.task_relev_obj_rot_nerf],
                obj_to_init_info=obj_to_init_info,
            )

            if plant is None:
                logging.info("Failed to setup environment")
                return None

        except Exception as e:
            print(f"Exception type {type(e)}")
            print(f"setup_env failed with exception {e} ... trying again")
            import ipdb

            ipdb.set_trace()
            plant, builder, scene_graph = self.setup_env(
                add_robot=True,
                add_robot_hand=False,
                robot_base_pos=start_cfg.robot_base_pos,
                robot_base_quat_wxyz=start_cfg.robot_base_quat_wxyz,
                task_relev_obj_pos_nerf=[start_cfg.task_relev_obj_pos_nerf],
                task_relev_obj_rot_nerf=[start_cfg.task_relev_obj_rot_nerf],
                obj_to_init_info=obj_to_init_info,
            )

        if self.meshcat is not None:
            visualizer = MeshcatVisualizer.AddToBuilder(
                builder,
                scene_graph,
                self.meshcat,
                MeshcatVisualizerParams(role=Role.kIllustration),
            )
            collision_visualizer = MeshcatVisualizer.AddToBuilder(
                builder,
                scene_graph,
                self.meshcat,
                MeshcatVisualizerParams(prefix="collision", role=Role.kProximity),
            )
            MeshcatVisualizer.AddToBuilder(
                builder,
                scene_graph,
                self.meshcat,
                MeshcatVisualizerParams(prefix="collision", role=Role.kProximity),
            )
            self.meshcat.SetProperty("collision", "visible", False)
        else:
            visualizer = None
            collision_visualizer = None

        diagram = builder.Build()
        context = diagram.CreateDefaultContext()

        # force update the visualizer
        if self.view_meshcat:
            visualizer_context = visualizer.GetMyContextFromRoot(context)
            visualizer.ForcedPublish(visualizer_context)
            collision_context = collision_visualizer.GetMyContextFromRoot(context)
            collision_visualizer.ForcedPublish(collision_context)

        initial_robot_joint_qpos = default_robot_joint_qpos
        initial_robot_gripper_qpos = (
            start_cfg.robot_gripper_qpos.cpu().numpy()
            if isinstance(start_cfg.robot_gripper_qpos, torch.Tensor)
            else start_cfg.robot_gripper_qpos
        )
        robot_env_cfgs = [start_cfg] + future_cfg_list
        for i, robot_env_cfg in enumerate(robot_env_cfgs):
            robot_qpos_ik, is_success = DrakeMotionPlanner.get_IK(
                plant,
                context,
                RigidTransform(
                    Quaternion(
                        robot_env_cfg.robot_ee_rot
                        if robot_env_cfg.robot_ee_rot is not None
                        else robot_env_cfg.robot_ee_quat_wxyz
                    ),
                    robot_env_cfg.robot_ee_pos,
                ),
                ee_frame_name=ROBOMIMIC_EE_SITE_FRAME_NAME,
                initial_robot_joint_qpos=initial_robot_joint_qpos,
                initial_robot_gripper_qpos=initial_robot_gripper_qpos,
                goal_robot_gripper_qpos=robot_env_cfg.robot_gripper_qpos,
                meshcat=self.meshcat,
                collision_visualizer=collision_visualizer,
                visualizer=visualizer,
                check_collisions=check_collisions,
            )

            if not is_success:
                return None

            initial_robot_joint_qpos = robot_qpos_ik[:7]
            initial_robot_gripper_qpos = robot_qpos_ik[7:]

            plant_context = plant.GetMyContextFromRoot(context)
            plant.SetPositions(plant_context, robot_qpos_ik)

            hand_robomimic_gripper_site_frame = plant.GetFrameByName(
                ROBOMIMIC_GRIPPER_SITE_FRAME_NAME
            )
            X_W_GripperSite_robomimic = (
                hand_robomimic_gripper_site_frame.CalcPoseInWorld(plant_context)
            )
            base_frame = plant.GetFrameByName(ROBOT_BASE_FRAME_NAME)
            X_RobotBase_GripperSite_robomimic = (
                hand_robomimic_gripper_site_frame.CalcPose(plant_context, base_frame)
            )

            save_gripper_qpos = robot_env_cfg.robot_gripper_qpos.copy()
            if convert_last_gripper_qpos_entry_negative:
                save_gripper_qpos[-1] *= -1  # correction for mujoco

            hand_robomimic_eef_site_frame = plant.GetFrameByName(
                ROBOMIMIC_EE_SITE_FRAME_NAME
            )
            X_W_EEFSite_robomimic = hand_robomimic_eef_site_frame.CalcPoseInWorld(
                plant_context
            )
            X_RobotBase_EEFSite_robomimic = hand_robomimic_eef_site_frame.CalcPose(
                plant_context, base_frame
            )

            DrakeMotionPlanner.update_results(
                results,
                robot_qpos_ik[:7],
                save_gripper_qpos,
                X_W_GripperSite_robomimic,
                X_W_EEFSite_robomimic,
                X_RobotBase_GripperSite_robomimic,
                X_RobotBase_EEFSite_robomimic,
                robot_env_cfg.task_relev_obj_pos,
                robot_env_cfg.task_relev_obj_rot,
            )

            if add_noise_to_task_relev_obj_pose:
                pos_noise = np.random.uniform(
                    -task_relev_obj_pos_noise_bound,
                    task_relev_obj_pos_noise_bound,
                    size=3,
                )
                task_relev_obj_pose = results["task_relev_obj_pose"][-1].numpy().copy()
                task_relev_obj_pose[:3, 3] += pos_noise
                results["task_relev_obj_pos"][-1] += torch.FloatTensor(pos_noise)
                rot_noise = random_rotation_matrix(task_relev_obj_rot_angle_noise_bound)
                task_relev_obj_pose[:3, :3] = np.matmul(
                    rot_noise[:3, :3], task_relev_obj_pose[:3, :3]
                )
                results["task_relev_obj_rot"][-1] = torch.FloatTensor(
                    task_relev_obj_pose[:3, :3].copy()
                )

        return results

    def get_eef_ik(
        self,
        X_W_ee_init: np.ndarray,
        q_gripper_init: np.ndarray,
        name_to_frame_info: Dict[str, Dict[str, Any]],  # for env setup
        kp_to_P_goal: Dict[str, np.ndarray],
        check_collisions: bool = False,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        q_gripper_init = np.abs(q_gripper_init)  # convert to drake convention
        plant, builder, scene_graph = self.setup_env(
            add_robot=False, add_robot_hand=True, name_to_frame_info=name_to_frame_info
        )
        if plant is None:
            logging.info(
                "drake_motion_planner.py: get_eef_ik() Failed to setup environment"
            )
            return None, None

        if self.meshcat is not None:
            visualizer = MeshcatVisualizer.AddToBuilder(
                builder,
                scene_graph,
                self.meshcat,
                MeshcatVisualizerParams(role=Role.kIllustration),
            )
            MeshcatVisualizer.AddToBuilder(
                builder,
                scene_graph,
                self.meshcat,
                MeshcatVisualizerParams(prefix="collision", role=Role.kProximity),
            )
            self.meshcat.SetProperty("collision", "visible", False)
        else:
            visualizer = None

        diagram = builder.Build()
        context = diagram.CreateDefaultContext()

        robot_qpos, is_success = DrakeMotionPlanner.get_IK(
            plant,
            context,
            visualizer=visualizer,
            meshcat=self.meshcat,
            ik_type=IKType.P_KPTS_TO_X_EE,
            X_W_ee_init=X_W_ee_init,
            q_gripper_init=q_gripper_init,
            kp_to_P_goal=kp_to_P_goal,
            check_collisions=check_collisions,
        )

        if not is_success:
            logging.info("IK failure for keypoints to EE pose (+ gripper qpos)")
            # import ipdb

            # ipdb.set_trace()

        # set robot qpos
        plant_context = plant.GetMyContextFromRoot(context)
        plant.SetPositions(plant_context, robot_qpos)

        # calculate gripper site pose
        ee_site_frame = plant.GetFrameByName(ROBOMIMIC_EE_SITE_FRAME_NAME)
        X_ee = ee_site_frame.CalcPoseInWorld(plant_context).GetAsMatrix4()
        q_gripper = robot_qpos[7:]

        q_gripper[-1] *= -1
        # debugging code for checking if scaling worked properly; seems like much ado about nothing
        # # compute positions of keypoints
        kpts = {}
        for kpt_name, kpt_info in name_to_frame_info.items():
            kpt_frame = plant.GetFrameByName(kpt_info["src_frame_of_offset_frame"])
            X_kpt = kpt_frame.CalcPoseInWorld(plant_context).GetAsMatrix4()
            kpts[kpt_name] = X_kpt[:3, 3]

        # print kpts and distance between them
        np.linalg.norm(kpts["panda_left_tip"] - kpts["panda_right_tip"])

        return X_ee, q_gripper

    def get_kpts_from_ee_gripper_objs_and_params(
        self,
        X_ee: np.ndarray,
        q_gripper: np.ndarray,
        name_to_kpt_info: Dict[str, Any],
        input_ee_pose_frame: str = ROBOMIMIC_EE_SITE_FRAME_NAME,
    ) -> Dict[str, np.ndarray]:
        """
        Get positions of keypoints with keypoint name from the *top level keys* in name_to_kpt_info
        and using ee, gripper, and name_to_kpt_info.
        """
        # convert to drake convention
        q_gripper = np.abs(q_gripper)

        kpts: Dict[str, np.ndarray] = {}
        plant, builder, scene_graph = self.setup_env(
            add_robot=False,
            add_robot_hand=True,
            name_to_frame_info=name_to_kpt_info,
        )
        if plant is None:
            logging.info(
                "drake_motion_planner.py: get_eef_ik() Failed to setup environment"
            )
            return None, None

        if self.meshcat is not None:
            visualizer = MeshcatVisualizer.AddToBuilder(
                builder,
                scene_graph,
                self.meshcat,
                MeshcatVisualizerParams(role=Role.kIllustration),
            )
            MeshcatVisualizer.AddToBuilder(
                builder,
                scene_graph,
                self.meshcat,
                MeshcatVisualizerParams(prefix="collision", role=Role.kProximity),
            )
            self.meshcat.SetProperty("collision", "visible", False)
        else:
            visualizer = None

        diagram = builder.Build()
        context = diagram.CreateDefaultContext()
        plant_context = plant.GetMyContextFromRoot(context)

        if input_ee_pose_frame == ROBOMIMIC_EE_SITE_FRAME_NAME:
            R = np.eye(3)
            X_gripsite_panda = np.eye(4)
            X_gripsite_panda[:3, 3] = -R.T @ DRAKE_TO_ROBOMIMIC_GRIPPER_OFFSET_POS
            X_gripsite_panda[:3, :3] = R.T
            X_ee = X_ee @ X_gripsite_panda

        robot_qpos = np.zeros(plant.num_positions())
        quat_xyzw_ee = st.Rotation.from_matrix(X_ee[:3, :3]).as_quat()
        quat_wxyz_ee = np.concatenate([[quat_xyzw_ee[3]], quat_xyzw_ee[:3]])
        robot_qpos[:7] = np.concatenate([quat_wxyz_ee, X_ee[:3, 3]])
        robot_qpos[7:] = q_gripper

        plant.SetPositions(plant_context, robot_qpos)

        for kpt_name in name_to_kpt_info.keys():
            frame = plant.GetFrameByName(kpt_name)
            X_W_frame = frame.CalcPoseInWorld(plant_context)
            kpts[kpt_name] = X_W_frame.translation()

        if self.meshcat is not None:
            for i, (frame_name, _) in enumerate(name_to_kpt_info.items()):
                X_W_tip = plant.GetFrameByName(frame_name).CalcPoseInWorld(
                    plant_context
                )
                self.meshcat.SetObject(
                    f"{frame_name}_sphere",
                    Sphere(0.0025),
                    rgba=Rgba(0.1 * i, 0.9 * i, 0.1 * i, 1),
                )
                self.meshcat.SetTransform(
                    f"{frame_name}_sphere", RigidTransform(X_W_tip.translation())
                )

            visualizer_context = visualizer.GetMyContextFromRoot(context)
            visualizer.ForcedPublish(visualizer_context)
        return kpts


def calculate_pose_deltas(
    pos: List[torch.Tensor], rot: List[torch.Tensor], output_rot_type: str = "matrix"
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Calculates the deltas between the reference pose (first element in the lists) and the
    current pose (second element onwards).

    Currently unused?
    """
    # Initialize lists to store deltas
    pos_deltas = []
    rot_deltas = []

    homog_deltas = []
    # Reference pose
    ref_pos = pos[0]
    ref_rot = rot[0]

    T1 = torch.eye(4)
    T1[:3, :3] = ref_rot
    T1[:3, 3] = ref_pos

    # Loop over all poses starting from the second pose
    for cur_pos, cur_rot in zip(pos[1:], rot[1:]):
        T2 = torch.eye(4)
        T2[:3, :3] = cur_rot
        T2[:3, 3] = cur_pos

        homog_deltas.append(torch.inverse(T1) @ T2)

        # Compute position delta
        pos_delta = cur_pos - ref_pos
        pos_deltas.append(pos_delta)

        # Compute orientation delta as relative rotation
        rot_delta = cur_rot * ref_rot.T
        if output_rot_type == "matrix":
            rot_deltas.append(rot_delta)
        else:
            rot_delta_xyzw = rot_delta.as_quat()
            rot_delta_wxyz = [
                rot_delta_xyzw[3],
                rot_delta_xyzw[0],
                rot_delta_xyzw[1],
                rot_delta_xyzw[2],
            ]
            rot_deltas.append(rot_delta_wxyz)

    return pos_deltas, rot_deltas, homog_deltas
