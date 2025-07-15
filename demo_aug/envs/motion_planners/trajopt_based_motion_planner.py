import datetime
import logging
import pathlib
import time
from typing import Dict, Literal, Optional, Tuple

import numpy as np
from pydrake.geometry import (
    Meshcat,
    MeshcatVisualizer,
)
from pydrake.math import BsplineBasis
from pydrake.multibody.inverse_kinematics import (
    MinimumDistanceConstraint,
)
from pydrake.multibody.plant import MultibodyPlant
from pydrake.planning import KinematicTrajectoryOptimization
from pydrake.solvers import MathematicalProgramResult, SnoptSolver, SolutionResult
from pydrake.systems.framework import Context
from pydrake.trajectories import BsplineTrajectory
from scipy.spatial.transform import Rotation as R

from demo_aug.envs.motion_planners.base_motion_planner import (
    BaseMotionPlanner,
)
from demo_aug.envs.motion_planners.motion_planning_space import MotionPlanningSpace
from demo_aug.utils.drake_utils import PublishPositionTrajectory
from demo_aug.utils.mathutils import (
    get_timesteps,
    interpolate_poses,
    random_rotation_matrix,
)
from demo_aug.utils.snopt_utils import (
    SNOPT_SOLVER_MAX_OPTIMALITY_VALUE,
    extract_feasiblity_optimality_values_from_snopt_log,
)


def extract_quat_pos_from_matrix(matrices):
    """
    Extracts the quaternion (WXYZ) and the position from a list of transformation matrices.
    """
    quats = []
    positions = []

    for mat in matrices:
        rotation_matrix = mat[:3, :3]
        position = mat[:3, 3]

        quat_wxyz = R.from_matrix(rotation_matrix).as_quat()
        quats.append(quat_wxyz)
        positions.append(position)

    return np.array(quats), np.array(positions)


# deprecated
class TrajoptBasedMotionPlanner(BaseMotionPlanner):
    def __init__(self):
        # super().__init__(robot, obstacles, start_conf, goal_conf, **kwargs)
        # self.planner = trajopt_planner
        return

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
        lb, ub = robot_kinematic_ranges[0], robot_kinematic_ranges[1]

        # Generate random waypoint configuration
        random_rot_mat = random_rotation_matrix(max_angle_deg=45)
        robot_pos_waypoint = np.random.uniform(lb[4:7], ub[4:7])
        robot_quat_wxyz_waypoint = (
            R.from_matrix(random_rot_mat) * R.from_quat(robot_joint_qpos_goal[0:4])
        ).as_quat()

        robot_joint_waypoint = np.concatenate(
            [robot_quat_wxyz_waypoint, robot_pos_waypoint]
        )
        robot_gripper_waypoint = robot_gripper_qpos_goal

        # Initialize trajectory
        q_guess = np.zeros((9, trajopt_num_control_points))

        # Calculate the number of control points for each segment
        num_control_points_per_segment = trajopt_num_control_points // 2

        # Interpolate between start, waypoint, and goal for each joint and gripper parameter
        # quat_wxyz slerp
        X_robot_start = np.eye(4)
        X_robot_start[:3, :3] = R.from_quat(robot_joint_qpos_start[0:4]).as_matrix()
        X_robot_start[:3, 3] = robot_joint_qpos_start[4:7]
        X_robot_waypoint = np.eye(4)
        X_robot_waypoint[:3, :3] = R.from_quat(robot_joint_waypoint[0:4]).as_matrix()
        X_robot_waypoint[:3, 3] = robot_joint_waypoint[4:7]
        X_robot_goal = np.eye(4)
        X_robot_goal[:3, :3] = R.from_quat(robot_joint_qpos_goal[0:4]).as_matrix()
        X_robot_goal[:3, 3] = robot_joint_qpos_goal[4:7]

        # interpolate there
        X_robot_start_to_wp = interpolate_poses(
            X_robot_start,
            X_robot_waypoint,
            trajopt_num_control_points - num_control_points_per_segment,
        )
        X_robot_wp_to_goal = interpolate_poses(
            X_robot_waypoint, X_robot_goal, num_control_points_per_segment + 1
        )

        # Extract quaternion (WXYZ) and positions from interpolated matrices
        quats_start_to_wp, positions_start_to_wp = extract_quat_pos_from_matrix(
            X_robot_start_to_wp
        )
        quats_wp_to_goal, positions_wp_to_goal = extract_quat_pos_from_matrix(
            X_robot_wp_to_goal[1:]
        )

        # Concatenate the two sequences
        all_quats = np.vstack([quats_start_to_wp, quats_wp_to_goal])
        all_positions = np.vstack([positions_start_to_wp, positions_wp_to_goal])

        # Populate q_guess with the extracted values
        q_guess[0:4, :] = all_quats.T
        q_guess[4:7, :] = all_positions.T

        # For the gripper, assume direct linear interpolation between start and goal
        q_guess[7:, :] = np.linspace(
            robot_gripper_qpos_start, robot_gripper_waypoint, trajopt_num_control_points
        ).T

        return BsplineTrajectory(trajopt_basis, q_guess)

    @staticmethod
    def setup_and_solve_trajopt_prog(
        plant: MultibodyPlant,
        context: Context,
        robot_joint_qpos_start: np.ndarray,
        robot_gripper_qpos_start: np.ndarray,
        robot_joint_qpos_goal: np.ndarray,
        robot_gripper_qpos_goal: np.ndarray,
        motion_planning_space: MotionPlanningSpace,
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
        q_start = np.concatenate([robot_joint_qpos_start, robot_gripper_qpos_start])
        q_goal = np.concatenate([robot_joint_qpos_goal, robot_gripper_qpos_goal])

        trajopt = KinematicTrajectoryOptimization(plant.num_positions(), 15)
        prog = trajopt.get_mutable_prog()
        # trajopt.AddDurationCost(1.0)
        trajopt.AddPathLengthCost(1.0)
        # trajopt.AddPositionBounds(plant.GetPositionLowerLimits(), plant.GetPositionUpperLimits())
        # update position bounds so that gripper positions are fixed
        trajopt.AddPositionBounds(
            np.concatenate(
                [plant.GetPositionLowerLimits()[:7], plant.GetPositionUpperLimits()[7:]]
            ),
            np.concatenate(
                [plant.GetPositionUpperLimits()[:7], plant.GetPositionUpperLimits()[7:]]
            ),
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

        # prog.AddQuadraticErrorCost(np.eye(num_q), q_start, trajopt.control_points()[:, 0])
        # prog.AddQuadraticErrorCost(np.eye(num_q), q_goal, trajopt.control_points()[:, -1])

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
        prog.SetSolverOption(snopt_solver.solver_id(), "Time limit", 2)
        prog.SetSolverOption(snopt_solver.solver_id(), "Timing level", 2)
        # try the below
        # prog.SetSolverOption(snopt_solver.solver_id(), "verbose", (0, 1))

        def PlotPath(control_points):
            traj = BsplineTrajectory(
                trajopt.basis(), control_points.reshape((num_q, -1))
            )

            hand_positions = []
            times = np.linspace(0, 1, 50)
            traj_vals = traj.vector_values(times.tolist())
            hand_positions = np.array(traj_vals).T[..., 4:7]
            hand_positions = hand_positions.T
            # convert to contiguous array
            hand_positions = np.ascontiguousarray(hand_positions)
            meshcat.SetLine("hand_positions", hand_positions)

        if meshcat is not None and viz_trajopt_process:
            # import ipdb; ipdb.set_trace()
            prog.AddVisualizationCallback(
                PlotPath, trajopt.control_points().reshape((-1,))
            )

        tic = time.time()
        result = snopt_solver.Solve(prog)

        if visualizer is not None:
            PublishPositionTrajectory(
                trajopt.ReconstructTrajectory(result), context, plant, visualizer
            )
        if collision_visualizer is not None:
            collision_visualizer.ForcedPublish(
                collision_visualizer.GetMyContextFromRoot(context)
            )

        return result, trajopt
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
        ccc = motion_planning_space.collision_checker.MakeStandaloneModelContext()

        collision_constraint = MinimumDistanceConstraint(
            motion_planning_space.collision_checker,
            minimum_distance=min_dist,
            collision_checker_context=ccc,
            penalty_function=None,
            influence_distance_offset=min_dist_thresh,
        )

        # only doing this because I know when things get close to contact ...
        evaluate_at_s = np.linspace(0, 0.5, 25)
        for s in evaluate_at_s:
            trajopt.AddPathPositionConstraint(collision_constraint, s)

        # add more constraints near the end and append to evaluate_at_s
        evaluate_at_s = np.append(evaluate_at_s, np.linspace(0.5, 0.99, 25))
        for s in evaluate_at_s:
            trajopt.AddPathPositionConstraint(collision_constraint, s)

        # create folder for trajopt results using pathlib
        solver_results_dir = pathlib.Path("trajopt_results")
        solver_results_dir.mkdir(parents=True, exist_ok=True)

        current_time = datetime.datetime.now().strftime("%b_%d_%H_%M_%S")
        file_path: pathlib.Path = (
            solver_results_dir / f"trajopt_shelves_demo_collision_{current_time}.snopt"
        )
        prog.SetSolverOption(snopt_solver.solver_id(), "Major iterations limit", 2000)
        prog.SetSolverOption(snopt_solver.solver_id(), "Iterations limit", 2000)
        prog.SetSolverOption(snopt_solver.solver_id(), "Print file", str(file_path))
        prog.SetSolverOption(snopt_solver.solver_id(), "Time limit", 15)
        prog.SetSolverOption(snopt_solver.solver_id(), "Timing level", 2)

        for i in range(num_init_guesses):
            if i > 0:
                init_guess_traj = TrajoptBasedMotionPlanner.generate_trajopt_init_guess(
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
                print("Time to extract feasibility and optimality values: ", toc - tic)
                logging.info(f"Optimal cost: {result.get_optimal_cost()}")
                logging.info(f"feasibility_value: {feasibility_value}")
                logging.info(f"optimality_value: {optimality_value}")
                logging.info(f"solver return value: {result.get_solver_details().info}")
                print(f"feasibility_value: {feasibility_value}")
                print(f"optimality_value: {optimality_value}")
                logging.info(f"Optimal cost: {result.get_optimal_cost()}")

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
        import ipdb

        ipdb.set_trace()
        return result, trajopt

    def get_optimal_trajectory(
        self,
        start_cfg: np.ndarray,
        goal_cfg: np.ndarray,
        motion_planning_space: MotionPlanningSpace,
        debug: bool = False,
        num_trajopt_guesses: int = 1,
        input_config_frame: Optional[
            Literal[
                "panda_hand_eef_site_robomimic", "panda_hand_gripper_site_robomimic"
            ]
        ] = None,
    ) -> np.ndarray:
        traj: np.ndarray = None
        self.action_freq = 10

        plant = motion_planning_space.plant
        context = motion_planning_space.context
        meshcat, visualizer, collision_visualizer = (
            motion_planning_space.meshcat,
            motion_planning_space.visualizer,
            motion_planning_space.collision_visualizer,
        )

        if input_config_frame == "panda_hand_eef_site_robomimic":
            # convert start and goal configs from panda_hand_eef_site_robomimic to panda_hand_gripper_site_robomimic
            start_cfg = motion_planning_space.convert_config_frame(
                start_cfg, "panda_hand_eef_site_robomimic"
            )
            goal_cfg = motion_planning_space.convert_config_frame(
                goal_cfg, "panda_hand_eef_site_robomimic"
            )
        # TODO(klin): hardcoded acc limits in setup_and_solve_trajopt_prog() mightn't work
        result, trajopt = self.setup_and_solve_trajopt_prog(
            plant,
            context,
            start_cfg[:7],
            start_cfg[7:],
            goal_cfg[:7],
            goal_cfg[7:],
            motion_planning_space=motion_planning_space,
            meshcat=meshcat,
            visualizer=visualizer,
            collision_visualizer=collision_visualizer,
            min_dist=0.0,
            min_dist_thresh=0.05,
            min_duration=0.5,
            max_duration=1,
            use_collision_constraint=True,
            robot_kinematic_ranges=motion_planning_space.bounds,
            num_init_guesses=num_trajopt_guesses,
            viz_trajopt_process=True,
        )
        if not result.is_success():
            import ipdb

            ipdb.set_trace()
            return None

        traj = trajopt.ReconstructTrajectory(result)
        timesteps = get_timesteps(traj.end_time(), 1 / self.action_freq)
        path = traj.vector_values(timesteps.tolist()).T

        # convert path from panda_hand_gripper_site_robomimic to panda_hand_eef_site_robomimic
        if input_config_frame == "panda_hand_eef_site_robomimic":
            path_in_eef_site = []
            for i in range(len(path)):
                X_ee_input_frame = motion_planning_space.get_end_effector_pose(
                    path[i], input_config_frame
                )
                # convert to quat_wxyz, pos, gripper
                path_in_eef_site.append(
                    np.concatenate(
                        [
                            np.roll(
                                R.from_matrix(X_ee_input_frame[:3, :3]).as_quat(),
                                shift=1,
                            ),
                            X_ee_input_frame[:3, 3],
                            path[i][-2:],
                        ]
                    )
                )
            path = path_in_eef_site
        return path

    def plan(self):
        # Call the trajopt planner to generate a trajectory
        traj = self.planner(self.robot, self.obstacles, self.start_conf, self.goal_conf)

        # Return the trajectory
        return traj
