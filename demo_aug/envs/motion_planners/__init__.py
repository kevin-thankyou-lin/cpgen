# Monkey patch curobo planner's retime trajectory to allow for time 'dilation' > 1


import time
from typing import List, Optional

import numpy as np
import torch
import torch.autograd.profiler as profiler
import trimesh

# CuRobo
from curobo.cuda_robot_model.cuda_robot_model import (
    CudaRobotModel,
    CudaRobotModelConfig,
)
from curobo.geom.sphere_fit import SphereFitType
from curobo.geom.types import (
    Material,
    Mesh,
    WorldConfig,
)
from curobo.rollout.cost.straight_line_cost import StraightLineCost
from curobo.rollout.rollout_base import Goal
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.state import JointState
from curobo.util.logger import log_error, log_info, log_warn
from curobo.util.torch_utils import get_torch_jit_decorator
from curobo.util.trajectory import InterpolateType, get_batch_interpolated_trajectory
from curobo.util.usd_helper import UsdHelper
from curobo.util_file import (
    get_robot_configs_path,
    join_path,
    load_yaml,
)
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenPlanConfig,
    MotionGenResult,
    MotionGenStatus,
)
from curobo.wrap.reacher.types import ReacherSolveState, ReacherSolveType
from scipy.spatial.transform import Rotation as R


def retime_trajectory(
    self,
    time_dilation_factor: float,
    interpolate_trajectory: bool = True,
    interpolation_dt: Optional[float] = None,
    interpolation_kind: InterpolateType = InterpolateType.LINEAR_CUDA,
    create_interpolation_buffer: bool = False,
):
    """Retime the optimized trajectory by a dilation factor.

    Args:
        time_dilation_factor: Time factor to slow down the trajectory. Should be less than 1.0.
        interpolate_trajectory: Interpolate the trajectory after retiming.
        interpolation_dt: Time between steps in the interpolated trajectory. If None,
            :attr:`MotionGenResult.interpolation_dt` is used.
        interpolation_kind: Interpolation type to use.
        create_interpolation_buffer: Create a new buffer for interpolated trajectory. Set this
            to True if existing buffer is not large enough to store new trajectory.
    """
    # allow for time dilation factor > 1 --- commented out this check
    # if time_dilation_factor > 1.0:
    #     log_error("time_dilation_factor should be less than 1.0")
    if time_dilation_factor == 1.0:
        return

    if len(self.path_buffer_last_tstep) > 1:
        log_error("only single result is supported")

    # unsqueeze interpolated plan if only 2 dimensions
    if len(self.interpolated_plan.shape) == 2:
        self.interpolated_plan = self.interpolated_plan.unsqueeze(0)

    new_dt = self.optimized_dt * (1.0 / time_dilation_factor)
    if len(self.optimized_plan.shape) == 3:
        new_dt = new_dt.view(-1, 1, 1)
    else:
        new_dt = new_dt.view(-1, 1)
    self.optimized_plan = self.optimized_plan.scale_by_dt(self.optimized_dt, new_dt)
    self.optimized_dt = new_dt.view(-1)
    if interpolate_trajectory:
        if interpolation_dt is not None:
            self.interpolation_dt = interpolation_dt
        self.interpolated_plan, last_tstep, _ = get_batch_interpolated_trajectory(
            self.optimized_plan,
            self.optimized_dt,
            self.interpolation_dt,
            kind=interpolation_kind,
            out_traj_state=self.interpolated_plan
            if not create_interpolation_buffer
            else None,
            tensor_args=self.interpolated_plan.tensor_args,
            optimize_dt=False,
        )

        self.path_buffer_last_tstep = [last_tstep[i] for i in range(len(last_tstep))]
        assert all([last_tstep[i] == last_tstep[0] for i in range(len(last_tstep))]), (
            "Assume all last tsteps are the same"
            "because assuming retiming only one trajectory"
        )
        # crop each element in self.interpolated_plan to the corres last_tstep in self.path_buffer_last_tstep
        self.interpolated_plan = self.interpolated_plan[
            :, : self.path_buffer_last_tstep[0]
        ]
        if len(self.optimized_plan.shape) == 2:
            self.interpolated_plan = self.interpolated_plan.squeeze(0)


@get_torch_jit_decorator()
def st_cost_custom(eef_positions: torch.Tensor, vec_weight, weight) -> torch.Tensor:
    """
    Computes the straightness metric for a batch of trajectories by calculating the minimum distance of each
    EEF position to a straight line connecting the first and last positions in the trajectory.

    Args:
        eef_positions (torch.Tensor): A tensor of shape (B, N, 3) representing B trajectories, each with N EEF positions in 3D.

    Returns:
        torch.Tensor: A tensor of shape (B,) containing the straightness metric for each trajectory.
    """
    # Extract the start and end points for each batch
    start_point = eef_positions[:, 0, :]  # Shape: (B, 3)
    end_point = eef_positions[:, -1, :]  # Shape: (B, 3)

    # Compute the line vector for each batch
    line_vector = end_point - start_point  # Shape: (B, 3)

    # Normalize the line vectors
    line_length = torch.norm(line_vector, dim=-1, keepdim=True)  # Shape: (B, 1)
    line_vector_normalized = line_vector / (
        line_length + 1e-8
    )  # Shape: (B, 3), avoid division by zero

    # Compute the projection lengths and projection points for all points in all batches
    start_to_points = eef_positions - start_point[:, None, :]  # Shape: (B, N, 3)
    projection_lengths = torch.sum(
        start_to_points * line_vector_normalized[:, None, :], dim=-1
    )  # Shape: (B, N)

    # Broadcast line_length to match projection_lengths dimensions (B, N)
    line_length_broadcast = line_length.expand_as(projection_lengths)

    # Clip projection lengths between 0 and line_length to keep points on the segment
    projection_lengths_clipped = torch.minimum(
        torch.maximum(projection_lengths, torch.zeros_like(projection_lengths)),
        line_length_broadcast,
    )  # Shape: (B, N)

    # Compute the projection points along the straight line
    projection_points = (
        start_point[:, None, :]
        + projection_lengths_clipped[:, :, None] * line_vector_normalized[:, None, :]
    )  # Shape: (B, N, 3)

    # Compute the orthogonal distances between the points and their projections
    distances = torch.norm(eef_positions - projection_points, dim=-1)  # Shape: (B, N)

    # Compute the average distance for each trajectory and use that for eah term in distance vector
    avg_distances = torch.mean(distances, dim=-1)  # Shape: (B,)

    # use the average distances for each value in the distance vector
    distances = avg_distances.unsqueeze(-1).expand_as(distances)  # Shape: (B, N)

    return distances * weight


# @get_torch_jit_decorator()
# def st_cost_custom(eef_positions: torch.Tensor, vec_weight, weight) -> torch.Tensor:
#     """
#     Computes the straightness metric by comparing actual positions with linearly interpolated
#     positions between start and end points.

#     Args:
#         eef_positions (torch.Tensor): A tensor of shape (B, N, 3) representing B trajectories, each with N EEF positions in 3D.

#     Returns:
#         torch.Tensor: A tensor of shape (B, N) containing the straightness metric for each point in each trajectory.
#     """
#     # Extract the start and end points for each batch
#     start_point = eef_positions[:, 0, :]  # Shape: (B, 3)
#     end_point = eef_positions[:, -1, :]  # Shape: (B, 3)

#     # Get the number of points
#     B, N, _ = eef_positions.shape

#     # Create interpolation factors from 0 to 1
#     t = torch.linspace(0, 1, N, device=eef_positions.device)  # Shape: (N,)
#     t = t.view(1, -1, 1).expand(B, N, 1)  # Shape: (B, N, 1)

#     # Compute ground truth positions through linear interpolation
#     # eef_position_gt = start + t * (end - start)
#     eef_positions_gt = start_point[:, None, :] + t * (end_point[:, None, :] - start_point[:, None, :])  # Shape: (B, N, 3)

#     # Compute L2 distances between actual and ground truth positions
#     distances = torch.norm(eef_positions - eef_positions_gt, dim=-1)  # Shape: (B, N)

#     return distances * weight


def straight_line_cost_forward(self, ee_pos_batch):
    cost = st_cost_custom(
        ee_pos_batch, self.vec_weight, self.weight
    )  # get the shape cost.shape
    return cost


def plan_single_js(
    self,
    start_state: JointState,
    goal_state: JointState,
    plan_config: MotionGenPlanConfig = MotionGenPlanConfig(),
    custom_trajopt_seed_traj: Optional[JointState] = None,
) -> MotionGenResult:
    """Plan a single motion to reach a goal joint state from a start joint state.

    This method uses trajectory optimization to find a collision-free path between the start
    and goal joint states. If trajectory optimization fails, it uses a graph planner to find
    a collision-free path to the goal joint state. The graph plan is then used as a seed for
    trajectory optimization.

    Args:
        start_state: Start joint state of the robot.
        goal_state: Goal joint state of the robot.
        plan_config: Planning parameters for motion generation.

    Returns:
        MotionGenResult: Result of motion generation. Check :attr:`MotionGenResult.success`
            attribute to see if the query was successful.
    """

    start_time = time.time()

    time_dict = {
        "solve_time": 0,
        "ik_time": 0,
        "graph_time": 0,
        "trajopt_time": 0,
        "trajopt_attempts": 0,
        "finetune_time": 0,
    }
    result = None
    # goal = Goal(goal_state=goal_state, current_state=start_state)
    solve_state = ReacherSolveState(
        ReacherSolveType.SINGLE,
        num_ik_seeds=1,
        num_trajopt_seeds=self.js_trajopt_solver.num_seeds,
        num_graph_seeds=self.js_trajopt_solver.num_seeds,
        batch_size=1,
        n_envs=1,
        n_goalset=1,
    )
    force_graph = plan_config.enable_graph
    valid_query = True
    if plan_config.check_start_validity:
        valid_query, status = self.check_start_state(start_state)
        if not valid_query:
            result = MotionGenResult(
                success=torch.as_tensor([False], device=self.tensor_args.device),
                valid_query=valid_query,
                status=status,
            )
            return result

    for n in range(plan_config.max_attempts):
        result = self._plan_js_from_solve_state(
            solve_state,
            start_state,
            goal_state,
            plan_config=plan_config,
            custom_trajopt_seed_traj=custom_trajopt_seed_traj,
        )
        time_dict["trajopt_time"] += result.trajopt_time
        time_dict["graph_time"] += result.graph_time
        time_dict["finetune_time"] += result.finetune_time
        time_dict["trajopt_attempts"] = n
        if plan_config.enable_graph_attempt is not None and (
            n >= plan_config.enable_graph_attempt - 1 and not plan_config.enable_graph
        ):
            plan_config.enable_graph = True
        if plan_config.disable_graph_attempt is not None and (
            n >= plan_config.disable_graph_attempt - 1 and not force_graph
        ):
            plan_config.enable_graph = False

        if result.success.item():
            break
        if not result.valid_query:
            break
        if time.time() - start_time > plan_config.timeout:
            break

    result.graph_time = time_dict["graph_time"]
    result.finetune_time = time_dict["finetune_time"]
    result.trajopt_time = time_dict["trajopt_time"]
    result.solve_time = result.trajopt_time + result.graph_time + result.finetune_time
    result.total_time = result.solve_time
    result.attempts = n
    if (
        plan_config.time_dilation_factor is not None
        and torch.count_nonzero(result.success) > 0
    ):
        result.retime_trajectory(
            plan_config.time_dilation_factor,
            interpolation_kind=self.js_trajopt_solver.interpolation_type,
        )
    return result


def _plan_js_from_solve_state(
    self,
    solve_state: ReacherSolveState,
    start_state: JointState,
    goal_state: JointState,
    plan_config: MotionGenPlanConfig = MotionGenPlanConfig(),
    custom_trajopt_seed_traj: Optional[JointState] = None,
) -> MotionGenResult:
    """Plan from a given reacher solve state for joint state.

    Args:
        solve_state: Reacher solve state for planning.
        start_state: Start joint state for planning.
        goal_state: Goal joint state to reach.
        plan_config: Planning parameters for motion generation.

    Returns:
        MotionGenResult: Result of planning.
    """
    trajopt_seed_traj = None
    trajopt_seed_success = None
    trajopt_newton_iters = self.js_trajopt_solver.newton_iters

    graph_success = 0
    if len(start_state.shape) == 1:
        log_error("Joint state should be not a vector (dof) should be (bxdof)")

    result = MotionGenResult(
        cspace_error=torch.zeros((1), device=self.tensor_args.device)
    )
    if self.store_debug_in_result:
        result.debug_info = {}
    # do graph search:
    if plan_config.enable_graph:
        start_config = torch.zeros(
            (solve_state.num_graph_seeds, self.js_trajopt_solver.dof),
            device=self.tensor_args.device,
            dtype=self.tensor_args.dtype,
        )
        goal_config = start_config.clone()
        start_config[:] = start_state.position
        goal_config[:] = goal_state.position
        interpolation_steps = None
        if plan_config.enable_opt:
            interpolation_steps = self.js_trajopt_solver.action_horizon
        log_info("MG: running GP")
        graph_result = self.graph_search(start_config, goal_config, interpolation_steps)
        trajopt_seed_success = graph_result.success

        graph_success = torch.count_nonzero(graph_result.success).item()
        result.graph_time = graph_result.solve_time
        result.solve_time += graph_result.solve_time
        if graph_success > 0:
            result.graph_plan = graph_result.interpolated_plan
            result.interpolated_plan = graph_result.interpolated_plan

            result.used_graph = True
            if plan_config.enable_opt:
                trajopt_seed = (
                    result.graph_plan.position.view(
                        1,  # solve_state.batch_size,
                        graph_success,  # solve_state.num_trajopt_seeds,
                        interpolation_steps,
                        self._dof,
                    )
                    .transpose(0, 1)
                    .contiguous()
                )
                trajopt_seed_traj = torch.zeros(
                    (
                        trajopt_seed.shape[0],
                        1,
                        self.trajopt_solver.action_horizon,
                        self._dof,
                    ),
                    device=self.tensor_args.device,
                    dtype=self.tensor_args.dtype,
                )
                trajopt_seed_traj[:, :, :interpolation_steps, :] = trajopt_seed
                trajopt_seed_success = graph_result.success

                trajopt_seed_success = trajopt_seed_success.view(
                    1, solve_state.num_trajopt_seeds
                )
                trajopt_newton_iters = self.graph_trajopt_iters
            else:
                _, idx = torch.topk(
                    graph_result.path_length[graph_result.success], k=1, largest=False
                )
                result.interpolated_plan = result.interpolated_plan[idx].squeeze(0)
                result.optimized_dt = self.tensor_args.to_device(self.interpolation_dt)
                result.optimized_plan = result.interpolated_plan[
                    : graph_result.path_buffer_last_tstep[idx.item()]
                ]
                idx = idx.view(-1) + self._batch_col
                result.cspace_error = torch.zeros((1), device=self.tensor_args.device)

                result.path_buffer_last_tstep = graph_result.path_buffer_last_tstep[
                    idx.item() : idx.item() + 1
                ]
                result.success = torch.as_tensor([True], device=self.tensor_args.device)
                return result
        else:
            result.success = torch.as_tensor([False], device=self.tensor_args.device)
            result.status = MotionGenStatus.GRAPH_FAIL
            if not graph_result.valid_query:
                result.valid_query = False
                if self.store_debug_in_result:
                    result.debug_info["graph_debug"] = graph_result.debug_info
                return result
            if plan_config.need_graph_success:
                return result

    # do trajopt:
    if plan_config.enable_opt:
        with profiler.record_function("motion_gen/setup_trajopt_seeds"):
            goal = Goal(
                current_state=start_state,
                goal_state=goal_state,
            )

            if (
                trajopt_seed_traj is None
                or graph_success < solve_state.num_trajopt_seeds * 1
            ):
                seed_goal = Goal(
                    current_state=start_state.repeat_seeds(
                        solve_state.num_trajopt_seeds
                    ),
                    goal_state=goal_state.repeat_seeds(solve_state.num_trajopt_seeds),
                )
                if trajopt_seed_traj is not None:
                    trajopt_seed_traj = trajopt_seed_traj.transpose(0, 1).contiguous()
                    # batch, num_seeds, h, dof
                    if trajopt_seed_success.shape[1] < self.js_trajopt_solver.num_seeds:
                        trajopt_seed_success_new = torch.zeros(
                            (1, solve_state.num_trajopt_seeds),
                            device=self.tensor_args.device,
                            dtype=torch.bool,
                        )
                        trajopt_seed_success_new[0, : trajopt_seed_success.shape[1]] = (
                            trajopt_seed_success
                        )
                        trajopt_seed_success = trajopt_seed_success_new
                # create seeds here:
                trajopt_seed_traj = self.js_trajopt_solver.get_seed_set(
                    seed_goal,
                    trajopt_seed_traj,  # batch, num_seeds, h, dof
                    num_seeds=self.js_trajopt_solver.num_seeds,
                    batch_mode=False,
                    seed_success=trajopt_seed_success,
                )
                if custom_trajopt_seed_traj is not None:
                    trajopt_seed_traj = custom_trajopt_seed_traj.position
                else:
                    trajopt_seed_traj = (
                        trajopt_seed_traj.view(
                            self.js_trajopt_solver.num_seeds * 1,
                            1,
                            self.trajopt_solver.action_horizon,
                            self._dof,
                        )
                        .contiguous()
                        .clone()
                    )
        if plan_config.enable_finetune_trajopt:
            og_value = self.trajopt_solver.interpolation_type
            self.js_trajopt_solver.interpolation_type = InterpolateType.LINEAR_CUDA
        with profiler.record_function("motion_gen/trajopt"):
            log_info("MG: running TO")
            traj_result = self._solve_trajopt_from_solve_state(
                goal,
                solve_state,
                trajopt_seed_traj,
                num_seeds_override=solve_state.num_trajopt_seeds,
                newton_iters=trajopt_newton_iters,
                return_all_solutions=plan_config.enable_finetune_trajopt,
                trajopt_instance=self.js_trajopt_solver,
            )
        if plan_config.enable_finetune_trajopt:
            self.trajopt_solver.interpolation_type = og_value
        if self.store_debug_in_result:
            result.debug_info["trajopt_result"] = traj_result
        if torch.count_nonzero(traj_result.success) == 0:
            result.status = MotionGenStatus.TRAJOPT_FAIL
        # run finetune
        if (
            plan_config.enable_finetune_trajopt
            and torch.count_nonzero(traj_result.success) > 0
        ):
            with profiler.record_function("motion_gen/finetune_trajopt"):
                seed_traj = traj_result.raw_action.clone()
                og_solve_time = traj_result.solve_time
                opt_dt = traj_result.optimized_dt
                opt_dt = torch.min(opt_dt[traj_result.success])
                finetune_time = 0
                newton_iters = None
                for k in range(plan_config.finetune_attempts):
                    scaled_dt = torch.clamp(
                        opt_dt
                        * plan_config.finetune_js_dt_scale
                        * (plan_config.finetune_dt_decay ** (k)),
                        self.js_trajopt_solver.minimum_trajectory_dt,
                    )

                    if self.optimize_dt:
                        self.finetune_js_trajopt_solver.update_solver_dt(
                            scaled_dt.item()
                        )
                    traj_result = self._solve_trajopt_from_solve_state(
                        goal,
                        solve_state,
                        seed_traj,
                        trajopt_instance=self.finetune_js_trajopt_solver,
                        num_seeds_override=solve_state.num_trajopt_seeds,
                        newton_iters=newton_iters,
                        return_all_solutions=False,
                    )

                    finetune_time += traj_result.solve_time
                    if (
                        torch.count_nonzero(traj_result.success) > 0
                        or not self.optimize_dt
                    ):
                        break
                    seed_traj = traj_result.optimized_seeds.detach().clone()
                    newton_iters = 4

                result.finetune_time = finetune_time

                traj_result.solve_time = og_solve_time
            if self.store_debug_in_result:
                result.debug_info["finetune_trajopt_result"] = traj_result
            if torch.count_nonzero(traj_result.success) == 0:
                result.status = MotionGenStatus.FINETUNE_TRAJOPT_FAIL
                if (
                    traj_result.debug_info is not None
                    and "dt_exception" in traj_result.debug_info
                    and traj_result.debug_info["dt_exception"]
                ):
                    result.status = MotionGenStatus.DT_EXCEPTION

        elif plan_config.enable_finetune_trajopt:
            traj_result.success = traj_result.success[0:1]
        result.solve_time += traj_result.solve_time + result.finetune_time
        result.trajopt_time = traj_result.solve_time
        result.trajopt_attempts = 1
        result.success = traj_result.success
        result.interpolated_plan = traj_result.interpolated_solution.trim_trajectory(
            0, traj_result.path_buffer_last_tstep[0]
        )

        result.interpolation_dt = self.trajopt_solver.interpolation_dt
        result.path_buffer_last_tstep = traj_result.path_buffer_last_tstep
        result.cspace_error = traj_result.cspace_error
        result.optimized_dt = traj_result.optimized_dt
        result.optimized_plan = traj_result.solution
        result.goalset_index = traj_result.goalset_index

    return result


def write_trajectory_animation(
    robot_model_file: str,
    world_model: WorldConfig,
    q_start: JointState,
    q_traj: JointState,
    dt: float = 0.02,
    save_path: str = "out.usd",
    tensor_args: TensorDeviceType = TensorDeviceType(),
    interpolation_steps: float = 1.0,
    robot_base_frame="robot",
    base_frame="/world",
    kin_model: Optional[CudaRobotModel] = None,
    visualize_robot_spheres: bool = True,
    robot_color: Optional[List[float]] = None,
    flatten_usd: bool = False,
    goal_pose: Optional[Pose] = None,
    goal_color: Optional[List[float]] = None,
):
    """
    EDITED: removes code related to visualizing robot meshes as they don't seem to exist.
    """
    if kin_model is None:
        config_file = load_yaml(join_path(get_robot_configs_path(), robot_model_file))
        if "robot_cfg" not in config_file:
            config_file["robot_cfg"] = config_file
        config_file["robot_cfg"]["kinematics"]["load_link_names_with_mesh"] = True
        robot_cfg = CudaRobotModelConfig.from_data_dict(
            config_file["robot_cfg"]["kinematics"], tensor_args=tensor_args
        )
        kin_model = CudaRobotModel(robot_cfg)
    m = kin_model.get_robot_link_meshes()

    robot_mesh_model = WorldConfig(mesh=m)
    if robot_color is not None:
        robot_mesh_model.add_color(robot_color)
        robot_mesh_model.add_material(Material(metallic=0.4))
    if goal_pose is not None:
        kin_model.link_names
        if kin_model.ee_link in kin_model.kinematics_config.mesh_link_names:
            index = kin_model.kinematics_config.mesh_link_names.index(kin_model.ee_link)
            gripper_mesh = m[index]
        if len(goal_pose.shape) == 1:
            goal_pose = goal_pose.unsqueeze(0)
        if len(goal_pose.shape) == 2:
            goal_pose = goal_pose.unsqueeze(0)
        for i in range(goal_pose.n_goalset):
            g = goal_pose.get_index(0, i).to_list()
            world_model.add_obstacle(
                Mesh(
                    file_path=gripper_mesh.file_path,
                    pose=g,
                    name="goal_idx_" + str(i),
                    color=goal_color,
                )
            )
    usd_helper = UsdHelper()
    usd_helper.create_stage(
        save_path,
        timesteps=q_traj.position.shape[0],
        dt=dt,
        interpolation_steps=interpolation_steps,
        base_frame=base_frame,
    )
    if world_model is not None:
        usd_helper.add_world_to_stage(world_model, base_frame=base_frame)

    # remove the following code because links don't always exist it seems
    # animation_links = kin_model.kinematics_config.mesh_link_names
    # animation_poses = kin_model.get_link_poses(q_traj.position.contiguous(), animation_links)
    # # add offsets for visual mesh:
    # for i, ival in enumerate(offsets):
    #     offset_pose = Pose.from_list(ival)
    #     new_pose = Pose(
    #         animation_poses.position[:, i, :], animation_poses.quaternion[:, i, :]
    #     ).multiply(offset_pose)
    #     animation_poses.position[:, i, :] = new_pose.position
    #     animation_poses.quaternion[:, i, :] = new_pose.quaternion

    # robot_base_frame = join_path(base_frame, robot_base_frame)

    # usd_helper.create_animation(
    #     robot_mesh_model, animation_poses, base_frame, robot_frame=robot_base_frame
    # )
    if visualize_robot_spheres:
        # visualize robot spheres:
        sphere_traj = kin_model.get_robot_as_spheres(q_traj.position)
        # change color:
        for s in sphere_traj:
            for k in s:
                k.color = [0, 0.27, 0.27, 1.0]
        usd_helper.create_obstacle_animation(
            sphere_traj, base_frame=base_frame, obstacles_frame="curobo/robot_collision"
        )
    usd_helper.write_stage_to_file(save_path, flatten=flatten_usd)


def attach_objects_to_robot(
    self,
    joint_state: JointState,
    object_names: List[str],
    surface_sphere_radius: float = 0.001,
    link_name: str = "attached_object",
    sphere_fit_type: SphereFitType = SphereFitType.VOXEL_VOLUME_SAMPLE_SURFACE,
    voxelize_method: str = "ray",
    world_objects_pose_offset: Optional[Pose] = None,
    remove_obstacles_from_world_config: bool = False,
    expand_meshes_factor: float = 0.002,
    viz_merged_trimesh_mesh_pts: bool = False,
) -> bool:
    """
    EDITED: combine all the objects into a single mesh and then get the bounding spheres for the combined mesh.
        Also expands the individual meshes by a factor of expand_meshes_factor before combining them in case
        original individual geometries are supposed to overlap but don't quite do so.

    Attach an object or objects from world to a robot's link.

    This method assumes that the objects exist in the world configuration. If attaching
    objects that are not in world, use :meth:`MotionGen.attach_external_objects_to_robot`.

    Args:
        joint_state: Joint state of the robot.
        object_names: Names of objects in the world to attach to the robot.
        surface_sphere_radius: Radius (in meters) to use for points sampled on surface of the
            object. A smaller radius will allow for generating motions very close to obstacles.
        link_name: Name of the link (frame) to attach the objects to. The assumption is that
            this link does not have any geometry and all spheres of this link represent
            attached objects.
        sphere_fit_type: Sphere fit algorithm to use. See :ref:`attach_object_note` for more
            details. The default method :attr:`SphereFitType.VOXEL_VOLUME_SAMPLE_SURFACE`
            voxelizes the volume of the objects and adds spheres representing the voxels, then
            samples points on the surface of the object, adds :attr:`surface_sphere_radius` to
            these points. This should be used for most cases.
        voxelize_method: Method to use for voxelization, passed to
            :py:func:`trimesh.voxel.creation.voxelize`.
        world_objects_pose_offset: Offset to apply to the object poses before attaching to the
            robot. This is useful when attaching an object that's in contact with the world.
            The offset is applied in the world frame before attaching to the robot.
        remove_obstacles_from_world_config: Remove the obstacles from the world cache after
            attaching to the robot to reduce memory usage. Note that when an object is attached
            to the robot, it's disabled in the world collision checker. This flag when enabled,
            also removes the object from world cache. For most cases, this should be set to
            False.
    """

    log_info("MG: Attach objects to robot")
    kin_state = self.compute_kinematics(joint_state)
    ee_pose = kin_state.ee_pose  # w_T_ee
    if world_objects_pose_offset is not None:
        # add offset from ee:
        ee_pose = world_objects_pose_offset.inverse().multiply(ee_pose)
        # new ee_pose:
        # w_T_ee = offset_T_w * w_T_ee
        # ee_T_w
    ee_pose = ee_pose.inverse()  # ee_T_w to multiply all objects later
    max_spheres = self.robot_cfg.kinematics.kinematics_config.get_number_of_spheres(
        link_name
    )
    n_spheres = int(max_spheres / len(object_names))
    sphere_tensor = torch.zeros((max_spheres, 4))
    sphere_tensor[:, 3] = -10.0
    sph_list = []
    if n_spheres == 0:
        log_warn(
            "MG: No spheres found, max_spheres: "
            + str(max_spheres)
            + " n_objects: "
            + str(len(object_names))
        )
        return False

    all_meshes = []
    for i, x in enumerate(object_names):
        obs = self.world_model.get_obstacle(x)
        if obs is None:
            log_error(
                "Object not found in world. Object name: "
                + x
                + " Name of objects in world: "
                + " ".join([i.name for i in self.world_model.objects])
            )
        trimesh_mesh = obs.get_trimesh_mesh()

        pose_matrix = np.eye(4)
        pose_matrix[:3, 3] = obs.pose[:3]
        quat_wxyz = obs.pose[3:7]
        rotation = R.from_quat(np.roll(quat_wxyz, -1)).as_matrix()
        pose_matrix[:3, :3] = rotation
        trimesh_mesh.apply_transform(pose_matrix)

        vertex_normals = trimesh_mesh.vertex_normals
        expanded_vertices = (
            trimesh_mesh.vertices + vertex_normals * expand_meshes_factor
        )
        vertex_colors = np.random.randint(
            0, 256, size=(expanded_vertices.shape[0], 4), dtype=np.uint8
        )
        expanded_trimesh_mesh = trimesh.Trimesh(
            vertices=expanded_vertices,
            faces=trimesh_mesh.faces,
            face_colors=np.random.randint(
                0, 256, size=expanded_vertices.shape, dtype=np.uint8
            ),
            vertex_colors=vertex_colors,
        )

        all_meshes.append(expanded_trimesh_mesh)
        self.world_coll_checker.enable_obstacle(enable=False, name=x)
        if remove_obstacles_from_world_config:
            self.world_model.remove_obstacle(x)
    log_info("MG: Computed spheres for attach objects to robot")

    if len(all_meshes) > 1:
        combined_mesh = trimesh.boolean.union(all_meshes)
    else:
        combined_mesh = all_meshes[0]

    if viz_merged_trimesh_mesh_pts:
        # Sample points on the surface of the merged mesh using the custom function
        sampled_points, _ = trimesh.sample.sample_surface_even(combined_mesh, 4000)
        # Convert sampled_points to a valid shape (flatten if nested)
        sampled_points = np.array(sampled_points)
        # Create a point cloud mesh for visualization
        points_mesh = trimesh.points.PointCloud(
            sampled_points, colors=[255, 255, 0]
        )  # Yellow points
        sampled_scene = trimesh.Scene([points_mesh])
        sampled_scene.show()

        # viz the merged mesh
        mesh_scene = trimesh.Scene([combined_mesh])
        mesh_scene.show()

    # visualize combined mesh
    combined_mesh_obstacle = Mesh(
        name="combined_mesh",
        pose=[0, 0, 0, 1, 0, 0, 0],
        vertices=combined_mesh.vertices.tolist(),
        faces=combined_mesh.faces.tolist(),
        vertex_colors=combined_mesh.visual.vertex_colors.tolist()
        if combined_mesh.visual.vertex_colors is not None
        else None,
        face_colors=combined_mesh.visual.face_colors.tolist()
        if combined_mesh.visual.face_colors is not None
        else None,
    )

    # Step 3: Get bounding spheres for the new combined Mesh obstacle
    max_spheres = self.robot_cfg.kinematics.kinematics_config.get_number_of_spheres(
        link_name
    )
    sphere_tensor = torch.zeros((max_spheres, 4))
    sphere_tensor[:, 3] = -10.0  # Default to inactive spheres

    spheres = combined_mesh_obstacle.get_bounding_spheres(
        max_spheres,
        surface_sphere_radius,
        pre_transform_pose=ee_pose,
        tensor_args=self.tensor_args,
        fit_type=sphere_fit_type,
        voxelize_method=voxelize_method,
    )

    sph_list = [s.position + [s.radius] for s in spheres]

    spheres = self.tensor_args.to_device(torch.as_tensor(sph_list))

    if spheres.shape[0] > max_spheres:
        spheres = spheres[: spheres.shape[0]]
    sphere_tensor[: spheres.shape[0], :] = spheres.contiguous()

    self.attach_spheres_to_robot(sphere_tensor=sphere_tensor, link_name=link_name)

    return True


MotionGenResult.retime_trajectory = retime_trajectory
StraightLineCost.forward = straight_line_cost_forward

MotionGen.plan_single_js = plan_single_js
MotionGen._plan_js_from_solve_state = _plan_js_from_solve_state
MotionGen.attach_objects_to_robot = attach_objects_to_robot

UsdHelper.write_trajectory_animation = staticmethod(write_trajectory_animation)
