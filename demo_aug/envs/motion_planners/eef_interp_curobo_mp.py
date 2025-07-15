import logging
import os
import pathlib
import time
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import imageio
import mujoco
import numpy as np
from mink import Configuration
from scipy.interpolate import interp1d

from demo_aug.envs.motion_planners.base_mp import MotionPlanner
from demo_aug.envs.motion_planners.curobo_mp import CuroboMotionPlanner
from demo_aug.envs.motion_planners.eef_interp_mp import EEFInterpMinkMotionPlanner
from demo_aug.envs.motion_planners.indexed_configuration import IndexedConfiguration
from demo_aug.utils.mujoco_utils import (
    check_geom_collisions,
    get_body_name,
    get_min_geom_distance,
    get_subtree_geom_ids_by_group,
    get_top_level_bodies,
)


def save_plan_video(
    env,
    plan: List[np.ndarray],
    save_plan_path: pathlib.Path,
    fps: int = 20,
    num_cameras: int = 6,
) -> None:
    """
    Save a video of a robot executing a motion plan.

    Args:
        env: The simulation environment containing the robot and its scene.
        plan: The motion plan containing joint positions (plan.position).
        save_plan_path (Path): Path to save the output video file.
        fps (int): Frames per second for the video. Defaults to 20.
        num_cameras (int): Number of cameras to render from. Defaults to 6.

    Raises:
        AssertionError: If the MUJOCO_GL environment variable is not set to 'egl' or 'osmesa'.
    """
    assert os.environ.get("MUJOCO_GL") in ["egl", "osmesa"], (
        f"Currently, MUJOCO_GL={os.environ.get('MUJOCO_GL')}. "
        "Need MUJOCO_GL=egl or MUJOCO_GL=osmesa for offscreen rendering."
    )

    save_plan_path.parent.mkdir(parents=True, exist_ok=True)
    video_writer = imageio.get_writer(save_plan_path, fps=fps)

    model = env.sim.model._model
    data = env.sim.data._data

    # Backup the current robot state
    qpos = data.qpos.copy()
    qvel = data.qvel.copy()
    qacc = data.qacc.copy()

    try:
        with mujoco.Renderer(model) as renderer:
            for joint_positions in plan:
                # Update robot joint state
                env.robots[0].set_robot_joint_positions(joint_positions)

                video_img = []
                for cam in range(num_cameras):
                    renderer.update_scene(data=data, camera=cam)
                    frame = renderer.render()
                    video_img.append(frame)

                video_img = np.concatenate(video_img, axis=1)
                video_writer.append_data(video_img)

    finally:
        # Restore the original robot state
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        data.qacc[:] = qacc

        video_writer.close()
        logging.info(f"Video saved to {save_plan_path}")


def retime_trajectory(
    positions: np.ndarray,
    distance_to_collision: np.ndarray,
    threshold: float = 0.015,
    slow_factor_min: float = 1.3,
    slow_factor_max: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Retimes a given trajectory based on distance to collision, adjusting the total time length as needed.

    Args:
        positions (np.ndarray): Original positions along the trajectory (N x D, where D is the dimensionality).
        distance_to_collision (np.ndarray): Distance to collision at each point.
        threshold (float): Distance threshold below which to slow down.
        slow_factor_min (float): Minimum factor by which to slow down if near the threshold.
        slow_factor_max (float): Maximum factor by which to slow down if distance is very low.

    Returns:
        tuple: Retimed positions and the corresponding retimed time vector.
    """
    N = len(positions)
    time = np.linspace(0, 1, N)  # Uniform time allocation from 0 to 1 seconds

    # Create a new time vector based on the distance to collision
    new_times = [time[0]]  # Start at the same initial time
    for i in range(1, N):
        dt = time[i] - time[i - 1]
        if distance_to_collision[i] < threshold:
            # Calculate slow factor based on distance to collision
            slow_factor = slow_factor_min + (slow_factor_max - slow_factor_min) * (
                1 - distance_to_collision[i] / threshold
            )
            slow_factor = min(slow_factor_max, max(slow_factor_min, slow_factor))
            dt *= slow_factor  # Slow down based on distance to collision
        new_times.append(new_times[-1] + dt)
    new_times = np.array(new_times)

    # Adjust the time vector to have the same dt but go up to the new final time
    final_time = new_times[-1]
    dt = time[1] - time[0]  # Original time step size
    retimed_time = np.arange(
        0, final_time, dt
    )  # Ensure we cover the entire range up to final_time

    # Ensure the final time is included, even if it doesn't align with the dt
    if final_time - retimed_time[-1] <= dt / 4:
        retimed_time[-1] = final_time
    else:
        retimed_time = np.append(retimed_time, final_time)

    # Interpolate each dimension of the retimed trajectory to match the adjusted time length using interp1d
    try:
        interp_func = interp1d(
            new_times, positions, kind="cubic", axis=0, fill_value="extrapolate"
        )
    except Exception:
        import ipdb

        ipdb.set_trace()
        interp_func = interp1d(
            new_times, positions, kind="cubic", axis=0, fill_value="extrapolate"
        )

    retimed_positions = interp_func(retimed_time)
    return retimed_positions, new_times


class EEFInterpCuroboMotionPlanner(MotionPlanner):
    name: str = "eef_interp_curobo"

    def __init__(
        self,
        env,
        save_dir: Optional[pathlib.Path] = None,
        num_steps: Optional[int] = None,
        max_eef_vel: float = 0.15,
        mink_robot_configuration: Optional[Configuration] = None,
        curobo_goal_type: Literal["joint", "pose", "pose_wxyz_xyz"] = "joint",
        robot_type: Literal["franka_umi", "franka"] = "franka",
    ):
        self.save_dir = save_dir
        self.env = env
        self.eef_interp_mink_planner: EEFInterpMinkMotionPlanner = (
            EEFInterpMinkMotionPlanner(env, save_dir, num_steps, max_eef_vel)
        )
        self.curobo_planner: CuroboMotionPlanner = CuroboMotionPlanner(
            env,
            save_dir=save_dir,
            goal_type=curobo_goal_type,
            robot_type=robot_type,
        )
        self.dof = self.curobo_planner.dof
        self.mink_robot_configuration = mink_robot_configuration

    def plan(
        self,
        X_ee_start: np.ndarray,
        X_ee_goal: np.ndarray,
        robot_configuration: IndexedConfiguration,
        q_start: np.ndarray,
        q_goal: np.ndarray,
        q_gripper_start: np.ndarray,
        ee_goal_curobo: Optional[np.ndarray] = None,
        X_base_goal: Optional[np.ndarray] = None,
        retract_q: Optional[np.ndarray] = None,
        retract_q_weights: Optional[np.ndarray] = None,
        curobo_goal_type: Literal["joint", "pose", "pose_wxyz_xyz"] = "joint",
        visualize: bool = False,
        eef_frame_name: str = "gripper0_right_grip_site",
        eef_frame_type: str = "site",
        verbose: bool = False,
        use_eef_warm_start: bool = False,
        duplicate_last_q: bool = False,  # hack for eef tracking error
        rescale_traj_based_on_min_dist: bool = False,  # currently hack implementation
    ) -> Tuple[Optional[np.ndarray], Union[np.ndarray, bool]]:
        if self.mink_robot_configuration is not None:
            q_start_full = np.concatenate(
                [q_start, np.zeros(2) if q_gripper_start is None else q_gripper_start]
            )
            self.mink_robot_configuration.update(q_start_full)
            X_ee_start_mink = (
                self.mink_robot_configuration.get_transform_frame_to_world(
                    "gripper0_right_grip_site", "site"
                ).as_matrix()
            )
            q_goal_full = np.concatenate(
                [q_goal, np.zeros(2) if q_gripper_start is None else q_gripper_start]
            )
            self.mink_robot_configuration.update(q_goal_full)
            X_ee_goal_mink = self.mink_robot_configuration.get_transform_frame_to_world(
                "gripper0_right_grip_site", "site"
            ).as_matrix()
            a = time.time()
            robot_qs, _ = self.eef_interp_mink_planner.plan(
                X_ee_start_mink,
                X_ee_goal_mink,
                q_start_full,
                robot_configuration=self.mink_robot_configuration,
                q_retract=retract_q,
                retract_q_weights=retract_q_weights,
                eef_frame_name=eef_frame_name,
                eef_frame_type=eef_frame_type,
            )
            b = time.time()
        if self.save_dir is not None:
            mink_save_plan_path = (
                self.save_dir / f"eef-mink-{time.strftime('%Y%m%d%H%M%S')}.mp4"
            )
            save_plan_video(
                self.env.env,
                plan=np.array(robot_qs)[..., : self.curobo_planner.dof],
                save_plan_path=mink_save_plan_path,
                num_cameras=3,
            )
            logging.info(f"saved mink path to: {mink_save_plan_path}")

        # Check for collisions between robot and environment #
        model = robot_configuration.model
        data = robot_configuration.data
        robot_geoms = get_subtree_geom_ids_by_group(
            model, model.body("gripper0_right_right_gripper").id, group=0
        )
        body_ids = get_top_level_bodies(model, exclude_prefixes=["robot", "gripper"])
        body_names = [get_body_name(model, body_id) for body_id in body_ids]
        if verbose:
            logging.info(
                f"EEFInterpCuroboMotionPlanner. Collision checking. Top-level body_names: {body_names}"
            )
        non_robot_geoms = [
            geom_id
            for body_id in body_ids
            for geom_id in get_subtree_geom_ids_by_group(model, body_id, group=0)
        ]
        geom_pairs_to_check: List[Tuple] = [(robot_geoms, non_robot_geoms)]
        # TODO: fix collision checking: robot_configuration doesn't match actual env
        is_collision = False
        for robot_q in robot_qs:
            robot_configuration.update(robot_q)
            is_collision = (
                len(
                    check_geom_collisions(
                        model,
                        data,
                        geom_pairs_to_check,
                        collision_activation_dist=0.002,
                    )
                )
                > 0
            )
            if is_collision:
                logging.info(
                    "Collision detected from using eef_interp motion planner! Falling back to curobo planner."
                )
                break
        c = time.time()
        # TODO(klin): fix mismatch between robot_configuration and actual env
        # If there are collisions between robot and environment, then directly return robot_qs
        if not is_collision:
            logging.info(
                "Not collision detected from using eef_interp motion planner! Using straight line path."
            )
            return np.array(robot_qs)[..., : self.curobo_planner.dof], True

        warm_start_traj = None
        if use_eef_warm_start:  # sometimes warm-start is worse
            # Otherwise, use curobo planner to plan a path. warm_start_traj should have curobo expected shape
            robot_qs, _ = self.eef_interp_mink_planner.plan(
                X_ee_start,
                X_ee_goal,
                q_start,
                robot_configuration=robot_configuration,
                q_retract=retract_q,
                retract_q_weights=retract_q_weights,
                eef_frame_name=eef_frame_name,
                eef_frame_type=eef_frame_type,
                num_steps=32,
            )
            warm_start_traj = np.array(robot_qs)[..., : self.curobo_planner.dof]

        q_goal_valid: bool = self.curobo_planner.motion_gen.ik_solver.check_valid(
            self.curobo_planner.motion_gen.tensor_args.to_device(
                q_goal[..., : self.dof]
            )
        )
        if not q_goal_valid:
            logging.info(
                "q_goal is invalid. Falling back to curobo planner will fail, exiting ..."
            )
            from curobo.wrap.reacher.motion_gen import MotionGenResult

            return MotionGenResult(success=False, valid_query=False), False
        d = time.time()
        # Use the eef_interp plan as the initial guess for curobo planner
        q, curobo_result = self.curobo_planner.plan(
            q_start,
            q_goal if curobo_goal_type == "joint" else ee_goal_curobo,
            warm_start_traj=warm_start_traj,
            goal_type=curobo_goal_type,
            robot_configuration=robot_configuration,
            visualize=visualize,
        )
        e = time.time()
        if rescale_traj_based_on_min_dist and q is not None:
            # setting q for env.sim doesn't also set the grasped object's pose ...
            # TODO(fix): robot_configuration.model isn't correct
            min_dists = np.ones(len(q))
            for i, q_i in enumerate(q):
                min_dists[i] = get_min_geom_distance(
                    model, data, geom_pairs_to_check, activation_dist=0.02
                )
            min_dists[-min(5, len(min_dists)) :] = 0.003
            q, retimed_times = retime_trajectory(q, min_dists)

        if duplicate_last_q and q is not None:
            # duplicate the last q N times
            duplication_times = 3
            last_elem_repeated = np.tile(q[-1], (duplication_times, 1))
            q = np.vstack([q, last_elem_repeated])
        logging.info(
            f"EEFInterpCuroboMotionPlanner: eef_interp time: {b-a:.3f}, curobo time: {e-d:.3f}. Collision checking time: {c-b:.3f}."
            f" Total time: {e-a:.3f}"
        )
        return q, curobo_result

    def visualize_motion(self, configs: np.ndarray):
        self.curobo_planner.visualize_motion(configs)

    def update_env(
        self,
        obs: Dict[str, Any],
        env,
        env_type: Literal["mujoco"] = "mujoco",
        attach_object_names: Optional[List[str]] = None,
        batch_size: int = 1,
    ):
        self.curobo_planner.update_env(
            obs, env, env_type, attach_object_names, batch_size
        )
