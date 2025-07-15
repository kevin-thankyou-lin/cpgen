import logging
import os
import pathlib
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import imageio
import mujoco
import numpy as np
import torch
import yaml
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModelConfig
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.sphere_fit import SphereFitType
from curobo.types.base import TensorDeviceType
from curobo.types.file_path import ContentPath
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.util_file import (
    get_robot_path,
    get_task_configs_path,
    join_path,
    load_yaml,
)
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
)
from mink import Configuration
from mujoco import MjData, MjModel, mj_fwdPosition, viewer

import demo_aug
from demo_aug.envs.motion_planners.base_mp import MotionPlanner
from demo_aug.envs.motion_planners.curobo_utils import (
    create_curobo_world_config,
    get_geom_poses,
    mjmodel_to_mjgeoms,
    update_curobo_world_config,
)


def get_configs_with_straight_line_cost(
    config_mapping: Dict[str, str],
) -> Dict[str, str]:
    """
    Takes a mapping of config names to yaml files, adds straight line cost config,
    saves new versions, and returns mapping to new filenames

    Args:
        config_mapping: dict mapping config names to yaml files
        e.g. {'base_cfg_file': 'base_cfg.yml', ...}

    Returns:
        dict mapping config names to modified yaml files
        e.g. {'base_cfg_file': 'base_cfg_with_stcost.yml', ...}
    """
    straight_line_cost = {
        "cost": {"straight_line_cfg": {"vec_weight": [1, 1, 1], "weight": 50000}},
    }

    updated_mapping = {}

    for config_name, yaml_file in config_mapping.items():
        # Load original yaml
        yaml_path = Path(get_task_configs_path()) / yaml_file
        with open(yaml_path, "r") as f:
            config_data = yaml.safe_load(f)

        # Update with straight line cost config
        if "cost" in config_data:
            config_data["cost"].update(straight_line_cost["cost"])
        else:
            config_data.update(straight_line_cost)

        # Generate new filename and save
        base_name = yaml_file.split(".")[0]
        updated_file = f"{base_name}_with_stcost.yml"
        updated_mapping[config_name] = updated_file

        save_path = Path(get_task_configs_path()) / updated_file
        with open(save_path, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False)

    return updated_mapping


def create_passive_viewer(
    model: MjModel,
    data: MjData,
    rate: float = 0.5,
    run_physics: bool = False,
    robot_configuration: Optional[Configuration] = None,
    **kwargs: Any,
) -> None:
    """
    Create a passive viewer for the given MuJoCo model and data.
    Args:
        model (MjModel): The MuJoCo model object.
        data (MjData): The MuJoCo data object.
        rate (float): The update rate for the viewer (in seconds).
        run_physics (bool): Whether to run physics in the background.
        **kwargs: Additional keyword arguments.
    """
    with viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as vis:
        mj_fwdPosition(model, data)  # Initialize positions
        qs = kwargs.get("qs", None)
        i: int = 0
        while vis.is_running():
            if run_physics:
                mujoco.mj_step(model, data)
            if robot_configuration is not None and qs is not None:
                if i == len(qs):
                    break
                robot_configuration.update(qs[i])
                i += 1
            vis.sync()
            time.sleep(rate)
        vis.close()


class CuroboMotionPlanner(MotionPlanner):
    name: str = "curobo"

    def __init__(
        self,
        env,
        save_dir: Optional[pathlib.Path] = None,
        robot_file: Optional[str] = None,
        robot_type: Literal["franka_umi", "franka"] = "franka",
        goal_type: Literal["joint", "pose", "pose_wxyz_xyz"] = "joint",
    ):
        super().__init__(env, save_dir)

        self.goal_type = goal_type  # useful for warming up correctly

        robot_file = (
            join_path(get_robot_path(), "franka.yml")
            if robot_file is None
            else robot_file
        )

        if robot_type == "franka":
            self.robot_config = load_yaml(str(robot_file))["robot_cfg"]
            self.robot_config["kinematics"]["extra_collision_spheres"][
                "attached_object"
            ] = 400
        elif robot_type == "franka_umi":
            # need to update the spheres stuff ... and urdf?
            demo_aug_path = pathlib.Path(demo_aug.__file__).parent
            robot_config_root_path = f"{demo_aug_path}/models/curobo/robot"
            self.robot_config = RobotConfig(
                CudaRobotModelConfig.from_content_path(
                    ContentPath(
                        robot_config_root_path=robot_config_root_path,
                        robot_config_file="franka_umi.yml",
                    )
                )
            )

        self.env = env.env
        # probably better to merge the geoms if belonging to the same body and no articulation actuation
        model = self.env.sim.model._model
        mj_data = self.env.sim.data._data
        self.mj_geoms = mjmodel_to_mjgeoms(model)
        self.base_pos = self.env.sim.data.get_body_xpos("robot0_base")
        geom_name_to_pose = get_geom_poses(self.mj_geoms, mj_data, self.base_pos)
        self.world_config = create_curobo_world_config(self.mj_geoms, geom_name_to_pose)
        self.body_to_geom_names = defaultdict(list)
        for mj_geom in self.mj_geoms:
            geom_name = mj_geom.name
            body_name = mj_geom.body_name
            self.body_to_geom_names[body_name].append(geom_name)

        original_config_files = {
            "base_cfg_file": "base_cfg.yml",
            "particle_ik_file": "particle_ik.yml",
            "gradient_ik_file": "gradient_ik.yml",
            "particle_trajopt_file": "particle_trajopt.yml",
            "gradient_trajopt_file": "gradient_trajopt.yml",
        }
        new_config_files = get_configs_with_straight_line_cost(original_config_files)

        self.motion_gen_config = MotionGenConfig.load_from_robot_config(
            self.robot_config,
            self.world_config,
            collision_checker_type=CollisionCheckerType.MESH,  # for mesh collision
            collision_activation_distance=0.012,  # higher tends to lead to finetune trajopt failure [for pose target]
            # rotation_threshold=0.0005,
            # position_threshold=0.001,
            num_graph_seeds=1,
            num_ik_seeds=32,
            num_trajopt_seeds=4,
            # trajopt_tsteps=32,
            # js_trajopt_tsteps=32,
            collision_cache={"obb": 10},
            interpolation_dt=1 / 30,
            use_cuda_graph=True,
            # doing so doesn't affect/help robot collision geometry after warm up
            # adding more spheres to robot collision doesn't register if warmup
            base_cfg_file=new_config_files["base_cfg_file"],
            particle_ik_file=new_config_files["particle_ik_file"],
            gradient_ik_file=new_config_files["gradient_ik_file"],
            particle_trajopt_file=new_config_files["particle_trajopt_file"],
            gradient_trajopt_file=new_config_files["gradient_trajopt_file"],
            # finetune_trajopt_iters=300 +300,
            # grad_trajopt_iters=200 + 200,
        )
        # TODO: add .yml files for straight line cost weights
        self.warmed_up = False
        self.motion_gen = MotionGen(self.motion_gen_config)

        self.tensor_args = TensorDeviceType(device=torch.device("cuda:0"))
        self.attached_object_names: List[str] = []
        self.dof = self.motion_gen.dof

    def plan(
        self,
        q_start: np.ndarray,
        goal: np.ndarray,
        warm_start_traj: Optional[np.ndarray] = None,
        goal_type: Literal["joint", "pose_wxyz_xyz", "pose"] = "pose_wxyz_xyz",
        retime_to_target_vel: bool = True,
        target_max_vel: float = 1.4,
        batch_size: int = 1,
        robot_configuration: Optional[Any] = None,
        visualize: bool = False,
        verbose: bool = False,
        save_plan_dir: Optional[Path] = None,
        save_usd_dir: Optional[Path] = None,
    ) -> Tuple[List[np.ndarray], Any]:
        save_plan_dir = save_plan_dir if save_plan_dir is not None else self.save_dir
        save_usd_dir = save_usd_dir if save_usd_dir is not None else save_plan_dir

        if not self.warmed_up:
            print("Warming up")
            self.motion_gen.warmup(batch=batch_size if batch_size > 1 else None)
            self.warmed_up = True

        q_start: Union[np.ndarray, JointState] = q_start[
            : self.dof
        ]  # automatically crop using robot config?
        q_start = JointState.from_position(
            self.motion_gen.tensor_args.to_device(np.atleast_2d(q_start))
        )

        if warm_start_traj is not None:
            assert (
                goal_type == "joint"
            ), "Only joint goal type supported for warm start. Need to convert to joint state"
            warm_start_traj = JointState.from_position(
                self.motion_gen.tensor_args.to_device(np.atleast_2d(warm_start_traj))[
                    2:30,
                    : self.dof,
                ].reshape(1, 1, -1, 7),
            )

        if goal_type == "joint":
            q_goal = JointState.from_position(
                self.motion_gen.tensor_args.to_device(np.atleast_2d(goal))
            )
            if verbose:
                print(f"q_start: {q_start.position}")
                print(f"q_goal: {q_goal.position}")
            result = self.motion_gen.plan_single_js(
                q_start,
                q_goal,
                MotionGenPlanConfig(max_attempts=50),
                custom_trajopt_seed_traj=warm_start_traj,
            )

            if result.valid_query and not result.success.any().item():
                # try again with differnt q_goal via IK
                X_ee_goal = self.motion_gen.compute_kinematics(q_goal).ee_pose
                # use motion_gen to do IK
                ik_result = self.motion_gen.solve_ik(
                    X_ee_goal,
                    return_seeds=8,
                )
                q_goals = ik_result.get_batch_unique_solution()[0]
                # loop trhough q_goal to find the one that is closest to the original goal
                for i, q_goal_new in enumerate(q_goals):
                    if ik_result.success[0][i].item():
                        result = self.motion_gen.plan_single_js(
                            q_start, JointState.from_position(q_goal_new.reshape(1, -1))
                        )
                        if result.success.any().item():
                            logging.info(
                                f"Found IK solution {i} that works with trajopt"
                            )
                            break
                        else:
                            logging.info(f"Motion planning on IK solution {i} failed")
            elif not result.valid_query and verbose:
                logging.warning(
                    f"Invalid query (start state likely in collision). "
                    f"Result: {result.status}. q_start: {q_start.position} goal: {goal}"
                )

        elif goal_type == "pose_wxyz_xyz":
            if isinstance(goal, torch.Tensor):
                assert goal.ndim == 1, "Goal for pose_wxyz_xyz should be a 1D tensor"
                goal_pose = Pose(
                    position=goal[4:],
                    quaternion=goal[:4],
                )
            else:
                goal_pose = Pose(
                    position=self.tensor_args.to_device([goal[4:]]),
                    quaternion=self.tensor_args.to_device([goal[:4]]),
                )
            result = self.motion_gen.plan_single(q_start, goal_pose)
            if batch_size == 1:
                result = self.motion_gen.plan_single(q_start, goal_pose)
            else:
                result = self.motion_gen.plan_batch(
                    q_start.repeat_seeds(batch_size),
                    goal_pose.repeat_seeds(batch_size),
                    MotionGenPlanConfig(enable_graph=True, enable_opt=True),
                )
        if result.success.any().item():
            if batch_size > 1:
                plan = result.get_successful_paths()[0]
            else:
                plan = result.get_interpolated_plan()
            # not sure why using optimized plan give even more non-straight paths
            # might be related to trying to optimize for minimum time

            if retime_to_target_vel:
                # retime the result based on target max velocity versus max velocity in the plan
                try:
                    max_vel = torch.max(torch.abs(plan.velocity))
                    time_scale_factor = target_max_vel / max_vel
                    if verbose:
                        logging.info(f"Scaling time by {time_scale_factor}")
                    result.retime_trajectory(time_scale_factor)
                except ValueError as e:
                    logging.error(f"Error in retiming trajectory: {e}")
                    import ipdb

                    ipdb.set_trace()
                    motion_plan = None
                    motion_cost = np.inf
                    return motion_plan, motion_cost

            if save_plan_dir is not None:
                assert os.environ.get("MUJOCO_GL") in ["egl", "osmesa"], (
                    f"Currently, MUJOCO_GL={os.environ.get('MUJOCO_GL')}. "
                    "Need MUJOCO_GL=egl or MUJOCO_GL=osmesa for offscreen rendering."
                )
                save_plan_dir.mkdir(parents=True, exist_ok=True)
                save_plan_path = (
                    save_plan_dir
                    / f"curobo_mp_success_{time.strftime('%Y%m%d%H%M%S')}.mp4"
                )
                video_writer = imageio.get_writer(save_plan_path, fps=20)
                model = self.env.sim.model._model
                data = self.env.sim.data._data
                # TODO: figure out how to not require self.env.sim's model/data to render from cameras
                qpos = data.qpos.copy()
                qvel = data.qvel.copy()
                qacc = data.qacc.copy()
                with mujoco.Renderer(model) as renderer:
                    for i in range(plan.position.shape[0]):
                        # update env's robot joint state
                        self.env.robots[0].set_robot_joint_positions(
                            plan.position[i].cpu().numpy()
                        )
                        video_img = []
                        for cam in range(0, 3):
                            renderer.update_scene(data=data, camera=cam)
                            frame = renderer.render()
                            video_img.append(frame)
                        video_img = np.concatenate(video_img, axis=1)
                        video_writer.append_data(video_img)
                video_writer.close()
                data.qpos[:] = qpos
                data.qvel[:] = qvel
                data.qacc[:] = qacc
                logging.info(f"Saving video to {save_plan_path}")

            plan = result.get_interpolated_plan()
            if visualize:
                assert (
                    robot_configuration is not None
                ), "Need robot configuration to visualize curobo plan"
                # code to visualize the plan
                full_q = plan.position.cpu().numpy()
                full_q_concat = np.concatenate(
                    [full_q, np.zeros((full_q.shape[0], 2))], axis=1
                )
                # update the q_goal
                create_passive_viewer(
                    robot_configuration.model,
                    robot_configuration.data,
                    robot_configuration=robot_configuration,
                    qs=full_q_concat,
                    rate=0.4,
                )
        else:
            logging.warning(
                f"Failed. Result: {result.status}. q_start: {q_start.position} goal: {goal}"
                "Pending fix for q0 joint and q1 joint unnecessarily different. Pose goals tend to give IK failure."
            )
            if result.optimized_plan is not None:
                plan = result.optimized_plan
            else:
                plan = q_start
                if goal_type == "joint":
                    # concatenate q_start and q_goal into a single plan iwith type joint state
                    plan = JointState(
                        position=self.motion_gen.tensor_args.to_device(
                            torch.cat(
                                [
                                    q_start.position.repeat(10, 1),
                                    q_goal.position.repeat(10, 1),
                                ],
                                dim=0,
                            ).reshape(1, -1, q_start.shape[-1])
                        ),
                    )

            if save_plan_dir is not None:
                assert os.environ.get("MUJOCO_GL") in ["egl", "osmesa"], (
                    f"Currently, MUJOCO_GL={os.environ.get('MUJOCO_GL')}. "
                    "Need MUJOCO_GL=egl or MUJOCO_GL=osmesa for offscreen rendering."
                )
                save_plan_dir.mkdir(parents=True, exist_ok=True)
                save_plan_path = (
                    save_plan_dir
                    / f"curobo_mp_valid-query={result.valid_query}_failure_status={str(result.status)}"
                    f"{time.strftime('%Y%m%d%H%M%S')}.mp4"
                )
                video_writer = imageio.get_writer(save_plan_path, fps=20)
                model = self.env.sim.model._model
                data = self.env.sim.data._data
                # TODO: put into helper function?
                # TODO: figure out how to not require self.env.sim's model/data to render from cameras
                qpos = data.qpos.copy()
                qvel = data.qvel.copy()
                qacc = data.qacc.copy()
                with mujoco.Renderer(model) as renderer:
                    if plan is not None:
                        positions = plan.position[0].cpu().numpy()
                    else:
                        positions = [q_start.position.cpu().numpy()]
                    for i, position in enumerate(positions):
                        # update env's robot joint state
                        self.env.robots[0].set_robot_joint_positions(position)
                        video_img = []
                        for cam in range(0, 3):
                            renderer.update_scene(data=data, camera=cam)
                            frame = renderer.render()
                            video_img.append(frame)
                        video_img = np.concatenate(video_img, axis=1)
                        video_writer.append_data(video_img)

                video_writer.close()
                data.qpos[:] = qpos
                data.qvel[:] = qvel
                data.qacc[:] = qacc
                logging.info(f"Saving video to {save_plan_path}")

            if visualize:
                # viz start and goal
                create_passive_viewer(
                    robot_configuration.model,
                    robot_configuration.data,
                    robot_configuration,
                    qs=[q_start.position.cpu().numpy(), q_goal.position.cpu().numpy()],
                )

            if save_usd_dir is not None:
                save_usd_dir.mkdir(parents=True, exist_ok=True)

                from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
                from curobo.util.usd_helper import UsdHelper

                cuda_robot_model = CudaRobotModel(self.motion_gen.robot_cfg.kinematics)
                save_path = (
                    save_usd_dir / f"{time.strftime('%Y%m%d%H%M%S')}-demo.usd"
                ).as_posix()
                try:
                    UsdHelper.write_trajectory_animation(
                        "franka.yml",  # dummy path overridden by kin_model
                        self.world_config,
                        q_start=JointState(plan.position[0][0][None]),
                        q_traj=JointState(plan.position[0]),
                        dt=0.1,
                        save_path=save_path,
                        base_frame="/world_base",
                        kin_model=cuda_robot_model,  # this argument overrides franka.yml
                    )
                except Exception as e:
                    logging.error(f"Error in saving trajectory to USD: {e}")
                    import ipdb

                    ipdb.set_trace()
                # Note: manually commented out lines in write_trajectory_animation to visualize trajectory
                logging.info(f"Saved trajectory to {save_path}")

            return None, result

        if save_usd_dir is not None:
            save_usd_dir.mkdir(parents=True, exist_ok=True)

            from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
            from curobo.util.usd_helper import UsdHelper

            cuda_robot_model = CudaRobotModel(self.motion_gen.robot_cfg.kinematics)
            save_path = (
                save_usd_dir / f"{time.strftime('%Y%m%d%H%M%S')}-demo.usd"
            ).as_posix()
            UsdHelper.write_trajectory_animation(
                "franka.yml",  # dummy path overridden by kin_model
                self.world_config,
                q_start=JointState(plan.position[0][0][None]),
                q_traj=JointState(plan.position[0]),
                dt=0.1,
                save_path=save_path,
                base_frame="/world_base",
                kin_model=cuda_robot_model,  # this argument overrides franka.yml
            )
            # Note: manually commented out lines in write_trajectory_animation to visualize trajectory
            logging.info(f"Saved trajectory to {save_path}")

        # skip first state as it's the start state
        return plan.position.cpu().numpy()[1:], result

    # need to update attach object names to maybe also map to the robot attachment frame
    def update_env(
        self,
        obs: Dict[str, Any],
        env,
        env_type: Literal["mujoco"] = "mujoco",
        attach_object_names: Optional[List[str]] = None,
        batch_size: int = 1,
    ):
        if not self.warmed_up:
            self.motion_gen.warmup(batch=batch_size if batch_size > 1 else None)
            self.warmed_up = True

        # env contains the location of the object in the environment;
        # if we need to attach, we update the robot 'joint-state' and object poses first
        # ideally env is just an observation dict, but for now we assume it's the env object
        if attach_object_names is None:
            attach_object_names = []
        if env_type == "mujoco":
            self.update_env_from_mujoco()  # first update poses of everything
            for attach_object_name in attach_object_names:
                if (
                    attach_object_name not in self.attached_object_names
                ):  # then attach things
                    joint_state = JointState.from_position(
                        self.motion_gen.tensor_args.to_device(
                            np.atleast_2d(obs["robot0_joint_pos"])
                        ),
                    )
                    self.motion_gen.attach_objects_to_robot(
                        joint_state,
                        self.body_to_geom_names[attach_object_name],
                        sphere_fit_type=SphereFitType.SAMPLE_SURFACE,
                        surface_sphere_radius=0.002,
                    )
                    # first modify the world config then attach
                    self.attached_object_names.append(attach_object_name)

            # Detach objects that are not in attach_object_names
            detach_object_names = [
                obj_name
                for obj_name in self.attached_object_names
                if obj_name not in attach_object_names
            ]
            for detach_object_name in detach_object_names:
                joint_state = JointState.from_position(
                    self.motion_gen.tensor_args.to_device(
                        np.atleast_2d(obs["robot0_joint_pos"])
                    ),
                )
                self.motion_gen.detach_object_from_robot()
                # re-enable the objects in the collision world
                for geom_name in self.body_to_geom_names[detach_object_name]:
                    self.motion_gen.world_coll_checker.enable_obstacle(
                        name=geom_name, enable=True
                    )
                # TODO(klin): map from attachment link to object and specify the attachment link to detach things from
                self.attached_object_names.remove(detach_object_name)
        return

    def update_env_from_mujoco(self):
        geom_name_to_pose = get_geom_poses(
            mj_geoms=self.mj_geoms,
            data=self.env.sim.data._data,
            base_pos=self.base_pos,
        )
        update_curobo_world_config(
            motion_gen=self.motion_gen,
            mj_geoms=self.mj_geoms,
            geom_name_to_pose=geom_name_to_pose,
        )
