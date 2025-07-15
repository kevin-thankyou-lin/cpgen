import logging
import pathlib
import time
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

# import acta_mp_vamp as vmp
import numpy as np
import torch
import xmltodict
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.sphere_fit import SphereFitType
from curobo.geom.types import Cuboid, Cylinder, Mesh, WorldConfig
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.util.logger import setup_curobo_logger
from curobo.util_file import (
    get_robot_path,
    join_path,
    load_yaml,
)
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
    MotionGenStatus,
)
from numpy.typing import NDArray

import demo_aug
from demo_aug.configs.robot_configs import (
    DRAKE_TO_ROBOMIMIC_GRIPPER_OFFSET_POS,  # TODO: different name
)
from demo_aug.envs.motion_planners.base_motion_planner import BaseMotionPlanner
from demo_aug.envs.motion_planners.motion_planning_space import (
    MotionPlanningSpace,
    ObjectInitializationInfo,
)

setup_curobo_logger("warning")


def _array_from_string(s):
    return np.fromiter(map(float, s.split()), dtype=np.float32)


def _load_geom_mujoco(
    geom: Dict,
    robot_base_xyz: NDArray,
    parent_xyz: NDArray,
    cubes: List[Tuple[int]],
    cylinders: List[Tuple[int]],
) -> None:
    """Updates the cubes and cylinders lists in place with the geometry information from the given mujoco geom dict."""
    # HACK: Utter hack. Reprehensible. Truly gross. Geometries seem to be named with a suffix convention for
    # visual shapes vs. collision shapes. So, we filter out any that end in "_visual"

    if geom.get("@name", None) is not None and geom["@name"].endswith("_visual"):
        print(
            f"Skipping visual geometry {geom['@name']} for curobo motion planner collision env."
        )
        return

    if geom["@type"] == "plane":
        # The third size parameter is unusual: it specifies the spacing between the grid subdivisions
        # of the plane for rendering purposes.
        half_extents = _array_from_string(geom["@size"]).tolist()
        half_extents[2] = 0.01
        position = (
            parent_xyz + _array_from_string(geom["@pos"]) - robot_base_xyz
        ).tolist()
        position[2] -= 0.1  # hack bc planes are supposed to be non geoms
        rotation = [1, 0, 0, 0]
        cubes.append((half_extents, position, rotation))
    elif geom["@type"] == "box":
        half_extents = (_array_from_string(geom["@size"])).tolist()
        if "@quat" in geom:
            rotation = _array_from_string(geom["@quat"])
        else:
            rotation = np.array([1, 0, 0, 0])

        position = (
            parent_xyz + _array_from_string(geom["@pos"]) - robot_base_xyz
        ).tolist()
        cubes.append((half_extents, position, rotation.tolist()))
    elif geom["@type"] == "cylinder":
        # NOTE: Mujoco cylinders seem to all be z-aligned?
        half_extents = _array_from_string(geom["@size"]).tolist()
        position = (
            parent_xyz + _array_from_string(geom["@pos"]) - robot_base_xyz
        ).tolist()
        if "@quat" in geom:
            rotation = _array_from_string(geom["@quat"])
        else:
            rotation = np.array([1, 0, 0, 0])
        cylinders.append((half_extents, position, rotation))
    else:
        print(f"Skipping unsupported geometry type '{geom['@type']}'!")


class CuroboMotionPlanner(BaseMotionPlanner):
    def __init__(
        self,
        planning_space: Optional[MotionPlanningSpace] = None,
        robot_file_path: Optional[pathlib.Path] = None,
        robot_ee_link: Optional[str] = None,
    ) -> None:
        """
        Initializes the Curobo motion planner.

        If planning space is not provided, use default values for the robot and environment.
        """
        # TODO add warmup dummy
        self._planning_space = planning_space
        self._robot_file_path = robot_file_path
        self._robot_ee_link = robot_ee_link
        self.world_config, self.robot_config, self.cubes = self._load_environment(
            self._planning_space, self._robot_file_path, self._robot_ee_link
        )
        # re-generating motion gen config and motion gen works --- maybe need to update the mesh cache?
        self.motion_gen_config = MotionGenConfig.load_from_robot_config(
            self.robot_config,
            self.world_config,
            collision_checker_type=CollisionCheckerType.MESH,  # for mesh collision
            collision_activation_distance=0.02,
            # rotation_threshold=0.0005,
            # position_threshold=0.001,
            num_ik_seeds=32,
            num_trajopt_seeds=4,
            collision_cache={"obb": 10},
            interpolation_dt=1 / 30,
            # doing so doesn't affect/help robot collision geometry after warm up
            # adding more spheres to robot collision doesn't register if warmup
        )

        self.motion_gen = MotionGen(self.motion_gen_config)
        self._has_warmed_up = False
        if planning_space.obj_to_init_info is not None:
            self._attach_object(planning_space.obj_to_init_info)
            # check if any obj_info contains weld_to_ee, if so, don't warm up b/c robot collision isn't updated
            weld_to_ee = any(
                [
                    obj_info.weld_to_ee
                    for obj_info in planning_space.obj_to_init_info.values()
                ]
            )
            if weld_to_ee:
                return
        self.motion_gen.warmup()
        self._has_warmed_up = True

    def update_env(self, planning_space: MotionPlanningSpace) -> None:
        self._planning_space = planning_space
        self.world_config, self.robot_config, self.cubes = self._load_environment(
            self._planning_space,
            self._robot_file_path,
            self._robot_ee_link,
            update_mesh_name=False,
        )

        self.motion_gen.clear_world_cache()
        self.motion_gen.update_world(self.world_config)
        if self._planning_space.obj_to_init_info is not None:
            a = time.time()
            self._attach_object(self._planning_space.obj_to_init_info)
            print(f"Time to attach objects: {time.time() - a}")
            weld_to_ee = any(
                [
                    obj_info.weld_to_ee
                    for obj_info in planning_space.obj_to_init_info.values()
                ]
            )
            if weld_to_ee:
                return

        if not self._has_warmed_up:
            self.motion_gen.warmup()
            self._has_warmed_up = True

    def _attach_object(self, object_info: Dict[str, ObjectInitializationInfo]) -> None:
        # go through obj to init info and attach objects if needed
        for obj_name, obj_info in object_info.items():
            if obj_info.weld_to_ee:
                obs = self.world_config.get_obstacle(obj_name)

                n_spheres = 200
                surface_sphere_radius = 0.002
                sphere_fit_type = SphereFitType.SAMPLE_SURFACE
                voxelize_method: str = "ray"
                link_name = "attached_object"

                X_drake_to_robomimic = np.eye(4)
                X_drake_to_robomimic[:3, 3] = DRAKE_TO_ROBOMIMIC_GRIPPER_OFFSET_POS
                X_drake_to_robomimic = Pose.from_matrix(
                    X_drake_to_robomimic[np.newaxis, ...]
                )

                original_object_pose = torch.tensor(obs.pose).to(
                    self.motion_gen.tensor_args.device
                )
                obs.pose = [
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                ]  # ObjectInitializationInfo assumes pose is 0, 0, 0, 1, 0, 0, 0

                # note: updated franka.yml's attached_object frame to be at DRAKE_TO_ROBOMIMIC_GRIPPER_OFFSET_POS
                transform_pose = Pose.from_matrix(
                    obj_info.X_parentframe_obj[np.newaxis, ...]
                )
                sph = obs.get_bounding_spheres(
                    n_spheres,
                    surface_sphere_radius,
                    pre_transform_pose=transform_pose,
                    tensor_args=self.motion_gen.tensor_args,
                    fit_type=sphere_fit_type,
                    voxelize_method=voxelize_method,
                )

                sph_list = [s.position + [s.radius] for s in sph]

                max_spheres = self.motion_gen.robot_cfg.kinematics.kinematics_config.get_number_of_spheres(
                    link_name
                )
                # remove original collision geometry since replacing with attached object
                try:
                    self.motion_gen.world_coll_checker.enable_obstacle(
                        enable=False, name=obj_name
                    )
                except Exception as e:
                    print(
                        f"Failed to disable original object: {e} TODO: investigate --- appears in"
                        " get_robot_and_object_traj"
                    )
                    # import ipdb; ipdb.set_trace()

                spheres = self.motion_gen.tensor_args.to_device(
                    torch.as_tensor(sph_list)
                )
                sphere_tensor = torch.zeros((max_spheres, 4))
                sphere_tensor[:, 3] = -10.0
                sphere_tensor[: spheres.shape[0], :] = spheres.contiguous()

                self.motion_gen.attach_spheres_to_robot(
                    sphere_tensor=sphere_tensor, link_name="attached_object"
                )
                obs.pose = (
                    original_object_pose.cpu().numpy().tolist()
                )  # convert obs pose back

                print(f"Attached object {obj_name} to robot")

    @staticmethod
    def _adjust_z_position(quat_wxyz: NDArray, pos: NDArray, z_offset: float):
        from scipy.spatial.transform import Rotation as R

        # Extract the position and quaternion from end_cfg
        pos = np.array(pos, dtype=float)
        quat_wxyz = np.array(quat_wxyz, dtype=float)
        # Create a rotation object from the quaternion (expects quat in scalar-last (x, y, z, w) format)
        rotation = R.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        # Define the z-offset in the end_cfg frame
        z_offset_vector = np.array([0, 0, z_offset])
        # Rotate the z-offset vector to align with the end_cfg's orientation
        z_offset_vector_rotated = rotation.apply(z_offset_vector)
        # Update the position by adding the rotated z-offset vector
        new_pos = pos + z_offset_vector_rotated
        # Return the updated configuration
        return np.concatenate([quat_wxyz, new_pos])

    def get_optimal_trajectory(
        self,
        start_cfg: NDArray,
        end_cfg: NDArray,
        goal_type: Literal["pose", "joint"] = "joint",
        gripper_dim: int = 6,
        *,
        max_iterations: Optional[int] = 3,
        visualize: bool = False,
        trajectory_vizualization_save_path: str = "demo.usd",
        retime_to_target_vel: bool = True,
        target_max_vel: float = 1.8,
    ) -> Optional[Tuple[List[NDArray], float]]:
        start_cfg = start_cfg[..., :-gripper_dim]
        start_state = JointState.from_position(
            self.motion_gen.tensor_args.to_device(np.atleast_2d(start_cfg))
        )
        goal_cfg = end_cfg[..., :-gripper_dim].copy()

        if visualize:
            from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
            from curobo.util.usd_helper import UsdHelper

            cuda_robot_model = CudaRobotModel(self.motion_gen.robot_cfg.kinematics)
            UsdHelper.write_trajectory_animation(
                "franka.yml",  # dummy path overridden by kin_model
                self.world_config,
                start_cfg,
                start_state,
                dt=0.1,
                save_path=trajectory_vizualization_save_path,
                base_frame="/world_base",
                kin_model=cuda_robot_model,  # this argument overrides franka.yml
            )
            # Note: manually commented out lines in write_trajectory_animation to visualize trajectory
            print("Saved trajectory to demo.usd")
            exit()

        if goal_type == "pose":
            # shift pose to be w.r.t. robot base
            adjust_z = (
                True if self._robot_ee_link == "panda_hand" else False
            )  # hardcode for now; to put vals into configs
            if adjust_z:
                goal_cfg[4:7] -= self._planning_space.robot_base_pos
                goal_cfg = CuroboMotionPlanner._adjust_z_position(
                    goal_cfg[:4], goal_cfg[4:7], -0.097
                )  # needed because curobo franka doesn't match robomimic frame
            goal_pose = Pose(
                self.motion_gen.tensor_args.to_device(np.atleast_2d(goal_cfg[4:7])),
                self.motion_gen.tensor_args.to_device(np.atleast_2d(goal_cfg[:4])),
            )

            # check if IK is possible
            start = time.time()
            result = self.motion_gen.plan_single(
                start_state, goal_pose, MotionGenPlanConfig(max_attempts=20)
            )
            print(f"Time to find single motion plan {time.time() - start}")

            if not result.success.item():
                print(f"result.success: {result.success}")
                print(f"result.status: {result.status}")
                if result.status == MotionGenStatus.FINETUNE_TRAJOPT_FAIL:
                    result.success = True
                    if result.motion_time.shape == torch.Size([]):
                        motion_cost = result.motion_time.item()
                        interpolated_plan = result.optimized_plan
                    else:
                        motion_cost = result.motion_time[-1].item()
                        interpolated_plan = result.optimized_plan[-1]

                    if retime_to_target_vel:
                        # retime the result based on target max velocity versus max velocity in the plan
                        try:
                            max_vel = torch.max(torch.abs(interpolated_plan.velocity))
                            time_scale_factor = target_max_vel / max_vel
                            result.retime_trajectory(time_scale_factor)
                        except ValueError as e:
                            print(f"Error in retiming trajectory: {e}")
                            # You might want to handle this error appropriately,
                            # such as setting a default time scale or skipping the retiming
                            # might want to investigate why retime_trajectory is failing
                            # File "/scr/thankyou/autom/demo-aug/demo_aug/envs/motion_planners/curobo_planner.py", line 352, in get_optimal_trajectory
                            # result.retime_trajectory(time_scale_factor)
                            # File "/scr/thankyou/autom/demo-aug/demo_aug/envs/motion_planners/__init__.py", line 37, in retime_trajectory
                            motion_plan = None
                            motion_cost = np.inf
                            return motion_plan, motion_cost
                    motion_plan = [
                        interpolated_plan[i].position.cpu().numpy()
                        for i in range(len(interpolated_plan))
                    ]
                    (
                        print("Position error: ", result.position_error),
                        print("rotation error [-1]: ", result.rotation_error),
                    )
                elif result.status == MotionGenStatus.IK_FAIL:
                    print(
                        "Failed to find a plan due to IK failure! Increasing num_ik_seeds for MotionGenConfig doesn't"
                        " seems helpful."
                    )
                    logging.info("Failed to find a curobo plan due to IK failure!")

                    motion_plan = None
                    motion_cost = np.inf
                else:
                    print(f"Failed to find a plan! {result.status}")
                    logging.info(f"Failed to find a curobo plan! {result.status}")
                    motion_plan = None
                    motion_cost = np.inf
                return motion_plan, motion_cost
        else:
            raise NotImplementedError(
                "Only pose goals are correctly supported for now."
                "Joint space planning in latest curobo 05/06/24 commit gives bad result that misses goal state by a lot"
            )
            goal_joint_state = JointState.from_position(
                self.motion_gen.tensor_args.to_device(np.atleast_2d(goal_cfg))
            )
            result = self.motion_gen.plan_single_js(
                start_state, goal_joint_state, MotionGenPlanConfig(max_attempts=50)
            )
            print(f"result.success: {result.success}")
            if not result.success.item():
                print("Failed to find a plan!")
                print(result.status)

        if result.success.item():
            interpolated_plan = result.get_interpolated_plan()
            if retime_to_target_vel:
                max_vel = torch.max(torch.abs(interpolated_plan.velocity))
                time_scale_factor = target_max_vel / max_vel
                result.retime_trajectory(time_scale_factor)
                interpolated_plan = result.get_interpolated_plan()

            if visualize:
                from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
                from curobo.util.usd_helper import UsdHelper

                interpolated_plan_viz = interpolated_plan.clone()
                # repeat each position for nicer replay ivz
                n_repeats = 16
                interpolated_plan_viz.position = torch.repeat_interleave(
                    interpolated_plan_viz.position, n_repeats, dim=0
                )
                cuda_robot_model = CudaRobotModel(self.motion_gen.robot_cfg.kinematics)
                UsdHelper.write_trajectory_animation(
                    "franka.yml",  # dummy path overridden by kin_model
                    self.world_config,
                    start_cfg,
                    # start_state, # start_state to debug b/c interpolated plan is None if failed
                    interpolated_plan_viz,
                    dt=result.interpolation_dt,
                    save_path=trajectory_vizualization_save_path,
                    base_frame="/world_base",
                    kin_model=cuda_robot_model,  # this argument overrides franka.yml
                )
                # Note: manually commented out lines in write_trajectory_animation to visualize trajectory
                print("Saved trajectory to demo.usd")

            # Extract and return the list of joint angles
            motion_cost = result.motion_time.item()
            motion_plan = [
                interpolated_plan[i].position.cpu().numpy()
                for i in range(len(interpolated_plan))
            ]

            replace_last_with_ik = False
            if replace_last_with_ik:
                # use ik solution instead of the last entry of the plan b/c last entry in plan always seems off
                regularization_joint_config = torch.tensor(
                    motion_plan[-1],
                    device=self.motion_gen.tensor_args.device,
                    dtype=torch.float32,
                )
                last_qpos = self.get_robot_trajectory(
                    end_cfg[np.newaxis, ..., :-gripper_dim],
                    regularization_joint_config=regularization_joint_config.unsqueeze(
                        0
                    ),
                )[0]
                motion_plan[-1] = last_qpos
            return motion_plan, motion_cost
        else:
            print("Failed to find a plan!")
            return None, np.inf

    def get_robot_trajectory(
        self,
        eef_quat_wxyz_pos: NDArray,  # shape: (num_poses, 7)
        regularization_joint_config: Optional[np.ndarray] = None,
    ) -> List[np.ndarray]:
        """Given a sequence of eef poses, return a sequence of joint angles that tracks that trajectory."""
        self._no_obstacles_ik_config = IKSolverConfig.load_from_robot_config(
            self.robot_config,
            None,
            rotation_threshold=0.005,
            position_threshold=0.0005,
            num_seeds=20,
            self_collision_check=True,
            self_collision_opt=False,
            tensor_args=self.motion_gen.tensor_args,
            use_cuda_graph=True,
        )
        self._no_obstacles_ik_solver = IKSolver(self._no_obstacles_ik_config)
        # first pre-process all eef poses to be in the right space for curobo

        # shift pose to be w.r.t. robot base
        eef_quat_wxyz_pos[..., 4:7] -= self._planning_space.robot_base_pos
        eef_quat_wxyzs_pos = eef_quat_wxyz_pos
        # this should I think we handled in urdf?? urdf should have a frame name??
        adjust_z = (
            True if self._robot_ee_link == "panda_hand" else False
        )  # hardcode for now; to put vals into configs
        if adjust_z:
            eef_quat_wxyzs_pos = np.array(
                [
                    CuroboMotionPlanner._adjust_z_position(
                        ee_quat_wxyz_pos[:4], ee_quat_wxyz_pos[4:7], -0.097
                    )
                    for ee_quat_wxyz_pos in eef_quat_wxyz_pos
                ]
            )
            # one way is to add obs frame vs action frame ... not sure if that's a thing w/ curobo's yaml?
        goal_pose = Pose(
            self.motion_gen.tensor_args.to_device(
                np.atleast_2d(eef_quat_wxyzs_pos[..., 4:7])
            ),
            self.motion_gen.tensor_args.to_device(
                np.atleast_2d(eef_quat_wxyzs_pos[..., :4])
            ),
        )
        if regularization_joint_config is not None:
            regularization_joint_config = torch.tensor(
                regularization_joint_config,
                device=self.motion_gen.tensor_args.device,
                dtype=self.motion_gen.tensor_args.dtype,
            )
        result = self._no_obstacles_ik_solver.solve_batch(
            goal_pose, retract_config=regularization_joint_config
        )
        if not result.success.all():
            print(f"Failed to find an IK plan for goal_pose: {goal_pose}!")
            logging.info(f"Failed to find an IK plan for goal_pose: {goal_pose}!")
            return None

        joint_angles = result.solution.cpu().numpy()
        joint_angles = joint_angles.squeeze(1)
        return joint_angles.tolist()

    @staticmethod
    def _create_curubo_world_config(
        cuboids: List[Tuple[float]],
        cylinders: List[Tuple[float]],
        robot_base_xyz: NDArray,
        meshes: Optional[Dict[str, ObjectInitializationInfo]] = None,
        update_mesh_name: bool = False,
    ) -> WorldConfig:
        meshs = []
        capsules = []
        spheres = []
        cylinders = [
            Cylinder(
                name=f"cylinder_{i+1}",
                radius=cylinder[0][0],
                height=cylinder[0][1],
                pose=cylinder[1] + cylinder[2],
                color=[0, 1.0, 0, 1.0],
            )
            for i, cylinder in enumerate(cylinders)
        ]
        cuboids = [
            Cuboid(
                name=f"cuboid_{i+1}",
                pose=cuboid[1] + cuboid[2],
                dims=[
                    dim * 2 for dim in cuboid[0]
                ],  # because isaac expects full extents for cubes
                color=[0.8, np.random.rand(), 0, 1],
            )
            for i, cuboid in enumerate(cuboids)
        ]
        if meshes is not None:
            for obj_name, obj_info in meshes.items():
                # if update_mesh_name:
                #     # add random prefix to mesh name to avoid name collision
                #     obj_name = f"{np.random.randint(10000)}_{obj_name}"
                pose = np.array([0.0, 0, 0, 1, 0, 0, 0], dtype=robot_base_xyz.dtype)
                pose[:3] -= robot_base_xyz
                new_mesh = Mesh(
                    name=obj_name,
                    file_path=obj_info.mesh_paths.obj_path,
                    pose=pose.tolist(),
                    color=[np.random.rand(), np.random.rand(), 1, 1.0],
                )
                new_mesh.file_path = (
                    obj_info.mesh_paths.obj_path
                )  # because curobo's Mesh always appends get_assets_path()
                # convert to absolute path new_mesh.file_path
                new_mesh.file_path = str(
                    Path(demo_aug.__file__).parent.parent / new_mesh.file_path
                )
                print(f"Loading mesh {obj_name} from {new_mesh.file_path}")

                # if update_mesh_name:
                #     try:
                #         new_mesh.file_path = "hello.obj"
                #     except Exception as e:
                #         print(e)
                #         import ipdb; ipdb.set_trace()

                meshs.append(new_mesh)

        # TODO: Klin: try from dict?
        return WorldConfig(
            mesh=meshs,
            cuboid=cuboids,
            cylinder=cylinders,
            capsule=capsules,
            sphere=spheres,
        )

    @staticmethod
    def save_shapes_to_yaml(converted_shapes: Dict[str, List[int]], filename: str):
        """
        Saves the converted shapes dictionary to a YAML file.

        Args:
        - converted_shapes (dict): The dictionary containing the shapes information.
        - filename (str): The name of the file to save the YAML data to.
        """
        import yaml

        with open(filename, "w") as file:
            # Use dump() to convert the dictionary to a YAML formatted string and write it to the file
            yaml.dump(converted_shapes, file, default_flow_style=False, sort_keys=False)

    @staticmethod
    def format_shape(
        shape: Tuple[List[int]], index: int, shape_type: Literal["cuboid", "cylinder"]
    ) -> Dict:
        """Formats a single shape to the desired output structure."""
        half_extents, position, rotation = shape
        # Handle the conversion from half_extents to full dimensions for cylinders and cubes (isaac expects full extents)
        dims = (
            [2 * extent for extent in half_extents]
            if shape_type == "cuboid"
            else [2 * half_extents[0], 2 * half_extents[0], half_extents[2]]
        )
        name = f"{shape_type}{index}"
        shape_dict = {
            name: {
                "dims": dims,
                "pose": list(position) + list(rotation),
                "color": [0.0, 1.0, 0.0, 1.0],
            }
        }
        return shape_dict

    @staticmethod
    def _convert_shapes(cubes: List[Tuple[int]], cylinders: List[Tuple[int]]):
        """Converts a list of shapes into the specified format used by curobo."""
        output = {"cuboid": [], "cylinder": []}
        cube_count, cylinder_count = 1, 1
        for shape in cubes:
            shape_type = "cuboid"
            cube_count += 1
            formatted_shape = CuroboMotionPlanner.format_shape(
                shape, cube_count, shape_type
            )
            output[shape_type].append(formatted_shape)
        for shape in cylinders:
            shape_type = "cylinder"
            cylinder_count += 1
            formatted_shape = CuroboMotionPlanner.format_shape(
                shape, cube_count, shape_type
            )
            output[shape_type].append(formatted_shape)
        return output

    def _load_environment(
        self,
        planning_space: MotionPlanningSpace,
        robot_file: Optional[pathlib.Path] = None,
        robot_ee_link: Optional[str] = None,
        update_mesh_name: bool = True,
    ) -> Tuple[WorldConfig, RobotConfig]:
        # load the xml environment file and return
        environment_path = planning_space.environment_xml_url
        if "package://" in str(environment_path):
            # HACK: Add more principled package path handling
            # NOTE: Double parent because it seems like the package.xml lives in models/, but the environment XML path is package://models/$something....
            package_root = Path(planning_space.package_root_path).parent.parent
            environment_path = package_root / environment_path[len("package://") :]

        with open(environment_path) as environment:
            arena = xmltodict.parse(environment.read())

        base_xyz = planning_space.robot_base_pos
        if base_xyz is None:
            base_xyz = np.zeros(3)

        base_quat = planning_space.robot_base_quat_wxyz
        if base_quat is None:
            base_quat = np.array([1.0, 0.0, 0.0, 0.0])

        # HACK: Assumes some structure, but simple parsing for now
        # HACK: Assumes no bodies have rotated frames, for now - easy to add
        cubes: List[Tuple[int]] = []
        cylinders: List[Tuple[int]] = []
        worldbody = arena["mujoco"]["worldbody"]
        for geom in worldbody["geom"]:
            _load_geom_mujoco(geom, base_xyz, np.zeros(3), cubes, cylinders)

        if worldbody.get("body", None) is not None:
            bodies = worldbody["body"]
            if not isinstance(bodies, list):
                # xmltodict quirk when there's a single element of a type:
                bodies = [bodies]

            for body in bodies:
                body_xyz = _array_from_string(body["@pos"])
                for geom in body["geom"]:
                    _load_geom_mujoco(geom, base_xyz, body_xyz, cubes, cylinders)

        # https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html
        # there is a guy] in mujoco -- it's a plane --- check if half extent or sometihng else etc think curbo auto updates to 2x??
        # TODO(klin): check if there's a cuboid added here?
        # nicer to pass mesh list for consistency, but would need to remember how mesh data is stored
        world_cfg = CuroboMotionPlanner._create_curubo_world_config(
            cubes,
            cylinders,
            base_xyz,
            meshes=planning_space.obj_to_init_info,
            update_mesh_name=update_mesh_name,
        )

        robot_file = (
            join_path(get_robot_path(), "franka.yml")
            if robot_file is None
            else robot_file
        )
        robot_cfg = load_yaml(str(robot_file))["robot_cfg"]

        if robot_ee_link is not None:
            # hardcode for now, convert ee_link to panda_link8; should pass this file or loaded config from env or something
            robot_cfg["kinematics"]["ee_link"] = robot_ee_link

        return world_cfg, robot_cfg, cubes

    def visualize_plan(self, plan: List[List[float]]):
        import acta_mp_vamp as vmp

        pb = vmp.PyBulletSimulator(
            str(Path.cwd() / "resources" / "panda_spherized.urdf"), True
        )

        pb.set_joint_positions([0, 0, 0, 0, 0, 0, 0, 0, 0, 0.065, 0.065])

        for cube in self.cubes:
            pb.add_cuboid(*cube)

        while True:
            for q in plan:
                pb.set_joint_positions(q)
                time.sleep(0.1)
