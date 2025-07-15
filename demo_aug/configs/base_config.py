import logging
import pathlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

import h5py
import numpy as np
import torch
from nerfstudio.utils.rich_utils import CONSOLE
from tyro.extras import from_yaml, to_yaml

from demo_aug.configs.env_configs import AnnotatedEnvConfigUnion
from demo_aug.envs.base_env import MotionPlannerType
from demo_aug.utils.run_script_utils import retry_on_exception


@dataclass
class CartesianSpaceAugConfigs:
    eef_pos_bound: float = 0.1  # Default for end effector position during sampling.
    eef_pos_x_bound: Optional[Tuple[float, float]] = None
    eef_pos_y_bound: Optional[Tuple[float, float]] = None
    eef_pos_z_bound: Optional[Tuple[float, float]] = None
    eef_rot_angle_z_bound: Tuple[float, float] = (
        -90,
        90,  # Bound for end effector rotation angle during sampling.
    )
    eef_rot_angle_y_bound: Tuple[float, float] = (
        -12,
        12,  # Bound for end effector rotation angle during sampling.
    )
    eef_rot_angle_x_bound: Tuple[float, float] = (
        -12,
        12,  # Bound for end effector rotation angle during sampling.
    )
    ee_pos: Optional[Tuple[float, float, float]] = (
        None  # End effector pose around which to sample, if any. If unspecified, samples around constraint pose.
    )
    ee_quat_xyzw: Optional[Tuple[float, float, float, float]] = (
        None  # End effector quaternion around which to sample, if any.
    )

    def __post_init__(self):
        if self.ee_pos is not None:
            self.ee_pos = np.array(self.ee_pos)
        if self.ee_quat_xyzw is not None:
            self.ee_quat_xyzw = np.array(self.ee_quat_xyzw)

        # if any of the pos_x/y/z bounds aren't specified, set them to the default pos bound
        if self.eef_pos_x_bound is None:
            self.eef_pos_x_bound = (-self.eef_pos_bound, self.eef_pos_bound)
        if self.eef_pos_y_bound is None:
            self.eef_pos_y_bound = (-self.eef_pos_bound, self.eef_pos_bound)
        if self.eef_pos_z_bound is None:
            self.eef_pos_z_bound = (-self.eef_pos_bound, self.eef_pos_bound)


@dataclass
class JointSpaceAugConfigs:
    joint_qpos_noise_magnitude: float = 0.1  # Noise magnitude for joint positions.


@dataclass
class StartAugmentationConfig:
    space: Literal["joint", "cartesian"] = (
        "joint"  # Specifies the type of space for start augmentation.
    )

    # Space-specific parameters
    joint_space_aug_configs: JointSpaceAugConfigs = JointSpaceAugConfigs()
    cartesian_space_aug_configs: CartesianSpaceAugConfigs = CartesianSpaceAugConfigs()

    # General parameters
    max_trials: int = 10
    weld_obj_to_ee_for_reach_traj: bool = (
        False  # if True, weld obj to ee for reach traj
    )
    set_gripper_action_close: bool = False  # should always be True if welding?
    end_with_tracking_start_gripper_qpos: bool = False
    only_use_tracking_start_gripper_qpos: bool = False

    # Heuristic only used in drake_motion_planner (kin-trajopt)
    use_collision_free_waypoint_heuristic: bool = True
    collision_free_waypoint_heuristic_max_trials: int = 10
    collision_free_waypoint_threshold: float = 0.1
    collision_free_waypoint_rotation_angle_bound: float = 0.0  # 0.3925
    collision_free_waypoint_sampling_radius: float = 0.0
    start_at_collision_free_waypoint: bool = False
    enforce_reach_traj_gripper_non_close: bool = True


@dataclass
class SE3AugmentationConfig:
    dx_range: Tuple[float, float] = (-0.05, 0.05)
    dy_range: Tuple[float, float] = (-0.05, 0.05)
    dz_range: Tuple[float, float] = (0.0, 0.0)
    dthetaz_range: Tuple[float, float] = (-0.1308, 0.916)
    # dthetaz_range: Tuple[float, float] = (0, 0)

    use_fixed_transf: bool = False
    use_abs_transform: bool = False
    use_robosuite_placement_initializer: bool = False
    use_biased_sampling_z_rot: bool = False  # mainly used if not doing 360 rotations
    # if we want to use absolute transform i.e. transform some source pt/pose to some range, instead of doing pure relative transforms,
    # we also need a pose or a point that we want to transform to some other pose or point in world frame.
    # currently, we're hardcoding the point to be the center point of a task relevant object (e.g. square nut).
    fixed_rot_z_transf: float = 0
    fixed_pos_transf: float = 0

    def __post_init__(self):
        if self.use_robosuite_placement_initializer:
            if not self.use_abs_transform:
                assert "Must use abs transf if using robosuite placement initializer. Please update se3_aug_cfg parameters."


@dataclass
class ScaleAugmentationConfig:
    scale_factor_range: List[float] = field(default_factory=lambda: [0.6, 1.7])
    apply_non_uniform_scaling: bool = False


@dataclass
class ShearAugmentationConfig:
    shear_factor_range: List[float] = field(default_factory=lambda: [-0.4, 0.1])


@dataclass
class WarpAugmentationConfig:
    warp_factor: float = 0.1


@dataclass
class EENoiseAugmentationConfig:
    transf_space: Literal["EEF", "World"] = "EEF"
    pos_x_min: float = -0.01
    pos_x_max: float = 0.01
    pos_y_min: float = -0.01
    pos_y_max: float = 0.01
    pos_z_min: float = -0.01
    pos_z_max: float = 0.01
    rot_bound: float = np.pi / 32

    # +z moves the end effector "up" in its own frame


@dataclass
class AppearanceAugmentationConfig:
    target_objects: List[str] = field(default_factory=list)
    appearance_aug_types: List[
        Literal[
            "brightness",
            "contrast",
            "noise",
            "color_jitter",
            "vignette",
            "blur",
            "sharpen",
        ]
    ] = field(default_factory=list)
    randomize_robot_texture: bool = False


# one of these for each constraint time range
# should roughly contain augmentation related configs ...
@dataclass
class AugmentationConfig:
    apply_start_aug: bool = True
    apply_se3_aug: bool = True
    apply_scale_aug: bool = True
    apply_shear_aug: bool = True
    apply_warp_aug: bool = False
    apply_ee_noise_aug: bool = False
    apply_appearance_aug: bool = False
    apply_camera_pose_aug: bool = False

    # for each aug, we provide some default configs and user can override as desired in
    # query_user_for_constraint_infos()
    start_aug_cfg: StartAugmentationConfig = field(
        default_factory=lambda: StartAugmentationConfig()
    )
    se3_aug_cfg: SE3AugmentationConfig = field(
        default_factory=lambda: SE3AugmentationConfig()
    )
    scale_aug_cfg: ScaleAugmentationConfig = field(
        default_factory=lambda: ScaleAugmentationConfig()
    )
    shear_aug_cfg: ShearAugmentationConfig = field(
        default_factory=lambda: ShearAugmentationConfig()
    )
    warp_aug_cfg: WarpAugmentationConfig = field(
        default_factory=lambda: WarpAugmentationConfig()
    )
    ee_noise_aug_cfg: EENoiseAugmentationConfig = field(
        default_factory=lambda: EENoiseAugmentationConfig()
    )
    appearance_aug_cfg: AppearanceAugmentationConfig = field(
        default_factory=lambda: AppearanceAugmentationConfig()
    )

    # To group into a more comprehensive config:
    # Note: reflect eef reflects eef poses within the constraint timesteps
    reflect_eef_and_select_min_cost: bool = False  # if True, plan to both variants of eef poses and select the min-cost variant
    randomly_reflect_eef: bool = False
    always_reflect_eef: bool = False

    name_to_kp_params: Dict[str, Dict[str, Any]] = field(
        default_factory={
            "panda_left_tip": {
                "src_frame_of_offset_frame": "panda_leftfinger",
                "offset_pos": (0, -0.002, 0.047),
                "offset_quat_wxyz": (1, 0.0, 0.0, 0.0),
            },
            "panda_right_tip": {
                "src_frame_of_offset_frame": "panda_rightfinger",
                "offset_pos": (0, 0.002, 0.047),
                "offset_quat_wxyz": (1, 0.0, 0.0, 0.0),
            },
        }.copy
    )
    # TODO(klin): currently can't pass this task_relev_obj_render_gt_obj_mesh to rendering
    # where do the rendering cfgs belong?
    task_relev_obj_render_gt_obj_mesh: bool = False

    do_traj_tracking: bool = (
        True  # should always be true? under new implementation, start w/ this first
    )
    goal_reaching_max_trials: int = 10
    # only relevant if doing goal reaching and if robot has a gripper
    init_gripper_type: Literal["open", "closed", "non-fixed"] = "non-fixed"
    # init_gripper_open: bool = True  # KL change back
    # only relevant if doing goal reaching and if robot has a gripper
    end_gripper_open: bool = False  # False would be more faithful (and easier to implement ... however trajopt doesn't like it)

    view_meshcat: bool = False
    # TODO(klin): implement collision_free_waypoint_rotation; would be good to start trajectories from random rotations

    def __post_init__(self):
        assert (
            self.end_gripper_open
            or self.start_aug_cfg.use_collision_free_waypoint_heuristic
        ), (
            "end_gripper_open must be true if use_collision_free_waypoint_heuristic is false."
            "Else motion planner usually can't solve if goal is near collision"
        )


@dataclass
class NeRFGenerationConfig:
    # timestamp(s) for images to use for nerf generation
    src_image_data_t: List[int] = field(default_factory=lambda: [-1])


@dataclass
class TaskRelevantObjectsConfig:
    """
    For a given timestep, specifies the configuration for task relevant objects.
    """

    obj_names: List[str] = field(default_factory=list)
    obj_to_nerf_generation_image_ts: Dict[str, Tuple[int]] = field(
        default_factory=dict
    )  # redundant with the below
    obj_to_nerf_generation_cfg: Optional[Dict[str, NeRFGenerationConfig]] = field(
        default_factory=dict
    )
    obj_to_3d_bounding_box: Dict[str, List[List[float]]] = field(default_factory=dict)
    # may need to have a full config object e.g. NeRFGenerationConfig ... simple method/config for now

    # assumption: only ever weld one object to the end effector
    # to reduce number of nerfs and make motion planning collision implementation easier
    weld_obj_to_ee: bool = (
        False  # if True, assume the object is welded for the entire time range
    )
    weld_obj_name: str = "welded_obj_name"
    weld_t_src: Optional[int] = (
        None  # ground truth timestamp at which to weld the object to the end effector
    )

    def __post_init__(self):
        # if there are multiple nerf_generation_image_ts, may need to enforce that we mask out the EEF
        # when generating the nerf
        for obj, nerf_gen_image_ts in self.obj_to_nerf_generation_image_ts.items():
            if len(nerf_gen_image_ts) > 1:
                print(
                    f"Remove this reminder; multiple nerf_gen_image_ts for {obj}remember to mask out EEF when NeRFing!"
                )


# constraint "info" is basically a augmentation config + time range for which these augmentation configs apply
# where do I set the nerf object stuff though?
@dataclass
class ConstraintInfo:
    """
    Specifies the time range, task relevant object configs and augmentation configs for a constraint within a demo.

    From the reconstruction perspective, ConstraintInfo tells us
        (1) which timesteps to generate nerf from and (task relevant object config)
        (2) which objects are task relevant

        (the latter can, e.g., be specified by 2D bounding box)

    From the variant generation perspective, ConstraintInfo tells us:
        (1) what we should augment (e.g. scale, shear, warp, ee noise, se3)
        (2) how we should augment (e.g. scale factor, shear factor, warp factor, ee noise factor, se3 factor)
        (3) what are the task relevant objects

    Let's think about how we'll use ConstraintInfo.

    Note:  ObjectReconstructionsManager should be outside of ConstraintInfo because reconstructions
        in ReconstructionsManager may be needed for multiple constraints.
    """

    constraint_name: str = "constraint_name"

    time_range: Tuple[int, int] = (50, 59)  # [t_start, t_end] (inclusive)
    aug_cfg: AugmentationConfig = field(default_factory=lambda: AugmentationConfig())

    # maybe easier to specify timerange-wide things here?
    weld_obj_name: str = "square_nut"
    weld_t_src: int = 85  # for now, assume weld t is within constraint time range so that we have eef poses
    weld_obj_to_ee: bool = True
    weld_t_range: Tuple[int, int] = (
        105,
        120,
    )  # [t_start, t_end] (inclusive)  # update this range to work for two ranges??
    # need to check if weld_t_range works for the given constraint_info time_range

    t_to_task_relev_objs_cfg: Optional[Dict[int, TaskRelevantObjectsConfig]] = None
    se3_origin_obj_name: str = "square_nut"
    scale_origin_obj_name: str = "square_nut"

    def get_task_relevant_objects_cfg(self, t: int) -> TaskRelevantObjectsConfig:
        """For the input time, return information required to generate the TaskRelevant object
        via TaskRelevantObjectsConfig.

        This class might contain relevant information to retrieve the TaskRelevant object for the given time
        such as the ground truth timestamp to use for the object, the object names, etc.

        from ObjectReconstructionsManager?

        e.g. rec_manager.get_task_relevant_objs(task_relev_obj_name, t)
        """
        return self.t_to_task_relev_objs_cfg[
            t
        ]  # basically gives gt timestep for each object

    def get_task_relevant_obj_names(self, t: int) -> List[str]:
        """Return the task relevant object names for the given time."""
        return self.t_to_task_relev_objs_cfg[t].obj_names

    def get_reconstruction_ts_for_t_and_obj(self, t: int, obj_name: str) -> Tuple[int]:
        """Return the reconstruction timesteps for the given time and object name."""
        return self.t_to_task_relev_objs_cfg[t].obj_to_nerf_generation_image_ts[
            obj_name
        ]

    def collect_reconstruction_timesteps(self) -> List[Tuple[int]]:
        """Collect list of image timesteps used for object/scene reconstruction."""
        reconstruction_timesteps: Set[Tuple[int]] = set()
        for gt_t in range(self.time_range[0], self.time_range[1] + 1):
            obj_to_nerf_generation_image_ts: Dict[str, Tuple[int]] = (
                self.t_to_task_relev_objs_cfg[gt_t].obj_to_nerf_generation_image_ts
            )
            for obj_name in obj_to_nerf_generation_image_ts.keys():
                reconstruction_timesteps.add(
                    tuple(obj_to_nerf_generation_image_ts[obj_name])
                )
        return list(reconstruction_timesteps)

    def collect_reconstruction_timesteps_to_obj_name(self) -> Dict[Tuple[int], str]:
        """Collect list of image timesteps used for object/scene reconstruction."""
        reconstruction_timesteps_to_obj_name: Dict[Tuple[int], str] = {}
        for gt_t in range(self.time_range[0], self.time_range[1] + 1):
            obj_to_nerf_generation_image_ts: Dict[str, Tuple[int]] = (
                self.t_to_task_relev_objs_cfg[gt_t].obj_to_nerf_generation_image_ts
            )
            for obj_name in obj_to_nerf_generation_image_ts.keys():
                reconstruction_timesteps_to_obj_name[
                    tuple(obj_to_nerf_generation_image_ts[obj_name])
                ] = obj_name
        return reconstruction_timesteps_to_obj_name

    # the below does not work
    # field(
    #     default_factory={
    #         t: TaskRelevantObjectsConfig(
    #             gt_t=t,
    #             nerf_generation_t=list([t]),
    #         ) for t in range(time_range[0], time_range[1] + 1)
    #     }.copy
    # )

    # for each timestep in the time range, we need to know how to:
    # i) render novel views for each object: need a nerf for each object:
    #   to generate a nerf, need to know:
    #       a) which images to use [i.e. which timestep(s)] and
    #       b) (for more advanced use cases), which nerfs to blend together
    # ii) avoid collisions with each object: need a collision mesh for each object
    # - obtain collision mesh from the nerf in i) or just do some dumb conservative bounding box thing
    def __post_init__(self):
        # default to having gripper action = close if we're welding and also
        # set gripper qpos to be the same as the tracking start gripper qpos
        if self.weld_obj_to_ee:
            assert (
                self.weld_t_range[1] - self.weld_t_range[0]
                <= self.time_range[1] - self.time_range[0]
            ), "weld_t_range must be within constraint time range."

        if self.aug_cfg.start_aug_cfg.weld_obj_to_ee_for_reach_traj:
            assert (
                self.weld_obj_to_ee
            ), "weld_obj_to_ee must be true if weld_obj_to_ee_for_reach_traj is true."
            assert self.aug_cfg.start_aug_cfg.end_with_tracking_start_gripper_qpos, (
                "Must set gripper qpos goal to be the same as tracking start gripper qpos if welding object to ee for"
                " reach traj is true."
            )
            assert self.aug_cfg.start_aug_cfg.set_gripper_action_close, "Must set gripper action to close for reach traj if welding object to ee for reach traj."
            assert self.aug_cfg.start_aug_cfg.only_use_tracking_start_gripper_qpos, "If welding object to ee for reach traj, only use tracking start gripper qpos for reach traj."

        if self.weld_obj_to_ee:
            assert self.aug_cfg.start_aug_cfg.end_with_tracking_start_gripper_qpos, "Must set gripper to be the same as tracking start gripper qpos if welding object to ee."

        if self.t_to_task_relev_objs_cfg is None:
            self.populate_default_obj_to_nerf_generation_image_ts()

        # assert that all ts in time_range are in t_to_task_relev_objs_cfg
        if self.t_to_task_relev_objs_cfg is not None:
            for t in range(self.time_range[0], self.time_range[1] + 1):
                assert (
                    t in self.t_to_task_relev_objs_cfg
                ), f"t={t} must be in t_to_task_relev_objs_cfg"

    def populate_default_obj_to_nerf_generation_image_ts(self) -> bool:
        """
        If the object does not have any nerf generation times specified, then we use the ground truth
        timestamp collected images for generating the nerf.
        """
        print(self.time_range)
        for t in range(self.time_range[0], self.time_range[1] + 1):
            for obj_name in self.t_to_task_relev_objs_cfg[t].obj_names:
                if (
                    obj_name
                    not in self.t_to_task_relev_objs_cfg[
                        t
                    ].obj_to_nerf_generation_image_ts
                ):
                    self.t_to_task_relev_objs_cfg[t].obj_to_nerf_generation_image_ts[
                        obj_name
                    ] = list([t])
        return True

        # if all(
        #     [
        #         len(self.task_relev_objs_cfg.objs_t_to_annotation_data_t[obj_name]) == 0
        #         for obj_name in self.task_relev_objs_cfg.objs_t_to_annotation_data_t
        #     ]
        # ):
        #     # Set default times for each object
        #     for obj_name in self.task_relev_objs_cfg.obj_names:
        #         print(f"obj_name: {obj_name}")
        #         self.task_relev_objs_cfg.objs_t_to_annotation_data_t[obj_name] = {
        #             t: t for t in range(self.time_range[0], self.time_range[1] + 1)
        #         }


@retry_on_exception(max_retries=5, retry_delay=1, exceptions=(BlockingIOError,))
def save_constraint_infos(
    demo_path, demo_name: str, constraint_infos: List[ConstraintInfo]
) -> None:
    """Update original hdf5 file at 'name' to include constraint_infos.

    Assumes demo_path is an hdf5 file containing key 'data/demo_name' and that demo_name is a group in the hdf5 file.
    """
    assert demo_path.endswith(
        ".hdf5"
    ), f"demo_path must be an hdf5 file, got {demo_path}"
    CONSOLE.log(f"Saving constraint timestep info to {demo_path}")
    with h5py.File(demo_path, "r+") as f:
        # create constraint_infos
        if "constraint_infos" not in f[f"data/{demo_name}"].keys():
            constraint_infos_group = f[f"data/{demo_name}"].create_group(
                "constraint_infos"
            )
        else:
            constraint_infos_group = f[f"data/{demo_name}"]["constraint_infos"]

        for i, constraint_info in enumerate(constraint_infos):
            if f"constraint_info_{i}" in constraint_infos_group.keys():
                # ask user if they want to overwrite
                # overwrite = input(
                #     f"Constraint info {i} already exists for demo {demo_name}. Overwrite? (y/n): "
                # ).lower()
                overwrite = "y"
                if overwrite == "y":
                    del constraint_infos_group[f"constraint_info_{i}"]
                    constraint_infos_group[f"constraint_info_{i}"] = to_yaml(
                        constraint_info
                    )
            else:
                constraint_infos_group[f"constraint_info_{i}"] = to_yaml(
                    constraint_info
                )


@retry_on_exception(max_retries=5, retry_delay=1, exceptions=(BlockingIOError,))
def load_constraint_infos(demo_path: str, demo_name: str) -> List[ConstraintInfo]:
    """Load constraint timestep info from the original hdf5 file at 'name'

    Returns True if successful, False otherwise
    """
    assert demo_path.endswith(
        ".hdf5"
    ), f"demo_path must be an hdf5 file, got {demo_path}"

    constraint_infos: List[ConstraintInfo] = []

    CONSOLE.log(f"Loading constraint timestep info from {demo_path}")
    with h5py.File(demo_path, "r") as f:
        if (
            "demo_name" not in f["data"].keys()
            or "constraint_infos" not in f["data"][demo_name].keys()
        ):
            logging.info(f"No demo group found for {demo_name}")
            return constraint_infos

        # e.g. f['data/demo_0'] then do f['data/demo_0/constraint_infos']['constraint_info_i']
        constraint_infos_group = f[f"data/{demo_name}/constraint_infos"]

        # Iterate over all the timestep range groups and extract the data
        for c_info_name in constraint_infos_group.keys():
            # key name doesn't really matter: values contain yaml-ified constraint info
            constraint_info = from_yaml(
                ConstraintInfo, constraint_infos_group[c_info_name][()]
            )
            constraint_infos.append(constraint_info)

        # Set the constraint timestep info list on the object
    CONSOLE.log(f"Loaded {len(constraint_infos)} constraint infos")
    return constraint_infos


@dataclass
class DemoAugConfig:
    """Configuration for demo augmentation."""

    env_cfg: AnnotatedEnvConfigUnion

    # number of trials to generate for each constraint timestep
    trials_per_constraint: int = 1
    # number of extra action steps to add to the end of the demo
    extra_action_steps: int = 0
    # source demo path
    demo_path: pathlib.Path = pathlib.Path(
        "~/autom/demo-aug/data/nerf_images/3d-trace-rand-free-all_1ep_succ_hold_12_1684117749_444264/demo.hdf5"
    ).expanduser()

    timestep_to_nerf_folder: Dict[int, pathlib.Path] = field(default_factory=dict)

    demo_name: str = "demo_0"

    save_base_dir: Optional[pathlib.Path] = pathlib.Path(
        "~/autom/diffusion_policy/data/nerf_env/3d-trace/augmented/"
    ).expanduser()

    save_file_name: Optional[str] = None

    nerf_generation_freq: float = 0.5  # how often to generate nerf in Hz
    n_obs_history: int = (
        1  # number of observations to use for the policy at each timestep
    )
    # could be OK to have 1 maybe ... not sure actually

    # path to timestep_annotation_data.json
    timestep_annotation_data_path: Optional[pathlib.Path] = None
    aug_cfg: AugmentationConfig = field(default_factory=AugmentationConfig)
    constraint_infos: List[ConstraintInfo] = field(default_factory=list)
    task_relev_obj_render_gt_obj_mesh: bool = False

    view_meshcat: bool = False  # relevant if using drake motion planning
    debug_mode: bool = False
    seed: int = 0

    use_wandb: bool = False
    overwrite_saved_nerfs: bool = False

    # used for storing default values for configs in 'query' user stage
    task_name: Optional[str] = None
    subtask_name: Optional[str] = None
    task_distribution: Optional[
        Literal[
            "narrow",
            "wide",
            "narrow-debug",
            "narrow-debug-w-scale",
            "narrow-debug-w-scale-opp-rot",
            "near-goal",
            "wide-figures",
            "single",
        ]
    ] = None

    method_name: Optional[str] = None  # use for implementing baseline methods

    @retry_on_exception(max_retries=5, retry_delay=1, exceptions=(BlockingIOError,))
    def __post_init__(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # generate file name if not specified
        if self.save_file_name is None:
            self.save_file_name = ""
            self.save_file_name += (
                f"{self.method_name if self.method_name is not None else ''}"
            )
            self.save_file_name += (
                f"{self.task_name if self.task_name is not None else ''}"
            )
            self.save_file_name += (
                f"{'_' + self.subtask_name if self.subtask_name is not None else ''}"
            )
            self.save_file_name += f"{'_' + self.task_distribution if self.task_distribution is not None else ''}"

            # add number of trials
            self.save_file_name += f"_trials{self.trials_per_constraint}"
            if self.aug_cfg.apply_scale_aug:
                self.save_file_name += "_scaug"
                self.save_file_name += f"-scrng{'-'.join(list(map(str, self.aug_cfg.scale_aug_cfg.scale_factor_range)))}"
            if self.aug_cfg.apply_shear_aug:
                self.save_file_name += "_shaug"
                self.save_file_name += f"-shrng{'-'.join(list(map(str, self.aug_cfg.shear_aug_cfg.shear_factor_range)))}"
            if self.aug_cfg.apply_warp_aug:
                self.save_file_name += "_warpaug"
            if self.aug_cfg.apply_ee_noise_aug:
                self.save_file_name += "_eenoiseaug"
                # noise sampling bounds
                self.save_file_name += f"-px{'-'.join(list(map(str, [self.aug_cfg.ee_noise_aug_cfg.pos_x_min, self.aug_cfg.ee_noise_aug_cfg.pos_x_max])))}"
                self.save_file_name += f"-py{'-'.join(list(map(str, [self.aug_cfg.ee_noise_aug_cfg.pos_y_min, self.aug_cfg.ee_noise_aug_cfg.pos_y_max])))}"
                self.save_file_name += f"-pz{'-'.join(list(map(str, [self.aug_cfg.ee_noise_aug_cfg.pos_z_min, self.aug_cfg.ee_noise_aug_cfg.pos_z_max])))}"
            if self.aug_cfg.apply_se3_aug:
                self.save_file_name += "_se3aug"
                # se3 sampling bounds
                self.save_file_name += (
                    f"-dx{'-'.join(list(map(str, self.aug_cfg.se3_aug_cfg.dx_range)))}"
                )
                self.save_file_name += (
                    f"-dy{'-'.join(list(map(str, self.aug_cfg.se3_aug_cfg.dy_range)))}"
                )
                self.save_file_name += (
                    f"-dz{'-'.join(list(map(str, self.aug_cfg.se3_aug_cfg.dz_range)))}"
                )
                self.save_file_name += f"-dthetz{'-'.join(list(map(str, self.aug_cfg.se3_aug_cfg.dthetaz_range)))}"
                self.save_file_name += f"{'-biassampzrot' if self.aug_cfg.se3_aug_cfg.use_biased_sampling_z_rot else ''}"
            if self.aug_cfg.apply_start_aug:
                self.save_file_name += "_staug"
                if self.aug_cfg.start_aug_cfg.space == "joint":
                    self.save_file_name += "_joint"
                    self.save_file_name += f"-jointqposnoise{str(self.aug_cfg.start_aug_cfg.joint_space_aug_configs.joint_qpos_noise_magnitude)}"
                elif self.aug_cfg.start_aug_cfg.space == "cartesian":
                    # eef sampling bounds
                    self.save_file_name += f"-eeposbnd{str(self.aug_cfg.start_aug_cfg.cartesian_space_aug_configs.eef_pos_bound)}"
                    self.save_file_name += (
                        f"-eerotbnd{str(self.aug_cfg.start_aug_cfg.cartesian_space_aug_configs.eef_rot_angle_z_bound)}"
                        f"{str(self.aug_cfg.start_aug_cfg.cartesian_space_aug_configs.eef_rot_angle_y_bound)}"
                        f"{str(self.aug_cfg.start_aug_cfg.cartesian_space_aug_configs.eef_rot_angle_x_bound)}"
                    )
            if (
                self.env_cfg.motion_planner_cfg.motion_planner_type
                == MotionPlannerType.PRM
            ):
                self.save_file_name += f"_envpad{self.env_cfg.motion_planner_cfg.sampling_based_motion_planner_cfg.env_collision_padding}"
                self.save_file_name += f"_envinfldist{self.env_cfg.motion_planner_cfg.sampling_based_motion_planner_cfg.env_influence_distance}"
                self.save_file_name += f"_edgestepsize{self.env_cfg.motion_planner_cfg.sampling_based_motion_planner_cfg.edge_step_size}"
                self.save_file_name += f"_envdistfact{self.env_cfg.motion_planner_cfg.sampling_based_motion_planner_cfg.env_dist_factor}"
            else:
                self.save_file_name += "_defcurobocfg"

            self.save_file_name += (
                f"_{self.env_cfg.motion_planner_cfg.motion_planner_type.name}"
            )

        if self.env_cfg.obs_cfg.use_mujoco_renderer_only:
            self.save_file_name += "_mjcrend"
        self.save_file_name += ".hdf5"

        # remove spaces, "(", ")" from save_dir
        self.save_file_name = self.save_file_name.replace(" ", "")
        self.save_file_name = self.save_file_name.replace("(", "")
        self.save_file_name = self.save_file_name.replace(")", "")
        # replace all commas with underscores
        self.save_file_name = self.save_file_name.replace(",", "_")

        # update env_cfg's robot_cfg's dataset path
        if "robomimic" in str(self.demo_path):
            self.env_cfg.robot_cfg.dataset = self.demo_path

            # load the timestep_to_nerf_folder dict
            with h5py.File(str(self.demo_path), "r") as f:
                assert (
                    "nerf_timestep_paths" in f[f"data/{self.demo_name}"].keys()
                ), "nerf_timestep_paths must be in demo file.This is a dict mapping timestep to nerf_trainable_folder"
                for timestep in f[f"data/{self.demo_name}/nerf_timestep_paths"].keys():
                    self.timestep_to_nerf_folder[timestep] = pathlib.Path(
                        f[f"data/{self.demo_name}/nerf_timestep_paths"][timestep][
                            ()
                        ].decode("utf-8")
                    )

                # TODO(klin): change to 'folder' instead of path in generate image file
            if "square" in str(self.demo_path):
                self.task_name = "square"
            elif "lift" in str(self.demo_path):
                self.task_name = "lift"
            elif "door" in str(self.demo_path):
                self.task_name = "door"
            elif "can" in str(self.demo_path):
                self.task_name = "can"
            elif "tool_hang" in str(self.demo_path):
                self.task_name = "tool_hang"
            elif "wine_glass" in str(self.demo_path):
                self.task_name = "wine_glass"
            else:
                raise ValueError(f"Unknown task name for demo path {self.demo_path}")

        # Hack: ideally, we check for these during the creation of the multi_camera_cfg
        # however, to use tyro's base configs, all the options for multi_camera_cfg must be created first
        multi_camera_cfg = self.env_cfg.multi_camera_cfg
        if (
            multi_camera_cfg.camera_cfgs is not None
            and len(multi_camera_cfg.camera_cfgs) == 0
        ):
            assert (
                multi_camera_cfg.camera_extrinsics_path is not None
            ), "Must specify camera extrinsics path if camera_cfgs is None"
            assert (
                multi_camera_cfg.camera_intrinsics_dir is not None
                or multi_camera_cfg.camera_intrinsics_path is not None
            ), "Must specify camera camera_intrinsics_dir or camera_intrinsics_path if camera_cfgs is None"

            multi_camera_cfg.load_configs_from_files()


@dataclass
class PolicyEvalConfig:
    demo_aug_cfg: DemoAugConfig
    policy_path: pathlib.Path = pathlib.Path("")
    compare_to_ground_truth: bool = False
    ground_truth_src_path: Optional[pathlib.Path] = None
    n_trials: int = 5
    save_dir: Optional[pathlib.Path] = None
    ood_variant: Optional[
        Literal["remove_bg", "shift_bg", "shift_gripper", "object_color", "robot_color"]
    ] = None
    obj_transf_variant: Literal[
        "demo_aug_aug", "fixed_uniform", "use_src_demo", "grid_sweep"
    ] = "demo_aug_aug"
    # demo_aug_aug: use the obj transf as from demo_aug configs
    bg_shift_type: Optional[Literal["fixed", "uniform"]] = None
    bg_shift_x_y: Tuple[float, float] = (0.04, 0.04)  # for fixed bg shift
    bg_shift_x_range: Tuple[float, float] = (-0.1, 0.1)  # for uniform bg shift
    bg_shift_y_range: Tuple[float, float] = (-0.1, 0.1)  # for uniform bg shift
    target_object: Optional[str] = None

    def __post_init__(self):
        if self.save_dir is None:
            self.save_dir = "policy_eval_outputs"

        if self.ood_variant is not None:
            self.save_dir = f"{self.save_dir}_{self.ood_variant}"

        if self.obj_transf_variant == "fixed_uniform":
            self.save_dir = f"{self.save_dir}_obj_transf=uniform"
        elif self.obj_transf_variant == "use_src_demo":
            self.save_dir = f"{self.save_dir}_obj_transf=use_src_demo"

        if self.ood_variant == "shift_bg":
            assert self.bg_shift_type is not None, "Must specify bg_shift_type"
            self.save_dir = f"{self.save_dir}_{self.bg_shift_type}"
            # if bg_shift_type is fixed, must specify bg_shift_x_y
            if self.bg_shift_type == "fixed":
                assert self.bg_shift_x_y is not None, "Must specify bg_shift_x_y"
                self.save_dir = f"{self.save_dir}_bg_shift_fixed{self.bg_shift_x_y[0]}_{self.bg_shift_x_y[1]}"
            elif self.bg_shift_type == "uniform":
                assert (
                    self.bg_shift_x_range is not None
                ), "Must specify bg_shift_x_range"
                assert (
                    self.bg_shift_y_range is not None
                ), "Must specify bg_shift_y_range"
                self.save_dir = (
                    f"{self.save_dir}_bg_shift_range_{self.bg_shift_x_range[0]}_{self.bg_shift_x_range[1]}"
                    f"_{self.bg_shift_y_range[0]}_{self.bg_shift_y_range[1]}"
                )

        if self.ood_variant == "object_color":
            assert self.target_object is not None, "Must specify target_object"
            self.save_dir = f"{self.save_dir}_{self.target_object}"

        # remove spaces, "(", ")" from save_dir
        self.save_dir = pathlib.Path(
            str(self.save_dir).replace(" ", "").replace("(", "").replace(")", "")
        )
