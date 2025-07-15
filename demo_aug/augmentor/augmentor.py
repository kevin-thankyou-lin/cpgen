import logging
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
from nerfstudio.utils.rich_utils import CONSOLE
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from demo_aug.aug_datastructures import (
    TimestepData,
)
from demo_aug.configs.base_config import (
    AppearanceAugmentationConfig,
    ConstraintInfo,
    DemoAugConfig,
)
from demo_aug.demo import Demo
from demo_aug.envs.motion_planners.base_motion_planner import RobotEnvConfig
from demo_aug.envs.motion_planners.motion_planning_space import ObjectInitializationInfo
from demo_aug.envs.nerf_robomimic_env import NeRFRobomimicEnv
from demo_aug.envs.sphere_nerf_env import EnvConfig
from demo_aug.objects.nerf_object import (
    ColorAugmentationWrapper,
    GSplatObject,
    MeshObject,
    NeRFObject,
    TransformationWrapper,
    TransformType,
)
from demo_aug.objects.reconstructor import ReconstructionType
from demo_aug.utils.mathutils import (
    mat2axisangle,
    multiply_with_X_transf,
    random_z_rotation,
)

OBJ_TYPE: Literal["nerf", "gsplat"] = "gsplat"


class Augmentor:
    _env: Optional[NeRFRobomimicEnv] = None

    @staticmethod
    def generate_augmented_demos(
        src_demo: Demo,
        demo_aug_cfg: DemoAugConfig,
        env_cfg: EnvConfig,
        constraints: List[ConstraintInfo],
    ):
        """For the constraints from a source demo, generate `demo's that satisfy the constraints.

        Note: this generated `demo' end when it satisfies the constraint.

        I guess we have the following:

        i) Demo: a list of observation and actions, a.k.a. List[TimestepData]
        ii) Annotated Demo: a Demo and the (user-supplied) set of constraints that the demo should satisfy.
        iii) (Augmented) Demo: same object as Demo, except it was generated from the original Annotated Demo
            and satisfies the constraints.
        """
        augmented_demos: List[Demo] = []
        for constraint_info in tqdm(
            constraints, desc="Generating augmentations for all constraints"
        ):
            augmented_demos.extend(
                Augmentor.generate_augmented_demos_from_single_constraint(
                    src_demo, demo_aug_cfg, env_cfg, constraint_info
                )
            )
        return augmented_demos

    @staticmethod
    def generate_augmented_demos_from_single_constraint(
        src_demo: Demo, demo_aug_cfg: DemoAugConfig, env_cfg, constraint: ConstraintInfo
    ):
        """For the constraints from a source demo, generate `demo's that satisfy the constraints.

        Note: this generated `demo' end when it satisfies the constraint.

        I guess we have the following:

        i) Demo: a list of observation and actions, a.k.a. List[TimestepData]
        ii) Annotated Demo: a Demo and the (user-supplied) set of constraints that the demo should satisfy.
        iii) (Augmented) Demo: same object as Demo, except it was generated from the original Annotated Demo
            and satisfies the constraints.
        """
        augmented_demos: List[Demo] = []
        for _ in tqdm(
            range(demo_aug_cfg.trials_per_constraint),
            desc="Generating augmentations for a specific constraint",
        ):
            # TODO(klin): extract env type from dataset (path)
            augmented_demo: Demo = (
                Augmentor.generate_demo_from_constraint_info_robomimic(
                    src_demo,
                    constraint,
                    demo_aug_cfg.env_cfg,
                )
            )
            if (
                augmented_demo is None
                or augmented_demo.name is None
                or len(augmented_demo.timestep_data) == 0
            ):
                logging.info("Failed to generate an augmentation, skipping ...")
                continue

            augmented_demos.append(augmented_demo)

        return augmented_demos

    @staticmethod
    def apply_augs(
        env,
        X_ee_lst,
        q_gripper_lst,
        c_objs_lst,  # how to generally specify the task relevant keypoints: need to enable frame offset definitions etc
        c_objs_mesh_lst,
        # and to let user specify the keypoints
        obj_to_X_se3: Optional[Dict[str, np.ndarray]] = None,
        obj_to_X_se3_origin: Optional[Dict[str, np.ndarray]] = None,
        obj_to_X_scale: Optional[Dict[str, np.ndarray]] = None,
        obj_to_X_scale_origin: Optional[Dict[str, np.ndarray]] = None,
        obj_to_X_shear: Optional[Dict[str, np.ndarray]] = None,
        obj_to_X_shear_origin: Optional[Dict[str, np.ndarray]] = None,
        obj_to_X_warp: Optional[Dict[str, np.ndarray]] = None,
        X_ee_noise: Optional[np.ndarray] = None,
        optimize_X_ee: bool = False,
        name_to_kp_params: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
        max_gripper_width: float = 0.08 - 0.001,
        obj_for_kp_transforms: Optional[str] = None,
    ) -> Tuple[Optional[List], Optional[List], Optional[List]]:
        """
        Apply augmentations to the environment.

        Args:
            obj_for_kp_transforms: the object whose transformations we use to transform the keypoints
        """

        def transform_objs(
            objs: Dict[str, Union[NeRFObject, MeshObject]],
            obj_to_X_transform: Dict[str, np.ndarray],
            obj_to_X_origin: Dict[str, np.ndarray],
            transform_type: TransformType,
            obj_to_transform_params: Dict[str, Dict[str, np.ndarray]],
            obj_type: Literal["mesh", "nerf", "gsplat"],
        ) -> Dict[str, Union[NeRFObject, MeshObject]]:
            assert obj_type in [
                "mesh",
                "nerf",
                "gsplat",
            ], f"obj_type must be mesh or nerf, got {obj_type}"

            def select_lambda(
                obj_type: str, obj_name: str
            ) -> Callable[[np.ndarray], np.ndarray]:
                X_transform_t = torch.tensor(
                    obj_to_X_transform[obj_name],
                    dtype=torch.float32,
                    device=torch.device("cuda"),
                )
                X_origin_t = torch.tensor(
                    obj_to_X_origin[obj_name],
                    dtype=torch.float32,
                    device=torch.device("cuda"),
                )
                if obj_type == "nerf":
                    return lambda x: multiply_with_X_transf(
                        X_origin_t,
                        multiply_with_X_transf(
                            torch.linalg.inv(X_transform_t),
                            multiply_with_X_transf(torch.linalg.inv(X_origin_t), x),
                        ),
                    )
                else:
                    return lambda x: multiply_with_X_transf(
                        X_origin_t,
                        multiply_with_X_transf(
                            X_transform_t,
                            multiply_with_X_transf(torch.linalg.inv(X_origin_t), x),
                        ),
                    )

            return {
                name: TransformationWrapper(
                    obj,
                    select_lambda(obj_type, name),
                    transform_type=transform_type,
                    transform_params=obj_to_transform_params[name],
                    transform_name=f"{transform_type.value.lower()}_aug",
                )
                for name, obj in objs.items()
            }

        X_ee_new_lst = []
        q_gripper_new_lst = []
        c_objs_new_lst = []
        c_objs_mesh_new_lst = []

        kpts_new_lst: List[Dict[str, np.ndarray]] = []  # should there be more to this?
        # validate transforms are kinematically feasible for the gripper
        # this transforms the directly actuated parts of the interation e.g. gripper or object welded to gripper
        compute_kpt_transforms = (
            obj_to_X_scale is not None
            or obj_to_X_shear is not None
            or obj_to_X_warp is not None
        )
        if compute_kpt_transforms:
            # NUNOCS-style representation would be good for knife object contact because it can do a smarter
            # weighted average keypoint importance transfer operation.
            # TODO(klin): extract the following from args
            both_gripper_positions_matter = True
            if not both_gripper_positions_matter:
                # if only one positions matters, presumably always kinematically feasible
                # if just one point traversal needed
                # bar collision constraints.
                # may need to handle individual tip scaling separately ...
                raise NotImplementedError(
                    "Only one gripper position matters not implemented yet"
                )

            # 0.08 is the max gripper width
            for i in range(len(X_ee_lst)):
                curr_X_ee = X_ee_lst[i]
                curr_q_gripper = q_gripper_lst[i]
                # name of keypoints is the frame name of that keypoint
                # get from arg frame_to_params (key point to params)
                # perhaps ignore keypoints on objects for now
                kpts: Dict[str, np.ndarray] = (
                    env.get_kpts_from_ee_gripper_objs_and_params(
                        curr_X_ee, curr_q_gripper, name_to_kp_params
                    )
                )
                kpts_homog_new = {
                    k: np.concatenate((v, np.ones(1))) for k, v in kpts.items()
                }
                for k, v in kpts_homog_new.items():
                    # assuming X_*_origin is also not None is X_* is not None
                    if obj_to_X_scale is not None:
                        # KL check if things work w/ rotation and scaling that
                        kpts_homog_new[k] = obj_to_X_scale_origin[
                            obj_for_kp_transforms
                        ] @ (
                            obj_to_X_scale[obj_for_kp_transforms]
                            @ (
                                np.linalg.inv(
                                    obj_to_X_scale_origin[obj_for_kp_transforms]
                                )
                                @ kpts_homog_new[k]
                            )
                        )
                    if obj_to_X_shear is not None:
                        kpts_homog_new[k] = obj_to_X_shear_origin[
                            obj_for_kp_transforms
                        ] @ (
                            obj_to_X_shear[obj_for_kp_transforms]
                            @ (
                                np.linalg.inv(
                                    obj_to_X_shear_origin[obj_for_kp_transforms]
                                )
                                @ kpts_homog_new[k]
                            )
                        )
                    if obj_to_X_warp is not None:
                        kpts_homog_new[k] = np.concatenate(
                            (obj_to_X_warp(kpts_homog_new[k]), np.ones(1))
                        )  # only apply warp to the position
                    if obj_to_X_se3 is not None:
                        kpts_homog_new[k] = (
                            obj_to_X_se3[obj_for_kp_transforms] @ kpts_homog_new[k]
                        )

                    if X_ee_noise is not None:
                        # ideal: apply noise w.r.t some valid set of grasps (if it's a grasp we're noising over)
                        # applying a constant transform w.r.t each new gripper frame (post IK) requires a small refactor
                        # of this function
                        # currently applying noise in absolute coordinates
                        kpts_homog_new[k] = X_ee_noise @ kpts_homog_new[k]

                kpts_new = {k: v[:3] for k, v in kpts_homog_new.items()}
                kpts_new_lst.append(kpts_new)
                # solve for gripper length
                gripper_tip_dist = np.linalg.norm(
                    kpts_new["panda_left_tip"] - kpts_new["panda_right_tip"]
                )
                if gripper_tip_dist > max_gripper_width:
                    return (
                        None,
                        None,
                        None,
                        None,
                    )  # kinematically infeasible transformed gripper width

        # this gives us a warm start for the IK problem assuming we used keypoints and need IK
        # pass in keyframe cfgs? from which contains drake relevant things?
        for i in range(len(X_ee_lst)):
            X_ee_new = X_ee_lst[i]
            q_gripper_new = q_gripper_lst[i]

            c_objs_new = c_objs_lst[i]
            c_objs_mesh_new = c_objs_mesh_lst[i]
            if obj_to_X_scale is not None:
                X_ee_new = obj_to_X_scale_origin[obj_for_kp_transforms] @ (
                    obj_to_X_scale[obj_for_kp_transforms]
                    @ (
                        np.linalg.inv(obj_to_X_scale_origin[obj_for_kp_transforms])
                        @ X_ee_new
                    )
                )
                X_ee_new[:3, :3] = X_ee_lst[i][
                    :3, :3
                ]  # keep the original rotation because scaling doesn't affect orientation, I believe
                obj_to_transf_params: Dict[str, Dict[str, np.ndarray]] = {}
                for obj_name in c_objs_new.keys():
                    obj_to_transf_params[obj_name] = {
                        "X_scale": obj_to_X_scale[obj_name],
                        "X_scale_origin": obj_to_X_scale_origin[obj_name],
                    }

                c_objs_new: Dict[str, TransformationWrapper] = transform_objs(
                    c_objs_new,
                    obj_to_X_scale,
                    obj_to_X_scale_origin,
                    TransformType.SCALE,
                    obj_to_transf_params,
                    obj_type=OBJ_TYPE,
                )
                c_objs_mesh_new: Dict[str, TransformationWrapper] = transform_objs(
                    c_objs_mesh_new,
                    obj_to_X_scale,
                    obj_to_X_scale_origin,
                    TransformType.SCALE,
                    obj_to_transf_params,
                    obj_type="mesh",
                )

            if obj_to_X_shear is not None:
                X_ee_new = obj_to_X_shear_origin[obj_for_kp_transforms] @ (
                    obj_to_X_shear[obj_for_kp_transforms]
                    @ (
                        np.linalg.inv(obj_to_X_shear_origin[obj_for_kp_transforms])
                        @ X_ee_new
                    )
                )
                X_ee_new[:3, :3] = X_ee_lst[i][
                    :3, :3
                ]  # keep the original rotation even though shearing does affect orientation; use for initialization

                obj_to_transf_params: Dict[str, Dict[str, np.ndarray]] = {}
                for obj_name in c_objs_new.keys():
                    obj_to_transf_params[obj_name] = {
                        "X_shear": obj_to_X_scale[obj_name],
                        "X_shear_origin": obj_to_X_scale_origin[obj_name],
                    }

                c_objs_new: Dict[str, TransformationWrapper] = transform_objs(
                    c_objs_new,
                    obj_to_X_shear,
                    obj_to_X_shear_origin,
                    TransformType.SHEAR,
                    obj_to_transf_params,
                    obj_type=OBJ_TYPE,
                )
                c_objs_mesh_new: Dict[str, TransformationWrapper] = transform_objs(
                    c_objs_mesh_new,
                    obj_to_X_shear,
                    obj_to_X_shear_origin,
                    TransformType.SHEAR,
                    obj_to_transf_params,
                    obj_type="mesh",
                )

            if obj_to_X_warp is not None:
                X_warp = None
                X_ee_new = X_warp @ X_ee_new
                X_warp_origin = np.eye(4)  # note this is hardcoded!
                X_ee_new = X_warp_origin @ (
                    X_warp @ (np.linalg.inv(X_warp_origin) @ X_ee_new)
                )

                # currrently not correctly implemented as warp implies non-affine transformation
                # c_objs_new = [TransformationWrapper(c_obj, lambda x: X_warp @ x) for c_obj in c_objs_new]
                raise NotImplementedError("Need to implement/test warp for meshes")
            if obj_to_X_se3 is not None:
                X_ee_new = obj_to_X_se3_origin[obj_for_kp_transforms] @ (
                    obj_to_X_se3[obj_for_kp_transforms]
                    @ (
                        np.linalg.inv(obj_to_X_se3_origin[obj_for_kp_transforms])
                        @ X_ee_new
                    )
                )

                obj_to_transf_params: Dict[str, Dict[str, np.ndarray]] = {}
                for obj_name in c_objs_new.keys():
                    obj_to_transf_params[obj_name] = {
                        "X_SE3": obj_to_X_se3[obj_name],
                        "X_SE3_origin": obj_to_X_se3_origin[obj_name],
                    }

                c_objs_new: Dict[str, TransformationWrapper] = transform_objs(
                    c_objs_new,
                    obj_to_X_se3,
                    obj_to_X_se3_origin,
                    TransformType.SE3,
                    obj_to_transf_params,
                    obj_type=OBJ_TYPE,
                )
                c_objs_mesh_new: Dict[str, TransformationWrapper] = transform_objs(
                    c_objs_mesh_new,
                    obj_to_X_se3,
                    obj_to_X_se3_origin,
                    TransformType.SE3,
                    obj_to_transf_params,
                    obj_type="mesh",
                )

            if X_ee_noise is not None:
                X_ee_new = X_ee_noise @ X_ee_new

            if compute_kpt_transforms:
                kpts_new = kpts_new_lst[i]
                # need user to tell us where we begin caring about q_gripper
                # currently not catching when eef ik fails ...
                # check why it fails on first round where gripper tip dist: 0.0771

                # OK looks like the thing's X gets shifted up by ~ 0.044m i.e the height of the gripper tip!
                # maybe we're optimizing the wrong frame?
                # aug'ed EE new adds on an extra 0.045! I see herein lies the bug I think
                X_ee_new, q_gripper_new = env.get_eef_ik(
                    X_W_ee_init=X_ee_new,
                    q_gripper_init=q_gripper_new,
                    name_to_frame_info=name_to_kp_params,
                    kp_to_P_goal=kpts_new,
                    check_collisions=False,  # TODO(klin): unclear why check_collisions needs to be False 02/11/2024
                )  # shouldn't be anything else in the env?
                # alternatively, pass in the transformed keypoint locations so we've a goal set?

            # ideally this returns the correct dimensionality. HACK to do so for now:
            if env.robot_obj.gripper_qpos_ndim == 6:
                # convert it back to 1 b/c technically it should just be 1 ---
                # for drake and robomimic, it's 6
                # maybe we should have drake be the one to convert to 6 dim
                q_gripper_new = q_gripper_new[:1]

            X_ee_new_lst.append(X_ee_new)
            q_gripper_new_lst.append(q_gripper_new)
            c_objs_new_lst.append(c_objs_new)
            c_objs_mesh_new_lst.append(c_objs_mesh_new)

        return X_ee_new_lst, q_gripper_new_lst, c_objs_new_lst, c_objs_mesh_new_lst

    def generate_demo_from_constraint_info_robomimic(
        src_demo: Demo, constraint_info: ConstraintInfo, env_cfg: EnvConfig
    ) -> Demo:
        # currently, env_cfg contains the correct object too
        # eventually want parts of env config to come from the loaded thing

        constraint_range = constraint_info.time_range
        orig_ee_goal_pos = src_demo.get_robot_env_cfg(
            constraint_range[0], constraint_range[1]
        )[0].robot_ee_pos

        # update the env cfg with the constraint info
        env_cfg.motion_planner_cfg.view_meshcat = constraint_info.aug_cfg.view_meshcat

        env = NeRFRobomimicEnv(env_cfg)
        if Augmentor._env is None:
            env = Augmentor._env = NeRFRobomimicEnv(env_cfg)
        else:
            env = Augmentor._env

        # Get pose of the object we use as the origin for the se3 transform
        obs_dct = src_demo.get_obs_for_range(constraint_range[0], constraint_range[1])[
            0
        ]
        if constraint_info.se3_origin_obj_name == "square_nut":
            obj_prefix = "SquareNut"
        elif constraint_info.se3_origin_obj_name == "red_cube":
            obj_prefix = "cube"
        elif constraint_info.se3_origin_obj_name == "can":
            obj_prefix = "Can"
        else:
            print(
                f"Unknown se3_origin_obj_name object name: {constraint_info.se3_origin_obj_name}"
            )
            obj_prefix = "wine_glass"

        if obs_dct.get(f"{obj_prefix}_pos") is None:
            print(
                "Task relevant object doesn't seem to exist: double check this. Should exist if using sim based demo."
            )
            task_relev_objs_meshes = src_demo.get_task_relev_objs_for_range(
                constraint_range[0],
                constraint_range[1],
                constraint_info,
                ReconstructionType.Mesh,
            )[0]
            obj_pos_at_constraint = task_relev_objs_meshes[
                constraint_info.se3_origin_obj_name
            ].get_center()
            obj_quat_xyzw_at_constraint = np.array([0, 0, 0, 1])
        else:
            # assuming this object's pose exists in the demo, if not, handle if by using the mesh's center
            obj_pos_at_constraint = obs_dct[f"{obj_prefix}_pos"]
            obj_quat_xyzw_at_constraint = obs_dct[f"{obj_prefix}_quat"]

        num_trials = constraint_info.aug_cfg.goal_reaching_max_trials

        tic = time.time()
        start_aug_succeeded: bool = False
        for trial_idx in range(num_trials):
            min_cost_reflect_eef_value = np.inf
            min_cost_overall_traj = None
            min_cost_reach_traj_steps = np.inf

            ########################
            # Sample augmentations #
            ########################
            ROBOMIMIC_TABLE_Z_HEIGHT = 0.8
            task_relev_obj_names = constraint_info.t_to_task_relev_objs_cfg[
                constraint_info.time_range[0]
            ].obj_names
            if constraint_info.aug_cfg.apply_scale_aug:
                X_scale, X_scale_origin = NeRFRobomimicEnv.sample_scale_transform(
                    constraint_info.aug_cfg.scale_aug_cfg.scale_factor_range,
                    np.eye(4),
                    apply_non_uniform_scaling=constraint_info.aug_cfg.scale_aug_cfg.apply_non_uniform_scaling,
                )

                obj_to_X_scale_origin: Dict[str, np.ndarray] = {}
                obj_to_X_scale: Dict[str, np.ndarray] = {}
                for obj_name in task_relev_obj_names:
                    obj_to_X_scale_origin[obj_name] = np.eye(4)
                    X_scale_origin = X_scale_origin.copy()
                    if (
                        obj_name == "square_peg"
                        or constraint_info.scale_origin_obj_name == "square_peg"
                    ):
                        # if there's a square peg at all, always set the scale origin to be around the square peg
                        X_scale_origin[:3, 3] = np.array(
                            [0.23, 0.1, ROBOMIMIC_TABLE_Z_HEIGHT]
                        )
                    elif obj_name == "square_nut" or obj_name == "red_cube":
                        X_scale_origin[:3, 3] = np.array(
                            [
                                obj_pos_at_constraint[0],
                                obj_pos_at_constraint[1],
                                ROBOMIMIC_TABLE_Z_HEIGHT,
                            ]
                        )
                        X_scale_origin[:3, :3] = R.from_quat(
                            obj_quat_xyzw_at_constraint
                        ).as_matrix()
                    elif obj_name == "can":
                        ROBOMIMIC_TABLE_Z_HEIGHT = 0.82
                        X_scale_origin[:3, 3] = np.array(
                            [
                                obj_pos_at_constraint[0],
                                obj_pos_at_constraint[1],
                                ROBOMIMIC_TABLE_Z_HEIGHT,
                            ]
                        )
                        X_scale_origin[:3, :3] = R.from_quat(
                            obj_quat_xyzw_at_constraint
                        ).as_matrix()
                    elif obj_name == "wine_glass":
                        Z_HEIGHT = 0
                        X_scale_origin[:3, 3] = np.array(
                            [
                                obj_pos_at_constraint[0],
                                obj_pos_at_constraint[1],
                                Z_HEIGHT,
                            ]
                        )
                        X_scale_origin[:3, :3] = R.from_quat(
                            obj_quat_xyzw_at_constraint
                        ).as_matrix()
                    elif obj_name == "background":
                        X_scale_origin = np.eye(4)
                        X_scale = np.eye(4)
                    else:
                        raise NotImplementedError(
                            "Need to handle scale origin for other objects"
                        )

                    obj_to_X_scale_origin[obj_name] = X_scale_origin
                    obj_to_X_scale[obj_name] = X_scale
            else:
                obj_to_X_scale, obj_to_X_scale_origin = None, None

            if constraint_info.aug_cfg.apply_shear_aug:
                obj_to_X_shear_origin: Dict[str, np.ndarray] = {}
                obj_to_X_shear: Dict[str, np.ndarray] = {}

                X_shear_origin = np.eye(4)
                X_shear_origin[:3, 3] = np.array([0, 0, ROBOMIMIC_TABLE_Z_HEIGHT])
                # user can override default values; below is overriden value
                X_shear, X_shear_origin = NeRFRobomimicEnv.sample_shear_transform(
                    constraint_info.aug_cfg.shear_aug_cfg.shear_factor_range,
                )

                for obj_name in task_relev_obj_names:
                    obj_to_X_shear_origin[obj_name] = np.eye(4)
                    obj_to_X_shear_origin[obj_name][:3, 3] = np.array(
                        [
                            obj_pos_at_constraint[0],
                            obj_pos_at_constraint[1],
                            ROBOMIMIC_TABLE_Z_HEIGHT,
                        ]
                    )
                    obj_to_X_shear_origin[obj_name][:3, :3] = R.from_quat(
                        obj_quat_xyzw_at_constraint
                    ).as_matrix()

                    obj_to_X_shear[obj_name] = X_shear
            else:
                obj_to_X_shear, obj_to_X_shear_origin = None, None

            if constraint_info.aug_cfg.apply_warp_aug:
                obj_to_X_warp_origin: Dict[str, np.ndarray] = {}
                obj_to_X_warp: Dict[str, np.ndarray] = {}

                # user can override default values; below is overriden value
                X_warp = NeRFRobomimicEnv.sample_warp_transform(
                    constraint_info.aug_cfg.warp_aug_cfg.warp_factor,
                )

                for obj_name in task_relev_obj_names:
                    obj_to_X_warp_origin[obj_name] = np.eye(4)
                    obj_to_X_warp_origin[obj_name][:3, 3] = np.array(
                        [
                            obj_pos_at_constraint[0],
                            obj_pos_at_constraint[1],
                            ROBOMIMIC_TABLE_Z_HEIGHT,
                        ]
                    )
                    obj_to_X_warp_origin[obj_name][:3, :3] = R.from_quat(
                        obj_quat_xyzw_at_constraint
                    ).as_matrix()

                    obj_to_X_warp[obj_name] = X_warp
            else:
                # TODO(klin): not implemented
                obj_to_X_warp = None

            if constraint_info.aug_cfg.apply_ee_noise_aug:
                X_ee_noise = NeRFRobomimicEnv.sample_ee_noise_transform(
                    constraint_info.aug_cfg.ee_noise_aug_cfg.pos_x_min,
                    constraint_info.aug_cfg.ee_noise_aug_cfg.pos_x_max,
                    constraint_info.aug_cfg.ee_noise_aug_cfg.pos_y_min,
                    constraint_info.aug_cfg.ee_noise_aug_cfg.pos_y_max,
                    constraint_info.aug_cfg.ee_noise_aug_cfg.pos_z_min,
                    constraint_info.aug_cfg.ee_noise_aug_cfg.pos_z_max,
                    constraint_info.aug_cfg.ee_noise_aug_cfg.rot_bound,
                )
            else:
                X_ee_noise = np.eye(4)
                # looks like x is moving the gripper 'forward' w.r.t to the square nut
                # z is moving the gripper 'up' or 'down' w.r.t to the square nut

            # by default, things should refer to the nerf? so test_nerf_robomimic has inconsistencies
            if constraint_info.aug_cfg.apply_se3_aug:
                X_se3_origin = np.eye(4)

                X_W_obj_curr = np.eye(4)
                X_W_obj_curr[:3, 3] = obj_pos_at_constraint
                X_W_obj_curr[:3, :3] = R.from_quat(
                    obj_quat_xyzw_at_constraint
                ).as_matrix()

                if constraint_info.aug_cfg.se3_aug_cfg.use_robosuite_placement_initializer:
                    # if we have the object's pose, we can use it to initialize the object in robomimic
                    X_se3 = env.sample_robosuite_placement_initializer(
                        # constraint_info.aug_cfg.se3_aug_cfg.dx_range,
                        # constraint_info.aug_cfg.se3_aug_cfg.dy_range,
                        z_range=constraint_info.aug_cfg.se3_aug_cfg.dz_range,
                        obj_name=constraint_info.se3_origin_obj_name,
                    )
                else:
                    # maybe get some 3D point somehwere?
                    X_se3 = env.sample_task_relev_obj_se3_transform(
                        constraint_info.aug_cfg.se3_aug_cfg.dx_range,
                        constraint_info.aug_cfg.se3_aug_cfg.dy_range,
                        constraint_info.aug_cfg.se3_aug_cfg.dz_range,
                        constraint_info.aug_cfg.se3_aug_cfg.dthetaz_range,
                        use_biased_sampling_z_rot=constraint_info.aug_cfg.se3_aug_cfg.use_biased_sampling_z_rot,
                    )

                # putting everything into the X_se3 matrix by something like X_orig @ X_se3 @ X_orig.inv()
                if (
                    constraint_info.aug_cfg.se3_aug_cfg.use_abs_transform
                    and X_W_obj_curr is not None
                ):
                    X_W_obj_target = X_se3
                    # transform obj_curr to go to some target location
                    # Simple case of transforming point: P_new = X_transform @ P_old
                    X_se3 = X_W_obj_target @ np.linalg.inv(X_W_obj_curr)
                    X_se3_origin = np.eye(4)

                pos_transf, rot_transf = X_se3[:3, 3], X_se3[:3, :3]
                # for debugging
                if constraint_info.aug_cfg.se3_aug_cfg.use_fixed_transf:
                    pos_transf = np.array(
                        constraint_info.aug_cfg.se3_aug_cfg.fixed_pos_transf
                    )
                    # note this rot_transf only works because original object and nerf have similar x-y origin
                    rot_transf = random_z_rotation(np.pi / 6, np.pi / 6)
                    rot_transf = random_z_rotation(-np.pi / 5, -np.pi / 5)
                    rot_transf = random_z_rotation(np.pi / 3, np.pi / 3)
                    rot_transf = random_z_rotation(
                        constraint_info.aug_cfg.se3_aug_cfg.fixed_rot_z_transf,
                        constraint_info.aug_cfg.se3_aug_cfg.fixed_rot_z_transf,
                    )
                    X_se3[:3, :3] = rot_transf
                    X_se3[:3, 3] = pos_transf
            else:
                pos_transf, rot_transf = np.zeros(3), np.eye(3)
                X_se3 = np.eye(4)
                X_se3_origin = np.eye(4)

            obj_to_X_se3 = {
                obj_name: X_se3
                if obj_name not in ["square_peg", "background"]
                else np.eye(4)
                for obj_name in task_relev_obj_names
            }
            obj_to_X_se3_origin = {
                obj_name: X_se3_origin
                if obj_name != "square_peg" or obj_name != "background"
                else np.eye(4)
                for obj_name in task_relev_obj_names
            }

            if constraint_info.aug_cfg.reflect_eef_and_select_min_cost:
                reflect_eef_idxs = [0, 1]
            elif constraint_info.aug_cfg.randomly_reflect_eef:
                reflect_eef_idxs = [np.random.choice([0, 1])]
            elif constraint_info.aug_cfg.always_reflect_eef:
                reflect_eef_idxs = [1]
            else:
                reflect_eef_idxs = [0]

            for reflect_eef_idx in reflect_eef_idxs:
                #########################################################
                # Apply reward equivariant object-and-action transforms #
                #########################################################
                overall_traj: Dict[str, torch.Tensor] = defaultdict(list)

                robot_key_prefix = (
                    "robot0_"
                    if "robot0_joint_pos" in src_demo.get_obs_for_range(0, 1)[0]
                    or "robot0_joint_qpos" in src_demo.get_obs_for_range(0, 1)[0]
                    else ""
                )
                X_ee_lst = [
                    np.block(
                        [
                            [
                                R.from_quat(
                                    src_demo.get_obs_for_range(i, i + 1)[0][
                                        f"{robot_key_prefix}eef_quat"
                                    ]
                                ).as_matrix(),
                                np.expand_dims(
                                    src_demo.get_obs_for_range(i, i + 1)[0][
                                        f"{robot_key_prefix}eef_pos"
                                    ],
                                    axis=1,
                                ),
                            ],
                            [np.array([0, 0, 0, 1])],
                        ]
                    )
                    for i in range(constraint_range[0], constraint_range[1])
                ]
                q_gripper_lst = [
                    src_demo.get_obs_for_range(i, i + 1)[0][
                        f"{robot_key_prefix}gripper_qpos"
                    ]
                    for i in range(constraint_range[0], constraint_range[1])
                ]
                quat_xyzw_ee = src_demo.get_obs_for_range(
                    constraint_info.weld_t_src, constraint_info.weld_t_src + 1
                )[0][f"{robot_key_prefix}eef_quat"]
                pos_ee = src_demo.get_obs_for_range(
                    constraint_info.weld_t_src, constraint_info.weld_t_src + 1
                )[0][f"{robot_key_prefix}eef_pos"]

                weld_t_src = constraint_info.weld_t_src
                X_ee_weld_t_src = (
                    np.block(
                        [
                            [
                                R.from_quat(
                                    src_demo.get_obs_for_range(
                                        weld_t_src, weld_t_src + 1
                                    )[0][f"{robot_key_prefix}eef_quat"]
                                ).as_matrix(),
                                np.expand_dims(
                                    src_demo.get_obs_for_range(
                                        weld_t_src, weld_t_src + 1
                                    )[0][f"{robot_key_prefix}eef_pos"],
                                    axis=1,
                                ),
                            ],
                            [np.array([0, 0, 0, 1])],
                        ]
                    )
                    if weld_t_src is not None
                    else None
                )

                q_gripper_weld_t_src = (
                    src_demo.get_obs_for_range(weld_t_src, weld_t_src + 1)[0][
                        f"{robot_key_prefix}gripper_qpos"
                    ]
                    if weld_t_src is not None
                    else None
                )

                if reflect_eef_idx == 1:
                    # TODO(klin): reflect eef might be better as an env method since it's pretty specific
                    # TODO: klin fix this hardcoded hand type
                    X_ee_lst, q_gripper_lst = Augmentor.reflect_eef(
                        env,
                        X_ee_lst,
                        q_gripper_lst,
                        end_effector_type="panda_hand",
                        name_to_kp_params=constraint_info.aug_cfg.name_to_kp_params,
                    )
                    [X_ee_weld_t_src], [q_gripper_weld_t_src] = Augmentor.reflect_eef(
                        env,
                        [X_ee_weld_t_src],
                        [q_gripper_weld_t_src],
                        end_effector_type="panda_hand",
                        name_to_kp_params=constraint_info.aug_cfg.name_to_kp_params,
                    )

                # X_W_OBJ = mujoco_state!
                # new welding that lets us just specify X_EE_NERF to use for the welding transform

                c_objs_lst: List[
                    Dict[str, Union[NeRFObject, GSplatObject, MeshObject]]
                ] = src_demo.get_task_relev_objs_for_range(
                    constraint_range[0],
                    constraint_range[1],
                    constraint_info,
                    ReconstructionType.NeRF
                    if OBJ_TYPE == "nerf"
                    else ReconstructionType.GaussianSplat,
                )

                # add the color augs here
                if constraint_info.aug_cfg.apply_appearance_aug:
                    c_objs_lst = Augmentor.apply_appearance_augs(
                        c_objs_lst, constraint_info.aug_cfg.appearance_aug_cfg
                    )

                # TODO(klin): should contain both mesh path (i.e. collision geometries) and nerf (rendering)
                # this is the core API required from constraint_info --- to get the task relevant objects;
                # right now, we're passing things through via names; this is OK for now (but doesn't work if two object
                # have same semantic class name)
                # c_objs_mesh_lst = src_demo.get_task_relev_objs_mesh_for_range(
                #     constraint_range[0], constraint_range[1], constraint_info
                # )
                c_objs_mesh_lst = src_demo.get_task_relev_objs_for_range(
                    constraint_range[0],
                    constraint_range[1],
                    constraint_info,
                    ReconstructionType.Mesh,
                )

                # TODO: find a way to specify scale origin to be different for each object; code in apply_augs
                # is a bit hardcoded, unfortunately
                # test this:
                X_ee_lst, q_gripper_lst, c_objs_lst, c_objs_mesh_lst = (
                    Augmentor.apply_augs(
                        env,
                        X_ee_lst,
                        q_gripper_lst,
                        c_objs_lst,
                        c_objs_mesh_lst,
                        None,
                        None,
                        obj_to_X_scale,
                        obj_to_X_scale_origin,
                        None,
                        None,
                        None,
                        None,
                        name_to_kp_params=constraint_info.aug_cfg.name_to_kp_params,
                        obj_for_kp_transforms=constraint_info.scale_origin_obj_name,
                    )
                )

                if c_objs_lst is None:
                    print(f"Failed to apply scale augs {obj_to_X_scale}. Skipping.")
                    break

                [X_ee_weld_t_src], [q_gripper_weld_t_src], _, _ = Augmentor.apply_augs(
                    env,
                    [X_ee_weld_t_src],
                    [q_gripper_weld_t_src],
                    c_objs_lst,
                    c_objs_mesh_lst,
                    None,
                    None,
                    obj_to_X_scale,
                    obj_to_X_scale_origin,
                    None,
                    None,
                    None,
                    None,
                    name_to_kp_params=constraint_info.aug_cfg.name_to_kp_params,
                    obj_for_kp_transforms=constraint_info.scale_origin_obj_name,
                )

                quat_xyzw_ee = R.from_matrix(X_ee_weld_t_src[:3, :3]).as_quat()
                pos_ee = X_ee_weld_t_src[:3, 3]

                X_W_EE = np.eye(4)
                X_W_EE[:3, :3] = R.from_quat(quat_xyzw_ee).as_matrix()
                X_W_EE[:3, 3] = pos_ee  # weld object to the ee at weld_t_src
                X_W_NERF = np.eye(4)
                X_EE_NERF = np.linalg.inv(X_W_EE) @ X_W_NERF

                c_objs_new_lst, c_objs_mesh_new_lst = Augmentor.apply_weld_transf(
                    c_objs_lst,
                    X_ee_lst,
                    c_objs_mesh_lst=c_objs_mesh_lst,
                    weld_src_idx=None,
                    weld_target_idx_range=(
                        constraint_info.weld_t_range[0] - constraint_info.time_range[0],
                        constraint_info.weld_t_range[1] - constraint_info.time_range[0],
                    ),
                    weld_obj_name=constraint_info.weld_obj_name,
                    weld_obj_to_ee=constraint_info.weld_obj_to_ee,
                    X_EE_NERF=X_EE_NERF,
                )

                X_ee_new_lst, q_gripper_new_lst, c_objs_new_lst, c_objs_mesh_new_lst = (
                    Augmentor.apply_augs(
                        env,
                        X_ee_lst,
                        q_gripper_lst,
                        # c_objs_lst,
                        c_objs_new_lst,
                        c_objs_mesh_new_lst,
                        obj_to_X_se3,
                        obj_to_X_se3_origin,
                        None,
                        None,
                        obj_to_X_shear,
                        obj_to_X_shear_origin,
                        obj_to_X_warp,
                        X_ee_noise,
                        name_to_kp_params=constraint_info.aug_cfg.name_to_kp_params,
                        obj_for_kp_transforms=constraint_info.scale_origin_obj_name,
                    )
                )

                # supposing obj is welded to EEF at first timestep here
                # can assume X_W_NERF = I; then X_EE_NERF = X_W_EE^-1 @ X_W_NERF = X_W_EE^-1
                X_W_OBJMESH = np.eye(4)
                X_W_EE = X_ee_new_lst[0]
                X_EE_OBJMESH = np.linalg.inv(X_W_EE) @ X_W_OBJMESH

                if X_ee_new_lst is None:
                    import ipdb

                    ipdb.set_trace()
                    logging.info(
                        "Augmentation resulted in kinematic infeasibility; trying again ..."
                    )
                    continue

                # can be in two ways
                robot_env_ts_to_te_cfgs = [
                    RobotEnvConfig(
                        robot_gripper_qpos=q_gripper_new_lst[
                            i
                        ],  # not sure why previously was taking abs
                        robot_ee_pos=X_ee_new_lst[i][:3, 3],
                        robot_ee_quat_wxyz=np.array(
                            [
                                R.from_matrix(X_ee_new_lst[i][:3, :3]).as_quat()[-1],
                                *R.from_matrix(X_ee_new_lst[i][:3, :3]).as_quat()[:-1],
                            ]
                        ),
                        robot_base_pos=env.robot_obj.base_pos,
                        robot_base_quat_wxyz=env.robot_obj.base_quat_wxyz,
                        # TODO(klin): retrieve these values perhaps from c_objs_new_lst: save info about transforms
                        # and also need to pass in new mesh paths
                        task_relev_obj_pos=np.array([0, 0, 0]),
                        task_relev_obj_rot=np.eye(3),
                        task_relev_obj_pos_nerf=np.array([0, 0, 0]),
                        task_relev_obj_rot_nerf=np.eye(3),
                        task_irrelev_obj_pos=None,
                        task_irrelev_obj_rot=None,
                    )
                    for i in range(0, len(q_gripper_new_lst))
                ]

                #  mainly just does IK it doesn't apply augs
                # it assumes augs are applied to goal_cfg and orig_future_cfg_list already
                # to use it, need to correctly populate goal_cfg and orig_future_cfg_list
                # kind of weird but extends to drakemotion planner too, so kkeep for now

                # hardcoding ee_obs_frame as the parent frame
                # because the observations of the end effector pose are in this frame
                obj_to_init_info: Dict[str, ObjectInitializationInfo] = {
                    obj_name: ObjectInitializationInfo(
                        parent_frame_name=(
                            "ee_obs_frame"
                            if obj_name == constraint_info.weld_obj_name
                            else None
                        ),
                        weld_to_ee=(
                            constraint_info.aug_cfg.start_aug_cfg.weld_obj_to_ee_for_reach_traj
                            if obj_name == constraint_info.weld_obj_name
                            else False
                        ),
                        X_parentframe_obj=(
                            X_EE_OBJMESH
                            if constraint_info.aug_cfg.start_aug_cfg.weld_obj_to_ee_for_reach_traj
                            and obj_name == constraint_info.weld_obj_name
                            # if "weld_to_ee" in c_objs_mesh_new_lst[0][obj_name].transform_name_seq
                            else np.eye(4)
                        ),
                        mesh_paths=c_objs_mesh_new_lst[0][obj_name].mesh_paths,
                    )
                    for obj_name in c_objs_mesh_new_lst[0].keys()
                }
                # TODO(klin): update this thing properly; what is get_robot_and_obj_trajectory?
                # it's when things aren't welded ...
                overall_traj = env.get_robot_and_obj_trajectory(
                    robot_env_ts_to_te_cfgs[0],
                    robot_env_ts_to_te_cfgs[1:],
                    check_collisions=False,  # hmm think this should still be false sometimes ... but not always
                    obj_to_init_info=obj_to_init_info,
                )

                # TO this point, we're following the actions from the original tracking trajectory.
                if overall_traj is None:
                    # import ipdb

                    # ipdb.set_trace()
                    logging.info(
                        "Augmentation resulted in kinematic infeasibility; trying again ..."
                    )
                    continue

                overall_traj["task_relev_obj"] = c_objs_new_lst
                overall_traj["task_relev_obj_mesh"] = c_objs_mesh_new_lst

                # debugging: speed up pipeline
                skip_reach_traj = False
                if skip_reach_traj:
                    # skip reach traj optimization
                    min_cost_overall_traj = overall_traj
                    min_cost_reach_traj_steps = 2
                    start_aug_succeeded = True
                    break

                q_robot_ts = overall_traj["robot_joint_qpos"][0]

                if constraint_info.aug_cfg.apply_start_aug:
                    # center around the goal!
                    center_eef_pos_sampling = (
                        X_ee_new_lst[0][:3, 3]
                        if constraint_info.aug_cfg.start_aug_cfg.cartesian_space_aug_configs.ee_pos
                        is None
                        else constraint_info.aug_cfg.start_aug_cfg.cartesian_space_aug_configs.ee_pos
                    )
                    center_eef_quat_xyzw_sampling = (
                        R.from_matrix(X_ee_new_lst[0][:3, :3]).as_quat()
                        if constraint_info.aug_cfg.start_aug_cfg.cartesian_space_aug_configs.ee_quat_xyzw
                        is None
                        else constraint_info.aug_cfg.start_aug_cfg.cartesian_space_aug_configs.ee_quat_xyzw
                    )
                    # apply z rotation: accident for square insertion: need to re-compute things for that task
                    # R1 = R.from_matrix(X_ee_new_lst[0][:3, :3]).as_matrix() @ R.from_euler("z", np.pi).as_matrix()
                    # center_eef_quat_xyzw_sampling = R.from_matrix(R1).as_quat()
                    print(
                        f"center_eef_quat_xyzw_sampling: {center_eef_quat_xyzw_sampling}"
                    )
                    if (
                        "door"
                        in constraint_info.t_to_task_relev_objs_cfg[
                            list(constraint_info.t_to_task_relev_objs_cfg.keys())[0]
                        ].obj_names[0]
                    ):
                        center_eef_pos_sampling[1] += 0.2

                    # sample around the goal pose because accuracy there is key
                    # TODO(klin) update naming to q_robot_start
                    robot_pose_start: Dict[str, torch.Tensor] = env.sample_robot_qpos(
                        sample_near_default_qpos=constraint_info.aug_cfg.start_aug_cfg.space
                        == "joint",
                        near_qpos_scaling=constraint_info.aug_cfg.start_aug_cfg.joint_space_aug_configs.joint_qpos_noise_magnitude,
                        sample_near_eef_pose=constraint_info.aug_cfg.start_aug_cfg.space
                        == "cartesian",
                        center_eef_pos=(
                            center_eef_pos_sampling
                            if constraint_info.aug_cfg.start_aug_cfg.cartesian_space_aug_configs.ee_pos
                            is None
                            else constraint_info.aug_cfg.start_aug_cfg.cartesian_space_aug_configs.ee_pos
                        ),
                        center_eef_quat_xyzw=center_eef_quat_xyzw_sampling,
                        sample_pos_x_bound=constraint_info.aug_cfg.start_aug_cfg.cartesian_space_aug_configs.eef_pos_x_bound,
                        sample_pos_y_bound=constraint_info.aug_cfg.start_aug_cfg.cartesian_space_aug_configs.eef_pos_y_bound,
                        sample_pos_z_bound=constraint_info.aug_cfg.start_aug_cfg.cartesian_space_aug_configs.eef_pos_z_bound,
                        sample_rot_angle_z_bound=constraint_info.aug_cfg.start_aug_cfg.cartesian_space_aug_configs.eef_rot_angle_z_bound,
                        sample_rot_angle_y_bound=constraint_info.aug_cfg.start_aug_cfg.cartesian_space_aug_configs.eef_rot_angle_y_bound,
                        sample_rot_angle_x_bound=constraint_info.aug_cfg.start_aug_cfg.cartesian_space_aug_configs.eef_rot_angle_x_bound,
                        # TODO(klin): this min height doesn't make sense if e.g. goal is at insertion point
                        sample_min_height=orig_ee_goal_pos[2] - 0.2,
                    )
                    robot_pose_start = {
                        k: v.numpy()
                        if v is not None and isinstance(v, torch.Tensor)
                        else v
                        for k, v in robot_pose_start.items()
                    }
                else:
                    import ipdb

                    ipdb.set_trace()
                    print("Not implemented properly yet!")
                    robot_pose_start["robot_qpos"] = (
                        env.robot_obj.default_joint_qpos.cpu().numpy()
                    )

                # Note: hardcoded this gripper qpos because, otherwise, traj-opt fails badly to find a solution
                robot_pose_start["robot_gripper_qpos"] = np.array([0.028, 0.028])
                assert env_cfg.robot_cfg.robot_model_type in [
                    "sim_panda_arm_hand",
                    "real_FR3_robotiq",
                ], "robot_model_type not supported for gripper qpos initialization"
                # TODO(klin): get these hardcoded values from the robot_object itself instead of hardcoding here
                if constraint_info.aug_cfg.init_gripper_type == "open":
                    if env_cfg.robot_cfg.robot_model_type == "sim_panda_arm_hand":
                        robot_gripper_qpos = np.array([0.04, -0.04])
                    elif env_cfg.robot_cfg.robot_model_type == "real_FR3_robotiq":
                        robot_gripper_qpos = np.array([0.0])
                elif constraint_info.aug_cfg.init_gripper_type == "closed":
                    if env_cfg.robot_cfg.robot_model_type == "sim_panda_arm_hand":
                        robot_gripper_qpos = np.array([0, 0])
                    elif env_cfg.robot_cfg.robot_model_type == "real_FR3_robotiq":
                        robot_gripper_qpos = np.array([0.804])
                elif constraint_info.aug_cfg.start_aug_cfg.only_use_tracking_start_gripper_qpos:
                    robot_gripper_qpos_second_flipped = (
                        overall_traj["robot_gripper_qpos"][0].cpu().numpy()
                    )
                    if env_cfg.robot_cfg.robot_model_type == "sim_panda_arm_hand":
                        robot_gripper_qpos = robot_gripper_qpos_second_flipped
                    elif env_cfg.robot_cfg.robot_model_type == "real_FR3_robotiq":
                        robot_gripper_qpos = robot_gripper_qpos_second_flipped
                else:
                    if env_cfg.robot_cfg.robot_model_type == "sim_panda_arm_hand":
                        robot_gripper_qpos = np.array([0.028, -0.028])
                    elif env_cfg.robot_cfg.robot_model_type == "real_FR3_robotiq":
                        robot_gripper_qpos = np.array([0.0])

                start_cfg = RobotEnvConfig(
                    robot_joint_qpos=(
                        robot_pose_start["robot_joint_qpos"]
                        if "robot_joint_qpos" in robot_pose_start.keys()
                        else None
                    ),
                    robot_gripper_qpos=robot_gripper_qpos,
                    robot_ee_pos=(
                        robot_pose_start["robot_ee_pos"]
                        if "robot_ee_pos" in robot_pose_start.keys()
                        else None
                    ),
                    robot_ee_quat_wxyz=(
                        robot_pose_start["robot_ee_quat_wxyz"]
                        if "robot_ee_quat_wxyz" in robot_pose_start.keys()
                        else None
                    ),
                    robot_base_pos=env.robot_obj.base_pos,
                    robot_base_quat_wxyz=env.robot_obj.base_quat_wxyz,
                    task_relev_obj_pos=X_se3[:3, 3],
                    task_relev_obj_rot=X_se3[:3, :3],
                    task_relev_obj_pos_nerf=X_se3[:3, 3],
                    task_relev_obj_rot_nerf=X_se3[:3, :3],
                    task_irrelev_obj_pos=None,
                    task_irrelev_obj_rot=None,
                )

                robot_env_ts_cfg_dct = robot_env_ts_to_te_cfgs[0].__dict__.copy()
                robot_env_ts_cfg_dct["robot_joint_qpos"] = q_robot_ts

                # by default the goal is the gripper qpos at the start of the tracking traj
                goal_cfg = RobotEnvConfig(
                    **robot_env_ts_cfg_dct,
                )

                tic_a = time.time()

                reach_traj = env.get_optimal_trajectory(
                    start_cfg,
                    goal_cfg,
                    obj_to_init_info=obj_to_init_info,
                    enforce_reach_traj_gripper_non_close=constraint_info.aug_cfg.start_aug_cfg.enforce_reach_traj_gripper_non_close,
                    default_robot_goal_joint_pos=overall_traj["robot_joint_qpos"][0]
                    .cpu()
                    .numpy(),
                    # use the first robot joint qpos after following the trajectory
                    use_collision_free_waypoint_heuristic=constraint_info.aug_cfg.start_aug_cfg.use_collision_free_waypoint_heuristic,
                    collision_free_waypoint_threshold=constraint_info.aug_cfg.start_aug_cfg.collision_free_waypoint_threshold,
                    collision_free_waypoint_rotation_angle_bound=constraint_info.aug_cfg.start_aug_cfg.collision_free_waypoint_rotation_angle_bound,
                    collision_free_waypoint_sampling_radius=constraint_info.aug_cfg.start_aug_cfg.collision_free_waypoint_sampling_radius,
                    start_at_collision_free_waypoint=constraint_info.aug_cfg.start_aug_cfg.start_at_collision_free_waypoint,
                    truncate_last_n_steps=env_cfg.motion_planner_cfg.truncate_last_n_steps,  # mp configs should be part of aug_cfg ...
                )
                toc_a = time.time()
                print(
                    f"Time to find a collision-free trajectory: {toc_a - tic_a} seconds."
                )

                if reach_traj is None:
                    logging.info(
                        f"Failed to find a collision-free trajectory after {str(trial_idx)} attempts."
                    )
                    print(
                        f"Failed to find a collision-free trajectory after {str(trial_idx)} attempts."
                    )
                    # import ipdb

                    # ipdb.set_trace()
                    continue

                reach_traj_steps = len(reach_traj["robot_joint_qpos"])

                # for visibility of args, separate out task_relev_obj_path from the RobotEnvConfig
                # which (for now, still) contains other task_relev_obj info e.g. task_relev_obj poses
                # task_relev_obj_pos_nerf thus corresponds to the guy loaded from task_relev_obj_path
                assert (
                    "overall_cost" in reach_traj.keys()
                ), "overall_cost not in reach_traj.keys()can't determine best eef-gripper configuration"

                # check the last val of this guy against the first of the X_ee_new_lst
                # TODO(klin): should be able to remove this if statement as part of #66
                if "task_relev_obj" not in reach_traj.keys():
                    X_ee_lst = [
                        np.block(
                            [
                                [
                                    reach_traj["robot_ee_rot_obs_frame_world"][
                                        i
                                    ].numpy(),
                                    np.expand_dims(
                                        reach_traj["robot_ee_pos_obs_frame_world"][
                                            i
                                        ].numpy(),
                                        axis=1,
                                    ),
                                ],
                                [np.array([0, 0, 0, 1])],
                            ]
                        )
                        for i in range(len(reach_traj["robot_ee_rot_obs_frame_world"]))
                    ]

                    # from welding information, update task_relev_obj using a wrapper
                    # unclear if doing the following will screw things up? i.e. duplicate objects
                    #  then update underlying stuff
                    c_objs_reach_traj_lst = [
                        overall_traj["task_relev_obj"][0] for _ in range(len(X_ee_lst))
                    ]
                    c_objs_mesh_reach_traj_lst = [
                        overall_traj["task_relev_obj_mesh"][0]
                        for _ in range(len(X_ee_lst))
                    ]

                    # hardcoded assuming 1x task relevant object to weld for now ...
                    # otherwise need multiple X_W_NERFs
                    # only weld if we actually want to weld
                    if constraint_info.aug_cfg.start_aug_cfg.weld_obj_to_ee_for_reach_traj:
                        X_W_EE = X_ee_new_lst[
                            0
                        ]  # re-welds object to the ee at the start of the tracking traj.
                        X_W_NERF = np.eye(4)
                        X_EE_NERF = np.linalg.inv(X_W_EE) @ X_W_NERF
                        c_objs_reach_traj_lst, c_objs_mesh_reach_traj_lst = (
                            Augmentor.apply_weld_transf(
                                c_objs_reach_traj_lst,
                                X_ee_lst,
                                weld_src_idx=None,
                                weld_target_idx_range=(0, len(X_ee_lst)),
                                c_objs_mesh_lst=c_objs_mesh_reach_traj_lst,
                                weld_obj_name=constraint_info.weld_obj_name,
                                weld_obj_to_ee=constraint_info.weld_obj_to_ee,
                                X_EE_NERF=X_EE_NERF,
                            )
                        )

                    # do have a dict of objects but it'll all be in the same spot ---
                    reach_traj["task_relev_obj"] = c_objs_reach_traj_lst
                    reach_traj["task_relev_obj_mesh"] = c_objs_mesh_reach_traj_lst

                for k, v in reach_traj.items():
                    if not isinstance(v, float) and not isinstance(v, int):
                        reach_traj[k].extend(overall_traj[k])
                        overall_traj[k] = reach_traj[k]

                if reach_traj["overall_cost"] < min_cost_reflect_eef_value:
                    min_cost_reflect_eef_value = reach_traj["overall_cost"]
                    min_cost_overall_traj = overall_traj
                    min_cost_reach_traj_steps = reach_traj_steps

            if min_cost_overall_traj is not None:
                start_aug_succeeded = True
                CONSOLE.log("Successfully created an augmented demo!")
                break

        if constraint_info.aug_cfg.apply_start_aug and not start_aug_succeeded:
            CONSOLE.log("Robot start configuration randomization failed.")
            print("Robot start configuration randomization failed.")
            # import ipdb

            # ipdb.set_trace()
            return None
        overall_traj = min_cost_overall_traj
        reach_traj_steps = min_cost_reach_traj_steps

        toc = time.time()
        logging.info(f"Augmentation took {toc - tic} seconds")
        print(f"Augmentation took {toc - tic} seconds")

        tic = time.time()
        # cropping to reduce NeRF rendering time during debugging
        crop_start = 0  # len(overall_traj["robot_joint_qpos"]) - 20
        crop_end = len(overall_traj["robot_joint_qpos"])

        # randomize camera pose resets the env so needs to be before randomize texture
        if constraint_info.aug_cfg.apply_camera_pose_aug:
            env.robot_obj.randomize_camera_pose()

        if constraint_info.aug_cfg.apply_appearance_aug:
            if constraint_info.aug_cfg.appearance_aug_cfg.randomize_robot_texture:
                env.robot_obj.randomize_robot_texture()

        obs_lst: List[Dict[str, torch.Tensor]] = env.get_observations(
            overall_traj["robot_joint_qpos"][crop_start:crop_end],
            overall_traj["robot_gripper_qpos"][crop_start:crop_end],
            task_relev_obj=overall_traj["task_relev_obj"][crop_start:crop_end],
            task_relev_obj_mesh=overall_traj["task_relev_obj_mesh"][
                crop_start:crop_end
            ],
            task_relev_obj_pose=overall_traj["task_relev_obj_pose"][
                crop_start:crop_end
            ],
        )

        toc = time.time()
        logging.info(f"get_observations took {toc - tic} seconds")

        # convert everything to numpy
        for k, v in overall_traj.items():
            if isinstance(v, torch.Tensor):
                overall_traj[k] = v.cpu().numpy()

        timestep_data_lst: List[TimestepData] = []
        for i, obs in enumerate(obs_lst):
            if i == len(obs_lst) - 1:
                break

            overall_obs = {k: v for k, v in obs.items()}

            for k, v in overall_traj.items():
                if isinstance(v, torch.Tensor):
                    overall_obs[k] = v[i].numpy()
                else:
                    try:
                        overall_obs[k] = v[i]
                    except Exception as e:
                        print(f"Error: {e}")
                        import ipdb

                        ipdb.set_trace()

            # all overall_traj's joint qpos in there too
            overall_obs["robot_joint_qpos"] = overall_traj["robot_joint_qpos"][i]
            overall_obs["robot_gripper_qpos"] = overall_traj["robot_gripper_qpos"][i]

            curr_ee_pos = overall_traj["robot_ee_pos_action_frame_world"][i]
            curr_ee_rot = overall_traj["robot_ee_rot_action_frame_world"][i]
            next_ee_pos = overall_traj["robot_ee_pos_action_frame_world"][i + 1]
            next_ee_rot = overall_traj["robot_ee_rot_action_frame_world"][i + 1]

            curr_gripper_qpos = overall_traj["robot_gripper_qpos"][i]
            next_gripper_qpos = overall_traj["robot_gripper_qpos"][i + 1]

            # TODO(klin): maybe get gripper action from the overall_traj.
            # no need to determine gripper action from outside?
            try:
                action_dict = get_action_dict(
                    curr_ee_pos,
                    curr_ee_rot,
                    next_ee_pos,
                    next_ee_rot,
                    curr_gripper_qpos,
                    next_gripper_qpos,
                    copy_original_gripper_action=i
                    >= reach_traj_steps
                    - 1,  # doesn't always work esp when obj is transformed; manually copy original action; only works for closing gripper; TODO better fix
                    original_gripper_action=src_demo.timestep_data[
                        i + 1 - reach_traj_steps + constraint_range[0]
                    ].action[-1],
                    set_gripper_action_close=i < reach_traj_steps
                    and constraint_info.aug_cfg.start_aug_cfg.weld_obj_to_ee_for_reach_traj,
                )
            except Exception as e:
                logging.info("get_action_dict failed with error: " + str(e))
                import ipdb

                ipdb.set_trace()
                action_dict = get_action_dict(
                    curr_ee_pos,
                    curr_ee_rot,
                    next_ee_pos,
                    next_ee_rot,
                    curr_gripper_qpos,
                    next_gripper_qpos,
                    copy_original_gripper_action=i
                    >= reach_traj_steps,  # doesn't always work esp when obj is transformed
                    original_gripper_action=src_demo.timestep_data[
                        i + 1 - reach_traj_steps + constraint_range[0]
                    ].action[-1],
                    set_gripper_action_close=i < reach_traj_steps
                    and constraint_info.aug_cfg.start_aug_cfg.weld_obj_to_ee_for_reach_traj,
                )
                return None

            # TODO(klin): check these actions e.g. by debugging in the sim
            action_abs = action_dict["action_abs"]
            action_alt_rep: Dict[str, np.ndarray] = {
                "action_abs": action_dict["action_abs"],
                "action_delta_world": action_dict["action_delta_world"],
            }

            # maybe store hmmm
            # checking how robomimic does the actions
            # TODO(klin): update the future actions section ...
            for k, v in overall_obs.items():
                if isinstance(v, torch.Tensor):
                    overall_obs[k] = v.cpu().numpy()

            mujoco_state = None
            if len(timestep_data_lst) == 0:
                SRC_DATA_TIME_SHIFT = (
                    2  # shift b/c at constraint time we might've shifted object pose
                )
                # need mjc time step corres. to mjc state!
                mujoco_state_t_idx = (
                    constraint_info.time_range[0] - SRC_DATA_TIME_SHIFT
                    if not constraint_info.weld_obj_to_ee
                    else weld_t_src
                )
                if src_demo.timestep_data[mujoco_state_t_idx].mujoco_state is not None:
                    # update_object_pose_in_mujoco_state = True
                    # if update_object_pose_in_mujoco_state:
                    # welding may use a different src mjc time step from constraint time range [0]
                    mujoco_state = src_demo.timestep_data[
                        mujoco_state_t_idx
                    ].mujoco_state.copy()
                    # update the robot joint qpos to match the qpos of the augmented demo
                    # bad: breaks api abstraction
                    joint_qpos_idxs = (
                        np.array(
                            env.robot_obj.env.env.robots[0]._ref_joint_pos_indexes,
                            dtype=np.uint8,
                        )
                        + 1
                    )
                    gripper_qpos_idxs = np.array(
                        [joint_qpos_idxs[-1] + 1, joint_qpos_idxs[-1] + 2],
                        dtype=np.uint8,
                    )
                    mujoco_state[joint_qpos_idxs] = (
                        overall_traj["robot_joint_qpos"][0].cpu().numpy()
                    )
                    mujoco_state[gripper_qpos_idxs] = (
                        overall_traj["robot_gripper_qpos"][0].cpu().numpy()
                    )

                    from demo_aug.objects.nerf_object import (
                        apply_transforms_to_pos_quat_wxyz,
                    )

                    obj_pos_idxs = np.array([10, 11, 12])
                    obj_quat_wxyz_idxs = np.array([13, 14, 15, 16])
                    if (
                        "can"
                        in constraint_info.t_to_task_relev_objs_cfg[
                            constraint_info.time_range[0]
                        ].obj_names
                    ):
                        obj_pos_idxs = np.array([31, 32, 33])
                        obj_quat_wxyz_idxs = np.array([34, 35, 36, 37])
                    obj_pos = mujoco_state[obj_pos_idxs]
                    obj_quat_wxyz = mujoco_state[obj_quat_wxyz_idxs]
                    # currently hardcoding a single object + single timestep, assuming the idxs are from 10 to 16
                    new_obj_pos, new_obj_quat_wxyz = apply_transforms_to_pos_quat_wxyz(
                        obj_pos,
                        obj_quat_wxyz,
                        overall_traj["task_relev_obj"][0][
                            constraint_info.se3_origin_obj_name
                        ].transform_params_seq,
                    )

                    mujoco_state[obj_pos_idxs] = new_obj_pos
                    mujoco_state[obj_quat_wxyz_idxs] = new_obj_quat_wxyz

            timestep_data_lst.append(
                TimestepData(
                    obs=overall_obs,
                    action=action_abs,
                    action_alt_rep=action_alt_rep,
                    future_actions=None,
                    mujoco_state=mujoco_state if len(timestep_data_lst) == 0 else None,
                    mujoco_model_xml=(
                        src_demo.timestep_data[0].mujoco_model_xml
                        if len(timestep_data_lst) == 0
                        else None
                    ),
                    objs_transf_type_seq={
                        obj_name: (
                            obj.transform_type_seq
                            if isinstance(obj, TransformationWrapper)
                            else "no-transf"
                        )  # TODO: investiage why some objs are never wrapper + add some dummt vals instaed of nothing?
                        for obj_name, obj in overall_obs["task_relev_obj"].items()
                    },
                    objs_transf_params_seq={
                        obj_name: obj.transform_params_seq
                        if isinstance(obj, TransformationWrapper)
                        else "no-transf"
                        for obj_name, obj in overall_obs["task_relev_obj"].items()
                    },
                    objs_transf_name_seq={
                        obj_name: obj.transform_name_seq
                        if isinstance(obj, TransformationWrapper)
                        else "no-transf"
                        for obj_name, obj in overall_obs["task_relev_obj"].items()
                    },
                )
            )

            # for some reason, need to use .keys() and then access the dict rather than use .items()
        return Demo(
            name=f"{src_demo.name}_{constraint_info.time_range}",
            demo_path="dummy-unclear-why-this-is-needed",  # oh needed for saving the new
            timestep_data=timestep_data_lst,
        )

    def reflect_eef(
        env: NeRFRobomimicEnv,
        X_ee_lst: List[np.ndarray],
        q_gripper_lst: List[np.ndarray],
        end_effector_type: Literal["panda_hand", "robotiq"] = "panda_hand",
        name_to_kp_params: Dict[str, Dict[str, Any]] = None,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Reflect the EEF about the symmetry plane and then do IK for the reverse gripper positions.

        Symmetry plane for the panda_hand end effector is the plane perpendicular to xy-plane and lying along the x-axis.
        """
        new_q_gripper_lst = []
        new_X_ee_lst = []

        if end_effector_type == "panda_hand":
            # rotate about z axis by np.pi to roughly simulate the reflection
            T_overall = np.eye(4)
            T_overall[0, 0] = -1
            T_overall[1, 1] = -1
        elif end_effector_type == "robotiq":
            raise NotImplementedError(
                "reflect_eef() not implented yet for robotiq end effector;that said, maybe don't need special init"
            )

        orig_gripper_kpts_lst = [
            env.get_kpts_from_ee_gripper_objs_and_params(
                X_ee_lst[i], q_gripper_lst[i], name_to_kp_params
            )
            for i in range(len(q_gripper_lst))
        ]

        # switch left w/ right and vice versa
        new_gripper_kpts_lst = []
        for orig_gripper_kpts in orig_gripper_kpts_lst:
            new_gripper_kpts = {}
            assert len(orig_gripper_kpts) == 2, "Assuming only two gripper tips for now"
            for k, v in orig_gripper_kpts.items():
                new_k = (
                    k.replace("left", "right")
                    if "left" in k
                    else k.replace("right", "left")
                )
                new_gripper_kpts[new_k] = v
            new_gripper_kpts_lst.append(new_gripper_kpts)

        # X_ee_lst = [np.eye(4) for _ in range(len(X_ee_lst))]
        X_ee_init_lst = [np.dot(X_ee, T_overall) for X_ee in X_ee_lst]
        q_gripper_init_lst = [
            np.array([-q_gripper[1], -q_gripper[0]]) for q_gripper in q_gripper_lst
        ]

        for i in range(len(q_gripper_lst)):
            kpts: Dict[str, np.ndarray] = new_gripper_kpts_lst[i]
            if i == 0:
                q_gripper_init = q_gripper_init_lst[i]
                X_ee_init = X_ee_init_lst[i]
            else:
                q_gripper_init = new_q_gripper_lst[i - 1]
                X_ee_init = new_X_ee_lst[i - 1]
            X_ee_new, q_gripper_new = env.get_eef_ik(
                X_W_ee_init=X_ee_init,
                q_gripper_init=q_gripper_init,
                name_to_frame_info=name_to_kp_params,
                kp_to_P_goal=kpts,
            )
            new_q_gripper_lst.append(q_gripper_new)
            new_X_ee_lst.append(X_ee_new)

        return new_X_ee_lst, new_q_gripper_lst

    # TODO: eventually, should be outputting a list of list of scene representations
    @staticmethod
    def apply_weld_transf(
        c_objs_lst: List[Dict[str, Union[NeRFObject, TransformationWrapper]]],
        X_ee_lst: List[np.ndarray],
        weld_src_idx: int,
        weld_target_idx_range,
        weld_obj_name: str,
        weld_obj_to_ee: bool,
        c_objs_mesh_lst: Optional[
            List[Dict[str, Union[NeRFObject, TransformationWrapper]]]
        ] = None,
        X_EE_NERF: Optional[np.ndarray] = None,
    ) -> List[List[TransformationWrapper]]:
        """
        Args:
            c_objs_lst: list of c_objs for each timestep
            X_ee_lst: list of X_ee for each timestep
            weld_src_idx: timestep index of the welding source
            weld_target_idx_range: timestep index range of the welding target
            weld_obj_name: name of the object to weld
            weld_obj_to_ee: whether to weld the object to the ee or the ee to the object
            X_EE_NERF: optional: X_EE_NERF to use for the welding transform.
                If specified, directly use this transform for all the objects in weld_target_idx_range

        Also need to specify the timesteps where we want to have welded to.

        I.e., need source welding idx and target welded indices.
        """
        # can extract X_se3_lst from X_ee_lst and weld_idx? mmm or maybe need poses???
        # pose?? wtf? hmm just need a relative transform right?
        # since we can assume X_W_NERF is eye(4) for the first object
        # hardcode to 0 for now; easier to get if it's absolute time though
        c_objs_new_lst: List[Dict[str, TransformationWrapper]] = []
        c_objs_mesh_new_lst: List[Dict[str, TransformationWrapper]] = []

        # define a closure function
        def transf_closure(
            transf_mat: torch.Tensor,
        ) -> Callable[[torch.Tensor], torch.Tensor]:
            def transf(x: torch.Tensor) -> torch.Tensor:
                return multiply_with_X_transf(torch.linalg.inv(transf_mat), x)

            return transf

        # define a closure function
        def transf_closure_for_mesh(
            transf_mat: torch.Tensor,
        ) -> Callable[[torch.Tensor], torch.Tensor]:
            def transf(x: torch.Tensor) -> torch.Tensor:
                return multiply_with_X_transf(transf_mat, x)

            return transf

        # TODO(klin): need to now what type of object we have ... hmmm
        # should the apply transforms be inside the object themselves?
        # yeah, probably let the SceneRepresentation / TransformationWrapper have something?
        # define a closure function
        def transf_closure_for_gsplat(
            transf_mat: torch.Tensor,
        ) -> Callable[[torch.Tensor], torch.Tensor]:
            def transf(x: torch.Tensor) -> torch.Tensor:
                return multiply_with_X_transf(transf_mat, x)

            return transf

        # note this closure isn't going to do anything for now
        transf_closure = (
            transf_closure if OBJ_TYPE == "nerf" else transf_closure_for_gsplat
        )

        # populate the first part of the list that doesn't need welding
        for i in range(0, weld_target_idx_range[0]):
            c_objs_new_lst.append(c_objs_lst[i])
            if c_objs_mesh_lst is not None:
                c_objs_mesh_new_lst.append(c_objs_mesh_lst[i])

        if X_EE_NERF is None:
            X_W_EE = X_ee_lst[weld_src_idx]
            # hardcoded assuming 1x task relevant object to weld for now ...
            # otherwise need multiple X_W_NERFs
            X_W_NERF = np.eye(4)
            # already precomputed!
            X_EE_NERF = (
                np.linalg.inv(X_W_EE) @ X_W_NERF
            )  # think this needs to computed post the other augs!

        # then, just set newest X_W_NERF = X_W_EE @ X_EE_NERF
        for i in range(weld_target_idx_range[0], weld_target_idx_range[1]):
            c_objs_new = c_objs_lst[i]
            X_W_EE = X_ee_lst[i]
            X_W_NERF = X_W_EE @ X_EE_NERF
            X_W_NERF_t = torch.tensor(
                X_W_NERF, dtype=torch.float32, device=torch.device("cuda")
            )
            c_objs_new = {
                name: (
                    TransformationWrapper(
                        c_obj,
                        transf_closure(X_W_NERF_t),
                        transform_type=TransformType.SE3,
                        transform_params={
                            "X_SE3": X_W_NERF.copy(),
                            "X_SE3_origin": np.eye(4),
                        },
                        transform_name="weld_to_ee",
                    )
                    if name == weld_obj_name and weld_obj_to_ee
                    else c_obj
                )
                for name, c_obj in c_objs_new.items()
            }
            c_objs_new_lst.append(c_objs_new)
            if c_objs_mesh_lst is not None:
                name_to_mesh_obj = c_objs_mesh_lst[i]
                name_to_mesh_obj = {
                    name: (
                        TransformationWrapper(
                            mesh_obj,
                            transf_closure_for_mesh(X_W_NERF_t),
                            transform_type=TransformType.SE3,
                            transform_params={
                                "X_SE3": X_W_NERF.copy(),
                                "X_SE3_origin": np.eye(4),
                            },
                            transform_name="weld_to_ee",
                        )
                        if name == weld_obj_name and weld_obj_to_ee
                        else mesh_obj
                    )
                    for name, mesh_obj in name_to_mesh_obj.items()
                }
                c_objs_mesh_new_lst.append(name_to_mesh_obj)

        # return other object without welding
        for i in range(weld_target_idx_range[1], len(c_objs_lst)):
            c_objs_new_lst.append(c_objs_lst[i])
            if c_objs_mesh_lst is not None:
                c_objs_mesh_new_lst.append(c_objs_mesh_lst[i])

        if c_objs_mesh_lst is None:
            return c_objs_new_lst
        return c_objs_new_lst, c_objs_mesh_new_lst

    @staticmethod
    def apply_appearance_augs(
        c_objs_lst: List[Dict[str, Union[NeRFObject, GSplatObject, MeshObject]]],
        appearance_augs: AppearanceAugmentationConfig,
    ) -> List[Dict[str, Union[NeRFObject, GSplatObject, MeshObject]]]:
        """Apply appearance augs to the objects in the scene."""
        # apply appearance augs
        for c_objs in c_objs_lst:
            for c_obj_name, c_obj in c_objs.items():
                if c_obj_name in appearance_augs.target_objects:
                    c_objs[c_obj_name] = ColorAugmentationWrapper(
                        c_obj, aug_types=appearance_augs.appearance_aug_types
                    )

        return c_objs_lst


def get_action_dict(
    curr_ee_pos: np.ndarray,
    curr_ee_rot: np.ndarray,
    next_ee_pos: np.ndarray,
    next_ee_rot: np.ndarray,
    curr_gripper_qpos: np.ndarray,
    next_gripper_qpos: np.ndarray,
    copy_original_gripper_action: bool = False,
    original_gripper_action: float = 0.0,
    set_gripper_action_close: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Returns a dictionary of different action representations given current and next EE pose and gripper qpos.
    """
    gripper_action = determine_gripper_action(curr_gripper_qpos, next_gripper_qpos)
    if copy_original_gripper_action:
        gripper_action = original_gripper_action

    if set_gripper_action_close:
        gripper_action = 1

    action_delta_world_homog = create_homog_mat(
        next_ee_rot, next_ee_pos
    ) @ np.linalg.inv(create_homog_mat(curr_ee_rot, curr_ee_pos))
    action_delta_world_pos = action_delta_world_homog[:3, 3]
    action_delta_world_aa = mat2axisangle(action_delta_world_homog[:3, :3])
    action_delta_world = np.concatenate(
        (action_delta_world_pos, action_delta_world_aa, [gripper_action])
    )
    # action_abs_pos = curr_ee_pos
    # action_abs_aa = mat2axisangle(curr_ee_rot)
    # TODO(klin): is the below correct? i.e. use next instead of cur positions for the action?
    action_abs_pos = next_ee_pos
    action_abs_aa = mat2axisangle(next_ee_rot)
    action_abs_gripper = gripper_action
    action_abs = np.concatenate((action_abs_pos, action_abs_aa, [action_abs_gripper]))

    return {
        "action_abs": action_abs,
        "action_delta_world": action_delta_world,
    }


def determine_gripper_action(current_qpos: List[float], next_qpos: List[float]) -> int:
    """Determine gripper action based on current and next qpos

    I think this function is OK for now.

    It's likely I'd need to check that this function works though (probably in sim).
    TODO(klin): this function isn't perfect (especially not for when I'm just copying the gripper action.)
    When copying gripper action, just copy the action and don't use this function
    """
    # convert any negative qpos values to positive
    current_qpos = np.abs(np.array(current_qpos))
    next_qpos = np.abs(np.array(next_qpos))

    eps_buffer = 1e-4  # buffer to account for floating point errors
    # we can bias if statement towards opening now, since we've a copy gripper action flag and a set gripper action flag
    cur_wider_than_next = np.all(np.abs(current_qpos) > np.abs(next_qpos) + eps_buffer)
    if cur_wider_than_next:
        # close gripper
        return 1
    else:
        return -1


def create_homog_mat(R: np.ndarray, t: np.ndarray):
    """Create a homogeneous transformation matrix from a rotation matrix and translation vector."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T
