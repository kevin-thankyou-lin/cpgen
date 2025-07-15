import logging
import pathlib
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from typing import Dict, List, Literal, Optional, Set, Tuple, Union

import cv2
import imageio
import numpy as np
import torch
import tyro
from diffusion_policy.dataset.base_dataset import LinearNormalizer
from r2d2.evaluation.policy_wrapper import PolicyWrapperRobomimic
from scipy.spatial.transform import Rotation as R

import wandb
from demo_aug.annotators.constraints import ConstraintAnnotator
from demo_aug.annotators.segmentation_masks import SegmentationMaskAnnotator
from demo_aug.configs.base_config import ConstraintInfo, PolicyEvalConfig
from demo_aug.demo import Demo
from demo_aug.envs.nerf_robomimic_env import NeRFRobomimicEnv
from demo_aug.objects.nerf_object import ColorAugmentationWrapper
from demo_aug.objects.reconstructor import ReconstructionManager, ReconstructionType
from demo_aug.utils.data_collection_utils import load_demos
from demo_aug.utils.mathutils import eval_error
from demo_aug.utils.viz_utils import plot_image_diff_map

# PYTHONPATH=.:../R2D2/:../consistency-policy/ python scripts/policy/eval_policy.py fr3-real-env

# Configure the root logger to write log messages to a file
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)  # noqa: E402


def update_logging_filename(new_filename):
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            root_logger.removeHandler(handler)
            handler.close()

    new_file_handler = logging.FileHandler(new_filename)
    new_file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    root_logger.addHandler(new_file_handler)


logging.basicConfig(level=logging.INFO)


def determine_reconstruction_timesteps(
    constraint_infos: List[ConstraintInfo],
):
    """Determine which timesteps require 3D reconstructions based on user input and existing data;
    ignores redundant reconstructions, for example, if we're using a single static NeRF model for multiple
    times.

    TODO: move elsewhere; maybe not Reconstructor class to keep it clean and not rely on ConstraintInfos
    """
    reconstruction_timesteps: Union[List, Set[Tuple[int]]] = set()
    for constraint in constraint_infos:
        reconstruction_timestep_for_constraint = (
            constraint.collect_reconstruction_timesteps()
        )
        reconstruction_timesteps.update(reconstruction_timestep_for_constraint)
    reconstruction_timesteps_list = list(reconstruction_timesteps)
    return reconstruction_timesteps_list


NORMALIZER_PREFIX_LENGTH = 11
MODEL_PREFIX_LENGTH = 6


def load_normalizer(workspace_state_dict):
    keys = workspace_state_dict["state_dicts"]["model"].keys()
    normalizer_keys = [key for key in keys if "normalizer" in key]
    normalizer_dict = {
        key[NORMALIZER_PREFIX_LENGTH:]: workspace_state_dict["state_dicts"]["model"][
            key
        ]
        for key in normalizer_keys
    }

    normalizer = LinearNormalizer()
    normalizer.load_state_dict(normalizer_dict)

    return normalizer


def load_policy(
    ckpt_path: str,
    old_policy_loading: bool = False,
) -> Tuple[PolicyWrapperRobomimic, Tuple[int, int], Tuple[int, int]]:
    import dill
    import hydra.utils
    import torch
    from consistency_policy.utils import get_policy_and_cfg
    from diffusion_policy.workspace.base_workspace import BaseWorkspace

    if old_policy_loading:
        # TODO(klin): test using CP's get_policy method
        # load DP/CP-repo trained checkpoint
        payload = torch.load(open(ckpt_path, "rb"), pickle_module=dill)
        cfg = payload["cfg"]
        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg)
        workspace: BaseWorkspace
        # workspace.load_payload(payload, exclude_keys=None, include_keys=None)
        workspace.load_checkpoint(path=ckpt_path, exclude_keys=["optimizer"])
        workspace_state_dict = torch.load(ckpt_path)
        normalizer = load_normalizer(workspace_state_dict)

        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model
        policy.set_normalizer(normalizer)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        policy.eval().to(device)
    else:
        policy, cfg = get_policy_and_cfg(ckpt_path)
        device = policy.device

    # set inference params
    policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1

    # policy.n_action_steps = 16  # hardcoded
    ### infer image size ###
    # policy's predict_action expects (H, W, C); policy expects (C, H, W); camera_reader handles resizing
    hand_im_shape = cfg["shape_meta"]["obs"]["hand_camera_image"]["shape"][1:3][
        ::-1
    ]  # inverse b/c cv2 uses W x H
    agenview_im_shape = cfg["shape_meta"]["obs"]["agentview_image"]["shape"][1:3][
        ::-1
    ]  # inverse b/c cv2 uses W x H

    action_space = (
        "cartesian_position"  # hardcoded gripper action space to match eef action space
    )

    # Prepare Policy Wrapper #
    data_processing_kwargs = dict(
        timestep_filtering_kwargs=dict(
            action_space=action_space,
            robot_state_keys=[
                "cartesian_position",
                "gripper_position",
                "joint_positions",
            ],
            # camera_extrinsics=[],
        ),
        image_transform_kwargs=dict(
            remove_alpha=True,
            bgr_to_rgb=False,
            to_tensor=False,
            augment=False,
        ),
    )
    timestep_filtering_kwargs = data_processing_kwargs.get(
        "timestep_filtering_kwargs", {}
    )
    image_transform_kwargs = data_processing_kwargs.get("image_transform_kwargs", {})

    policy_timestep_filtering_kwargs = image_transform_kwargs.get(
        "timestep_filtering_kwargs", {}
    )
    policy_image_transform_kwargs = image_transform_kwargs.get(
        "image_transform_kwargs", {}
    )

    policy_timestep_filtering_kwargs.update(timestep_filtering_kwargs)
    policy_image_transform_kwargs.update(image_transform_kwargs)

    wrapped_policy = PolicyWrapperRobomimic(
        policy=policy,
        timestep_filtering_kwargs=policy_timestep_filtering_kwargs,
        image_transform_kwargs=policy_image_transform_kwargs,
        frame_stack=cfg.n_obs_steps,
        eval_mode=True,
        cfg=cfg,
        device=device,
        n_acts=8,  # if using only 8 action, only last is close gripper
    )

    return wrapped_policy, hand_im_shape, agenview_im_shape


OBJ_TYPE: Literal["nerf", "gsplat"] = "gsplat"


def extract_obs_action(
    src_demo: Demo,
    convert_axis_angle_to_euler_xyz: bool = False,
    start_idx: int = 0,
) -> Tuple[Dict, List[np.ndarray]]:
    gt_obs = src_demo.timestep_data[start_idx].obs
    actions = np.array(
        [ts.action for ts in src_demo.timestep_data[start_idx : start_idx + 16]]
    )
    if convert_axis_angle_to_euler_xyz:
        from robosuite.utils.transform_utils import axisangle2quat

        for i in range(len(actions)):
            curr_axis_angle = actions[i][3:6]
            quat = axisangle2quat(curr_axis_angle)
            actions[i][3:6] = R.from_quat(quat).as_euler("xyz", degrees=False)
    return gt_obs, actions


def dict_to_torch_with_prefix(
    obs: dict, prefix: str = "robot_"
) -> Dict[str, torch.Tensor]:
    prefixed_obs = {}
    for key, value in obs.items():
        new_key = f"{prefix}{key}"
        if isinstance(value, np.ndarray):
            prefixed_obs[new_key] = torch.tensor(value, dtype=torch.float32)
        if "quat" in new_key:
            new_key = new_key.replace("quat", "quat_wxyz")
            if "eef" in new_key:
                new_key = new_key.replace("eef", "ee")
            # add quat_wxyz to the dict --- need to update the value by rolling +1
            prefixed_obs[new_key] = torch.tensor(np.roll(value, 1), dtype=torch.float32)
        if "robot0_joint_qpos" in new_key:
            # add robot1 to the dict
            prefixed_obs["robot_joint_qpos"] = torch.tensor(value, dtype=torch.float32)
        if "eef" in new_key:
            # swap the "eef" to "ee"
            prefixed_obs[new_key.replace("eef", "ee")] = torch.tensor(
                value, dtype=torch.float32
            )

    return prefixed_obs


def process_reconstructions(
    rec_manager: ReconstructionManager,
    reconstruction_ts_list: List[Tuple[int]],
    timestep_to_nerf_folder: Dict[str, str],
    constraints: List[ConstraintInfo],
    src_demo: Demo,
    demo_aug_cfg: PolicyEvalConfig,
) -> None:
    for ts in reconstruction_ts_list:
        rec_manager.reconstruct(ts, timestep_to_nerf_folder)

    for constraint in constraints:
        for (
            rec_ts,
            obj_name,
        ) in constraint.collect_reconstruction_timesteps_to_obj_name().items():
            segmentation_mask = SegmentationMaskAnnotator.get_segmentation_masks(
                src_demo, demo_aug_cfg, rec_manager, rec_ts, obj_name
            )
            rec_manager.reconstruct(
                rec_ts,
                timestep_to_nerf_folder,
                obj_name,
                segmentation_mask,
                ReconstructionType.NeRF,
            )
            rec_manager.reconstruct(
                rec_ts,
                timestep_to_nerf_folder,
                obj_name,
                segmentation_mask,
                ReconstructionType.GaussianSplat,
            )
            rec_manager.reconstruct(
                rec_ts,
                timestep_to_nerf_folder,
                obj_name,
                segmentation_mask,
                ReconstructionType.Mesh,
            )

    src_demo.add_reconstruction_manager(rec_manager)
    src_demo.add_constraint_infos(constraints)


def compare_gen_obs_policy_action_to_ground_truth(
    env: NeRFRobomimicEnv,
    cfg: PolicyEvalConfig,
    src_demo: Demo,
    policy: PolicyWrapperRobomimic,
    src_demo_start_idx: int = 10,
    ee_pos_offset: Optional[np.ndarray] = None,
    viz_obs_seq=True,
) -> None:
    robot_state: Dict[str, np.ndarray]
    if cfg.ground_truth_src_path is None:
        robot_state: Dict[str, np.ndarray] = src_demo.timestep_data[
            src_demo_start_idx
        ].obs
        robot_state = dict_to_torch_with_prefix(robot_state)
        objs_transf_params_seq = src_demo.timestep_data[
            src_demo_start_idx
        ].objs_transf_params_seq

        if ee_pos_offset is not None:
            robot_state["robot_ee_pos"] = robot_state["robot_eef_pos"] + ee_pos_offset

        gt_obs, gt_action = extract_obs_action(
            src_demo, convert_axis_angle_to_euler_xyz=True, start_idx=src_demo_start_idx
        )
    else:
        import pickle

        with open(cfg.ground_truth_src_path, "rb") as f:
            inference_data = pickle.load(f)
        gt_obs = inference_data["obs"]
        for key, value in gt_obs.items():
            if "image" in key:
                gt_obs[key] = (
                    value.squeeze(0).squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                    * 255
                ).astype(np.uint8)

        gt_action = inference_data["predicted_actions"][0].detach().cpu().numpy()
        real_world_raw_obs = inference_data["raw_obs"]

        robot_state = dict_to_torch_with_prefix(
            {
                "eef_pos": real_world_raw_obs["robot_state"]["cartesian_position"][:3],
                "eef_quat": R.from_euler(
                    "xyz", real_world_raw_obs["robot_state"]["cartesian_position"][3:]
                ).as_quat(),
                "gripper_qpos": np.array(
                    [real_world_raw_obs["robot_state"]["gripper_position"]]
                ),
                "robot0_joint_qpos": real_world_raw_obs["robot_state"][
                    "joint_positions"
                ],
            }
        )

        objs_transf_params_seq = src_demo.timestep_data[-20].objs_transf_params_seq
        for obj_name, obj_transf_params in objs_transf_params_seq.items():
            for param_name, _ in obj_transf_params.items():
                objs_transf_params_seq[obj_name][param_name] = np.eye(4)

    obs, _ = env.reset(
        seed=0,
        robot_state=robot_state,
        objs_transf_params_seq=objs_transf_params_seq,
    )
    policy.reset()
    action = policy.forward(obs)
    predicted_actions = [action]
    for _ in range(12):
        action = policy.action_chunker.get_action()
        predicted_actions.append(action)

    gt_action = gt_action[:12]
    predicted_actions = predicted_actions[:12]
    import ipdb

    ipdb.set_trace()
    for i in range(12):
        print(f"Predicted    action {i}: {predicted_actions[i]}")
        print(f"Ground truth action {i}: {gt_action[i]}")
        print(
            f"Diff is              : {np.array(predicted_actions[i][:3] - gt_action[i][:3])}"
        )
        print(
            f"{i} max diff abs: {np.max(np.abs((predicted_actions[i][:3] - gt_action[i][:3])))}"
        )

    # mse loss
    mse_loss = np.mean(
        np.square(
            np.array(predicted_actions)[..., :3] - np.array(gt_action)[..., :3]
        ).sum(axis=1)
    )
    print(f"MSE loss: {mse_loss}")

    generated_hand_camera = obs["image"]["12391924_left_image"]
    generated_agentview = obs["image"]["27432424_left_image"]

    loaded_hand_camera = gt_obs["hand_camera_image"]
    loaded_agentview = gt_obs["agentview_image"]

    generated_hand_camera = cv2.resize(
        generated_hand_camera, loaded_hand_camera.shape[:2][::-1]
    )

    plot_image_diff_map(
        generated_hand_camera.astype(np.int16),
        loaded_hand_camera.astype(np.int16),
        "Hand Camera Image Diff",
        save_dir=cfg.save_dir,
        print_max_diff=True,
    )
    plot_image_diff_map(
        generated_agentview.astype(np.int16),
        loaded_agentview.astype(np.int16),
        "Agentview Image Diff",
        save_dir=cfg.save_dir,
        print_max_diff=True,
    )

    if viz_obs_seq:
        obs_seq_hand = []
        obs_seq_agentview = []
        for i in range(11):
            obs, _, _, _ = env.step(predicted_actions[i])
            obs_seq_hand.append(obs["image"]["12391924_left_image"])
            obs_seq_agentview.append(obs["image"]["27432424_left_image"])

        # save images as mp4 via imageio
        save_dir = pathlib.Path(cfg.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        hand_mp4_path = save_dir / "obs_seq_hand.mp4"
        imageio.mimsave(hand_mp4_path, obs_seq_hand, fps=10)
        print(f"Saved hand images to {hand_mp4_path}")
        agentview_mp4_path = save_dir / "obs_seq_agentview.mp4"
        imageio.mimsave(agentview_mp4_path, obs_seq_agentview, fps=10)


def main(cfg: PolicyEvalConfig) -> None:
    demo_aug_cfg = cfg.demo_aug_cfg
    if demo_aug_cfg.use_wandb:
        wandb.init(project="eval-policy", entity="kevin-lin")
        # append wandb.run.id to demo_aug_cfg.save_file_name
        curr_save_path = pathlib.Path(demo_aug_cfg.save_file_name)
        demo_aug_cfg.save_file_name = (
            f"{curr_save_path.stem}_{wandb.run.id + curr_save_path.suffix}"
        )
        wandb.config.update(asdict(demo_aug_cfg))

    src_demos: List[Demo] = load_demos(
        demo_aug_cfg.demo_path, load_images=True, debug_mode=True
    )
    for i, src_demo in enumerate(src_demos):
        x_se3 = src_demo.timestep_data[0].objs_transf_params_seq["wine_glass"][
            "1:X_SE3"
        ]
        print(i, x_se3[0, 3], x_se3[1, 3])

    if "robomimic" in str(demo_aug_cfg.demo_path):
        import robomimic.utils.file_utils as FileUtils

        env_cfg = FileUtils.get_env_metadata_from_dataset(
            dataset_path=demo_aug_cfg.demo_path
        )
    else:
        # handling real world dataset
        env_cfg = demo_aug_cfg.env_cfg

    # env_info contains robot + camera (intrinsics) info + camera frame transform w.r.t robot EE frame
    # helps figure out corresponding robot action post obj pose transform
    # env_info may be good to load into each demo as an entry then?
    # dataset should contain information include the transforms betweens cameras and robot
    for i, src_demo in enumerate(src_demos):
        src_demo_name = str(src_demo.name).split(".")[0]
        generated_demos_save_path = (
            demo_aug_cfg.save_base_dir
            / pathlib.Path(
                src_demo_name + f"{demo_aug_cfg.trials_per_constraint}trials"
            )
            / datetime.now().strftime("%Y-%m-%d")
            / wandb.run.id
            / demo_aug_cfg.save_file_name
            if demo_aug_cfg.use_wandb
            else pathlib.Path("debug") / demo_aug_cfg.save_file_name
        )

        # create directory if it doesn't exist
        generated_demos_save_path.parent.mkdir(parents=True, exist_ok=True)
        update_logging_filename(str(generated_demos_save_path.parent / "app.log"))

        constraints = ConstraintAnnotator.get_constraints(src_demo, demo_aug_cfg)

        # if rec_manager is None:
        rec_manager = ReconstructionManager(str(src_demo.demo_path), src_demo.name)

        reconstruction_ts_list = determine_reconstruction_timesteps(constraints)
        # couldn't give demo_aug_cfg key as a tuple on the CLI, so manually converting here
        timestep_to_nerf_folder = {
            str(key): value
            for key, value in demo_aug_cfg.timestep_to_nerf_folder.items()
        }

        process_reconstructions(
            rec_manager,
            reconstruction_ts_list,
            timestep_to_nerf_folder,
            constraints,
            src_demo,
            demo_aug_cfg,
        )

        constraint_info = constraints[0]
        constraint_range = constraint_info.time_range
        c_objs_lst = src_demo.get_task_relev_objs_for_range(
            constraint_range[0],
            constraint_range[1],
            constraint_info,
            ReconstructionType.NeRF
            if OBJ_TYPE == "nerf"
            else ReconstructionType.GaussianSplat,
        )

        # currently handcoded because generated demo dataset currently doesn't contain src demo generation data
        orig_ee_goal_pos = np.array([0.43016782, 0.3153621, 0.15920483])
        orig_ee_goal_quat_xyzw = np.array(
            [0.59590929, 0.44576579, -0.51371556, 0.42694413]
        )
        orig_ee_goal_pose = np.eye(4)
        orig_ee_goal_pose[:3, :3] = R.from_quat(orig_ee_goal_quat_xyzw).as_matrix()
        orig_ee_goal_pose[:3, 3] = orig_ee_goal_pos

        t_idx = 0
        if cfg.ood_variant == "remove_bg":
            c_objs_lst[t_idx].pop("background")

        # TODO(klin): 8/8 check in / write details re color augs; not optimizal but okay for now
        if cfg.ood_variant == "object_color":
            # loop through c_objs_lst[t_idx] and apply ColorAugmentationWrapper to the specific object
            for obj_name in c_objs_lst[t_idx].keys():
                if obj_name == cfg.target_object:
                    c_objs_lst[t_idx][obj_name] = ColorAugmentationWrapper(
                        c_objs_lst[t_idx][obj_name],
                        aug_types=["brightness", "contrast", "noise"],
                    )

        env = NeRFRobomimicEnv(
            env_cfg, renderable_objs=c_objs_lst[t_idx], aug_cfg=constraint_info.aug_cfg
        )

        if cfg.ood_variant == "robot_color":
            env.robot_obj.randomize_robot_texture()

        # load policy
        policy, hand_im_shape, agentview_im_shape = load_policy(
            cfg.policy_path, old_policy_loading=True
        )

        # loop begins here
        max_steps = len(src_demo.timestep_data) + 20

        if cfg.compare_to_ground_truth:
            ee_pos_offset = np.array([0.0, 0.0, 0.0])
            compare_gen_obs_policy_action_to_ground_truth(
                env,
                cfg,
                src_demo,
                policy,
                src_demo_start_idx=43 - 16,
                ee_pos_offset=ee_pos_offset,
            )

        n_trials = cfg.n_trials

        ee_pos_offsets = [np.zeros(3)]
        for x in np.linspace(-0.04, 0.04, 2):
            for y in np.linspace(-0.04, 0.04, 2):
                for z in np.linspace(-0.04, 0.04, 2):
                    ee_pos_offsets.append(np.array([x, y, z]))

        if cfg.obj_transf_variant == "grid_sweep":
            x_vals = np.linspace(-0.05, 0.05, 4)
            y_vals = np.linspace(-0.05, 0.05, 4)
            x_vals, y_vals = np.meshgrid(x_vals, y_vals)
            x_vals = x_vals.flatten()
            y_vals = y_vals.flatten()
            n_trials = len(x_vals)

        # n_trials = 1
        ee_pos_offsets = [np.zeros(3)] * n_trials
        errors_at_constraints: List[float] = []

        breakpt = False
        for trial in range(n_trials):
            constraint_eef_pose = orig_ee_goal_pose.copy()
            transform_task_relev_obj = "wine_glass"
            test_on_training_data = False
            test_on_input_data = True
            if breakpt:
                import ipdb

                ipdb.set_trace()
            if test_on_training_data:
                robot_state: Dict[str, np.ndarray] = src_demo.timestep_data[0].obs
                robot_state = dict_to_torch_with_prefix(src_demo.timestep_data[0].obs)
                objs_transf_params_seq = src_demo.timestep_data[
                    0
                ].objs_transf_params_seq
                ee_pos_offset = ee_pos_offsets[trial]
                robot_state["robot_ee_pos"] = robot_state[
                    "robot_eef_pos"
                ] + torch.tensor(ee_pos_offset, dtype=torch.float32)
            elif test_on_input_data:
                if False:
                    import pickle

                    with open(cfg.ground_truth_src_path, "rb") as f:
                        inference_data = pickle.load(f)
                    gt_obs = inference_data["obs"]
                    # for each image, squeeze the first two dims than permute the first channel to the last
                    for key, value in gt_obs.items():
                        if "image" in key:
                            gt_obs[key] = (
                                value.squeeze(0)
                                .squeeze(0)
                                .permute(1, 2, 0)
                                .detach()
                                .cpu()
                                .numpy()
                                * 255
                            )
                            gt_obs[key] = gt_obs[key].astype(np.uint8)

                    gt_action = (
                        inference_data["predicted_actions"][0].detach().cpu().numpy()
                    )
                    real_world_raw_obs = inference_data["raw_obs"]

                    # scipy transform to get the eef_pos and eef_quat
                    # convert the above obs to {"eef_pos", "eef_quat", "gripper_qpos", "joint_qpos"}

                    robot_state: Dict[str, np.ndarray] = {
                        "eef_pos": real_world_raw_obs["robot_state"][
                            "cartesian_position"
                        ][:3],
                        "eef_quat": R.from_euler(
                            "xyz",
                            real_world_raw_obs["robot_state"]["cartesian_position"][3:],
                        ).as_quat(),  # is it xyzw or wxyz?
                        "gripper_qpos": np.array(
                            [real_world_raw_obs["robot_state"]["gripper_position"]]
                        ),
                        "robot0_joint_qpos": real_world_raw_obs["robot_state"][
                            "joint_positions"
                        ],
                    }
                    robot_state = dict_to_torch_with_prefix(robot_state)
                else:
                    robot_state: Dict[str, torch.Tensor] = env.sample_robot_qpos(
                        sample_near_default_qpos=True,
                        near_qpos_scaling=0.06,
                        sample_near_eef_pose=False,
                        forward_kinematics_body_name="camera_mount",
                    )
                    robot_state["robot_gripper_qpos"][:] = 0

                obj_to_x_y: Dict[str, Tuple] = defaultdict(tuple)
                if cfg.obj_transf_variant == "fixed_uniform":
                    # let x range from -0.05 to 0.05 based on the number of trials
                    x_vals = np.linspace(-0.05, 0.05, cfg.n_trials)
                    y_vals = np.linspace(-0.05, 0.05, cfg.n_trials)
                    x_val = x_vals[trial]
                    y_val = y_vals[trial]
                elif cfg.obj_transf_variant == "grid_sweep":
                    x_vals = np.linspace(-0.05, 0.05, 4)
                    y_vals = np.linspace(-0.05, 0.05, 4)
                    # meshgrid
                    x_vals, y_vals = np.meshgrid(x_vals, y_vals)
                    x_vals = x_vals.flatten()
                    y_vals = y_vals.flatten()
                    x_val = x_vals[trial]
                    y_val = y_vals[trial]
                else:
                    se3 = env.sample_task_relev_obj_se3_transform(
                        cfg.demo_aug_cfg.aug_cfg.se3_aug_cfg.dx_range,
                        cfg.demo_aug_cfg.aug_cfg.se3_aug_cfg.dy_range,
                        cfg.demo_aug_cfg.aug_cfg.se3_aug_cfg.dz_range,
                        cfg.demo_aug_cfg.aug_cfg.se3_aug_cfg.dthetaz_range,
                    )
                    x_val = se3[0, 3]
                    y_val = se3[1, 3]

                obj_to_x_y["wine_glass"] = (x_val, y_val)

                if cfg.obj_transf_variant == "use_src_demo":
                    obj_to_x_y.pop("wine_glass")

                if cfg.ood_variant == "shift_bg":
                    x_vals = np.linspace(
                        cfg.bg_shift_x_range[0], cfg.bg_shift_x_range[1], cfg.n_trials
                    )
                    y_vals = np.linspace(
                        cfg.bg_shift_x_range[0], cfg.bg_shift_x_range[1], cfg.n_trials
                    )
                    x_val = x_vals[trial]
                    y_val = y_vals[trial]
                    if cfg.bg_shift_type == "fixed":
                        x_val, y_val = cfg.bg_shift_x_y

                    obj_to_x_y["background"] = (x_val, y_val)

                objs_transf_params_seq = src_demo.timestep_data[
                    t_idx
                ].objs_transf_params_seq
                # TODO(klin): Aug/7 check enable bg shift too --- check pipeline works
                # loop though and set all values to np.eye(4)
                for obj_name, obj_transf_params in objs_transf_params_seq.items():
                    for param_name, param_value in obj_transf_params.items():
                        if len(obj_to_x_y[obj_name]) == 0:
                            continue
                        if obj_name == "wine_glass" and param_name == "1:X_SE3":
                            objs_transf_params_seq[obj_name][param_name] = np.eye(4)
                            objs_transf_params_seq[obj_name][param_name][0, 3] = (
                                obj_to_x_y[obj_name][0]
                            )
                            objs_transf_params_seq[obj_name][param_name][1, 3] = (
                                obj_to_x_y[obj_name][1]
                            )
                        elif obj_name == "background" and param_name == "0:X_SE3":
                            objs_transf_params_seq[obj_name][param_name] = np.eye(4)
                            objs_transf_params_seq[obj_name][param_name][0, 3] = (
                                obj_to_x_y[obj_name][0]
                            )
                            objs_transf_params_seq[obj_name][param_name][1, 3] = (
                                obj_to_x_y[obj_name][1]
                            )
                        else:
                            objs_transf_params_seq[obj_name][param_name] = np.eye(4)
                constraint_eef_pose = (
                    objs_transf_params_seq[transform_task_relev_obj]["1:X_SE3"]
                    @ constraint_eef_pose
                )
                # update eef_pos and eef_quat based on the obj_to_x_y
                # TODO: just coinvert to cosntraint_eef_pose
            else:
                robot_state = None
                objs_transf_params_seq = None

            # Get ground truth actions
            gt_obs, gt_action = extract_obs_action(
                src_demo, convert_axis_angle_to_euler_xyz=True
            )

            obs, info = env.reset(
                seed=trial,
                robot_state=robot_state,
                objs_transf_params_seq=objs_transf_params_seq,
                use_ik_for_qpos_update=True,
            )
            # reset the policy wrapper
            policy.reset()

            images_hand = []
            images_agentview = []
            pos_errors = []
            ori_errors = []
            gripper_actions = []

            images_hand.append(obs["image"]["12391924_left_image"])
            images_agentview.append(obs["image"]["27432424_left_image"])
            for i in range(max_steps):
                # convert obs' image shape to match policy's input shape of hand and agentview images
                obs["image"]["12391924_left_image"] = cv2.resize(
                    obs["image"]["12391924_left_image"],
                    hand_im_shape,
                )
                obs["image"]["27432424_left_image"] = cv2.resize(
                    obs["image"]["27432424_left_image"],
                    agentview_im_shape,
                )

                with torch.no_grad():
                    action = policy.forward(obs)
                obs, reward, done, info = env.step(action)
                gripper_actions.append(action[-1])

                # acquire obs eef pose
                ee_pose = np.eye(4)
                eef_pos = obs["robot_state"]["cartesian_position"][:3]
                eef_euler_xyz = obs["robot_state"]["cartesian_position"][3:]
                ee_pose[:3, :3] = R.from_euler("xyz", eef_euler_xyz).as_matrix()
                ee_pose[:3, 3] = eef_pos

                pos_error, ori_err = eval_error(ee_pose, constraint_eef_pose)
                pos_errors.append(pos_error)
                ori_errors.append(ori_err)

                images_hand.append(obs["image"]["12391924_left_image"])
                images_agentview.append(obs["image"]["27432424_left_image"])
                if i == 69:
                    print(i)
                    print(
                        f"obs['robot_state']['cartesian_position']: {obs['robot_state']['cartesian_position']}"
                    )
                    print(
                        f"pos diff from 0.43135732 0.3159827  0.15919787 is: {obs['robot_state']['cartesian_position'][:3] - np.array([0.43135732, 0.3159827, 0.15919787])}"
                    )

            plot_error = True
            if plot_error:
                # plot the pos error chart
                import matplotlib.pyplot as plt

                # Clear the current figure
                plt.clf()

                # Plotting the arrays
                plt.plot(gripper_actions, label="gripper_actions")
                plt.plot(pos_errors, label="pos_errors")

                # Adding labels and title
                plt.xlabel("Index")
                plt.ylabel("Value")
                plt.title("Plot of gripper_actions and pos_errors")

                # set xlim
                plt.xlim([0, 70])
                plt.ylim([-0.0, 0.05])
                # Adding a legend
                plt.legend()

                # Save the plot as an image file
                output_path = f"plot_gripper_actions_pos_errors_trial_{trial}.png"
                plt.savefig(output_path)
                print(f"Saved plot to {output_path}")

            # find first index where gripper action goes to 1 for more than 2 steps
            def find_gripper_close_index(arr: np.ndarray) -> int:
                first_positive_index = -1
                # Check for consecutive positives
                for j in range(len(arr) - 1):
                    if arr[j] > 0 and arr[j + 1] > 0:
                        return j
                # If no consecutive positives, check for first positive
                if first_positive_index == -1:
                    for value in arr:
                        if value > 0:
                            first_positive_index = j
                            break
                return first_positive_index

            # find the index at which gripper closes
            gripper_close_index = find_gripper_close_index(gripper_actions)
            # get the pos error at the index where gripper closes
            if gripper_close_index != -1:
                errors_at_constraints.append(pos_errors[gripper_close_index])
                print(
                    f"pos error at gripper close index {gripper_close_index}: {pos_errors[gripper_close_index]}"
                )
            else:
                print("Gripper never closes; skip error calculation")

            # also plot the point at which gripper closes
            policy.reset()

            # save images as mp4
            save_dir = pathlib.Path(cfg.save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            hand_mp4_path = (
                save_dir
                / f"{src_demo_name}_{trial}_ee_pos_offset_{ee_pos_offsets[trial]}_hand.mp4"
            )
            agentview_mp4_path = (
                save_dir
                / f"{src_demo_name}_{trial}_ee_pos_offset_{ee_pos_offsets[trial]}_agentview.mp4"
            )
            # add obj trans params to the filename
            hand_mp4_path = hand_mp4_path.parent / (
                hand_mp4_path.stem
                + f"_obj_transf_{obj_to_x_y['wine_glass'][0]:.2f}_{obj_to_x_y['wine_glass'][1]:.2f}"
                + hand_mp4_path.suffix
            )
            agentview_mp4_path = agentview_mp4_path.parent / (
                agentview_mp4_path.stem
                + f"_obj_transf_{obj_to_x_y['wine_glass'][0]:.2f}_{obj_to_x_y['wine_glass'][1]:.2f}"
                + agentview_mp4_path.suffix
            )
            # remove "(" and ")" from the filename and "," and " " from the ee_pos_offset
            hand_mp4_path = hand_mp4_path.with_name(
                hand_mp4_path.name.replace("(", "")
                .replace(")", "")
                .replace(",", "")
                .replace(" ", "")
            )
            agentview_mp4_path = agentview_mp4_path.with_name(
                agentview_mp4_path.name.replace("(", "")
                .replace(")", "")
                .replace(",", "")
                .replace(" ", "")
            )
            if True:
                imageio.mimsave(hand_mp4_path, images_hand, fps=15)
                imageio.mimsave(agentview_mp4_path, images_agentview, fps=15)
                print(f"Saved hand images to {hand_mp4_path}")
                print(f"Saved agentview images to {agentview_mp4_path}")
                print(f"errors_at_constraints: {errors_at_constraints}")
                print(f"mean error at constraints: {np.mean(errors_at_constraints)}")

        save_path = f"{cfg.save_dir}/{src_demo_name}_errors_at_constraints.txt"
        # write this to a file
        with open(save_path, "w") as f:
            for error in errors_at_constraints:
                f.write(f"{error}\n")
            # also save the mean error
            f.write(f"# mean error at constraints: {np.mean(errors_at_constraints)}")

        print(f"saving errors at constraints to {save_path}")
        import ipdb

        ipdb.set_trace()


if __name__ == "__main__":
    # load demonstrations file
    tyro.cli(main)
