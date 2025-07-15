"""
A script to generate some data to train a nerf and check how the nerf's quality is.
"""

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import cv2
import h5py
import numpy as np
import torch
import tyro
from tqdm import tqdm

from demo_aug.utils.camera_utils import (
    get_camera_extrinsic_matrix,
    get_camera_intrinsic_matrix,
    get_real_depth_map,
    get_real_distance_map,
)

try:
    pass
except ImportError:
    raise ImportError(
        "Robomimic must be installed to run this script. Please follow instructions at "
        "https://robomimic.github.io/docs/datasets/robomimic_v0.1.html"
        "Note: also need to install robosuite and may need to reference this github issue"
        "to resolve GLEW initialization errors: https://github.com/ARISE-Initiative/robosuite/issues/114"
    )
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
from mujoco_py.generated import const
from robosuite.utils.transform_utils import mat2quat

from demo_aug.utils.mujoco_utils import (
    add_camera_to_xml,
    add_camera_xml_to_xml,
    get_body_bounding_box,
    remove_camera_from_xml,
)
from demo_aug.utils.viz_utils import (
    get_lookat,
    render_image_collage_from_image_list,
)

logging.basicConfig(level=logging.INFO)


@dataclass
class Config:
    """Script to generate images using the given arguments."""

    dataset: str
    """Path to hdf5 dataset."""

    save_dir: str = None
    """(optional) Directory to save images to."""

    num_cameras: int = 30
    """Number of cameras per camera type (eye_in_hand_camera, target_object_camera) to render from."""

    radius: float = 0.5
    """The radius value."""

    image_height: int = 512
    """Image height."""

    image_width: int = 512
    """Image width."""

    render_depths: bool = True
    """Whether to render depth images."""

    render_pitcher: bool = False
    """Whether to render pitcher: hardcoded transparent object that requires manually imported blender export."""

    timesteps: List[int] = field(default_factory=list)
    """Timesteps to render images."""

    overwrite_all: bool = False
    """Whether to overwrite all images and saved data."""

    reset_to_loaded_xml: bool = True
    """Whether to reset the environment to the loaded xml."""

    use_eye_in_hand_cameras: bool = False
    """Whether to render images from eye-in-hand camera(s)."""

    use_camera_xml_cameras: bool = True
    """Whether to use cameras from camera xml."""

    camera_xml_file: str = "/scr/thankyou/autom/demo-aug/models/all-16-cam-transf-from-transform_140_deg_view_coordinate_frame.xml"
    """The path to the camera xml file."""

    camera_xml_parent_body_name: Optional[str] = "robot0_right_hand"
    """The name of the parent body for the camera xml cameras."""

    remove_default_cameras: bool = False
    """Whether to remove default cameras from the xml."""

    save_mask_for_end_effector: bool = False
    """Whether to save a mask for the end effector."""

    save_mask_for_grasped_object: bool = False
    """Whether to save a mask for any object grasped by end effector.
    If nothing is being grasped, this flag should be false."""

    use_target_obj_cameras: bool = False
    """Whether to use cameras that are centered on the target object."""

    target_obj_name: str = None
    """The name of the target object to use for the target object cameras."""

    dataparser_type: Literal["nerfstudio-data", "blender-data"] = "nerfstudio-data"

    def __post_init__(self):
        if self.use_target_obj_cameras and self.target_obj_name is None:
            raise ValueError(
                "target_obj_name must be specified if use_target_obj_cameras is True."
            )


def get_pose(
    eye: torch.Tensor, target: torch.Tensor, up: torch.Tensor
) -> Tuple[np.ndarray, np.ndarray]:
    """Get c2w matrix from eye, target and up vectors."""
    eye = eye.numpy().copy()
    target = target.numpy().copy()
    up = up.numpy().copy()
    rotation_l, translation_l = get_lookat(eye, target, up)
    pose_translation = -np.dot(rotation_l.T, translation_l)
    pose_rotation = rotation_l.T
    assert np.allclose(
        pose_translation, eye
    ), "Pose translation does not match eye position"
    return pose_rotation, pose_translation


def add_cameras_to_env(
    env,
    num_cameras: int,
    target: torch.Tensor,
    up: torch.Tensor,
    radius: float = 0.75,
    image_height: int = 256,
    image_width: int = 256,
    use_eye_in_hand_cameras: bool = False,
    remove_default_cameras: bool = False,
    use_target_obj_cameras: bool = False,
    target_obj_name: Optional[str] = None,
    use_camera_xml: bool = False,
    camera_xml_file: Optional[str] = None,
    camera_xml_parent_body_name: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Add num_cameras cameras to the env.

    Returns a list of per frame info for each camera in the form of a list of dicts
    """
    per_camera_info: Dict[str, Dict[str, Any]] = {}
    initial_mjstate = env.env.sim.get_state().flatten()
    xml = env.env.sim.model.get_xml()

    if remove_default_cameras:
        env.env.num_cameras = 0
        xml = remove_camera_from_xml(xml, "agentview")
        xml = remove_camera_from_xml(xml, "robot0_eye_in_hand")

    all_camera_positions = []
    all_poses = []
    all_quats_wxyz = []
    new_camera_names = []

    num_default_cameras = env.env.num_cameras
    num_eye_in_hand_cameras = 0
    num_target_obj_cameras = 0
    num_xml_cameras = 0

    if use_eye_in_hand_cameras:
        circle_center = torch.tensor([0.0, 0.0, 0.0])
        eye_in_hand_target = torch.tensor([0.0, 0.0, 0.25])
        eye_in_hand_cam_radius = 0.13

        def generate_camera_positions_circle(
            num_cameras: int, radius: float, center: torch.Tensor
        ) -> List[torch.Tensor]:
            """
            Generate num_cameras camera positions on a circle evenly spaced.
            """
            positions = []
            for i in range(num_cameras):
                angle = 2 * np.pi * i / num_cameras
                x = center[0] + radius * np.cos(angle)
                y = center[1] + radius * np.sin(angle)
                z = 0
                positions.append(torch.tensor([x, y, z]))
            return positions

        camera_positions = generate_camera_positions_circle(
            num_cameras, eye_in_hand_cam_radius, circle_center
        )
        num_eye_in_hand_cameras = len(camera_positions)

        poses = [
            get_pose(camera_positions[i], eye_in_hand_target, up)
            for i in range(num_eye_in_hand_cameras)
        ]
        quats_wxyz = [
            [mat2quat(R)[3], mat2quat(R)[0], mat2quat(R)[1], mat2quat(R)[2]]
            for R, t in poses
        ]

        all_poses.extend(poses)
        all_quats_wxyz.extend(quats_wxyz)
        all_camera_positions.extend(camera_positions)

    if use_camera_xml:
        xml = add_camera_xml_to_xml(camera_xml_file, xml, camera_xml_parent_body_name)
        env.env.reset_from_xml_string(
            xml
        )  # add cameras to the underlying simulation env

        # TODO: remove the hardcoding of things used for backwards compatibility of old camera adding code
        # get number of cameras from xml
        num_xml_cameras = 16  # TODO: hardcoded for now; should be read from xml
        # the following data is used only for padding up the camera info because
        # the later camera adding assumes a certain ordering.
        camera_positions = [
            torch.tensor([0.0, 0.0, 0.0]) for _ in range(num_xml_cameras)
        ]
        poses = [
            get_pose(camera_positions[i], target, up) for i in range(num_xml_cameras)
        ]
        quats_wxyz = [
            [mat2quat(R)[3], mat2quat(R)[0], mat2quat(R)[1], mat2quat(R)[2]]
            for R, t in poses
        ]
        # dummy values for now
        all_poses.extend(poses)
        all_quats_wxyz.extend(quats_wxyz)
        all_camera_positions.extend(camera_positions)

    # given an mjcf xml with cameras, add those cameras to xml

    if use_target_obj_cameras:
        circle_center = target + torch.tensor([0.0, 0.0, 0.3])

        # camera_positions = generate_camera_positions(num_cameras, radius, circle_center, z_min=-0.1, z_max=0.8)
        def generate_camera_positions_circle(
            num_cameras: int, radius: float, center: torch.Tensor
        ) -> List[torch.Tensor]:
            """
            Generate num_cameras camera positions on a circle evenly spaced.
            """
            positions = []
            for i in range(num_cameras):
                angle = 2 * np.pi * i / num_cameras
                x = center[0] + radius * np.cos(angle)
                y = center[1] + radius * np.sin(angle)
                z = center[2]
                positions.append(torch.tensor([x, y, z]))
            return positions

        camera_positions = generate_camera_positions_circle(
            num_cameras, radius, circle_center
        )
        num_target_obj_cameras = len(camera_positions)
        poses = [
            get_pose(camera_positions[i], target, up)
            for i in range(num_target_obj_cameras)
        ]
        quats_wxyz = [
            [mat2quat(R)[3], mat2quat(R)[0], mat2quat(R)[1], mat2quat(R)[2]]
            for R, t in poses
        ]

        all_poses.extend(poses)
        all_quats_wxyz.extend(quats_wxyz)
        all_camera_positions.extend(camera_positions)

    num_new_cameras = num_xml_cameras + num_eye_in_hand_cameras + num_target_obj_cameras

    for i in tqdm(range(num_new_cameras), desc="Adding cameras to env"):
        pos = [pos for pos in all_camera_positions[i].numpy()]
        pos_str = [str(x) for x in pos]
        quat_wxyz = all_quats_wxyz[i]
        quat_str = [str(x) for x in quat_wxyz]

        # TODO(klin): these vars are wrong if not both use_camera_xml and use_eye_in_hand_cameras are true
        is_xml_camera_camera = use_camera_xml and i < num_xml_cameras
        is_eye_in_hand_camera = (
            use_eye_in_hand_cameras
            and num_xml_cameras < i < num_xml_cameras + num_eye_in_hand_cameras
        )
        if is_eye_in_hand_camera:
            camera_name = f"eye-in-hand-{i}"
        elif is_xml_camera_camera:
            camera_name = f"camera-sensor-{i}"
        else:
            camera_name = f"target-obj-{i}"

        new_camera_names.append(camera_name)

        if not is_xml_camera_camera:
            # have already added cameras from xml
            xml = add_camera_to_xml(
                xml,
                camera_name=camera_name,
                camera_pos=" ".join(pos_str),
                camera_quat=" ".join(quat_str),
                is_eye_in_hand_camera=is_eye_in_hand_camera,
            )

    if remove_default_cameras:
        env.env.camera_depths = []
        env.env.camera_heights = []
        env.env.camera_widths = []
        env.env.camera_names = []
    else:
        env.env.camera_depths = [True] * num_default_cameras
        env.env.camera_heights = env.env.camera_heights[:num_default_cameras]
        env.env.camera_widths = env.env.camera_widths[:num_default_cameras]
        env.env.camera_names = env.env.camera_names[:num_default_cameras]

    env.env.num_cameras = num_default_cameras + num_new_cameras
    # manually set camera depths to true
    env.env.camera_depths.extend([True] * num_new_cameras)
    env.env.camera_heights.extend([image_height] * num_new_cameras)
    env.env.camera_widths.extend([image_width] * num_new_cameras)

    env.env.camera_names.extend(new_camera_names)
    env.env.camera_heights.extend(env.env._input2list(image_height, num_new_cameras))
    env.env.camera_widths.extend(env.env._input2list(image_width, num_new_cameras))
    env.env.camera_depths.extend(env.env._input2list(True, num_new_cameras))

    env.env.reset_from_xml_string(xml)
    env.env.sim.reset()
    env.env.sim.set_state_from_flattened(initial_mjstate)
    env.env.sim.forward()

    for i in range(env.env.num_cameras):
        intrinsic_matrix = get_camera_intrinsic_matrix(
            env.env.sim,
            env.env.camera_names[i],
            env.env.camera_heights[i],
            env.env.camera_widths[i],
        )
        per_camera_info[env.env.camera_names[i]] = {
            "name": env.env.camera_names[i],
            "height": env.env.camera_heights[i],
            "width": env.env.camera_widths[i],
            "depth": env.env.camera_depths[i],
            "fl_x": intrinsic_matrix[0, 0],
            "fl_y": intrinsic_matrix[1, 1],
            "cx": intrinsic_matrix[0, 2],
            "cy": intrinsic_matrix[1, 2],
        }
        fl_x = intrinsic_matrix[0, 0]
        fl_y = intrinsic_matrix[1, 1]
        if fl_x == fl_y:
            per_camera_info[env.env.camera_names[i]]["camera_angle_x"] = 2 * np.arctan(
                0.5 * image_width / intrinsic_matrix[0, 0]
            )
            # for compatibility with nerfstudio's blender nerf-synthetic dataparser

    return per_camera_info


def render_and_save_images(
    env,
    states: List[Any],
    camera_names: List[str],
    save_dir: str,
    per_camera_info: Dict[str, Dict[str, Any]],
    hdf5_save_group: h5py.Group,
    train_val_split: float = 0.9,
    print_qpos: bool = False,
    timesteps: List[int] = field(default_factory=lambda: []),
    model_file: Optional[str] = None,
    overwrite_all: bool = False,
    use_random_hand_state: bool = False,
    save_mask_for_end_effector: bool = False,
    save_mask_for_grasped_object: bool = False,
    dataparser_type: Literal["nerfstudio-data", "blender-data"] = "nerfstudio-data",
):
    """
    Render and save images from the environment.

    Args:
        env: The environment to render images from
        states: The states to render images from
        model_file: The model file to load. Make sure to either ensure we've the correct model_xml
            (by resetting the env once with the correct xml) or add model_file to the reset dict.
            There is still a discrepancy between resetting env in robosuite vs resetting env in these scripts ...
        dataset_path: the path to the original demo dataset; store the nerf folder here
        save_mask_for_end_effector: whether to save a mask for the end effector (mask masks *out* the end effector)
        save_mask_for_grasped_object: whether to save a mask for the grasped object (mask masks *out* the grasped object)
    """
    save_seg_mask = save_mask_for_end_effector or save_mask_for_grasped_object

    # Loop through each action and render the scene
    for t in timesteps:
        state = states[t]
        # randomly update indices 1 to y
        if use_random_hand_state:
            state[1:7] += np.random.uniform(-0.8, 0.8, 6)

        all_images_timestep: List[Tuple[str, np.ndarray]] = []

        all_image_infos: List[Dict] = []
        all_depth_images: List[Tuple[str, np.ndarray]] = []

        # Create save directory if it doesn't exist
        new_save_dir = Path(f"{save_dir}") / str(t)

        if use_random_hand_state:
            new_save_dir = Path(f"{save_dir}") / f"{t}_random_hand_state"

        Path(new_save_dir).mkdir(parents=True, exist_ok=True)

        if "nerf_timestep_paths" not in hdf5_save_group:
            hdf5_save_group.create_group("nerf_timestep_paths")

        if str(t) in hdf5_save_group["nerf_timestep_paths"]:
            # Ask user if they want to overwrite the existing data
            if not overwrite_all:
                user_response = input(
                    f"Do you want to overwrite the existing nerf timestep path for timestep {t}? (yes/no): "
                )
            else:
                user_response = "yes"

            if user_response.lower() == "yes":
                # If the user wants to overwrite, delete the existing data
                del hdf5_save_group["nerf_timestep_paths"][str(t)]
                hdf5_save_group["nerf_timestep_paths"][str(t)] = str(new_save_dir)
                print(
                    f"Nerf timestep path for timestep {t} has been overwritten in hdf5 {hdf5_save_group}."
                )
            else:
                # If the user doesn't want to overwrite, you can choose to handle this case accordingly.
                print("No changes were made to the existing data.")
        else:
            hdf5_save_group["nerf_timestep_paths"][str(t)] = str(new_save_dir)

        reset_dict = {"states": state}
        if model_file is not None:
            reset_dict["model_file"] = model_file
        init_state = env.reset_to(reset_dict)

        if print_qpos:
            joint_qpos_idxs = (
                np.array(env.env.robots[0]._ref_joint_pos_indexes, dtype=np.uint8) + 1
            )
            gripper_qpos_idxs = np.array(
                [joint_qpos_idxs[-1] + 1, joint_qpos_idxs[-1] + 2], dtype=np.uint8
            )
            print(f"base_pos: {env.env.robots[0].base_pos}")
            print(f"base_quat_xyzw: {env.env.robots[0].base_ori}")
            print(f"joint_qpos: {state[joint_qpos_idxs]}")
            print(f"gripper_qpos: {state[gripper_qpos_idxs]}")
            print(f"init_state['robot0_eef_pos']: {init_state['robot0_eef_pos']}")
            print(f"init_state['robot0_eef_quat']: {init_state['robot0_eef_quat']}")
            task_relev_obj_pos = env.env.sim.data.body_xpos[env.env.cube_body_id]
            task_relev_obj_quat_wxyz = env.env.sim.data.body_xquat[env.env.cube_body_id]
            print(f"task_relev_obj_pos: {task_relev_obj_pos}")
            print(f"task_relev_obj_quat_wxyz: {task_relev_obj_quat_wxyz}")

        # Loop through each camera and render and save the image
        for cam_name in tqdm(camera_names, desc="Rendering images"):
            if "camera-sensor" in cam_name:
                # hardcoded for my cameras - in real, 512 x 512 is probably the resolution I'd use
                intrinsic_matrix = get_camera_intrinsic_matrix(
                    env.env.sim,
                    cam_name,
                    512,
                    512,
                )
                per_camera_info[cam_name] = {
                    "name": cam_name,
                    "height": 512,
                    "width": 512,
                    "depth": True,
                    "fl_x": intrinsic_matrix[0, 0],
                    "fl_y": intrinsic_matrix[1, 1],
                    "cx": intrinsic_matrix[0, 2],
                    "cy": intrinsic_matrix[1, 2],
                }
                fl_x = intrinsic_matrix[0, 0]
                fl_y = intrinsic_matrix[1, 1]
                if fl_x == fl_y:
                    per_camera_info[cam_name]["camera_angle_x"] = 2 * np.arctan(
                        0.5 * 512 / intrinsic_matrix[0, 0]
                    )
                    # for compatibility with nerfstudio's blender nerf-synthetic dataparser

            camera_info = per_camera_info[cam_name]
            cam_h = camera_info["height"] if "height" in camera_info else 512
            cam_w = camera_info["width"] if "width" in camera_info else 512

            rgb, depth = env.env.sim.render(
                camera_name=cam_name,
                height=cam_h,
                width=cam_w,
                depth=camera_info["depth"] if "depth" in camera_info else False,
            )

            if save_seg_mask:
                raw_mjc_seg = env.env.sim.render(
                    camera_name=cam_name,
                    height=cam_h,
                    width=cam_w,
                    depth=False,
                    segmentation=True,
                )
                # see https://github.com/openai/mujoco-py/issues/516 for how to get segs:
                # note that latest robosuite has nicer implementation (instance segmentation, etc.)
                # note: mjc seg renderer may need antialiasing turned *off* to remove artifacts (which mostly are OK)
                # for this case
                # https://github.com/deepmind/dm_control/issues/395
                types = raw_mjc_seg[:, :, 0]
                ids = raw_mjc_seg[:, :, 1]
                geoms = types == const.OBJ_GEOM
                geoms_ids = np.unique(ids[geoms])
                seg_mask = np.ones((cam_h, cam_w), dtype=np.uint8) * 255
                for i in geoms_ids:
                    name = env.env.sim.model.geom_id2name(i)
                    if name is None:
                        # peg's geoms have no name by default
                        continue
                    if save_mask_for_end_effector and (
                        "robot" in name or "gripper" in name
                    ):
                        seg_mask[ids == i] = 0
                    if save_mask_for_grasped_object and "Nut" in name:
                        seg_mask[ids == i] = 0

            # recompute camera extrinsics
            # TODO(klin): unclear why necessary to flip things
            rgb = rgb[::-1]
            depth = depth[::-1]
            if save_seg_mask:
                seg_mask = seg_mask[::-1]

            if (depth == 0).sum() > 0:
                import ipdb

                ipdb.set_trace()
                logging.warning(
                    "Need to handle zero depth case (if mujoco renders zero depth for inf depth)                        "
                    "      for alpha compositing"
                )

            # TODO(klin): use real distance map instead of real depth map!
            real_depth = get_real_depth_map(env.env.sim, depth)
            use_distance_for_depth: bool = False
            if use_distance_for_depth:
                cam_intrinsic_matrix = get_camera_intrinsic_matrix(
                    env.env.sim,
                    cam_name,
                    cam_h,
                    cam_w,
                )
                real_depth = get_real_distance_map(
                    real_depth,
                    cam_intrinsic_matrix[0, 0],
                    cam_intrinsic_matrix[1, 1],
                    cam_intrinsic_matrix[0, 2],
                    cam_intrinsic_matrix[1, 2],
                )

            # convert to np.uint16 and mm
            real_depth = (real_depth * 1000).astype(np.uint16)
            vis_depth = real_depth.copy()
            vis_depth = cv2.normalize(vis_depth, None, 0, 255, cv2.NORM_MINMAX).astype(
                "uint8"
            )

            file_path = f"images/frame_{str(t).zfill(5)}_{cam_name}_rgb.png"
            full_file_path = Path(new_save_dir) / file_path
            depth_file_path = f"images/frame_{str(t).zfill(5)}_{cam_name}_depth.png"
            full_depth_file_path = Path(new_save_dir) / depth_file_path
            visible_depth_file_path = (
                f"images/frame_{str(t).zfill(5)}_{cam_name}_depth_vis.png"
            )
            full_visible_depth_file_path = Path(new_save_dir) / visible_depth_file_path
            seg_mask_file_path = (
                f"images/frame_{str(t).zfill(5)}_{cam_name}_seg_mask.png"
            )
            full_seg_mask_file_path = Path(new_save_dir) / seg_mask_file_path

            # create directory if it doesn't exist
            full_file_path.parent.mkdir(parents=True, exist_ok=True)

            # save normal image
            success = cv2.imwrite(
                str(full_file_path), cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            )
            if not success:
                logging.warning(f"Failed to save image to {full_file_path}")
            success = cv2.imwrite(str(full_depth_file_path), real_depth)
            if not success:
                logging.warning(f"Failed to save depth image to {full_depth_file_path}")
            success = cv2.imwrite(str(full_visible_depth_file_path), vis_depth)
            if not success:
                logging.warning(
                    f"Failed to save depth image to {full_visible_depth_file_path}"
                )
            if save_seg_mask:
                success = cv2.imwrite(str(full_seg_mask_file_path), seg_mask)
                if not success:
                    logging.warning(
                        f"Failed to save segmentation image to {full_seg_mask_file_path}"
                    )

            camera_info = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in camera_info.items()
            }
            # update transform_matrix
            camera_info["transform_matrix"] = get_camera_extrinsic_matrix(
                env.env.sim, cam_name
            ).tolist()
            # Get the camera information for the image
            image_info = {
                "file_path": str(file_path),
                "depth_file_path": str(depth_file_path),
                "visible_depth_file_path": str(visible_depth_file_path),
                "camera_name": cam_name,
                "time": t,
                **camera_info,
            }

            if save_seg_mask:
                image_info["mask_path"] = str(seg_mask_file_path)

            if dataparser_type == "blender-data":
                # using non-nerfstudio dataparaser; use another flag to indicate this
                # loop through image_info: for keys with "path" in them, cut down to just the filename
                for k, v in image_info.items():
                    if "path" in k:
                        image_info[k] = str(Path(v).stem)
            elif dataparser_type == "nerfstudio-data":
                # convert 'height' key to 'h', 'weight' key to 'w'
                image_info["h"] = image_info.pop("height")
                image_info["w"] = image_info.pop("width")

            print(f"image_info: {image_info}")
            all_image_infos.append(image_info)
            all_images_timestep.append((file_path, rgb))

            all_depth_images.append((depth_file_path, depth))

        render_image_collage_from_image_list(
            all_images_timestep, Path(new_save_dir) / f"all_images_time{t}.png"
        )

        per_camera_info[camera_names[0]]
        {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in camera_info.items()
        }

        if dataparser_type == "blender-data":
            random.shuffle(all_image_infos)
            split_index = int(len(all_image_infos) * train_val_split)
            # train_json = single_camera_info.copy()
            # val_json = single_camera_info.copy()
            train_json = {}
            val_json = {}
            train_json["frames"] = all_image_infos[:split_index]
            val_json["frames"] = all_image_infos[split_index:]
            # save to transforms_train.json and transforms_val.json
            with open(Path(new_save_dir) / "transforms_train.json", "w") as f:
                json.dump(train_json, f)
            with open(Path(new_save_dir) / "transforms_val.json", "w") as f:
                json.dump(val_json, f)
            # save to transforms_train.json and transforms_val.json
            with open(Path(new_save_dir) / "transforms_train_src.json", "w") as f:
                json.dump(train_json, f)
            with open(Path(new_save_dir) / "transforms_val_src.json", "w") as f:
                json.dump(val_json, f)
            # save to transforms_test.json for viewer purposes
            with open(Path(new_save_dir) / "transforms_test.json", "w") as f:
                json.dump(val_json, f)
        elif dataparser_type == "nerfstudio-data":
            overall_json = {}
            # overall_json = single_camera_info.copy()
            overall_json["frames"] = all_image_infos
            # Save all image infos to a json file
            with open(Path(new_save_dir) / "transforms.json", "w") as f:
                json.dump(overall_json, f)
        else:
            raise NotImplementedError(f"Unsupported dataparser_type: {dataparser_type}")

        print(f"all_image_infos: {all_image_infos}")
        print(f"saved to {new_save_dir}")
        # save new_save_dir to hdf5 file in dataset_path
        # with open(dataset_path, "w") as f:
        #     pickle.dump(new_save_dir, f)
    # TODO(klin): have UI allowing user to see all images in a panel and then
    # delete incorrect segmentations
    print("hdf5_save_group", hdf5_save_group["nerf_timestep_paths"].keys())
    logging.info(
        f"Finished saving images to hdf5 file; directory for images is {new_save_dir}"
    )
    # render_image_collage_from_image_list(all_images, Path(save_dir) / "all_images.png")
    env.reset()


def generate_images(cfg: Config):
    num_cameras = cfg.num_cameras

    # create environment
    # need to make sure ObsUtils knows which observations are images, but it doesn't matter
    # for playback since observations are unused. Pass a dummy spec here.
    # TODO(klin): maybe remove dummy spec: try removing and checking
    dummy_spec = dict(
        obs=dict(
            low_dim=["robot0_eef_pos"],
            rgb=[],
        ),
    )
    assert (
        cfg.render_depths
    ), "render_depths must be True for current render_and_save_images function"
    ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=dummy_spec)
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=cfg.dataset)
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta, render_offscreen=True, use_image_obs=False
    )

    env_name = env_meta["env_name"]

    f = h5py.File(
        cfg.dataset, "a"
    )  # risky move here: use 'a' instead of 'w'. Safer = create new file but annoying
    demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    for ind, ep in enumerate(demos):
        # prepare initial state to reload from
        states = f["data/{}/states".format(ep)][()]
        reset_dict = dict(states=states[cfg.timesteps[0]])
        if cfg.reset_to_loaded_xml:
            model_file = f["data/{}".format(ep)].attrs["model_file"]
            reset_dict["model"] = model_file
            env.reset_to(reset_dict)
            reset_dict["model"] = model_file
            env.reset_to(reset_dict)
        else:
            print("Likely going to get an error ...")
            # ValueError: could not broadcast input array from shape (7,) into shape (15,)
            env.reset()

        if env_name == "Door":
            camera_target = torch.tensor(env.env._handle_xpos)
            radius = 0.19
        elif env_name == "Lift":
            camera_target = torch.tensor([0.02, 0.02, 0.83])
            radius = cfg.radius
            if cfg.render_pitcher:
                camera_target = torch.tensor([0.2, 0, 1.03])
                radius = 0.19
        elif env_name == "PickPlaceCan":
            radius = 0.19
            camera_target = torch.tensor([0.2, 0, 1.03])
        elif env_name == "NutAssemblySquare":
            if cfg.use_target_obj_cameras:
                if cfg.target_obj_name == "square_peg":
                    peg1_body_id = env.env.sim.model.body_name2id("peg1")
                    camera_target = torch.tensor(
                        env.env.sim.model.body_pos[peg1_body_id]
                    )
                    radius = cfg.radius
                elif cfg.target_obj_name == "square_nut":
                    bbox_min, bbox_max = get_body_bounding_box(
                        env.env.sim, "SquareNut_main"
                    )
                    target = (bbox_min + bbox_max) / 2
                    camera_target = torch.tensor(target)
                    radius = 0.2
                    print(f"target: {target}")
                    print(f"bbox_min: {bbox_min}")
                    print(f"bbox_max: {bbox_max}")
                    print(f"sizes (bbox_max - bbox_min): {bbox_max - bbox_min}")
            else:
                radius = 0.19
                camera_target = torch.tensor([0.2, 0, 1.03])
        else:
            raise NotImplementedError(
                f"Unsupported camera_target for env_name: {env_name}"
            )

        per_camera_info: Dict[str, Dict[str, Any]] = add_cameras_to_env(
            env,
            num_cameras=num_cameras,
            radius=radius,
            up=torch.tensor([0, 0, 1]),
            target=camera_target,
            image_height=cfg.image_height,
            image_width=cfg.image_width,
            use_eye_in_hand_cameras=cfg.use_eye_in_hand_cameras,
            remove_default_cameras=cfg.remove_default_cameras,
            use_target_obj_cameras=cfg.use_target_obj_cameras,
            target_obj_name=cfg.target_obj_name,
            use_camera_xml=cfg.use_camera_xml_cameras,
            camera_xml_file=cfg.camera_xml_file,
            camera_xml_parent_body_name="robot0_right_hand",
        )

        model_file = env.env.sim.model.get_xml()
        reset_dict["model"] = model_file
        env.reset_to(reset_dict)

        render_and_save_images(
            env=env,
            states=states,
            hdf5_save_group=f[f"data/{ep}"],
            camera_names=env.env.camera_names,
            per_camera_info=per_camera_info,
            save_dir=f"{cfg.save_dir}",
            train_val_split=1,
            timesteps=cfg.timesteps,
            model_file=model_file,
            overwrite_all=cfg.overwrite_all,
            save_mask_for_end_effector=cfg.save_mask_for_end_effector,
            save_mask_for_grasped_object=cfg.save_mask_for_grasped_object,
            dataparser_type=cfg.dataparser_type,
        )
        # TODO(klin): handle more than 1 snapshot in time by naming things differently
        if ind == 0:
            break

    f.close()


if __name__ == "__main__":
    tyro.cli(generate_images)
