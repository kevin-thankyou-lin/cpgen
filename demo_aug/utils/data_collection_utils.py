import datetime
import json
import logging
import pathlib
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pprint import pprint
from typing import Any, Dict, List, Optional, Tuple

import cv2
import h5py
import numpy as np
import torch
import tyro

from demo_aug.demo import Demo, TimestepData
from demo_aug.envs.motion_planners.base_motion_planner import RobotEnvConfig
from demo_aug.envs.sphere_nerf_env import EnvConfig
from demo_aug.utils.run_script_utils import retry_on_exception


@retry_on_exception(max_retries=15, retry_delay=1, exceptions=(BlockingIOError,))
def get_env_cfg_from_dataset(dataset_path: pathlib.Path) -> EnvConfig:
    """
    Retrieves EnvConfig from dataset. At first used by toy envs (SphereNeRF, NeRF3DTrace, etc.)

    Args:
        dataset_path (str): path to dataset

    Returns:
        EnvConfig
    """
    dataset_path = dataset_path.expanduser()
    with h5py.File(dataset_path, "r") as f:
        config_json = f["data"].attrs["env_args"]
        config_loaded = json.loads(config_json)
        env_cfg = tyro.extras.from_yaml(EnvConfig, config_loaded["env_kwargs"])
    return env_cfg


def print_hdf5_file_structure(
    file_path: str, print_stats: bool = True, skip_key_prefix: str = ""
):
    with h5py.File(file_path, "r") as hdf_file:
        print("File structure:", file_path)
        print("-" * 50)
        # Recursively print all groups and datasets
        print_hdf5_item_structure(
            hdf_file, "", print_stats=print_stats, skip_key_prefix=skip_key_prefix
        )


def print_hdf5_item_structure(
    item, indent, print_stats: bool = False, skip_key_prefix: str = ""
):
    for key in item.keys():
        if skip_key_prefix != "" and key.startswith(skip_key_prefix):
            continue
        print(indent, end="")
        print(key)
        if (
            isinstance(item[key], h5py.Dataset)
            and print_stats
            and key != "dones"
            and key != "rewards"
        ):
            print(indent + "  " + "Dataset:", key)
            print_dataset_statistics(item[key], indent=indent)
        elif isinstance(item[key], h5py.Group):
            print(indent + "  " + "Group:", key)
            print_hdf5_item_structure(
                item[key],
                indent + "    ",
                print_stats=print_stats,
                skip_key_prefix=skip_key_prefix,
            )


def print_dataset_statistics(dataset, indent="    "):
    print(indent + " " + "Shape:", dataset.shape)
    print(indent + " " + "Data type:", dataset.dtype)


# NOTE: this function isn't very good; used once to convert demo_1 to demo_0 and saved in new path
def convert_hdf5_demo_name(
    demo_path: pathlib.Path,
    old_demo_name: str,
    new_demo_name: str,
    save_path: pathlib.Path,
) -> str:
    with h5py.File(demo_path, "r") as file:
        src_demos = file["data"]

        # Choose the index of the demo you want to save
        demo_index = 0  # Replace with the desired index
        if demo_index < len(src_demos):
            # Create a new HDF5 file to save the selected demo
            with h5py.File(save_path, "w") as tgt_file:
                tgt_file_grp = tgt_file.create_group("data")
                tgt_file_grp_demo = tgt_file_grp.create_group(new_demo_name)
                for k, v in file[f"data/{old_demo_name}"].items():
                    tgt_file_grp_demo.create_dataset(k, data=v[()])
                tgt_file_grp_demo.attrs["model_file"] = file[
                    f"data/{old_demo_name}"
                ].attrs["model_file"]
                for attr_name, attr_value in file["data"].attrs.items():
                    print(f"attr_name: {attr_name}, attr_value: {attr_value}")
                    tgt_file["data"].attrs[attr_name] = attr_value
                    print(
                        f"tgt_file['data'].attrs[attr_name]: {tgt_file['data'].attrs[attr_name]}"
                    )
                # print(f"tgt_file['data/{demo_name}']: {tgt_file['data/{demo_name}']}")

            print(f"Selected demo saved to {save_path}.")


def take_n_demos(
    demo_path: pathlib.Path,
    save_path: pathlib.Path,
    n: int,
    random_sample: bool = False,
):
    """
    Takes n demos from an HDF5 file and saves them to the specified path.

    :param demo_path: The path to the HDF5 file containing the demos.
    :type demo_path: pathlib.Path
    :param save_path: The path to save the selected demos.
    :type save_path: pathlib.Path
    :param n: Number of demos to take.
    :type n: int
    :param random_sample: If True, samples n demos randomly without replacement. Defaults to False.
    :type random_sample: bool
    """
    import random

    with h5py.File(demo_path, "r") as file:
        src_demos = file["data"]
        total_demos = len(src_demos)

        if n > total_demos:
            raise ValueError(f"Requested {n} demos, but only {total_demos} available.")

        # Determine demo indices to extract
        if random_sample:
            selected_indices = random.sample(range(total_demos), n)
        else:
            selected_indices = list(range(n))

        # Save selected demos to new HDF5 file
        with h5py.File(save_path, "w") as tgt_file:
            tgt_file.create_group("data")

            for idx in selected_indices:
                demo_name = list(src_demos.keys())[idx]
                file.copy(f"data/{demo_name}", tgt_file["data"])

            # Copy attributes from data group
            for attr_name, attr_value in src_demos.attrs.items():
                tgt_file["data"].attrs[attr_name] = attr_value

            # Copy attributes from the root file level
            for attr_name, attr_value in file.attrs.items():
                tgt_file.attrs[attr_name] = attr_value

        print(f"Selected {n} demos saved to {save_path}.")


def take_one_demo(
    demo_path: pathlib.Path,
    demo_name: str,
    save_path: pathlib.Path,
    demo_index: int = 0,
):
    """
    Takes one demo from an HDF5 file and saves it to the specified path.

    :param demo_path: The path to the HDF5 file containing the demos.
    :type demo_path: pathlib.Path
    :param save_path: The path to save the selected demo.
    :type save_path: pathlib.Path
    """
    with h5py.File(demo_path, "r") as file:
        src_demos = file["data"]

        # Choose the index of the demo you want to save
        if demo_index < len(src_demos):
            # Create a new HDF5 file to save the selected demo
            with h5py.File(save_path, "w") as tgt_file:
                tgt_file.create_group("data")
                file.copy(f"data/{demo_name}", tgt_file["data"])
                for attr_name, attr_value in file["data"].attrs.items():
                    print(f"attr_name: {attr_name}, attr_value: {attr_value}")
                    tgt_file["data"].attrs[attr_name] = attr_value
                    print(
                        f"tgt_file['data'].attrs[attr_name]: {tgt_file['data'].attrs[attr_name]}"
                    )

            print(f"Selected demo saved to {save_path}.")
        else:
            print(f"Invalid demo index: {demo_index}. No demo found.")


def subsample_demo(
    demo_path: pathlib.Path,
    demo_name: str,
    save_path: pathlib.Path,
    subsample_rate: int = 4,
    convert_to_abs_actions: bool = False,
):
    """
    Takes one demo from an HDF5 file, subsample it and save to the specified path.
    """
    logging.info(f"Subsampling and saving demo {demo_name} to {save_path}")
    with h5py.File(demo_path, "r") as file:
        file["data"][demo_name]

        with h5py.File(save_path, "w") as tgt_file:
            tgt_file.create_group("data")
            file.copy(f"data/{demo_name}", tgt_file["data"])
            for attr_name, attr_value in file["data"].attrs.items():
                print(f"attr_name: {attr_name}, attr_value: {attr_value}")
                tgt_file["data"].attrs[attr_name] = attr_value
                print(
                    f"tgt_file['data'].attrs[attr_name]: {tgt_file['data'].attrs[attr_name]}"
                )
            # subsample actions and states
            # might need to delete the original actions and states first
            orig_actions = tgt_file["data"][demo_name]["actions"]
            orig_states = tgt_file["data"][demo_name]["states"]
            if convert_to_abs_actions:
                from demo_aug.utils.robomimic_utils import (
                    RobomimicAbsoluteActionConverter,
                )

                converter = RobomimicAbsoluteActionConverter(demo_path)
                orig_actions = converter.convert_idx(0)

            del tgt_file["data"][demo_name]["actions"]
            del tgt_file["data"][demo_name]["states"]

            tgt_file["data"][demo_name]["actions"] = orig_actions[
                subsample_rate::subsample_rate
            ]
            tgt_file["data"][demo_name]["states"] = orig_states[
                subsample_rate::subsample_rate
            ]
            tgt_file["data"][demo_name].attrs["subsample_rate"] = subsample_rate
        # check if there are "obs" fields that I need to deal with?
    logging.info(f"Subsampled demo saved to {save_path}.")


def build_robot_env_config(src_demo, j: int):
    robot_obs = src_demo["obs"]
    robot_obs_keys = list(robot_obs.keys())
    gripper_key = [key for key in robot_obs_keys if "gripper" in key][0]
    eef_pos_key = [key for key in robot_obs_keys if "eef_pos" in key][0]
    eef_quat_key = [key for key in robot_obs_keys if "eef_quat" in key][0]
    joint_pos_key = [
        key for key in robot_obs_keys if ("joint_pos" in key or "joint_qpos" in key)
    ][0]

    return RobotEnvConfig(
        robot_joint_qpos=src_demo["obs"][joint_pos_key][()][j],
        robot_gripper_qpos=src_demo["obs"][gripper_key][()][j],
        robot_ee_pos=src_demo["obs"][eef_pos_key][()][j],
        robot_ee_quat_wxyz=np.concatenate(
            (
                [src_demo["obs"][eef_quat_key][()][j][-1]],
                src_demo["obs"][eef_quat_key][()][j][:-1],
            )
        ),
        robot_base_pos=np.array([-0.56, 0.0, 0.912]),
        robot_base_quat_wxyz=np.array([1, 0, 0, 0]),
    )


def recursively_unpack_h5py(obj) -> Dict[str, Any]:
    if isinstance(obj, h5py.Group):
        return {key: recursively_unpack_h5py(obj[key]) for key in obj.keys()}
    elif isinstance(obj, h5py.Dataset):
        data = obj[()]
        if isinstance(data, np.ndarray):
            if data.dtype.kind == "S":  # Binary string
                return np.char.decode(data, "utf-8")
            elif data.dtype.kind == "O":  # Object array, possibly containing strings
                return np.array(
                    [
                        item.decode("utf-8") if isinstance(item, bytes) else item
                        for item in data
                    ]
                )
        elif isinstance(data, bytes):  # Single binary string
            return data.decode("utf-8")
        return data
    else:
        return obj


def load_single_demo(
    file, demo_index: int, demo_path: pathlib.Path, load_images: bool = False
):
    print(f"demo_index: {demo_index}")
    src_demo = file[f"data/demo_{demo_index}"]
    timestep_data = []

    demo_len = len(src_demo["obs"]["agentview_image"])
    for j in range(demo_len):
        # unfortunately the saved hdf5 doesn't contain the object's pose ...
        obs_dct = {}
        for key, value in src_demo["obs"].items():
            if "image" not in key and isinstance(value, h5py.Dataset):
                try:
                    if isinstance(value[j], torch.Tensor):
                        obs_dct[key] = value[j].numpy()
                    else:
                        obs_dct[key] = value[j]
                except Exception as e:
                    import ipdb

                    ipdb.set_trace()
                    print(f"Error processing key {key}: {str(e)}")
                    continue

        # if load images, load the images
        if load_images:
            obs_dct.update(
                {
                    key: np.array(src_demo["obs"][key][j], dtype=np.uint8)
                    for key in src_demo["obs"].keys()
                    if "image" in key
                }
            )

        if src_demo.get(f"timestep_{j}", None) is not None:
            curr_data = recursively_unpack_h5py(src_demo[f"timestep_{j}"])
        else:
            curr_data = {}

        timestep_data.append(
            TimestepData(
                obs=obs_dct,
                action=src_demo["actions"][()][j],
                robot_pose=(
                    src_demo["states"]["robot_pose"][()][j]
                    if "states" in src_demo
                    and isinstance(src_demo["states"], h5py.Group)
                    and "robot_pose" in src_demo["states"]
                    else None
                ),
                target_pose=(
                    src_demo["states"]["target_pose"][()][j]
                    if "states" in src_demo
                    and isinstance(src_demo["states"], h5py.Group)
                    and "target_pose" in src_demo["states"]
                    else None
                ),
                mujoco_state=(
                    src_demo["states"][()][j]
                    if "states" in src_demo and len(src_demo["states"][()]) > j
                    else None
                ),
                robot_env_cfg=build_robot_env_config(src_demo, j),
                mujoco_model_xml=src_demo.attrs["model_file"]
                if "model_file" in src_demo.attrs
                else None,
                objs_transf_type_seq=curr_data.get("objs_transf_type_seq", None),
                objs_transf_params_seq=curr_data.get("objs_transf_params_seq", None),
                objs_transf_name_seq=curr_data.get("objs_transf_name_seq", None),
            )
        )

    constraint_infos = (
        src_demo["constraint_infos"][()] if "constraint_timesteps" in src_demo else None
    )
    print(f"demo_index end: {demo_index}")
    return Demo(
        name=f"demo_{demo_index}",
        demo_path=demo_path,
        timestep_data=timestep_data,
        constraint_infos=constraint_infos,
    )


@retry_on_exception(max_retries=15, retry_delay=1, exceptions=(BlockingIOError,))
def load_demos(
    demo_path: str,
    max_demos: int = -1,
    start_idx: int = 0,
    debug_mode: bool = False,
    load_images: bool = False,
    use_threads: bool = False,
) -> List[Demo]:
    """Load demos sequentially from the given HDF5 file."""
    start = time.time()
    src_demo_objs = []
    with h5py.File(demo_path, "r") as file:
        src_demos = file["data"]
        num_demos = min(len(src_demos), max_demos if max_demos > 0 else len(src_demos))
        if debug_mode:
            num_demos = min(num_demos, 2)

        if use_threads:
            num_threads = 16
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                results = executor.map(
                    lambda i: load_single_demo(
                        file, i, demo_path, load_images=load_images
                    ),
                    range(start_idx, start_idx + num_demos),
                )
                src_demo_objs.extend(results)

        else:
            for i in range(start_idx, start_idx + num_demos):
                demo = load_single_demo(file, i, demo_path, load_images=load_images)
                src_demo_objs.append(demo)

    end = time.time()
    print(f"Loaded demos in {end - start:.2f} seconds.")
    return src_demo_objs


@retry_on_exception(max_retries=15, retry_delay=1, exceptions=(BlockingIOError,))
def save_demos(
    demos: List[Demo],
    save_path: pathlib.Path,
    env_info: str,
    env_cfg: Optional[Dict] = None,
    save_quat_with_x_positve: bool = True,
    camera_names: List[str] = [],
):
    """Save demos to disk.

    Camera names: list of camera names to save images for. If empty, no images are saved.
    """

    def process_images(demo: Demo, ep_obs_group: h5py.Group, camera_names: List[str]):
        """
        Process images, stored in demo, for each camera and timestep. Then, save to the hdf5 file within ep_obs_group.
        """
        for camera_name in camera_names:
            # Create the key names
            key_name = f"{camera_name}_image"

            # Process images for the current camera
            obs_images = np.array(
                [timestep.obs[key_name] for timestep in demo.timestep_data]
            )
            if obs_images.shape[-1] != 3:
                obs_images = (obs_images * 255).astype(np.uint8).transpose(0, 2, 3, 1)

            # Remove old dataset if exists and create a new one
            if key_name in ep_obs_group:
                del ep_obs_group[key_name]
            ep_obs_group.create_dataset(key_name, data=obs_images)

    logging.info(f"Saving demos to {save_path}")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    demo_type: str = "robomimic"
    if demo_type == "robomimic":
        assert env_cfg is not None, "env_cfg must be specified for robomimic demos"
    with h5py.File(save_path, "a") as f:
        # create "data" group if it doesn't exist
        if "data" not in f:
            grp = f.create_group("data")
        else:
            # ask user if they want to overwrite
            if input(f"Overwrite existing data at {save_path}? (y/n) ").lower() != "y":
                return
            grp = f["data"]

        # set attributes of "data" group
        grp.attrs["date"] = "2023-04-19"  # or use the actual date
        grp.attrs["time"] = "00:00:00"  # or use the actual time
        grp.attrs["env"] = demos[0].name
        for i, demo in enumerate(demos):
            # check if demo already exists
            if "demo_{}".format(i) in list(grp.keys()):
                ep_data_grp = grp["demo_{}".format(i)]
            else:
                ep_data_grp = grp.create_group("demo_{}".format(i))

            for t, timestep in enumerate(demo.timestep_data):
                # if already exists, delete
                if "timestep_{}".format(t) in list(ep_data_grp.keys()):
                    del ep_data_grp["timestep_{}".format(t)]

                # Create group for each timestep
                timestep_grp = ep_data_grp.create_group(f"timestep_{t}")

                # Create sub-groups for each data type
                type_grp = timestep_grp.create_group("objs_transf_type_seq")
                params_grp = timestep_grp.create_group("objs_transf_params_seq")
                name_grp = timestep_grp.create_group("objs_transf_name_seq")

                # Flatten and store the data
                for key, transf_type_seq in timestep.objs_transf_type_seq.items():
                    var_len_str_dt = h5py.special_dtype(vlen=str)
                    type_dataset = type_grp.create_dataset(
                        key, (len(transf_type_seq)), dtype=var_len_str_dt
                    )
                    type_dataset[:] = [val.value for val in transf_type_seq]

                for key, transf_params_seq in timestep.objs_transf_params_seq.items():
                    # transf_params_seq is a list of dicts, each dict is [str, np.ndarray]
                    sub_params_grp = params_grp.create_group(key)
                    for k, params_dict in enumerate(transf_params_seq):
                        for sub_key, sub_value in params_dict.items():
                            sub_key_with_seq = f"{k}:{sub_key}"
                            sub_params_grp.create_dataset(
                                sub_key_with_seq, data=np.array(sub_value)
                            )

                for key, transf_name_seq in timestep.objs_transf_name_seq.items():
                    var_len_str_dt = h5py.special_dtype(vlen=str)
                    name_dataset = name_grp.create_dataset(
                        key, (len(transf_name_seq)), dtype=var_len_str_dt
                    )
                    name_dataset[:] = transf_name_seq

            states = demo.timestep_data[0].mujoco_state
            # write datasets for actions, observations, robot poses, and auxiliary images
            # check if dataset already exists
            if "actions" in ep_data_grp:
                del ep_data_grp["actions"]

            # TODO(klin): decide on using tensors as the data type: when loading data, load to cpu?
            actions = np.array(
                [np.array(timestep.action) for timestep in demo.timestep_data]
            )
            ep_data_grp.create_dataset("actions", data=actions)

            if "obs" in ep_data_grp:
                ep_obs_group = ep_data_grp["obs"]
            else:
                ep_obs_group = ep_data_grp.create_group("obs")

            if "action_alt_rep" in ep_data_grp:
                del ep_data_grp["action_alt_rep"]
            ep_action_alt_rep_group = ep_data_grp.create_group("action_alt_rep")
            ep_action_alt_rep_group.create_dataset(
                "action_abs",
                data=np.array(
                    [
                        timestep.action_alt_rep["action_abs"]
                        for timestep in demo.timestep_data
                    ]
                ),
            )
            ep_action_alt_rep_group.create_dataset(
                "action_delta_world",
                data=np.array(
                    [
                        timestep.action_alt_rep["action_delta_world"]
                        for timestep in demo.timestep_data
                    ]
                ),
            )

            if "states" in ep_data_grp:
                del ep_data_grp["states"]

            if states is not None:
                ep_data_grp.create_dataset("states", data=np.array([states]))

            process_images(demo, ep_obs_group, camera_names)

            if demo_type == "robomimic":
                obs_robo0_eef_pos = np.array(
                    [
                        timestep.obs["robot_ee_pos_action_frame_world"]
                        for timestep in demo.timestep_data
                    ]
                )
                obs_robo0_eef_quat = np.array(
                    [
                        np.concatenate(
                            [
                                timestep.obs["robot_ee_quat_wxyz_action_frame_world"][
                                    1:
                                ],
                                [
                                    timestep.obs[
                                        "robot_ee_quat_wxyz_action_frame_world"
                                    ][0]
                                ],
                            ]
                        )
                        for timestep in demo.timestep_data
                    ]
                )
                if save_quat_with_x_positve:
                    # loop through and replace element with element *-1 if first entry of quat is negative
                    for i in range(obs_robo0_eef_quat.shape[0]):
                        if obs_robo0_eef_quat[i, 0] < 0:
                            obs_robo0_eef_quat[i] *= -1

                obs_robo0_eef_pos_eef_site = np.array(
                    [
                        timestep.obs["robot_ee_pos_obs_frame_world"]
                        for timestep in demo.timestep_data
                    ]
                )
                obs_robo0_eef_quat_eef_site = np.array(
                    [
                        np.concatenate(
                            [
                                timestep.obs["robot_ee_quat_wxyz_obs_frame_world"][1:],
                                [timestep.obs["robot_ee_quat_wxyz_obs_frame_world"][0]],
                            ]
                        )
                        for timestep in demo.timestep_data
                    ]
                )
                if save_quat_with_x_positve:
                    # loop through and replace element with element *-1 if first entry of quat is negative
                    for i in range(obs_robo0_eef_quat_eef_site.shape[0]):
                        if obs_robo0_eef_quat_eef_site[i, 0] < 0:
                            obs_robo0_eef_quat_eef_site[i] *= -1

                # gripper site
                if "robot0_eef_pos_action_frame" in ep_obs_group:
                    del ep_obs_group["robot0_eef_pos_action_frame"]
                ep_obs_group.create_dataset(
                    "robot0_eef_pos_action_frame", data=np.array(obs_robo0_eef_pos)
                )
                if "robot0_eef_quat_action_frame" in ep_obs_group:
                    del ep_obs_group["robot0_eef_quat_action_frame"]
                ep_obs_group.create_dataset(
                    "robot0_eef_quat_action_frame", data=np.array(obs_robo0_eef_quat)
                )  # in xyzw

                # eef site
                if "robot0_eef_pos" in ep_obs_group:
                    del ep_obs_group["robot0_eef_pos"]
                ep_obs_group.create_dataset(
                    "robot0_eef_pos", data=np.array(obs_robo0_eef_pos_eef_site)
                )
                if "robot0_eef_quat" in ep_obs_group:
                    del ep_obs_group["robot0_eef_quat"]
                ep_obs_group.create_dataset(
                    "robot0_eef_quat", data=np.array(obs_robo0_eef_quat_eef_site)
                )

                if "robot0_gripper_qpos" in ep_obs_group:
                    del ep_obs_group["robot0_gripper_qpos"]
                ep_obs_group.create_dataset(
                    "robot0_gripper_qpos",
                    data=np.array(
                        [
                            timestep.obs["robot_gripper_qpos"]
                            for timestep in demo.timestep_data
                        ]
                    ),
                )
                if "robot0_joint_qpos" in ep_obs_group:
                    del ep_obs_group["robot0_joint_qpos"]
                ep_obs_group.create_dataset(
                    "robot0_joint_qpos",
                    data=np.array(
                        [
                            timestep.obs["robot_joint_qpos"]
                            for timestep in demo.timestep_data
                        ]
                    ),
                )

            if demo.timestep_data[0].mujoco_model_xml is not None:
                ep_data_grp.attrs["model_file"] = demo.timestep_data[0].mujoco_model_xml

        # write dataset attributes (metadata)
        now = datetime.datetime.now()
        grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
        grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
        # grp.attrs["repository_version"] = suite.__version__
        grp.attrs["env"] = demos[0].name
        grp.attrs["env_info"] = env_info
        # commenting because real world env currently doesn't have a legit env_cfg to store
        grp.attrs["env_args"] = json.dumps(
            env_cfg
        )  # re-comment if using real world env


def convert_nerf_env_dataset(dataset_path: pathlib.Path):
    """
    Converts a dataset collected using nerf_env into an hdf5 compatible with robomimic.

    Args:
        dataset_path (str): path to the input hdf5 dataset

    Returns:
        None
    """
    f = h5py.File(dataset_path, "a")  # edit mode

    # store env meta
    env_name = f["data"].attrs["env"]
    env_info = f["data"].attrs["env_info"]

    env_meta = dict(
        type="nerf_env",
        env_name=env_name,
        env_kwargs=env_info,
    )
    if "env_args" in f["data"].attrs:
        del f["data"].attrs["env_args"]
    f["data"].attrs["env_args"] = json.dumps(env_meta, indent=4)

    print("====== Stored env info ======")
    pprint(asdict(tyro.extras.from_yaml(EnvConfig, env_info)))

    # store metadata about number of samples
    total_samples = 0
    for ep in f["data"]:
        # add "num_samples" into per-episode metadata
        if "num_samples" in f["data/{}".format(ep)].attrs:
            del f["data/{}".format(ep)].attrs["num_samples"]
        n_sample = f["data/{}/actions".format(ep)].shape[0]
        f["data/{}".format(ep)].attrs["num_samples"] = n_sample
        total_samples += n_sample

    # add total samples to global metadata
    if "total" in f["data"].attrs:
        del f["data"].attrs["total"]
    f["data"].attrs["total"] = total_samples
    print("====== Stored total samples ======")
    print(f["data"].attrs["total"])
    print("====== Stored shape of data/agentview_image from single episode ======")
    print(f["data/demo_0/obs/agentview_image"].shape)
    f.close()


def resize_images_in_dataset(
    input_file_path: str,
    output_file_path: str,
    image_key_resizing_map: Dict[str, Tuple[int, int]],
) -> None:
    logging.info(f"Starting to process file: {input_file_path}")
    logging.info(f"Output will be saved to: {output_file_path}")
    logging.info(f"Processing keys and resolutions: {image_key_resizing_map}")

    with h5py.File(input_file_path, "r") as input_file, h5py.File(
        output_file_path, "w"
    ) as output_file:
        # Copy all attributes from the input file to the output file
        for attr_name, attr_value in input_file.attrs.items():
            output_file.attrs[attr_name] = attr_value

        def process_item(name: str, obj: h5py.HLObject) -> None:
            if isinstance(obj, h5py.Group):
                output_group = output_file.create_group(name)
                for attr_name, attr_value in obj.attrs.items():
                    output_group.attrs[attr_name] = attr_value
            elif isinstance(obj, h5py.Dataset):
                if "image" in name and any(
                    key in name for key in image_key_resizing_map
                ):
                    original_images = obj[()]
                    if len(original_images) == 0:
                        logging.warning(f"Dataset {name} is empty. Skipping.")
                        return

                    for key, new_resolution in image_key_resizing_map.items():
                        if key in name:
                            original_resolution = original_images[0].shape[:2]
                            logging.info(
                                f"Resizing dataset: {name} from {original_resolution} to {new_resolution}"
                            )
                            print(f"original_images.shape: {original_images.shape}")
                            resized_images = np.array(
                                [
                                    cv2.resize(
                                        img, (new_resolution[1], new_resolution[0])
                                    )  # cv2.resize takes (width, height)
                                    for img in original_images
                                ]
                            )
                            print(f"resized_images.shape: {resized_images.shape}")
                            output_name = name
                            output_file.create_dataset(
                                output_name, data=resized_images, compression="gzip"
                            )
                            logging.info(
                                f"Resized {len(resized_images)} images in dataset {output_name}"
                            )
                else:
                    output_file.create_dataset(name, data=obj[()], compression="gzip")

        # Process all items in the input file
        input_file.visititems(process_item)

    # Verify the output file
    with h5py.File(output_file_path, "r") as verify_file:
        if len(verify_file.keys()) == 0:
            logging.error("Output file is empty. No datasets were created.")
        else:
            logging.info(
                f"Output file created successfully with {len(verify_file.keys())} top-level items."
            )
            logging.info(f"Top-level items: {list(verify_file.keys())}")

    logging.info(f"Processing complete. Output saved to {output_file_path}")


def rename_keys(
    input_file_path: str, output_file_path: str, old_to_new_names: Dict[str, str]
):
    with h5py.File(input_file_path, "r") as input_file, h5py.File(
        output_file_path, "w"
    ) as output_file:

        def copy_rename(name, obj):
            for old_name, new_name in old_to_new_names.items():
                if old_name in name:
                    new_name = name.replace(old_name, new_name)
                    break
            else:
                new_name = name

            if isinstance(obj, h5py.Dataset):
                output_file.create_dataset(new_name, data=obj[:], dtype=obj.dtype)
                # Copy attributes
                for key, value in obj.attrs.items():
                    output_file[new_name].attrs[key] = value
            elif isinstance(obj, h5py.Group):
                output_file.create_group(new_name)
                # Copy attributes
                for key, value in obj.attrs.items():
                    output_file[new_name].attrs[key] = value

        input_file.visititems(copy_rename)
