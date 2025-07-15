"""
A script to collect a batch of human demonstrations that can be used
to generate a learning curriculum (see `demo_learning_curriculum.py`).

The demonstrations can be played back using the `playback_demonstrations_from_pkl.py`
script.
"""

import datetime
import json
import os
import pathlib
import shutil
import time
from dataclasses import asdict, dataclass, field
from glob import glob
from pprint import pprint

import h5py
import imageio
import numpy as np
import tyro
from PIL import Image
from tqdm import tqdm

import wandb
from demo_aug.envs.base_env import EnvConfig, EnvType
from demo_aug.envs.nerf_3d_trace_env import NeRF3DTraceEnv
from demo_aug.envs.sphere_nerf_env import SphereNeRFEnv
from demo_aug.envs.wrapper.data_collection_wrapper import (
    DataCollectionWrapper,  # , VisualizationWrapper
)


@dataclass
class DatasetGenerationConfig:
    save_dir: pathlib.Path = pathlib.Path("data/nerf_images")
    temp_dir: pathlib.Path = pathlib.Path("~/temp").expanduser()
    file_name: str = "demo.hdf5"
    episodes: int = 20
    task_completion_hold_count_goal: int = 12
    exp_name: str = "sphere_rand_robot_fixed_target_fixed_x_plane"
    env_cfg: EnvConfig = field(default_factory=lambda: EnvConfig())


def collect_expert_trajectory(
    env: SphereNeRFEnv,
    task_completion_hold_count_goal: int,
):
    """
    Use the device (keyboard or SpaceNav 3D mouse) to collect a demonstration.
    The rollout trajectory is saved to files in npz format.
    Modify the DataCollectionWrapper wrapper to add new fields or change data formats.

    Args:
        env (MujocoEnv): environment to control
        arms (str): which arm to control (eg bimanual) 'right' or 'left'
        env_configuration (str): specified environment configuration
    """
    env.reset()
    env.render()

    task_completion_hold_count = (
        -1
    )  # counter to collect task_completion_hold_count_goal timesteps

    # Loop until we get a reset from the input or the task completes
    while True:
        action = env.optimal_action()
        _, _, done, info = env.step(action)
        env.render()

        # Also break if we complete the task
        if task_completion_hold_count == 0:
            break

        # state machine to check for having a success for 8 consecutive timesteps
        if done:
            if task_completion_hold_count > 0:
                task_completion_hold_count -= 1  # latched state, decrement count
            else:
                task_completion_hold_count = task_completion_hold_count_goal  # reset count on first success timestep
        else:
            task_completion_hold_count = -1  # null the counter if there's no success

    # cleanup for end of data collection episodes
    env.close()


def gather_demonstrations_as_hdf5(
    directory: pathlib.Path, out_dir: pathlib.Path, env_info: str
) -> None:
    """
    Gathers the demonstrations saved in @directory into a
    single hdf5 file.

    The strucure of the hdf5 file is as follows.

    data (group)
        date (attribute) - date of collection
        time (attribute) - time of collection
        env (attribute) - environment name on which demos were collected

        demo1 (group) - every demonstration has a group
            states (group) - group of states for resetting environment to these states
            actions (dataset) - actions applied during demonstration
            obs (group) - observations from the environment
            rewards (dataset) - rewards from the environment
            dones (dataset) - dones from the environment
            infos (dataset) - infos from the environment

        demo2 (group)
        ...

    Args:
        directory (str): Path to the directory containing raw demonstrations.
        out_dir (str): Path to where to store the hdf5 file.
        env_info (str): JSON-encoded string containing environment information,
            including controller and robot info
    """
    hdf5_path = os.path.join(out_dir, "demo.hdf5")
    f = h5py.File(hdf5_path, "w")

    # store some metadata in the attributes of one group
    grp = f.create_group("data")

    num_eps = 0
    env_name = None  # will get populated at some point

    for ep_directory in os.listdir(directory):
        state_paths = os.path.join(directory, ep_directory, "state_*.npz")
        if len(glob(state_paths)) == 0:
            continue

        actions = []
        rewards = []
        dones = []
        obs_agentview_images = []
        next_obs_agentview_images = []
        obs_c2ws = []
        next_obs_c2ws = []
        robot_poses = []
        target_poses = []
        for state_file in sorted(glob(state_paths)):
            dic = np.load(state_file, allow_pickle=True)
            env_name = str(dic["env"])
            robot_poses.extend(dic["states"].item()["robot_pose"])
            target_poses.extend(dic["states"].item()["target_pose"])
            for ai in dic["action_infos"]:
                actions.append(ai["actions"])

            # new stuff
            rewards.extend(dic["rewards"])
            dones.extend(dic["dones"])
            obs_agentview_images.extend(dic["obs"].item()["agentview_image"])
            next_obs_agentview_images.extend(dic["obs"].item()["agentview_image"])
            obs_c2ws.extend(dic["obs"].item()["agentview_c2w"])
            next_obs_c2ws.extend(dic["obs"].item()["agentview_c2w"])

        obs_agentview_images = obs_agentview_images[:-1]
        next_obs_agentview_images = next_obs_agentview_images[1:]
        obs_c2ws = obs_c2ws[:-1]
        next_obs_c2ws = next_obs_c2ws[1:]

        # Delete the last state. This is because when the DataCollector wrapper
        # recorded the states and atask_completion_hold_count_goalctions,
        # the states were recorded AFTER playing that action,
        # so we end up with an extra state at the end. Isn't this perfect? the next state is next_obs
        assert (
            len(actions)
            == len(obs_c2ws)
            == len(next_obs_c2ws)
            == len(obs_agentview_images)
            == len(next_obs_agentview_images)
        ), "Lengths don't match!"

        ep_data_grp = grp.create_group("demo_{}".format(num_eps))

        # write datasets for states, actions, observations, next_observations, rewards, dones
        ep_data_grp.create_dataset("actions", data=np.array(actions))
        ep_data_grp.create_dataset("rewards", data=np.array(rewards))
        ep_data_grp.create_dataset("dones", data=np.array(dones))

        ep_obs_group = ep_data_grp.create_group("obs")
        ep_data_grp.create_group("next_obs")
        ep_states_group = ep_data_grp.create_group("states")

        # # camera intrinsics
        ep_obs_group.attrs["agentview_cam_fl_x"] = dic["obs"].item()[
            "agentview_cam_fl_x"
        ]
        ep_obs_group.attrs["agentview_cam_fl_y"] = dic["obs"].item()[
            "agentview_cam_fl_y"
        ]
        ep_obs_group.attrs["agentview_cam_cx"] = dic["obs"].item()["agentview_cam_cx"]
        ep_obs_group.attrs["agentview_cam_cy"] = dic["obs"].item()["agentview_cam_cy"]
        # camera extrinsics
        ep_obs_group.create_dataset("agentview_c2w", data=np.array(obs_c2ws))
        # images
        obs_agentview_images = (
            (np.array(obs_agentview_images) * 255)
            .astype(np.uint8)
            .transpose(0, 2, 3, 1)
        )
        # obs_agentview_images = np.array(obs_agentview_images)
        ep_obs_group.create_dataset("agentview_image", data=obs_agentview_images)

        ep_states_group.create_dataset("robot_pose", data=np.array(robot_poses))
        ep_states_group.create_dataset("target_pose", data=np.array(target_poses))

        num_eps += 1

    # write dataset attributes (metadata)
    now = datetime.datetime.now()
    grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
    # grp.attrs["repository_version"] = suite.__version__
    grp.attrs["env"] = env_name
    grp.attrs["env_info"] = env_info

    f.close()


def generate_dataset(dataset_generation_cfg: DatasetGenerationConfig) -> None:
    t1, t2 = str(time.time()).split(".")

    wandb_exp_name = (
        f"{dataset_generation_cfg.exp_name}_{dataset_generation_cfg.episodes}"
        f"ep_succ_hold_{dataset_generation_cfg.task_completion_hold_count_goal}_{t1}_{t2}"
    )
    wandb.init(config=dataset_generation_cfg, name=wandb_exp_name)

    # TODO(klin) propably have a make env function or EnvFactory pattern?
    if dataset_generation_cfg.env_cfg.env_type == EnvType.SPHERE:
        env = SphereNeRFEnv(dataset_generation_cfg.env_cfg)
    elif dataset_generation_cfg.env_cfg.env_type == EnvType.NERF_3D_TRACE:
        env = NeRF3DTraceEnv(dataset_generation_cfg.env_cfg)
    else:
        raise NotImplementedError(
            f"{dataset_generation_cfg.env_cfg.env_type} not implemented"
        )

    # elif dataset_generation_cfg.env_cfg.env_type == "trac":
    tmp_directory = dataset_generation_cfg.temp_dir / f"{t1}_{t2}"

    # wrap the environment with data collection wrapper
    env = DataCollectionWrapper(env, tmp_directory)

    # make a new timestamped directory
    new_dir = dataset_generation_cfg.save_dir / wandb_exp_name
    pathlib.Path(new_dir).mkdir(parents=True, exist_ok=True)
    print("Saving data to {}".format(new_dir))

    env_info = tyro.extras.to_yaml(dataset_generation_cfg.env_cfg)

    for _ in tqdm(
        range(dataset_generation_cfg.episodes), desc="Collecting expert trajectories"
    ):
        collect_expert_trajectory(
            env, dataset_generation_cfg.task_completion_hold_count_goal
        )
    gather_demonstrations_as_hdf5(tmp_directory, new_dir, env_info)

    create_gifs_from_h5py_file(new_dir / dataset_generation_cfg.file_name, new_dir)
    convert_nerf_env_dataset(new_dir / dataset_generation_cfg.file_name)

    # delete the temporary directory
    shutil.rmtree(tmp_directory)


def create_gifs_from_h5py_file(file_path: pathlib.Path, gif_dir: pathlib.Path) -> None:
    with h5py.File(file_path, "r") as f:
        demo_groups = [
            g for g in f["data"].values() if g.name.startswith("/data/demo_")
        ]
        for i, demo_group in enumerate(demo_groups):
            obs_dataset = demo_group["obs/agentview_image"]
            obs_images = np.array(obs_dataset)

            images = []
            for j in range(obs_images.shape[0]):
                img = Image.fromarray(obs_images[j])
                images.append(img)

            gif_path = f"{gif_dir}/demo_{i}.gif"
            print(f"Saving gif to {gif_path}")
            imageio.mimsave(gif_path, images, duration=0.1)


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


def print_hdf5_tree(hdf5_path: str) -> None:
    """Print the tree structure of an hdf5 file."""

    def print_attrs(name, obj):
        print(name)
        if isinstance(obj, h5py.Dataset):
            print("Type:", type(obj))
            print("Shape:", obj.shape)
            print("Data type:", obj.dtype)

    with h5py.File(hdf5_path, "r") as f:
        f.visititems(print_attrs)


if __name__ == "__main__":
    tyro.cli(generate_dataset)
