"""
A script to visualize dataset trajectories by loading the simulation states
one by one or loading the first state and playing actions back open-loop.
The script can generate videos as well, by rendering simulation frames
during playback. The videos can also be generated using the image observations
in the dataset (this is useful for real-robot datasets) by using the
--use-obs argument.

Args:
    dataset (str): path to hdf5 dataset

    filter_key (str): if provided, use the subset of trajectories
        in the file that correspond to this filter key

    n (int): if provided, stop after n trajectories are processed

    use-obs (bool): if flag is provided, visualize trajectories with dataset
        image observations instead of simulator

    use-actions (bool): if flag is provided, use open-loop action playback
        instead of loading sim states

    render (bool): if flag is provided, use on-screen rendering during playback

    video_path (str): if provided, render trajectories to this video file path

    video_skip (int): render frames to a video every @video_skip steps

    render_image_names (str or [str]): camera name(s) / image observation(s) to
        use for rendering on-screen or to video

    first (bool): if flag is provided, use first frame of each episode for playback
        instead of the entire episode. Useful for visualizing task initializations.

Example usage below:

    # force simulation states one by one, and render agentview and wrist view cameras to video
    python playback_dataset.py --dataset /path/to/dataset.hdf5 \
        --render_image_names agentview robot0_eye_in_hand \
        --video_path /tmp/playback_dataset.mp4

    # playback the actions in the dataset, and render agentview camera during playback to video
    python playback_dataset.py --dataset /path/to/dataset.hdf5 \
        --use-actions --render_image_names agentview \
        --video_path /tmp/playback_dataset_with_actions.mp4

    # use the observations stored in the dataset to render videos of the dataset trajectories
    python playback_dataset.py --dataset /path/to/dataset.hdf5 \
        --use-obs --render_image_names agentview_image \
        --video_path /tmp/obs_trajectory.mp4

    # visualize depth observations along with image observations
    python playback_dataset.py --dataset /path/to/dataset.hdf5 \
        --use-obs --render_image_names agentview_image \
        --render_depth_names agentview_depth \
        --video_path /tmp/obs_trajectory.mp4

    # visualize initial states in the demonstration data
    python playback_dataset.py --dataset /path/to/dataset.hdf5 \
        --first --render_image_names agentview \
        --video_path /tmp/dataset_task_inits.mp4
"""

import argparse
import json
from collections import defaultdict
from types import SimpleNamespace

import h5py
import imageio
import numpy as np
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.envs.env_base import EnvBase, EnvType

# Define default cameras to use for each env type
DEFAULT_CAMERAS = {
    EnvType.ROBOSUITE_TYPE: ["agentview"],
    EnvType.IG_MOMART_TYPE: ["rgb"],
    EnvType.GYM_TYPE: ValueError("No camera names supported for gym type env!"),
}


def playback_trajectory_with_env(
    env,
    initial_state,
    states,
    actions=None,
    render=False,
    video_writer=None,
    video_skip=5,
    camera_names=None,
    first=False,
    action_noise_std: float = 0.02,
    constraint_timesteps: list = None,
    noise_schedule_decay_timesteps: int = 6,
    min_noise_scale: float = 0.25,
    max_noise_scale: float = 1.0,
):
    """
    Helper function to playback a single trajectory using the simulator environment.
    If @actions are not None, it will play them open-loop after loading the initial state.
    Otherwise, @states are loaded one by one.

    Args:
        env (instance of EnvBase): environment
        initial_state (dict): initial simulation state to load
        states (np.array): array of simulation states to load
        actions (np.array): if provided, play actions back open-loop instead of using @states
        render (bool): if True, render on-screen
        video_writer (imageio writer): video writer
        video_skip (int): determines rate at which environment frames are written to video
        camera_names (list): determines which camera(s) are used for rendering. Pass more than
            one to output a video with multiple camera views concatenated horizontally.
        first (bool): if True, only use the first frame of each episode.
    """
    recorded_states = []
    recorded_actions = []
    rec_obs = defaultdict(list)

    assert isinstance(env, EnvBase)

    write_video = video_writer is not None
    video_count = 0
    assert not (render and write_video)

    # load the initial state
    env.reset()
    env.reset_to(initial_state)

    traj_len = states.shape[0]
    action_playback = actions is not None
    if action_playback:
        assert states.shape[0] == actions.shape[0]

    recorded_states.append(env.get_state()["states"])
    for i in range(traj_len):
        if action_playback:
            # noisy_action = actions[i] + np.random.normal(0, action_noise_std, actions[i].shape)

            effective_scale = max_noise_scale
            if constraint_timesteps and len(constraint_timesteps) > 0:
                dmin = min(abs(i - t) for t in constraint_timesteps)
                if dmin < noise_schedule_decay_timesteps:
                    effective_scale = min_noise_scale + (
                        max_noise_scale - min_noise_scale
                    ) * (dmin / noise_schedule_decay_timesteps)
            effective_std = action_noise_std * effective_scale
            noisy_action = actions[i] + np.random.normal(
                0, effective_std, actions[i].shape
            )
            recorded_actions.append(noisy_action)
            recorded_states.append(env.get_state()["states"])
            obs, reward, done, info = env.step(noisy_action)
            for k, v in obs.items():
                rec_obs[k].append(v)

            if i < traj_len - 1:
                # check whether the actions deterministically lead to the same recorded states
                state_playback = env.get_state()["states"]
                if not np.all(np.equal(states[i + 1], state_playback)):
                    err = np.linalg.norm(states[i + 1] - state_playback)
                    print("warning: playback diverged by {} at step {}".format(err, i))
        else:
            env.reset_to({"states": states[i]})

        # on-screen render
        if render:
            env.render(mode="human", camera_name=camera_names[0])

        # video render
        if write_video:
            if video_count % video_skip == 0:
                video_img = []
                for cam_name in camera_names:
                    video_img.append(
                        env.render(
                            mode="rgb_array",
                            height=128,
                            width=128,
                            camera_name=cam_name,
                        )
                    )
                video_img = np.concatenate(
                    video_img, axis=1
                )  # concatenate horizontally
                video_writer.append_data(video_img)
            video_count += 1

        if first:
            break
    is_success = env.is_success()["task"]

    return (
        recorded_states[: len(recorded_actions)],
        recorded_actions,
        rec_obs,
        is_success,
    )


def save_demo(demo: dict, save_path: str, env_meta: str):
    # exit if not successful
    if not demo["success"]:
        print("Demo not successful, skipping save...")
        return

    # Save the demo with new states and actions (others remain unchanged).
    with h5py.File(save_path, "w") as f:
        demo_group = f.create_group("data/demo_0")
        demo_group.create_dataset("actions", data=np.array(demo["actions"]))
        demo_group.create_dataset("states", data=np.array(demo["states"]))
        # for k, v in demo["obs"].items():
        #     demo_group.create_dataset("obs/{}".format(k), data=np.array(v))
        demo_group.attrs["success"] = (
            demo["success"] if demo["success"] is not None else False
        )
        demo_group.attrs["failure_type"] = (
            demo["failure_type"] if demo["failure_type"] is not None else "None"
        )
        if demo["constraint_sequence"]:
            cgrp = demo_group.create_group("constraint_data")
            for i, c in enumerate(demo["constraint_sequence"]):
                cdict = c if isinstance(c, dict) else c.to_constraint_data_dict()
                cgrp.create_dataset(f"constraint_{i}", data=json.dumps(cdict))
        demo_group.attrs["model_file"] = demo["model_file"]
        f["data"].attrs["env_args"] = json.dumps(env_meta)
    print("Saved demo to {}".format(save_path))


def playback_dataset(args):
    # some arg checking
    write_video = args.video_path is not None
    assert not (args.render and write_video)  # either on-screen or video but not both

    # Auto-fill camera rendering info if not specified
    if args.render_image_names is None:
        # We fill in the automatic values
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
        env_type = EnvUtils.get_env_type(env_meta=env_meta)
        args.render_image_names = DEFAULT_CAMERAS[env_type]

    if args.render:
        # on-screen rendering can only support one camera
        assert len(args.render_image_names) == 1

    if args.use_obs:
        assert write_video, "playback with observations can only write to video"
        assert (
            not args.use_actions
        ), "playback with observations is offline and does not support action playback"

    if args.render_depth_names is not None:
        assert (
            args.use_obs
        ), "depth observations can only be visualized from observations currently"

    # create environment only if not playing back with observations
    if not args.use_obs:
        # need to make sure ObsUtils knows which observations are images, but it doesn't matter
        # for playback since observations are unused. Pass a dummy spec here.
        dummy_spec = dict(
            obs=dict(
                low_dim=["robot0_eef_pos"],
                rgb=[],
            ),
        )
        ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=dummy_spec)

        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
        env = EnvUtils.create_env_from_metadata(
            env_meta=env_meta, render=args.render, render_offscreen=write_video
        )

        # some operations for playback are robosuite-specific, so determine if this environment is a robosuite env
        is_robosuite_env = EnvUtils.is_robosuite_env(env_meta)

    f = h5py.File(args.dataset, "r")

    # list of all demonstration episodes (sorted in increasing number order)
    if args.filter_key is not None:
        print("using filter key: {}".format(args.filter_key))
        demos = [
            elem.decode("utf-8")
            for elem in np.array(f["mask/{}".format(args.filter_key)])
        ]
    else:
        demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    # maybe reduce the number of demonstrations to playback
    if args.n is not None:
        demos = demos[: args.n]

    # maybe dump video
    video_writer = None
    if write_video:
        video_writer = imageio.get_writer(args.video_path, fps=20)

    success_demo_paths = []
    failure_demo_paths = []

    from tqdm import tqdm

    for ind in tqdm(range(len(demos)), desc="Processing demos"):
        ep = demos[ind]
        print("Playing back episode: {}".format(ep))

        constraints = f["data/{}".format(ep)].get("constraint_data", None)
        constraint_timesteps = None
        if constraints is not None:
            constraint_timesteps = []
            for key in constraints:
                constraint_timesteps.extend(
                    json.loads(constraints[key][()])["timesteps"]
                )

        # prepare initial state to reload from
        states = f["data/{}/states".format(ep)][()]
        initial_state = dict(states=states[0])
        if is_robosuite_env:
            initial_state["model"] = f["data/{}".format(ep)].attrs["model_file"]

        # supply actions if using open-loop action playback
        actions = None
        if args.use_actions:
            actions = f["data/{}/actions".format(ep)][()]

        # Re-use metadata from the original demo but with new states/actions.
        demo_group = f["data/{}".format(ep)]
        demo = {
            "success": demo_group.attrs.get("success", False),
            "failure_type": demo_group.attrs.get("failure_type", "None"),
            "model_file": demo_group.attrs.get("model_file", ""),
            "constraint_sequence": None,
        }
        if "constraint_data" in demo_group:
            constraint_seq = []
            for key in demo_group["constraint_data"]:
                constraint_seq.append(
                    json.loads(demo_group["constraint_data"][key][()])
                )
            demo["constraint_sequence"] = constraint_seq

        attempt = 0
        while attempt < args.num_retries:
            action_noise_decay_factor = (args.num_retries - attempt) / args.num_retries
            print(
                f"Attempt {attempt+1} for episode {ep} with action noise decay factor {action_noise_decay_factor}"
            )
            rec_states, rec_actions, rec_obs, is_success = playback_trajectory_with_env(
                env=env,
                initial_state=initial_state,
                states=states,
                actions=actions,
                render=args.render,
                video_writer=video_writer,
                video_skip=args.video_skip,
                camera_names=args.render_image_names,
                first=args.first,
                action_noise_std=args.action_noise_std * action_noise_decay_factor,
                constraint_timesteps=constraint_timesteps,
            )

            if is_success:
                print(f"Episode {ep} playback successful.")
                break  # Successful playback; exit retry loop.
            else:
                print(f"Attempt {attempt+1} for episode {ep} failed")
                attempt += 1

        if attempt == args.num_retries:
            print(
                f"Episode {ep} failed after {args.num_retries + 1} attempts. Using original states/actions"
            )
            rec_states, rec_actions = states, actions
            is_success = True

        demo["actions"] = np.array(rec_actions)
        demo["states"] = np.array(rec_states)

        import pathlib

        # create path
        pathlib.Path(args.save_demo_dir).mkdir(exist_ok=True)
        save_path = pathlib.Path(args.save_demo_dir) / f"demo_{ep}_w_noise.hdf5"
        save_demo(demo, save_path, env_meta)
        if is_success:
            success_demo_paths.append(save_path)
        else:
            failure_demo_paths.append(save_path)

    f.close()
    if write_video:
        video_writer.close()

    from robomimic.scripts.dataset_states_to_obs import dataset_states_to_obs

    from demo_aug.utils.file_utils import merge_demo_files
    from scripts.dataset.mp4_from_h5 import Config as MP4H5Config
    from scripts.dataset.mp4_from_h5 import generate_videos_from_hdf5

    if success_demo_paths:
        merge_demo_save_path = (
            pathlib.Path(args.save_demo_dir) / "merged_success_demos.hdf5"
        )
        merge_demo_files(success_demo_paths, save_path=merge_demo_save_path)

        render_camera_names = ["agentview", "robot0_eye_in_hand"]
        camera_height, camera_width = 84, 84
        depth = False
        dataset_states_to_obs_args = SimpleNamespace(
            dataset=merge_demo_save_path,
            output_name=f"{str(pathlib.Path(merge_demo_save_path).stem)}_obs.hdf5",
            n=None,
            shaped=False,
            camera_names=render_camera_names,
            camera_height=camera_height,
            camera_width=camera_width,
            depth=depth,
            done_mode=1,  # done = done or (t == traj_len)
            copy_rewards=False,
            copy_dones=False,
            exclude_next_obs=True,
            compress=False,
            use_actions=False,
        )
        dataset_states_to_obs(dataset_states_to_obs_args)
        print(f"output_name: {str(pathlib.Path(merge_demo_save_path).stem)}_obs.hdf5")

        video_h5_path = (
            merge_demo_save_path.parent / f"{merge_demo_save_path.stem}_obs.hdf5"
        )
        generate_videos_from_hdf5(
            MP4H5Config(
                h5_file_path=video_h5_path,
                all_demos=True,
                fps=20,
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="path to hdf5 dataset",
    )
    parser.add_argument(
        "--filter_key",
        type=str,
        default=None,
        help="(optional) filter key, to select a subset of trajectories in the file",
    )

    # number of trajectories to playback. If omitted, playback all of them.
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="(optional) stop after n trajectories are played",
    )

    # Use image observations instead of doing playback using the simulator env.
    parser.add_argument(
        "--use-obs",
        action="store_true",
        help="visualize trajectories with dataset image observations instead of simulator",
    )

    # Playback stored dataset actions open-loop instead of loading from simulation states.
    parser.add_argument(
        "--use-actions",
        action="store_true",
        help="use open-loop action playback instead of loading sim states",
    )

    # Whether to render playback to screen
    parser.add_argument(
        "--render",
        action="store_true",
        help="on-screen rendering",
    )

    # Dump a video of the dataset playback to the specified path
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="(optional) render trajectories to this video file path",
    )

    # How often to write video frames during the playback
    parser.add_argument(
        "--video_skip",
        type=int,
        default=5,
        help="render frames to video every n steps",
    )

    # camera names to render, or image observations to use for writing to video
    parser.add_argument(
        "--render_image_names",
        type=str,
        nargs="+",
        default=None,
        help="(optional) camera name(s) / image observation(s) to use for rendering on-screen or to video. Default is"
        "None, which corresponds to a predefined camera for each env type",
    )

    # depth observations to use for writing to video
    parser.add_argument(
        "--render_depth_names",
        type=str,
        nargs="+",
        default=None,
        help="(optional) depth observation(s) to use for rendering to video",
    )

    # Only use the first frame of each episode
    parser.add_argument(
        "--first",
        action="store_true",
        help="use first frame of each episode",
    )

    # Action noise std for open-loop action playback
    parser.add_argument(
        "--action_noise_std",
        type=float,
        default=0.02,
        help="std of noise to add to actions during open-loop playback",
    )
    # num_retries
    parser.add_argument(
        "--num_retries",
        type=int,
        default=1,
        help="number of retries for each demo",
    )
    # save demo dir
    parser.add_argument(
        "--save_demo_dir",
        type=str,
        default=None,
        help="(optional) path to save the new demo with injected action noise",
    )

    args = parser.parse_args()
    # playback_dataset(args)
    import time

    a = time.time()
    playback_dataset(args)
    b = time.time()
    print(f"Total time taken: {b-a}")
