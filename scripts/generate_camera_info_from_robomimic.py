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

    # visualize initial states in the demonstration data
    python playback_dataset.py --dataset /path/to/dataset.hdf5 \
        --first --render_image_names agentview \
        --video_path /tmp/dataset_task_inits.mp4
"""

import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import h5py
import imageio
import matplotlib.pyplot as plt
import numpy as np
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import torch
from robomimic.envs.env_base import EnvBase, EnvType
from robosuite.utils.transform_utils import mat2quat, pose2mat

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
    assert isinstance(env, EnvBase)

    write_video = video_writer is not None
    video_count = 0
    assert not (render and write_video)

    traj_len = states.shape[0]
    action_playback = actions is not None
    if action_playback:
        assert states.shape[0] == actions.shape[0]

    for i in range(traj_len):
        if action_playback:
            obs, reward, done, _ = env.step(actions[i])
            render_image_collage(obs)

            # env.step(actions[i])
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
            camera_names = ["sideview1"]
            env.render(mode="human", camera_name=camera_names[0])

        # video render
        if write_video:
            if video_count % video_skip == 0:
                video_img = []
                for cam_name in camera_names:
                    print(cam_name)
                    video_img.append(
                        env.render(
                            mode="rgb_array",
                            height=512,
                            width=512,
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


def playback_trajectory_with_obs(
    traj_grp,
    video_writer,
    video_skip=5,
    image_names=None,
    first=False,
):
    """
    This function reads all "rgb" observations in the dataset trajectory and
    writes them into a video.

    Args:
        traj_grp (hdf5 file group): hdf5 group which corresponds to the dataset trajectory to playback
        video_writer (imageio writer): video writer
        video_skip (int): determines rate at which environment frames are written to video
        image_names (list): determines which image observations are used for rendering. Pass more than
            one to output a video with multiple image observations concatenated horizontally.
        first (bool): if True, only use the first frame of each episode.
    """
    assert (
        image_names is not None
    ), "error: must specify at least one image observation to use in @image_names"
    video_count = 0

    traj_len = traj_grp["actions"].shape[0]
    for i in range(traj_len):
        if video_count % video_skip == 0:
            # concatenate image obs together
            im = [traj_grp["obs/{}".format(k)][i] for k in image_names]
            frame = np.concatenate(im, axis=1)
            video_writer.append_data(frame)
        video_count += 1

        if first:
            break


def render_image_collage(image_dict):
    """
    Renders and displays a collage of images from the provided dictionary.

    image_dict (dict): Dictionary of images with keys as image names and values as image arrays.

    Returns:
    None
    """
    # Filter the image dictionary to include only entries with keys containing "image"
    filtered_dict = {key: value for key, value in image_dict.items() if "image" in key}

    # Calculate the number of rows and columns for the collage grid
    num_images = len(filtered_dict)
    num_cols = 3  # Adjust the number of columns as desired
    num_rows = (num_images + num_cols - 1) // num_cols

    # Create a figure and axes for the collage
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 8))

    # Flatten the axes array if there is only one row
    if num_rows == 1:
        axs = axs.flatten()

    # Loop over the filtered image dictionary and render each image
    for i, (image_name, image_array) in enumerate(filtered_dict.items()):
        row_index = i // num_cols
        col_index = i % num_cols
        axs[row_index, col_index].imshow(image_array)
        axs[row_index, col_index].set_title(image_name)
        axs[row_index, col_index].axis("off")

    # Hide any remaining empty subplots
    for j in range(num_images, num_rows * num_cols):
        row_index = j // num_cols
        col_index = j % num_cols
        axs[row_index, col_index].axis("off")

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Display the collage
    plt.show()


def add_camera_to_xml(xml, camera_name, camera_pos, camera_quat):
    """
    Adds a new camera to the XML by attaching it to a body element,
    allowing for camera movement by manipulating the body.

    xml (str): Mujoco sim XML file as a string
    camera_name (str): Name of the new camera
    camera_pos (str): Position of the camera in the format "x y z"
    camera_quat (str): Quaternion rotation of the camera in the format "w x y z"

    Returns:
    str: Modified XML with the new camera
    """
    tree = ET.fromstring(xml)

    # Find the parent element to attach the camera body
    worldbody_elem = tree.find("worldbody")  # Modify this to match your XML structure
    assert worldbody_elem is not None, "No <worldbody> element found in the XML."

    # Create the camera body element
    new_camera = ET.SubElement(worldbody_elem, "camera")
    new_camera.set("mode", "fixed")
    new_camera.set("name", camera_name)
    new_camera.set("pos", camera_pos)
    new_camera.set("quat", camera_quat)

    # Return the modified XML
    return ET.tostring(tree, encoding="utf8").decode("utf8")


_EPS = np.finfo(float).eps * 4.0


def anorm(x, axis=None, keepdims=False):
    """Compute L2 norms alogn specified axes."""
    return np.sqrt((x * x).sum(axis=axis, keepdims=keepdims))


def normalize(v, axis=None, eps=1e-10):
    """L2 Normalize along specified axes."""
    return v / max(anorm(v, axis=axis, keepdims=True), eps)


def normalize_with_norm(x: torch.Tensor, dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Normalize tensor along axis and return normalized value with norms.

    Args:
        x: tensor to normalize.
        dim: axis along which to normalize.

    Returns:
        Tuple of normalized tensor and corresponding norm.
    """

    norm = torch.maximum(
        torch.linalg.vector_norm(x, dim=dim, keepdims=True), torch.tensor([_EPS]).to(x)
    )
    return x / norm, norm


def extrinsic_to_pos_rot(lookat: torch.Tensor, up: torch.Tensor, pos: torch.Tensor):
    L = normalize(lookat)
    up = normalize(up)
    s = normalize(torch.cross(L, up))
    u_pr = normalize(torch.cross(s, L))

    R = torch.stack([s, u_pr, -L])
    t = -R @ pos
    return R, t


def get_lookat(eye, target=[0, 0, 0], up=[0, 1, 0]):
    """Generate LookAt modelview matrix."""
    eye = np.float32(eye)
    forward = normalize(target - eye)
    side = normalize(np.cross(forward, up))
    up = np.cross(side, forward)
    M = np.eye(4, dtype=np.float32)
    R = M[:3, :3]
    R[:] = [side, up, -forward]
    M[:3, 3] = -R.dot(eye)
    return M[:3, :3], M[:3, 3]


def get_pose(eye, target, up):
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


def fibonacci_sphere(samples: int = 40):
    points = []
    phi = math.pi * (math.sqrt(5.0) - 1.0)  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        points.append((x, y, z))

    return points


def generate_camera_positions(K: int, r: float, center: torch.Tensor):
    """
    Generate K random camera positions on a hemisphere.
    """
    points = []
    fib_points = fibonacci_sphere(samples=K * 2 + 2)
    # filter out camera positions with z values < 0
    fib_points = [p for p in fib_points if p[2] > 0]
    # add center point and apply radius to each fibonacci point
    for p in fib_points:
        points.append(torch.tensor(p) * r + center)

    assert (
        len(points) == K
    ), "Generated {} points instead of {}; easiest fix: update samples above".format(
        len(points), K
    )
    return points


def get_camera_intrinsic_matrix(sim, camera_name, camera_height, camera_width):
    """
    Obtains camera intrinsic matrix.

    Args:
        sim (MjSim): simulator instance
        camera_name (str): name of camera
        camera_height (int): height of camera images in pixels
        camera_width (int): width of camera images in pixels
    Return:
        K (np.array): 3x3 camera matrix
    """
    cam_id = sim.model.camera_name2id(camera_name)
    fovy = sim.model.cam_fovy[cam_id]
    f = 0.5 * camera_height / np.tan(fovy * np.pi / 360)
    K = np.array([[f, 0, camera_width / 2], [0, f, camera_height / 2], [0, 0, 1]])
    return K


def add_cameras_to_env(
    env,
    num_cameras,
    target,
    up: torch.tensor,
    radius: float = 0.75,
    image_height: int = 256,
    image_width: int = 256,
) -> Dict[str, Dict[str, Any]]:
    """
    Add cameras to the env.

    Returns a list of per frame info for each camera in the form of a list of dicts
    """
    per_camera_info: Dict[str, Dict[str, Any]] = {}
    per_camera_poses: List[torch.Tensor] = []
    initial_mjstate = env.env.sim.get_state().flatten()
    xml = env.env.sim.model.get_xml()

    env.env.num_cameras = num_cameras
    env.env.camera_heights[0] = image_height
    env.env.camera_widths[0] = image_width
    circle_center = target + torch.tensor([0.0, 0.0, 0.1])

    camera_positions = generate_camera_positions(num_cameras, radius, circle_center)

    poses = [get_pose(camera_positions[i], target, up) for i in range(num_cameras)]
    quats_wxyz = [
        [mat2quat(R)[3], mat2quat(R)[0], mat2quat(R)[1], mat2quat(R)[2]]
        for R, t in poses
    ]

    for i in range(env.env.num_cameras):
        pos = [pos for pos in camera_positions[i].numpy()]
        pos_str = [str(x) for x in pos]
        quat_wxyz = quats_wxyz[i]
        quat_str = [str(x) for x in quat_wxyz]
        xml = add_camera_to_xml(
            xml,
            camera_name=f"a{i}",
            camera_pos=" ".join(pos_str),
            camera_quat=" ".join(quat_str),
        )
        quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
        per_camera_poses.append(pose2mat((pos, quat_xyzw)))

    env.env.camera_names = [f"a{i}" for i in range(env.env.num_cameras)]
    env.env.camera_heights = env.env._input2list(
        env.env.camera_heights[0], env.env.num_cameras
    )
    env.env.camera_widths = env.env._input2list(
        env.env.camera_widths[0], env.env.num_cameras
    )
    env.env.camera_depths = env.env._input2list(False, env.env.num_cameras)

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
            "transform_matrix": per_camera_poses[i],  # c2w i.e. camera pose
        }
    return per_camera_info


def render_and_save_images(
    env,
    actions: List[Any],
    states: List[Any],
    camera_names: List[str],
    save_dir: str,
    per_camera_info: Dict[str, Dict[str, Any]],
):
    all_image_infos: List[Dict] = []
    # Create save directory if it doesn't exist
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Loop through each action and render the scene
    for i, state in enumerate(states):
        env.reset_to({"states": states[i]})

        # Loop through each camera and render and save the image
        for cam_name in camera_names:
            camera_info = per_camera_info[cam_name]
            if i != 10:
                continue
            print(f"Rendering image for camera: {cam_name}")
            image = env.render(
                mode="rgb_array",
                camera_name=cam_name,
                height=camera_info["height"],
                width=camera_info["width"],
            )
            file_name = f"frame_{str(i).zfill(5)}_{cam_name}.jpeg"
            file_path = Path(save_dir) / file_name
            cv2.imwrite(str(file_path), cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            camera_info = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in camera_info.items()
            }
            # Get the camera information for the image
            image_info = {
                "file_path": str(file_path),
                "camera_name": cam_name,
                "time": i,
                **camera_info,
            }
            all_image_infos.append(image_info)
            print(f"Saved image: {file_path} | Camera Name: {cam_name} | Time: {i}")

    # Save all image infos to a json file
    with open(Path(save_dir) / "image_infos.json", "w") as f:
        json.dump(all_image_infos, f)

    env.reset()


def playback_dataset(args):
    image_height = 512
    image_width = 512
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
        write_video = True
        env_meta["env_kwargs"]["camera_heights"] = image_height
        env_meta["env_kwargs"]["camera_widths"] = image_width
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
        video_writer = imageio.get_writer(args.video_path, fps=2)

    for ind in range(len(demos)):
        ep = demos[ind]
        print("Playing back episode: {}".format(ep))

        if args.use_obs:
            playback_trajectory_with_obs(
                traj_grp=f["data/{}".format(ep)],
                video_writer=video_writer,
                video_skip=args.video_skip,
                image_names=args.render_image_names,
                first=args.first,
            )
            continue

        # prepare initial state to reload from
        states = f["data/{}/states".format(ep)][()]
        initial_state = dict(states=states[0])
        if is_robosuite_env:
            initial_state["model"] = f["data/{}".format(ep)].attrs["model_file"]

        # supply actions if using open-loop action playback
        actions = None
        if args.use_actions:
            actions = f["data/{}/actions".format(ep)][()]

        # load the initial state
        env.reset()
        env.reset_to(initial_state)

        # per_camera_info: Dict[str, Dict[str, Any]] = add_cameras_to_env(
        #     env,
        #     num_cameras=num_cameras,
        #     radius=radius,
        #     up=torch.tensor([0, 0, 1]),
        #     target=torch.tensor([0, 0, 0.83]),
        #     image_height=image_height,
        #     image_width=image_width
        # )

        # render_and_save_images(
        #     env=env,
        #     actions=actions,
        #     states=states,
        #     camera_names=env.env.camera_names,
        #     per_camera_info=per_camera_info,
        #     save_dir=args.save_dir,
        # )
        playback_trajectory_with_env(
            env=env,
            initial_state=initial_state,
            states=states,
            actions=actions,
            render=args.render,
            video_writer=video_writer,
            video_skip=args.video_skip,
            camera_names=env.env.camera_names,  # args.render_image_names,
            first=args.first,
        )

        if ind == 0:
            break

    f.close()
    if write_video:
        video_writer.close()


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
        help=(
            "(optional) camera name(s) / image observation(s) to use for rendering on-screen or to video. Default is"
            "None, which corresponds to a predefined camera for each env type"
        ),
    )

    # Only use the first frame of each episode
    parser.add_argument(
        "--first",
        action="store_true",
        help="use first frame of each episode",
    )

    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="(optional) directory to save images to",
    )

    args = parser.parse_args()
    playback_dataset(args)
