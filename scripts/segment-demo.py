import os
from typing import Dict, List, Tuple

import cv2
import h5py
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import ruptures as rpt

from demo_aug.utils.mujoco_utils import get_joint_name_to_qpos


class ObjectMotionSegmentation:
    def __init__(
        self, min_segment_size: int = 3, kernel: str = "rbf", penalty: float = 10.0
    ):
        """
        Temporal segmentation based on object motion.

        Args:
            min_segment_size (int): Minimum segment size for change detection.
            kernel (str): Kernel type for Ruptures CPD (e.g., "rbf").
            penalty (float): Penalty value for change point detection.
        """
        self.min_segment_size = min_segment_size
        self.kernel = kernel
        self.penalty = penalty
        self.segments = None

    def segment_object_trajectories(
        self, object_poses: np.ndarray
    ) -> List[Tuple[int, int]]:
        """
        Segments object motion based on velocity change points and significant position changes.

        Args:
            object_poses (np.ndarray): (T, D) array of object positions.
            contacts (np.ndarray): (T,) binary array (1 if in contact, 0 otherwise).

        Returns:
            List[Tuple[int, int]]: Temporal segments as (start_idx, end_idx).
        """
        velocities = np.diff(object_poses, axis=0)  # (T-1, D)
        velocity_norms = np.linalg.norm(velocities, axis=1)  # (T-1,)

        # Change point detection using velocity
        signal = velocity_norms[:, None]
        algo = rpt.KernelCPD(kernel=self.kernel, min_size=self.min_segment_size)
        algo.fit(signal)
        change_points = algo.predict(pen=self.penalty)

        # Filter change points based on position changes and if we were in contact with the object
        filtered_change_points = [change_points[0]]  # Always keep the start
        position_threshold = 0.01  # 1 cm threshold

        for i in range(1, len(change_points) - 1):
            start_idx = filtered_change_points[-1]
            end_idx = change_points[i]

            position_change = np.linalg.norm(
                object_poses[end_idx - 1] - object_poses[start_idx]
            )
            if position_change > position_threshold:
                filtered_change_points.append(end_idx)

        filtered_change_points.append(len(object_poses))  # Always keep the end
        change_points = filtered_change_points

        # Merge change points based on velocity and position thresholds
        self.segments = [
            (change_points[i], change_points[i + 1])
            for i in range(len(change_points) - 1)
        ]
        return self.segments


def load_hdf5_data(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads end-effector positions and images from an HDF5 file.
    If images are not available, it creates an environment to render them.

    Args:
        file_path (str): Path to the HDF5 file.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - End-effector positions (T, 3) -> (x, y, z).
            - Agent view images (T, H, W, 3) -> RGB frames.
    """
    with h5py.File(file_path, "r") as f:
        demo_key = list(f["data"].keys())[0]  # Assuming a single demo in the file
        state_key = f"data/{demo_key}/states"
        states = np.array(f[state_key])  # Load full states

    dummy_spec = dict(
        obs=dict(
            low_dim=["robot0_eef_pos"],
            rgb=["agentview_image", "agentview"],
        ),
    )
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=file_path)
    ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=dummy_spec)
    from robosuite.controllers import load_composite_controller_config

    controller_config = load_composite_controller_config(robot="Panda")
    env_meta["env_kwargs"]["controller_configs"] = controller_config
    src_env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        use_image_obs=True,
        render_offscreen=True,
        render=False,
    )

    from demo_aug.generate import CPEnv

    src_env = CPEnv(src_env)
    joint_obs, images = get_joint_observations_from_states(states, src_env)
    eef_positions = np.array(joint_obs["mug_joint0"])[..., :3]
    eef_positions = np.array(joint_obs["coffee_machine_lid_main_joint0"])[
        ..., :1
    ]  # Keep only x, y, z
    return eef_positions, images


def get_top_level_body_names(model: mujoco.MjModel) -> List[int]:
    """Get names of all top-level bodies (direct children of the world body).
    Args:
        model: MuJoCo model.
    Returns:
        A list containing IDs of top-level bodies.
    """
    top_level_ids = [
        body_id
        for body_id in range(1, model.nbody)
        if model.body_parentid[body_id] == 0
    ]
    top_level_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        for body_id in top_level_ids
    ]
    return top_level_names


def get_body_joints_recursive(model: mujoco.MjModel, body_name: str) -> List[str]:
    """Get names of all joints belonging to a body, including joints of its descendant bodies.

    Args:
        model: MuJoCo model.
        body_name: Name of the body.

    Returns:
        A list containing names of joints belonging to the body and all its descendants.
    """
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)

    joint_names = []

    def collect_joints(body_id: int):
        """Recursively collects joints for the given body and all its child bodies."""
        # Add joints belonging to this body
        for joint_id in range(model.njnt):
            if model.jnt_bodyid[joint_id] == body_id:
                joint_names.append(
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
                )

        # Find and recurse into child bodies
        for child_body_id in range(model.nbody):
            if model.body_parentid[child_body_id] == body_id:
                collect_joints(child_body_id)

    collect_joints(body_id)
    return joint_names


def get_top_level_joints_qpos(
    model: mujoco.MjModel, data: mujoco.MjData, exclude_prefixes: Tuple[str, ...] = ()
) -> Dict[str, float]:
    """Returns joint values for all joints under top-level bodies, excluding specified prefixes."""
    joint_values = {}
    top_level_bodies = get_top_level_body_names(model)

    for body in top_level_bodies:
        if any(body.startswith(prefix) for prefix in exclude_prefixes):
            continue
        joints = get_body_joints_recursive(model, body)
        joint_values.update(
            {
                j: get_joint_name_to_qpos(model, data).get(j)
                for j in joints
                if j in get_joint_name_to_qpos(model, data)
            }
        )

    np.set_printoptions(precision=3)
    return joint_values


def get_joint_observations_from_states(
    states: np.ndarray, src_env, exclude_prefixes: Tuple[str] = ("robot",)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates an environment, sets states, and renders observations.

    Args:
        states (np.ndarray): Array of full environment states (T, ...).

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - End-effector positions (T, 3).
            - Rendered images (T, H, W, 3) in RGB format.
    """
    images = []
    model, data = src_env.env.env.sim.model._model, src_env.env.env.sim.data._data
    joints = get_top_level_joints_qpos(model, data, exclude_prefixes=exclude_prefixes)

    joint_obs = {j_name: [] for j_name in joints.keys()}
    for t in range(len(states)):
        src_env.env.reset_to({"states": states[t]})
        obs = src_env.get_observation()
        joints = get_top_level_joints_qpos(
            model, data, exclude_prefixes=exclude_prefixes
        )
        for jnt in joints.keys():
            joint_obs[jnt].append(joints[jnt])

        images.append(
            (obs["agentview_image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        )

    return joint_obs, np.array(images)


def visualize_segments(object_poses: np.ndarray, segments: List[Tuple[int, int]]):
    """
    Visualizes the segmented object trajectory.

    Args:
        object_poses (np.ndarray): End-effector positions (T, 3).
        segments (List[Tuple[int, int]]): List of segment start/end indices.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for i, (start, end) in enumerate(segments):
        segment_poses = object_poses[start:end]
        ax.plot(segment_poses[:, 0], segment_poses[:, 1], label=f"Segment {i+1}")

    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_title("Segmented End-Effector Trajectory")
    ax.legend()

    # save the plot to disk
    plt.savefig("segmented_trajectory.png")
    print("Saved segmented trajectory plot to segmented_trajectory.png")


def overlay_segments_on_images(
    images: np.ndarray, segments: List[Tuple[int, int]], save_path: str
):
    """
    Overlays segment labels on images and saves them as a visualization.

    Args:
        images (np.ndarray): Sequence of agent view images (T, H, W, 3).
        segments (List[Tuple[int, int]]): Temporal segments as (start_idx, end_idx).
        save_path (str): Path to save output images.
    """
    os.makedirs(save_path, exist_ok=True)

    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(len(images)):
        img = images[i].copy()
        segment_id = None
        for j, (start, end) in enumerate(segments):
            if start <= i < end:
                segment_id = j + 1
                break

        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if segment_id is not None:
            cv2.putText(
                img_bgr,
                f"{segment_id}",
                (10, 10),
                font,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        if segment_id is not None:
            cv2.imwrite(
                os.path.join(save_path, f"segment_{segment_id}_frame_{i:04d}.png"),
                img_bgr,
            )
        else:
            cv2.imwrite(os.path.join(save_path, f"frame_{i:04d}.png"), img_bgr)

    print(f"Saved segmented images to {save_path}")


def create_segmented_video(
    images: np.ndarray, segments: List[Tuple[int, int]], save_path: str, fps: int = 30
):
    """
    Creates a video where each segment is played sequentially.

    Args:
        images (np.ndarray): Sequence of agent view images (T, H, W, 3).
        segments (List[Tuple[int, int]]): Temporal segments as (start_idx, end_idx).
        save_path (str): Path to save the output video.
        fps (int): Frames per second for the video.
    """
    height, width, _ = images[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    for start, end in segments:
        # Stay on the first frame of the segment
        for _ in range(fps):  # 1 second pause
            video_writer.write(cv2.cvtColor(images[start], cv2.COLOR_RGB2BGR))

        # Play the segment
        for i in range(start, end):
            video_writer.write(cv2.cvtColor(images[i], cv2.COLOR_RGB2BGR))

        # Stay on the last frame of the segment
        for _ in range(fps):  # 1 second pause
            video_writer.write(cv2.cvtColor(images[end - 1], cv2.COLOR_RGB2BGR))

    video_writer.release()
    print(f"Saved segmented video to {save_path}")


# ==== MAIN EXECUTION ====
file_path = "datasets/source/coffee_preparation.hdf5"
save_path = "output_segmented_images"

# Load data
eef_positions, images = load_hdf5_data(file_path)

# Initialize and run segmentation
segmenter = ObjectMotionSegmentation(min_segment_size=3, kernel="rbf", penalty=5)
segments = segmenter.segment_object_trajectories(eef_positions)

segments = [seg for seg in segments if len(seg) > 0]

# visualize_segments(eef_positions, segments)
overlay_segments_on_images(images, segments, save_path)
print(f"saved images to {save_path}")


# Create segmented video
video_save_path = "segmented_video.mp4"
create_segmented_video(images, segments, video_save_path)
