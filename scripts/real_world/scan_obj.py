from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from demo_aug.utils.camera_utils import get_transform_matrix, lookat_matrix


def compute_camera_positions(angles: List[float], radius: float) -> List[np.ndarray]:
    positions = []
    for angle in angles:
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        position = np.array([x, y, radius])
        positions.append(position)
    return positions


def compute_camera_rotations(
    positions: List[np.ndarray], target: np.ndarray
) -> List[np.ndarray]:
    rotations = []
    for position in positions:
        up = np.array([0, 0, 1])
        rotation_matrix = lookat_matrix(position, target, up)[:3, :3]
        rpy = R.from_matrix(rotation_matrix).as_euler("xyz", degrees=False)
        rotations.append(rpy)
    return rotations


def compute_eef_poses(
    X_W_cams: List[np.ndarray], X_ee_cam: np.ndarray
) -> List[np.ndarray]:
    eef_poses = []
    for X_W_cam in X_W_cams:
        position = X_W_cam[:3]
        rpy = X_W_cam[3:]
        rotation_matrix = R.from_euler("xyz", rpy, degrees=False).as_matrix()

        X_W_cam = np.eye(4)
        X_W_cam[:3, :3] = rotation_matrix
        X_W_cam[:3, 3] = position
        # want X_W_ee = X_W_cam @ X_cam_ee
        X_W_ee = X_W_cam @ np.linalg.inv(X_ee_cam)
        eef_position = X_W_ee[:3, 3]
        eef_rpy = R.from_matrix(X_W_ee[:3, :3]).as_euler("xyz", degrees=False)

        eef_pose = np.hstack((eef_position, eef_rpy))
        eef_poses.append(eef_pose)
    return eef_poses


def get_poses_to_scan_obj(
    angles: List[float], radius: float, ee_to_camera: np.ndarray, target: np.ndarray
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    positions = compute_camera_positions(angles, radius)
    # convert to target frame # TODO: update position to be centered around target!
    positions = [position + target for position in positions]
    rotations = compute_camera_rotations(positions, target)
    camera_poses = [np.hstack((pos, rot)) for pos, rot in zip(positions, rotations)]
    eef_poses = compute_eef_poses(camera_poses, ee_to_camera)
    return camera_poses, eef_poses


def plot_poses_with_axes(poses: List[np.ndarray]) -> None:
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot each pose
    axes_length = 0.1  # Shorter axes for better visualization in larger space
    for pose in poses:
        # Extract the origin
        origin = pose[:3, 3]

        # Plot the origin
        ax.scatter(origin[0], origin[1], origin[2], color="k", s=10)  # Origin point

        # Draw the axes
        x_axis = origin + axes_length * pose[:3, 0]
        y_axis = origin + axes_length * pose[:3, 1]
        z_axis = origin + axes_length * pose[:3, 2]

        ax.quiver(
            origin[0],
            origin[1],
            origin[2],
            x_axis[0] - origin[0],
            x_axis[1] - origin[1],
            x_axis[2] - origin[2],
            color="r",
            label="X axis",
        )

        ax.quiver(
            origin[0],
            origin[1],
            origin[2],
            y_axis[0] - origin[0],
            y_axis[1] - origin[1],
            y_axis[2] - origin[2],
            color="g",
            label="Y axis",
        )

        ax.quiver(
            origin[0],
            origin[1],
            origin[2],
            z_axis[0] - origin[0],
            z_axis[1] - origin[1],
            z_axis[2] - origin[2],
            color="b",
            label="Z axis",
        )

    # Plot world axes
    ax.quiver(0, 0, 0, 0.5, 0, 0, color="r", linestyle="--", alpha=0.5)
    ax.quiver(0, 0, 0, 0, 0.5, 0, color="g", linestyle="--", alpha=0.5)
    ax.quiver(0, 0, 0, 0, 0, 0.5, color="b", linestyle="--", alpha=0.5)

    # Set labels and title
    ax.set_xlabel("X axis (+X right)")
    ax.set_ylabel("Y axis (+Y down)")
    ax.set_zlabel("Z axis (+Z forward)")
    ax.set_title("Poses with Coordinate Axes")

    # Set xyz limits
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    # Display the plot
    plt.show()


def interpolate_poses(
    X_W_ee_interp_start: np.ndarray, X_W_ee_interp_end: np.ndarray, num_steps: int
) -> np.ndarray:
    # Extract translation components
    start_translation = X_W_ee_interp_start[:3, 3]
    end_translation = X_W_ee_interp_end[:3, 3]

    # Extract rotation components and convert to quaternions
    start_rotation = R.from_matrix(X_W_ee_interp_start[:3, :3])
    end_rotation = R.from_matrix(X_W_ee_interp_end[:3, :3])

    slerp_obj = Slerp(
        [0, num_steps],
        R.from_matrix([start_rotation.as_matrix(), end_rotation.as_matrix()]),
    )
    interp_rots = slerp_obj([i for i in range(num_steps)])
    # Initialize an array to hold the interpolated poses
    interpolated_poses = np.zeros((num_steps, 4, 4))

    for i in range(num_steps):
        alpha = i / (num_steps - 1)  # Interpolation parameter from 0 to 1

        # Interpolate translations linearly
        interp_translation = (1 - alpha) * start_translation + alpha * end_translation

        # Interpolate rotations using slerp
        interp_rotation = interp_rots[i].as_matrix()

        # Construct the interpolated transformation matrix
        interp_pose = np.eye(4)
        interp_pose[:3, :3] = interp_rotation
        interp_pose[:3, 3] = interp_translation

        interpolated_poses[i] = interp_pose

    return interpolated_poses


X_W_ee = np.array(
    [
        0.34495192766189575,
        0.005398919340223074,
        0.4042896330356598,
        -3.1362310489746488,
        0.1644460822668845,
        -0.017932463136487282,
    ]
)
X_ee_cam = np.array(
    [
        -0.07496436728692035,
        0.03375652826239056,
        0.01352048433972477,
        -0.34256567561092544,
        0.021615754419753275,
        -1.5874660014129223,
    ]
)

X_W_ee = get_transform_matrix(X_W_ee[:3], X_W_ee[3:])
X_ee_cam = get_transform_matrix(X_ee_cam[:3], X_ee_cam[3:])
X_W_cam = X_W_ee @ X_ee_cam

angles = np.linspace(0, 2 * np.pi, 10)
radius = 0.3
X_EE_CAM = X_ee_cam  # Example transformation matrix, replace with actual
target = np.array([0, 0, 0])

up = np.array([0.0, 0.0, 1.0])
position = np.array([0.268, -0.0269, 0.403])
target = np.array([0.33, 0.0, 0.0])
target = X_W_cam[:3, 3]

camera_poses, eef_poses = get_poses_to_scan_obj(angles, radius, X_EE_CAM, target)

# convert camera poses to 4x4 matrices
for i in range(len(camera_poses)):
    camera_poses[i] = get_transform_matrix(camera_poses[i][:3], camera_poses[i][3:])

# convert eef poses to 4x4 matrices
for i in range(len(eef_poses)):
    eef_poses[i] = get_transform_matrix(eef_poses[i][:3], eef_poses[i][3:])

# now get extra eef poses to go from X_W_ee to the first eef pose via interpolation
X_W_ee_interp_end = eef_poses[0]
X_W_ee_interp_start = X_W_ee

X_W_ee_interp_lst = interpolate_poses(X_W_ee_interp_start, X_W_ee_interp_end, 5)
# concatenate the interpolated poses with the original eef poses
eef_poses = np.vstack((X_W_ee_interp_lst, eef_poses))
# convert eef poses to 3 pos 3 euler xyz
for i in range(len(eef_poses)):
    eef_poses[i] = np.concatenate(
        [eef_poses[i][:3, 3], R.from_matrix(eef_poses[i][:3, :3]).as_euler("xyz")]
    )
plot_poses_with_axes(camera_poses)
plot_poses_with_axes(eef_poses)
