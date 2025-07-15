from typing import List

import matplotlib.pyplot as plt
import numpy as np

from demo_aug.utils.camera_utils import get_transform_matrix


def normalize(x: np.ndarray) -> np.ndarray:
    """Returns a normalized vector."""
    return x / np.linalg.norm(x)


def plot_poses_with_axes(poses: List[np.ndarray]) -> None:
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot each pose
    axes_length = 0.2  # Shorter axes for better visualization in larger space
    for pose in poses:
        # Extract the origin
        origin = pose[:3, 3]

        # Plot the origin
        ax.scatter(origin[0], origin[1], origin[2], color="k", s=100)  # Origin point

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


if __name__ == "__main__":
    X_W_hand = np.array(
        [
            0.34468984603881836,
            0.003972956445068121,
            0.4039565324783325,
            -3.132271350474009,
            0.16433151099725118,
            -0.02449520831057671,
        ]
    )
    X_W_hand = np.array(
        [
            0.34495192766189575,
            0.005398919340223074,
            0.4042896330356598,
            -3.1362310489746488,
            0.1644460822668845,
            -0.017932463136487282,
        ]
    )
    X_hand_cam = np.array(
        [
            -0.07496436728692035,
            0.03375652826239056,
            0.01352048433972477,
            -0.34256567561092544,
            0.021615754419753275,
            -1.5874660014129223,
        ]
    )
    X_hand_cam_2 = np.array(
        [
            -0.07644216443928634,
            -0.02730158707694055,
            0.012627235466440005,
            -0.3429787912920421,
            0.0175529883724852,
            -1.5911095137360665,
        ]
    )

    X_W_hand = get_transform_matrix(X_W_hand[:3], X_W_hand[3:])
    X_hand_cam = get_transform_matrix(X_hand_cam[:3], X_hand_cam[3:])
    X_hand_cam_2 = get_transform_matrix(X_hand_cam_2[:3], X_hand_cam_2[3:])
    X_W_cam = X_W_hand @ X_hand_cam
    X_W_cam_2 = X_W_hand @ X_hand_cam_2
    plot_poses_with_axes([X_W_hand, X_W_cam, X_W_cam_2])
