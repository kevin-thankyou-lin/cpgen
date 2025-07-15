import pathlib
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tyro
from scipy.spatial.transform import Rotation as R

from demo_aug.demo import Demo
from demo_aug.utils.data_collection_utils import load_demos
from demo_aug.utils.mathutils import compute_quaternion_distances


def compute_distances(
    pos: List[np.ndarray], quat: List[np.ndarray]
) -> Tuple[List[float], List[float]]:
    pos_distances = [np.linalg.norm(pos[i] - pos[i + 1]) for i in range(len(pos) - 1)]
    quat_distances = compute_quaternion_distances(quat)
    return pos_distances, quat_distances


def plot_demos_dists_3d(
    demos: List[Demo],
    save_path: pathlib.Path,
    max_timestep: int = -1,
    plot_quiver: bool = False,
) -> None:
    fig = plt.figure(figsize=(18, 10))

    ax1 = fig.add_subplot(2, 3, 1, projection="3d")  # Top view
    ax2 = fig.add_subplot(2, 3, 2, projection="3d")  # Side view
    ax3 = fig.add_subplot(2, 3, 3, projection="3d")  # Perspective view

    ax4 = fig.add_subplot(2, 3, 4)  # Position distances
    ax5 = fig.add_subplot(2, 3, 5)  # Quaternion distances
    ax6 = fig.add_subplot(2, 3, 6, projection="3d")  # Lowest Z value point

    # Views (elevation, azimuth)
    views = [(None, None), (90, 0), (0, 0)]  # Top, Side, Perspective
    view_names = ["Perspective", "Top", "Side"]

    # Create a list of distinct colors, one for each demo
    colors = plt.cm.jet(np.linspace(0, 1, len(demos)))

    for demo_idx, demo in enumerate(demos):
        obs_lst = demo.get_obs_for_range(0, len(demo.timestep_data))

        obs_eef_pos = [obs["robot0_eef_pos"] for obs in obs_lst][:max_timestep]
        obs_eef_quat = [obs["robot0_eef_quat"] for obs in obs_lst][:max_timestep]

        obs_eef_pos = np.array(obs_eef_pos)
        obs_eef_quat = np.array(obs_eef_quat)

        # Plotting for each 3D view
        for ax, view, view_name in zip([ax1, ax2, ax3], views, view_names):
            # Calculate direction vectors
            # Standard direction vector (can be adjusted)
            standard_direction = np.array([1, 0, 0])

            # Convert quaternions to rotation matrices and apply to the standard direction
            directions = np.array(
                [R.from_quat(quat).apply(standard_direction) for quat in obs_eef_quat]
            )

            if plot_quiver:
                # all indices
                idx = np.arange(len(obs_eef_pos))
                ax.quiver(
                    obs_eef_pos[idx, 0],
                    obs_eef_pos[idx, 1],
                    obs_eef_pos[idx, 2],
                    directions[idx, 0],
                    directions[idx, 1],
                    directions[idx, 2],
                    length=0.02,
                    normalize=True,
                    color="red",
                    arrow_length_ratio=0.3,
                )

            # Plotting 3D positions with gradient
            cmap = plt.get_cmap("viridis")
            num_points = len(obs_eef_pos)
            for i in range(num_points - 1):
                x, y, z = zip(obs_eef_pos[i], obs_eef_pos[i + 1])
                ax.plot(x, y, z, color=cmap(i / (num_points - 1)), marker="o")

            # Set the view for the 3D plot
            ax.set_title(f"3D Positions ({view_name} view)")
            ax.view_init(elev=view[0], azim=view[1])

            # # Add axes labels
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")

        pos_distances, quat_distances = compute_distances(obs_eef_pos, obs_eef_quat)

        # Plotting Position and Quaternion Distances
        # ax4.plot(pos_distances, marker="o", label=f"Demo {demos.index(demo)}")
        # ax5.plot(quat_distances, marker="o", label=f"Demo {demos.index(demo)}")

        lowest_z_value_idx = np.argmin(obs_eef_pos[:, 2])  # Find the minimum Z value
        lowest_point_pos = obs_eef_pos[
            lowest_z_value_idx
        ]  # Find the corresponding point

        start_z_value_idx = np.argmin(obs_eef_pos[0, 2])  # Find the minimum Z value
        start_point_pos = obs_eef_pos[start_z_value_idx]  # Find the corresponding point

        # Plot the lowest Z value point
        ax6.plot(
            [lowest_point_pos[0]],
            [lowest_point_pos[1]],
            [lowest_point_pos[2]],
            "o",
            color=colors[demo_idx],
            markersize=2,
        )  # Plot as a red dot
        ax6.plot(
            [start_point_pos[0]],
            [start_point_pos[1]],
            [start_point_pos[2]],
            "*",
            color=colors[demo_idx],
        )  # Plot as a green dot

        # Plot the direction vector of the lowest Z value point
        ax6.quiver(
            lowest_point_pos[0],
            lowest_point_pos[1],
            lowest_point_pos[2],
            directions[lowest_z_value_idx, 0],
            directions[lowest_z_value_idx, 1],
            directions[lowest_z_value_idx, 2],
            length=0.02,
            normalize=True,
            color="red",
            arrow_length_ratio=0.3,
        )

        # Plot the direction vector of the start point
        ax6.quiver(
            start_point_pos[0],
            start_point_pos[1],
            start_point_pos[2],
            directions[start_z_value_idx, 0],
            directions[start_z_value_idx, 1],
            directions[start_z_value_idx, 2],
            length=0.02,
            normalize=True,
            color="green",
            arrow_length_ratio=0.3,
        )

        ax6.set_title("Start and lowest z Value Point; Top down view")
        ax6.view_init(elev=90, azim=0)
        ax6.set_xlabel("X")
        ax6.set_ylabel("Y")
        ax6.set_zlabel("Z")

        # Plot the start point
        ax6.plot(
            [start_point_pos[0]],
            [start_point_pos[1]],
            [start_point_pos[2]],
            "*",
            color="green",
            markersize=10,
            label="Start Point (*)",
        )

        # Plot the lowest Z value point
        ax6.plot(
            [lowest_point_pos[0]],
            [lowest_point_pos[1]],
            [lowest_point_pos[2]],
            "o",
            color="red",
            markersize=10,
            label="Lowest Z Point (o)",
        )

        # Ensure the legend is displayed
        ax6.legend(loc="upper right")

    # loop over ax1, ax2, ax3, ax6
    for ax in [ax1, ax2, ax3, ax6]:
        ax.set_xlim(-0.25, 0.25)
        ax.set_ylim(-0.25, 0.25)
        ax.set_zlim(0.8, 1.05)

    ax4.set_title("Position L2 Distances")
    ax4.set_xlabel("Index")
    ax4.set_ylabel("Distance")
    ax4.legend()

    ax5.set_title("Quaternion Angular Distances")
    ax5.set_xlabel("Index")
    ax5.set_ylabel("Distance")
    ax5.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot to {save_path}")


@dataclass
class Config:
    # The path to the dataset
    dataset_path: Optional[str] = None

    task_name: Optional[str] = None

    # The path to the output plot
    plot_path: Optional[str] = None

    plot_path_extra_name: Optional[str] = None

    # The maximum number of demos to analyze
    max_demos: int = 20

    # max timestep to plot for each demo
    max_timestep: int = -1

    plot_type: Literal["human", "gen"] = "gen"

    plot_quiver: bool = False

    def __post_init__(self):
        # if task_name is given, hardcode the dataset path
        if self.task_name is not None:
            if self.plot_type == "human":
                if self.task_name == "lift":
                    self.dataset_path = "/juno/u/thankyou/autom/diffusion_policy/data/robomimic/datasets/lift/ph/image_abs.hdf5"
                elif self.task_name == "square":
                    self.dataset_path = "/juno/u/thankyou/autom/diffusion_policy/data/robomimic/datasets/square/ph/1_demo.hdf5"
                    self.dataset_path = "/scr/thankyou/autom/Consistency-Policy/data/robomimic/datasets/square/ph/image_abs.hdf5"
            elif self.plot_type == "gen":
                if self.task_name == "lift":
                    self.dataset_path = "/juno/u/thankyou/autom/consistency_policy/data/robomimic/datasets/augmented/demo_025trials/2023-11-12/aefnb5l1/lift_trials_25_scaleaug-scalefactor0.67-1.33_se3aug-dx-0.22-0.22-dz-0.02-0.02-dthetaz-0.4-1.1_startaug-eeposbound0.15-eerotbound0.85_envpad0_envinfldist0.08_edgestepsize0.005_envdistfact0.3_PRM_mujocorendereronly.hdf5"
                elif self.task_name == "square":
                    if self.dataset_path is None:
                        self.dataset_path = "/juno/u/thankyou/autom/consistency_policy/data/robomimic/datasets/augmented/square/demo_025trials/2023-11-26/3lj18amd/square_trials_25_se3aug-dx-0.08-0.08-dz-0.0-0.0-dthetaz0.0-0.0_startaug-eeposbound0.15-eerotbound0.79_envpad0_envinfldist0.08_edgestepsize0.005_envdistfact0.3_PRM_mujocorendereronly.hdf5"

        if self.plot_path is None:
            if self.plot_type == "human":
                self.plot_path = pathlib.Path(
                    "human_demo_dists.png"
                    if self.task_name is None
                    else f"human_{self.task_name}_demo_dists.png"
                )
            elif self.plot_type == "gen":
                self.plot_path = pathlib.Path(
                    "gen_demo_dists.png"
                    if self.task_name is None
                    else f"gen_{self.task_name}_demo_dists.png"
                )

            # add max_time_step to plot path
            if self.max_timestep > 0:
                self.plot_path = self.plot_path.parent / (
                    self.plot_path.stem
                    + f"_maxtimestep{self.max_timestep}"
                    + self.plot_path.suffix
                )

            if self.plot_path_extra_name is not None:
                self.plot_path = self.plot_path.parent / (
                    self.plot_path.stem
                    + f"_{self.plot_path_extra_name}"
                    + self.plot_path.suffix
                )

        assert (
            self.dataset_path is not None or self.task_name is not None
        ), "Either dataset_path or task_name must be given"


def main(cfg: Config):
    # Load the demo from the specified demo path
    demos: List[Demo] = load_demos(cfg.dataset_path, max_demos=cfg.max_demos)
    demos = demos[: cfg.max_demos]

    plot_demos_dists_3d(
        demos,
        save_path=cfg.plot_path,
        plot_quiver=cfg.plot_quiver,
        max_timestep=cfg.max_timestep,
    )


if __name__ == "__main__":
    tyro.cli(main)
