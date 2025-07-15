import dataclasses
import pathlib
from typing import List, Literal, Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import tyro
from scipy.spatial.transform import Rotation as R


def create_env(
    dataset_path: pathlib.Path,
):
    # We fill in the automatic values
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=dataset_path)

    # need to make sure ObsUtils knows which observations are images, but it doesn't matter
    # for playback since observations are unused. Pass a dummy spec here.
    dummy_spec = dict(
        obs=dict(
            low_dim=["robot0_eef_pos"],
            rgb=[],
        ),
    )
    ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=dummy_spec)
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=dataset_path)
    # convert old controller_configs to new format
    if (
        env_meta["env_kwargs"]["controller_configs"].get(
            "body_parts_controller_configs"
        )
        is not None
    ):
        env_meta["env_kwargs"]["controller_configs"]["body_parts"] = env_meta[
            "env_kwargs"
        ]["controller_configs"]["body_parts_controller_configs"]

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta, render=False, render_offscreen=True
    )
    return env


def plot_reset_distributions(
    dataset_paths: List[pathlib.Path],
    task: Literal["lift", "square"] = "square",
    save_path: str = "reset_poses.png",
):
    if task == "lift":
        xlim = [-0.04, 0.04]
        ylim = [-0.04, 0.04]
    elif task == "square":
        xlim = [-0.115, -0.11]
        xlim = [-0.13, -0.1]
        ylim = [0.09, 0.24]
    else:
        xlim = [-0.14, -0.08]
        ylim = [0.05, 0.3]

    from demo_aug.utils.mujoco_utils import get_body_pose

    square_poses = []
    rewards = []

    for dataset_path in dataset_paths:
        env = create_env(dataset_path)
        with h5py.File(dataset_path, "r") as f:
            for i in range(len(f["data"].keys())):
                # reset env to the first state, then get the obs of the task relevant objects
                env.reset_to({"states": f[f"data/demo_{i}/states"][()][0]})
                squarenut_pos, squarenut_quat_wxyz = get_body_pose(
                    env.env.sim.data._data, env.env.sim.model._model, "SquareNut_main"
                )
                squarenut_quat_xyzw = squarenut_quat_wxyz[[1, 2, 3, 0]]
                if "rewards" in f[f"data/demo_{i}"].keys():
                    max_reward = np.max(f[f"data/demo_{i}/rewards"][()])
                    rewards.append(max_reward)
                else:
                    rewards.append(0.5)

                square_poses.append((squarenut_pos.copy(), squarenut_quat_xyzw.copy()))

    # Extract yaw angles
    gt_yaw_angles = [
        R.from_quat(quat).as_euler("xyz")[-1] for pos, quat in square_poses
    ]
    # Extract positions
    gt_positions = np.array([pos for pos, quat in square_poses])

    # Plotting the quiver plot
    fig, ax = plt.subplots(figsize=(6, 6))

    rewards = np.array(rewards)
    gt_positions = gt_positions
    gt_yaw_angles = gt_yaw_angles
    rewards = rewards

    colors = [
        "red" if reward == 0 else "green" if reward == 1 else "black"
        for reward in rewards
    ]
    # Add legend for colors
    red_patch = plt.Line2D(
        [], [], color="red", marker="o", markersize=5, label="reward=0"
    )
    green_patch = plt.Line2D(
        [], [], color="green", marker="o", markersize=5, label="reward=1"
    )
    black_patch = plt.Line2D(
        [], [], color="black", marker="o", markersize=5, label="reward=None"
    )
    # Modify legend font size
    legend_font_size = 8
    ax.legend(
        handles=[red_patch, green_patch, black_patch],
        loc="upper right",
        prop={"size": legend_font_size},
    )
    ax.quiver(
        gt_positions[:, 0],
        gt_positions[:, 1],
        np.cos(gt_yaw_angles),
        np.sin(gt_yaw_angles),
        color=colors,
        scale=50,  # Decrease the size of the quiver
        headwidth=4,
        headlength=2,
        headaxislength=2,
    )

    ax.legend(
        handles=[red_patch, green_patch, black_patch],
        loc="upper right",
        prop={"size": legend_font_size},
    )

    for i, reward in enumerate(rewards):
        if reward == 0:
            ax.annotate(
                str(i),
                (gt_positions[i, 0], gt_positions[i, 1]),
                color="red",
                fontsize=2,
            )

    ax.set_title("Ground Truth Env Reset Poses")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    plt.tight_layout()
    plt.savefig(save_path, dpi=700)
    print(f"Saved reset poses plot to {save_path} with increased resolution")


@dataclasses.dataclass
class Config:
    dataset_paths: List[pathlib.Path]
    task: Literal["lift", "square"] = "lift"
    plot_save_path: Optional[str] = None

    def __post_init__(self):
        if self.plot_save_path is None:
            self.plot_save_path = f"{self.dataset_paths[0].parent}/{self.dataset_paths[0].stem}_reset_poses.png"


def main(cfg: Config):
    plot_reset_distributions(
        dataset_paths=cfg.dataset_paths, task=cfg.task, save_path=cfg.plot_save_path
    )


if __name__ == "__main__":
    tyro.cli(main)
