import dataclasses
import pathlib
from typing import List

import h5py
import matplotlib.pyplot as plt
import numpy as np
import tyro


def get_max_demos(dataset_paths: List[pathlib.Path]) -> int:
    max_demos_to_plot = np.inf
    if not isinstance(dataset_paths, List):
        dataset_paths = [dataset_paths]
    for dataset_path in dataset_paths:
        max_key_idx = 0
        with h5py.File(dataset_path, "r") as f:
            for key in f["data"].keys():
                max_key_idx = max(max_key_idx, int(key.split("_")[-1]))
        max_demos_to_plot = min(max_demos_to_plot, max_key_idx)
    return max_demos_to_plot


def extract_eef_positions(f, demo_index: int, timesteps: List[int]) -> np.ndarray:
    eef_positions = []

    for timestep in timesteps:
        try:
            ee_pos = f[f"data/demo_{demo_index}/obs/eef_pos"][()][timestep]
        except KeyError as e:
            print(f"KeyError: {e}; try to get eef_pos from robot0_eef_pos")
            ee_pos = f[f"data/demo_{demo_index}/obs/robot0_eef_pos"][()][timestep]

        eef_positions.append(ee_pos[:3])  # Only take x and y coordinates

    return np.array(eef_positions)


def plot_relative_eef_poses(
    dataset_path: pathlib.Path,
    src_eef_pose_idx: List[int] = [-1],
    save_dir: pathlib.Path = None,
    custom_save_path_prefix: str = "",
):
    """Plot the relative EEF poses at the given timesteps relative to the final EEF pose."""
    max_demos_to_plot = get_max_demos(dataset_path)
    print(f"Plotting {max_demos_to_plot} demos")

    # positions for now
    eef_poses_src_idx: List[np.ndarray] = []
    eef_poses_final_idx: List[np.ndarray] = []

    with h5py.File(dataset_path, "r") as f:
        for demo_index in range(max_demos_to_plot):
            eef_pos_src = extract_eef_positions(f, demo_index, [src_eef_pose_idx])
            eef_poses_src_idx.append(eef_pos_src)
            eef_pos_final = extract_eef_positions(f, demo_index, [-1])
            eef_poses_final_idx.append(eef_pos_final)

    # get the relative eef poses for each demo by subtracting the final eef pose
    eef_poses_relative = [
        eef_pos_src - eef_pos_final
        for eef_pos_src, eef_pos_final in zip(eef_poses_src_idx, eef_poses_final_idx)
    ]

    # plot in 3D
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    for eef_pos_rel in eef_poses_relative:
        ax.scatter(eef_pos_rel[:, 0], eef_pos_rel[:, 1], eef_pos_rel[:, 2])

    # also plot 0 0 0 for reference
    ax.scatter(0, 0, 0, color="red", label="Final EEF position")
    # also plot a 0.01 0.01 0.01 axes plot in xyz axes for reference
    ax.plot([0, 0.01], [0, 0], [0, 0], color="red", label="X axis")
    ax.plot([0, 0], [0, 0.01], [0, 0], color="green", label="Y axis")
    ax.plot([0, 0], [0, 0], [0, 0.01], color="blue", label="Z axis")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(
        f"Relative EEF Poses at Timestep {src_eef_pose_idx} Relative to Final EEF Pose"
    )

    # set aspect ratio to be equal
    ax.set_box_aspect([1, 1, 1])
    # set limits for each to be +- 0.1
    ax.set_xlim(-0.05, 0.05)
    ax.set_ylim(-0.05, 0.05)
    ax.set_zlim(-0.05, 0.05)

    plt.tight_layout()
    # make save_dir if it doesn't exist
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        save_dir
        / f"relative_eef_poses_timestep_{src_eef_pose_idx}{custom_save_path_prefix}.png"
    )
    print(
        f"Saved {save_dir / f'relative_eef_poses_timestep_{src_eef_pose_idx}{custom_save_path_prefix}.png'}"
    )

    # top down view
    ax.view_init(elev=90, azim=0)
    plt.tight_layout()
    ax.set_title(
        f"Relative EEF Poses at Timestep {src_eef_pose_idx} Relative to Final EEF Pose (Top Down)"
    )

    save_path = (
        save_dir
        / f"relative_eef_poses_timestep_{src_eef_pose_idx}_top_down{custom_save_path_prefix}.png"
    )
    plt.savefig(save_path)
    print(f"Saved {save_path}")

    # update view to side view
    ax.view_init(elev=0, azim=0)
    plt.tight_layout()
    ax.set_title(
        f"Relative EEF Poses at Timestep {src_eef_pose_idx} Relative to Final EEF Pose (Side View)"
    )

    save_path = (
        save_dir
        / f"relative_eef_poses_timestep_{src_eef_pose_idx}_side_view{custom_save_path_prefix}.png"
    )
    plt.savefig(save_path)
    print(f"Saved {save_path}")


@dataclasses.dataclass
class Config:
    dataset_path: pathlib.Path = "/scr/thankyou/autom/consistency-policy/data/real_world/datasets/augmented/wine_glass_hanging/grasp/narrow/demo_025trials/2024-07-20/nullwine_glass_hanging_grasp_narrow_trials25_se3aug-dx-0.05-0.05-dy-0.05-0.05-dz0.0-0.0-dthetz0.0-0.0-biassampzrot_staug_joint-jointqposnoise0.1_defcurobocfg_CUROBO_1j9so2i4_200demos_resized_renamed.hdf5"
    save_dir: pathlib.Path = pathlib.Path("relative_eef")
    src_eef_pos_idx: int = 27
    custom_save_path_prefix: str = ""


def main(cfg: Config):
    plot_relative_eef_poses(
        cfg.dataset_path,
        src_eef_pose_idx=cfg.src_eef_pos_idx,
        save_dir=cfg.save_dir,
        custom_save_path_prefix=cfg.custom_save_path_prefix,
    )


if __name__ == "__main__":
    tyro.cli(main)
