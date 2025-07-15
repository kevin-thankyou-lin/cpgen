import dataclasses
import pathlib
from typing import List, Literal, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import tyro
from scipy.spatial.transform import Rotation as R


def get_max_demos(dataset_paths: List[pathlib.Path]) -> int:
    max_demos_to_plot = np.inf
    for dataset_path in dataset_paths:
        max_key_idx = 0
        with h5py.File(dataset_path, "r") as f:
            for key in f["data"].keys():
                max_key_idx = max(max_key_idx, int(key.split("_")[-1]))
        max_demos_to_plot = min(max_demos_to_plot, max_key_idx)
    return max_demos_to_plot


def extract_relative_poses(
    f, demo_index: int, timesteps: List[int]
) -> Tuple[np.ndarray, np.ndarray]:
    obj_pos = f[f"data/demo_{demo_index}/states"][()][0, 10:13]
    obj_quat_wxyz = f[f"data/demo_{demo_index}/states"][()][0, 13:17]

    relative_positions = []
    relative_rots = []

    for timestep in timesteps:
        ee_pos = f[f"data/demo_{demo_index}/obs/robot0_eef_pos"][()][timestep]
        ee_quat_xyzw = f[f"data/demo_{demo_index}/obs/robot0_eef_quat"][()][timestep]
        obj_quat_xyzw = obj_quat_wxyz[[1, 2, 3, 0]]

        obj_rot = R.from_quat(obj_quat_xyzw)
        ee_rot = R.from_quat(ee_quat_xyzw)

        relative_pos = obj_rot.inv().apply(ee_pos - obj_pos)
        relative_rot = obj_rot.inv() * ee_rot

        relative_euler = relative_rot.as_euler("xyz")

        relative_positions.append(relative_pos)
        relative_rots.append(relative_euler)

    return np.array(relative_positions), np.array(relative_rots)


def plot_relative_positions(
    ax,
    relative_positions: np.ndarray,
    mid_x: float,
    mid_y: float,
    mid_z: float,
    max_range: float,
):
    ax.scatter(
        relative_positions[:, 0],
        relative_positions[:, 1],
        relative_positions[:, 2],
        label="Relative Position",
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.legend()


def plot_relative_position_at_grasp(ax, pt_x: float, pt_y: float, pt_z: float):
    ax.scatter(pt_x, pt_y, pt_z, label="Grasp Relative Position", color="red", s=60)
    ax.legend()


def plot_relative_poses(
    dataset_paths: List[pathlib.Path],
    titles: List[str],
    demo_timesteps: List[int] = [-1],
    task: Literal["lift", "square"] = "square",
):
    max_demos_to_plot = get_max_demos(dataset_paths)
    print(f"Plotting {max_demos_to_plot} demos")

    relative_positions_all_datasets = [[] for _ in dataset_paths]
    relative_rots_all_datasets = [[] for _ in dataset_paths]

    for dataset_idx, dataset_path in enumerate(dataset_paths):
        with h5py.File(dataset_path, "r") as f:
            for demo_index in range(max_demos_to_plot):
                rel_pos, rel_rot = extract_relative_poses(f, demo_index, demo_timesteps)
                relative_positions_all_datasets[dataset_idx].append(rel_pos)
                relative_rots_all_datasets[dataset_idx].append(rel_rot)
                # print(f"rel_pos: {rel_pos}")

    if task == "lift":
        max_range = 0.3
        mid_x, mid_y, mid_z = 0.07382217566221545, -0.004, 0.00262
    elif task == "square":
        max_range = 0.1
        mid_x, mid_y, mid_z = 0.07382217566221545, -0.004, 0.00262
    else:
        raise ValueError(f"Unknown task: {task}")

    fig, axs = plt.subplots(
        len(demo_timesteps),
        len(dataset_paths),
        figsize=(5 * len(dataset_paths), 5 * len(demo_timesteps)),
        subplot_kw={"projection": "3d"},
    )

    for ax_row in axs:
        for ax in ax_row:
            ax.view_init(elev=90, azim=0)  # Set the top-down view

    for timestep_idx, timestep in enumerate(demo_timesteps):
        print(f"Plotting timestep {timestep}")
        print(f"max_range: {max_range}")
        print(f"mid_x: {mid_x}, mid_y: {mid_y}, mid_z: {mid_z}")
        for dataset_idx in range(len(dataset_paths)):
            timestep_relative_positions = np.array(
                [
                    pos[timestep_idx]
                    for pos in relative_positions_all_datasets[dataset_idx]
                ]
            )
            plot_relative_positions(
                axs[timestep_idx][dataset_idx],
                timestep_relative_positions,
                mid_x,
                mid_y,
                mid_z,
                max_range,
            )
            axs[timestep_idx][dataset_idx].set_title(
                f"{titles[dataset_idx]}: Timestep {timestep}"
            )
            plot_relative_position_at_grasp(
                axs[timestep_idx][dataset_idx],
                np.array(relative_positions_all_datasets[dataset_idx])[..., -1, 0],
                np.array(relative_positions_all_datasets[dataset_idx])[..., -1, 1],
                np.array(relative_positions_all_datasets[dataset_idx])[..., -1, 2],
            )

    # convert demo_timesteps to string
    demo_timesteps = [str(t) for t in demo_timesteps]

    plt.tight_layout()
    plt.savefig(f'relative_pos_timestep_{"_".join(demo_timesteps)}.png')
    print(f'Saved relative_pos_timestep_{"_".join(demo_timesteps)}.png')


@dataclasses.dataclass
class Config:
    task: Literal["lift", "square"] = "square"


def main(cfg: Config):
    gen_dataset_path_post_mjc_success_failure = pathlib.Path(
        "/scr/thankyou/autom/consistency-policy/data/robomimic/datasets/augmented/square/demo_025trials/2024-05-17-grasp-jnt-rand-gr-non-fix-mjc-still-fails/square_trials_25_se3aug-dx-0.2--0.07-dy0.05-0.28-dz0.829958-0.829958-dthetaz-2.32-3.92_startaug-eeposbound0.1-eerotbound18022.522.5_default_curobo_cfg_CUROBO_mujocorendereronly_200demos.hdf5"
    )
    gen_dataset_path_mjc_success = pathlib.Path(
        "/scr/thankyou/autom/consistency-policy/data/robomimic/datasets/augmented/square/demo_025trials/2024-05-07-mjc-render-works!/square_trials_25_se3aug-dx-0.2--0.07-dz0.829958-0.829958-dthetaz-2.32-3.92_startaug-eeposbound0.1-eerotbound0.3925_default_curobo_cfg_CUROBO_mujocorendereronly_200demos.hdf5"
    )
    gen_dataset_path_gsplat_render_failed = pathlib.Path(
        "/scr/thankyou/autom/consistency-policy/data/robomimic/datasets/augmented/square/demo_025trials/2024-05-17/square_trials_25_se3aug-dx-0.2--0.07-dy0.05-0.28-dz0.829958-0.829958-dthetaz-2.32-3.92_startaug-eeposbound0.1-eerotbound18022.522.5_default_curobo_cfg_CUROBO_200demos.hdf5"
    )
    human_dataset_path = pathlib.Path(
        "/juno/u/thankyou/autom/diffusion_policy/data/robomimic/datasets/square/ph/low_dim_abs.hdf5"
    )
    # dataset_paths = [gen_dataset_path, gen_dataset_path_mjc_success]
    gen_dataset_path_regen_from_ws_17_good_train_SR = pathlib.Path(
        "/scr/thankyou/autom/demo-aug/square_trials_25_se3aug-dx-0.2--0.07-dz0.829958-0.829958-dthetaz-2.32-3.92_startaug-eeposbound0.1-eerotbound0.3925_default_curobo_cfg_CUROBO_mujocorendereronly_200demos.hdf5"
    )
    gen_dataset_path_225_gsplat_success = pathlib.Path(
        "/scr/thankyou/autom/consistency-policy/data/robomimic/datasets/augmented/square/demo_025trials/2024-05-07-extra-gsplat/square_trials_25_se3aug-dx-0.2--0.07-dz0.829958-0.829958-dthetaz-2.32-3.92_startaug-eeposbound0.1-eerotbound0.3925_default_curobo_cfg_CUROBO_225demos.hdf5"
    )
    gen_dataset_extra_close_action = pathlib.Path(
        "/scr/thankyou/autom/consistency-policy/data/robomimic/datasets/augmented/square/demo_025trials/2024-05-19/square_trials_25_se3aug-dx-0.2--0.07-dy0.05-0.28-dz0.829958-0.829958-dthetaz-2.32-3.92_startaug-eeposbound0.1-eerotbound18022.522.5_default_curobo_cfg_CUROBO_mujocorendereronly_200demos.hdf5"
    )
    gen_94_100_replace_last_trajopt_w_ik = pathlib.Path(
        "/scr/thankyou/autom/consistency-policy/data/robomimic/datasets/augmented/square/grasp/demo_025trials/2024-05-20-94-100-replace-last-trajopt-w-ik/square_trials_25_se3aug-dx-0.2--0.07-dy0.05-0.28-dz0.829958-0.829958-dthetaz-2.32-3.92_startaug-eeposbound0.1-eerotbound18022.522.5_default_curobo_cfg_CUROBO_mujocorendereronly_200demos.hdf5"
    )
    dataset_paths = [
        # gen_dataset_path,
        gen_dataset_extra_close_action,
        gen_dataset_path_post_mjc_success_failure,
        gen_dataset_path_gsplat_render_failed,
        gen_dataset_path_225_gsplat_success,
        gen_dataset_path_regen_from_ws_17_good_train_SR,
        gen_dataset_path_mjc_success,
        gen_94_100_replace_last_trajopt_w_ik,
    ]

    lift_generated = pathlib.Path(
        "/scr/thankyou/autom/consistency-policy/data/robomimic/datasets/augmented/square/demo_025trials/2024-05-16/square_trials_25_se3aug-dx-0.2--0.07-dy0.05-0.28-dz0.829958-0.829958-dthetaz-2.32-3.92_startaug-eeposbound0.1-eerotbound18022.522.5_default_curobo_cfg_CUROBO_200demos.hdf5"
    )
    # generated_dataset_path = pathlib.Path("/scr/thankyou/autom/consistency-policy/data/robomimic/datasets/augmented/square/demo_025trials/2024-05-07-extra-gsplat/square_trials_25_se3aug-dx-0.2--0.07-dz0.829958-0.829958-dthetaz-2.32-3.92_startaug-eeposbound0.1-eerotbound0.3925_default_curobo_cfg_CUROBO_225demos.hdf5")
    lift_human = pathlib.Path(
        "/juno/u/thankyou/autom/diffusion_policy/data/robomimic/datasets/lift/ph/low_dim_abs.hdf5"
    )

    square_generated_start_near_goal = pathlib.Path(
        "/scr/thankyou/autom/consistency-policy/data/robomimic/datasets/augmented/square/grasp/wide/demo_05trials/2024-05-28/6ma4b8wt/square_grasp_wide_trials5_scaleaug-scalerng0.8-1.1_se3aug-dx-0.2--0.07-dy0.05-0.28-dz0.829958-0.829958-dthetaz-2.32-3.92_startaug-eeposbnd0.05-eerotbnd901212_defcurobocfg_CUROBO_mjcrendonly_6ma4b8wt.hdf5"
    )
    square_generated_start_near_goal_bnd_0pt1 = pathlib.Path(
        "/scr/thankyou/autom/consistency-policy/data/robomimic/datasets/augmented/square/grasp/wide/demo_025trials/2024-05-28/m6vlxgnn/square_grasp_wide_trials25_scaleaug-scalerng0.8-1.1_se3aug-dx-0.2--0.07-dy0.05-0.28-dz0.829958-0.829958-dthetaz-2.32-3.92_startaug-eeposbnd0.1-eerotbnd901212_defcurobocfg_CUROBO_mjcrendonly_m6vlxgnn.hdf5"
    )
    square_generated_start_near_goal_bnd_0pt1_eef_centered = pathlib.Path(
        "/scr/thankyou/autom/consistency-policy/data/robomimic/datasets/augmented/square/grasp/wide/demo_010trials/2024-05-29/hzihg6ba/square_grasp_wide_trials10_scaleaug-scalerng0.8-1.1_se3aug-dx-0.2--0.07-dy0.05-0.28-dz0.829958-0.829958-dthetaz-2.32-3.92_startaug-eeposbnd0.1-eerotbnd901212_defcurobocfg_CUROBO_mjcrendonly_hzihg6ba.hdf5"
    )

    # dataset_paths = [lift_human, lift_generated]
    print(f"dataset_paths: {dataset_paths}")
    dataset_paths = [human_dataset_path, square_generated_start_near_goal_bnd_0pt1]
    print(f"dataset_paths: {dataset_paths}")
    # get title from python variable name
    titles = []
    locals_lst = locals().copy()
    for var_name in locals_lst:
        if locals_lst[var_name] in dataset_paths:
            titles.append(var_name)
    print(titles)
    print(f"dataset_paths: {dataset_paths}")

    demo_timesteps = [-30, -25, -18, -14, -11, -10, -9, -8, -7]
    plot_relative_poses(
        dataset_paths, titles=titles, demo_timesteps=demo_timesteps, task=cfg.task
    )


if __name__ == "__main__":
    tyro.cli(main)
