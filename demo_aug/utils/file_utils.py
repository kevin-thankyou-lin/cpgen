import pathlib
from typing import List

import h5py
import numpy as np


def count_total_demos(demo_files: List[pathlib.Path]) -> int:
    """
    Count the total number of demos in a list of K .hdf5 demo files.

    :param demo_files: List of paths to the K .hdf5 demo files.
    :type demo_files: List[pathlib.Path]
    :return: The total number of demos in the list of demo files.
    :rtype: int
    """
    total_demos = 0
    for demo_file in demo_files:
        with h5py.File(demo_file, "r") as file:
            src_demos = file["data"]
            for src_demo in src_demos.values():
                if "actions" in src_demo.keys():
                    total_demos += 1
    return total_demos


def merge_demo_files(demo_files: List[pathlib.Path], save_path: str):
    """
    Merges multiple K .hdf5 demo files into a single hdf5 file, including constraints if they exist.
    """
    pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    merged_file = h5py.File(save_path, "w")

    demo_index = 0  # start_idx for merged_file demos
    grp = merged_file.create_group("data")

    for idx, demo_file in enumerate(demo_files):
        print(f"idx: {idx}, demo_file: {demo_file}")
        with h5py.File(demo_file, "r") as file:
            src_demos = file["data"]

            # Merge any global attributes like wandb_run_url
            if "wandb_run_url" in file.attrs:
                if isinstance(file.attrs["wandb_run_url"], str):
                    src_wandb_url_lst = [file.attrs["wandb_run_url"]]
                else:
                    src_wandb_url_lst = file.attrs["wandb_run_url"].tolist()

                if "wandb_run_url" not in merged_file.attrs:
                    merged_file.attrs["wandb_run_url"] = src_wandb_url_lst
                else:
                    # Retrieve the existing attribute, convert to list, append, and reassign
                    existing_ids = (
                        merged_file.attrs["wandb_run_url"].tolist()
                        if isinstance(merged_file.attrs["wandb_run_url"], np.ndarray)
                        else [merged_file.attrs["wandb_run_url"]]
                    )
                    existing_ids.extend(src_wandb_url_lst)
                    merged_file.attrs["wandb_run_url"] = existing_ids

            # Copy any top-level attributes from src_demos to the merged 'data' group
            for k in src_demos.attrs.keys():
                grp.attrs[k] = src_demos.attrs[k]

            # Loop through each demo in the source file
            for demo_key in src_demos.keys():
                src_demo = src_demos[demo_key]
                if "actions" not in src_demo.keys():
                    print(
                        f"Skipping {demo_file} because it has no actions in {demo_key}"
                    )
                    continue

                # Create or get the new group in the merged file for this demo
                group_name = f"demo_{demo_index}"
                if group_name not in grp.keys():
                    grp.create_group(group_name)

                # Copy the top-level attributes from src_demo (if any) to the merged group
                for k, v in src_demo.attrs.items():
                    grp[group_name].attrs[k] = v

                # Copy the usual datasets like "actions", "states", "obs"
                grp[group_name]["actions"] = src_demo["actions"][()]
                if "states" in src_demo.keys():
                    grp[group_name]["states"] = src_demo["states"][()]
                else:
                    print(f"Warning: No states in {demo_file}/{demo_key}")

                # Copy per-timestep data, e.g. "timestep_0", "timestep_1", ...
                for key in src_demo.keys():
                    if key.startswith("timestep_"):
                        src_demo.copy(key, grp[group_name])

                # Copy observations if they exist
                if "obs" not in grp[group_name].keys() and "obs" in src_demo.keys():
                    grp[group_name].create_group("obs")
                if "obs" in src_demo.keys():
                    for obs_key in src_demo["obs"].keys():
                        grp[group_name]["obs"][obs_key] = src_demo["obs"][obs_key][()]

                # Copy constraint_data
                if "constraint_data" in src_demo.keys():
                    src_constraint_data_grp = src_demo["constraint_data"]
                    merged_constraint_data_grp = grp[group_name].create_group(
                        "constraint_data"
                    )

                    # Copy each constraint subgroup, e.g. 'constraint_0', 'constraint_1'
                    for c_key in src_constraint_data_grp.keys():
                        src_constraint_data = src_constraint_data_grp[c_key]
                        # Create the corresponding subgroup in the merged file
                        merged_constraint_data_grp.create_dataset(
                            c_key, data=src_constraint_data[()]
                        )

                # Optionally copy any top-level attributes from the *file's* "data" group into the new demo
                for attr_name, attr_value in src_demos.attrs.items():
                    grp.attrs[attr_name] = attr_value

                print(
                    f"Demo {demo_index} from file {demo_file} merged into {save_path}."
                )

                demo_index += 1

    merged_file.close()
    print(f"All demos merged into {save_path}")
    print(f"Total demos: {demo_index}")
