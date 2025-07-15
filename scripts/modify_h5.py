import argparse

import h5py
import numpy as np


def modify_h5_actions(file_path, demo_name, start_index, end_index):
    with h5py.File(file_path, "r+") as f:
        dataset = f[f"data/{demo_name}/actions"]

        # Get the starting XY values and Z value
        start_xy = dataset[start_index, 0:2]
        start_z = dataset[start_index, 2]
        end_z = 1.0

        num_steps = end_index - start_index + 1
        z_values = np.linspace(start_z, end_z, num_steps)

        # Update the actions
        for i in range(start_index, end_index + 1):
            new_action = np.copy(dataset[i])
            new_action[0:2] = start_xy
            new_action[2] = z_values[i - start_index]
            dataset[i] = new_action

        print("Updated actions:")
        for i in range(start_index, end_index + 1):
            print(f"  {i}: {dataset[i, 0:3]}")

        f.flush()


def modify_h5_observations(file_path, demo_name, start_index, end_index):
    with h5py.File(file_path, "r+") as f:
        dataset = f[f"data/{demo_name}/obs/robot0_eef_pos"]
        start_xy = dataset[start_index, 0:2]
        start_z = dataset[start_index, 2]
        end_z = 1.0  # hardcoded to 1.0; originally for modifying the can task actions to move up more

        num_steps = end_index - start_index + 1
        z_values = np.linspace(start_z, end_z, num_steps)

        # Update the observations
        for i in range(start_index, end_index + 1):
            new_obs = np.copy(dataset[i])
            new_obs[0:2] = start_xy
            new_obs[2] = z_values[i - start_index]
            dataset[i] = new_obs

        print("Updated observations:")
        for i in range(start_index, end_index + 1):
            print(f"  {i}: {dataset[i]}")

        f.flush()


def modify_h5_observations_drop(file_path, demo_name, start_index, end_index):
    with h5py.File(file_path, "r+") as f:
        dataset = f[f"data/{demo_name}/obs/robot0_eef_pos"]
        start_x = 0.16
        end_x = 0.17
        start_y = 0.33
        end_y = 0.34  # New end_x is 0.35 further

        z = dataset[start_index, 2]

        num_steps = end_index - start_index + 1
        y_values = np.linspace(start_y, end_y, num_steps)
        x_values = np.linspace(start_x, end_x, num_steps)

        # Update the observations
        for i in range(start_index, end_index + 1):
            new_obs = np.copy(dataset[i])
            new_obs[0] = x_values[i - start_index] + np.random.uniform(-0.01, 0.01)
            new_obs[1] = y_values[i - start_index] + np.random.uniform(-0.005, 0.005)
            new_obs[2] = z
            dataset[i] = new_obs

        print("Updated observations:")
        for i in range(start_index, end_index + 1):
            print(f"  {i}: {dataset[i]}")

        f.flush()


def main():
    parser = argparse.ArgumentParser(description="Modify H5 file")
    parser.add_argument(
        "--file_path",
        help="Path to the H5 file",
        type=str,
        default="../diffusion_policy/data/robomimic/datasets/can/ph/1_demo_from_robomimic_backup.hdf5",
    )
    parser.add_argument(
        "--demo_name", help="Name of the demo", type=str, default="demo_0"
    )
    parser.add_argument("--start_index", type=int, help="Start index", default=57)
    parser.add_argument("--end_index", type=int, help="End index", default=69)

    args = parser.parse_args()

    np.set_printoptions(precision=3, suppress=True)
    # modify_h5_actions(args.file_path, args.demo_name,
    #           args.start_index, args.end_index)
    modify_h5_observations(
        args.file_path, args.demo_name, args.start_index, args.end_index
    )

    print("Hardcoded for the can drop subtask")
    import ipdb

    ipdb.set_trace()
    start_idx = 105
    end_idx = 117
    modify_h5_observations_drop(args.file_path, args.demo_name, start_idx, end_idx)


if __name__ == "__main__":
    main()
