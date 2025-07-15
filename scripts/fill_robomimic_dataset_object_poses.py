"""
Set robomimic environment to a particular state, read out object poses and update robomimic demonstration data hdf5
"""

import pathlib
from collections import defaultdict
from typing import Dict, List

import h5py
import numpy as np
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils


class RobomimicEnvironmentHandler:
    def __init__(self, dataset_path: pathlib.Path):
        self.dataset_path = dataset_path
        self.env = self.setup_environment()

    def setup_environment(self):
        """
        Set up the Robomimic environment based on the dataset's metadata.
        """
        obs_spec = {
            "obs": {
                "low_dim": ["robot0_eef_pos"],  # Specify other observations if needed
                "rgb": [],
            },
        }
        ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=obs_spec)
        env_meta = FileUtils.get_env_metadata_from_dataset(
            dataset_path=self.dataset_path
        )
        return EnvUtils.create_env_from_metadata(
            env_meta=env_meta, render_offscreen=True, use_image_obs=False
        )

    def set_env_state_and_extract_obs(
        self,
        state: np.ndarray,
        model: str,
        let_env_settle: bool = False,
        obs_after_settling: List[str] = ["SquareNut_pos", "SquareNut_quat"],
    ) -> Dict[str, np.ndarray]:
        """
        Set the environment to a given state and extract observations.
        """
        self.env.reset_to({"states": state, "model_file": model})
        obs = (
            self.env.env._get_observations()
        )  # Assumes the environment has a method `get_obs`
        if let_env_settle:
            # let env settle
            for _ in range(10):
                self.env.env.step(np.zeros(self.env.env.action_dim))
            new_obs = self.env.env._get_observations()
            for key in obs_after_settling:
                obs[key] = new_obs[key]

        return obs


def update_demo_obs_data(demo: h5py.Group, new_observations_obj: Dict[str, np.ndarray]):
    """
    Update the demonstration data with the new observations.
    """
    # loop through the keys in the new observations and update the demo group's observations
    for key, value in new_observations_obj.items():
        # create a new dataset for the key if it doesn't exist
        assert key not in demo["obs"], f"Key {key} already exists in the demo group"
        demo["obs"].create_dataset(key, data=value)
        # demo["obs"][key][:] = value


if __name__ == "__main__":
    # Example usage
    dataset_path = "../diffusion_policy/data/robomimic/datasets/lift/ph/1_demo.hdf5"
    task = "lift"
    env_handler = RobomimicEnvironmentHandler(dataset_path=dataset_path)

    # open up dataset and read out the states
    with h5py.File(dataset_path, "r+") as file:
        src_demos = file["data"]
        for src_demo_key in src_demos:
            src_demo = src_demos[src_demo_key]
            model = src_demo.attrs["model_file"]
            states = src_demo["states"]

            dummy_state = states[0]

            env_obs: Dict[str, np.ndarray] = env_handler.set_env_state_and_extract_obs(
                dummy_state, model
            )
            print(f"Environment observations: {env_obs}")

            # compare to the existing observations in the demo
            curr_dataset_obs = list(src_demo["obs"].keys())
            print(f"Current dataset observations: {curr_dataset_obs}")
            # get the keys that are in env_obs but not in curr_dataset_obs
            new_keys = set(env_obs.keys()) - set(curr_dataset_obs)
            print(f"New keys: {new_keys}")

            # get a list of all the values corresping to the new keys
            new_observations = defaultdict(list)
            for t_idx, state in enumerate(states):
                let_env_settle = False
                if task == "square":
                    let_env_settle = True if t_idx < 20 else False
                elif task == "can":
                    let_env_settle = True if t_idx < 20 else False

                # set the environment to a particular state and extract observations
                env_obs: Dict[str, np.ndarray] = (
                    env_handler.set_env_state_and_extract_obs(
                        state,
                        model,
                        let_env_settle=let_env_settle,
                        obs_after_settling=new_keys,
                    )
                )
                # for keys in new_keys, append the values to the new_observations dict
                for key in new_keys:
                    new_observations[key].append(env_obs[key])

            # now update the dataset with the new observations
            update_demo_obs_data(src_demo, new_observations)
