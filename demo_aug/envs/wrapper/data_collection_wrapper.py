"""
This file implements a wrapper for saving simulation states to disk.
This data collection wrapper is useful for collecting demonstrations.
"""

import os
import pathlib
import time
from collections import defaultdict
from typing import Dict, List

import numpy as np

from demo_aug.envs.wrapper.wrapper import Wrapper


class DataCollectionWrapper(Wrapper):
    def __init__(
        self, env, directory: pathlib.Path, collect_freq: int = 1, flush_freq: int = 100
    ):
        """
        Initializes the data collection wrapper.

        Args:
            env (NeRFEnv): The environment to monitor.
            directory (str): Where to store collected data.
            collect_freq (int): How often to save simulation state, in terms of environment steps.
            flush_freq (int): How frequently to dump data to disk, in terms of environment steps.
        """
        super().__init__(env)

        # the base directory for all logging
        self.directory = directory

        # in-memory cache for simulation states and action info
        self.states: Dict[str, List] = defaultdict(list)
        self.action_infos = []  # stores information about actions taken
        self.rewards = []  # stores rewards from the environment
        self.dones = []  # stores dones from the environment
        self.obs: Dict[str, List] = defaultdict(
            list
        )  # stores observations from the environment

        # how often to save simulation state, in terms of environment steps
        self.collect_freq = collect_freq

        # how frequently to dump data to disk, in terms of environment steps
        self.flush_freq = flush_freq

        directory.mkdir(parents=True, exist_ok=True)

        # store logging directory for current episode
        self.ep_directory = None

        # remember whether any environment interaction has occurred
        self.has_interaction = False

        # some variables for remembering the current episode's initial state and model xml
        self._current_task_instance_state = None

    def _start_new_episode(self):
        """
        Bookkeeping to do at the start of each new episode.
        """

        # flush any data left over from the previous episode if any interactions have happened
        if self.has_interaction:
            self._flush()

        # timesteps in current episode
        self.t = 0
        self.has_interaction = False

    def _flush(self):
        """
        Method to flush internal state to disk.
        """
        if len(self.states) == 0:
            print("No data to flush.")
            return

        t1, t2 = str(time.time()).split(".")
        state_path = os.path.join(self.ep_directory, "state_{}_{}.npz".format(t1, t2))
        if hasattr(self.env, "unwrapped"):
            env_name = self.env.unwrapped.__class__.__name__
        else:
            env_name = self.env.__class__.__name__
        np.savez(
            state_path,
            states=np.array(self.states),
            action_infos=self.action_infos,
            rewards=np.array(self.rewards),
            dones=np.array(self.dones),
            obs=self.obs,
            env=env_name,
        )

        print("self.states: ", len(self.states["robot_pose"]))
        print("len(self.action_infos): ", len(self.action_infos))
        print("len(self.rewards): ", len(self.rewards))
        print("len(self.dones): ", len(self.dones))
        print("self.obs['agentview_image']: ", len(self.obs["agentview_image"]))
        self.states = defaultdict(list)
        self.obs = defaultdict(list)
        self.action_infos = []

    def reset(self):
        """
        Extends vanilla reset() function call to accommodate data collection

        Returns:
            OrderedDict: Environment observation space after reset occurs
        """
        ret = super().reset()
        self._start_new_episode()
        return ret

    def step(self, action):
        """
        Extends vanilla step() function call to accommodate data collection

        Args:
            action (np.array): Action to take in environment

        Returns:
            4-tuple:

                - (OrderedDict) observations from the environment
                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) misc information
        """
        ret = super().step(action)
        self.t += 1

        # on the first time step, make directories for logging
        if not self.has_interaction:
            self._on_first_interaction()

        # collect the current simulation state if necessary
        if self.t % self.collect_freq == 0:
            state = self.env.get_env_state()
            obs = self.env.render()
            self.obs["agentview_image"].append(obs["agentview_image"].copy())
            self.obs["agentview_c2w"].append(obs["agentview_c2w"].copy())
            self.states["robot_pose"].append(state["robot_pose"].copy())
            self.states["target_pose"].append(state["target_pose"].copy())
            # seems like states can be a dict as convert_robotmimic_to_replay doesn't explicitly do anything to states

            info = {}
            info["actions"] = np.array(action)
            self.action_infos.append(info)

        # flush collected data to disk if necessary
        if self.t % self.flush_freq == 0:
            self._flush()

        return ret

    def close(self):
        """
        Override close method in order to flush left over data
        """
        if self.has_interaction:
            self._flush()

    def _on_first_interaction(self):
        """
        Bookkeeping for first timestep of episode.
        This function is necessary to make sure that logging only happens after the first
        step call to the simulation, instead of on the reset (people tend to call
        reset more than is necessary in code).

        Raises:
            AssertionError: [Episode path already exists]
        """

        self.has_interaction = True

        # create a directory with a timestamp
        t1, t2 = str(time.time()).split(".")
        self.ep_directory = os.path.join(self.directory, "ep_{}_{}".format(t1, t2))
        assert not os.path.exists(self.ep_directory)
        print("DataCollectionWrapper: making folder at {}".format(self.ep_directory))
        os.makedirs(self.ep_directory)

        # save initial state and action
        assert len(self.states) == 0
        state = self.env.get_env_state()
        obs = self.env.render()
        try:
            self.obs["agentview_image"].append(obs["agentview_image"].copy())
            self.obs["agentview_c2w"].append(obs["agentview_c2w"].copy())
            self.states["robot_pose"].append(state["robot_pose"].copy())
            self.states["target_pose"].append(state["target_pose"].copy())
        except Exception:
            import ipdb

            ipdb.set_trace()
            self.obs["agentview_image"].append(obs["agentview_image"].copy())
            self.obs["agentview_c2w"].append(obs["agentview_c2w"].copy())
            self.states["robot_pose"].append(state["robot_pose"].copy())
            self.states["target_pose"].append(state["target_pose"].copy())
