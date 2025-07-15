import copy

import h5py
import numpy as np
import robomimic.scripts.dataset_states_to_obs as dataset_states_to_obs_file
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.config import config_factory
from scipy.spatial.transform import Rotation


# Monkey patch robomimic get_camera_info to use v1.5.0
def get_camera_info(
    env,
    camera_names=None,
    camera_height=84,
    camera_width=84,
):
    """
    Helper function to get camera intrinsics and extrinsics for cameras being used for observations.
    """
    import robomimic.utils.env_utils as EnvUtils

    # TODO: make this function more general than just robosuite environments
    assert EnvUtils.is_robosuite_env(env=env)

    if camera_names is None:
        return None

    camera_info = dict()
    for cam_name in camera_names:
        K = env.get_camera_intrinsic_matrix(
            camera_name=cam_name, camera_height=camera_height, camera_width=camera_width
        )
        R = env.get_camera_extrinsic_matrix(
            camera_name=cam_name
        )  # camera pose in world frame
        if "eye_in_hand" in cam_name:
            # convert extrinsic matrix to be relative to robot eef control frame
            assert cam_name.startswith("robot0")
            eef_site_name = (
                env.base_env.robots[0]
                .composite_controller.part_controllers["right"]
                .ref_name
            )
            eef_pos = np.array(
                env.base_env.sim.data.site_xpos[
                    env.base_env.sim.model.site_name2id(eef_site_name)
                ]
            )
            eef_rot = np.array(
                env.base_env.sim.data.site_xmat[
                    env.base_env.sim.model.site_name2id(eef_site_name)
                ].reshape([3, 3])
            )
            eef_pose = np.zeros((4, 4))  # eef pose in world frame
            eef_pose[:3, :3] = eef_rot
            eef_pose[:3, 3] = eef_pos
            eef_pose[3, 3] = 1.0
            eef_pose_inv = np.zeros((4, 4))
            eef_pose_inv[:3, :3] = eef_pose[:3, :3].T
            eef_pose_inv[:3, 3] = -eef_pose_inv[:3, :3].dot(eef_pose[:3, 3])
            eef_pose_inv[3, 3] = 1.0
            R = R.dot(eef_pose_inv)  # T_E^W * T_W^C = T_E^C
        camera_info[cam_name] = dict(
            intrinsics=K.tolist(),
            extrinsics=R.tolist(),
        )
    return camera_info


dataset_states_to_obs_file.get_camera_info = get_camera_info


def playback_trajectory_with_env(
    env,
    initial_state,
    states,
    actions=None,
    render=False,
    video_writer=None,
    video_skip=5,
    camera_names=None,
    first=False,
    print_diffs=False,
):
    """
    Helper function to playback a single trajectory using the simulator environment.
    If @actions are not None, it will play them open-loop after loading the initial state.
    Otherwise, @states are loaded one by one.

    Args:
        env (instance of EnvBase): environment
        initial_state (dict): initial simulation state to load
        states (np.array): array of simulation states to load
        actions (np.array): if provided, play actions back open-loop instead of using @states
        render (bool): if True, render on-screen
        video_writer (imageio writer): video writer
        video_skip (int): determines rate at which environment frames are written to video
        camera_names (list): determines which camera(s) are used for rendering. Pass more than
            one to output a video with multiple camera views concatenated horizontally.
        first (bool): if True, only use the first frame of each episode.
    """
    write_video = video_writer is not None
    video_count = 0
    assert not (render and write_video)

    # load the initial state
    env.reset()
    env.reset_to(initial_state)

    traj_len = states.shape[0]
    action_playback = actions is not None
    if action_playback:
        assert states.shape[0] == actions.shape[0]

    for i in range(traj_len):
        if action_playback:
            obs, _, _, _ = env.step(actions[i])
            if print_diffs and i > 0:
                pos_diff = actions[i][:3] - obs["robot0_eef_pos"]
                print(f"Step {i}: pos diff: {pos_diff}")

            if i < traj_len - 1:
                # check whether the actions deterministically lead to the same recorded states
                state_playback = env.get_state()["states"]
                if not np.all(np.equal(states[i + 1], state_playback)):
                    err = np.linalg.norm(states[i + 1] - state_playback)
                    print("warning: playback diverged by {} at step {}".format(err, i))
        else:
            env.reset_to({"states": states[i]})

        # on-screen render
        if render:
            env.render(mode="human", camera_name=camera_names[0])

        from PIL import Image, ImageDraw

        # video render
        if write_video:
            if video_count % video_skip == 0:
                video_img = []
                for cam_name in camera_names:
                    video_img.append(
                        env.render(
                            mode="rgb_array",
                            height=128,
                            width=128,
                            camera_name=cam_name,
                        )
                    )
                video_img = np.concatenate(
                    video_img, axis=1
                )  # concatenate horizontally
                # Add frame number to the video image
                img_pil = Image.fromarray(video_img)
                draw = ImageDraw.Draw(img_pil)
                draw.text((10, 10), f"Frame: {video_count}", fill="white")
                video_img = np.array(img_pil)
                video_writer.append_data(video_img)
            video_count += 1

        if first:
            break


class RobomimicAbsoluteActionConverter:
    def __init__(self, dataset_path: str, algo_name: str = "bc"):
        # default BC config
        config = config_factory(algo_name=algo_name)

        # read config to set up metadata for observation modalities (e.g. detecting rgb observations)
        # must ran before create dataset
        ObsUtils.initialize_obs_utils_with_config(config)

        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
        abs_env_meta = copy.deepcopy(env_meta)
        abs_env_meta["env_kwargs"]["controller_configs"]["control_delta"] = False

        env = EnvUtils.create_env_from_metadata(
            env_meta=env_meta,
            render=False,
            render_offscreen=False,
            use_image_obs=False,
        )
        assert len(env.env.robots) in (1, 2)

        abs_env = EnvUtils.create_env_from_metadata(
            env_meta=abs_env_meta,
            render=False,
            render_offscreen=False,
            use_image_obs=False,
        )
        assert not abs_env.env.robots[0].controller.use_delta

        self.env = env
        self.abs_env = abs_env
        self.file = h5py.File(dataset_path, "r")

    def __len__(self):
        return len(self.file["data"])

    def convert_actions(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """
        Given state and delta action sequence
        generate equivalent goal position and orientation for each step
        keep the original gripper action intact.
        """
        # in case of multi robot
        # reshape (N,14) to (N,2,7)
        # or (N,7) to (N,1,7)
        stacked_actions = actions.reshape(*actions.shape[:-1], -1, 7)

        env = self.env
        # generate abs actions
        action_goal_pos = np.zeros(
            stacked_actions.shape[:-1] + (3,), dtype=stacked_actions.dtype
        )
        action_goal_ori = np.zeros(
            stacked_actions.shape[:-1] + (3,), dtype=stacked_actions.dtype
        )
        action_gripper = stacked_actions[..., [-1]]
        for i in range(len(states)):
            _ = env.reset_to({"states": states[i]})

            # taken from robot_env.py L#454
            for idx, robot in enumerate(env.env.robots):
                # run controller goal generator
                robot.control(stacked_actions[i, idx], policy_step=True)

                # read pos and ori from robots
                controller = robot.controller
                action_goal_pos[i, idx] = controller.goal_pos
                action_goal_ori[i, idx] = Rotation.from_matrix(
                    controller.goal_ori
                ).as_rotvec()

        stacked_abs_actions = np.concatenate(
            [action_goal_pos, action_goal_ori, action_gripper], axis=-1
        )
        abs_actions = stacked_abs_actions.reshape(actions.shape)
        return abs_actions

    def convert_idx(self, idx):
        file = self.file
        demo = file[f"data/demo_{idx}"]
        # input
        states = demo["states"][:]
        actions = demo["actions"][:]

        # generate abs actions
        abs_actions = self.convert_actions(states, actions)
        return abs_actions

    def convert_and_eval_idx(self, idx):
        env = self.env
        abs_env = self.abs_env
        file = self.file
        # first step have high error for some reason, not representative
        eval_skip_steps = 1

        demo = file[f"data/demo_{idx}"]
        # input
        states = demo["states"][:]
        actions = demo["actions"][:]

        # generate abs actions
        abs_actions = self.convert_actions(states, actions)

        # verify
        robot0_eef_pos = demo["obs"]["robot0_eef_pos"][:]
        robot0_eef_quat = demo["obs"]["robot0_eef_quat"][:]

        delta_error_info = self.evaluate_rollout_error(
            env,
            states,
            actions,
            robot0_eef_pos,
            robot0_eef_quat,
            metric_skip_steps=eval_skip_steps,
        )
        abs_error_info = self.evaluate_rollout_error(
            abs_env,
            states,
            abs_actions,
            robot0_eef_pos,
            robot0_eef_quat,
            metric_skip_steps=eval_skip_steps,
        )

        info = {"delta_max_error": delta_error_info, "abs_max_error": abs_error_info}
        return abs_actions, info

    @staticmethod
    def evaluate_rollout_error(
        env, states, actions, robot0_eef_pos, robot0_eef_quat, metric_skip_steps=1
    ):
        # first step have high error for some reason, not representative

        # evaluate abs actions
        rollout_next_states = list()
        rollout_next_eef_pos = list()
        rollout_next_eef_quat = list()
        obs = env.reset_to({"states": states[0]})
        for i in range(len(states)):
            obs = env.reset_to({"states": states[i]})
            obs, reward, done, info = env.step(actions[i])
            obs = env.get_observation()
            rollout_next_states.append(env.get_state()["states"])
            rollout_next_eef_pos.append(obs["robot0_eef_pos"])
            rollout_next_eef_quat.append(obs["robot0_eef_quat"])
        rollout_next_states = np.array(rollout_next_states)
        rollout_next_eef_pos = np.array(rollout_next_eef_pos)
        rollout_next_eef_quat = np.array(rollout_next_eef_quat)

        next_state_diff = states[1:] - rollout_next_states[:-1]
        max_next_state_diff = np.max(np.abs(next_state_diff[metric_skip_steps:]))

        next_eef_pos_diff = robot0_eef_pos[1:] - rollout_next_eef_pos[:-1]
        next_eef_pos_dist = np.linalg.norm(next_eef_pos_diff, axis=-1)
        max_next_eef_pos_dist = next_eef_pos_dist[metric_skip_steps:].max()

        next_eef_rot_diff = (
            Rotation.from_quat(robot0_eef_quat[1:])
            * Rotation.from_quat(rollout_next_eef_quat[:-1]).inv()
        )
        next_eef_rot_dist = next_eef_rot_diff.magnitude()
        max_next_eef_rot_dist = next_eef_rot_dist[metric_skip_steps:].max()

        info = {
            "state": max_next_state_diff,
            "pos": max_next_eef_pos_dist,
            "rot": max_next_eef_rot_dist,
        }
        return info

    def close(self):
        self.env.env.close()
        self.abs_env.env.close()
        self.file.close()
