"""
A convenience script to playback random demonstrations from
a set of demonstrations stored in a hdf5 file.

Arguments:
    --path (str): Path to demonstrations
    --use-actions (optional): If this flag is provided, the actions are played back
        through the MuJoCo simulator, instead of loading the simulator states
        one by one.
    --visualize-gripper (optional): If set, will visualize the gripper site

Example:
    $ python playback_demonstrations_from_hdf5.py --path ../models/assets/demonstrations/SawyerPickPlace/
"""

import json
import pathlib
from collections import defaultdict
from dataclasses import dataclass

import h5py
import imageio
import numpy as np
import robosuite
import robosuite.utils.transform_utils as T
import torch
import trimesh
import tyro
from PIL import Image
from robosuite.utils.mjcf_utils import postprocess_model_xml
from scipy.spatial.transform import Rotation as R

from demo_aug.utils.mathutils import multiply_with_X_transf
from demo_aug.utils.robomimic_utils import RobomimicAbsoluteActionConverter


def get_state(env):
    """
    Get current environment simulator state as a dictionary. Should be compatible with @reset_to.
    """
    xml = env.sim.model.get_xml()  # model xml file
    # save as file called "pb_mujoco.xml"
    with open("pb_mujoco.xml", "w") as f:
        f.write(xml)
    state = np.array(env.sim.get_state().flatten())  # simulator state
    return dict(model=xml, states=state)


@dataclass
class Config:
    """Script to configure demonstration parameters."""

    demo_path: str
    """Path to your demonstration path that contains the demo.hdf5 file."""

    use_actions: bool = False
    """Whether to use actions."""

    transform_objs: bool = False
    """Whether to transform objects."""

    offscreen_render: bool = False
    """Whether to use offscreen rendering."""

    convert_to_abs_actions: bool = False
    """Whether to convert to absolute actions."""


def main(cfg: Config):
    f = h5py.File(cfg.demo_path, "r")
    actions = np.array(f["data/{}/actions".format("demo_0")][()])
    states = np.array(f["data/{}/states".format("demo_0")][()])
    print(actions.shape)
    print(states.shape)
    print(actions[-2])
    print(actions[-1])
    # exit()
    if cfg.convert_to_abs_actions:
        # need to deal w/ fact that env is being created in this line
        converter = RobomimicAbsoluteActionConverter(cfg.demo_path)
        abs_actions = converter.convert_idx(0)
        actions = abs_actions

        # actions = actions[50:]
        # run every 5th action
        actions = actions[6::6]
        converter.close()

    # actions = actions[2::2]

    env_info = json.loads(f["data"].attrs["env_args"])["env_kwargs"]
    env_info["env_name"] = json.loads(f["data"].attrs["env_args"])["env_name"]

    if cfg.offscreen_render:
        env_info["has_renderer"] = False
        env_info["has_offscreen_renderer"] = True
        env_info["ignore_done"] = True
        env_info["use_camera_obs"] = True
    else:
        env_info["has_renderer"] = True
        env_info["has_offscreen_renderer"] = False
        env_info["ignore_done"] = True
        env_info["use_camera_obs"] = False

    env_info["controller_configs"]["control_delta"] = False
    # if aaditya_data:
    #     print("no modifications to env_info")
    # elif "1_demo.hdf5" in cfg.demo_path or "1_demo_180.hdf5" in cfg.demo_path or "image.hdf5" in cfg.demo_path:
    #     env_info["controller_configs"]["control_delta"] = True
    # else:
    #     env_info["controller_configs"]["control_delta"] = False
    #     env_info["controller_configs"]["policy_freq"] = 5
    #     env_info["control_freq"] = 5

    if cfg.convert_to_abs_actions:
        env_info["controller_configs"]["control_delta"] = False
        # env_info["controller_configs"]["policy_freq"] = 5
        # env_info["control_freq"] = 5

    print(env_info)

    env_info["camera_names"] = ["agentview", "robot0_eye_in_hand"]

    env_info["camera_heights"] = 256
    env_info["camera_widths"] = 256
    env = robosuite.make(
        **env_info,
    )

    # list of all demonstrations episodes
    demos = list(f["data"].keys())

    result_dict = defaultdict(bool)

    for ep in demos:
        # ep = "demo_1"
        print("Playing back episode: {}".format(ep))

        # env.viewer.set_camera(0)
        np.set_printoptions(precision=3, suppress=True)

        # load the flattened mujoco states
        states = f["data/{}/states".format(ep)][()]
        if cfg.transform_objs:
            objs_transf_name_seq = f[
                "data/{}/timestep_0/objs_transf_name_seq".format(ep)
            ]
            objs_transf_params_seq = f[
                "data/{}/timestep_0/objs_transf_params_seq".format(ep)
            ]
            objs_transf_type_seq = f[
                "data/{}/timestep_0/objs_transf_type_seq".format(ep)
            ]

            cube_transf_type_seq = [
                transf_type.decode("utf-8")
                for transf_type in objs_transf_type_seq["cube"]
            ]
            cube_transf_params_seq = {
                k: objs_transf_params_seq["cube"][k][()]
                for k in objs_transf_params_seq["cube"].keys()
            }
            [
                transf_name.decode("utf-8")
                for transf_name in objs_transf_name_seq["cube"]
            ]
            # convert to dict
            # ideally, objs_transfs only provides the se3 transform to apply ... and meshes are given elsewhere
            # note: mesh will be 0 centered and won't be in same coordinate frames as the nerf (as nerf is I by default)
            cube_mesh_path = "/juno/u/thankyou/autom/demo-aug/models/assets/objects/meshes/cube_mesh.stl"
            cube_mesh = trimesh.load(cube_mesh_path)

        images = []
        if cfg.use_actions:
            randomize_init_state_robot_pose = False
            if randomize_init_state_robot_pose:
                states[0][1:6] = (
                    states[0][1:6] + np.random.normal(0, 1, states[0][1:6].shape) * 0.1
                )

            if cfg.transform_objs:
                obj_pos = states[0][10:13]
                obj_quat_wxyz = states[0][13:17]
                obj_quat_xyzw = np.array(
                    [
                        obj_quat_wxyz[1],
                        obj_quat_wxyz[2],
                        obj_quat_wxyz[3],
                        obj_quat_wxyz[0],
                    ]
                )

                # apply obj_pos obj_quat_xyzw to cube_mesh vertices which is at p=0, R=I
                # convert quat to rotation matrix
                obj_rot = R.from_quat(obj_quat_xyzw).as_matrix()

                # convert to homogeneous matrix
                X_se3_mesh = np.eye(4)
                X_se3_mesh[:3, :3] = obj_rot
                X_se3_mesh[:3, 3] = obj_pos

                # apply transform to mesh
                cube_vertices_homog = np.hstack(
                    (
                        np.array(cube_mesh.vertices),
                        np.ones((cube_mesh.vertices.shape[0], 1)),
                    )
                )
                cube_vertices_homog = multiply_with_X_transf(
                    torch.tensor(X_se3_mesh, dtype=torch.float32),
                    torch.tensor(cube_vertices_homog, dtype=torch.float32),
                )  # maybe this can handle both numpy and torch?

                # update mesh
                cube_mesh.vertices = cube_vertices_homog[:, :3].numpy()

                # apply other transforms
                for i, transf_type in enumerate(cube_transf_type_seq):
                    # get the params
                    if transf_type == "SCALE":
                        X_scale_origin_key = f"{i}:X_scale_origin"
                        X_scale_origin = torch.tensor(
                            cube_transf_params_seq[X_scale_origin_key],
                            dtype=torch.float32,
                        )
                        X_scale_key = f"{i}:X_scale"
                        X_scale = torch.tensor(
                            cube_transf_params_seq[X_scale_key], dtype=torch.float32
                        )
                        cube_vertices_homog = multiply_with_X_transf(
                            torch.linalg.inv(X_scale_origin), cube_vertices_homog
                        )
                        cube_vertices_homog = multiply_with_X_transf(
                            X_scale, cube_vertices_homog
                        )
                        cube_vertices_homog = multiply_with_X_transf(
                            X_scale_origin, cube_vertices_homog
                        )
                    elif transf_type == "SE3":
                        X_se3_key = f"{i}:X_SE3"
                        X_se3 = torch.tensor(
                            cube_transf_params_seq[X_se3_key], dtype=torch.float32
                        )
                        cube_vertices_homog = multiply_with_X_transf(
                            X_se3, cube_vertices_homog
                        )

                    # update mesh
                    cube_mesh.vertices = cube_vertices_homog[:, :3].numpy()

                # convert back to numpy
                cube_vertices = cube_vertices_homog[:, :3].numpy()
                # update mesh
                cube_mesh.vertices = cube_vertices

                # save mesh
                cube_mesh.export("cube_mesh_transformed.stl")
            # need to generate a convex decomposition of the following: perhaps
            # directly call coacd here though then need to save things ...

            # set mujoco from xml
            model_xml = f["data/{}".format(ep)].attrs["model_file"]
            xml = postprocess_model_xml(model_xml)

            # can we directly update the xml to use the new mesh(es)?
            # with open('model.xml', 'w') as f:
            #     f.write(xml)

            # # load model.xml into env
            # use_updated = True
            if cfg.transform_objs:
                with open("updated_model.xml", "r") as newf:
                    xml = newf.read()
                states[0][10:13] = np.array([0.0, 0.0, 0.0])
                states[0][13:17] = np.array([1.0, 0.0, 0.0, 0.0])

            env.reset_from_xml_string(xml)
            env.sim.reset()
            print(f"states[0]: {states[0]}")
            # load the initial state
            obs = env.sim.set_state_from_flattened(states[0])

            env.sim.forward()

            env.robots[0].gripper.current_action = np.zeros(1)

            # take every 5th action
            # actions = actions[::5]

            # add a couple of dummy actions
            # zeros_array = np.zeros((13, actions.shape[1]))
            # num_copies = 30
            # dummy_actions = actions[0:1, :]
            # copied_actions = np.tile(dummy_actions, (num_copies, 1))
            # actions = np.vstack((copied_actions, actions))

            print(f"actions: {actions[:, -1]}")
            # drake_obs = f["data/{}/obs".format(ep)]
            num_positives = np.sum(actions[:, -1] == 1)
            num_negatives = np.sum(actions[:, -1] == -1)
            num_actions = actions.shape[0]

            print(
                f"num_actions: {num_actions}, num_positives: {num_positives}, num_negatives: {num_negatives}"
            )
            if not cfg.offscreen_render:
                env.render()

            first_z = None
            manual_compute_reward = False
            for j, action in enumerate(actions):
                obs, reward, done, info = env.step(action)
                # img = Image.fromarray(obs["agentview_image"][::-1])
                # img2 = Image.fromarray(obs["robot0_eye_in_hand_image"][::-1])
                # # save to png
                # img.save(f"{pathlib.Path(cfg.demo_path).expanduser().parent}/{ep}-playback-actions-{j}.png")
                # print(f"saved to {pathlib.Path(cfg.demo_path).expanduser().parent}/{ep}-playback-actions-{j}.png")

                if j > 0 and manual_compute_reward:
                    cur_z = env.sim.get_state().flatten()[10:17][2]
                    if cur_z >= first_z + 0.04:
                        reward = 1
                        print("success")
                        result_dict[ep] = True

                # break
                if reward == 1:
                    print("success")
                    result_dict[ep] = True
                if not cfg.offscreen_render:
                    env.render()
                else:
                    # print(f"obs['agentview_image'].max(): {obs['agentview_image'].max()}")
                    img = Image.fromarray(obs["agentview_image"][::-1])
                    img2 = Image.fromarray(obs["robot0_eye_in_hand_image"][::-1])
                    # # # save to png
                    # img.save(f"{pathlib.Path(cfg.demo_path).expanduser().parent}/{ep}-playback-actions-{j}.png")
                    # print(f"saved to {pathlib.Path(cfg.demo_path).expanduser().parent}/{ep}-playback-actions-{j}.png")
                    # import ipdb; ipdb.set_trace()
                    # concat images
                    img = np.concatenate((img, img2), axis=1)
                    images.append(img)

                np.array(
                    env.sim.data.site_xpos[
                        env.sim.model.site_name2id(env.robots[0].controller.eef_name)
                    ]
                )  # gripper0_grip_site

                state_playback = env.sim.get_state().flatten()
                if j == 0:
                    first_z = state_playback[10:17][2]

                print(f"{j}: state_playback: {state_playback[10:17]}")
                if j < num_actions - 1:
                    # ensure that the actions deterministically lead to the same recorded states
                    state_playback = env.sim.get_state().flatten()

                    # if not np.all(np.equal(states[j + 1], state_playback)):
                    #     err = np.linalg.norm(states[j + 1] - state_playback)
                    #     print(f"[warning] playback diverged by {err:.3f} for ep {ep} at step {j}")

            # save images as mp4
            if cfg.offscreen_render and len(images) > 0:
                mp4_path = f"{pathlib.Path(cfg.demo_path).expanduser().parent}/{ep}-playback-actions.mp4"
                print(f"Saving mp4 to {mp4_path}")
                # take every 10th image
                # images = images[::5]
                images = images
                # save to png
                for i, img in enumerate(images):
                    img = Image.fromarray(img)
                    img.save(
                        f"{pathlib.Path(cfg.demo_path).expanduser().parent}/{ep}-playback-actions-{i}.png"
                    )

                imageio.mimsave(mp4_path, images, fps=10)
                gif_path = f"{pathlib.Path(cfg.demo_path).expanduser().parent}/{ep}-playback-actions.gif"
                imageio.mimsave(gif_path, images, duration=len(images) / 20)

            save_constraint_pose = False
            constraint_pose_idx = 54
            if save_constraint_pose:
                img_to_save = images[constraint_pose_idx]
                img_to_save = Image.fromarray(img_to_save)
                img_to_save.save(
                    f"{pathlib.Path(cfg.demo_path).expanduser().parent}/{ep}-playback-actions-constraint.png"
                )

            save_original_start_pose = False
            original_start_pose_idx = 0
            if save_original_start_pose:
                img_to_save = images[original_start_pose_idx]
                img_to_save = Image.fromarray(img_to_save)
                if randomize_init_state_robot_pose:
                    img_to_save.save(
                        f"{pathlib.Path(cfg.demo_path).expanduser().parent}/{ep}-playback-actions-rand-start2.png"
                    )
                else:
                    img_to_save.save(
                        f"{pathlib.Path(cfg.demo_path).expanduser().parent}/{ep}-playback-actions-original-start.png"
                    )

            if not result_dict[ep]:
                # TODO(klin): need to comment this out or update the reward computation
                import ipdb

                ipdb.set_trace()
        else:
            # force the sequence of internal mujoco states one by one
            if "1_demo.hdf5" in cfg.demo_path or "1_demo_180.hdf5" in cfg.demo_path:
                for i, state in enumerate(states):
                    print(f"i: {i}")
                    env.sim.set_state_from_flattened(state)
                    env.sim.forward()

                    np.array(
                        env.sim.data.site_xpos[
                            env.sim.model.site_name2id(
                                env.robots[0].controller.eef_name
                            )
                        ]
                    )  # gripper0_grip_site
                    ee_ori_mat = np.array(
                        env.sim.data.site_xmat[
                            env.sim.model.site_name2id(
                                env.robots[0].controller.eef_name
                            )
                        ].reshape([3, 3])
                    )
                    T.mat2quat(ee_ori_mat)
                    env.sim.data.get_body_xpos("robot0_right_hand")
                    env.sim.data.get_body_xquat("robot0_right_hand")
                    np.array(
                        env.sim.data.site_xpos[
                            env.sim.model.site_name2id(
                                env.robots[0].controller.eef_name
                            )
                        ]
                    )
                    env.render()
            else:
                state = states[0]
                actions = np.array(f["data/{}/actions".format(ep)][()])
                obs = f["data/{}/states".format(ep)]
                for idx, _ in enumerate(actions):
                    cur_robot_joint_pos = obs["robot0_joint_qpos"][idx]
                    state[1 : len(cur_robot_joint_pos) + 1] = cur_robot_joint_pos
                    cur_gripper_pos = obs["robot0_gripper_qpos"][idx]
                    state[
                        len(cur_robot_joint_pos) + 1 : len(cur_robot_joint_pos)
                        + 1
                        + len(cur_gripper_pos)
                    ] = cur_gripper_pos
                    env.sim.set_state_from_flattened(state)
                    env.sim.forward()
                    if not cfg.offscreen_render:
                        env.render()

                    # gett ee_pos:
                    np.array(
                        env.sim.data.site_xpos[
                            env.sim.model.site_name2id(
                                env.robots[0].controller.eef_name
                            )
                        ]
                    )  # gripper0_grip_site

            if cfg.offscreen_render:
                mp4_path = f"{pathlib.Path(cfg.demo_path).expanduser().parent}/{ep}-playback-states.mp4"
                print(f"Saving mp4 to {mp4_path}")
                imageio.mimsave(mp4_path, images, fps=10)
                gif_path = f"{pathlib.Path(cfg.demo_path).expanduser().parent}/{ep}-playback-states.gif"
                imageio.mimsave(gif_path, images, duration=len(images) / 20)

    f.close()


if __name__ == "__main__":
    tyro.cli(main)  # replace "your_function" with the actual function to call
