"""
A convenience script to playback augmented demonstrations from
a set of demonstrations stored in a hdf5 file. Purpose is to
debug the augmentation process specifically for the more rigid
augmentations (e.g. object translation, rotation, scaling, warping)
that don't require NeRFs at multiple timesteps for representing an object.

Example:
    $ python scripts/playback_aug_demos_from_hdf5_v2.py   --cfg.offscreen-render --cfg.use-actions \
    --cfg.demo-path /juno/u/thankyou/autom/demo-aug/debug/square-start2wp2goal-wp-z-offset0.08-wp-sample-radius-0.hdf5 \
    --cfg.transform-objs

    (nerfstudio) thankyou@bohg-ws-14:/juno/u/thankyou/autom/demo-aug$ python scripts/playback_aug_demos_from_hdf5_v2.py \
        --cfg.offscreen-render --cfg.use-actions \
        --cfg.demo-path ../diffusion_policy/data/robomimic/datasets/augmented/demo_01trials/2023-08-29/\
        jqums14p/square-pick-insert-start2wp2goal-wp-z-offset0.08-wp-sample-radius-0.hdf5 \
        --cfg.transform-objs


To compare recorded obs with actual obs:
- set the flags --cfg.compare-recorded-with-actual-obs and --cfg.set-joint-qpos

TODO(klin): add the demo number to the image at the bottom.
"""

import json
import pathlib
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import List

import h5py
import imageio
import numpy as np
import robosuite
import tyro
from PIL import Image, ImageDraw, ImageFont
from robosuite.utils.mjcf_utils import postprocess_model_xml
from scipy.spatial.transform import Rotation as R

from demo_aug.utils.mujoco_utils import update_model_xml


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

    camera_heights: int = 256
    """Height of rendered images."""

    camera_widths: int = 256
    """Width of rendered images."""

    camera_names: List[str] = field(
        default_factory=lambda: ["agentview", "robot0_eye_in_hand"]
    )
    """Camera names to render from."""

    set_joint_qpos: bool = False
    """Whether to set the joint qpos instead of using the controller to execute actions."""

    fix_body_to_world: bool = False
    """Whether to fix the body to the world (because, for example, initial position isn't static)."""

    remove_table: bool = False
    """Whether to remove the table for visualization or debugging purposes."""

    compare_recorded_with_actual_obs: bool = False
    """Whether to compare the recorded obs with the actual obs obtained by setting states and querying obs."""

    manually_check_success: bool = False
    """Whether to manually check success (because objects are now transformed)."""

    add_text: bool = False
    """Whether to add text to the rendered images (for debugging)."""

    save_frames_individually: bool = False
    """Whether to save the frames individually (rather than as a video)."""

    recenter_vertices: bool = False
    """Whether to recenter vertices (doing so might make fix_body_to_world not work properly)"""

    n_action_repeats: int = 1
    """Number of times to repeat each action during demo playback."""

    def __post_init__(self):
        if self.compare_recorded_with_actual_obs:
            if not self.set_joint_qpos:
                raise ValueError(
                    "Please set set_joint_qpos to True to better compare recorded obs with actual obs!"
                )

        if self.fix_body_to_world:
            assert (
                not self.recenter_vertices
            ), "recenter vertices should be false if fix_body_to_world"


def find_mesh_path(obj_name: str, model_xml: str):
    import xml.etree.ElementTree as ET

    # Parse the model XML using ElementTree
    root = ET.fromstring(model_xml)

    # Assuming each mesh element has a 'name' attribute
    for mesh in root.findall(".//mesh"):
        mesh_name = mesh.get("name")

        # If the mesh name matches the target object name, retrieve the path
        if mesh_name == obj_name:
            # Assuming the mesh path is specified as a 'path' attribute
            mesh_path = mesh.get("path")
            # check if there is a mesh path
            return pathlib.Path(mesh_path)  # Convert to pathlib.Path object

    return None  # Return None if the mesh path is not found


def get_mesh_paths(objs: List[str], model_xml: str) -> List[pathlib.Path]:
    obj_mesh_paths = {}
    # save the model_xml to 'temp.xml'
    with open("temp.xml", "w") as f:
        f.write(model_xml)

    for obj in objs:
        # check if there is a mesh with name obj
        if not find_mesh_path(obj, model_xml):
            # create a mesh with name obj
            obj_mesh_paths[obj] = model_xml.replace(
                "model.xml", "assets/objects/meshes/{}.stl".format(obj)
            )
    return obj_mesh_paths


def main(cfg: Config):
    f = h5py.File(cfg.demo_path, "r")

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

    use_delta_actions = False
    env_info["controller_configs"]["control_delta"] = use_delta_actions
    env_info["controller_configs"]["control_freq"] = 5
    env_info["control_freq"] = 5
    env_info["camera_names"] = cfg.camera_names
    env_info["camera_heights"] = cfg.camera_heights
    env_info["camera_widths"] = cfg.camera_widths
    print(f"env_info: {env_info}")

    env = robosuite.make(
        **env_info,
    )

    demos = list(f["data"].keys())
    total_successes = 0

    print(f"demos: {demos}")
    for idx, ep in enumerate(demos):
        # reset so that gripper dynamics etc are reset correctly; env.sim.reset() not enough; unclear which is the fix
        obs = env.reset()
        env.robots[0].gripper.current_action = np.zeros(1)

        np.set_printoptions(precision=3, suppress=True)
        # load the flattened mujoco states
        states = f["data/{}/states".format(ep)][()]
        rec_obs = f["data/{}/obs".format(ep)]
        actions = np.array(f["data/{}/actions".format(ep)][()])

        # manually increase the actions by copying the last action 5 times in a for loop

        print(f"actions: {actions[:, -1]}")

        if use_delta_actions:
            actions_delta = np.array(
                f[f"data/{ep}/action_alt_rep/action_delta_world"][()]
            )
            actions = actions_delta

        orig_eef_quat = rec_obs["robot0_eef_quat"][()].copy()  # is this xyzw?
        rec_obs["robot0_eef_pos"][()].copy()

        num_positives = np.sum(actions[:, -1] == 1)
        num_negatives = np.sum(actions[:, -1] == -1)
        num_actions = actions.shape[0]

        print(
            f"num_actions: {num_actions}, num_positives: {num_positives}, num_negatives: {num_negatives}"
        )

        if cfg.transform_objs:
            # seems difficult to initialize sim with object already in hand
            # would need to initialize sim, freeze the object somehow, the close the gripper (+ensure obj doesn't move)
            objs_transf_name_seq = f[f"data/{ep}/timestep_0/objs_transf_name_seq"]
            objs_transf_params_seq = f[f"data/{ep}/timestep_0/objs_transf_params_seq"]
            objs_transf_type_seq = f[f"data/{ep}/timestep_0/objs_transf_type_seq"]

            obj_to_transf_type_seq = {
                obj: [
                    obj_transf_type.decode("utf-8")
                    for obj_transf_type in obj_transf_type_seq
                ]
                for obj, obj_transf_type_seq in objs_transf_type_seq.items()
            }
            obj_to_transf_params_seq = {
                obj: {
                    obj_name: transf_params[()]
                    for obj_name, transf_params in transf_params_seq.items()
                }
                for obj, transf_params_seq in objs_transf_params_seq.items()
            }
            obj_to_transf_name_seq = {
                obj: [transf_name.decode("utf-8") for transf_name in transf_name_seq]
                for obj, transf_name_seq in objs_transf_name_seq.items()
            }

            # remove shear transforms from obj_to_transf_params_seq
            for obj_name, transf_params_seq in obj_to_transf_params_seq.items():
                print(obj_name)
                keys_to_remove = []
                for transf_name, transf_params in transf_params_seq.items():
                    print(f"{transf_name}: {transf_params}")
                    if "scale" in transf_name and "orig" not in transf_name:
                        print(transf_name, transf_params)
                    if "shear" in transf_name:
                        # remove transf_name from transf_params_seq
                        keys_to_remove.append(transf_name)

                    print(f"transf_name: {transf_name}")
                    print(f"transf_params: {transf_params}")

                # for key in keys_to_remove:
                #     del transf_params_seq[key]

            # get the model_xml
            list(obj_to_transf_name_seq.keys())
            model_xml = f["data/{}".format(ep)].attrs["model_file"]

        images = []
        imgs_1, imgs_2 = [], []
        success: bool = False
        if cfg.use_actions:
            model_xml = f["data/{}".format(ep)].attrs["model_file"]
            xml = postprocess_model_xml(model_xml)
            # xml = env.edit_model_xml(model_xml)

            model_xml = xml
            update_xml_with_custom_meshes = cfg.transform_objs
            if update_xml_with_custom_meshes:
                if "lift" in cfg.demo_path:
                    obj_to_xml_body_name = {
                        "red_cube": "cube_main",
                    }
                elif "square" in cfg.demo_path:
                    obj_to_xml_body_name = {
                        "square_nut": "SquareNut_main",
                        "square_peg": "peg1",
                    }
                elif "can" in cfg.demo_path:
                    obj_to_xml_body_name = {
                        "can": "Can_main",
                    }
                else:
                    raise ValueError(
                        "Please specify the obj_to_xml_body_name for your demo path!"
                    )
                # import ipdb; ipdb.set_trace()
                output = update_model_xml(
                    xml,
                    obj_to_xml_body_name,
                    obj_to_transf_type_seq,
                    obj_to_transf_params_seq,
                    remove_body_free_joint=cfg.fix_body_to_world,  # if remove, the object is fixed to the world
                    apply_all_transforms=True,
                    set_obj_collision_free=False,  # for pure rendering purposes, disable collisions
                    recenter_vertices=cfg.recenter_vertices,
                )
                if cfg.recenter_vertices:
                    xml, new_obj_pos = output
                else:
                    xml = output
                    new_obj_pos = np.zeros(3)

                if cfg.remove_table:
                    # find the 'body' element with name 'table'
                    root = ET.fromstring(xml)
                    # get the xml tag for the body with name SquareNut_main
                    body = root.find(".//body[@name='table']")
                    # update pos to 0 0 0
                    body.set("pos", "0 0 0")

                model_xml = xml

            state = states[0].copy()

            use_pure_nerf_mesh = False  # maybe I'll need this later?
            dont_update_things_here = cfg.transform_objs
            if use_pure_nerf_mesh:
                obj_pose = np.eye(4)
                obj_pose = obj_pose @ obj_to_transf_params_seq["square_nut"]["2:X_SE3"]
                obj_pose = obj_pose @ obj_to_transf_params_seq["square_nut"]["1:X_SE3"]
                obj_pose = obj_pose @ obj_to_transf_params_seq["square_nut"]["0:X_SE3"]
                obj_quat_xyzw = R.from_matrix(obj_pose[:3, :3]).as_quat()
                np.array(
                    [
                        obj_quat_xyzw[3],
                        obj_quat_xyzw[0],
                        obj_quat_xyzw[1],
                        obj_quat_xyzw[2],
                    ]
                )
                obj_pose[:3, 3]

            # if False:
            if not dont_update_things_here:
                # looks like the gripper is originally closed!?
                # need to apply the task relevant object's transf
                # state[10:13] = obj_pos
                # state[13:17] = obj_quat_wxyz
                print("using saved state")
                print(state)
            else:
                state[10:13] = new_obj_pos
                state[13:17] = [1, 0, 0, 0]

            manual_reset_from_xml_string = True
            if manual_reset_from_xml_string:
                # set the peg pos/quat?
                env.reset_from_xml_string(model_xml)
                env.sim.reset()
            manual_reset_from_state = True
            if manual_reset_from_state:
                env.sim.data.qpos[:] = state[1:][: len(env.sim.data.qpos)]
                env.sim.forward()
                env._update_observables(force=True)
                obs = env._get_observations()

            if not cfg.offscreen_render:
                env.render()
                env.viewer.set_camera(
                    5
                )  # agentview (I think) env.viewer.set_camera(2) for eye in hand
            else:
                env._update_observables(force=True)
                obs = env._get_observations()

                img = Image.fromarray(obs["agentview_image"][::-1])
                img2 = Image.fromarray(obs["robot0_eye_in_hand_image"][::-1])
                # save img to png
                img.save(f"{pathlib.Path(cfg.demo_path).expanduser().parent}/img1.png")
                img = np.array(img)
                img2 = np.array(img2)
                if cfg.save_frames_individually:
                    imgs_1.append(img)
                    imgs_2.append(img2)
                # img = np.concatenate((img, img2), axis=1)
                img = np.array(img)
                images.append(img)

            if cfg.manually_check_success:
                if "Lift" in env_info["env_name"]:
                    start_z_pos = env.sim.data.body_xpos[env.cube_body_id][2]

            for i in range(num_actions):
                action = np.copy(actions[i])
                for _ in range(cfg.n_action_repeats):
                    if cfg.set_joint_qpos:
                        robot_qpos = rec_obs["robot0_joint_qpos"][i]
                        env.sim.data.qpos[:7] = robot_qpos
                        gripper_qpos = rec_obs["robot0_gripper_qpos"][i]
                        env.sim.data.qpos[7:9] = gripper_qpos
                        env.sim.forward()
                        env._update_observables(force=True)

                        obs = env._get_observations()
                    else:
                        obs, reward, done, info = env.step(action)
                        success = success or reward

                        for cam_name in env.camera_names:
                            # check if any image is all zeros
                            if np.all(obs[f"{cam_name}_image"] == 0):
                                rgb = env.sim.render(
                                    camera_name=cam_name,
                                    height=256,
                                    width=256,
                                )
                                obs[f"{cam_name}_image"] = rgb.copy()

                    if not cfg.offscreen_render:
                        env.render()
                    else:
                        img = Image.fromarray(obs["agentview_image"][::-1])
                        img2 = Image.fromarray(obs["robot0_eye_in_hand_image"][::-1])
                        if cfg.add_text:
                            draw = ImageDraw.Draw(img)

                            diff_str = f"diff: {np.array(action[:3]) - np.array(obs['robot0_eef_pos'])}"
                            # diff_str = f"diff: {np.array(orig_eef_pos[i + 1]) - np.array(obs['robot0_eef_pos'])}"
                            if cfg.compare_recorded_with_actual_obs:
                                if i + 1 < len(orig_eef_quat):
                                    diff_str = (
                                        "diff:"
                                        f" {np.array(rec_obs['robot0_eef_pos'][i]) - np.array(obs['robot0_eef_pos'])}"
                                    )
                                else:
                                    diff_str = ""

                            if i + 1 < len(orig_eef_quat):
                                diff_quat_str = f"diff quat: {np.array(orig_eef_quat[i + 1]) - np.array(obs['robot0_eef_quat'])}"
                            else:
                                diff_quat_str = ""
                            draw.text((0, 0), diff_str)
                            draw.text((0, 10), diff_quat_str)

                            # at the bottom of the image, store the demo number and the parent folder of the demo
                            demo_number = ep.split("_")[-1]
                            parent_folder = (
                                pathlib.Path(cfg.demo_path).expanduser().parent.name
                            )
                            draw.text((0, 20), f"demo: {demo_number}")
                            draw.text((0, 30), f"folder: {parent_folder}")

                            img = np.array(img)
                        if cfg.save_frames_individually:
                            imgs_1.append(img)
                            imgs_2.append(img2)

                        img = np.array(img)
                        images.append(img)

            if cfg.manually_check_success:
                if "Lift" in env_info["env_name"]:
                    success = False
                    # TODO(klin): don't think this is a good way to check for failures
                    # need a displacement threshold not absolute threshold
                    print(
                        f"start_z_pos: {start_z_pos} end_z_pos: {env.sim.data.body_xpos[env.cube_body_id][2]}"
                    )
                    print(
                        f"Diff: {env.sim.data.body_xpos[env.cube_body_id][2] - start_z_pos}"
                    )
                    ORIGINAL_SUCCESS_DELTA = 0.02
                    # get scale factor in z direction
                    scale_factor = 1
                    for transf_name, transf_params in obj_to_transf_params_seq[
                        "red_cube"
                    ].items():
                        if "scale" in transf_name and "scale_origin" not in transf_name:
                            scale_factor = transf_params[2, 2]
                    SCALED_SUCCESS_DELTA = ORIGINAL_SUCCESS_DELTA * scale_factor
                    if (
                        env.sim.data.body_xpos[env.cube_body_id][2] - start_z_pos
                        > SCALED_SUCCESS_DELTA
                    ):
                        print("Success!")
                        success = True
                        # breakpoint()
                    else:
                        print("Not success!")
                        success = False

                total_successes += success
                print(f"total successes / total demos: {total_successes} / {idx}")

            if cfg.offscreen_render:
                parent_path = pathlib.Path(cfg.demo_path).expanduser().parent
                set_qpos = "-set-qpos" if cfg.set_joint_qpos else ""
                fix_body = "-fix-body-to-world" if cfg.fix_body_to_world else ""
                success_status = (
                    "-success" if cfg.manually_check_success or success else "-fail"
                )

                # Add frame numbers to images
                font = ImageFont.load_default()
                for i, img in enumerate(images):
                    pil_img = Image.fromarray(img)
                    draw = ImageDraw.Draw(pil_img)
                    draw.text(
                        (10, pil_img.height - 30),
                        f"Frame: {i}",
                        font=font,
                        fill=(255, 255, 255),
                    )
                    images[i] = np.array(pil_img)

                mp4_path = f"{parent_path}/{ep}-playback-actions-test-convex-shapes{set_qpos}{fix_body}{success_status}.mp4"
                print(f"Saving mp4 to {mp4_path}")
                imageio.mimsave(mp4_path, images, fps=20)

                gif_path = f"{parent_path}/{ep}-playback-actions-test-convex-shapes{set_qpos}{fix_body}{success_status}.gif"
                imageio.mimsave(gif_path, images, duration=len(images) / 20, loop=0)
                print(f"Saving gif to {gif_path}")

            if cfg.save_frames_individually:
                # create directory
                pathlib.Path(
                    f"{pathlib.Path(cfg.demo_path).expanduser().parent}/imgs/eps{ep}"
                ).mkdir(parents=True, exist_ok=True)
                # save imgs_1 imgs_2 as individual frames
                for i in range(len(imgs_1)):
                    img1 = imgs_1[i]
                    img2 = imgs_2[i]
                    # save images using imageio
                    imageio.imwrite(
                        f"{pathlib.Path(cfg.demo_path).expanduser().parent}/imgs/eps{ep}/img1-{i}.png",
                        img1,
                    )
                    imageio.imwrite(
                        f"{pathlib.Path(cfg.demo_path).expanduser().parent}/imgs/eps{ep}/img2-{i}.png",
                        img2,
                    )
    f.close()


if __name__ == "__main__":
    tyro.cli(main)  # replace "your_function" with the actual function to call
