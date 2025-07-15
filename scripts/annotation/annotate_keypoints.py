"""
Script for finding desired keypoint position relative to a robot link link_name.


Test script for relative pose logic (TODO):
Use one link as src frame and another link as target frame, then get relative position of sphere in target frame.
Then, set the sphere position in the target frame and check if the world frame sphere positions match.

Example:
python scripts/annotation/annotate_keypoints.py --cfg.scene-reset-type from_xml --cfg.link-name leftgripper --cfg.other-link-names SquareNut_main
python scripts/annotation/annotate_keypoints.py --cfg.scene-reset-type robomimic --cfg.demo-path datasets/source/square-demo-152.hdf5 --cfg.demo-name demo_152 --cfg.demo-timestep 115 --cfg.link-name SquareNut_main
python scripts/annotation/annotate_keypoints.py --cfg.scene-reset-type robomimic --cfg.demo-path datasets/source/square-demo-152.hdf5 --cfg.demo-name demo_152 --cfg.demo-timestep 65 --cfg.link-name gripper0_right_leftfinger --cfg.other-link-names SquareNut_main --cfg.fix-sim-state --cfg.set-sphere-position
"""

import pathlib
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple

import h5py
import mujoco
import mujoco.viewer
import numpy as np
import tyro

#########################
### robosuite methods ###
#########################


def array_to_string(array):
    return " ".join(["{}".format(x) for x in array])


def convert_to_string(inp):
    if type(inp) in {list, tuple, np.ndarray}:
        return array_to_string(inp)
    elif type(inp) in {int, float, bool}:
        return str(inp).lower()
    elif type(inp) in {str, np.str_}:
        return inp
    else:
        raise ValueError("Unsupported type received: got {}".format(type(inp)))


def new_element(tag, name, **kwargs):
    # Name will be set if it's not None
    if name is not None:
        kwargs["name"] = name
    # Loop through all attributes and pop any that are None, otherwise convert them to strings
    for k, v in kwargs.copy().items():
        if v is None:
            kwargs.pop(k)
        else:
            kwargs[k] = convert_to_string(v)
    element = ET.Element(tag, attrib=kwargs)
    return element


def new_geom(name, type, size, pos=(0, 0, 0), group=0, **kwargs):
    kwargs["type"] = type
    kwargs["size"] = size
    kwargs["pos"] = pos
    kwargs["group"] = group if group is not None else None
    return new_element(tag="geom", name=name, **kwargs)


def new_body(name, pos=(0, 0, 0), **kwargs):
    kwargs["pos"] = pos
    return new_element(tag="body", name=name, **kwargs)


#########################
### robosuite methods ###
#########################


def get_relative_position(
    model: mujoco.MjModel, data: mujoco.MjData, sphere_name: str, link_name: str
) -> np.ndarray:
    """Get the position of the sphere relative to a specific robot link."""
    mocap_id = model.body(sphere_name).mocapid[0]

    link_body_id = model.body(link_name).id

    sphere_pos = data.mocap_pos[mocap_id]  # World position of the sphere
    link_pos = data.xpos[link_body_id]  # World position of the robot link
    link_ori = data.xquat[link_body_id]  # World orientation of the robot link
    link_pose = np.eye(4)
    link_mat = np.ones((9, 1), dtype=np.float64)
    link_mat = np.ascontiguousarray(link_mat)
    mujoco.mju_quat2Mat(link_mat, link_ori)
    link_pose[:3, :3] = link_mat.reshape(3, 3)
    link_pose[:3, 3] = link_pos
    # Compute the relative position of the sphere with respect to the robot link
    relative_pos = np.linalg.inv(link_pose) @ np.append(sphere_pos, 1)
    relative_pos = relative_pos[:3]
    return relative_pos


def set_sphere_position(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    sphere_name: str,
    link_name: str,
    position: np.ndarray,
) -> None:
    """Set the position of the sphere in the frame of a specific robot link."""
    mocap_id = model.body(sphere_name).mocapid[0]

    link_body_id = model.body(link_name).id
    link_pos = data.xpos[link_body_id]  # World position of the robot link
    link_ori = data.xquat[link_body_id]  # World orientation of the robot link
    link_pose = np.eye(4)
    link_mat = np.ones((9, 1), dtype=np.float64)
    link_mat = np.ascontiguousarray(link_mat)
    mujoco.mju_quat2Mat(link_mat, link_ori)
    link_pose[:3, :3] = link_mat.reshape(3, 3)
    link_pose[:3, 3] = link_pos

    # Compute the absolute position of the sphere based on the link position and the given relative position
    absolute_pos = (link_pose @ np.append(position, 1))[:3]
    # Set the position of the sphere
    data.mocap_pos[mocap_id] = absolute_pos
    mujoco.mj_kinematics(model, data)


def set_target_pose(sim, target_pos=None, target_mat=None, mocap_name: str = "target"):
    mocap_id = sim.model.body(mocap_name).mocapid[0]
    if target_pos is not None:
        sim.data.mocap_pos[mocap_id] = target_pos
    if target_mat is not None:
        # convert mat to quat
        target_quat = np.empty(4)
        mujoco.mju_mat2Quat(target_quat, target_mat.reshape(9, 1))
        sim.data.mocap_quat[mocap_id] = target_quat


def add_mocap_sphere(xml_path: str, new_xml_path: str) -> None:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # find worldbody
    worldbody = root.find("worldbody")

    # Add the mocap sphere to the XML model
    mocap_body_1 = new_body(name="sphere", pos="0 0 -1", mocap=True)
    mocap_body_1_geom = new_geom(
        name="sphere",
        type="sphere",
        size="0.02 0.02 0.02",
        # size="0.1 0.1 0.1",
        rgba="0.898 0.420 0.435 0.5",
        conaffinity="0",
        contype="0",
        group="2",
    )
    mocap_body_1.append(mocap_body_1_geom)
    worldbody.append(mocap_body_1)

    # Create an ElementTree object
    tree = ET.ElementTree(root)

    # Save the modified XML model to new file
    tree.write(new_xml_path, encoding="utf-8", xml_declaration=True)
    print(f"Added mocap sphere to {new_xml_path}")


@dataclass
class Config:
    scene_reset_type: Literal["from_xml", "robomimic"] = "from_xml"
    demo_path: Optional[str] = None
    demo_name: Optional[str] = None
    demo_timestep: int = 0
    xml_path: pathlib.Path = pathlib.Path(
        "/home/thankyou/autom/robosuite-dev/robosuite/models/assets/grippers/panda_gripper.xml"
    )
    temp_xml_path: Optional[pathlib.Path] = None
    sphere_name: str = "sphere"
    link_name: str = "rightfinger"
    other_link_names: List[str] = field(default_factory=lambda: [])
    sphere_in_link_pos: Tuple[int, int, int] = field(
        default_factory=lambda: (0, 0, 0.0447)
    )
    # sphere_in_link_pos: Tuple[float, float, float] = field(default_factory=lambda: ([0, 0.0, -0.01]))
    show_left_ui: bool = False
    show_right_ui: bool = True
    print_interval: int = 100
    set_sphere_position: bool = False
    fix_sim_state: bool = False

    def __post_init__(self):
        print(f"Using {self.scene_reset_type} to reset the scene.")
        if self.scene_reset_type == "robomimic":
            assert self.demo_path is not None, "Please provide a path to the demo file."
            assert (
                self.demo_name is not None
            ), "Please provide the name of the demo to load."
        elif self.scene_reset_type == "from_xml":
            assert (
                self.xml_path is not None and self.xml_path.expanduser().exists()
            ), f"XML file not found at {self.xml_path.expanduser()}."
            if self.temp_xml_path is None:
                self.temp_xml_path = (
                    self.xml_path.parent / f"{self.xml_path.stem}_temp.xml"
                )

        if not self.set_sphere_position:
            print(
                "Sphere position will not be set --- please choose some value for sphere_in_link_pos"
                "by dragging around the sphere in the viewer."
            )


def main(cfg: Config) -> None:
    """
    Workflow for finding desired keypoint position relative to a robot link link_name:

    1) Drag the sphere to the desired position in the viewer and record the printed relative position.
    2) Set sphere_in_link_pos to the recorded position and run the script with set_sphere_position=True.
        Check if the sphere is placed at the desired position.
    """
    assert mujoco.__version__ >= "3.1.0", "Please upgrade to MuJoCo 3.1.0 or later."
    np.set_printoptions(precision=3, suppress=True)
    if cfg.scene_reset_type == "robomimic":
        import robomimic.utils.env_utils as EnvUtils
        import robomimic.utils.file_utils as FileUtils

        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=cfg.demo_path)
        env = EnvUtils.create_env_from_metadata(env_meta=env_meta)
        xml = env.env.sim.model.get_xml()
        f = h5py.File(cfg.demo_path, "r")
        # xml = f["data"][cfg.demo_name].attrs["model_file"]
        state = f["data"][cfg.demo_name]["states"][cfg.demo_timestep]
        f.close()
        # save to a temp file near the demo file
        temp_xml_path = pathlib.Path(cfg.demo_path).parent / f"{cfg.demo_name}_temp.xml"
        with open(temp_xml_path, "w") as f:
            f.write(xml)
        print(f"Saved temp xml file to {temp_xml_path}")
        cfg.temp_xml_path = temp_xml_path
        add_mocap_sphere(temp_xml_path.as_posix(), temp_xml_path.as_posix())
    else:
        add_mocap_sphere(cfg.xml_path.as_posix(), cfg.temp_xml_path.as_posix())

    model = mujoco.MjModel.from_xml_path(cfg.temp_xml_path.as_posix())
    cfg.temp_xml_path.unlink()

    data = mujoco.MjData(model)
    if cfg.scene_reset_type == "robomimic":
        # take the first state from the demo file using h5py
        data.qpos[:] = state[1 : 1 + model.nq]
        data.qvel[:] = state[1 + model.nq :]

    if cfg.fix_sim_state:
        fixed_qpos = data.qpos.copy()
        fixed_qvel = data.qvel.copy()

    np.set_printoptions(precision=4, suppress=True)
    i = 0
    # Viewer to visualize and interact with the simulation
    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=True
    ) as viewer:
        while viewer.is_running():
            step_start = time.time()

            # Perform one simulation step
            mujoco.mj_step(model, data)

            if cfg.set_sphere_position:
                set_sphere_position(
                    model, data, cfg.sphere_name, cfg.link_name, cfg.sphere_in_link_pos
                )

            if i % cfg.print_interval == 0:
                # Get the relative position of the sphere w.r.t the specified robot link
                relative_position = get_relative_position(
                    model, data, cfg.sphere_name, cfg.link_name
                )
                print(
                    f"Relative Position of {cfg.sphere_name} w.r.t {cfg.link_name}: {relative_position}"
                )
                for link_name in cfg.other_link_names:
                    relative_position = get_relative_position(
                        model, data, cfg.sphere_name, link_name
                    )
                    print(
                        f"Relative Position of {cfg.sphere_name} w.r.t {link_name}: {relative_position}"
                    )

            # Ensure real-time stepping
            time_to_wait = model.opt.timestep - (time.time() - step_start)
            if time_to_wait > 0:
                time.sleep(time_to_wait)

            viewer.sync()

            if cfg.fix_sim_state:
                data.qpos[:] = fixed_qpos
                data.qvel[:] = fixed_qvel

            i += 1


if __name__ == "__main__":
    tyro.cli(main)
