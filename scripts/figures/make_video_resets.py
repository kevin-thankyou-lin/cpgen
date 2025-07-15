"""
Convenience script to make a video out of initial environment
configurations. This can be a useful debugging tool to understand
what different sampled environment configurations look like.
"""

import argparse
import pathlib
import xml.etree.ElementTree as ET
from typing import Literal, Tuple

import imageio
import numpy as np
import robosuite as suite
from robosuite.controllers import load_controller_config


def generate_plane_and_border_cylinders(
    pos_xyz: Tuple[float, float, float], size_xyz: Tuple[float, float, float]
):
    """
    Generate geom dictionaries for a transparent plane and surrounding cylinders at a given position and size.

    Parameters:
        pos_xyz (list or tuple): The (x, y) position of the plane's center.
        size_xy (list or tuple): The half-size (sx, sy) of the plane in the x and y directions.
        pz (float): The z-coordinate where the plane and cylinders are located. Default is 0.8.

    Returns:
        list: A list of dictionaries defining the plane and the cylinders.
    """
    px, py, pz = pos_xyz
    sx, sy, sz = size_xyz

    # Generate the plane dictionary
    plane_dict = {
        "type": "box",
        "pos": f"{px} {py} {pz}",
        "size": f"{sx} {sy} {sz}",  # Thickness of the plane in z-direction
        "rgba": "0 0.8 0.8 0.5",  # Semi-transparent gray color
        "group": "1",
        "contype": "0",
        "conaffinity": "0",
    }

    # Compute the four corners of the plane
    c1 = (px - sx, py - sy)
    c2 = (px + sx, py - sy)
    c3 = (px + sx, py + sy)
    c4 = (px - sx, py + sy)

    cylinders = []

    # Define the properties common to all cylinders
    cylinder_template = {
        "type": "cylinder",
        "size": "0.001",  # Radius of the cylinder
        "rgba": "0 0.8 0.8 1",  # Color (black)
        "group": "1",
        "contype": "0",
        "conaffinity": "0",
    }

    # List of corner pairs to define each cylinder (edge)
    corner_pairs = [(c1, c2), (c2, c3), (c3, c4), (c4, c1)]

    # Generate a cylinder for each edge
    for from_corner, to_corner in corner_pairs:
        cylinder = cylinder_template.copy()
        cylinder["fromto"] = (
            f"{from_corner[0]} {from_corner[1]} {pz} {to_corner[0]} {to_corner[1]} {pz}"
        )
        cylinders.append(cylinder)

    # Combine the plane and cylinders into one list
    geoms = [plane_dict] + cylinders

    return geoms


def add_geoms_to_worldbody(
    xml_string: str,
    task: Literal[
        "lift-narrow",
        "lift-wide",
        "can-narrow",
        "can-wide",
        "square-narrow",
        "square-wide",
    ],
):
    """
    Takes an existing MuJoCo XML string, adds specified geoms to the worldbody,
    and returns the updated XML string.
    """
    # Parse the existing XML string into an ElementTree object
    root = ET.fromstring(xml_string)

    # Find the worldbody element
    worldbody = root.find("worldbody")
    if worldbody is None:
        # If worldbody doesn't exist, create one
        worldbody = ET.SubElement(root, "worldbody")

    if task == "lift-wide":
        reset_center = (0, 0, 0.8)
        size_xyz = (0.1, 0.1, 0.001)
    elif task == "lift-narrow":
        reset_center = (0, 0, 0.8)
        size_xyz = (0.03, 0.03, 0.001)
    elif task == "can-narrow":
        # <geom size="0.175 0.225 0.001" pos="0.1 -0.25 0.82" type="box" contype="0" conaffinity="0" group="1" rgba="0 0.4 0.8 0.4"/>
        reset_center = (0.1, -0.25, 0.83)
        size_xyz = (0.175, 0.225, 0.001)
    elif task == "can-wide":
        reset_center = (0.1, -0.25, 0.83)
        size_xyz = (0.175, 0.225, 0.001)
    elif task == "square-narrow":
        reset_center = (-0.1125, 0.1675, 0.82)
        # reset_center = (-0.2, 0.1675, 0.82)
        size_xyz = (0.01, 0.115, 0.001)
    elif task == "square-wide":
        reset_center = (-0.1125, 0.1675, 0.82)
        # reset_center = (-0.2, 0.1675, 0.82)
        size_xyz = (0.01, 0.115, 0.001)

    new_geoms = generate_plane_and_border_cylinders(reset_center, size_xyz=size_xyz)

    # Add each new geom to the worldbody
    for geom_attrs in new_geoms:
        ET.SubElement(worldbody, "geom", attrib=geom_attrs)

    # Convert the updated ElementTree back into a string
    updated_xml_string = ET.tostring(root, encoding="unicode")
    return updated_xml_string


def update_xml(
    xml_string: str,
    task: str,  # e.g., "lift-narrow", "lift-wide", etc.
    scaling_origin: tuple = (0.0, 0.0, 0.0),  # Default origin for scaling
    scale_factor: float = 1.0,
) -> str:
    """
    Updates the XML string by scaling the peg object based on the task,
    around a specified origin.
    """
    # Parse the existing XML string into an ElementTree object
    root = ET.fromstring(xml_string)

    scale_factor = np.array(
        [scale_factor, scale_factor, scale_factor]
    )  # Uniform scaling

    scaling_origin = np.array(scaling_origin)  # Convert origin to NumPy array

    # Convert all file paths in 'file' attributes to absolute paths
    # Assuming the base directory is known
    base_dir = (
        pathlib.Path.cwd()
    )  # Replace with the appropriate base directory if known

    for elem in root.findall(".//*[@file]"):
        file_attr = elem.get("file")
        if file_attr:
            # Resolve the absolute path
            absolute_path = (base_dir / file_attr).resolve()
            elem.set("file", str(absolute_path))

    # Find the 'body' element with name 'peg1' (change 'peg1' to the correct name if needed)
    peg_body = root.find(".//body[@name='peg1']")  # Change to 'peg_2' if necessary

    if peg_body is not None:
        # Scale the position of the peg body itself
        peg_pos_str = peg_body.get("pos")
        if peg_pos_str:
            peg_pos = np.array([float(x) for x in peg_pos_str.strip().split()])
            peg_pos = scaling_origin + scale_factor * (peg_pos - scaling_origin)
            peg_body.set("pos", " ".join(map(str, peg_pos)))

        # Scale geoms within the peg body
        for geom in peg_body.findall(".//geom"):
            # Scale 'pos' attribute
            pos_str = geom.get("pos")
            if pos_str:
                pos = np.array([float(x) for x in pos_str.strip().split()])
                pos = scaling_origin + scale_factor * (pos - scaling_origin)
                geom.set("pos", " ".join(map(str, pos)))

            # Scale 'size' attribute
            size_str = geom.get("size")
            if size_str:
                size = np.array([float(x) for x in size_str.strip().split()])
                size *= scale_factor
                geom.set("size", " ".join(map(str, size)))

        # Scale sites within the peg body, if any
        for site in peg_body.findall(".//site"):
            # Scale 'pos' attribute
            pos_str = site.get("pos")
            if pos_str:
                pos = np.array([float(x) for x in pos_str.strip().split()])
                pos = scaling_origin + scale_factor * (pos - scaling_origin)
                site.set("pos", " ".join(map(str, pos)))

            # Scale 'size' attribute
            size_str = site.get("size")
            if size_str:
                size = np.array([float(x) for x in size_str.strip().split()])
                size *= scale_factor
                site.set("size", " ".join(map(str, size)))
    else:
        print("Peg body not found in XML.")

    # Convert the updated ElementTree back into a string
    updated_xml_string = ET.tostring(root, encoding="unicode")
    return updated_xml_string


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # camera to use for generating frames
    parser.add_argument(
        "--camera",
        type=str,
        default="agentview",
    )

    # number of frames in output video
    parser.add_argument(
        "--frames",
        type=int,
        default=10,
    )

    # path to output video
    parser.add_argument(
        "--output",
        type=str,
        default=None,
    )

    parser.add_argument("--task", type=str, default="lift-narrow")
    parser.add_argument(
        "--use-canonical-pose",
        action="store_true",
    )
    parser.add_argument(
        "--viz-reset-plane",
        action="store_true",
    )
    args = parser.parse_args()
    camera_name = args.camera
    num_frames = args.frames
    output_path = args.output

    # Create dict to hold options that will be passed to env creation call
    options = {}

    # print welcome info
    print("Welcome to robosuite v{}!".format(suite.__version__))
    print(suite.__logo__)

    # Choose environment and add it to options
    task = args.task
    output_path = f"resets/{task}-{'viz-plane-' if args.viz_reset_plane else ''}{'canon-pose-' if args.use_canonical_pose else ''}resets.mp4"
    pathlib.Path(output_path).parent.mkdir(exist_ok=True, parents=True)

    update_object_scale: bool = False
    if "wide" in task:
        update_object_scale = True
    if "lift" in task:
        if "wide" in task:
            options["env_name"] = "LiftWide"
        elif "narrow" in task:
            options["env_name"] = "Lift"
    elif "can" in task:
        if "narrow" in task:
            options["env_name"] = "PickPlaceCan"
        elif "wide" in task:
            options["env_name"] = "PickPlaceCanWide"
    elif "square" in task:
        if "narrow" in task:
            options["env_name"] = "NutAssemblySquare"
        elif "wide" in task:
            options["env_name"] = "SquareWide"

    # Else, we simply choose a single (single-armed) robot to instantiate in the environment
    else:
        pass
        # options["robots"] = choose_robots(exclude_bimanual=True)

    options["robots"] = ["Panda"]
    # Load the controller
    options["controller_configs"] = load_controller_config(
        default_controller="OSC_POSE"
    )

    # initialize the task
    env = suite.make(
        **options,
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
    )

    # write a video
    video_writer = imageio.get_writer(output_path, fps=5)
    for i in range(num_frames):
        env.reset()
        post_reset_state = env.sim.get_state()
        xml = env.sim.model.get_xml()
        if update_object_scale:
            if "can" in task:
                scale = np.linspace(0.6, 1.2, num=num_frames)[i]
                env.set_can_scale(scale=scale)
                env.reset()
            if "lift" in task:
                size = np.linspace(0.014, 0.024, num=num_frames)[i]
                size = np.array([size, size, size]) + np.random.uniform(-0.003, 0.003)
                if i == 0:
                    size[2] = 0.018
                env.set_cube_size(size=size)
                env.reset()
            elif "square" in task:
                scale = np.linspace(0.8, 1.1, num=num_frames)[i]
                env.set_square_scale(scale=scale)
                env.reset()
                xml_string = env.sim.model.get_xml()
                scaling_origin = np.array([0.23, 0.1, 0.8])
                xml = update_xml(
                    xml_string,
                    task="square-wide",
                    scaling_origin=scaling_origin,
                    scale_factor=scale,
                )
                env.reset_from_xml_string(xml)
                env.sim.set_state(post_reset_state)
        if args.use_canonical_pose:
            if "lift" in task:
                env.set_cube_pos_quat(
                    pos=np.array([0, 0, 0.8]) - env.cube.bottom_offset,
                    quat=[0.92, 0, 0, 0.383],
                )
            elif "can" in task:
                env.set_can_pos_quat(
                    pos=np.array([0.1, -0.25, 0.82]) - env.objects[-1].bottom_offset,
                    quat=[1, 0, 0, 0],
                )
            elif "square" in task:
                env.set_square_pos_quat(
                    pos=np.array([-0.2, 0.1675, 0.80]) - env.nuts[0].bottom_offset,
                    quat=[0.92, 0, 0, 0.383],
                )
        if args.viz_reset_plane:
            xml = add_geoms_to_worldbody(xml, task=task)
            env.reset_from_xml_string(xml)
            env.sim.set_state(post_reset_state)
        env.sim.forward()

        video_img = env.sim.render(height=512, width=512, camera_name=camera_name)[::-1]
        output_frame_path = pathlib.Path(output_path).with_suffix(
            ""
        )  # Remove .mp4 suffix
        output_frame_path = output_frame_path.with_name(
            f"{output_frame_path.stem}_frame_{i}.png"
        )
        imageio.imwrite(output_frame_path, video_img)

        video_writer.append_data(video_img)
    video_writer.close()
    print(output_path)
