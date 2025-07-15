from pathlib import Path
from typing import List, Optional, Union

import mujoco
import numpy as np
from lxml import etree
from lxml import etree as ET
from scipy.spatial.transform import Rotation as R


def remove_actuator_tag(xml_input: Union[Path, str]) -> str:
    # Parse XML input based on type
    if isinstance(xml_input, Path):
        tree = etree.parse(str(xml_input))
    else:
        tree = etree.ElementTree(etree.fromstring(xml_input))

    root = tree.getroot()

    # Find and remove the <actuator> tag
    actuator_tag = root.find("actuator")
    if actuator_tag is not None:
        root.remove(actuator_tag)
    else:
        print("No <actuator> tag found.")

    # Return the modified XML as a string
    return etree.tostring(root, encoding="unicode", pretty_print=True)


def add_free_joint(xml_input: str, body_name: str, free_joint_name: str) -> str:
    """
    Adds a free joint to the given body in the XML string, removing any existing joint.

    Args:
        xml_input: The input XML string.
        body_name: The name of the body to which the free joint will be added.
        free_joint_name: The name of the free joint to be added.
    """
    # Parse the XML
    root = etree.fromstring(xml_input, parser=None)

    # Find the body with the given name
    body = root.find(f".//body[@name='{body_name}']")
    if body is None:
        raise ValueError(f"Body with name '{body_name}' not found in the XML.")

    # Remove existing joint if any
    existing_joint = body.find("joint")
    if existing_joint is not None:
        body.remove(existing_joint)

    # Create new free joint
    free_joint = etree.Element(
        "joint",
        {"name": free_joint_name, "type": "free"},
        nsmap=None,
    )

    # Insert the free joint as the first child of the body
    body.insert(0, free_joint)

    # Return the modified XML as a string
    return etree.tostring(root, encoding="unicode", pretty_print=True)


def remove_arm_keep_eef(
    input_xml: str,
    robot_base_prefix: str,
    robot_eef_prefix: str,
    add_free_joint: bool = True,
    free_joint_name: Optional[str] = None,
) -> str:
    """
    Removes the robot arm from the given XML while keeping the end effector and, by default, adding a free joint
    to the end effector.

    Args:
        input_xml: The input XML string representing the robot.
        robot_base_prefix: The prefix of the robot base body name to identify the robot arm.
        robot_eef_prefix: The name of the robot hand body to be kept.
        add_free_joint (bool, optional): If True, adds a free joint to the hand. Defaults to True.
    Returns:
        str: The modified XML string with the robot arm removed and the hand kept, and all else the same.
    """

    # Parse the XML
    root = etree.fromstring(input_xml, parser=None)

    # Find the world body
    worldbody = root.find("worldbody")
    assert worldbody is not None, "worldbody not found"

    # Find the base body that starts with the given robot_base_prefix
    base = (
        worldbody.xpath(f".//body[starts-with(@name, '{robot_base_prefix}')]")[0]
        if worldbody.xpath(f".//body[starts-with(@name, '{robot_base_prefix}')]")
        else None
    )

    if base is not None:
        # Find the hand body
        hand = base.find(f".//body[@name='{robot_eef_prefix}']")
        if hand is not None:
            if add_free_joint:
                # Remove existing joint if any
                existing_joint = hand.find("joint")
                if existing_joint is not None:
                    hand.remove(existing_joint)

                if free_joint_name is None:
                    free_joint_name = f"{hand.get('name')}_free_joint"
                # Create new free joint
                free_joint = etree.Element(
                    "joint",
                    {"name": free_joint_name, "type": "free"},
                    nsmap=None,
                )

                # Insert the free joint as the first child of the hand body
                hand.insert(0, free_joint)

            # Remove hand from its parent
            hand_parent = hand.getparent()
            if hand_parent is not None:
                hand_parent.remove(hand)

            # Add hand directly to worldbody
            worldbody.append(hand)

        # Remove the base (which contains the robot arm)
        base_parent = base.getparent()
        if base_parent is not None:
            base_parent.remove(base)
    else:
        import ipdb

        ipdb.set_trace()
        print(f"No base body found with prefix '{robot_base_prefix}'")

    # Convert to string
    return etree.tostring(root, encoding="unicode", pretty_print=True)


def mat2quat(rmat: np.ndarray) -> np.ndarray:
    """
    Returns:
        np.array: (w, x,y,z) float quaternion angles
    """
    return np.roll(R.from_matrix(rmat).as_quat(), 1)


def qpos_to_pos_quat(model: mujoco.MjModel, data: mujoco.MjData, body_id: str):
    # Get world position
    pos = data.xpos[body_id]

    # Get rotation matrix
    rot_mat = data.xmat[body_id].reshape(3, 3)

    # Convert rotation matrix to quaternion (w, x, y, z)
    quat = mat2quat(rot_mat)

    return pos, quat


def update_xml_with_mjmodel(
    xml_string: str,
    model: mujoco.MjModel,
    exclude_body_and_children: List[str] = None,
) -> str:
    """
    Update the positions and orientations of bodies in an XML string based on a MuJoCo model.

    Args:
        xml_string (str): The input XML string.
        model (mujoco.MjModel): The MuJoCo model containing body positions and orientations.
        exclude_body_and_children (List[str], optional): List of body names to exclude, including their children.

    Returns:
        str: The updated XML string.
    """
    # Parse the XML string
    parser = ET.XMLParser(remove_blank_text=True)
    root = ET.fromstring(xml_string, parser)

    # Create set of bodies to exclude (including their children)
    excluded_bodies = set()
    if exclude_body_and_children:
        for body_name in exclude_body_and_children:
            # Add the body itself
            excluded_bodies.add(body_name)
            # Find the body element
            body_elem = root.find(f".//body[@name='{body_name}']")
            if body_elem is not None:
                # Add all child body names
                for child in body_elem.findall(".//body"):
                    child_name = child.get("name")
                    if child_name:
                        excluded_bodies.add(child_name)

    # Iterate through all bodies in the XML
    for body in root.findall(".//body"):
        body_name = body.get("name")
        if body_name and body_name not in excluded_bodies:
            try:
                body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            except ValueError:
                continue  # Skip if the body name is not found in the model

            # Retrieve position and orientation from the model
            pos = model.body_pos[body_id]
            quat = model.body_quat[body_id]

            # Update pos and quat as attributes
            pos_str = f"{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}"
            quat_str = f"{quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}"

            body.set("pos", pos_str)
            body.set("quat", quat_str)

    # Convert to string and return
    return ET.tostring(root, encoding="unicode", pretty_print=True)


def update_xml_with_mjdata(
    xml_string: str,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    exclude_body_and_children: List[str] = None,
) -> str:
    # Parse the XML string
    parser = ET.XMLParser(remove_blank_text=True)
    root = ET.fromstring(xml_string, parser)

    # Create set of bodies to exclude (including their children)
    excluded_bodies = set()
    if exclude_body_and_children:
        for body_name in exclude_body_and_children:
            # Add the body itself
            excluded_bodies.add(body_name)
            # Find the body element
            body_elem = root.find(f".//body[@name='{body_name}']")
            if body_elem is not None:
                # Add all child body names
                for child in body_elem.findall(".//body"):
                    child_name = child.get("name")
                    if child_name:
                        excluded_bodies.add(child_name)
    print("updating xml with mjdata")
    # Iterate through all bodies in the XML
    for body in root.findall(".//body"):
        body_name = body.get("name")
        if body_name and body_name not in excluded_bodies:
            print(f"body_name: {body_name}, id: {model.body(body_name).id}")
            try:
                body_id = model.body(body_name).id
            except Exception:
                continue  # allow cases where model doesn't match the xml
            pos, quat = qpos_to_pos_quat(model, data, body_id)

            # Update pos and quat as attributes
            pos_str = f"{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}"
            quat_str = f"{quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}"

            body.set("pos", pos_str)
            body.set("quat", quat_str)

    # Convert to string and return
    return ET.tostring(root, encoding="unicode", pretty_print=True)


if __name__ == "__main__":
    # Test the function
    input_xml = """<mujoco model="robot">
        <worldbody>
            <body name="robot_base">
      <inertial mass="0.629769" pos="-0.041018 -0.00014 0.049974"
        fullinertia="0.00315 0.00388 0.004285 8.2904e-7 0.00015 8.2299e-6"/>
                <joint name="base_joint" type="free"/>
                <body name="arm_link1">
      <inertial mass="0.629769" pos="-0.041018 -0.00014 0.049974"
        fullinertia="0.00315 0.00388 0.004285 8.2904e-7 0.00015 8.2299e-6"/>
                    <joint name="joint1" type="hinge"/>
                    <body name="arm_link2">
      <inertial mass="0.629769" pos="-0.041018 -0.00014 0.049974"
        fullinertia="0.00315 0.00388 0.004285 8.2904e-7 0.00015 8.2299e-6"/>
                        <joint name="joint2" type="hinge"/>
                        <body name="hand">
                            <geom name="hand_geom" type="sphere" size="0.05"/>
      <inertial mass="0.629769" pos="-0.041018 -0.00014 0.049974"
        fullinertia="0.00315 0.00388 0.004285 8.2904e-7 0.00015 8.2299e-6"/>
                            <joint name="hand_joint" type="hinge"/>
                        </body>
                    </body>
                </body>
            </body>
        <body name="random-to-stay">
                <geom type="box" size="0.05 0.5 0.05" rgba="1 0.4 0.4 1"/>
            </body>
        </worldbody>
    </mujoco>"""

    model = mujoco.MjModel.from_xml_string(input_xml)
    data = mujoco.MjData(model)

    data.qpos[:] = np.random.randn(model.nq)
    # forward kinematics
    mujoco.mj_fwdPosition(model, data)
    print(data.qpos)
    xml_string = update_xml_with_mjdata(input_xml, model, data)
    print(xml_string)
    exit()
    # Now you can call the function with different prefixes
    modified_xml = remove_arm_keep_eef(input_xml, "robot_", "hand", add_free_joint=True)
    print(modified_xml)
    # save the modified xml to a file
    with open("modified.xml", "w") as f:
        f.write(modified_xml)
