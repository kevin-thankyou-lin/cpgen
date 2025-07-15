import datetime
import logging
import pathlib
import uuid
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Set, Tuple, Union
from xml.dom import minidom

import imageio
import mujoco
import numpy as np
import trimesh
from lxml import etree
from mujoco import MjModel, mj_id2name, mjtObj
from scipy.spatial.transform import Rotation as R

import demo_aug


def set_body_pose(
    data: mujoco.MjData,
    model: mujoco.MjModel,
    body_name: str,
    pos_xyz: np.ndarray,
    quat_wxyz: np.ndarray,
) -> None:
    """
    Set the pose of a specified body in the model. Updates data.

    Args:
        data (MjData): The MuJoCo data object.
        model (MjModel): The MuJoCo model object.
        body_name (str): Name of the body to set pose for.
        pose (np.ndarray): Target [x, y, z] pose.
    """
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    # Retrieve the joint type
    joint_id = model.body_jntadr[body_id]
    if joint_id >= 0 and model.jnt_type[joint_id] == 0:  # type 0 indicates a free joint
        # Set qpos for free joint: [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z]
        qpos_start = model.jnt_qposadr[joint_id]
        data.qpos[qpos_start : qpos_start + 3] = pos_xyz
        data.qpos[qpos_start + 3 : qpos_start + 7] = quat_wxyz
    else:
        raise ValueError(
            f"Body '{body_name}' does not have a free joint or is missing a joint."
        )


def update_fixed_joint_objects_in_xml(model: MjModel, xml_string: str) -> str:
    """
    Update the poses of all bodies with *fixed joints* in the XML string,
    to match the pose of the corresponding bodies in the model.

    Why? Some mujoco environments directly update the mujoco model state
    (especially for objects with fixed joints). Thus, syncing all joints wouldn't fully sync the models.
    """
    # Parse XML from string
    root = etree.fromstring(xml_string.encode("utf-8"))

    # Iterate over all bodies in the XML
    for body in root.findall(".//body"):
        # Get the name of the body
        body_name = body.get("name")
        if body_name is None:
            continue

        # Get the body ID in MjModel, ignoring if not found
        body_id = model.body_name2id(body_name)
        if body_id < 0:
            continue

        # Get the joint associated with this body
        joint_id = model.body_jntadr[body_id]
        if joint_id >= 0:
            continue
        # joint_id == -1 implicitly means fixed joint!

        # Update position and orientation based on the current model state
        pos = model.body_pos[body_id]
        quat = model.body_quat[body_id]

        # Convert numpy to string for XML
        pos_str = " ".join(map(str, pos))
        quat_str = " ".join(map(str, quat))

        # Update the XML attributes
        body.set("pos", pos_str)
        body.set("quat", quat_str)

    # Convert updated XML back to string
    updated_xml_string = etree.tostring(
        root, encoding="utf-8", xml_declaration=True, pretty_print=True
    ).decode("utf-8")
    # save updated xml to a file
    with open("updated_xml.xml", "w") as f:
        f.write(updated_xml_string)
    return updated_xml_string


def get_body_pose(
    mj_model: mujoco.MjModel, mj_data: mujoco.MjData, obj_name: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get the pose of an object in the environment.

    Args:
        mj_model (mujoco.MjModel): The MuJoCo model.
        mj_data (mujoco.MjData): The MuJoCo data.
        obj_name (str): Name of the object.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - xyz (np.ndarray): 3D position of the object.
            - wxyz (np.ndarray): Quaternion (w, x, y, z) representing the object's orientation.
    """
    body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
    if body_id == -1:
        raise ValueError(f"Object '{obj_name}' not found in the model.")

    xyz = mj_data.xpos[body_id]
    wxyz = mj_data.xquat[body_id]  # MuJoCo stores quaternions as (w, x, y, z)

    return np.concatenate([xyz, wxyz])


def get_body_velocity(
    mj_model: mujoco.MjModel, mj_data: mujoco.MjData, obj_name: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get the velocity of an object in the environment.

    Args:
        mj_model (mujoco.MjModel): The MuJoCo model.
        mj_data (mujoco.MjData): The MuJoCo data.
        obj_name (str): Name of the object.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - linear_vel (np.ndarray): 3D linear velocity of the object.
            - angular_vel (np.ndarray): 3D angular velocity of the object.
    """
    body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
    if body_id == -1:
        raise ValueError(f"Object '{obj_name}' not found in the model.")

    return mj_data.cvel[body_id]


def has_joint(model: MjModel, joint_name: str) -> bool:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    return joint_id >= 0


def get_body_joints_recursive(
    model: mujoco.MjModel, body_name: str
) -> Dict[str, Tuple[str, int, Optional[Tuple[float, float]]]]:
    """Get names, types, DOFs, and limits of all joints belonging to a body, including its descendants.

    Args:
        model: MuJoCo model.
        body_name: Name of the body.

    Returns:
        A dictionary mapping joint names to (joint type, DOF, (lower limit, upper limit) or None if no limits).
    """
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)

    joint_info = {}

    def collect_joints(body_id: int):
        """Recursively collects joint data for the given body and all its child bodies."""
        for joint_id in range(model.njnt):
            if model.jnt_bodyid[joint_id] == body_id:
                joint_name = mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_JOINT, joint_id
                )
                joint_type = model.jnt_type[joint_id]

                # Determine DOF based on joint type
                if joint_type == mujoco.mjtJoint.mjJNT_FREE:
                    dof = 7  # 3 translation, 4 rotation (quaternion)
                    joint_type_str = "free"
                    joint_limits = None  # Free joints have no meaningful limits
                elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
                    dof = 4  # quaternion rotation
                    joint_type_str = "ball"
                    joint_limits = None  # Ball joints have no meaningful limits
                elif joint_type == mujoco.mjtJoint.mjJNT_HINGE:
                    dof = 1  # single-axis rotation
                    joint_type_str = "hinge"
                    joint_limits = (
                        (model.jnt_range[joint_id, 0], model.jnt_range[joint_id, 1])
                        if model.jnt_limited[joint_id]
                        else None
                    )
                elif joint_type == mujoco.mjtJoint.mjJNT_SLIDE:
                    dof = 1  # single-axis translation
                    joint_type_str = "slide"
                    joint_limits = (
                        (model.jnt_range[joint_id, 0], model.jnt_range[joint_id, 1])
                        if model.jnt_limited[joint_id]
                        else None
                    )
                else:
                    raise ValueError(f"Unexpected joint type: {joint_type}")

                joint_info[joint_name] = (joint_type_str, dof, joint_limits)

        # Recurse into child bodies
        for child_body_id in range(model.nbody):
            if model.body_parentid[child_body_id] == body_id:
                collect_joints(child_body_id)

    collect_joints(body_id)
    return joint_info


def get_top_level_body_names(
    model: mujoco.MjModel, exclude_prefixes: Tuple[str]
) -> List[int]:
    """Get names of all top-level bodies (direct children of the world body).
    Args:
        model: MuJoCo model.
    Returns:
        A list containing IDs of top-level bodies.
    """
    top_level_ids = [
        body_id
        for body_id in range(1, model.nbody)
        if model.body_parentid[body_id] == 0
    ]
    top_level_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        for body_id in top_level_ids
    ]
    res = []
    for name in top_level_names:
        test = [prefix in name for prefix in exclude_prefixes]
        if not any(test):
            res.append(name)
    return res


def get_body_children(
    model: mujoco.MjModel, parent_body_name: str, recursive: bool = False
) -> List[str]:
    """Get all children of a specified body.

    Args:
        model: MuJoCo model.
        parent_body_name: Name of the parent body to find children for.
        recursive: If True, also include all descendants (children of children).

    Returns:
        A list containing names of all child bodies.
    """
    parent_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, parent_body_name)
    if parent_id == -1:
        raise ValueError(f"Body '{parent_body_name}' not found in the model.")

    # Function to get direct children of a body by ID
    def get_direct_children(body_id: int) -> List[int]:
        return [
            child_id
            for child_id in range(model.nbody)
            if model.body_parentid[child_id] == body_id
        ]

    # If recursive, use a helper function to collect all descendants
    if recursive:
        all_children_ids = []

        def collect_descendants(body_id: int):
            children = get_direct_children(body_id)
            all_children_ids.extend(children)
            for child_id in children:
                collect_descendants(child_id)

        collect_descendants(parent_id)
        children_ids = all_children_ids
    else:
        # Get only direct children
        children_ids = get_direct_children(parent_id)

    # Convert IDs to names
    children_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, child_id)
        for child_id in children_ids
    ]

    # Filter out None values (bodies without names)
    return [name for name in children_names if name is not None]


def sync_fixed_joint_objects_mjmodel(
    src_mjmodel: MjModel, target_mjmodel: MjModel, verbose: bool = False
) -> None:
    """
    Syncs the poses of all fixed bodies from source MuJoCo model to the target MuJoCo model.

    Why? Some mujoco environments directly update the mujoco model state
    (especially for objects with fixed joints). Thus, syncing all joints wouldn't fully sync the models.

    Also, I think we don't wanto to sync bodies with non-fixed joints.
    """
    # Iterate over all bodies in the source model
    for body_id in range(src_mjmodel.nbody):
        body_name = mujoco.mj_id2name(src_mjmodel, mujoco.mjtObj.mjOBJ_BODY, body_id)

        if body_name is None:
            continue

        pos = src_mjmodel.body_pos[body_id]
        quat = src_mjmodel.body_quat[body_id]

        # Get the joint associated with this body
        joint_id = src_mjmodel.body_jntadr[body_id]
        if joint_id >= 0:
            # joint_id == -1 implicitly means fixed joint!
            continue

        target_body_id = mujoco.mj_name2id(
            target_mjmodel, mujoco.mjtObj.mjOBJ_BODY, body_name
        )
        if target_body_id < 0:
            continue

        # Update position and orientation based on the source model state
        pos = src_mjmodel.body_pos[body_id]
        quat = src_mjmodel.body_quat[body_id]
        # Set the target model's body position and orientation
        target_mjmodel.body_pos[target_body_id] = pos
        target_mjmodel.body_quat[target_body_id] = quat

        if verbose:
            print(f"Synced body {body_name} to {pos, quat}.")


def sync_geoms_mjmodel(
    src_mjmodel: mujoco.MjModel,
    target_mjmodel: mujoco.MjModel,
    verbose: bool = False,
) -> None:
    """
    Syncs the geom positions from the source model to the target model based on geom names.
    """
    # sync geom positions
    for i in range(src_mjmodel.ngeom):
        geom_name = mujoco.mj_id2name(src_mjmodel, mujoco.mjtObj.mjOBJ_GEOM, i)
        if geom_name is None:
            continue

        target_geom_id = mujoco.mj_name2id(
            target_mjmodel, mujoco.mjtObj.mjOBJ_GEOM, geom_name
        )
        if target_geom_id < 0:
            continue

        # Update geom position
        target_mjmodel.geom_pos[target_geom_id] = src_mjmodel.geom_pos[i]
        target_mjmodel.geom_quat[target_geom_id] = src_mjmodel.geom_quat[i]

        if verbose:
            print(
                f"Synced geom {geom_name} to {src_mjmodel.geom_pos[i], src_mjmodel.geom_quat[i]}."
            )


def sync_mjdata(
    src_mjmodel: mujoco.MjModel,
    src_mjdata: mujoco.MjData,
    target_mjmodel: mujoco.MjModel,
    target_mjdata: mujoco.MjData,
    verbose: bool = False,
) -> None:
    """
    Syncs the joint values from `src_mjdata` to `target_mjdata` based on the joint names
    in `src_mjmodel` and `target_mjmodel`.

    Args:
        src_mjdata: The source mjdata object.
        src_mjmodel: The source mjmodel object.
        target_mjdata: The target mjdata object.
        target_mjmodel: The target mjmodel object.
        verbose: If True, prints detailed synchronization information.
    """
    # Map joint names to indexes for both models
    src_joint_name_to_indexes = get_joint_name_to_indexes(src_mjmodel)
    target_joint_name_to_indexes = get_joint_name_to_indexes(target_mjmodel)

    for joint_name, target_joint_indexes in target_joint_name_to_indexes.items():
        if joint_name not in src_joint_name_to_indexes:
            if verbose:
                print(
                    f"Skipping {joint_name} as it does not exist in the source model."
                )
            continue

        src_joint_indexes = src_joint_name_to_indexes[joint_name]

        if verbose:
            print(f"Syncing joint: {joint_name}")
            print(f"Source qpos: {src_mjdata.qpos[src_joint_indexes]}")
            print(
                f"Target qpos before sync: {target_mjdata.qpos[target_joint_indexes]}"
            )

        # Synchronize joint positions
        target_mjdata.qpos[target_joint_indexes] = src_mjdata.qpos[src_joint_indexes]

        if verbose:
            print(f"Target qpos after sync: {target_mjdata.qpos[target_joint_indexes]}")


def get_geom_name_to_indexes(model: mujoco.MjModel) -> Dict[str, List[int]]:
    """
    Creates a mapping from geom names to their indices in the given MuJoCo model.
    Unnamed geoms are assigned a name based on their parent body's name.
    If the parent body has no name, a random 8-character hex string is used.

    Args:
        model: The MuJoCo model.

    Returns:
        A dictionary mapping geom names to lists of indices.
    """
    geom_name_to_indexes: Dict[str, List[int]] = {}
    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        if not name:
            body_id = model.geom_bodyid[i]
            parent_body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if not parent_body:
                parent_body = uuid.uuid4().hex[:8]
            name = f"{parent_body}_g{i}"
        geom_name_to_indexes.setdefault(name, []).append(i)
    return geom_name_to_indexes


def sync_sizes_mjmodel(
    src_mjmodel: mujoco.MjModel,
    target_mjmodel: mujoco.MjModel,
    verbose: bool = False,
) -> None:
    """
    Synchronizes geom sizes from the source model to the target model based on geom names.

    Args:
        src_mjmodel: Source model with reference geom sizes.
        target_mjmodel: Target model to update with source geom sizes.
        verbose: If True, prints detailed synchronization logs.
    """
    src_geom_indexes = get_geom_name_to_indexes(src_mjmodel)
    target_geom_indexes = get_geom_name_to_indexes(target_mjmodel)

    for geom_name, tgt_indexes in target_geom_indexes.items():
        if geom_name not in src_geom_indexes:
            if verbose:
                print(f"Skipping geom '{geom_name}': not present in source model.")
            continue

        src_indexes = src_geom_indexes[geom_name]
        if verbose:
            print(f"Syncing geom '{geom_name}'")
            print(f"Source sizes: {src_mjmodel.geom_size[src_indexes]}")
            print(f"Target sizes before sync: {target_mjmodel.geom_size[tgt_indexes]}")
        target_mjmodel.geom_size[tgt_indexes] = src_mjmodel.geom_size[src_indexes]
        if verbose:
            print(f"Target sizes after sync: {target_mjmodel.geom_size[tgt_indexes]}")


def sync_mjmodel_mjdata(
    src_mjmodel: mujoco.MjModel,
    src_mjdata: mujoco.MjData,
    target_mjmodel: mujoco.MjModel,
    target_mjdata: mujoco.MjData,
    verbose: bool = False,
) -> None:
    """
    Synchronize positions and orientations of top-level bodies from the source MuJoCo model/data
    to the target MuJoCo model/data.

    Args:
        src_mjmodel: The source MuJoCo model.
        src_mjdata: The source MuJoCo data.
        target_mjmodel: The target MuJoCo model.
        target_mjdata: The target MuJoCo data.
        verbose: If True, print the names of bodies that are skipped.

    Note: if a body is fixed to an object, syncing all joints wouldn't fully sync the models
    """
    sync_sizes_mjmodel(src_mjmodel, target_mjmodel, verbose=verbose)
    sync_mjdata(src_mjmodel, src_mjdata, target_mjmodel, target_mjdata, verbose=verbose)
    sync_fixed_joint_objects_mjmodel(src_mjmodel, target_mjmodel, verbose=verbose)
    sync_geoms_mjmodel(src_mjmodel, target_mjmodel, verbose=verbose)


def get_joint_name_to_indexes(model: MjModel) -> Dict[str, np.ndarray]:
    """
    Create a mapping from joint names to their corresponding qpos indices.
    Takes into account different joint types:
    - Free joint: 7 values (3 for position, 4 for quaternion)
    - Ball joint: 4 values (quaternion)
    - Slide/Hinge joint: 1 value
    - Fixed joint: 0 values (not included in qpos)

    Args:
        model (MjModel): The MuJoCo model object.

    Returns:
        Dictionary mapping joint names to arrays of qpos indices.
    """
    name_to_indices = {}
    qpos_index = 0

    for i in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if not joint_name:
            continue

        joint_type = model.jnt_type[i]

        # Determine number of DOFs based on joint type
        if joint_type == mujoco.mjtJoint.mjJNT_FREE:
            dof_count = 7  # 3 for position, 4 for quaternion
        elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
            dof_count = 4  # quaternion
        elif joint_type in [mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE]:
            dof_count = 1
        elif joint_type == mujoco.mjtJoint.mjJNT_FIXED:
            continue  # Skip fixed joints as they don't appear in qpos
        else:
            raise ValueError(f"Unexpected joint type: {joint_type}")

        indices = np.arange(qpos_index, qpos_index + dof_count)
        name_to_indices[joint_name] = indices
        qpos_index += dof_count

    return name_to_indices


def get_joint_name_to_qpos(
    mjmodel: mujoco.MjModel, mjdata: mujoco.MjData
) -> Dict[str, float]:
    """
    Extract a mapping from joint names to their associated qpos values.

    Args:
        mjmodel: The MuJoCo model.
        mjdata: The MuJoCo data.

    Returns:
        A dictionary mapping joint names to their corresponding qpos values.
    """
    joint_name_to_qpos = {}

    for joint_id in range(mjmodel.njnt):
        joint_name = mujoco.mj_id2name(mjmodel, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if joint_name is not None:
            joint_addr = mjmodel.jnt_qposadr[joint_id]
            joint_type = mjmodel.jnt_type[joint_id]

            # Determine number of DOFs based on joint type
            if joint_type == mujoco.mjtJoint.mjJNT_FREE:
                dof_count = 7  # 3 for position, 4 for quaternion
            elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
                dof_count = 4  # quaternion
            elif joint_type in [
                mujoco.mjtJoint.mjJNT_HINGE,
                mujoco.mjtJoint.mjJNT_SLIDE,
            ]:
                dof_count = 1
            elif joint_type == mujoco.mjtJoint.mjJNT_FIXED:
                continue  # Skip fixed joints as they don't appear in qpos
            else:
                raise ValueError(f"Unexpected joint type: {joint_type}")

            indices = np.arange(joint_addr, joint_addr + dof_count)
            joint_name_to_qpos[joint_name] = mjdata.qpos[indices]

    return joint_name_to_qpos


def get_joint_name_to_qvel(
    mjmodel: mujoco.MjModel, mjdata: mujoco.MjData
) -> Dict[str, float]:
    """
    Extract a mapping from joint names to their associated qvel values.

    Args:
        mjmodel: The MuJoCo model.
        mjdata: The MuJoCo data.

    Returns:
        A dictionary mapping joint names to their corresponding qvel values.
    """
    joint_name_to_qvel = {}

    for joint_id in range(mjmodel.njnt):
        joint_name = mujoco.mj_id2name(mjmodel, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if joint_name is not None:
            joint_addr = mjmodel.jnt_dofadr[joint_id]
            joint_type = mjmodel.jnt_type[joint_id]

            # Determine number of DOFs based on joint type
            if joint_type == mujoco.mjtJoint.mjJNT_FREE:
                dof_count = 6  # 3 for linear velocity, 3 for angular velocity
            elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
                dof_count = 3  # angular velocity
            elif joint_type in [
                mujoco.mjtJoint.mjJNT_HINGE,
                mujoco.mjtJoint.mjJNT_SLIDE,
            ]:
                dof_count = 1
            elif joint_type == mujoco.mjtJoint.mjJNT_FIXED:
                continue  # Skip fixed joints as they don't appear in qvel
            else:
                raise ValueError(f"Unexpected joint type: {joint_type}")

            indices = np.arange(joint_addr, joint_addr + dof_count)
            joint_name_to_qvel[joint_name] = mjdata.qvel[indices]

    return joint_name_to_qvel


def set_joint_name_to_qpos(
    joint_name_to_qpos: Dict[str, float],
    target_mjmodel: mujoco.MjModel,
    target_mjdata: mujoco.MjData,
) -> None:
    """
    Set the qpos values of joints in the target model and data using a mapping from joint names.

    Args:
        joint_name_to_qpos: A dictionary mapping joint names to qpos values.
        target_mjmodel: The target MuJoCo model.
        target_mjdata: The target MuJoCo data.
    """
    for joint_name, qpos_value in joint_name_to_qpos.items():
        joint_id = mujoco.mj_name2id(
            target_mjmodel, mujoco.mjtObj.mjOBJ_JOINT, joint_name
        )
        if joint_id == -1:
            print(f"Warning: Joint '{joint_name}' not found in the target model.")
            continue

        joint_addr = target_mjmodel.jnt_qposadr[joint_id]
        target_mjdata.qpos[joint_addr] = qpos_value


def get_body_geom_ids_by_group(
    model: mujoco.MjModel, body_id: int, group: int
) -> List[int]:
    """Get all geoms belonging to a given body that are in a specified group.

    Adapted from https://github.com/kevinzakka/mink/blob/main/mink/utils.py
    """
    geom_start = model.body_geomadr[body_id]
    geom_end = geom_start + model.body_geomnum[body_id]
    return [
        geom_id
        for geom_id in range(geom_start, geom_end)
        if model.geom_group[geom_id] == group
    ]


def get_subtree_geom_ids_by_group(
    model: mujoco.MjModel, body_id: int, group: int
) -> list[int]:
    """Get all geoms belonging to a subtree starting at a given body, filtered by group.

    Args:
        model: MuJoCo model.
        body_id: ID of body where subtree starts.
        group: Group ID to filter geoms.

    Returns:
        A list containing all subtree geom ids in the specified group.

    Adapted from https://github.com/kevinzakka/mink/blob/main/mink/utils.py
    """

    def gather_geoms(body_id: int) -> list[int]:
        geoms: list[int] = []
        geom_start = model.body_geomadr[body_id]
        geom_end = geom_start + model.body_geomnum[body_id]
        geoms.extend(
            geom_id
            for geom_id in range(geom_start, geom_end)
            if model.geom_group[geom_id] == group
        )
        children = [i for i in range(model.nbody) if model.body_parentid[i] == body_id]
        for child_id in children:
            geoms.extend(gather_geoms(child_id))
        return geoms

    return gather_geoms(body_id)


def get_all_body_ids(model: mujoco.MjModel) -> List[int]:
    """Get all body IDs in the MuJoCo model."""
    return list(range(model.nbody))


def get_all_body_names(
    model: mujoco.MjModel, exclude_keywords: Optional[List[str]] = None
) -> List[str]:
    """Get all body names in the MuJoCo model, excluding those containing specified keywords."""
    exclude_keywords = exclude_keywords or []

    return [
        name
        for body_id in range(model.nbody)
        if (name := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id))
        and not any(keyword in name for keyword in exclude_keywords)
    ]


def get_top_level_bodies(
    model: mujoco.MjModel, exclude_prefixes: Optional[List[str]] = None
) -> List[int]:
    """Get IDs of all top-level bodies (direct children of the world body),
    optionally excluding bodies with names that start with any specified prefixes.

    Args:
        model: MuJoCo model.
        exclude_prefixes: Optional list of prefix strings to exclude (e.g., ["robot", "sensor"]).

    Returns:
        A list containing IDs of top-level bodies that do not start with any of the specified prefixes.
    """

    def has_prefix(name: str) -> bool:
        return any(name.startswith(prefix) for prefix in exclude_prefixes or [])

    top_level_bodies = []
    for body_id in range(1, model.nbody):
        if model.body_parentid[body_id] == 0:
            body_name = model.names[model.name_bodyadr[body_id] :].decode()
            if not has_prefix(body_name):
                top_level_bodies.append(body_id)

    return top_level_bodies


def get_top_level_body_contact_pairs(
    mj_model: mujoco.MjModel, mj_data: mujoco.MjData
) -> list[tuple[str, str]]:
    """
    Extracts contact pairs at the highest-level body representation in MuJoCo.

    Args:
        mj_model (MjModel): MuJoCo model containing body and geometry information.
        mj_data (MjData): MuJoCo data containing the current state, including contacts.

    Returns:
        list[tuple[str, str]]: Unique contact pairs at the top-level body level.
    """

    def get_top_level_body(body_id: int) -> int:
        """Recursively finds the top-level parent of a given body ID."""
        while mj_model.body_parentid[body_id] != 0:  # 0 is the world body (root)
            body_id = mj_model.body_parentid[body_id]
        return body_id

    contact_pairs = set()

    for i in range(mj_data.ncon):
        contact = mj_data.contact[i]  # Get individual contact

        geom1_id, geom2_id = contact.geom1, contact.geom2  # Contacting geometries

        # Map geoms to their highest-level parent body
        body1_id = get_top_level_body(mj_model.geom_bodyid[geom1_id])
        body2_id = get_top_level_body(mj_model.geom_bodyid[geom2_id])

        # Retrieve body names
        body1_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, body1_id)
        body2_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, body2_id)

        if body1_name and body2_name and body1_name != body2_name:
            contact_pairs.add((body1_name, body2_name))

    return list(contact_pairs)


def filter_body_ids_by_prefixes(
    model: mujoco.MjModel, body_ids: List[int], prefixes: List[str]
) -> List[int]:
    """Filter out body IDs with names starting with any of the specified prefixes.

    Args:
        model: MuJoCo model.
        body_ids: List of body IDs to filter.
        prefixes: List of prefix strings to filter out (e.g., ["robot", "sensor"]).

    Returns:
        A list containing body IDs that do not start with any of the specified prefixes.
    """

    def has_prefix(name: str) -> bool:
        return any(name.startswith(prefix) for prefix in prefixes)

    filtered_body_ids = []
    for body_id in body_ids:
        body_name = model.names[model.name_bodyadr[body_id] :].decode()
        if not has_prefix(body_name):
            filtered_body_ids.append(body_id)

    return filtered_body_ids


def get_body_name(model: mujoco.MjModel, body_id: int) -> str:
    """Get the name of a body given its ID.

    Args:
        model: MuJoCo model.
        body_id: ID of the body.

    Returns:
        The name of the body as a string.
    """
    name_start = model.name_bodyadr[body_id]
    return model.names[name_start:].decode().split("\0", 1)[0]


def get_body_contact_pairs(
    mj_model: mujoco.MjModel, mj_data: mujoco.MjData, body_names: list[str]
) -> list[tuple[str, str]]:
    """
    Extracts contact pairs for the specified bodies in MuJoCo, ensuring that contacts involving
    child bodies of the specified ones are also included under the first matching ancestor in body_names.

    Args:
        mj_model (MjModel): MuJoCo model containing body and geometry information.
        mj_data (MjData): MuJoCo data containing the current state, including contacts.
        body_names (list[str]): List of body names to check for contacts.

    Returns:
        list[tuple[str, str]]: Unique contact pairs involving the specified bodies.
    """

    def get_nearest_specified_body(body_id: int) -> str:
        """Walk up the hierarchy to find the first matching body in body_names."""
        while body_id != 0:  # Stop at the world body (ID 0)
            body_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            print(f"body_name: {body_name}")
            if body_name in body_names:
                return body_name  # Return the first encountered specified body
            body_id = mj_model.body_parentid[body_id]  # Move up one level
        return None  # No relevant body found

    contact_pairs = set()
    for i in range(mj_data.ncon):
        contact = mj_data.contact[i]  # Get individual contact

        geom1_id, geom2_id = contact.geom1, contact.geom2  # Contacting geometries

        # Get corresponding body IDs
        body1_id = mj_model.geom_bodyid[geom1_id]
        body2_id = mj_model.geom_bodyid[geom2_id]

        # Find nearest specified bodies
        body1_name = get_nearest_specified_body(body1_id)
        body2_name = get_nearest_specified_body(body2_id)

        # Only include contacts if both bodies are in body_names and are distinct
        if body1_name and body2_name and body1_name != body2_name:
            contact_pairs.add((body1_name, body2_name))

    return list(contact_pairs)


def get_geom_names(model: mujoco.MjModel, geom_ids: List[int]) -> List[str]:
    """
    Returns the geom names for given geom IDs in a MuJoCo model.

    :param model: MjModel, the MuJoCo model.
    :param geom_ids: List[int], list of geom IDs.
    :return: List[str], list of geom names.
    """
    return [
        mj_id2name(model, mjtObj.mjOBJ_GEOM, geom_id)
        if mj_id2name(model, mjtObj.mjOBJ_GEOM, geom_id) is not None
        else ""
        for geom_id in geom_ids
    ]


def check_geom_collisions(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom_pairs: List[Tuple[List[int], List[int]]],
    collision_activation_dist: float = 0.015,
    verbose: bool = False,
) -> List[Tuple[Set[int], Set[int]]]:
    """
    For a list of geom_pairs, checks for collisions between specified pairs of of geom lists.

    Parameters:
        model (MjModel): The MuJoCo model object.
        data (MjData): The MuJoCo data object.
        geom_pairs (List[Tuple[List[int], List[int]]]): List of tuples with sets of geom IDs to check for collisions.

    Returns:
        List[Tuple[Set[int], Set[int]]]: List of geom sets that are in collision.
    """
    mujoco.mj_fwdPosition(model, data)  # Update positions and contacts

    # Convert each set of geom pairs to a frozenset for fast lookup
    geom_pairs_set = {(frozenset(pair[0]), frozenset(pair[1])) for pair in geom_pairs}
    collisions = []

    # Iterate through detected contacts to find matching sets
    for i in range(data.ncon):
        contact = data.contact[i]
        geom1, geom2 = contact.geom1, contact.geom2

        # Check if either geom is in one set and the other geom in the corresponding paired set
        for set_a, set_b in geom_pairs_set:
            if (geom1 in set_a and geom2 in set_b) or (
                geom2 in set_a and geom1 in set_b
            ):
                if verbose:
                    collision_geom_names = get_geom_names(model, [geom1, geom2])
                    print(
                        f"Collision between {collision_geom_names}; distance: {contact.dist}"
                    )
                collisions.append((set_a, set_b))
                return collisions  # Found a collision for this contact, no need to check further sets

    fromto = np.empty(6)
    for geoms_1, geoms_2 in geom_pairs_set:
        for geom_id_1 in geoms_1:
            for geom_id_2 in geoms_2:
                dist = mujoco.mj_geomDistance(
                    model,
                    data,
                    geom_id_1,
                    geom_id_2,
                    collision_activation_dist,
                    fromto=fromto,
                )
                if dist < collision_activation_dist:
                    if verbose:
                        print(
                            f"Collision between {geom_id_1} and {geom_id_2}; distance: {dist}"
                        )
                    collisions.append((geoms_1, geoms_2))
                    break

    return collisions


def get_min_geom_distance(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom_pairs: List[Tuple[List[int], List[int]]],
    verbose: bool = False,
    activation_dist: float = 0.02,
) -> float:
    """
    Computes the minimum distance between pairs of geoms specified in geom_pairs.

    Parameters:
        model (MjModel): The MuJoCo model object.
        data (MjData): The MuJoCo data object.
        geom_pairs (List[Tuple[List[int], List[int]]]): List of tuples with sets of geom IDs to compute distances.
        verbose (bool): If True, prints detailed distance information.
    """
    mujoco.mj_fwdPosition(model, data)  # Update positions and contacts

    fromto = np.empty(6)
    min_dist = activation_dist

    # Iterate over the specified geom pairs
    for geoms_1, geoms_2 in geom_pairs:
        for geom_id_1 in geoms_1:
            for geom_id_2 in geoms_2:
                dist = mujoco.mj_geomDistance(
                    model,
                    data,
                    geom_id_1,
                    geom_id_2,
                    activation_dist,
                    fromto=fromto,
                )
                if verbose:
                    print(
                        f"Distance between geom {geom_id_1} and geom {geom_id_2}: {dist}"
                    )
                min_dist = min(min_dist, dist)

    return min_dist


def convert_xml_to_transformed_mesh(
    xml: str,
    output_dir: pathlib.Path,
    transform_type_seq: List[str],
    transform_params_seq: Dict[str, np.ndarray],
    apply_all_transforms: bool = False,
    original_obj_joint_qpos: Optional[np.ndarray] = None,  # pos + quat_wxyz
    verbose: bool = False,
    recenter_vertices: bool = False,
    mesh_name_to_mesh_path: Optional[Dict[str, str]] = None,
) -> Tuple[
    pathlib.Path,
    Dict[str, pathlib.Path],
    Dict[str, Dict[str, Union[np.ndarray, float]]],
]:
    """
    Args:
        recenter_vertices: Whether to recenter the vertices to be centered at the origin. Only perform a translation.

    TODO(klin): continue

    Creates, transforms and exports meshes from an XML string. Only supports box geoms.

    Return a list of paths to the exported meshes.
    """
    overall_mesh_path: pathlib.Path = None
    mesh_name_to_component_mesh_path: Dict[str, pathlib.Path] = {}
    mesh_name_to_mass_properties: Dict[str, Dict[str, Union[np.ndarray, float]]] = {}

    root = ET.fromstring(xml)
    overall_mesh = trimesh.Trimesh()

    body_pos = [
        float(x)
        for x in (root.get("pos") if root.get("pos") is not None else "0 0 0").split()
    ]

    # original mujoco pose of the object when we first nerf'ed things
    if original_obj_joint_qpos is None:
        if "Nut" in root.get("name"):
            # original_obj_joint_qpos = np.array([-0.11118988, 0.17443746, 0.89, -0.40031812, 0.0, 0.0, 0.91637624])
            original_obj_joint_qpos = np.array(
                [-0.11118988, 0.17443746, 0.834, -0.40031812, 0.0, 0.0, 0.91637624]
            )
            # from timestep=50 for square
            original_obj_joint_qpos = np.array(
                [-0.11118766, 0.17443986, 0.82997895, -0.40031811, 0, 0, 0.91637636]
            )
            # robosuite's _get_observations() returns xyzw?
            # the above is wxyz I believe - check what does env.step() give?
            print(
                f"TODO(klin): automatically get the correct original_obj_joint_qpos for {root.get('name')}."
                "Needed for using mujoco geometries for objects (rather than using nerf extracted meshes)."
            )
            # original_obj_joint_qpos = np.array(
            #     [0.20072371, 0.07480421, 0.96665007, -0.06685157, 0.01189282, 0.00234283, 0.9976893]
            # )
        elif "peg" in root.get("name"):
            # imperfect interface to transforms later on --- if don't specify this, fails later I think
            original_obj_joint_qpos = np.array([0, 0, 0, 1, 0, 0, 0])
        elif "cube" in root.get("name"):
            original_obj_joint_qpos = np.array(
                [
                    2.64492582e-02,
                    2.69810917e-02,
                    8.21208589e-01,
                    2.46631196e-01,
                    1.80507473e-07,
                    3.25642196e-07,
                    9.69109412e-01,
                ]
            )
        elif "can" in root.get("name"):
            original_obj_joint_qpos = np.array(
                [
                    0.12000000000000001,
                    -0.20000000000000004,
                    0.8600000000000001,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            )
        else:
            original_obj_joint_qpos = np.array([0, 0, 0, 1, 0, 0, 0])

    new_obj_pos = np.array([0, 0, 0])
    # convert to 4 x 4 matrix
    original_pose = np.eye(4)
    original_quat_wxyz = original_obj_joint_qpos[3:]
    original_quat_xyzw = np.array(
        [
            original_quat_wxyz[1],
            original_quat_wxyz[2],
            original_quat_wxyz[3],
            original_quat_wxyz[0],
        ]
    )

    original_pose[:3, :3] = R.from_quat(original_quat_xyzw).as_matrix()
    original_pose[:3, 3] = original_obj_joint_qpos[:3]

    # append current time to output_dir
    output_dir = (
        output_dir
        / datetime.datetime.now().strftime("%Y-%m-%d")
        / datetime.datetime.now().strftime("%H-%M-%S-%f")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    for geom in root.findall(".//geom"):
        geom_type = geom.get("type")
        geom_pos = [
            float(x)
            for x in (
                geom.get("pos") if geom.get("pos") is not None else "0 0 0"
            ).split()
        ]
        if geom_type == "box":
            geom_size = [float(x) for x in geom.get("size").split()]
            geom_size = np.array(geom_size) * 2  # Double the size to get the extents
            mesh = trimesh.creation.box(extents=geom_size)
            mesh.apply_translation(body_pos)
            mesh.apply_translation(geom_pos)
            mesh.apply_transform(original_pose)

            if apply_all_transforms:
                for i, transf_type in enumerate(transform_type_seq):
                    # get the parameters for the transformation
                    # get keys corresponding to time i
                    original_vertices = np.array(mesh.vertices)  # (N, 3)

                    original_vertices_homog = np.concatenate(
                        [original_vertices, np.ones((original_vertices.shape[0], 1))],
                        axis=1,
                    )
                    keys = [
                        k
                        for k in transform_params_seq.keys()
                        if int(k.split(":")[0]) == i
                    ]
                    if transf_type == "SCALE":
                        assert (
                            len(keys) == 2
                        ), "scale transform should have 2 keys: X_scale and X_scale_origin"
                        X_scale = transform_params_seq[keys[0]]
                        X_scale_origin = transform_params_seq[keys[1]]
                        mesh.vertices = (
                            X_scale_origin
                            @ (
                                X_scale
                                @ (
                                    np.linalg.inv(X_scale_origin)
                                    @ original_vertices_homog.T
                                )
                            )
                        ).T[:, :3]
                    elif transf_type == "SE3":
                        assert (
                            len(keys) == 2
                        ), "scale transform should have 2 keys: X_scale and X_scale_origin"
                        # update the vertices of mesh with an SE3 transform
                        X_SE3 = transform_params_seq[keys[0]]
                        X_SE3_origin = transform_params_seq[keys[1]]
                        mesh.vertices = (
                            X_SE3_origin
                            @ (
                                X_SE3
                                @ (
                                    np.linalg.inv(X_SE3_origin)
                                    @ original_vertices_homog.T
                                )
                            )
                        ).T[:, :3]
                    elif transf_type == "SHEAR":
                        X_shear = transform_params_seq[keys[0]]
                        X_shear_origin = transform_params_seq[keys[1]]
                        mesh.vertices = (
                            X_shear_origin
                            @ (
                                X_shear
                                @ (
                                    np.linalg.inv(X_shear_origin)
                                    @ original_vertices_homog.T
                                )
                            )
                        ).T[:, :3]
                    else:
                        raise NotImplementedError(
                            f"Transform type {transf_type} not supported"
                        )

            mesh_name = (
                geom.get("name")
                if geom.get("name")
                else root.get("name") + str(len(mesh_name_to_component_mesh_path))
            )

            # TODO(klin) convex decomp here!
            output_filename = f"{mesh_name}.stl"
            mesh.export(output_dir / output_filename)

            overall_mesh += mesh  # Add to the overall mesh
            mesh_name_to_component_mesh_path[mesh_name] = output_dir / output_filename
            mesh_name_to_mass_properties[mesh_name] = mesh.mass_properties
            if verbose:
                logging.info(f"Saved {mesh_name} to {output_dir / output_filename}")

        elif geom_type == "mesh":
            geom_name = (
                geom.get("name")
                if geom.get("name")
                else root.get("name") + str(len(mesh_name_to_component_mesh_path))
            )
            geom_mesh_name = geom.get("mesh")
            geom_mesh_path = mesh_name_to_mesh_path[geom_mesh_name]
            mesh = trimesh.load_mesh(geom_mesh_path)
            mesh.apply_translation(body_pos)
            mesh.apply_translation(geom_pos)
            mesh.apply_transform(original_pose)

            if apply_all_transforms:
                for i, transf_type in enumerate(transform_type_seq):
                    # get the parameters for the transformation
                    # get keys corresponding to time i
                    original_vertices = np.array(mesh.vertices)  # (N, 3)

                    original_vertices_homog = np.concatenate(
                        [original_vertices, np.ones((original_vertices.shape[0], 1))],
                        axis=1,
                    )
                    keys = [
                        k
                        for k in transform_params_seq.keys()
                        if int(k.split(":")[0]) == i
                    ]
                    if transf_type == "SCALE":
                        assert (
                            len(keys) == 2
                        ), "scale transform should have 2 keys: X_scale and X_scale_origin"
                        X_scale = transform_params_seq[keys[0]]
                        X_scale_origin = transform_params_seq[keys[1]]
                        mesh.vertices = (
                            X_scale_origin
                            @ (
                                X_scale
                                @ (
                                    np.linalg.inv(X_scale_origin)
                                    @ original_vertices_homog.T
                                )
                            )
                        ).T[:, :3]
                    elif transf_type == "SE3":
                        assert (
                            len(keys) == 2
                        ), "scale transform should have 2 keys: X_scale and X_scale_origin"
                        # update the vertices of mesh with an SE3 transform
                        X_SE3 = transform_params_seq[keys[0]]
                        X_SE3_origin = transform_params_seq[keys[1]]
                        mesh.vertices = (
                            X_SE3_origin
                            @ (
                                X_SE3
                                @ (
                                    np.linalg.inv(X_SE3_origin)
                                    @ original_vertices_homog.T
                                )
                            )
                        ).T[:, :3]
                    elif transf_type == "SHEAR":
                        X_shear = transform_params_seq[keys[0]]
                        X_shear_origin = transform_params_seq[keys[1]]
                        mesh.vertices = (
                            X_shear_origin
                            @ (
                                X_shear
                                @ (
                                    np.linalg.inv(X_shear_origin)
                                    @ original_vertices_homog.T
                                )
                            )
                        ).T[:, :3]
                    else:
                        raise NotImplementedError(
                            f"Transform type {transf_type} not supported"
                        )

            output_filename = f"{geom_name}.stl"
            mesh.export(output_dir / output_filename)
            overall_mesh += mesh  # Add to the overall mesh
            mesh_name_to_component_mesh_path[geom_name] = output_dir / output_filename
            mesh_name_to_mass_properties[geom_name] = mesh.mass_properties
        else:
            raise NotImplementedError(f"Geom type {geom_type} not supported")

    if recenter_vertices:
        # Compute the center of mass of the overall mesh
        center_of_mass = overall_mesh.center_mass
        new_obj_pos = center_of_mass
        # Translate all vertices of each component mesh and the overall mesh
        for mesh_name, component_mesh_path in mesh_name_to_component_mesh_path.items():
            component_mesh = trimesh.load_mesh(component_mesh_path)
            component_mesh.apply_translation(-center_of_mass)
            component_mesh.export(component_mesh_path)  # Overwrite the existing file
            mesh_name_to_mass_properties[mesh_name] = (
                component_mesh.mass_properties
            )  # Update mass properties
        overall_mesh.apply_translation(-center_of_mass)

    # overall_mesh is for visualization purposes only
    overall_mesh_path = output_dir / "overall_mesh.stl"
    overall_mesh.export(overall_mesh_path)
    if verbose:
        logging.info(f"Saved overall mesh to {overall_mesh_path}")
        logging.info(f"Number of components: {len(mesh_name_to_component_mesh_path)}")
        logging.info(f"overall_mesh_path: {overall_mesh_path}")
    return (
        overall_mesh_path,
        mesh_name_to_component_mesh_path,
        mesh_name_to_mass_properties,
        new_obj_pos,
    )


def add_camera_xml_to_xml(
    source_xml_file: str,
    destination_xml_str: str,
    parent_body_name: str = "robot0_right_hand",
    debug_file_path: str = "",
) -> str:
    """Assumes that the source XML file contains camera elements that need to be appended to the destination XML file."""
    # Parse the source XML file to extract camera elements
    source_tree = ET.parse(source_xml_file)
    source_root = source_tree.getroot()
    cameras = list(source_root.findall("camera"))

    # Parse the destination XML file to find the specified parent body
    dest_root = ET.fromstring(destination_xml_str)

    parent_body = dest_root.find(f".//body[@name='{parent_body_name}']")

    # Check if the parent body exists
    if parent_body is None:
        import ipdb

        ipdb.set_trace()
        raise ValueError(
            f"No body with name {parent_body_name} found in destination XML."
        )

    # Append each camera element to the parent body
    for camera in cameras:
        parent_body.append(camera)

    xml_str = ET.tostring(dest_root, "utf-8")
    xml_str = minidom.parseString(xml_str).toprettyxml(indent="    ")

    if debug_file_path:
        with open(debug_file_path, "w") as file:
            file.write(xml_str)

    return xml_str


def update_model_xml(
    model_xml: str,
    obj_to_xml_body_name: Dict[str, str],
    obj_to_transf_type_seq: Dict[str, List[str]],
    obj_to_transf_params_seq: Dict[str, Dict[str, np.ndarray]],
    remove_body_free_joint: bool = False,
    apply_all_transforms: bool = True,
    obj_to_original_joint_qpos: Optional[Dict[str, np.ndarray]] = None,
    set_obj_collision_free: bool = False,
    recenter_vertices: bool = False,
) -> Union[Tuple[str, np.ndarray], str]:
    """
    Args:
        remove_body_free_joint (bool): Whether to remove the free joint from all bodies being looped over.
        recenter_vertices (bool): Whether to recenter the vertices to be centered at the origin.
            Goal: decrease behavior where object squirms away.
    Currently hardcoded for SquareNut from NutAssemblyEnv.

    Uses the mesh paths from objs to update the model_xml.

    For each obj in objs:
        1. Add meshes that are not in the model_xml already
            e.g. inside the <asset> tag, add a new <mesh> tag for each mesh_path corresponding to obj
        2. Update the geometries in the model_xml
            e.g. inside the <body> tag, update the <geom> tags: remove the size tag and pos tag
            (unless the mesh doesn't 'contain' the pos/quat of the object) and ensure the
            "type" attribute is set to "mesh" and the "mesh" attribute is set to the name of the mesh
    """
    import xml.etree.ElementTree as ET

    # Parse the model XML using ElementTree
    root = ET.fromstring(model_xml)
    # save model_xml to a file
    # with open("model_xml.xml", "w") as f:
    #     f.write(model_xml)
    # go from model_xml to dict of mesh_name to mesh_path using xmltodict
    import xmltodict

    mesh_name_path_dict_lst = xmltodict.parse(model_xml)["mujoco"]["asset"]["mesh"]

    def convert_mujoco_xmltodict_list_to_dict(
        name_file_dict_lst: List[Dict[str, str]],
    ) -> Dict[str, str]:
        result = {}
        for item in name_file_dict_lst:
            name = item["@name"]
            file = item["@file"]
            result[name] = file
        return result

    mesh_name_to_mesh_path = convert_mujoco_xmltodict_list_to_dict(
        mesh_name_path_dict_lst
    )

    # Add meshes that are not in the model_xml already
    # e.g. inside the <asset> tag, add a new <mesh> tag for each obj in objs
    for obj, xml_body_name in obj_to_xml_body_name.items():
        # get the xml tag for the body with name SquareNut_main
        body = root.find(f".//body[@name='{xml_body_name}']")
        # convert that body to a string
        obj_xml_str = ET.tostring(body, encoding="unicode")
        output_dir = (
            pathlib.Path(demo_aug.__file__).parent / "models/assets/objects/meshes"
        )
        (
            overall_mesh_path,
            component_mesh_name_to_mesh_path,
            component_mesh_name_to_mass_properties,
            new_object_pos,
        ) = convert_xml_to_transformed_mesh(
            obj_xml_str,
            output_dir,
            obj_to_transf_type_seq[obj],
            obj_to_transf_params_seq[obj],
            apply_all_transforms=apply_all_transforms,
            original_obj_joint_qpos=(
                obj_to_original_joint_qpos.get(obj, None)
                if obj_to_original_joint_qpos is not None
                else None
            ),
            recenter_vertices=recenter_vertices,
            mesh_name_to_mesh_path=mesh_name_to_mesh_path,
        )

        # set the pos attribute from the body tag to 0 0 0
        body.set("pos", "0 0 0")

        # Add meshes to assets section
        # for i, mesh_path in enumerate(component_mesh_name_to_mesh_path):
        for i, (mesh_name, mesh_path) in enumerate(
            component_mesh_name_to_mesh_path.items()
        ):
            mesh_path = str(mesh_path)
            if mesh_path not in model_xml:
                # create a mesh with name obj
                mesh_element = ET.Element("mesh")
                mesh_element.set("name", mesh_name)
                mesh_element.set("file", mesh_path)
                # Locate the <asset> tag and append the new mesh
                asset = root.find("asset")
                if asset is not None:
                    # Formatting: determine current indentation (assumes the XML is properly indented)
                    current_indent = (asset.tail or "").count(" ")
                    mesh_element.tail = "\n" + " " * current_indent
                    asset.append(mesh_element)
            else:
                # overwrite the mesh path
                mesh = root.find(".//mesh[@name='{}']".format(mesh_name))
                mesh.set("file", mesh_path)

        for i, geom in enumerate(body.findall("geom")):
            # this geom_name needs to be consistent with name used in convert_xml_to_transformed_mesh
            geom_name = (
                geom.get("name") if geom.get("name") else body.get("name") + str(i)
            )
            # if geom_name is in objs, delete size + pos
            if "size" in geom.attrib:
                del geom.attrib["size"]
            if "pos" in geom.attrib:
                del geom.attrib["pos"]

            geom.set("type", "mesh")
            geom.set("mesh", geom_name)

            # Update inertial properties (if needed) #
            if body.find("inertial") is not None:
                body.remove(body.find("inertial"))

            if remove_body_free_joint and body.find("joint") is not None:
                body.remove(body.find("joint"))

            # then create a new inertial tag with attributes mass 0.001 and pos 0 0 0
            inertial = ET.Element("inertial")

            geom_mass_properties = component_mesh_name_to_mass_properties[geom_name]
            center_mass = geom_mass_properties["center_mass"]  # np array

            if set_obj_collision_free:
                geom.set("contype", "0")
                geom.set("conaffinity", "0")
                geom.set("group", "1")

            # inertial.set("mass", "0.0158068")
            # inertial.set("diaginertia", "2.91963e-5 1.87672e-5 1.14829e-5")
            # scale mass, diaginertia by 1000 because the default values from trimesh are too small
            inertial.set("mass", str(geom_mass_properties["mass"] * 1000))
            inertial.set("pos", " ".join([str(x) for x in center_mass]))
            diaginertia = np.diagonal(geom_mass_properties["inertia"]) * 1000
            inertial.set("diaginertia", " ".join([str(x) for x in diaginertia]))
            body.append(inertial)

    if recenter_vertices:
        return ET.tostring(root, encoding="unicode"), new_object_pos
    else:
        return ET.tostring(root, encoding="unicode")


def get_geom_bounding_box(geom_type, geom_size):
    """
    Compute AABB based on geometry type and size.
    """
    # if geom_type == mujoco_py.const.GEOM_BOX:  # type: ignore
    #     return -geom_size, geom_size
    # elif geom_type == mujoco_py.const.GEOM_SPHERE:  # type: ignore
    #     return -geom_size * np.ones(3), geom_size * np.ones(3)
    # elif geom_type == mujoco_py.const.GEOM_CYLINDER:  # type: ignore
    #     r, h = geom_size[0], geom_size[1]
    #     return np.array([-r, -r, -h / 2]), np.array([r, r, h / 2])
    # ... handle other geometry types as needed
    # else:
    #     raise NotImplementedError(
    #         f"Bounding box computation not implemented for geom_type {geom_type}"
    #     )
    return


def get_body_bounding_box(sim, body_name):
    """
    Get the bounding box (AABB) of a body given its name in terms of world coordinates.

    :param sim: MuJoCo simulator object.
    :param body_name: Name of the body.
    :return: (aabb_min, aabb_max) - min and max of the bounding box in world coordinates.
    """

    # Get body id from name
    body_id = sim.model.body_name2id(body_name)

    # Get body position (world coordinates)
    body_pos = sim.data.body_xpos[body_id].copy()

    # Get body rotation (quaternion)
    # body_quat = sim.data.body_xquat[body_id].copy()

    # Obtain the body's geometric properties
    geom_idx = np.where(sim.model.geom_bodyid == body_id)[0]

    # Initialize the bounding box
    aabb_min = np.array([float("inf")] * 3)
    aabb_max = np.array([-float("inf")] * 3)

    for idx in geom_idx:
        # Get geom type and size
        geom_type = sim.model.geom_type[idx].copy()
        geom_size = sim.model.geom_size[idx].copy()

        # # Get local AABB
        # local_aabb_min, local_aabb_max = sim.model.geom_aabb[idx].reshape(2, 3)
        # Get local AABB
        local_aabb_min, local_aabb_max = get_geom_bounding_box(geom_type, geom_size)
        # apply geom_pos
        local_aabb_min += sim.model.geom_pos[idx]
        local_aabb_max += sim.model.geom_pos[idx]

        # Get all 8 corners of the bounding box
        corners = np.array(
            [
                local_aabb_min,
                [local_aabb_max[0], local_aabb_min[1], local_aabb_min[2]],
                [local_aabb_max[0], local_aabb_max[1], local_aabb_min[2]],
                [local_aabb_min[0], local_aabb_max[1], local_aabb_min[2]],
                [local_aabb_min[0], local_aabb_min[1], local_aabb_max[2]],
                [local_aabb_max[0], local_aabb_min[1], local_aabb_max[2]],
                [local_aabb_max[0], local_aabb_max[1], local_aabb_max[2]],
                local_aabb_max,
            ]
        )

        # multiply by homogeous transformation matrix created by body_pos and body_quat
        body_transformation = np.eye(4)

        buffer = np.empty(9, dtype=np.float64)
        # mujoco_py.functions.mju_quat2Mat(buffer, body_quat)  # type: ignore
        rot_matrix = buffer.reshape(3, 3)
        body_transformation[:3, :3] = rot_matrix
        body_transformation[:3, 3] = body_pos
        body_transformation[3, 3] = 1

        corners = np.concatenate([corners, np.ones((8, 1))], axis=1)
        updated_corners = np.matmul(body_transformation, corners.T).T[:, :3]
        corners = updated_corners

        # update the aabb
        aabb_min = np.minimum(aabb_min, corners.min(axis=0))
        aabb_max = np.maximum(aabb_max, corners.max(axis=0))

    return aabb_min, aabb_max


def add_camera_to_xml(
    xml: str,
    camera_name: str,
    camera_pos: str,
    camera_quat: str,
    parent_body_name: str = "robot0_right_hand",
    fovy: str = "75",
    is_eye_in_hand_camera: bool = False,
    fx_pixel: Optional[float] = None,
    fy_pixel: Optional[float] = None,
    cx_pixel: Optional[float] = None,
    cy_pixel: Optional[float] = None,
) -> str:
    """
    Adds a new camera to the XML by attaching it to a body element,
    allowing for camera movement by manipulating the body.

    xml (str): Mujoco sim XML file as a string
    camera_name (str): Name of the new camera
    camera_pos (str): Position of the camera in the format "x y z"
    camera_quat (str): Quaternion rotation of the camera in the format "w x y z"

    Returns:
    str: Modified XML with the new camera
    """
    tree = ET.fromstring(xml)

    # Find the parent element to attach the camera body
    worldbody_elem = tree.find("worldbody")  # Modify this to match your XML structure
    assert worldbody_elem is not None, "No <worldbody> element found in the XML."

    if is_eye_in_hand_camera:
        # Create new camera elements
        new_camera = ET.Element(
            "camera"
        )  # , name=camera_name, pos=camera_pos, quat=camera_quat) #, fovy='75')

        # Find the body element to which you want to add cameras
        body_element = tree.find(f".//body[@name='{parent_body_name}']")

        new_camera.set("mode", "fixed")
        new_camera.set("name", camera_name)
        new_camera.set("pos", camera_pos)
        new_camera.set("quat", camera_quat)
        if (
            fx_pixel is not None
            and fy_pixel is not None
            and cx_pixel is not None
            and cy_pixel is not None
        ):
            print(
                "Warning: setting focalpixel and principalpixel for cameras is not supported properly yet."
            )
            import ipdb

            ipdb.set_trace()
            new_camera.set("focalpixel", f"{fx_pixel} {fy_pixel}")
            new_camera.set("principalpixel", f"{cx_pixel} {cy_pixel}")
        else:
            # add fovx
            new_camera.set("fovy", fovy)

        # Append the new camera elements to the body element
        body_element.append(new_camera)
    else:
        # Create the camera body element
        new_camera = ET.SubElement(worldbody_elem, "camera")
        new_camera.set("mode", "fixed")
        new_camera.set("name", camera_name)
        new_camera.set("pos", camera_pos)
        new_camera.set("quat", camera_quat)

    # Return the modified XML
    return ET.tostring(tree, encoding="utf8").decode("utf8")


def update_camera_in_xml(
    xml_str: str, camera_name: str, pos: List[float], quat: List[float]
) -> None:
    root = ET.fromstring(xml_str)

    # Find the camera element by name
    for camera in root.findall(".//camera"):
        if camera.get("name") == camera_name:
            # Update position
            camera.set("pos", " ".join(map(str, pos)))
            # Update quaternion
            camera.set("quat", " ".join(map(str, quat)))
            break

    # Return the updated XML string
    return ET.tostring(root, encoding="unicode")


def get_parent_map(tree: ET.Element):
    return {c: p for p in tree.iter() for c in p}


def update_xml_body_pos(xml: str, new_pos: List[float], body_name: str) -> str:
    assert (
        new_pos is not None
    ), "new_pos must be provided if _update_xml_body_pos is called"
    print("Patching xml model for real robot rendering ...")
    import xml.etree.ElementTree as ET

    # Parse the XML data
    tree = ET.ElementTree(ET.fromstring(xml))
    root = tree.getroot()

    # Find the body with name 'right_hand' and update its position
    for body in root.findall(f".//body[@name='{body_name}']"):
        body.set("pos", f"{new_pos[0]} {new_pos[1]} {new_pos[2]}")

    # Convert the tree back to a string and print it
    return ET.tostring(root, encoding="unicode")


def remove_body_from_xml(xml: str, body_name: str):
    """
    Intention: remove task relevant object when rendering
    However, ***Currently unused*** and instead teleport object away

    Removes a body from the XML by attaching it to a body element,
    allowing for camera movement by manipulating the body.
    """
    # Parse the XML file
    tree = ET.fromstring(xml)

    # Create a dictionary of child-parent relationships
    parent_map = get_parent_map(tree)

    # Find the body element by its name
    bodies = tree.findall(".//body")
    for body in bodies:
        if body.get("name") == body_name:
            # Remove the body element from its parent
            parent = parent_map[body]
            parent.remove(body)

    # Returns the modified XML
    return ET.tostring(tree, encoding="utf8").decode("utf8")


def remove_camera_from_xml(xml: str, camera_name: str):
    """
    Intention: remove camera when rendering
    """
    # Parse the XML file
    tree = ET.fromstring(xml)

    # Create a dictionary of child-parent relationships
    parent_map = get_parent_map(tree)

    # Find the body element by its name
    bodies = tree.findall(".//camera")
    # print(bodies)
    # import ipdb; ipdb.set_trace()
    for body in bodies:
        if body.get("name") == camera_name:
            # Remove the body element from its parent
            parent = parent_map[body]
            parent.remove(body)

    # Returns the modified XML
    return ET.tostring(tree, encoding="utf8").decode("utf8")


def remove_all_cameras_from_xml(xml: str):
    """
    Intention: remove camera when rendering
    """
    # Parse the XML file
    tree = ET.fromstring(xml)

    # Create a dictionary of child-parent relationships
    parent_map = get_parent_map(tree)

    # Find the body element by its name
    bodies = tree.findall(".//camera")
    for body in bodies:
        # Remove the body element from its parent
        parent = parent_map[body]
        parent.remove(body)

    # Returns the modified XML
    return ET.tostring(tree, encoding="utf8").decode("utf8")


# def create_passive_viewer(
#     model: MjModel,
#     data: MjData,
#     rate: float = 0.5,
#     run_physics: bool = False,
#     robot_configuration: Optional[Configuration] = None,
#     **kwargs: Any,
# ) -> None:
#     """
#     Create a passive viewer for the given MuJoCo model and data.
#     Args:
#         model (MjModel): The MuJoCo model object.
#         data (MjData): The MuJoCo data object.
#         rate (float): The update rate for the viewer (in seconds).
#         run_physics (bool): Whether to run physics in the background.
#         **kwargs: Additional keyword arguments.
#     """
#     with viewer.launch_passive(
#         model=model, data=data, show_left_ui=False, show_right_ui=False
#     ) as vis:
#         mj_fwdPosition(model, data)  # Initialize positions
#         qs = kwargs.get("qs", None)
#         i: int = 0
#         while vis.is_running():
#             if run_physics:
#                 mujoco.mj_step(model, data)
#             if robot_configuration is not None and qs is not None:
#                 if i == len(qs):
#                     break
#                 robot_configuration.update(qs[i])
#                 i += 1
#             vis.sync()
#             time.sleep(rate)
#         vis.close()


def render_image(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cam: str = "agentview",
    image_prefix: str = "eef_config",
):
    """
    Renders from camera cam and saves an image of the model.

    Args:
        model: MuJoCo model
        data: MuJoCo data
        cam: Camera name
        image_prefix: Prefix for the saved image filename

    Returns:
        str: Path to the saved image
    """
    mujoco.mj_forward(model, data)
    with mujoco.Renderer(model) as renderer:
        renderer.update_scene(data=data, camera=cam)
        frame = renderer.render()
        # Save the rendered frame to disk
        img_dir = pathlib.Path("debug_images")
        img_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        img_path = img_dir / f"{image_prefix}_{timestamp}.png"
        imageio.imsave(img_path, frame)
        logging.info(f"Saved image to {img_path}")
        print(f"Saved image to {img_path}")
