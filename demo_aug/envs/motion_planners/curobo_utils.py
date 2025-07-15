import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import mujoco
import numpy as np
import robosuite.utils.transform_utils as T
from curobo.geom.types import Capsule, Cuboid, Cylinder, Mesh, Sphere, WorldConfig
from curobo.types.math import Pose
from curobo.wrap.reacher.motion_gen import MotionGen


@dataclass
class MjGeom:
    name: str
    type: str  # Sphere, Cuboid, Cylinder, etc.
    dimensions: List[float]
    pose_xyz_wxyz: np.ndarray
    file_path: Optional[str] = None  # Only for mesh

    # Additional properties
    contype: int = 0
    conaffinity: int = 0
    group: int = 0

    # Body, geom and mesh ID information
    body_name: Optional[str] = None
    body_id: Optional[int] = None
    mesh_id: Optional[int] = None
    geom_id: Optional[int] = None


def _extract_mj_names(
    mjmodel: mujoco.MjModel, num_obj: int, obj_type
) -> Tuple[Tuple[str], Dict[str, int], Dict[int, str]]:
    """
    Extracts names and IDs from MuJoCo model, assigning a unique name if name is None.

    Adapted from robosuite.
    """
    id2name = {}
    name2id = {}
    for i in range(num_obj):
        name = mujoco.mj_id2name(mjmodel, obj_type, i)

        # Generate a unique name if the object is unnamed
        if not name:
            # Create a unique name that combines object type and ID, avoiding clashes
            name = f"unnamed_{str(obj_type).replace('.', '_')}_{i}"
            while name in name2id:
                name = f"unnamed_{str(obj_type).replace('.', '_')}_{i}_{random.randint(1000, 9999)}"  # Ensures unique suffix if necessary

        name2id[name] = i
        id2name[i] = name

    # Sort by increasing ID for deterministic order and return as tuple and dictionaries
    return tuple(id2name[nid] for nid in sorted(id2name.keys())), name2id, id2name


def get_path_from_pathsadr(model: mujoco.MjModel, pathsadr: int) -> str:
    """Extract individual path from model.paths binary string."""
    paths_bytes = model.paths
    # Find the null terminator for this path
    end_idx = paths_bytes.find(b"\x00", pathsadr)
    if end_idx == -1:
        end_idx = len(paths_bytes)
    # Extract and decode the path
    path = paths_bytes[pathsadr:end_idx].decode("utf-8")
    return path


def mjmodel_to_mjgeoms(
    mjmodel: mujoco.MjModel,
    exclude_geom_prefixes: List[str] = ["robot", "gripper"],
    verbose: bool = False,
) -> List[MjGeom]:
    """Convert MjModel geoms to MjGeom list for intermediate representation."""
    _, _, geom_id2name = _extract_mj_names(
        mjmodel, mjmodel.ngeom, mujoco.mjtObj.mjOBJ_GEOM
    )
    _, _, body_id2name = _extract_mj_names(
        mjmodel, mjmodel.nbody, mujoco.mjtObj.mjOBJ_BODY
    )

    mj_geoms = []
    for geom_id in range(mjmodel.ngeom):
        geom_name = geom_id2name.get(geom_id)
        geom_type = mjmodel.geom_type[geom_id]

        # Set dimensions based on geom type
        if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
            dimensions = [mjmodel.geom_size[geom_id][0]]  # Radius
        elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
            dimensions = [
                mjmodel.geom_size[geom_id][0],
                mjmodel.geom_size[geom_id][1],
            ]  # Radius, Height
        elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
            dimensions = [
                mjmodel.geom_size[geom_id][0],
                mjmodel.geom_size[geom_id][1],
            ]  # Radius, Height
        elif geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            dimensions = mjmodel.geom_size[
                geom_id
            ].tolist()  # x_length, y_length, z_length
        elif geom_type == mujoco.mjtGeom.mjGEOM_MESH:
            mesh_id = mjmodel.geom_dataid[geom_id]
            dimensions = mjmodel.mesh_scale[
                mesh_id
            ]  # mujoco is using m, curobo is using cm
            file_path = get_path_from_pathsadr(mjmodel, mjmodel.mesh_pathadr[mesh_id])
        else:
            continue

        body_name = body_id2name.get(mjmodel.geom_bodyid[geom_id], None)
        geom_group = mjmodel.geom_group[geom_id]

        if geom_group != 0:
            if verbose:
                print(f"Skipping {geom_name}: not in the collision group.")
            continue
        elif geom_group == 1:
            if verbose:
                print(f"Skipping {geom_name}: visual geom.")
            continue
        elif (geom_name is not None) and any(
            geom_name.startswith(geom_prefix) for geom_prefix in exclude_geom_prefixes
        ):
            if verbose:
                print(
                    f"Skipping {geom_name}: part of exclude_geom_prefixes {exclude_geom_prefixes}."
                )
            continue

        # Create the MjGeom instance
        mj_geom = MjGeom(
            name=geom_name,
            type=geom_type,
            geom_id=geom_id,
            dimensions=dimensions,
            pose_xyz_wxyz=np.concatenate(
                [mjmodel.geom_pos[geom_id], mjmodel.geom_quat[geom_id]]
            ),
            file_path=file_path if geom_type == mujoco.mjtGeom.mjGEOM_MESH else None,
            contype=mjmodel.geom_contype[geom_id],
            conaffinity=mjmodel.geom_conaffinity[geom_id],
            group=geom_group,
            body_name=body_name,
            body_id=mjmodel.geom_bodyid[geom_id],
            mesh_id=mesh_id if geom_type == mujoco.mjtGeom.mjGEOM_MESH else None,
        )
        mj_geoms.append(mj_geom)

    return mj_geoms


def get_geom_poses(
    mj_geoms: List[MjGeom], data: mujoco.MjData, base_pos: np.ndarray
) -> Dict[str, Dict[str, Any]]:
    """
    Computes world-frame poses for each geom in `mj_geoms` using `data`.

    TODO: maybe let this method return world state poses instead of manually updating to body frame.
    """
    geom_name_to_pose = {}
    for mj_geom in mj_geoms:
        geom_id = mj_geom.geom_id  # Assume geom_id can be taken from mesh_id
        # Get geom pose in world frame
        geom_pos = np.array(data.geom_xpos[geom_id])
        geom_rot = np.array(data.geom_xmat[geom_id].reshape((3, 3)))
        geom_pose = T.make_pose(geom_pos, geom_rot)
        if mj_geom.type == mujoco.mjtGeom.mjGEOM_MESH:
            offset_pose = T.make_pose(
                mj_geom.pose_xyz_wxyz[:3],
                T.quat2mat(np.roll(np.array(mj_geom.pose_xyz_wxyz[3:]), -1)),
            )
        else:
            offset_pose = np.eye(4)

        geom_pose = geom_pose @ np.linalg.inv(offset_pose)
        # convert to pose xyz_wxyz
        pose_xyz_wxyz = np.concatenate(
            [geom_pose[:3, 3], np.roll(T.mat2quat(geom_pose[:3, :3]), 1)]
        )
        pose_xyz_wxyz[:3] = pose_xyz_wxyz[:3] - base_pos

        geom_name_to_pose[mj_geom.name] = {
            "geom_id": geom_id,
            "geom_pose": pose_xyz_wxyz,
        }
    return geom_name_to_pose


def create_curobo_world_config(
    mj_geoms: List[MjGeom],
    geom_name_to_pose: Dict[str, np.ndarray],
    exclude_geom_prefixes=("robot", "gripper"),
    verbose: bool = False,
) -> WorldConfig:
    """
    Creates curobo world representation from mujoco world. Takes in a list of `MjGeom`
    objects and a dictionary of poses, where keys are geom names and values are poses in world frame.
    """
    cuboid_obstacles = []
    capsule_obstacles = []
    cylinder_obstacles = []
    sphere_obstacles = []
    mesh_obstacles = []

    for geom in mj_geoms:
        geom_group = geom.group
        geom_name = geom.name

        if geom_group != 0:
            if verbose:
                print(f"Skipping {geom_name}: not in the collision group.")
            continue
        elif geom_group == 1:
            if verbose:
                print(f"Skipping {geom_name}: visual geom.")
            continue
        elif (geom_name is not None) and any(
            geom_name.startswith(geom_prefix) for geom_prefix in exclude_geom_prefixes
        ):
            if verbose:
                print(
                    f"Skipping {geom_name}: part of exclude_geom_prefixes {exclude_geom_prefixes}."
                )
            continue

        geom_name = geom.name
        # Retrieve pose and color for the obstacle
        geom_pose_xyz_wxyz = geom_name_to_pose[geom_name]["geom_pose"]
        if geom.type == mujoco.mjtGeom.mjGEOM_BOX:
            obstacle = Cuboid(
                name=geom_name,
                pose=geom_pose_xyz_wxyz,
                # mujoco uses half-extents
                dims=[2 * dim for dim in geom.dimensions],
            )
            cuboid_obstacles.append(obstacle)
        elif geom.type == mujoco.mjtGeom.mjGEOM_CAPSULE:
            obstacle = Capsule(
                name=geom_name,
                pose=geom_pose_xyz_wxyz,
                radius=geom.dimensions[0],
                base=[0, 0, 0],
                # mujoco uses half-extents
                tip=[0, 0, 2 * geom.dimensions[1]],
            )
            capsule_obstacles.append(obstacle)
        elif geom.type == mujoco.mjtGeom.mjGEOM_CYLINDER:
            obstacle = Cylinder(
                name=geom_name,
                pose=geom_pose_xyz_wxyz,
                radius=geom.dimensions[0],
                # mujoco uses half-extents
                height=(2 * geom.dimensions[1]),
            )
            cylinder_obstacles.append(obstacle)
        elif geom.type == mujoco.mjtGeom.mjGEOM_SPHERE:
            obstacle = Sphere(
                name=geom_name,
                pose=geom_pose_xyz_wxyz,
                radius=geom.dimensions[0],
            )
            sphere_obstacles.append(obstacle)
        elif geom.type == mujoco.mjtGeom.mjGEOM_MESH:
            if not geom.file_path:
                raise ValueError(f"Mesh file path for geom '{geom_name}' is missing.")
            obstacle = Mesh(
                name=geom_name,
                pose=geom_pose_xyz_wxyz,
                file_path=geom.file_path,
                scale=geom.dimensions,  # Assuming `dimensions` holds scale for meshes
            )
            mesh_obstacles.append(obstacle)
        else:
            raise ValueError(f"Invalid geom type '{geom.type}' for geom '{geom_name}'")

    # Construct Curobo world model from the obstacles
    world_model = WorldConfig(
        mesh=mesh_obstacles,
        cuboid=cuboid_obstacles,
        capsule=capsule_obstacles,
        cylinder=cylinder_obstacles,
        sphere=sphere_obstacles,
    )

    # Optionally add collision support (if needed for Curobo)
    world_model = WorldConfig.create_collision_support_world(world_model)

    return world_model


def update_curobo_world_config(
    motion_gen: MotionGen,
    mj_geoms: List[MjGeom],
    geom_name_to_pose: Dict[str, np.ndarray],
):
    """
    Updates curobo world (in MotionGen object) using the latest geom pose
    information from mujoco.
    """
    for mj_geom in mj_geoms:
        geom_name = mj_geom.name
        geom_pose = geom_name_to_pose[geom_name]["geom_pose"]

        # update internal representation for collision checking on GPU
        motion_gen.world_coll_checker.update_obstacle_pose(
            name=geom_name,
            w_obj_pose=Pose.from_list(geom_pose),
            update_cpu_reference=True,
        )
        motion_gen.graph_planner.reset_buffer()


def main():
    model = mujoco.MjModel.from_xml_path("new-model.xml")
    mj_data = mujoco.MjData(model)

    # forward the model
    mujoco.mj_forward(model, mj_data)

    mj_geoms = mjmodel_to_mjgeoms(model)
    geom_name_to_pose = get_geom_poses(mj_geoms, mj_data)

    # create curobo world config
    world_model = create_curobo_world_config(mj_geoms, geom_name_to_pose)

    world_model.save_world_as_mesh("world_mesh.obj")

    import trimesh

    # Load the mesh from the OBJ file
    mesh = trimesh.load_mesh("world_mesh.obj")

    mesh.show()


if __name__ == "__main__":
    main()
