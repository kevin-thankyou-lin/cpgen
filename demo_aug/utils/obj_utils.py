import argparse
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from pydrake.common import configure_logging as _configure_logging

from demo_aug.utils.drake_utils import MeshModelMaker, MultiMeshModelMaker


def get_renderable_obj_timestamps(
    obj_timestamps_and_image_timestamps_for_training: List[Tuple[int, Tuple[int]]],
    image_ts_to_trained_nerf_paths: Dict[Tuple[int, ...], str],
) -> List[int]:
    """Figures out which object timestamps we can render novel views for.

    1. Get a list of all unique object timestamps.
    2. For each object timestamp, check if all corresponding image timestamps have a trained NeRF path.
    3. If so, add the object timestamp to a list of renderable object timestamps.
    """

    # Step 1: Get a list of all unique object timestamps.
    unique_obj_timestamps = list(
        set([obj_ts for obj_ts, _ in obj_timestamps_and_image_timestamps_for_training])
    )

    # Step 3: Create a list to hold renderable object timestamps.
    renderable_obj_timestamps = []

    # Step 2: For each object timestamp, check if all corresponding image timestamps have a trained NeRF path.
    for obj_t in unique_obj_timestamps:
        all_image_ts_for_obj_t = [
            image_ts
            for o_t, image_ts in obj_timestamps_and_image_timestamps_for_training
            if o_t == obj_t
        ]

        # Check if all image timestamps for current object timestamp have a trained NeRF path.
        can_render = all(
            [
                image_ts in image_ts_to_trained_nerf_paths.keys()
                for image_ts in all_image_ts_for_obj_t
            ]
        )

        # If all image timestamps for current object timestamp have trained NeRF paths, add to the renderable list.
        if can_render:
            renderable_obj_timestamps.append(obj_t)

    return renderable_obj_timestamps


def subsample_timestamps_for_training_nerf(
    obj_timestamps_and_image_timestamps_for_training: List[Tuple[int, Tuple[int]]],
    image_ts_to_trained_nerf_paths: Dict[Tuple[int, ...], str],
    K: int,
) -> List[Tuple[int, Tuple[int]]]:
    """
    Subsamples obj_timestamps_and_image_timestamps_for_training based on already
    trained image timestamps and K interval.

    Parameters:
    - obj_timestamps_and_image_timestamps_for_training: A list of tuples,
        where each tuple contains an object timestamp and a list of corresponding image timestamps for training.
    - image_ts_to_trained_nerf_paths: A dictionary mapping tuples of image
        timestamps to their corresponding trained NeRF paths.
    - K: Interval to ensure that there's at least one timestamp with all objects having a trained NeRF.

    Returns:
    - Subsampled list containing only the combinations needed for training.
    - List of object timestamps for which we will be able to do NeRF rendering for

    TODO(klin): handle case where there might be a timestamp where one object has a trained NeRF but another object
    doesn't. However, we skip this timestamp because it's not >= last_trained_time + K. This case could
    cause an issue on downstream NeRF rendering generation.

    We should ensure that we use a timestamp completely or not at all.
    """
    # Step 1: Convert the list to a dict mapping obj_timestamp to a list of image_timestamps.
    obj_to_image_ts: Dict[int, List[Tuple[int]]] = {}
    for (
        obj_timestamp,
        image_timestamps,
    ) in obj_timestamps_and_image_timestamps_for_training:
        if obj_timestamp not in obj_to_image_ts:
            obj_to_image_ts[obj_timestamp] = []
        obj_to_image_ts[obj_timestamp].append(image_timestamps)

    # Step 2: Get a list of only obj_timestamps.
    unique_obj_timestamps = list(obj_to_image_ts.keys())

    last_trained_time = -np.inf
    subsampled_list = []

    # Step 3: Do the subsampling using the list of only obj_timestamps.
    for obj_timestamp in unique_obj_timestamps:
        image_ts_list = obj_to_image_ts[obj_timestamp]

        # If obj_timestamp is greater than or equal to last_trained_time + K, ensure all image_ts that don't have
        # a trained nerf path to subsampled_list
        if obj_timestamp >= last_trained_time + K:
            for image_ts in image_ts_list:
                # only add if image_ts not in image_ts_to_trained_nerf_paths
                if image_ts not in image_ts_to_trained_nerf_paths.keys():
                    subsampled_list.append((obj_timestamp, image_ts))
                    # design choice below:
                    # only update last_trained_time if we actually added something?
                    last_trained_time = obj_timestamp
                    image_ts_to_trained_nerf_paths[image_ts] = None

            # design choice below: or update last_trained_time regardless?
            # last_trained_time = obj_timestamp

    return subsampled_list


def extract_model_name_from_sdf(sdf_file_path) -> str:
    """Parses an SDF file and extracts the model name.

    Args:
    - sdf_file_path (str): Path to the SDF file to be parsed.

    Returns:
    - str: The name of the model.
    """
    assert (
        Path(sdf_file_path).suffix == ".sdf"
    ), f"File {sdf_file_path} is not an SDF file."
    assert Path(sdf_file_path).exists(), f"SDF file {sdf_file_path} does not exist."

    tree = ET.parse(sdf_file_path)
    root = tree.getroot()

    # Find the model name
    model_name = root.find(".//model").attrib["name"]

    return model_name


def extract_mesh_paths_from_sdf(sdf_file_path: str) -> Tuple[List[str], List[str]]:
    """Parses an SDF file and extracts visual and collision mesh paths.

    Args:
    - sdf_file_path (str): Path to the SDF file to be parsed.

    Returns:
    - tuple: A tuple containing two lists:
        1. List of visual mesh URIs.
        2. List of collision mesh URIs.
    """
    tree = ET.parse(sdf_file_path)
    root = tree.getroot()

    # Find all visual mesh URIs
    visual_mesh_uris = [
        elem.text for elem in root.findall(".//visual//geometry//mesh//uri")
    ]

    # Find all collision mesh URIs
    collision_mesh_uris = [
        elem.text for elem in root.findall(".//collision//geometry//mesh//uri")
    ]

    # concatenate each elem with the parent folder of sdf_file_path
    # assumption = .obj files are in the same folder as the sdf file
    visual_mesh_paths = [
        str(Path(sdf_file_path).parent / Path(uri)) for uri in visual_mesh_uris
    ]
    collision_mesh_paths = [
        str(Path(sdf_file_path).parent / Path(uri)) for uri in collision_mesh_uris
    ]

    # check these mesh paths actually exist
    for mesh_path in visual_mesh_paths + collision_mesh_paths:
        assert Path(mesh_path).exists(), (
            f"Mesh path {mesh_path} does not exist."
            "Please ensure that the .obj files are in the same folder as the SDF file"
            "or use drake's package parser to get the correct full path of the meshes."
        )

    return (visual_mesh_paths, collision_mesh_paths)


_logger = logging.getLogger("drake")


def _CommaSeparatedXYZ(arg: str) -> np.ndarray:
    """An argparse parser for an x,y,z vector."""
    x, y, z = [float(item) for item in arg.split(",")]
    return np.array([x, y, z])


# In the event that the user passes in a malformed value for --body-origin,
# this makes the error message nicer.
_CommaSeparatedXYZ.__name__ = "comma-separated triple"


# Note: Modify _SDF_TEMPLATE to use placeholders {visual} and {collision} to
# insert the multiple visual and collision geometry XML strings, respectively.


def convert_mesh_to_model(
    # mesh_path: Path,
    mesh_paths: List[Path],
    scale: float = 1,
    density: float = None,
    mass: float = None,
    at_com: bool = False,
    body_origin: np.ndarray = None,
    encoded_package: str = None,
    output_dir: Path = None,
    filename_prefix: str = None,
) -> str:
    """Converts a .obj mesh file to a drake-compatible SDFormat file

    See pydrake.multibody.mesh_to_model for more information.

    Drake uses meters as the unit of length. Modelling packages may frequently
    use other units (e.g. centimeters). This program will *assume* the units
    are meters. If they are not, provide the scale conversion factor to go from
    mesh units to meters. For example, if your mesh is in centimeters, use 0.01
    for the scale.

    If output_dir is not given, the file will be written in the mesh's directory.

    Returns the path to the generated model file.
    """
    _configure_logging()

    default_maker = MeshModelMaker(mesh_path=None, output_dir=None)

    args = argparse.Namespace(
        model_name=None,  # uses mesh_path's filename if None
        scale=scale if scale else default_maker.scale,
        density=density if density else default_maker.density,
        mass=mass,
        at_com=at_com,
        p_GoBo=body_origin,
        encoded_package=encoded_package
        if encoded_package
        else default_maker.encoded_package,
        output_dir=output_dir,
        mesh_path=mesh_paths[0],
    )

    maker = MultiMeshModelMaker(
        visual_mesh_paths=mesh_paths, collision_mesh_paths=mesh_paths, **vars(args)
    )
    output_path = maker.make_model(filename_prefix)

    return output_path


if __name__ == "__main__":
    # Example usage
    mesh_path = Path("path/to/my_models/mesh.obj")
    scale = 0.01
    density = 1500  # kg/m³
    output_dir = Path("output_directory/")

    convert_mesh_to_model(mesh_path, scale, density=density, output_dir=output_dir)
