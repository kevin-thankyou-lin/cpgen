import pathlib
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import yaml
from nerfstudio.utils.rich_utils import CONSOLE
from tyro.extras import from_yaml, to_yaml

from demo_aug.envs.motion_planners.base_motion_planner import RobotEnvConfig
from demo_aug.objects.nerf_object import MeshPaths, TransformType
from demo_aug.utils.run_script_utils import retry_on_exception


@dataclass(frozen=True)
class TimestepData:
    """Data for a single timestep.

    In terms of hardware setup, assuming train/eval camera-to-EE
    transform is the same, so obs includes all the info that the policy might take in.

    Attributes:
        obs: observation
        action: action
        info (note there may be overlap between obs and pose)
        robot_pose: robot pose
        target_pose: target pose
        auxiliary_image_and_camera_cfg: auxiliary image and camera config
    """

    obs: Dict[str, np.ndarray]  # includes robot pose and RGB images
    action: np.ndarray  # perhaps default to absolute position as the `action`?
    # alternative action representations: EE pose delta w.r.t. world/base/curr-EE frame,
    action_alt_rep: Optional[Dict[str, np.ndarray]] = None
    mujoco_state: Optional[np.ndarray] = None  # used for storing robomimic states
    # the below is like the "info" field in a gym env
    # used to remove images of the robot when getting 3D model
    robot_pose: Optional[np.ndarray] = None
    # future K actions: src demonstrations don't have this option;
    # only augmented demos may have this option depending on the size of K
    future_actions: Optional[np.ndarray] = None
    # privileged data to skip learning the NeRF for now
    target_pose: Optional[np.ndarray] = None

    # this stuff is not exactly TimestepData;
    # should move them elsewhere
    robot_env_cfg: Optional[RobotEnvConfig] = None
    # this is more for replaying mujoco states
    mujoco_model_xml: Optional[str] = None

    objs_transf_type_seq: Optional[Dict[str, List[TransformType]]] = (
        None  # object transforms (X_se3's to apply to the mesh)
    )
    objs_transf_params_seq: Optional[Dict[str, List[Dict[str, np.ndarray]]]] = None
    objs_transf_name_seq: Optional[Dict[str, List[str]]] = None

    # TODO(klin): have 2 pipelines?
    # i) for sim verification of transforms and actions \
    # ii) for motion planning for for augmentations

    # cube pipeline i:
    # 1) apply se3 to mesh (this se3 is in the mujoco state?)
    # 2) apply scaling/shearing/warping to mesh --- need to store this info
    # 3) apply se3 to mesh again - also need to store this info; maybe the latter
    #    two infos can be stored in the same place


@dataclass
class TimestepAnnotationData:
    """Auxiliary annotation data corresponding to a single timestep.
    Contains nerf paths, sam3d masks, etc. nerfs should be lazily loaded.

    Separate from TimestepData to not accidentally overwrite the original data.
    Hmmm, maybe user decides which timesteps to generate these data from ... maybe don't want to
    a priori generate nerf ... maybe generate nerfs (default would be every 0.5s) according to the user's desires?
    """

    timestep: int

    task_relev_obj_poses: Optional[np.ndarray] = None  # for nerf3d == target_pose
    task_irrelev_objs_pose: Optional[np.ndarray] = None

    sam_points: Optional[np.ndarray] = None  # segment anything points
    sam_box: Optional[np.ndarray] = None
    sam_text: Optional[np.ndarray] = None
    sam_mask: Optional[np.ndarray] = None
    # perhaps best formalized post SAM3D?
    # use config to load trained NeRF: loading assumes a fixed file path
    # in terms of storing data, should be whatever SAM3D needs to get the masks?
    # sam3D masks
    # TODO(klin): these will be implementation dependent, for now these would mostly be none?
    task_relev_sam3d_mask: Optional[np.ndarray] = None
    task_relev_sam3d_mask_path: Optional[pathlib.Path] = None  # TODO(klin)

    # assume we can SAM things and NeRF separately if we know apriori that there are no transparent objects in the scene
    # for now, assume we don't SAM things and NeRF separately
    task_relev_obj_nerf_paths: Optional[Dict[str, pathlib.Path]] = None
    task_relev_obj_mesh_paths: Optional[Dict[str, str]] = None

    # TODO(klin): convert to 3D segmentation mask info
    task_relev_obj_nerf_bounding_boxes: Optional[Dict[str, np.ndarray]] = None

    task_irrelev_obj_paths: Optional[List[pathlib.Path]] = None
    task_irrelev_obj_poses: Optional[List[np.ndarray]] = None


@retry_on_exception(max_retries=5, retry_delay=1, exceptions=(BlockingIOError,))
def load_timestep_annotation_data(
    demo_path: str, demo_name: str
) -> Dict[int, TimestepAnnotationData]:
    assert demo_path.endswith(
        ".hdf5"
    ), f"demo_path must be an hdf5 file, got {demo_path}"

    timestep_annotation_data_dict: Dict[int, TimestepAnnotationData] = {}
    CONSOLE.log(f"Loading timestep annotation data from {demo_path}")
    with h5py.File(demo_path, "r") as f:
        # check if f"data/{demo_name}" exists
        if (
            demo_name not in f["data"].keys()
            or "timestep_annotation_data" not in f[f"data/{demo_name}"].keys()
        ):
            CONSOLE.log(
                f"No data found for demo {demo_name}; returning empty timestep_annotation_data_dict"
            )
            return timestep_annotation_data_dict

        # e.g. f['data/demo_0'] then do f['data/demo_0/constraint_infos']['constraint_info_i']
        # f['data/demo_0/timestep_annotation_data']['timestep_i']
        timestep_annotation_data_group = f[f"data/{demo_name}/timestep_annotation_data"]

        for timestep in timestep_annotation_data_group.keys():
            timestep_annotation_data_dict[int(timestep)] = from_yaml(
                TimestepAnnotationData, timestep_annotation_data_group[timestep][()]
            )
    CONSOLE.log(f"Loaded {len(timestep_annotation_data_dict)} timesteps")
    return timestep_annotation_data_dict


@retry_on_exception(max_retries=5, retry_delay=1, exceptions=(BlockingIOError,))
def load_image_ts_to_trained_nerf_path(
    demo_path: str, demo_name: str
) -> Dict[Tuple[int], str]:
    assert demo_path.endswith(
        ".hdf5"
    ), f"demo_path must be an hdf5 file, got {demo_path}"

    image_ts_to_trained_nerf_path: Dict[Tuple[int], str] = {}
    CONSOLE.log(f"Loading image_ts_to_trained_nerf_path from {demo_path}")
    with h5py.File(demo_path, "r") as f:
        # check if f"data/{demo_name}" exists
        if (
            demo_name not in f["data"].keys()
            or "image_ts_to_trained_nerf_path" not in f[f"data/{demo_name}"].keys()
        ):
            CONSOLE.log(
                f"No data found for demo {demo_name}; returning empty image_ts_to_trained_nerf_path"
            )
            return image_ts_to_trained_nerf_path

        # e.g. f['data/demo_0'] then do f['data/demo_0/constraint_infos']['constraint_info_i']
        # f['data/demo_0/timestep_annotation_data']['timestep_i']
        image_ts_to_trained_nerf_path_group = f[
            f"data/{demo_name}/image_ts_to_trained_nerf_path"
        ]

        for image_ts in image_ts_to_trained_nerf_path_group.keys():
            # eval() to convert from string to tuple; decode from bytes to string
            image_ts_to_trained_nerf_path[eval(image_ts)] = (
                image_ts_to_trained_nerf_path_group[image_ts][()].decode("utf-8")
            )
    CONSOLE.log(f"Loaded {len(image_ts_to_trained_nerf_path)} timesteps")
    return image_ts_to_trained_nerf_path


@retry_on_exception(max_retries=5, retry_delay=1, exceptions=(BlockingIOError,))
def load_image_ts_to_trained_gsplat_path(
    demo_path: str, demo_name: str
) -> Dict[Tuple[int], str]:
    assert demo_path.endswith(
        ".hdf5"
    ), f"demo_path must be an hdf5 file, got {demo_path}"

    image_ts_to_trained_gsplat_path: Dict[Tuple[int], str] = {}
    CONSOLE.log(f"Loading image_ts_to_trained_gsplat_path from {demo_path}")
    with h5py.File(demo_path, "r") as f:
        # check if f"data/{demo_name}" exists
        if (
            demo_name not in f["data"].keys()
            or "image_ts_to_trained_gsplat_path" not in f[f"data/{demo_name}"].keys()
        ):
            CONSOLE.log(
                f"No data found for demo {demo_name}; returning empty image_ts_to_trained_gsplat_path"
            )
            return image_ts_to_trained_gsplat_path

        # e.g. f['data/demo_0'] then do f['data/demo_0/constraint_infos']['constraint_info_i']
        # f['data/demo_0/timestep_annotation_data']['timestep_i']
        image_ts_to_trained_gsplat_path_group = f[
            f"data/{demo_name}/image_ts_to_trained_gsplat_path"
        ]

        for image_ts in image_ts_to_trained_gsplat_path_group.keys():
            # eval() to convert from string to tuple; decode from bytes to string
            image_ts_to_trained_gsplat_path[eval(image_ts)] = (
                image_ts_to_trained_gsplat_path_group[image_ts][()].decode("utf-8")
            )
    CONSOLE.log(f"Loaded {len(image_ts_to_trained_gsplat_path)} timesteps")
    return image_ts_to_trained_gsplat_path


@retry_on_exception(max_retries=5, retry_delay=1, exceptions=(BlockingIOError,))
def load_obj_t_to_mesh_path(
    demo_path: str, demo_name: str
) -> Dict[Tuple[str, int], str]:
    assert demo_path.endswith(
        ".hdf5"
    ), f"demo_path must be an hdf5 file, got {demo_path}"

    obj_t_to_mesh_path: Dict[Tuple[str, int], str] = {}
    CONSOLE.log(f"Loading obj_t_to_mesh_path from {demo_path}")
    with h5py.File(demo_path, "r") as f:
        # check if f"data/{demo_name}" exists
        if (
            demo_name not in f["data"].keys()
            or "obj_t_to_mesh_path" not in f[f"data/{demo_name}"].keys()
        ):
            CONSOLE.log(
                f"No data found for demo {demo_name}; returning empty obj_t_to_mesh_path"
            )
            return obj_t_to_mesh_path

        # e.g. f['data/demo_0'] then do f['data/demo_0/constraint_infos']['constraint_info_i']
        # f['data/demo_0/timestep_annotation_data']['timestep_i']
        obj_t_to_mesh_path_group = f[f"data/{demo_name}/obj_t_to_mesh_path"]

        for obj_t in obj_t_to_mesh_path_group.keys():
            # eval() to convert from string to tuple; decode from bytes to string
            obj_t_to_mesh_path[eval(obj_t)] = obj_t_to_mesh_path_group[obj_t][
                ()
            ].decode("utf-8")

    CONSOLE.log(f"Loaded {len(obj_t_to_mesh_path)} timesteps")
    return obj_t_to_mesh_path


@retry_on_exception(max_retries=5, retry_delay=1, exceptions=(BlockingIOError,))
def load_image_ts_obj_name_to_mesh_paths(
    demo_path: str, demo_name: str
) -> Dict[Tuple[Tuple[int], str], MeshPaths]:
    assert demo_path.endswith(
        ".hdf5"
    ), f"demo_path must be an hdf5 file, got {demo_path}"

    image_ts_obj_name_to_mesh_paths: Dict[Tuple[Tuple[int], str], str] = {}

    CONSOLE.log(f"Loading image_ts_obj_name_to_mesh_paths from {demo_path}")
    with h5py.File(demo_path, "r") as f:
        # check if f"data/{demo_name}" exists
        # if demo_name not in f["data"].keys() or "image_ts_obj_name_to_mesh_paths" not in f[f"data/{demo_name}"].keys():
        #     CONSOLE.log(f"No data found for demo {demo_name}; returning empty image_ts_obj_name_to_mesh_paths")
        #     return image_ts_obj_name_to_mesh_paths

        # e.g. f['data/demo_0'] then do f['data/demo_0/constraint_infos']['constraint_info_i']
        # f['data/demo_0/timestep_annotation_data']['timestep_i']
        # check if image_ts_obj_name_to_mesh_paths exists
        demo_group = f[f"data/{demo_name}"]
        if "image_ts_obj_name_to_mesh_paths" not in demo_group.keys():
            CONSOLE.log(
                f"No data found for demo {demo_name}; returning empty image_ts_obj_name_to_mesh_paths"
            )
            return image_ts_obj_name_to_mesh_paths
        image_ts_obj_name_to_mesh_paths_group = demo_group[
            "image_ts_obj_name_to_mesh_paths"
        ]

        for image_ts_obj_name in image_ts_obj_name_to_mesh_paths_group.keys():
            loaded = yaml.safe_load(
                image_ts_obj_name_to_mesh_paths_group[image_ts_obj_name][()].decode(
                    "utf-8"
                )
            )
            if isinstance(loaded, str):
                # no longer working with single str path for mesh; need to clear out these entries
                continue
            # eval() to convert from string to tuple; decode from bytes to string
            image_ts_obj_name_to_mesh_paths[eval(image_ts_obj_name)] = MeshPaths(
                **yaml.safe_load(
                    image_ts_obj_name_to_mesh_paths_group[image_ts_obj_name][()].decode(
                        "utf-8"
                    )
                )
            )

    CONSOLE.log(f"Loaded {len(image_ts_obj_name_to_mesh_paths)} timesteps")
    return image_ts_obj_name_to_mesh_paths


@retry_on_exception(max_retries=5, retry_delay=1, exceptions=(BlockingIOError,))
def load_image_ts_obj_name_to_bounding_box(
    demo_path: str, demo_name: str
) -> Dict[Tuple[Tuple[int], str], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    assert demo_path.endswith(
        ".hdf5"
    ), f"demo_path must be an hdf5 file, got {demo_path}"

    image_ts_obj_name_to_bounding_box: Dict[
        Tuple[int, str], Tuple[np.ndarray, np.ndarray, np.ndarray]
    ] = {}

    CONSOLE.log(f"Loading image_ts_obj_name_to_bounding_box from {demo_path}")
    with h5py.File(demo_path, "r") as f:
        # check if f"data/{demo_name}" exists
        if (
            demo_name not in f["data"].keys()
            or "image_ts_obj_name_to_bounding_box" not in f[f"data/{demo_name}"].keys()
        ):
            CONSOLE.log(
                f"No data found for demo {demo_name}; returning empty image_ts_obj_name_to_bounding_box"
            )
            return image_ts_obj_name_to_bounding_box

        # e.g. f['data/demo_0'] then do f['data/demo_0/constraint_infos']['constraint_info_i']
        # f['data/demo_0/timestep_annotation_data']['timestep_i']
        image_ts_obj_name_to_bounding_box_group = f[
            f"data/{demo_name}/image_ts_obj_name_to_bounding_box"
        ]

        for image_ts_obj_name in image_ts_obj_name_to_bounding_box_group.keys():
            # eval() to convert from string to tuple; decode from bytes to string
            image_ts_obj_name_to_bounding_box[eval(image_ts_obj_name)] = (
                image_ts_obj_name_to_bounding_box_group[image_ts_obj_name][()]
            )

    CONSOLE.log(f"Loaded {len(image_ts_obj_name_to_bounding_box)} timesteps")
    return image_ts_obj_name_to_bounding_box


@retry_on_exception(max_retries=5, retry_delay=1, exceptions=(BlockingIOError,))
def save_timestep_annotation_data(
    demo_path: str,
    demo_name: str,
    timestep_annotation_data_dict: Dict[int, TimestepAnnotationData],
) -> None:
    """
    Update original hdf5 file at 'name' to include timestep_annotation_data.

    Also, may want to create an offshoot (safer) of the original hdf5 file instead of modifying it directly (unsafe).
    """
    assert demo_path.endswith(
        ".hdf5"
    ), f"demo_path must be an hdf5 file, got {demo_path}"

    CONSOLE.log(f"Saving timestep annotation data to {demo_path}")
    with h5py.File(demo_path, "r+") as f:
        # check if f"data/{demo_name}" exists
        if demo_name not in f["data"].keys():
            CONSOLE.log(
                f"No data found for demo {demo_name}; not saving timestep annotation data."
            )
            return

        demo_group = f["data"][demo_name]

        # e.g. f['data/demo_0'] then do f['data/demo_0/constraint_infos']['constraint_info_i']
        # f['data/demo_0/timestep_annotation_data']['timestep_i']
        if "timestep_annotation_data" not in demo_group.keys():
            demo_group.create_group("timestep_annotation_data")

        timestep_annotation_data_group = f[f"data/{demo_name}/timestep_annotation_data"]
        for timestep, timestep_annotation_data in timestep_annotation_data_dict.items():
            if str(timestep) in timestep_annotation_data_group.keys():
                # ask user if they want to overwrite
                # overwrite = input(
                #     f"timestep_annotation_data timestep {timestep} already exists for demo {demo_name}."
                #     "Overwrite? (y/n): "
                # ).lower()
                overwrite = "y"
                if overwrite == "y":
                    del timestep_annotation_data_group[str(timestep)]
                    timestep_annotation_data_group[str(timestep)] = to_yaml(
                        timestep_annotation_data
                    )
                else:
                    continue
            else:
                timestep_annotation_data_group[str(timestep)] = to_yaml(
                    timestep_annotation_data
                )


@retry_on_exception(max_retries=5, retry_delay=1, exceptions=(BlockingIOError,))
def save_image_ts_to_trained_nerf_path(
    demo_path: str, demo_name: str, image_ts_to_trained_nerf_path: Dict[Tuple[int], str]
) -> None:
    """
    Update original hdf5 file at 'name' to include image_ts_to_trained_nerf_path.

    May want to create an offshoot (safer) of the original hdf5 file instead of modifying it directly (unsafe).
    """
    assert demo_path.endswith(
        ".hdf5"
    ), f"demo_path must be an hdf5 file, got {demo_path}"

    CONSOLE.log(f"Saving image_ts_to_trained_nerf_path to {demo_path}")
    with h5py.File(demo_path, "r+") as f:
        # check if f"data/{demo_name}" exists
        if demo_name not in f["data"].keys():
            CONSOLE.log(
                f"No data found for demo {demo_name}; not saving image_ts_to_trained_nerf_path."
            )
            return

        demo_group = f["data"][demo_name]

        # e.g. f['data/demo_0'] then do f['data/demo_0/constraint_infos']['constraint_info_i']
        # f['data/demo_0/timestep_annotation_data']['timestep_i']
        if "image_ts_to_trained_nerf_path" not in demo_group.keys():
            demo_group.create_group("image_ts_to_trained_nerf_path")

        image_ts_to_trained_nerf_path_group = f[
            f"data/{demo_name}/image_ts_to_trained_nerf_path"
        ]
        for image_ts, trained_nerf_path in image_ts_to_trained_nerf_path.items():
            if str(image_ts) in image_ts_to_trained_nerf_path_group.keys():
                # ask user if they want to overwrite
                # overwrite = input(
                #     f"image_ts_to_trained_nerf_path timestep {image_ts} already exists for demo {demo_name}."
                #     "Overwrite? (y/n): "
                # ).lower()
                overwrite = "y"
                if overwrite == "y":
                    del image_ts_to_trained_nerf_path_group[str(image_ts)]
                    image_ts_to_trained_nerf_path_group[str(image_ts)] = (
                        trained_nerf_path
                    )
                else:
                    continue
            else:
                image_ts_to_trained_nerf_path_group[str(image_ts)] = trained_nerf_path


# TODO: consolidate this function with the above nerf saving/loading function!
@retry_on_exception(max_retries=5, retry_delay=1, exceptions=(BlockingIOError,))
def save_image_ts_to_trained_gsplat_path(
    demo_path: str,
    demo_name: str,
    image_ts_to_trained_gsplat_path: Dict[Tuple[int], str],
) -> None:
    """
    Update original hdf5 file at 'name' to include image_ts_to_trained_gsplat_path.

    May want to create an offshoot (safer) of the original hdf5 file instead of modifying it directly (unsafe).
    """
    assert demo_path.endswith(
        ".hdf5"
    ), f"demo_path must be an hdf5 file, got {demo_path}"

    CONSOLE.log(f"Saving image_ts_to_trained_gsplat_path to {demo_path}")
    with h5py.File(demo_path, "r+") as f:
        # check if f"data/{demo_name}" exists
        if demo_name not in f["data"].keys():
            CONSOLE.log(
                f"No data found for demo {demo_name}; not saving image_ts_to_trained_gsplat_path."
            )
            return

        demo_group = f["data"][demo_name]

        # e.g. f['data/demo_0'] then do f['data/demo_0/constraint_infos']['constraint_info_i']
        # f['data/demo_0/timestep_annotation_data']['timestep_i']
        if "image_ts_to_trained_gsplat_path" not in demo_group.keys():
            demo_group.create_group("image_ts_to_trained_gsplat_path")

        image_ts_to_trained_gsplat_path_group = f[
            f"data/{demo_name}/image_ts_to_trained_gsplat_path"
        ]
        for image_ts, trained_gsplat_path in image_ts_to_trained_gsplat_path.items():
            if str(image_ts) in image_ts_to_trained_gsplat_path_group.keys():
                overwrite = "y"
                if overwrite == "y":
                    del image_ts_to_trained_gsplat_path_group[str(image_ts)]
                    image_ts_to_trained_gsplat_path_group[str(image_ts)] = (
                        trained_gsplat_path
                    )
                else:
                    continue
            else:
                image_ts_to_trained_gsplat_path_group[str(image_ts)] = (
                    trained_gsplat_path
                )


@retry_on_exception(max_retries=5, retry_delay=1, exceptions=(BlockingIOError,))
def save_obj_t_to_mesh_paths(
    demo_path: str, demo_name: str, obj_t_to_mesh_paths: Dict[Tuple[str, int], str]
) -> None:
    """
    Update original hdf5 file at 'name' to include obj_t_to_mesh_paths.

    May want to create an offshoot (safer) of the original hdf5 file instead of modifying it directly (unsafe).
    """
    assert demo_path.endswith(
        ".hdf5"
    ), f"demo_path must be an hdf5 file, got {demo_path}"

    CONSOLE.log(f"Saving obj_t_to_mesh_paths to {demo_path}")
    with h5py.File(demo_path, "r+") as f:
        # check if f"data/{demo_name}" exists
        if demo_name not in f["data"].keys():
            CONSOLE.log(
                f"No data found for demo {demo_name}; not saving obj_t_to_mesh_paths."
            )
            return

        demo_group = f["data"][demo_name]

        # e.g. f['data/demo_0'] then do f['data/demo_0/constraint_infos']['constraint_info_i']
        # f['data/demo_0/timestep_annotation_data']['timestep_i']
        if "obj_t_to_mesh_paths" not in demo_group.keys():
            demo_group.create_group("obj_t_to_mesh_paths")

        obj_t_to_mesh_paths_group = f[f"data/{demo_name}/obj_t_to_mesh_paths"]
        for obj_t, mesh_paths in obj_t_to_mesh_paths.items():
            if str(obj_t) in obj_t_to_mesh_paths_group.keys():
                # ask user if they want to overwrite
                # overwrite = input(
                #     f"obj_t_to_mesh_paths timestep {obj_t} already exists for demo {demo_name}."
                #     "Overwrite? (y/n): "
                # ).lower()
                overwrite = "y"
                if overwrite == "y":
                    del obj_t_to_mesh_paths_group[str(obj_t)]
                    obj_t_to_mesh_paths_group[str(obj_t)] = mesh_paths
                else:
                    continue
            else:
                obj_t_to_mesh_paths_group[str(obj_t)] = mesh_paths


@retry_on_exception(max_retries=5, retry_delay=1, exceptions=(BlockingIOError,))
def save_image_ts_obj_name_to_mesh_paths(
    demo_path: str,
    demo_name: str,
    image_ts_obj_name_to_mesh_paths: Dict[Tuple[str, str], MeshPaths],
) -> None:
    assert demo_path.endswith(
        ".hdf5"
    ), f"demo_path must be an hdf5 file, got {demo_path}"

    CONSOLE.log(f"Saving image_ts_obj_name_to_mesh_paths to {demo_path}")
    with h5py.File(demo_path, "r+") as f:
        # check if f"data/{demo_name}" exists
        if demo_name not in f["data"].keys():
            CONSOLE.log(
                f"No data found for demo {demo_name}; not saving image_ts_obj_name_to_mesh_paths."
            )
            return

        demo_group = f["data"][demo_name]

        # e.g. f['data/demo_0'] then do f['data/demo_0/constraint_infos']['constraint_info_i']
        # f['data/demo_0/timestep_annotation_data']['timestep_i']
        if "image_ts_obj_name_to_mesh_paths" not in demo_group.keys():
            demo_group.create_group("image_ts_obj_name_to_mesh_paths")

        image_ts_obj_name_to_mesh_paths_group = f[
            f"data/{demo_name}/image_ts_obj_name_to_mesh_paths"
        ]
        for image_ts_obj_name, mesh_paths in image_ts_obj_name_to_mesh_paths.items():
            if str(image_ts_obj_name) in image_ts_obj_name_to_mesh_paths_group.keys():
                # ask user if they want to overwrite
                # overwrite = input(
                #     f"image_ts_obj_name_to_mesh_paths timestep {image_ts_obj_name} already exists for demo {demo_name}."
                #     "Overwrite? (y/n): "
                # ).lower()
                overwrite = "y"
                if overwrite == "y":
                    del image_ts_obj_name_to_mesh_paths_group[str(image_ts_obj_name)]
                    image_ts_obj_name_to_mesh_paths_group[str(image_ts_obj_name)] = (
                        yaml.safe_dump(asdict(mesh_paths))
                    )
                else:
                    continue
            else:
                image_ts_obj_name_to_mesh_paths_group[str(image_ts_obj_name)] = (
                    yaml.safe_dump(asdict(mesh_paths))
                )


@retry_on_exception(max_retries=5, retry_delay=1, exceptions=(BlockingIOError,))
def save_image_ts_obj_name_to_bounding_box(
    demo_path: str,
    demo_name: str,
    image_ts_obj_name_to_bounding_box: Dict[Tuple[str, str], np.ndarray],
) -> None:
    assert demo_path.endswith(
        ".hdf5"
    ), f"demo_path must be an hdf5 file, got {demo_path}"

    CONSOLE.log(f"Saving image_ts_obj_name_to_bounding_box to {demo_path}")
    with h5py.File(demo_path, "r+") as f:
        # check if f"data/{demo_name}" exists
        if demo_name not in f["data"].keys():
            CONSOLE.log(
                f"No data found for demo {demo_name}; not saving image_ts_obj_name_to_bounding_box."
            )
            return

        demo_group = f["data"][demo_name]

        if "image_ts_obj_name_to_bounding_box" not in demo_group.keys():
            demo_group.create_group("image_ts_obj_name_to_bounding_box")

        image_ts_obj_name_to_bounding_box_group = f[
            f"data/{demo_name}/image_ts_obj_name_to_bounding_box"
        ]
        for (
            image_ts_obj_name,
            bounding_box,
        ) in image_ts_obj_name_to_bounding_box.items():
            if str(image_ts_obj_name) in image_ts_obj_name_to_bounding_box_group.keys():
                overwrite = "y"
                if overwrite == "y":
                    del image_ts_obj_name_to_bounding_box_group[str(image_ts_obj_name)]
                    image_ts_obj_name_to_bounding_box_group[str(image_ts_obj_name)] = (
                        bounding_box
                    )
                else:
                    continue
            else:
                image_ts_obj_name_to_bounding_box_group[str(image_ts_obj_name)] = (
                    bounding_box
                )
