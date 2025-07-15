from __future__ import annotations

import copy
import datetime
import gc
import json
import pathlib
import random
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import point_cloud_utils as pcu
import torch
import trimesh
import yaml
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.configs.dataparser_configs import dataparsers

# from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.configs.method_configs import TrainerConfig, all_methods, method_configs
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.model_components.renderers import BackgroundColor
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.scripts.exporter import ExportPoissonMesh
from nerfstudio.scripts.train import main as train_scene_representation
from nerfstudio.utils.eval_utils import eval_load_checkpoint
from scipy.spatial.transform import Rotation as R

import demo_aug
from demo_aug.objects.nerfstudio.nerf_to_mesh import load_nerf_field, nerf_to_mesh
from demo_aug.objects.nerfstudio.nerfstudio_field_utils import (
    get_density_nerfacto_field,
    get_density_tensorf_field,
    get_outputs_nerfacto_field,
    get_outputs_tensorf_field,
)
from demo_aug.objects.nerfstudio.splatfacto_utils import (
    get_outputs as get_outputs_splatfacto,
)
from demo_aug.utils.obj_utils import convert_mesh_to_model, extract_mesh_paths_from_sdf

# TODO(klin): might want to move this to a more general location or move scripts/ into demo_aug/
from scripts.convex_decomp import Config as ConvexDecompConfig
from scripts.convex_decomp import main as convex_decomp


@dataclass
class MeshPaths:
    sdf_path: Optional[str] = None
    obj_path: Optional[str] = None


# at some point, should be able to view things in the browser
# for now, just assume we want direct control over a few parameters
def generate_mesh(
    nerf_config_path: pathlib.Path,
    output_dir: pathlib.Path,
    num_points: int = 10000,
    bounding_box_min: Tuple[float, float, float] = (-1, -1, -1),
    bounding_box_max: Tuple[float, float, float] = (1, 1, 1),
    obb_center: Optional[Tuple[float, float, float]] = None,
    obb_rotation: Optional[Tuple[float, float, float]] = None,
    obb_scale: Optional[Tuple[float, float, float]] = None,
    remove_outliers: bool = True,
    target_num_faces: int = 2500,
    num_pixels_per_side: int = 400,
    mesh_name: str = "unnamed",
    save_point_cloud: bool = True,
    mesh_gen_method: Literal["level-set", "poisson"] = "level-set",
) -> MeshPaths:
    """
    1. Generate a .obj mesh from a nerf object
    2. Convert mesh to water tight mesh
    3. Compute convex decomp
    4. Converts .obj to .sdf format
    5. Return the .sdf path.
    """
    if mesh_gen_method == "level-set":
        obb_center = obb_center
        obb_rotation = obb_rotation
        obb_scale = obb_scale

        if obb_center is not None:
            lb = obb_center - obb_scale * 0.5
            ub = obb_center + obb_scale * 0.5

            print(f"lb: {lb}, ub: {ub}")

        if "holder" in str(nerf_config_path):
            level = 15  # important hyperparameter to adjust!
            npts = 60
        elif "glass" in str(nerf_config_path):
            level = 15  # important hyperparameter to adjust!
            npts = 31
        else:
            level = 5
            npts = 31

        field = load_nerf_field(nerf_config_path)
        print(f"water tight mesh saved to {output_dir / 'watertight-mesh.obj'}")
        nerf_to_mesh(
            field,
            level=level,
            npts=npts,
            lb=bounding_box_min,
            ub=bounding_box_max,
            save_path=str(output_dir / "watertight-mesh.obj"),
        )
    else:
        print(
            "NOTE: if mesh generation takes a long time, it likely means the NeRF was bad."
        )
        exporter = ExportPoissonMesh(
            load_config=nerf_config_path,
            output_dir=output_dir,
            num_points=num_points,
            bounding_box_min=bounding_box_min,
            bounding_box_max=bounding_box_max,
            obb_center=obb_center,
            obb_rotation=obb_rotation,
            obb_scale=obb_scale,
            remove_outliers=remove_outliers,
            target_num_faces=target_num_faces,
            num_pixels_per_side=num_pixels_per_side,
            save_point_cloud=save_point_cloud,
        )

        exporter.main()
        exporter.pipeline = None
        del exporter
        gc.collect()
        torch.cuda.empty_cache()

        # convert mesh to watertight mesh
        v, f, c = pcu.load_mesh_vfc(output_dir / "mesh.obj")
        print(f"Mesh saved to {output_dir / 'mesh.obj'}")
        vw, fw = pcu.make_mesh_watertight(v, f)
        pcu.save_mesh_vf(str(output_dir / "watertight-mesh.obj"), vw, fw)

    # compute convex decomposition
    suffix_to_decomped_mesh_paths: Dict[str, List[pathlib.Path]] = convex_decomp(
        ConvexDecompConfig(
            input_file=str(output_dir / "watertight-mesh.obj"),
            output_dir=output_dir,
            output_file_prefix=f"{mesh_name}-watertight-mesh-convex-decomp",
            output_file_suffixes=[".obj", ".stl"],
        )
    )

    # convert mesh to sdf
    mesh_sdf_path: str = convert_mesh_to_model(
        mesh_paths=suffix_to_decomped_mesh_paths[".obj"],
        output_dir=output_dir,
    )

    return MeshPaths(
        sdf_path=mesh_sdf_path, obj_path=str(output_dir / "watertight-mesh.obj")
    )


def get_nerf_folder_path(
    image_ts_for_training: List[int],
    t_to_nerf_folder_path: Dict[int, str],
) -> str:
    """
    If len(image_ts_for_training) == 1, then returns the path to the nerf folder.
    Otherwise, merge the information from the relevant nerf folders into a new folder.

    nerf_folder structure:

    folder_name_1/
        images/
            000000.png
            000001.png
            ...
        transforms.json
    folder_name_2/
        images/
            000000.png
            000001.png
            ...
        transforms.json
    ...

    transforms.json looks as follows:

    {
        "frames": [
            {
            "file_path": "images/000000.png",
            "depth_file_path": "depth/000000.exr",
            "mask_path": "masks/mask.jpeg",
            "transform_matrix": ...,
            "h": ...,
            "w": ...,
            ...
            }
        ]
    }

    Create new folder with name the concatenation of image_ts_for_training.

    Merge all the transforms.json files into one file. Do not merge the images folders.
    Instead, update the paths in the merged transforms.json file to point to the correct image paths.

    TODO(klin): ensure all data in nerf folders have a consistent format.
    """
    # convert image_ts to strings
    image_ts_for_training = list(map(str, image_ts_for_training))

    if len(image_ts_for_training) == 1:
        return str(t_to_nerf_folder_path[image_ts_for_training[0]])

    # Parent folder determination based on the first path
    parent_folder = Path(t_to_nerf_folder_path[image_ts_for_training[0]]).parent

    merged_frames = []

    for image_t in image_ts_for_training:
        nerf_folder_path = Path(t_to_nerf_folder_path[image_t])
        transforms_path = nerf_folder_path / "transforms.json"

        with transforms_path.open("r") as f:
            transforms_data = json.load(f)

            # Update the paths in each frame
            for frame in transforms_data["frames"]:
                for key, value in frame.items():
                    if key.endswith("_path"):
                        # provide absolute path; alternatively, could provide relative path
                        # by using the relative locations of the source and new folders
                        frame[key] = str(
                            pathlib.Path(demo_aug.__file__).parent.parent
                            / nerf_folder_path
                            / value
                        )

            merged_frames.extend(transforms_data["frames"])

    # Generate the new nerf folder path
    new_nerf_folder_name = "_".join(map(str, image_ts_for_training))
    new_nerf_folder_path = parent_folder / new_nerf_folder_name

    # Ensure the new folder exists
    new_nerf_folder_path.mkdir(parents=True, exist_ok=True)

    # Save the merged transforms.json to the new folder
    merged_transforms_path = new_nerf_folder_path / "transforms.json"
    with merged_transforms_path.open("w") as f:
        json.dump({"frames": merged_frames}, f, indent=4)

    # TODO(klin): if there are multiple timestamps, check images have EEs masked out
    return str(new_nerf_folder_path)


# TODO(klin): look into merging the generate nerf and generate gsplat method:
# if it's nerfacto do something, otherwise, do something else?
def generate_scene_representation(
    data_dir: pathlib.Path,
    output_dir: pathlib.Path,
    method: Literal["nerfacto", "splatfacto"] = "nerfacto",
    dataparser: str = "blender-data",
    near_plane: float = 0.01,
    far_plane: float = 3.0,
    max_num_iterations: int = 5000,
    steps_per_save: int = 2500,
    vis: str = "wandb",
    camera_optimizer_mode: str = "off",
    splatfacto_init_scale: float = 4.0,
    splatfacto_stop_split_at: int = 8000,
    random_scale: float = 3.0,
    steps_per_eval_image: int = 5000,
    steps_per_eval_all_images: int = 5000,
    sh_degree: int = 0,  # setting to 0 means only view independent color
) -> str:
    """
    Generates a nerf object from a set of images and returns the path to the config file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # data_dir = "data/robomimic/lift/newest/timestep_8"
    trainer_config: TrainerConfig = copy.deepcopy(method_configs[method])
    if type(trainer_config.pipeline.datamanager.dataparser) is not type(
        dataparsers[dataparser]
    ):
        # a method's data parser's defaults might be different from the data parser's defaults
        trainer_config.pipeline.datamanager.dataparser = dataparsers[dataparser]
    trainer_config.data = data_dir
    trainer_config.output_dir = output_dir
    trainer_config.experiment_name = f"{method}-{data_dir.name}"
    base_dir = trainer_config.get_base_dir()

    if method == "nerfacto":
        # TODO(klin): is it bad to have two wandb scripts running at the same time?
        trainer_config.pipeline.model.near_plane = near_plane
        trainer_config.pipeline.model.far_plane = far_plane
        trainer_config.pipeline.model.predict_normals = True
        trainer_config.pipeline.model.collider_params["near_plane"] = near_plane
        trainer_config.pipeline.model.collider_params["far_plane"] = far_plane
        trainer_config.pipeline.model.camera_optimizer.mode = camera_optimizer_mode
    elif method == "splatfacto":
        trainer_config.pipeline.model.splatfacto_init_scale = splatfacto_init_scale
        trainer_config.pipeline.model.splatfacto_stop_split_at = (
            splatfacto_stop_split_at
        )
        trainer_config.pipeline.model.sh_degree = sh_degree
        trainer_config.pipeline.model.random_scale = random_scale
        trainer_config.steps_per_eval_image = steps_per_eval_image
        trainer_config.steps_per_eval_all_images = steps_per_eval_all_images

        # TODO(klin): after ns==1.0.3, can set trainer_config.pipeline.model.camera_optimizer.mode
    trainer_config.steps_per_save = steps_per_save
    trainer_config.max_num_iterations = max_num_iterations
    vis = "wandb"
    trainer_config.vis = vis

    # TODO(klin): might need to debug these settings
    trainer_config.pipeline.datamanager.dataparser.auto_scale_poses = False
    trainer_config.pipeline.datamanager.dataparser.center_method = "none"
    trainer_config.pipeline.datamanager.dataparser.orientation_method = "none"

    memory_allocated = torch.cuda.memory_allocated("cuda:0")
    print(f"Before training: GPU memory allocated: {memory_allocated / 1024**3} GB")
    memory_cached = torch.cuda.memory_cached()
    print(f"Memory cached: {memory_cached / (1024 ** 3):.2f} GB")
    memory_reserved = torch.cuda.memory_reserved()
    print(f"Memory reserved: {memory_reserved / (1024 ** 3):.2f} GB")

    train_scene_representation(trainer_config)
    base_dir = trainer_config.get_base_dir()

    trainer_config.pipeline.model = None
    trainer_config.pipeline.datamanager = None
    trainer_config.pipeline = None
    trainer_config = None
    del trainer_config
    gc.collect()
    torch.cuda.empty_cache()

    return str(base_dir / "config.yml")


def eval_setup(
    config_path: Path,
    eval_num_rays_per_chunk: Optional[int] = None,
    test_mode: Literal["test", "val", "inference"] = "inference",
    predict_normals: bool = False,
    num_proposal_iterations: int = 2,
    num_proposal_samples_per_ray: Tuple[int, ...] = (512, 512),
    background_color: BackgroundColor = "black",
) -> Tuple[TrainerConfig, Pipeline, Path, int]:
    """Shared setup for loading a saved pipeline for evaluation.

    Using this code to update predict_normals to False because currently
    not successfully handling the case where predict_normals is True
    when no rays hit the aabb box.

    Args:
        config_path: Path to config YAML file.
        eval_num_rays_per_chunk: Number of rays per forward pass
        test_mode:
            'val': loads train/val datasets into memory
            'test': loads train/test dataset into memory
            'inference': does not load any dataset into memory


    Returns:
        Loaded config, pipeline module, corresponding checkpoint, and step
    """
    # load save config
    config = yaml.load(config_path.read_text(), Loader=yaml.Loader)
    assert isinstance(config, TrainerConfig)

    config.pipeline.model.predict_normals = predict_normals
    config.pipeline.model.num_proposal_iterations = num_proposal_iterations
    config.pipeline.model.num_proposal_samples_per_ray = num_proposal_samples_per_ray
    config.pipeline.model.background_color = background_color

    config.pipeline.datamanager._target = all_methods[
        config.method_name
    ].pipeline.datamanager._target
    if eval_num_rays_per_chunk:
        config.pipeline.model.eval_num_rays_per_chunk = eval_num_rays_per_chunk

    # load checkpoints from wherever they were saved
    # TODO: expose the ability to choose an arbitrary checkpoint
    config.load_dir = config.get_checkpoint_dir()
    if isinstance(config.pipeline.datamanager, VanillaDataManagerConfig):
        config.pipeline.datamanager.eval_image_indices = None

    # setup pipeline (which includes the DataManager)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = config.pipeline.setup(device=device, test_mode=test_mode)
    assert isinstance(pipeline, Pipeline)
    pipeline.eval()

    # load checkpointed information
    checkpoint_path, step = eval_load_checkpoint(config, pipeline)

    return config, pipeline, checkpoint_path, step


def extract_obj_files_from_sdf(sdf_path: str) -> List[str]:
    """
    Extracts the obj files from the sdf file (for drake).
    """
    import xml.etree.ElementTree as ET

    obj_files = []
    with open(sdf_path, "r") as file:
        tree = ET.parse(file)
        root = tree.getroot()
        for collision in root.findall(".//collision/geometry/mesh/uri"):
            obj_files.append(collision.text)
    return obj_files


class MeshObject:
    def __init__(
        self,
        mesh_paths: MeshPaths,  # sdf and .obj
        obj_name: str,
        image_ts: Optional[List[Union[int, str]]] = None,
    ):
        """
        Args:
            mesh_path (pathlib.Path): path to mesh; generate the .sdf file from this mesh too.
                Performance optimization: take in a list of meshes (which, presumably, are convex decomp'ed already)
                then transform each of them and convex decomp each of them.
                I see, the convex decomp may be slow ... or maybe not?
            obj_name (str): name of object
            image_ts (Optional[List[int]]): list of image timestamps from which the
                nerf and hence mesh was generated; used for saving the mesh path
        """
        self.mesh_paths = mesh_paths  # update this to include .obj path
        self.obj_name = obj_name
        self.image_ts = image_ts if image_ts is not None else ["unspecified_image_ts"]
        self.obj_type = "mesh"  # maybe not the best way to distinguish between NeRF and Mesh Objects

    def get_center(self) -> np.array:
        """
        Returns the mean of the mesh object's vertices. Use sdf file by default
        """
        if ".sdf" in self.mesh_paths.sdf_path:
            # Find the mesh path from the sdf file
            mesh_paths = extract_mesh_paths_from_sdf(self.mesh_paths.sdf_path)[0]

            # Aggregate vertices from all .obj files using trimesh
            all_vertices = np.array([]).reshape(0, 3)  # Starting with an empty array
            for mesh_path in mesh_paths:
                vertices = trimesh.load(mesh_path).vertices
                all_vertices = np.vstack([all_vertices, vertices])

            return (
                np.mean(all_vertices, axis=0)
                if len(all_vertices) > 0
                else np.array([0, 0, 0])
            )

        elif ".obj" in self.mesh_paths.obj_path:
            mesh = trimesh.load_mesh(self.mesh_paths.obj_path, process=False)
            return np.array(mesh.vertices.mean(axis=0))
        else:
            raise ValueError(f"Unsupported mesh path type: {self.mesh_path}")


class NeRFObject:
    """
    NeRF object class that implements a render method.
    Collision checking method (for RRT*) is TBD; might use a mesh for simplicity.
    """

    _config_path_to_pipeline: Dict[pathlib.Path, Pipeline] = {}

    def __init__(
        self,
        config_path: pathlib.Path,
        bounding_box_min: Optional[torch.Tensor] = None,
        bounding_box_max: Optional[torch.Tensor] = None,
        lazy_load: bool = True,
        # TODO(klin): also store the original object pose from mujoco here to save on env.reset() costs?
    ):
        """
        Args:
            config_path (pathlib.Path): path to nerfstudio training config file
            bounding_box_min (torch.Tensor): min bounding box
                Optional because supplied at SegWrapper; use SegWrapper to save loading pipeline multiple times.
            bounding_box_max (torch.Tensor): max bounding box
                Optional because supplied at SegWrapper; use SegWrapper to save loading pipeline multiple times.
            lazy_load (bool): whether to lazy load the pipeline (for GPU memory efficiency purposes)

            TODO(klin): More generally, bounding boxes are 3D segmentation masks for the object.
        """
        if isinstance(config_path, str):
            config_path = pathlib.Path(config_path)

        # maybe have lazy loading?
        self.config_path = config_path
        # bounding box only for inference; training bounding box is stored elsewhere
        self.bounding_box_min = bounding_box_min
        self.bounding_box_max = bounding_box_max

        self._transforms = None

        self._pipeline = None
        if not lazy_load:
            self.load_pipeline(config_path)
            # the aabb_box needs to also be transformed by the scaling transform / other transforms ...
            # maybe just scale for now?

        self.obj_type = "nerf"  # maybe not the best way to distinguish between NeRF and Mesh Objects

    # try  .transformed()

    def segment(self, seg_params: np.ndarray) -> NeRFObject:
        """
        Segments the object using the given segmentation wrapper.
        """
        return NeRFObject(
            self.config_path,
        )

    def transform(self, transf_params: np.ndarray) -> NeRFObject:
        return NeRFObject(self.config_path, self.transforms @ transf_params)

    # obj_repr_new = obj_repr.transformed(hello).transformed(hello).transformed(hello)

    @property
    def original_timestep(self) -> Optional[int]:
        """
        Assumes config_path contains the timestep corresponding to images used to train the NeRF.

        r'\d+' is the regular expression pattern:
           - \d matches any digit (0-9).
           - + matches one or more occurrences of the preceding pattern.
        """
        match = re.search(r"\d+", str(self.config_path))
        if match:
            first_number = int(match.group())
        else:
            return None

        return first_number

    def load_pipeline(self, config_path: pathlib.Path):
        """
        Loads the pipeline for inference.
        """
        if isinstance(config_path, str):
            config_path = pathlib.Path(config_path)
        # TODO(klin): set background color to transparent somehow
        # indicate via alpha component or via depth
        _, pipeline, _, _ = eval_setup(
            config_path=config_path,
            test_mode="inference",
        )
        self._pipeline = pipeline

    @property
    def pipeline(self):
        if self._pipeline is None:
            self.load_pipeline(self.config_path)
        return self._pipeline

    def render(
        self,
        obj2w: torch.Tensor,
        c2w: torch.Tensor,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        height: Optional[int] = None,
        width: Optional[int] = None,
        camera_type: CameraType = CameraType.PERSPECTIVE,
        inference_aabb_box: Optional[SceneBox] = None,
        upsample_then_downsample: bool = True,
        upsample_scale: int = 2,
    ) -> Dict[str, torch.Tensor]:
        """
        Renders an object in the given world coordinate system.

        Args:
            obj2w (torch.Tensor): A tensor of shape (4, 4) representing the object
                                  to world transformation matrix i.e. object pose. Assume always eye(4).
            c2w (torch.Tensor): A tensor of shape (4, 4) representing the camera to
                                world transformation matrix.
            upsample_then_downsample (bool): whether to upsample then downsample the image
                to the original size; useful for anti-aliasing
            upsample_scale (float): scale factor to upsample by; only used if upsample_then_downsample is True

            Q: what is the world coordinate frame?
            A: If we use a physics simulator, there'd need to be some conversion.
            A simulator's z axis would be parallel to gravity but a NeRF frame mightn't.

        Returns:
            outputs (Dict[str, torch.Tensor]): A dictionary of outputs.
        """
        # Transform the c2w matrix according to the given obj2w pose.
        obj2w = torch.eye(3)  # hardcode for now / forever ...
        R_obj_inv = torch.eye(4).T
        T_obj_inv = -torch.matmul(R_obj_inv, obj2w[:3, 3:])
        R_cam = c2w[:3, :3]
        T_cam = c2w[:3, 3:]
        R_nerf_cam = torch.matmul(R_obj_inv, R_cam)
        T_nerf_cam = T_obj_inv + torch.matmul(R_obj_inv, T_cam)

        c2w_nerf = torch.eye(4)[:3]
        c2w_nerf[:3, :3] = R_nerf_cam
        c2w_nerf[:3, 3:] = T_nerf_cam

        if self.pipeline is None:
            self.load_pipeline(self.config_path)

        if upsample_then_downsample:
            fx *= upsample_scale
            fy *= upsample_scale
            cx *= upsample_scale
            cy *= upsample_scale
            height *= upsample_scale
            width *= upsample_scale

        cameras = Cameras(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            height=height,
            width=width,
            camera_to_worlds=c2w_nerf,
            camera_type=camera_type,
        ).to(self.pipeline.device)

        # print(f"how do I update the points that'll be sampled from camera_ray_bundle? Literally scale everything up?")
        # no, don't update the sampling, just update what comes out of sampling + things for the depth maps e.g.
        # euclidean_to_sampler_fn. Generate rays from the camera at the given pose.
        if not hasattr(self, "inference_aabb_box"):
            self.inference_aabb_box = SceneBox(
                self.pipeline.model.field.aabb.to(self.pipeline.device)
            )
            self.inference_aabb_box = SceneBox(
                torch.tensor([[-0.4, -0.4, 0.8], [0.4, 0.4, 1.05]]).to(
                    self.pipeline.device
                )
            )

        # somehow we've gotten to this render of the nerf object: usually should always render
        # from SceneRepresentation3DSegWrapper or above
        # purpose of above wrapper is to only need to load pipeline once.
        camera_ray_bundle = cameras.generate_rays(
            camera_indices=0, aabb_box=self.inference_aabb_box
        )

        with torch.no_grad():
            outputs = self.pipeline.model.get_outputs_for_camera_ray_bundle(
                camera_ray_bundle
            )

        if upsample_then_downsample:
            for key, value in outputs.items():
                outputs[key] = torch.nn.functional.interpolate(
                    value, scale_factor=1 / upsample_scale, mode="linear"
                )

        if upsample_then_downsample:
            original_height = int(height / upsample_scale)
            original_width = int(width / upsample_scale)
            for key, value in outputs.items():
                outputs[key] = (
                    torch.nn.functional.interpolate(
                        value.permute(2, 0, 1).unsqueeze(0),
                        size=(original_height, original_width),
                        mode="linear",
                    )
                    .squeeze(0)
                    .permute(1, 2, 0)
                )

        return outputs


class GSplatObject:
    def __init__(
        self,
        config_path: pathlib.Path,
        lazy_load: bool = True,
        # TODO(klin): also store the original object pose from mujoco here to save on env.reset() costs?
    ):
        """
        Args:
            config_path (pathlib.Path): path to nerfstudio training config file
            lazy_load (bool): whether to lazy load the pipeline (for GPU memory efficiency purposes)
        """
        if isinstance(config_path, str):
            config_path = pathlib.Path(config_path)

        # maybe have lazy loading?
        self.config_path = config_path

        self._pipeline = None
        if not lazy_load:
            self.load_pipeline(config_path)
            # the aabb_box needs to also be transformed by the scaling transform / other transforms ...
            # maybe just scale for now?

        self.obj_type = "gsplat"  # maybe not the best way to distinguish between NeRF and Mesh Objects

    @property
    def original_timestep(self) -> Optional[int]:
        """
        Assumes config_path contains the timestep corresponding to images used to train the NeRF.

        r'\d+' is the regular expression pattern:
           - \d matches any digit (0-9).
           - + matches one or more occurrences of the preceding pattern.
        """
        match = re.search(r"\d+", str(self.config_path))
        if match:
            first_number = int(match.group())
        else:
            return None

        return first_number

    def load_pipeline(self, config_path: pathlib.Path):
        """
        Loads the pipeline for inference.
        """
        if isinstance(config_path, str):
            config_path = pathlib.Path(config_path)
        # TODO(klin): set background color to transparent somehow
        # indicate via alpha component or via depth
        _, pipeline, _, _ = eval_setup(
            config_path=config_path,
            test_mode="inference",
        )
        self._pipeline = pipeline

    @property
    def pipeline(self):
        if self._pipeline is None:
            self.load_pipeline(self.config_path)
        return self._pipeline

    def render(
        self,
        obj2w: torch.Tensor,
        c2w: torch.Tensor,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        height: Optional[int] = None,
        width: Optional[int] = None,
        camera_type: CameraType = CameraType.PERSPECTIVE,
        camera_names: Optional[List[str]] = None,
        inference_aabb_box: Optional[SceneBox] = None,
        upsample_then_downsample: bool = True,
        upsample_scale: int = 2,
        obb_center: Optional[torch.Tensor] = None,
        obb_rpy: Optional[torch.Tensor] = None,
        obb_scale: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Renders an object in the given world coordinate system.

        Args:
            obj2w (torch.Tensor): A tensor of shape (4, 4) representing the object
                                  to world transformation matrix i.e. object pose.
            c2w (torch.Tensor): A tensor of shape (4, 4) representing the camera to
                                world transformation matrix.
            upsample_then_downsample (bool): whether to upsample then downsample the image
                to the original size; useful for anti-aliasing
            upsample_scale (float): scale factor to upsample by; only used if upsample_then_downsample is True

            Q: what is the world coordinate frame?
            A: If we use a physics simulator, there'd need to be some conversion.
            A simulator's z axis would be parallel to gravity but a NeRF frame mightn't.

        Returns:
            outputs (Dict[str, torch.Tensor]): A dictionary of outputs.
        """
        c2w_nerf = c2w[:3]

        # convert to a list if not a list
        if isinstance(fx, float):
            fx = [fx]
            fy = [fy]
            cx = [cx]
            cy = [cy]
            height = [height]
            width = [width]

        # convert everything to tensors
        c2w_nerf = torch.stack(c2w_nerf)[:, :3]
        fx = torch.tensor(fx).unsqueeze(-1)
        fy = torch.tensor(fy).unsqueeze(-1)
        cx = torch.tensor(cx).unsqueeze(-1)
        cy = torch.tensor(cy).unsqueeze(-1)
        height = torch.tensor(height).unsqueeze(-1)
        width = torch.tensor(width).unsqueeze(-1)

        if upsample_then_downsample:
            fx *= upsample_scale
            fy *= upsample_scale
            cx *= upsample_scale
            cy *= upsample_scale
            height *= upsample_scale
            width *= upsample_scale

        cameras = Cameras(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            height=height,
            width=width,
            camera_to_worlds=c2w_nerf,
            camera_type=camera_type,
        ).to(self.pipeline.device)

        if self.pipeline is None:
            self.load_pipeline(self.config_path)

        camera_to_outputs: Dict[str, Dict[str, torch.Tensor]] = {}

        for camera_idx in range(cameras.size):
            self.pipeline.model.get_outputs = lambda cameras: get_outputs_splatfacto(
                self.pipeline.model,
                cameras,
                obb_center=obb_center,
                obb_rpy=obb_rpy,
                obb_scale=obb_scale,
                transform_params_seq=[],
                output_distances=True,
                # by default [05/02/24], gsplat's depth is z-buffer depth, not ray distances
                # https://github.com/nerfstudio-project/nerfstudio/issues/2743
                # https://github.com/nerfstudio-project/gsplat/issues/99
            )
            with torch.no_grad():
                outputs = self.pipeline.model.get_outputs(cameras[None, camera_idx])

            if upsample_then_downsample:
                original_height = int(
                    (cameras.height[camera_idx] / upsample_scale).item()
                )
                original_width = int(
                    (cameras.width[camera_idx] / upsample_scale).item()
                )
                for key, value in outputs.items():
                    if key == "background":
                        continue

                    outputs[key] = (
                        torch.nn.functional.interpolate(
                            value.permute(2, 0, 1).unsqueeze(0),
                            size=(original_height, original_width),
                            mode="bilinear",
                        )
                        .squeeze(0)
                        .permute(1, 2, 0)
                    )

            camera_to_outputs[camera_names[camera_idx]] = outputs

        return camera_to_outputs


class SceneRepresentation3DSegWrapper:
    """
    Wrapper for NeRFObject that allows for 3D segmentation masks.

    Added to avoid needing to load the pipeline multiple times for the same underlying object.

    For GSplat: maybe I can first get the object in its cropped form without transforms, and then apply the transforms
    to the cropped object.

    For rendering: I'd first crop and then update the means, scales and rotations of the gaussians.
    """

    def __init__(
        self,
        scene_representation: Union[NeRFObject],
        obb_center: np.ndarray,
        obb_rpy: np.ndarray,
        obb_scale: np.ndarray,
    ):
        self._obj = scene_representation
        self.obb_center = obb_center
        self.obb_rpy = obb_rpy
        self.obb_scale = obb_scale
        # need to generalize to 3D segmentation masks for the object
        # currently have been using bounding box only ...
        # for now, let's do only bounding box but we'll need to generalize this later

    # Optionally, if you want to be able to access any method of the original object
    # without explicitly defining a wrapper method, you can use Python's __getattr__ magic method
    def __getattr__(self, name):
        # This will only get called for attributes/methods not explicitly defined in the wrapper
        return getattr(self._obj, name)

    def render(
        self,
        obj2w: torch.Tensor,
        c2w: torch.Tensor,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        height: Optional[int] = None,
        width: Optional[int] = None,
        camera_type: CameraType = CameraType.PERSPECTIVE,
        camera_names: Optional[List[str]] = None,
        inference_aabb_box: Optional[SceneBox] = None,
        upsample_then_downsample: bool = True,
        upsample_scale: int = 2,
    ):
        return self._obj.render(
            obj2w=obj2w,
            c2w=c2w,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            height=height,
            width=width,
            camera_type=camera_type,
            camera_names=camera_names,
            inference_aabb_box=inference_aabb_box,
            upsample_then_downsample=upsample_then_downsample,
            upsample_scale=upsample_scale,
            obb_center=self.obb_center,
            obb_rpy=self.obb_rpy,
            obb_scale=self.obb_scale,
        )


# TODO(klin): what if we performed the transforms inside the TransformWrapper object?
# since we can get the transforms from the transform params anyways?
class TransformType(Enum):
    SE3 = "SE3"
    SCALE = "SCALE"
    SHEAR = "SHEAR"
    WARP = "WARP"


class TransformationWrapper:
    """
    NeRF object class that implements a render method.
    Collision checking method (for RRT*) is TBD; might use a mesh for simplicity.
    """

    def __init__(
        self,
        obj: Union[NeRFObject, MeshObject],
        transform_fn: Callable,
        transform_type: TransformType,
        transform_params: Dict = field(default_factory=dict),
        transform_name: str = "",
    ):
        """
        Args:
            obj: NeRFObject or MeshObject to be wrapped.
            transform: transformation to be applied points. Takes in a point (tensor) and returns a point (tensor).
        """
        self._obj = obj
        self.coordinate_transform_fn = transform_fn
        self._transform_type = transform_type
        self._transform_params = transform_params
        self._transform_name = transform_name

    @cached_property
    def transform_name_seq(self) -> List[str]:
        """Returns a sequence of transform names, from earliest transform's name to latest transform name."""
        if not isinstance(self._obj, TransformationWrapper):
            return [self._transform_name]
        res = [self._transform_name]
        res.extend(self._obj.transform_name_seq[::-1])
        return res[::-1]

    @cached_property
    def transform_params_seq(self) -> List[Dict[str, np.ndarray]]:
        """Returns a sequence of transform params, from earliest transform's params to latest transform params."""
        if not isinstance(self._obj, TransformationWrapper):
            return [self._transform_params]
        res = [self._transform_params.copy()]
        res.extend(self._obj.transform_params_seq[::-1])
        return res[::-1]

    @cached_property
    def transform_type_seq(self) -> List[TransformType]:
        """Returns a sequence of transform types, from earliest transform's type to latest transform."""
        if not isinstance(self._obj, TransformationWrapper):
            return [self._transform_type]
        res = [self._transform_type]
        res.extend(self._obj.transform_type_seq[::-1])
        return res[::-1]

    def get_overall_transform_fn(self) -> Callable:
        if not isinstance(self._obj, TransformationWrapper):

            def transform_fn(x):  # fns instead of lambdas for early binding
                return self.coordinate_transform_fn(x)

        else:
            # to understand the order of transforms, need to understand the math
            # useful to think in terms of spaces
            if self.obj_type == "mesh":
                # assumes that se3 gets called after scale/shearing/other things
                def transform_fn(x):
                    return self.coordinate_transform_fn(
                        self._obj.get_overall_transform_fn()(x)
                    )

            elif self.obj_type == "nerf":

                def transform_fn(x):
                    return self._obj.get_overall_transform_fn()(
                        self.coordinate_transform_fn(x)
                    )

            elif self.obj_type == "gsplat":

                def transform_fn(x):
                    return self._obj.get_overall_transform_fn()(
                        self.coordinate_transform_fn(x)
                    )

        return transform_fn

    def overall_inverse_transform_fn(self) -> Callable:
        """
        Returns the inverse of the overall transform function.

        Used for transforming point in object space to world space, useful for determining aabb
        for camera ray sampling during NeRF rendering.
        """
        raise NotImplementedError("Inverse transform not implemented yet.")
        return lambda x: x

    @property
    def pipeline(self):
        """
        Need to make pipeline a property, rather than let it be an
        attribute so that pipeline can be lazily loaded in NeRFObject.
        """
        assert self._obj is not None, "Object is not initialized."
        return self._obj.pipeline

    def render(
        self,
        obj2w: List[torch.Tensor],
        c2w: List[torch.Tensor],
        fx: List[float],
        fy: List[float],
        cx: List[float],
        cy: List[float],
        height: Optional[List[int]] = None,
        width: Optional[List[int]] = None,
        camera_type: List[CameraType] = field(
            default_factory=lambda: CameraType.PERSPECTIVE
        ),
        camera_names: Optional[List[str]] = None,
        upsample_then_downsample: bool = True,
        upsample_scale: int = 2,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Renders an object in the given world coordinate system.

        Note: implementation might break if doing multiprocessing.

        Args:
            obj2w (torch.Tensor): A tensor of shape (4, 4) representing the object
                                  to world transformation matrix i.e. object pose.
            c2w (torch.Tensor): A tensor of shape (4, 4) representing the camera to
                                world transformation matrix.
            Q: what is the world coordinate frame?
            A: If we use a physics simulator, there'd need to be some conversion.
            A simulator's z axis would be parallel to gravity but a NeRF frame mightn't.

        Returns:
            camera_to_outputs (Dict[str, torch.Tensor]): A dictionary of outputs. From each camera name
                                                  to a dictionary of outputs. Each output dictionary
        """
        # TODO(klin): need better / correct way to turns of predict_normals
        # self.pipeline.model.predict_normals = False
        overall_transform_fn = self.get_overall_transform_fn()
        c2w_nerf = c2w[:3]

        # convert to a list if not a list
        if isinstance(fx, float):
            fx = [fx]
            fy = [fy]
            cx = [cx]
            cy = [cy]
            height = [height]
            width = [width]

        # convert everything to tensors
        c2w_nerf = torch.stack(c2w_nerf)[:, :3]
        fx = torch.tensor(fx).unsqueeze(-1)
        fy = torch.tensor(fy).unsqueeze(-1)
        cx = torch.tensor(cx).unsqueeze(-1)
        cy = torch.tensor(cy).unsqueeze(-1)
        height = torch.tensor(height).unsqueeze(-1)
        width = torch.tensor(width).unsqueeze(-1)

        if self.obj_type == "gsplat":
            upsample_then_downsample = True
            upsample_scale = 4

        if upsample_then_downsample:
            fx *= upsample_scale
            fy *= upsample_scale
            cx *= upsample_scale
            cy *= upsample_scale
            height *= upsample_scale
            width *= upsample_scale

        cameras = Cameras(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            height=height,
            width=width,
            camera_to_worlds=c2w_nerf,
            camera_type=camera_type,
        ).to(self.pipeline.device)

        # don't always have inference_aabb_box already because using the wrapper causes self to be updated to
        # a subclass object, which has a different inference_aabb_box (or none at all)
        if not hasattr(self, "inference_aabb_box"):
            # TODO(klin): get a tight aabb box based on the original 3D bounding box and the transforms
            # for higher quality rendering
            # somehow, making this box too large causes weird rendering artifacts
            self.inference_aabb_box = SceneBox(
                torch.tensor([[-0.4, -0.4, 0.8], [0.4, 0.4, 1.1]]).to(
                    self.pipeline.device
                )
            )
        camera_to_outputs: Dict[str, Dict[str, torch.Tensor]] = {}

        if self.obj_type == "gsplat":
            for camera_idx in range(cameras.size):
                self.pipeline.model.get_outputs = (
                    lambda cameras: get_outputs_splatfacto(
                        self.pipeline.model,
                        cameras,
                        obb_center=self.obb_center,
                        obb_rpy=self.obb_rpy,
                        obb_scale=self.obb_scale,
                        transform_params_seq=self.transform_params_seq,
                        output_distances=True,
                        # by default [05/02/24], gsplat's depth is z-buffer depth, not ray distances
                        # https://github.com/nerfstudio-project/nerfstudio/issues/2743
                        # https://github.com/nerfstudio-project/gsplat/issues/99
                    )
                )
                with torch.no_grad():
                    outputs = self.pipeline.model.get_outputs(cameras[None, camera_idx])

                if upsample_then_downsample:
                    original_height = int(
                        (cameras.height[camera_idx] / upsample_scale).item()
                    )
                    original_width = int(
                        (cameras.width[camera_idx] / upsample_scale).item()
                    )
                    for key, value in outputs.items():
                        if key == "background":
                            continue

                        outputs[key] = (
                            torch.nn.functional.interpolate(
                                value.permute(2, 0, 1).unsqueeze(0),
                                size=(original_height, original_width),
                                mode="bilinear",
                            )
                            .squeeze(0)
                            .permute(1, 2, 0)
                        )

                camera_to_outputs[camera_names[camera_idx]] = outputs

        else:
            for camera_idx in range(cameras.size):
                camera_ray_bundle = cameras.generate_rays(
                    camera_indices=camera_idx, aabb_box=self.inference_aabb_box
                )
                # update pipeline.model.field.get_outputs; need to allow other args
                # note: this implementation will affect other wrappers that use the same underlying pipeline
                # potentially breaks multiprocessing @klin

                if type(self.pipeline.model.field).__name__ == "TensoRFField":
                    get_outputs = get_outputs_tensorf_field
                    get_density = get_density_tensorf_field
                elif type(self.pipeline.model.field).__name__ == "NerfactoField":
                    get_outputs = get_outputs_nerfacto_field
                    get_density = get_density_nerfacto_field
                else:
                    raise NotImplementedError(
                        f"Unknown field type: {type(self.pipeline.model.field).__name__}"
                        "Only TensorField and NerfactoField are supported currently."
                    )

                # a crappy way is to modify the model for this render, then put things back after this render
                self.pipeline.model.field.get_outputs = (
                    lambda x, density_embedding=None: get_outputs(
                        self.pipeline.model.field,
                        x,
                        bounding_box_min=self.obb_center - self.obb_scale / 2,
                        bounding_box_max=self.obb_center + self.obb_scale / 2,
                        density_embedding=density_embedding,
                        overall_transform_fn=overall_transform_fn,
                        transform_params_seq=self.transform_params_seq,
                    )
                )
                self.pipeline.model.field.get_density = lambda x: get_density(
                    self.pipeline.model.field,
                    x,
                    bounding_box_min=self.obb_center - self.obb_scale / 2,
                    bounding_box_max=self.obb_center + self.obb_scale / 2,
                    overall_transform_fn=overall_transform_fn,
                    transform_params_seq=self.transform_params_seq,
                )

                self.pipeline.model.proposal_sampler.generate_ray_samples = (
                    lambda ray_bundle, density_fns, chunk_idx=0: generate_ray_samples(
                        self.pipeline.model.proposal_sampler,
                        ray_bundle=ray_bundle,
                        density_fns=density_fns,
                        chunk_idx=chunk_idx,
                        overall_transform_fn=overall_transform_fn,
                    )
                )

                with torch.no_grad():
                    outputs = self.pipeline.model.get_outputs_for_camera_ray_bundle(
                        camera_ray_bundle,
                    )

                if upsample_then_downsample:
                    original_height = int(
                        (cameras.height[camera_idx] / upsample_scale).item()
                    )
                    original_width = int(
                        (cameras.width[camera_idx] / upsample_scale).item()
                    )
                    for key, value in outputs.items():
                        outputs[key] = (
                            torch.nn.functional.interpolate(
                                value.permute(2, 0, 1).unsqueeze(0),
                                size=(original_height, original_width),
                                mode="bilinear",
                            )
                            .squeeze(0)
                            .permute(1, 2, 0)
                        )

                camera_to_outputs[camera_names[camera_idx]] = outputs

        # flawed way of doing 2D color based augmentation
        # 1. changes color in 2D space, no always 3D consistent
        # 2. code logic is off b/c color wrapper mightn't just be subclass of TransformationWrapper
        # using this method for now b/c transformation wrapper doesn't call wrapped object's render method
        # due to need to modify field.get_outputs

        def apply_color_augmentation() -> None:
            current_obj = self._obj

            while hasattr(current_obj, "_obj"):
                if isinstance(current_obj, ColorAugmentationWrapper):
                    current_obj._apply_augmentation(camera_to_outputs)
                    return

                current_obj = getattr(current_obj, "_obj")

            # Final check if the last object in the chain is a ColorAugmentationWrapper
            if isinstance(current_obj, ColorAugmentationWrapper):
                current_obj._apply_augmentation(camera_to_outputs)

        apply_color_augmentation()

        return camera_to_outputs

    @cached_property
    def obj_name(self) -> str:
        """Returns the name of the base mesh object."""
        assert isinstance(self._obj, MeshObject) or isinstance(
            self._obj, TransformationWrapper
        ), "Must be wrapping a MeshObject to call obj_name."
        return self._obj.obj_name

    @cached_property
    def image_ts(self) -> Optional[List[int]]:
        """Returns the image timestamps of the base mesh object."""
        assert isinstance(self._obj, MeshObject) or isinstance(
            self._obj, TransformationWrapper
        ), "Must be wrapping a MeshObject to call image_ts."
        return self._obj.image_ts

    @cached_property
    def base_mesh_paths(self) -> MeshPaths:
        """Returns the mesh path of the base mesh object."""
        assert isinstance(self._obj, MeshObject) or isinstance(
            self._obj, TransformationWrapper
        ), "Must be wrapping a MeshObject to call base_mesh_paths."
        if isinstance(self._obj, TransformationWrapper):
            return self._obj.base_mesh_paths
        else:
            return self._obj.mesh_paths

    @cached_property
    def mesh_paths(self) -> MeshPaths:
        """Apply

        overall_transform_fn = self.get_overall_transform_fn()

        to the subclass mesh (in trimesh)
        and save the mesh to a new file.
        """
        overall_transform_fn = self.get_overall_transform_fn()
        # load the mesh from the base mesh path
        base_mesh_paths = self.base_mesh_paths

        new_mesh_paths = MeshPaths()

        # get both transformed .obj and .sdf paths
        for path_type, path in asdict(base_mesh_paths).items():
            print(f"PATH: {path}\nPATH: {path}\nPATH: {path}\n")
            if path_type == "sdf_path":
                # expect the base mesh path to be a .sdf file that contains at least one .obj file
                _, collision_mesh_paths = extract_mesh_paths_from_sdf(
                    pathlib.Path(path)
                )
            elif path_type == "obj_path":
                _ = collision_mesh_paths = [path]

            new_collision_mesh_paths = []

            # create new meshdir to avoid file being overwritten between mesh .sdf generation and drake loading of the mesh
            # TODO(klin): when drake (https://github.com/RobotLocomotion/drake/issues/15263)
            # enables in-memory mesh manipulation, can likely avoid saving mesh to disk?
            curr_t = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
            new_meshdir = pathlib.Path(base_mesh_paths.sdf_path).parent / curr_t
            new_meshdir.mkdir(parents=True, exist_ok=True)

            for mesh_path in collision_mesh_paths:
                mesh = trimesh.load(mesh_path)

                # transform the mesh vertices (TODO(klin): do a convex decomp here?)
                transf_vertices = np.array(mesh.vertices)
                transf_vertices = np.concatenate(
                    [transf_vertices, np.ones((transf_vertices.shape[0], 1))], axis=-1
                )
                transf_vertices = overall_transform_fn(transf_vertices)
                transf_vertices = transf_vertices[:, :-1]

                mesh.vertices = transf_vertices
                new_mesh_path = new_meshdir / (
                    pathlib.Path(mesh_path).stem + "_transformed.obj"
                )
                mesh.export(str(new_mesh_path))
                new_collision_mesh_paths.append(new_mesh_path)

            if path_type == "sdf_path":
                # create a new sdf file with the new mesh files
                new_path: str = convert_mesh_to_model(
                    new_collision_mesh_paths,
                    filename_prefix=f"{self.obj_name}_{'-'.join([str(t) for t in self.image_ts])}_{curr_t}",
                )
            elif path_type == "obj_path":
                new_path = str(new_collision_mesh_paths[0])

            setattr(new_mesh_paths, path_type, new_path)
        print(f"new_mesh_paths: {new_mesh_paths}")
        return new_mesh_paths

    # Optionally, if you want to be able to access any method of the original object
    # without explicitly defining a wrapper method, you can use Python's __getattr__ magic method
    def __getattr__(self, name):
        # This will only get called for attributes/methods not explicitly defined in the wrapper
        return getattr(self._obj, name)


class ColorAugmentationWrapper:
    def __init__(
        self,
        obj: Union[TransformationWrapper, NeRFObject, GSplatObject],
        aug_types: List[
            Literal[
                "brightness",
                "contrast",
                "noise",
                "color_jitter",
                "vignette",
                "blur",
                "sharpen",
            ]
        ],
    ):
        """
        Wrapper that applies image based color augmentations to rgb rendering of the object.
        """
        self._obj = obj
        self.aug_types = aug_types

    def _apply_augmentation(self, camera_to_outputs: Dict[str, torch.Tensor]):
        """
        Modifies the rgb outputs of the camera_to_outputs dictionary in place.
        """
        for _, outputs in camera_to_outputs.items():
            if "rgb" in outputs:
                rgb = outputs["rgb"]
                augmented_rgb = self._apply_augmentation_to_rgb(rgb)
                outputs["rgb"] = augmented_rgb

    def _apply_augmentation_to_rgb(self, rgb: torch.Tensor) -> torch.Tensor:
        if "brightness" in self.aug_types:
            factor = random.uniform(0.75, 1.25)
            rgb = torch.clamp(rgb * factor, 0, 1)

        if "contrast" in self.aug_types:
            factor = random.uniform(0.75, 1.25)
            mean = rgb.mean(dim=(0, 1), keepdim=True)
            rgb = torch.clamp((rgb - mean) * factor + mean, 0, 1)

        if "noise" in self.aug_types:
            noise_level = random.uniform(0.0, 0.04)
            noise = torch.randn_like(rgb) * noise_level
            rgb = torch.clamp(rgb + noise, 0, 1)

        return rgb

    # pass through all other methods to the underlying object
    def __getattr__(self, name):
        return getattr(self._obj, name)


# TODO(klin): see if I can improve this method because seems like could be better?
# could at least use Wrapper?
def generate_ray_samples(
    self,
    ray_bundle: Optional[RayBundle] = None,
    density_fns: Optional[List[Callable]] = None,
    chunk_idx: int = 0,
    overall_transform_fn: Optional[Callable] = None,
) -> Tuple[RaySamples, List, List]:
    assert ray_bundle is not None, "ray_bundle must be provided"
    assert density_fns is not None, "density_fns must be provided"

    weights_list = []
    ray_samples_list = []

    n = self.num_proposal_network_iterations
    weights = None
    ray_samples = None
    updated = (
        self._steps_since_update > self.update_sched(self._step) or self._step < 10
    )
    for i_level in range(n + 1):
        is_prop = i_level < n
        num_samples = (
            self.num_proposal_samples_per_ray[i_level]
            if is_prop
            else self.num_nerf_samples_per_ray
        )
        if i_level == 0:
            # Uniform sampling because we need to start with some samples
            ray_samples = self.initial_sampler(ray_bundle, num_samples=num_samples)
        else:
            # PDF sampling based on the last samples and their weights
            # Perform annealing to the weights. This will be a no-op if self._anneal is 1.0.
            assert weights is not None, "weights must be provided"
            annealed_weights = torch.pow(weights, self._anneal)
            ray_samples = self.pdf_sampler(
                ray_bundle, ray_samples, annealed_weights, num_samples=num_samples
            )
        # is_prop = False
        if is_prop:
            if updated:
                raw_positions = ray_samples.frustums.get_positions()
                raw_positions = torch.cat(
                    [raw_positions, torch.ones_like(raw_positions[..., :1])], dim=-1
                )
                transformed_positions = overall_transform_fn(raw_positions)[..., :-1]
                # always update on the first step or the inf check in grad scaling crashes
                density = density_fns[i_level](transformed_positions)
            else:
                with torch.no_grad():
                    density = density_fns[i_level](transformed_positions)
            weights = ray_samples.get_weights(density)
            weights_list.append(weights)  # (num_rays, num_samples)
            ray_samples_list.append(ray_samples)

    if updated:
        self._steps_since_update = 0

    assert ray_samples is not None, "ray_samples must be provided"
    return ray_samples, weights_list, ray_samples_list


class NeRFObjectManager:
    """
    Manages a set of NeRF objects.
    Maps NeRF paths to NeRF objects that've already been instantiated in memory to avoid
    re-loading the same underyling NeRF object multiple times.

    Also manages 3D segmentation masks for each NeRF object and avoids loading new nerf if only the
    3D segmentation mask is different.
    """

    def __init__(self):
        self._nerf_objects = {}
        self._nerf_objects_3d_seg = {}

    def get(
        self, config_path: str, object_name: str
    ) -> Optional[SceneRepresentation3DSegWrapper]:
        """
        Gets the NeRF object associated with the given config path.
        """
        if config_path not in self._nerf_objects.keys():
            return None
        nerf_3d_seg_key = str(config_path) + "_" + object_name
        if nerf_3d_seg_key not in self._nerf_objects_3d_seg.keys():
            # how would Ousterhout handle this? something about minimizing assertions / error handling?
            return None
        return self._nerf_objects_3d_seg[nerf_3d_seg_key]

    def add(
        self,
        config_path: str,
        object_name: str,
        obb_center: Tuple[float, float, float],
        obb_rpy: Tuple[float, float, float],
        obb_scale: Tuple[float, float, float],
    ) -> None:
        """
        Adds a NeRF object with its name, config_path and 3D segmentation info to the manager.
        """
        config_path = str(config_path)

        # if there's already a NeRF object with the same config path, skip
        if config_path in self._nerf_objects:
            nerf_object = self._nerf_objects[config_path]
        else:
            nerf_object = NeRFObject(config_path)
            self._nerf_objects[str(config_path)] = nerf_object

        nerf_3d_seg_key = str(config_path) + "_" + object_name
        if nerf_3d_seg_key not in self._nerf_objects_3d_seg:
            nerf_object_3d_seg = SceneRepresentation3DSegWrapper(
                nerf_object, obb_center, obb_rpy, obb_scale
            )
            self._nerf_objects_3d_seg[nerf_3d_seg_key] = nerf_object_3d_seg


#
class GSplatObjectManager:
    """
    Manages a set of NeRF objects.
    Maps NeRF paths to NeRF objects that've already been instantiated in memory to avoid
    re-loading the same underyling NeRF object multiple times.

    Also manages 3D segmentation masks for each NeRF object and avoids loading new nerf if only the
    3D segmentation mask is different.
    """

    def __init__(self):
        self._gsplat_objects = {}
        self._gsplat_objects_3d_seg = {}

    def get(
        self, config_path: str, object_name: str
    ) -> Optional[SceneRepresentation3DSegWrapper]:
        """
        Gets the gsplat object associated with the given config path.
        """
        if config_path not in self._gsplat_objects.keys():
            return None
        gsplat_3d_seg_key = str(config_path) + "_" + object_name
        if gsplat_3d_seg_key not in self._gsplat_objects_3d_seg.keys():
            # how would Ousterhout handle this? something about minimizing assertions / error handling?
            return None
        return self._gsplat_objects_3d_seg[gsplat_3d_seg_key]

    def add(
        self,
        config_path: str,
        object_name: str,
        obb_center: Tuple[float, float, float],
        obb_rpy: Tuple[float, float, float],
        obb_scale: Tuple[float, float, float],
    ) -> None:
        """
        Adds a gsplat object with its name, config_path and 3D segmentation info to the manager.
        """
        config_path = str(config_path)

        # if there's already a gsplat object with the same config path, skip
        if config_path in self._gsplat_objects:
            gsplat_object = self._gsplat_objects[config_path]
        else:
            gsplat_object = GSplatObject(config_path)
            self._gsplat_objects[str(config_path)] = gsplat_object

        gsplat_3d_seg_key = str(config_path) + "_" + object_name
        if gsplat_3d_seg_key not in self._gsplat_objects_3d_seg:
            gsplat_object_3d_seg = SceneRepresentation3DSegWrapper(
                gsplat_object, obb_center, obb_rpy, obb_scale
            )
            self._gsplat_objects_3d_seg[gsplat_3d_seg_key] = gsplat_object_3d_seg


ReconstructionKey = Tuple[Tuple[int], str]


# In Progress
class ObjectReconstructionManager:
    """
    E.g.
    add_reconstruction("obj_name_1", 0, nerf_obj_1, mesh_obj_1)
    add_reconstruction("obj_name_2", 1, nerf_obj_2, mesh_obj_2)

    get_reconstruction("obj_name_1", 0)

    Many different ts from the outside might use the same t here after
    asking the constraint_info what t should be used.
    """

    def __init__(self):
        self.reconstructions: Dict[
            ReconstructionKey, Tuple[NeRFObject, MeshObject]
        ] = {}

    def add_reconstruction(
        self,
        source_ts: Tuple[int],
        obj_name: str,
        nerf_obj: NeRFObject,
        mesh_obj: MeshObject,
    ) -> None:
        """
        Adds a reconstruction to the manager.

        Reconstruction is identified by the source timesteps used to generate reconstruction,
        as well as the object name.
        """
        nerf_obj = NeRFObject(nerf_obj.config_path)
        self.reconstructions[(source_ts, obj_name)] = (nerf_obj, mesh_obj)

    def get_reconstruction(
        self, source_ts: Tuple[int], obj_name: str
    ) -> Tuple[NeRFObject, MeshObject]:
        for rec in self.reconstructions:
            if rec.obj_name == obj_name and rec.source_ts == source_ts:
                return rec

        # otherwise, generate the reconstruction; hmm should be able to add non-segmented version too?
        raise ValueError(
            f"Reconstruction of object {obj_name} using images fromtime(s) {source_ts} not found."
        )


def apply_transforms_to_pos_quat_wxyz(
    pos: List[float],
    quat_wxyz: List[float],
    transform_params_seq: List[Dict[str, Union[float, np.ndarray]]],
) -> Tuple[List[float], List[float]]:
    """
    Applies a sequence of transformations to a position and quaternion that represent a pose.

    Args:
        pos (List[float]): The position to transform.
        quat_wxyz (List[float]): The quaternion to transform.
        transform_params_seq (List[Dict[str, Union[float, np.ndarray]]]): The sequence of transformations to apply.
    """
    overall_transform_matrix = np.eye(4)
    # put pos quaternion into a 4x4 matrix
    overall_transform_matrix[:3, 3] = pos
    quat_xyzw = np.roll(np.array(quat_wxyz), -1)
    overall_transform_matrix[:3, :3] = R.from_quat(quat_xyzw).as_matrix()

    for transform_params in transform_params_seq:
        X_origin_t = next(
            (v for k, v in transform_params.items() if "origin" in k), None
        )
        assert (
            len([k for k in transform_params if "origin" in k]) == 1
        ), "Only one origin should be provided for a given transform"
        X_transform_t = next(
            (v for k, v in transform_params.items() if "origin" not in k), None
        )
        assert (
            len([k for k in transform_params if "origin" not in k]) == 1
        ), "Only one non-origin should be provided for a given transform"
        # convert to tensors
        X_origin_t = np.array(X_origin_t)
        X_transform_t = np.array(X_transform_t)
        overall_transform_matrix = np.matmul(
            X_origin_t,
            np.matmul(
                X_transform_t,
                np.matmul(np.linalg.inv(X_origin_t), overall_transform_matrix),
            ),
        )

    pos = overall_transform_matrix[:3, 3]
    quat_wxyz = np.roll(R.from_matrix(overall_transform_matrix[:3, :3]).as_quat(), 1)
    return pos, quat_wxyz
