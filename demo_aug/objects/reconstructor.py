import pathlib
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from demo_aug.aug_datastructures import (
    load_image_ts_obj_name_to_bounding_box,
    load_image_ts_obj_name_to_mesh_paths,
    load_image_ts_to_trained_gsplat_path,
    load_image_ts_to_trained_nerf_path,
    save_image_ts_obj_name_to_bounding_box,
    save_image_ts_obj_name_to_mesh_paths,
    save_image_ts_to_trained_gsplat_path,
    save_image_ts_to_trained_nerf_path,
)
from demo_aug.objects.nerf_object import (
    GSplatObject,
    GSplatObjectManager,
    MeshObject,
    MeshPaths,
    NeRFObject,
    NeRFObjectManager,
    generate_mesh,
    generate_scene_representation,
    get_nerf_folder_path,
)


@dataclass
class SegmentationMask:
    """For now, using oriented bounding box as segmentation mask."""

    oriented_bbox_pos: Optional[Tuple[float, float, float]] = None
    oriented_bbox_rpy: Optional[Tuple[float, float, float]] = None
    oriented_bbox_scale: Optional[Tuple[float, float, float]] = None


class ReconstructionType(Enum):
    NeRF = "NeRF"
    Mesh = "Mesh"
    GaussianSplat = "GaussianSplat"


# TODO(klin): merge GaussianSplat + NeRF reconstructor
class NeRFReconstructor:
    @staticmethod
    def reconstruct(
        input_data_path: str, dataparser: str, trainer_configs: Optional[Any] = None
    ) -> str:
        """
        Reconstructs an object or scene from the given input data.
        :param input_data: Data for reconstruction.
        :return: Path to reconstructed object.
        """
        trained_nerf_path = generate_scene_representation(
            # this method should take in the standard NeRFStudio configs too?
            # e.g. a deepcopy of the default config?
            pathlib.Path(input_data_path),
            method="nerfacto",
            output_dir=pathlib.Path("nerf_outputs"),
            max_num_iterations=12001,  # for transparent objects; otherwise, 6001 is fine
            steps_per_save=3000,
            dataparser=dataparser,
            camera_optimizer_mode="SO3xR3",  # seems really important for real world data nerf quality
        )

        return trained_nerf_path


class GaussianSplatReconstructor:
    @staticmethod
    def reconstruct(
        input_data_path: str, dataparser: str, trainer_configs: Optional[Any] = None
    ) -> str:
        """
        Reconstructs an object or scene from the given input data.
        :param input_data: Data for reconstruction.
        :return: Path to reconstructed object.
        """
        trained_gsplat_path = generate_scene_representation(
            # this method should take in the standard splatStudio configs too?
            # e.g. a deepcopy of the default config?
            pathlib.Path(input_data_path),
            output_dir=pathlib.Path("splat_outputs"),
            method="splatfacto",
            # max_num_iterations=20001,
            # steps_per_save=5000,
            # hardcode for loading pc only
            max_num_iterations=301,
            steps_per_save=300,
            steps_per_eval_image=5000,
            steps_per_eval_all_images=5000,
            dataparser=dataparser,
            splatfacto_init_scale=4.0,
            splatfacto_stop_split_at=6000,
        )

        return trained_gsplat_path


class MeshReconstructor:
    @staticmethod
    def reconstruct(
        input_data_path: str,
        final_output_folder: str,
        segmentation_mask: SegmentationMask,
        mesh_reconstruction_configs: Optional[Any] = None,
        mesh_name: Optional[str] = None,
    ) -> MeshPaths:
        """
        Reconstructs an object or scene from the given input data.
        :param input_data: Data for reconstruction.
        :return: Path to reconstructed object.
        """
        print(
            f"Reconstructing mesh for {input_data_path} with mask {segmentation_mask}."
        )
        mesh_paths = generate_mesh(
            nerf_config_path=pathlib.Path(input_data_path),
            output_dir=pathlib.Path("demo_aug/models/assets/task_relevant")
            / final_output_folder,
            bounding_box_min=segmentation_mask.oriented_bbox_pos
            - segmentation_mask.oriented_bbox_scale / 2,
            bounding_box_max=segmentation_mask.oriented_bbox_pos
            + segmentation_mask.oriented_bbox_scale / 2,
            mesh_name=mesh_name,
        )
        return mesh_paths


class ReconstructionManager:
    """Returns NeRFObject or MeshObject."""

    def __init__(self, reconstruction_path: str, demo_name: str):
        self._reconstruction_path = reconstruction_path
        self._demo_name = demo_name

        self._nerf_object_manager = NeRFObjectManager()
        self._gsplat_object_manager = GSplatObjectManager()
        self._mesh_object_manager = (
            None  # not using this because Meshes are cheaper to hold in memory
        )

        self._image_ts_to_trained_nerf_path: Dict[Tuple[int], str] = {}
        self._image_ts_obj_name_to_mesh_paths: Dict[
            Tuple[Tuple[int], str], MeshPaths
        ] = {}
        self._image_ts_obj_name_to_oriented_bounding_box: Dict[
            Tuple[Tuple[int], str], np.ndarray
        ] = {}

        self.load_reconstructions(reconstruction_path, demo_name)

    def load_reconstructions(
        self, reconstruction_manager_path: str, demo_name: str
    ) -> None:
        """Load existing 3D reconstructions for the given demonstration.

        TODO: decouple from the entire Demo object. Just happens that the reconstruction storing stuff
        is inside the Demo object for now.
        """
        self._image_ts_to_trained_nerf_path: Dict[Tuple[int], str] = (
            load_image_ts_to_trained_nerf_path(reconstruction_manager_path, demo_name)
        )
        self._image_ts_to_trained_gsplat_path: Dict[Tuple[int], str] = (
            load_image_ts_to_trained_gsplat_path(reconstruction_manager_path, demo_name)
        )
        # # pop ((2, ), "background") from the dict
        self._image_ts_to_trained_gsplat_path.pop((2,), "background")
        # self._image_ts_to_trained_gsplat_path = {}
        self._image_ts_obj_name_to_mesh_paths: Dict[
            Tuple[Tuple[int], str], MeshPaths
        ] = load_image_ts_obj_name_to_mesh_paths(reconstruction_manager_path, demo_name)
        self._image_ts_obj_name_to_oriented_bounding_box: Dict[
            Tuple[Tuple[int], str], Tuple[np.ndarray, np.ndarray, np.ndarray]
        ] = load_image_ts_obj_name_to_bounding_box(
            reconstruction_manager_path, demo_name
        )

    def get_reconstruction(
        self,
        reconstruction_ts: Tuple[int],
        obj_name: Optional[str] = None,
        reconstruction_type: ReconstructionType = ReconstructionType.NeRF,
    ) -> Optional[Union[NeRFObject, MeshObject, GSplatObject]]:
        """
        Get the reconstruction for the given timestamp and object name.
        :param reconstruction_ts: Timestamp of the reconstruction.
        :param obj_name: Name of the object.

        :return: The reconstruction for the given timestamp and object name. Returns None
        if the reconstruction does not exist.
        """
        # Get the reconstruction
        if reconstruction_type == ReconstructionType.NeRF:
            nerf_path = self._image_ts_to_trained_nerf_path.get(reconstruction_ts, None)
            obj = self._nerf_object_manager.get(nerf_path, obj_name)
        elif reconstruction_type == ReconstructionType.GaussianSplat:
            gsplat_path = self._image_ts_to_trained_gsplat_path.get(
                reconstruction_ts, None
            )
            obj = self._gsplat_object_manager.get(gsplat_path, obj_name)
        elif reconstruction_type == ReconstructionType.Mesh:
            mesh_paths = self._image_ts_obj_name_to_mesh_paths.get(
                (reconstruction_ts, obj_name), None
            )
            if mesh_paths is not None:
                obj = MeshObject(mesh_paths, obj_name, reconstruction_ts)
            else:
                obj = None
        else:
            raise ValueError(f"Invalid reconstruction type: {reconstruction_type}")

        return obj

    # what should the stuff take in? Should take in timesteps and name?
    def reconstruct(
        self,
        image_ts: Tuple[int],
        t_to_nerf_folder_path: Dict[str, pathlib.Path],
        obj_name: Optional[str] = None,
        segmentation_mask: Optional[SegmentationMask] = None,
        reconstruction_type: ReconstructionType = ReconstructionType.NeRF,
    ) -> None:
        """Generate reconstruction using images.

        Either reconstruct to NeRFObject; if no bounding box.
        If segmentation_mask provided, then also reconstruct MeshObject.
        """
        if self.get_reconstruction(image_ts, obj_name, reconstruction_type) is not None:
            print(
                f"Reconstruction of obj {obj_name} at times {image_ts} of reconstruction type "
                f"{reconstruction_type.name} already exists. Passing."
            )
            return

        if segmentation_mask is not None:
            self._image_ts_obj_name_to_oriented_bounding_box[(image_ts, obj_name)] = (
                np.array(segmentation_mask.oriented_bbox_pos),
                np.array(segmentation_mask.oriented_bbox_rpy),
                np.array(segmentation_mask.oriented_bbox_scale),
            )
            save_image_ts_obj_name_to_bounding_box(
                self._reconstruction_path,
                self._demo_name,
                self._image_ts_obj_name_to_oriented_bounding_box,
            )

        # TODO(klin): duplicated code for nerf and gsplat bounding box retrieval
        if reconstruction_type == ReconstructionType.NeRF:
            nerf_path = self._image_ts_to_trained_nerf_path.get(image_ts, None)
            if nerf_path is None:
                nerf_folder_path = get_nerf_folder_path(image_ts, t_to_nerf_folder_path)
                nerf_path = NeRFReconstructor.reconstruct(
                    nerf_folder_path, dataparser="nerfstudio-data"
                )
                self._image_ts_to_trained_nerf_path[image_ts] = nerf_path
                save_image_ts_to_trained_nerf_path(
                    self._reconstruction_path,
                    self._demo_name,
                    self._image_ts_to_trained_nerf_path,
                )
                print(
                    f"Reconstructed NeRFObject for obj {obj_name} at times {image_ts}."
                )

            obb = self._image_ts_obj_name_to_oriented_bounding_box.get(
                (image_ts, obj_name), None
            )
            if obb is not None:
                # ideally, object should have a name; not enforcing for now?
                self._nerf_object_manager.add(
                    nerf_path, obj_name, obb[0], obb[1], obb[2]
                )
                print(f"Added NeRFObject w obb for obj {obj_name} at times {image_ts}.")
        elif reconstruction_type == ReconstructionType.GaussianSplat:
            gsplat_path = self._image_ts_to_trained_gsplat_path.get(image_ts, None)
            if gsplat_path is None:
                gsplat_folder_path = get_nerf_folder_path(
                    image_ts, t_to_nerf_folder_path
                )
                gsplat_path = GaussianSplatReconstructor.reconstruct(
                    gsplat_folder_path, dataparser="nerfstudio-data"
                )
                self._image_ts_to_trained_gsplat_path[image_ts] = gsplat_path
                save_image_ts_to_trained_gsplat_path(
                    self._reconstruction_path,
                    self._demo_name,
                    self._image_ts_to_trained_gsplat_path,
                )
                print(
                    f"Reconstructed GSplatObj for obj {obj_name} at times {image_ts}."
                )

            obb = self._image_ts_obj_name_to_oriented_bounding_box.get(
                (image_ts, obj_name), None
            )
            if obb is not None:
                # ideally, object should have a name; not enforcing for now?
                self._gsplat_object_manager.add(
                    gsplat_path, obj_name, obb[0], obb[1], obb[2]
                )
                print(
                    f"Added GSplatObject w obb for obj {obj_name} at times {image_ts}."
                )
        elif reconstruction_type == ReconstructionType.Mesh:
            mesh_paths = self._image_ts_obj_name_to_mesh_paths.get(
                (image_ts, obj_name), None
            )
            if mesh_paths is None:
                trained_nerf_config_path = self._image_ts_to_trained_nerf_path.get(
                    image_ts, None
                )
                if trained_nerf_config_path is None:
                    raise ValueError(
                        f"NeRF reconstruction does not exist for obj {obj_name} at times {image_ts}. "
                        "Please reconstruct NeRFObject first."
                    )
                # Reconstruct the object based on the constraint
                mesh_paths = MeshReconstructor.reconstruct(
                    trained_nerf_config_path,
                    str(
                        pathlib.Path(trained_nerf_config_path).parent / "meshes"
                    ),  # decide on an output dir
                    segmentation_mask,
                    mesh_name=obj_name + str(image_ts),
                )
                self._image_ts_obj_name_to_mesh_paths[(image_ts, obj_name)] = mesh_paths
                save_image_ts_obj_name_to_mesh_paths(
                    self._reconstruction_path,
                    self._demo_name,
                    self._image_ts_obj_name_to_mesh_paths,
                )
                print(
                    f"Reconstructed MeshObject for obj {obj_name} at times {image_ts}."
                )
        else:
            import ipdb

            ipdb.set_trace()
            raise ValueError(f"Invalid reconstruction type: {reconstruction_type}")

        return
