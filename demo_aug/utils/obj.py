"""
Object class for NeRF that can be rendered from a provided camera pose.
"""

import pathlib
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.utils import plotly_utils as vis


# Think this is unused
class RenderableCollidableObject:
    """Class for object that can be rendered from a provided camera pose and has collision geometry.

    RenderableCollidableObject.is_colliding(obj1, obj2)

    Unclear how to exactly check collisions between two (nerf) objects.
    """

    def __init__(
        self,
        trained_nerf_path: pathlib.Path,
        trained_nerf: Optional[torch.nn.Module] = None,
        nerf_training_data_path: Optional[pathlib.Path] = None,
        obj2w: Optional[np.ndarray] = None,
    ):
        """
        Initializes a new instance of RenderableCollidableObject. If there was a trained_nerf already,
        use it. Otherwise, use the provided nerf_training_data_path and train the nerf.

        Once complete, save the trained nerf to the provided path. Unfortunately, we
        can't save to hdf5 file directly.

        :param trained_nerf_path: A pathlib.Path object for the trained NERF path.
        :param trained_nerf: An optional torch.nn.Module object for the trained NERF.
        :param nerf_training_data_path: An optional pathlib.Path object for the NERF training data path.
        :param obj2w: An optional numpy ndarray object for obj2w.
        """
        self.trained_nerf_path = trained_nerf_path
        self.trained_nerf = trained_nerf
        self.nerf_training_data_path = nerf_training_data_path
        self.obj2w = obj2w

    def render(
        self,
        obj2w: torch.Tensor,
        c2w: torch.Tensor,
        cx: float,
        cy: float,
        fx: float,
        fy: float,
        height: int,
        width: int,
        camera_type: CameraType = CameraType.PERSPECTIVE,
        rendered_output_names: List[str] = None,
        save_dir: Optional[pathlib.Path] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Class for rendering NeRFStudio trained objects.

        Args:
            obj2w (torch.Tensor): 3 x 4 object-to-world transformation matrix
            c2w (torch.Tensor): 3 x 4 camera-to-world transformation matrix
            cx (float): horizontal focal length
            cy (float): vertical focal length
            fx (float): horizontal principal point
            fy (float): vertical principal point
            height (int): image height
            width (int): image width
            camera_type (CameraType): camera type
            rendered_output_names (List[str], optional): list of names of rendered output. Defaults to None.

        Returns:
            Dict[str, torch.Tensor]: dictionary of output tensors
        """
        if rendered_output_names is None:
            rendered_output_names = ["rgb", "depth", "accumulation"]
        camera, ray_bundle = self._get_camera_and_ray_bundle(
            obj2w, c2w, cx, cy, fx, fy, height, width, camera_type
        )
        outputs = self._get_ray_outputs(
            ray_bundle, self.cylinder_center, self.cylinder_radius, self.cylinder_height
        )

        reshaped_outputs = {}
        for output_name in rendered_output_names:
            if output_name == "rgb":
                reshaped_outputs[output_name] = outputs[output_name].reshape(
                    (camera.height, camera.width, -1)
                )
            else:
                reshaped_outputs[output_name] = outputs[output_name].reshape(
                    (camera.height, camera.width)
                )

        if save_dir is not None:
            rgb, depth, accumulation = (
                outputs["rgb"],
                outputs["depth"],
                outputs["accumulation"],
            )
            save_dir.mkdir(parents=True, exist_ok=True)
            # convert to numpy
            rgb = rgb.numpy()
            depth = depth.numpy()
            accumulation = accumulation.numpy()
            plt.imsave(save_dir / "rgb_aab.png", rgb)
            plt.imsave(save_dir / "depth_aab.png", depth)
            plt.imsave(save_dir / "accumulation_aab.png", accumulation)
            fig = vis.vis_camera_rays(camera)
            fig.show()

        return reshaped_outputs

    def _get_camera_and_ray_bundle(
        self,
        obj2w: torch.Tensor,
        c2w: torch.Tensor,
        cx: float,
        cy: float,
        fx: float,
        fy: float,
        height: int,
        width: int,
        camera_type: CameraType = CameraType.PERSPECTIVE,
    ) -> Tuple[Cameras, RayBundle]:
        if isinstance(obj2w, np.ndarray):
            obj2w = torch.as_tensor(obj2w, dtype=torch.float32)
        elif not isinstance(obj2w, torch.Tensor):
            raise TypeError("Input must be a NumPy array or PyTorch tensor.")

        if isinstance(c2w, np.ndarray):
            c2w = torch.as_tensor(c2w, dtype=torch.float32)
        elif not isinstance(c2w, torch.Tensor):
            raise TypeError("Input must be a NumPy array or PyTorch tensor.")

        if c2w.dtype != obj2w.dtype:
            c2w = c2w.to(dtype=obj2w.dtype)

        R_obj_inv = obj2w[:3, :3].T
        T_obj_inv = -torch.matmul(R_obj_inv, obj2w[:3, 3:])
        R_cam = c2w[:3, :3]
        T_cam = c2w[:3, 3:]
        R_nerf_cam = torch.matmul(R_obj_inv, R_cam)
        T_nerf_cam = T_obj_inv + torch.matmul(R_obj_inv, T_cam)

        c2w_nerf = torch.eye(4)[:3]
        c2w_nerf[:3, :3] = R_nerf_cam
        c2w_nerf[:3, 3:] = T_nerf_cam
        c2w = c2w_nerf
        camera = Cameras(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            width=width,
            height=height,
            camera_to_worlds=c2w,
            camera_type=camera_type,
        )
        # Generate rays from the camera at the given pose.
        ray_bundle = camera.generate_rays(camera_indices=0, aabb_box=None)
        return camera, ray_bundle
