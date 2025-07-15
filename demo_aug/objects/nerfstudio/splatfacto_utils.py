from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from gsplat.cuda_legacy._torch_impl import quat_to_rotmat

try:
    from gsplat.rendering import rasterization
except ImportError:
    print("Please install gsplat>=1.0.0")
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.scene_box import OrientedBox

# need following import for background color override
from nerfstudio.models.splatfacto import get_viewmat

from demo_aug.utils.camera_utils import get_real_distance_map
from demo_aug.utils.mathutils import rotmat_to_quat


def apply_transforms_to_gaussians(
    means: torch.Tensor,
    quats: torch.Tensor,
    transform_params_seq: Optional[List[Dict[str, Union[float, np.ndarray]]]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies a sequence of transformations to a set of gaussians.

    Args:
        means: Tensor of shape (N, 3) containing the means of the gaussians.
        quats: Tensor of shape (N, 4) containing the quaternions of the gaussians.
        transform_params_seq: List of dictionaries containing the transformation parameters for each transformation in the sequence.
            We apply the transformations in the order they appear in the list.
    Returns:
        Tuple of tensors containing the transformed means and quaternions.
    """
    # generate the overall transformation function
    overall_transform_matrix = torch.eye(4, device=means.device, dtype=means.dtype)
    if transform_params_seq is None:
        transform_params_seq = []
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
        X_origin_t = torch.tensor(X_origin_t, device=means.device, dtype=means.dtype)
        X_transform_t = torch.tensor(
            X_transform_t, device=means.device, dtype=means.dtype
        )
        overall_transform_matrix = torch.matmul(
            X_origin_t,
            torch.matmul(
                X_transform_t,
                torch.matmul(torch.linalg.inv(X_origin_t), overall_transform_matrix),
            ),
        )

    # Convert means + quats to SE3 matrices
    src_matrices = torch.zeros(
        (quats.size(0), 4, 4), device=means.device, dtype=means.dtype
    )
    rotmats = quat_to_rotmat(quats)
    src_matrices[:, :3, :3] = rotmats
    src_matrices[:, :3, 3] = means
    src_matrices[:, 3, 3] = 1

    # Apply transformation
    transformed_matrices = torch.matmul(
        overall_transform_matrix.unsqueeze(0), src_matrices
    )

    # Extract transformed means and rotation matrices
    transformed_means = transformed_matrices[:, :3, 3]
    transformed_rotmats = transformed_matrices[:, :3, :3]

    # Convert the rotation matrices back to quaternions
    transformed_quats = rotmat_to_quat(transformed_rotmats)
    transformed_quats = transformed_quats / torch.norm(
        transformed_quats, dim=-1, keepdim=True
    )

    return transformed_means, transformed_quats


def get_outputs(
    self,
    camera: Cameras,
    obb_center: Optional[Tuple[float, float, float]] = None,
    obb_rpy: Optional[Tuple[float, float, float]] = None,
    obb_scale: Optional[Tuple[float, float, float]] = None,
    transform_params_seq: Optional[List[Dict[str, np.ndarray]]] = None,
    output_distances: bool = True,
) -> Dict[str, Union[torch.Tensor, List]]:
    """Takes in a Ray Bundle and returns a dictionary of outputs.

    Args:
        ray_bundle: Input bundle of rays. This raybundle should have all the
        needed information to compute the outputs.

    Returns:
        Outputs of model. (ie. rendered colors)
    """
    if not isinstance(camera, Cameras):
        print("Called get_outputs with not a camera")
        return {}

    if self.training:
        assert camera.shape[0] == 1, "Only one camera at a time"
        optimized_camera_to_world = self.camera_optimizer.apply_to_camera(camera)
    else:
        optimized_camera_to_world = camera.camera_to_worlds

    # updated cropping code
    if obb_center is not None:
        crop_box = OrientedBox.from_params(pos=obb_center, rpy=obb_rpy, scale=obb_scale)
    else:
        crop_box = self.crop_box

    if crop_box is not None and not self.training:
        crop_ids = crop_box.within(self.means).squeeze()
        if crop_ids.sum() == 0:
            background = self._get_background_color()

            rgb = background.repeat(
                int(camera.height.item()), int(camera.width.item()), 1
            )
            depth = background.new_ones(*rgb.shape[:2], 1) * 10
            accumulation = background.new_zeros(*rgb.shape[:2], 1)
            return {
                "rgb": rgb,
                "depth": depth,
                "accumulation": accumulation,
                "background": background,
            }
    else:
        crop_ids = None

    # # cropping
    # if self.crop_box is not None and not self.training:
    #     crop_ids = self.crop_box.within(self.means).squeeze()
    #     if crop_ids.sum() == 0:
    #         return self.get_empty_outputs(
    #             int(camera.width.item()), int(camera.height.item()), self.background_color
    #         )
    # else:
    #     crop_ids = None

    if crop_ids is not None:
        opacities_crop = self.opacities[crop_ids]
        means_crop = self.means[crop_ids]
        features_dc_crop = self.features_dc[crop_ids]
        features_rest_crop = self.features_rest[crop_ids]
        scales_crop = self.scales[crop_ids]
        quats_crop = self.quats[crop_ids]
    else:
        opacities_crop = self.opacities
        means_crop = self.means
        features_dc_crop = self.features_dc
        features_rest_crop = self.features_rest
        scales_crop = self.scales
        quats_crop = self.quats

    means_crop, quats_crop = apply_transforms_to_gaussians(
        means_crop, quats_crop, transform_params_seq
    )
    colors_crop = torch.cat((features_dc_crop[:, None, :], features_rest_crop), dim=1)

    BLOCK_WIDTH = (
        16  # this controls the tile size of rasterization, 16 is a good default
    )
    camera_scale_fac = self._get_downscale_factor()
    camera.rescale_output_resolution(1 / camera_scale_fac)
    viewmat = get_viewmat(optimized_camera_to_world)
    K = camera.get_intrinsics_matrices().cuda()
    W, H = int(camera.width.item()), int(camera.height.item())
    self.last_size = (H, W)
    camera.rescale_output_resolution(camera_scale_fac)  # type: ignore

    # apply the compensation of screen space blurring to gaussians
    if self.config.rasterize_mode not in ["antialiased", "classic"]:
        raise ValueError("Unknown rasterize_mode: %s", self.config.rasterize_mode)

    if self.config.output_depth_during_training or not self.training:
        render_mode = "RGB+ED"
    else:
        render_mode = "RGB"

    if self.config.sh_degree > 0:
        sh_degree_to_use = min(
            self.step // self.config.sh_degree_interval, self.config.sh_degree
        )
    else:
        colors_crop = torch.sigmoid(colors_crop)
        sh_degree_to_use = None

    render, alpha, info = rasterization(
        means=means_crop,
        quats=quats_crop / quats_crop.norm(dim=-1, keepdim=True),
        scales=torch.exp(scales_crop),
        opacities=torch.sigmoid(opacities_crop).squeeze(-1),
        colors=colors_crop,
        viewmats=viewmat,  # [1, 4, 4]
        Ks=K,  # [1, 3, 3]
        width=W,
        height=H,
        tile_size=BLOCK_WIDTH,
        packed=False,
        near_plane=0.01,
        far_plane=1e10,
        render_mode=render_mode,
        sh_degree=sh_degree_to_use,
        sparse_grad=False,
        absgrad=True,
        rasterize_mode=self.config.rasterize_mode,
        # set some threshold to disregrad small gaussians for faster rendering.
        # radius_clip=3.0,
    )
    if self.training and info["means2d"].requires_grad:
        info["means2d"].retain_grad()
    self.xys = info["means2d"]  # [1, N, 2]
    self.radii = info["radii"][0]  # [N]
    alpha = alpha[:, ...]

    background = self._get_background_color()
    rgb = render[:, ..., :3] + (1 - alpha) * background
    rgb = torch.clamp(rgb, 0.0, 1.0)

    if render_mode == "RGB+ED":
        depth_im = render[:, ..., 3:4]
        depth_im = torch.where(alpha > 0, depth_im, depth_im.detach().max()).squeeze(0)
    else:
        depth_im = None

    if background.shape[0] == 3 and not self.training:
        background = background.expand(H, W, 3)

    if output_distances:
        # calculate the FOV of the camera given fx and fy, width and height
        cx = camera.cx.item()
        cy = camera.cy.item()
        W, H = int(camera.width.item()), int(camera.height.item())
        self.last_size = (H, W)

        depth_im = get_real_distance_map(
            depth_im.squeeze(-1),
            camera.fx.item(),
            camera.fy.item(),
            cx,
            cy,
        ).unsqueeze(-1)

    return {
        "rgb": rgb.squeeze(0),  # type: ignore
        "depth": depth_im,  # type: ignore
        "accumulation": alpha.squeeze(0),  # type: ignore
        "background": background,  # type: ignore
    }  # type: ignore
