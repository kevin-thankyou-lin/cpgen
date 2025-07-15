"""
Object class for NeRF that can be rendered from a provided camera pose.
"""

import pathlib
from typing import Dict, List, Optional, Tuple, Union

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from nerfstudio.cameras.camera_utils import viewmatrix
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.cameras.rays import RayBundle

from demo_aug.utils.composite import alpha_composite


class NeRF:
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
            from nerfstudio.utils import plotly_utils as vis

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


class SphereNeRF(NeRF):
    def __init__(
        self,
        sphere_center: Optional[torch.Tensor] = None,
        sphere_radius: Optional[float] = 1,
        rgb: Optional[torch.Tensor] = None,
    ) -> None:
        self.sphere_center = (
            sphere_center
            if sphere_center is not None
            else torch.tensor([0, 0, 0], dtype=torch.float32)
        )
        self.sphere_radius = sphere_radius
        self.sphere_rgb = (
            rgb if rgb is not None else torch.tensor([0, 0, 1], dtype=torch.float32)
        )

    def render(
        self,
        obj2w: Union[torch.Tensor, np.ndarray],
        c2w: Union[torch.Tensor, np.ndarray],
        cx: float,
        cy: float,
        fx: float,
        fy: float,
        height: Optional[int] = None,
        width: Optional[int] = None,
        camera_type: CameraType = CameraType.PERSPECTIVE,
        rendered_output_names: List[str] = None,
        save_dir: Optional[pathlib.Path] = None,
    ) -> Dict[str, torch.Tensor]:
        if rendered_output_names is None:
            rendered_output_names = ["rgb", "depth", "accumulation"]
        camera, ray_bundle = self._get_camera_and_ray_bundle(
            obj2w, c2w, cx, cy, fx, fy, height, width, camera_type
        )
        outputs = self._get_ray_outputs(
            ray_bundle, self.sphere_center, self.sphere_radius
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
            from nerfstudio.utils import plotly_utils as vis

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

    def _get_ray_outputs(
        self, ray_bundle: RayBundle, sphere_center: torch.Tensor, sphere_radius: float
    ) -> Dict[str, torch.Tensor]:
        """
        Get the rgb and depth of each ray in the given RayBundle for a scene with a sphere.

        Args:
            ray_bundle: A RayBundle object containing the ray parameters.
            sphere_center: A torch.Tensor of shape (3,) specifying the center of the sphere.
            sphere_radius: A float specifying the radius of the sphere.

        Returns:
            A dict containing the rgb, depth and accumulation torch tensors for each ray in the bundle.
            NB: depth values are 0 if ray has infinite depth.

        # Reference https://raytracing.github.io/books/RayTracingInOneWeekend.html#addingasphere/ray-sphereintersection
        """
        # Calculate the vector from the ray origin to the sphere center
        oc = ray_bundle.origins - sphere_center

        # Calculate the coefficients of the quadratic equation
        a = torch.sum(ray_bundle.directions**2, dim=-1)
        b = 2 * torch.sum(ray_bundle.directions * oc, dim=-1)
        c = torch.sum(oc**2, dim=-1) - sphere_radius**2

        # Calculate the discriminant
        discriminant = b**2 - 4 * a * c
        # Find the rays that intersect the sphere
        t = (-b - torch.sqrt(discriminant)) / (2 * a)
        valid_mask = (discriminant >= 0) & (t > 0)

        # Check if t is between near and far; only checking the first "intersection"
        if ray_bundle.nears is not None:
            valid_mask = valid_mask & (t >= ray_bundle.nears)
        if ray_bundle.fars is not None:
            valid_mask = valid_mask & (t <= ray_bundle.fars)

        # Set the rgb and depth of each ray
        rgb = torch.ones(
            (ray_bundle.origins.shape[0], ray_bundle.origins.shape[1], 3),
            dtype=torch.float32,
        )
        rgb[valid_mask] = self.sphere_rgb

        depth = (
            torch.ones(
                (ray_bundle.origins.shape[0], ray_bundle.origins.shape[1]),
                dtype=torch.float32,
            )
            * torch.inf
        )
        depth[valid_mask] = t[valid_mask]

        accumulation = torch.zeros_like(depth, dtype=torch.float32)
        accumulation[valid_mask] = 1

        return {
            "rgb": rgb,
            "depth": depth,
            "accumulation": accumulation,
        }


class CylinderNeRF(NeRF):
    def __init__(
        self,
        cylinder_center: Optional[torch.Tensor] = None,
        cylinder_radius: Optional[float] = 0.25,
        cylinder_height: Optional[float] = 3,
        cylinder_body_rgb: Optional[torch.Tensor] = None,
        cylinder_top_rgb: Optional[torch.Tensor] = None,
        cylinder_bottom_rgb: Optional[torch.Tensor] = None,
    ) -> None:
        self.cylinder_center = (
            cylinder_center
            if cylinder_center is not None
            else torch.tensor([0, 0, 0], dtype=torch.float32)
        )
        self.cylinder_radius = cylinder_radius
        self.cylinder_height = cylinder_height
        self.cylinder_body_rgb = (
            cylinder_body_rgb
            if cylinder_body_rgb is not None
            else torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
        )
        self.cylinder_top_rgb = (
            cylinder_top_rgb
            if cylinder_top_rgb is not None
            else torch.tensor([0, 1, 0], dtype=torch.float32)
        )
        self.cylinder_bottom_rgb = (
            cylinder_bottom_rgb
            if cylinder_bottom_rgb is not None
            else torch.tensor([1, 0, 0], dtype=torch.float32)
        )

    def _find_intersection_cylinder(
        self,
        intersection_type: str,
        ray_bundle: RayBundle,
        cylinder_center: torch.Tensor,
        cylinder_radius: float,
        cylinder_height: float,
        cylinder_direction: Optional[torch.Tensor] = None,
        cylinder_direction_origin: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Finds intersection points between rays and a cylinder.
        """
        assert (
            cylinder_direction is None
        ), "Implementation of cylinder_direction not yet implemented; would need to update this function"
        assert (
            cylinder_direction_origin is None
        ), "Implementation of cylinder_direction_origin not yet implemented"

        o = ray_bundle.origins
        d = ray_bundle.directions

        if intersection_type == "body":
            a = d[..., 0] ** 2 + d[..., 2] ** 2
            b = 2 * (o[..., 0] * d[..., 0] + o[..., 2] * d[..., 2])
            c = o[..., 0] ** 2 + o[..., 2] ** 2 - cylinder_radius**2
            discriminant = b**2 - 4 * a * c

            t = (-b - torch.sqrt(discriminant)) / (2 * a)
            t[discriminant < 0] = torch.inf

            p_int = o + t[..., None] * d

            # if the y value is outside h/2 and -h/2, set t to torch.inf
            t[torch.where(p_int[..., 1] > (cylinder_height / 2))] = torch.inf
            t[torch.where(p_int[..., 1] < -(cylinder_height / 2))] = torch.inf

        elif intersection_type == "top":
            # check for intersection with the plane of y = h/2
            t = (cylinder_height / 2 - o[..., 1]) / d[..., 1]
            t[d[..., 1] == 0] = torch.inf
            p_int = o + t[..., None] * d
            # check if point of intersection is within the radius of the cylinder
            dist_p_int_cyl = torch.sqrt(p_int[..., 0] ** 2 + p_int[..., 2] ** 2)
            t[dist_p_int_cyl > cylinder_radius] = torch.inf
        elif intersection_type == "bottom":
            # check for intersection with the plane of y = -h/2
            t = (-cylinder_height / 2 - o[..., 1]) / d[..., 1]
            t[d[..., 1] == 0] = torch.inf
            p_int = o + t[..., None] * d
            # check if point of intersection is within the radius of the cylinder
            dist_p_int_cyl = torch.sqrt(p_int[..., 0] ** 2 + p_int[..., 2] ** 2)
            t[dist_p_int_cyl > cylinder_radius] = torch.inf
        else:
            raise ValueError(f"Unknown intersection type {intersection_type}")

        # set to inf if t is not inbody near/far
        if ray_bundle.nears is not None:
            t[t < ray_bundle.nears] = torch.inf
        if ray_bundle.fars is not None:
            t[t > ray_bundle.fars] = torch.inf

        assert not torch.isnan(t).any().item(), "t contains nans"
        return p_int, t

    def _get_ray_outputs(
        self,
        ray_bundle: RayBundle,
        cylinder_center: torch.Tensor,
        cylinder_radius: float,
        cylinder_height: float,
        cylinder_direction: Optional[torch.Tensor] = None,
        cylinder_direction_origin: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        _, t_int_body = self._find_intersection_cylinder(
            "body",
            ray_bundle,
            cylinder_center,
            cylinder_radius,
            cylinder_height,
            cylinder_direction,
            cylinder_direction_origin,
        )
        _, t_int_top = self._find_intersection_cylinder(
            "top",
            ray_bundle,
            cylinder_center,
            cylinder_radius,
            cylinder_height,
            cylinder_direction,
            cylinder_direction_origin,
        )
        _, t_int_bottom = self._find_intersection_cylinder(
            "bottom",
            ray_bundle,
            cylinder_center,
            cylinder_radius,
            cylinder_height,
            cylinder_direction,
            cylinder_direction_origin,
        )

        # compute the rgb and depth of each ray in the bundle
        # rgb is the color of the ray with the closest intersection
        rgb = torch.ones_like(ray_bundle.origins)
        depth = (
            torch.ones((ray_bundle.origins.shape[0], ray_bundle.origins.shape[1]))
            * torch.inf
        )
        accumulation = torch.ones_like(depth)

        # compute color arrays for each intersection type
        color_int_body = self.cylinder_body_rgb[None, None, :].repeat(
            rgb.shape[0], rgb.shape[1], 1
        )
        color_int_top = self.cylinder_top_rgb[None, None, :].repeat(
            rgb.shape[0], rgb.shape[1], 1
        )
        color_int_bottom = self.cylinder_bottom_rgb[None, None, :].repeat(
            rgb.shape[0], rgb.shape[1], 1
        )

        # sort by t_int and set color corresponding to closest intersection
        t_int = torch.cat(
            [t_int_body[..., None], t_int_top[..., None], t_int_bottom[..., None]],
            dim=-1,
        )

        # closest t_int has determines color
        min_indices = torch.argmin(t_int, dim=2)

        color = torch.cat(
            [
                color_int_body[..., None],
                color_int_top[..., None],
                color_int_bottom[..., None],
            ],
            dim=-1,
        )

        N, M = ray_bundle.origins.shape[:2]
        index_N = torch.arange(N).view(-1, 1).expand(N, M)
        index_M = torch.arange(M).view(1, -1).expand(N, M)

        rgb = color[index_N, index_M, :, min_indices]
        t_int = t_int[index_N, index_M, min_indices]

        # if any of the t_ints for a given ray is inf, set corresponding rgb color to all 1s
        rgb[~torch.isfinite(t_int)] = 1
        depth[torch.isfinite(t_int)] = t_int[torch.isfinite(t_int)]
        # depth[torch.]
        accumulation[torch.isfinite(t_int)] = 1

        return {
            "rgb": rgb,
            "depth": depth,
            "accumulation": accumulation,
        }


class CylinderWithTwoSpheresNeRF(NeRF):
    def __init__(
        self,
        sphere1_center: Optional[torch.Tensor] = None,
        sphere1_radius: Optional[float] = 0.5,
        sphere1_rgb: Optional[torch.Tensor] = None,
        sphere2_center: Optional[torch.Tensor] = None,
        sphere2_radius: Optional[float] = 0.5,
        sphere2_rgb: Optional[torch.Tensor] = None,
        cylinder_center: Optional[torch.Tensor] = None,
        cylinder_radius: Optional[float] = 0.25,
        cylinder_height: Optional[float] = 3,
        cylinder_body_rgb: Optional[torch.Tensor] = None,
        cylinder_top_rgb: Optional[torch.Tensor] = None,
        cylinder_bottom_rgb: Optional[torch.Tensor] = None,
    ) -> None:
        self.sphere1 = SphereNeRF(
            sphere_center=(
                sphere1_center
                if sphere1_center is not None
                else torch.tensor([0, cylinder_height / 2, 0], dtype=torch.float32)
            ),
            sphere_radius=sphere1_radius,
            rgb=sphere1_rgb
            if sphere1_rgb is not None
            else torch.tensor([0, 1, 0], dtype=torch.float32),
        )
        self.sphere2 = SphereNeRF(
            sphere_center=(
                sphere2_center
                if sphere2_center is not None
                else torch.tensor([0, -cylinder_height / 2, 0], dtype=torch.float32)
            ),
            sphere_radius=sphere2_radius,
            rgb=sphere2_rgb
            if sphere2_rgb is not None
            else torch.tensor([1, 0, 0], dtype=torch.float32),
        )
        self.cylinder = CylinderNeRF(
            cylinder_center=cylinder_center,
            cylinder_radius=cylinder_radius,
            cylinder_height=cylinder_height,
            cylinder_body_rgb=(
                cylinder_body_rgb
                if cylinder_body_rgb is not None
                else torch.tensor([0, 0, 0], dtype=torch.float32)
            ),
            cylinder_top_rgb=(
                cylinder_top_rgb
                if cylinder_top_rgb is not None
                else torch.tensor([0, 1, 0], dtype=torch.float32)
            ),
            cylinder_bottom_rgb=(
                cylinder_bottom_rgb
                if cylinder_bottom_rgb is not None
                else torch.tensor([1, 0, 0], dtype=torch.float32)
            ),
        )

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
        if rendered_output_names is None:
            rendered_output_names = ["rgb", "depth", "accumulation"]

        camera, ray_bundle = self._get_camera_and_ray_bundle(
            obj2w, c2w, cx, cy, fx, fy, height, width, camera_type
        )

        sphere1_outputs = self.sphere1._get_ray_outputs(
            ray_bundle, self.sphere1.sphere_center, self.sphere1.sphere_radius
        )
        sphere2_outputs = self.sphere2._get_ray_outputs(
            ray_bundle, self.sphere2.sphere_center, self.sphere2.sphere_radius
        )
        cylinder_outputs = self.cylinder._get_ray_outputs(
            ray_bundle,
            self.cylinder.cylinder_center,
            self.cylinder.cylinder_radius,
            self.cylinder.cylinder_height,
        )

        # Stack the RGB, depth, and accumulation tensors from the three inputs
        rgb_list = torch.stack(
            [sphere1_outputs["rgb"], sphere2_outputs["rgb"], cylinder_outputs["rgb"]]
        )
        disp_list = torch.stack(
            [
                1 / sphere1_outputs["depth"],
                1 / sphere2_outputs["depth"],
                1 / cylinder_outputs["depth"],
            ]
        )
        acc_list = torch.stack(
            [
                sphere1_outputs["accumulation"],
                sphere2_outputs["accumulation"],
                cylinder_outputs["accumulation"],
            ]
        )

        # Perform alpha compositing and obtain the output and alpha values
        obs, alph = alpha_composite(rgb_list, disp_list, acc_list)

        outputs = {}
        for output_name in rendered_output_names:
            if output_name == "rgb":
                outputs[output_name] = obs
            elif output_name == "depth":
                outputs[output_name] = (
                    1 / torch.sort(disp_list, dim=0, descending=True).values[0]
                )
            elif output_name == "accumulation":
                outputs[output_name] = alph
            else:
                raise ValueError(f"Unknown output name: {output_name}")

        if save_dir is not None:
            from nerfstudio.utils import plotly_utils as vis

            sphere2_depth_np = 1 / sphere2_outputs["depth"].cpu().numpy()
            cylinder_depth_np = 1 / cylinder_outputs["depth"].cpu().numpy()
            sphere2_depth_np = sphere2_outputs["depth"].cpu().numpy()
            cylinder_depth_np = cylinder_outputs["depth"].cpu().numpy()

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

            # Plot the depths for sphere2
            im1 = ax1.imshow(sphere2_depth_np, cmap="viridis")
            ax1.set_title("Depth for Sphere2")
            fig.colorbar(im1, ax=ax1)

            # Plot the depths for cylinder
            im2 = ax2.imshow(cylinder_depth_np, cmap="viridis")
            ax2.set_title("Depth for Cylinder")
            fig.colorbar(im2, ax=ax2)

            # Save the figure
            plt.savefig(save_dir / "depths.png")

            fig, ax = plt.subplots(1, len(rendered_output_names), figsize=(20, 5))
            for i, output_name in enumerate(rendered_output_names):
                ax[i].imshow(outputs[output_name].cpu().numpy())
                ax[i].set_title(output_name)
                fig.colorbar(ax[i].imshow(outputs[output_name].cpu().numpy()), ax=ax[i])

            plt.savefig("render_outputs.png")

            fig = vis.vis_camera_rays(camera)
            fig.show()

        return outputs


if __name__ == "__main__":
    near: float = 0.1
    far: float = 10.0
    cx: float = 48.0
    cy: float = 48.0
    fx: float = 48.0
    fy: float = 48.0
    height: Optional[int] = None
    width: Optional[int] = None
    camera_type = CameraType.PERSPECTIVE

    center: torch.Tensor = torch.tensor([-6.0, 0.0, 0.0], dtype=torch.float32)
    target: torch.Tensor = torch.tensor([0.0, 0, 0.0], dtype=torch.float32)
    up: torch.Tensor = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
    lookat = center - target
    c2w = viewmatrix(lookat, up, center)

    # Define camera positions around the cylinder
    num_frames = 36
    radius = 2

    camera_positions = []
    for i in range(num_frames):
        theta = 2 * np.pi * i / num_frames
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        z = 3
        center = torch.tensor([x, y, z], dtype=torch.float32)
        target = torch.tensor([0.0, 0.2, 0], dtype=torch.float32)
        up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
        lookat = center - target
        c2w = viewmatrix(lookat, up, center)
        camera_positions.append(c2w)

    outdir = pathlib.Path("test1/")
    outdir.mkdir(parents=True, exist_ok=True)

    cylinder_nerf = CylinderNeRF()
    cylinder_pose = torch.eye(4)[:3]

    # Render each frame from the camera positions
    images = []
    for i, c2w in enumerate(camera_positions):
        obs_cy = cylinder_nerf.render(
            cylinder_pose,
            c2w,
            cx,
            cy,
            fx,
            fy,
            height,
            width,
            camera_type,
        )
        # Store the RGB image in the list of images
        images.append(obs_cy["rgb"])
        filename = outdir / f"frame_{i:04d}.png"
        image = (obs_cy["rgb"].numpy() * 255).astype(np.uint8)
        imageio.imwrite(filename, image)

    # Create a gif from the list of images
    imageio.mimsave("cylinder.gif", images, fps=10)

    cylinder_sphere_nerf = CylinderWithTwoSpheresNeRF()
    cylinder_pose = torch.eye(4)[:3]

    # Render each frame from the camera positions
    images = []
    for i, c2w in enumerate(camera_positions):
        obs_cy = cylinder_sphere_nerf.render(
            cylinder_pose,
            c2w,
            cx,
            cy,
            fx,
            fy,
            height,
            width,
            camera_type,
        )
        # Store the RGB image in the list of images
        images.append(obs_cy["rgb"])
        filename = outdir / f"frame_{i:04d}.png"
        image = (obs_cy["rgb"].numpy() * 255).astype(np.uint8)
        imageio.imwrite(filename, image)

    # Create a gif from the list of images
    imageio.mimsave("cylinder_sphere.gif", images, fps=10)
    obs_cy = cylinder_sphere_nerf.render(
        cylinder_pose,
        c2w,
        cx,
        cy,
        fx,
        fy,
        height,
        width,
        camera_type,
    )
    obs_cy = cylinder_nerf.render(
        cylinder_pose,
        c2w,
        cx,
        cy,
        fx,
        fy,
        height,
        width,
        camera_type,
    )
    # Create a figure and set its size
    fig, axes = plt.subplots(1, len(obs_cy), figsize=(15, 5))

    # Loop through the dictionary and plot the images
    for ax, (key, value) in zip(axes, obs_cy.items()):
        if value.ndim == 3:  # RGB image
            ax.imshow(value)
        else:  # Grayscale image
            ax.imshow(value, cmap="gray")
        ax.set_title(key)
        ax.axis("off")

    # Save the figure to a file
    plt.savefig("all_obs.png", bbox_inches="tight")
