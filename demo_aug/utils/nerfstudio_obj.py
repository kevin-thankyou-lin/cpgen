"""
Class for rendering NeRFStudio trained objects.
"""

import sys
from typing import List, Tuple

import numpy as np
import torch
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.models.base_model import BaseModel
from rich.console import Console

CONSOLE = Console(width=120)


class NeRFStudioObject:
    model: BaseModel
    aabb_box: SceneBox = None

    def render(
        self,
        obj2w: torch.Tensor,  # 3 x 4 object-to-world transformation matrix
        c2w: torch.Tensor,
        cx: float,
        cy: float,
        fx: float,
        fy: float,
        camera_type: CameraType,
        rendered_output_names: List[
            str
        ],  # something like ["rgb", "depth", "accumulation"]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # should be variable length
        """
        Renders the "scene" that consists only the current NeRF from camera pose c2w and
        assuming the object has pose obj2w.

        Args:
            c2w: Camera-to-world matrix.
            rendered_output_names: List of output names to render.

        Returns:
            tensors specified by rendered_output_names
        """
        # Transform the c2w matrix according to the given obj2w pose.
        R_obj, t_obj = obj2w[:3, :3], obj2w[:3, 3:]
        R_cam, t_cam = c2w[:3, :3], c2w[:3, 3:]
        render_c2w_R = torch.matmul(R_obj, R_cam)  # Rotate
        render_c2w_t = t_obj - torch.matmul(R_cam.T, t_cam)  # Translate
        render_c2w = torch.cat((render_c2w_R, render_c2w_t.unsqueeze(-1)), dim=-1)
        # check if shape is (3, 4)
        assert render_c2w.shape == (3, 4)

        camera = Cameras(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            camera_to_worlds=render_c2w,
            camera_type=camera_type,
        )

        # Generate rays from the camera at the given pose.
        camera_ray_bundle = camera.generate_rays(camera_indices=0, aabb_box=None)

        # Query, or "render", the object model and return the results.
        with torch.no_grad():
            outputs = self.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
            render_image = []
            for rendered_output_name in rendered_output_names:
                if rendered_output_name not in outputs:
                    CONSOLE.rule("Error", style="red")
                    CONSOLE.print(
                        f"Could not find {rendered_output_name} in the model outputs",
                        justify="center",
                    )
                    CONSOLE.print(
                        f"Please set --rendered_output_name to one of: {outputs.keys()}",
                        justify="center",
                    )
                    sys.exit(1)
                output_image = outputs[rendered_output_name].cpu().numpy()
                if output_image.shape[-1] == 1:
                    output_image = np.concatenate((output_image,) * 3, axis=-1)
                render_image.append(output_image)

        return render_image


# TODO(klin): loading in a trained nerf model and render object; first use nerfstudio itself
# to visualize spiral model


def entrypoint():
    CONSOLE.rule("Hello, world!", style="green")


if __name__ == "__main__":
    entrypoint()
