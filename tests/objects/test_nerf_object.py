import pathlib

import cv2
import numpy as np
import torch
from nerfstudio.cameras.cameras import CameraType

from demo_aug.objects.nerf_object import NeRFObject, NeRFObject3DSegWrapper

if __name__ == "__main__":
    # TODO(klin): after training, store metadata e.g. 3D segmentation mask (in the form of bounding box works too).
    # Note the bug w/ tensorf when bbox becomes too small
    # cfg_path = pathlib.Path(
    #     "~/autom/nerfstudio/outputs/robomimic_lift_2023-05-25/tensorf/2023-05-25_160044/config.yml"
    # ).expanduser()
    cfg_path = pathlib.Path(
        "/scr/thankyou/autom/demo-aug/outputs/nerfacto-v034-square-35-25-03-no-appear-embed--no-scale-or-orient-or-center/nerfacto/2024-03-25_192054/config.yml"
    )
    nerf_obj = NeRFObject(
        cfg_path,
        bounding_box_min=torch.tensor([-0.3, -0.3, 0.7]),
        bounding_box_max=torch.tensor([0.3, 0.3, 1.25]),
    )
    height = 512
    width = 512
    fl_x = 618.0386719675123
    fl_y = 618.0386719675123
    cx = 256.0
    cy = 256.0
    c2w = torch.tensor(
        [
            [
                -0.9932743906974792,
                -0.09902417659759521,
                0.060000598430633545,
                0.03497029095888138,
            ],
            [
                0.1157836765050888,
                -0.8494995832443237,
                0.5147275328636169,
                0.30000001192092896,
            ],
            [
                1.4901161193847656e-08,
                0.5182128548622131,
                0.855251669883728,
                1.3284684419631958,
            ],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    nerf_obj = NeRFObject3DSegWrapper(
        nerf_obj, torch.tensor([-0.4, -0.4, 0.7]), torch.tensor([0.4, 0.4, 1.155])
    )
    outputs = nerf_obj.render(
        torch.eye(4),
        c2w=c2w[:3],
        fx=fl_x,
        fy=fl_y,
        cx=cx,
        cy=cy,
        height=height,
        width=width,
        camera_type=CameraType.PERSPECTIVE,
        upsample_then_downsample=False,  # True causes a bug w interpolating
    )
    output_dir = pathlib.Path("aa_images_0.2")
    output_dir.mkdir(parents=True, exist_ok=True)
    # Save the rendered images
    for key, image in outputs.items():
        # Convert the image to PIL format
        if key == "rgb":
            image *= 255
            image = cv2.cvtColor(
                image.cpu().numpy().astype(np.uint8), cv2.COLOR_RGB2BGR
            )
        elif key == "depth":
            image = image.cpu().numpy()[..., 0]
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
        else:
            image = image.cpu().numpy()[..., 0]
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")

        file_path = output_dir / f"{key}.png"
        cv2.imwrite(str(file_path), image)
        print(f"Saved image to {file_path}")
