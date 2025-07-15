from dataclasses import dataclass, field
from typing import Dict, Tuple

import tyro

from demo_aug.utils.data_collection_utils import resize_images_in_dataset


@dataclass
class Config:
    input_file_path: str = "/scr/thankyou/autom/consistency-policy/data/robomimic/datasets/augmented/wine_glass_hanging/grasp/narrow/demo_05trials/2024-07-07/nullwine_glass_hanging_grasp_narrow_trials5_se3aug-dx-0.1-0.1-dy-0.1-0.1-dz0.0-0.0-dthetz0.0-0.0-biassampzrot_staug_joint-jointqposnoise0.06_defcurobocfg_CUROBO_bp4o80qx_45demos.hdf5"
    output_file_path: str = "/scr/thankyou/autom/consistency-policy/data/robomimic/datasets/augmented/wine_glass_hanging/grasp/narrow/demo_05trials/2024-07-07/nullwine_glass_hanging_grasp_narrow_trials5_se3aug-dx-0.1-0.1-dy-0.1-0.1-dz0.0-0.0-dthetz0.0-0.0-biassampzrot_staug_joint-jointqposnoise0.06_defcurobocfg_CUROBO_bp4o80qx_45demos_resized.hdf5"
    image_key_resizing_map: Dict[str, Tuple[int, int]] = field(
        default_factory=lambda: {
            "27432424_left": (240, 427),  # no need to resize for agentview
            "12391924_left": (84, 84),
        }
    )


def main(cfg: Config) -> None:
    """
    Main function to resize images in the dataset.
    """
    resize_images_in_dataset(
        cfg.input_file_path, cfg.output_file_path, cfg.image_key_resizing_map
    )


if __name__ == "__main__":
    tyro.cli(main)
