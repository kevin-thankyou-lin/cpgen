from dataclasses import dataclass, field
from typing import Dict, Tuple

import tyro

from demo_aug.utils.data_collection_utils import (
    print_hdf5_file_structure,
    rename_keys,
    resize_images_in_dataset,
)


@dataclass
class Config:
    input_file_path: str = "/scr/thankyou/autom/consistency-policy/data/robomimic/datasets/augmented/wine_glass_hanging/grasp/narrow/demo_01trials/2024-07-12/emtqpt1v/nullwine_glass_hanging_grasp_narrow_trials1_se3aug-dx-0.1-0.1-dy-0.1-0.1-dz0.0-0.0-dthetz0.0-0.0-biassampzrot_staug_joint-jointqposnoise0.06_defcurobocfg_CUROBO_emtqpt1v.hdf5"
    skip_key_prefix: str = "timestep_"
    image_key_resizing_height_width_map: Dict[str, Tuple[int, int]] = field(
        default_factory=lambda: {
            "27432424_left": (
                240,
                427,
            ),  # no need to resize for agentview; passing in as height x width
            "12391924_left": (84, 84),
        }
    )
    old_to_new_names: Dict[str, str] = field(
        default_factory=lambda: {
            "12391924_left_image": "hand_camera_image",
            "27432424_left_image": "agentview_image",
            "robot0_eef_pos": "eef_pos",
            "robot0_eef_quat": "eef_quat",
            "robot0_gripper_qpos": "gripper_qpos",
        }
    )


def main(cfg: Config) -> None:
    resize_output_file_path = cfg.input_file_path.replace(".hdf5", "_resized.hdf5")
    key_rename_output_file_path = resize_output_file_path.replace(
        ".hdf5", "_renamed.hdf5"
    )

    # first resize the images
    resize_images_in_dataset(
        cfg.input_file_path,
        resize_output_file_path,
        cfg.image_key_resizing_height_width_map,
    )

    # then rename the keys
    rename_keys(
        resize_output_file_path, key_rename_output_file_path, cfg.old_to_new_names
    )

    # then optionally view the structure of the generated HDF5 file
    print_hdf5_file_structure(
        key_rename_output_file_path, skip_key_prefix=cfg.skip_key_prefix
    )

    print(f"Processed dataset saved to {key_rename_output_file_path}")


if __name__ == "__main__":
    tyro.cli(main)
