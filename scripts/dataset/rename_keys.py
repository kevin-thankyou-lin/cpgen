from dataclasses import dataclass, field
from typing import Dict, Optional

import tyro

from demo_aug.utils.data_collection_utils import (
    rename_keys,
)


@dataclass
class Config:
    input_file_path: str = "/scr/thankyou/autom/consistency-policy/data/robomimic/datasets/augmented/wine_glass_hanging/grasp/narrow/demo_05trials/2024-07-07/nullwine_glass_hanging_grasp_narrow_trials5_se3aug-dx-0.1-0.1-dy-0.1-0.1-dz0.0-0.0-dthetz0.0-0.0-biassampzrot_staug_joint-jointqposnoise0.06_defcurobocfg_CUROBO_bp4o80qx_45demos_resized.hdf5"
    output_file_path: Optional[str] = None
    old_to_new_names: Dict[str, str] = field(
        default_factory=lambda: {
            "12391924_left_image": "hand_camera_image",
            "27432424_left_image": "agentview_image",
            "robot0_eef_pos": "eef_pos",
            "robot0_eef_quat": "eef_quat",
            "robot0_gripper_qpos": "gripper_qpos",
        }
    )

    def __post_init__(self):
        if self.output_file_path is None:
            self.output_file_path = self.input_file_path.replace(
                ".hdf5", "_renamed.hdf5"
            )


def main(cfg: Config) -> None:
    rename_keys(cfg.input_file_path, cfg.output_file_path, cfg.old_to_new_names)


if __name__ == "__main__":
    tyro.cli(main)
