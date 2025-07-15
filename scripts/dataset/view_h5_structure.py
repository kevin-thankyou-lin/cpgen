from dataclasses import dataclass

import tyro

from demo_aug.utils.data_collection_utils import print_hdf5_file_structure


@dataclass
class Config:
    file_path: str = "/scr/thankyou/autom/consistency-policy/data/robomimic/datasets/augmented/wine_glass_hanging/grasp/narrow/demo_05trials/2024-07-07/nullwine_glass_hanging_grasp_narrow_trials5_se3aug-dx-0.1-0.1-dy-0.1-0.1-dz0.0-0.0-dthetz0.0-0.0-biassampzrot_staug_joint-jointqposnoise0.06_defcurobocfg_CUROBO_bp4o80qx_45demos_46demos_resized.hdf5"
    skip_key_prefix: str = "timestep_"


def main(cfg: Config) -> None:
    """
    Main function to print the HDF5 file structure.
    """
    print_hdf5_file_structure(cfg.file_path, skip_key_prefix=cfg.skip_key_prefix)


if __name__ == "__main__":
    tyro.cli(main)
