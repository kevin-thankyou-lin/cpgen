import logging
import pathlib  # noqa: E402
from dataclasses import asdict  # noqa: E402
from datetime import datetime  # noqa: E402
from typing import List, Set, Tuple, Union  # noqa: E402

import tyro  # noqa: E402

import wandb  # noqa: E402
from demo_aug.annotators.constraints import ConstraintAnnotator
from demo_aug.annotators.segmentation_masks import SegmentationMaskAnnotator
from demo_aug.augmentor.augmentor import Augmentor
from demo_aug.configs.base_config import ConstraintInfo, DemoAugConfig
from demo_aug.demo import Demo  # noqa: E402
from demo_aug.objects.reconstructor import ReconstructionManager, ReconstructionType
from demo_aug.utils.data_collection_utils import (
    load_demos,
    save_demos,
)
from demo_aug.utils.viz_utils import create_gifs_from_h5py_file  # noqa: E402

# Configure the root logger to write log messages to a file
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)  # noqa: E402


def update_logging_filename(new_filename):
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            root_logger.removeHandler(handler)
            handler.close()

    new_file_handler = logging.FileHandler(new_filename)
    new_file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    root_logger.addHandler(new_file_handler)


logging.basicConfig(level=logging.INFO)


def determine_reconstruction_timesteps(
    constraint_infos: List[ConstraintInfo],
):
    """Determine which timesteps require 3D reconstructions based on user input and existing data;
    ignores redundant reconstructions, for example, if we're using a single static NeRF model for multiple
    times.

    TODO: move elsewhere; maybe not Reconstructor class to keep it clean and not rely on ConstraintInfos
    """
    reconstruction_timesteps: Union[List, Set[Tuple[int]]] = set()
    for constraint in constraint_infos:
        reconstruction_timestep_for_constraint = (
            constraint.collect_reconstruction_timesteps()
        )
        reconstruction_timesteps.update(reconstruction_timestep_for_constraint)
    reconstruction_timesteps_list = list(reconstruction_timesteps)
    return reconstruction_timesteps_list


def main(demo_aug_cfg: DemoAugConfig) -> None:
    if demo_aug_cfg.use_wandb:
        wandb.init(project="gen-aug-script", entity="kevin-lin")
        # append wandb.run.id to demo_aug_cfg.save_file_name
        curr_save_path = pathlib.Path(demo_aug_cfg.save_file_name)
        demo_aug_cfg.save_file_name = (
            f"{curr_save_path.stem}_{wandb.run.id + curr_save_path.suffix}"
        )
        wandb.config.update(asdict(demo_aug_cfg))

    src_demos: List[Demo] = load_demos(demo_aug_cfg.demo_path)
    if "robomimic" in str(demo_aug_cfg.demo_path):
        import robomimic.utils.file_utils as FileUtils

        env_cfg = FileUtils.get_env_metadata_from_dataset(
            dataset_path=demo_aug_cfg.demo_path
        )
        env_info = str(env_cfg)
    else:
        # handling real world dataset
        env_cfg = demo_aug_cfg.env_cfg
        env_info = str(env_cfg)

    # env_info contains robot + camera (intrinsics) info + camera frame transform w.r.t robot EE frame
    # helps figure out corresponding robot action post obj pose transform
    # env_info may be good to load into each demo as an entry then?
    # dataset should contain information include the transforms betweens cameras and robot
    for src_demo in src_demos:
        src_demo_name = str(src_demo.name).split(".")[0]
        generated_demos_save_path = (
            demo_aug_cfg.save_base_dir
            / pathlib.Path(
                src_demo_name + f"{demo_aug_cfg.trials_per_constraint}trials"
            )
            / datetime.now().strftime("%Y-%m-%d")
            / wandb.run.id
            / demo_aug_cfg.save_file_name
            if demo_aug_cfg.use_wandb
            else pathlib.Path("debug") / demo_aug_cfg.save_file_name
        )

        # create directory if it doesn't exist
        generated_demos_save_path.parent.mkdir(parents=True, exist_ok=True)
        update_logging_filename(str(generated_demos_save_path.parent / "app.log"))

        constraints = ConstraintAnnotator.get_constraints(src_demo, demo_aug_cfg)

        rec_manager = ReconstructionManager(str(src_demo.demo_path), src_demo.name)

        reconstruction_ts_list = determine_reconstruction_timesteps(constraints)
        # couldn't give demo_aug_cfg key as a tuple on the CLI, so manually converting here
        timestep_to_nerf_folder = {
            str(key): value
            for key, value in demo_aug_cfg.timestep_to_nerf_folder.items()
        }
        for ts in reconstruction_ts_list:
            rec_manager.reconstruct(ts, timestep_to_nerf_folder)

        # code for getting all relevant times
        for constraint in constraints:
            for (
                rec_ts,
                obj_name,
            ) in constraint.collect_reconstruction_timesteps_to_obj_name().items():
                segmentation_mask = SegmentationMaskAnnotator.get_segmentation_masks(
                    src_demo, demo_aug_cfg, rec_manager, rec_ts, obj_name
                )
                # my sense is that these functions are more segmentation and/or exporting, rather than reconstructing
                rec_manager.reconstruct(
                    rec_ts,
                    timestep_to_nerf_folder,
                    obj_name,
                    segmentation_mask,
                    ReconstructionType.NeRF,
                )
                rec_manager.reconstruct(
                    rec_ts,
                    timestep_to_nerf_folder,
                    obj_name,
                    segmentation_mask,
                    ReconstructionType.GaussianSplat,
                )
                rec_manager.reconstruct(
                    rec_ts,
                    timestep_to_nerf_folder,
                    obj_name,
                    segmentation_mask,
                    ReconstructionType.Mesh,
                )

        src_demo.add_reconstruction_manager(rec_manager)
        # adding constraint infos to demos can be removed once we use the "augmentor" by passing
        # the constraints as an argument
        src_demo.add_constraint_infos(constraints)

        generated_demos: List[Demo] = Augmentor.generate_augmented_demos(
            src_demo, demo_aug_cfg, env_cfg, constraints
        )

        print(
            f"generated {len(generated_demos)} demos; expected: {demo_aug_cfg.trials_per_constraint}"
        )
        src_demo_name = str(src_demo.name).split(".")[0]
        generated_demos_save_path = (
            demo_aug_cfg.save_base_dir
            / pathlib.Path(
                src_demo_name + f"{demo_aug_cfg.trials_per_constraint}trials"
            )
            / datetime.now().strftime("%Y-%m-%d")
            / (wandb.run.id if demo_aug_cfg.use_wandb else pathlib.Path("debug"))
            / demo_aug_cfg.save_file_name
        )

        save_demos(
            generated_demos,
            generated_demos_save_path,
            env_info,  # env info is really just env config and is used in diffusion policy to reset the evaluation env!
            env_cfg,
            camera_names=demo_aug_cfg.env_cfg.multi_camera_cfg.camera_names,
        )
        print(f"saved to {generated_demos_save_path}")
        break

    save_dir = generated_demos_save_path.parent
    save_dir.mkdir(parents=True, exist_ok=True)

    create_gifs_from_h5py_file(
        save_dir / demo_aug_cfg.save_file_name,
        save_dir,
        [
            cam_name + "_image"
            for cam_name in demo_aug_cfg.env_cfg.multi_camera_cfg.camera_names
        ],
    )


if __name__ == "__main__":
    # load demonstrations file
    tyro.cli(main)
