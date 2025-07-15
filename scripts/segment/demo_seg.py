"""
SquareReal
python scripts/segment/demo_seg.py --dataset datasets/source/square_real_3_6.hdf5 --interaction-threshold 0.15 --interactions gripper0_right_right_gripper:SquareNut_main,SquareNut_main:peg1

Stack3Real
python scripts/segment/demo_seg.py --dataset datasets/source/stack_three_real.hdf5  --interaction-threshold 0.03 --interactions gripper0_right_right_gripper:cubeC_main,cubeC_main:cubeA_main,gripper0_right_right_gripper:cubeB_main,cubeB_main:cubeC_main

Threading
# manually choose second object's segmentation time range --- also don't use open-gripper for post constraint action by default
python scripts/segment/demo_seg.py --dataset datasets/source/threading.hdf5  --interaction-threshold 0.02 --interactions gripper0_right_right_gripper:needle_obj_root,needle_obj_root:tripod_obj_root

Stack3
python scripts/segment/demo_seg.py --dataset datasets/source/stack_three.hdf5  --interaction-threshold 0.02 --interactions gripper0_right_right_gripper:cubeA_main,cubeA_main:cubeB_main,gripper0_right_right_gripper:cubeC_main,cubeC_main:cubeA_main

3PA
python scripts/segment/demo_seg.py --dataset datasets/source/three_piece_assembly.hdf5  --interaction-threshold 0.03 --interactions gripper0_right_right_gripper:piece_1_root,piece_1_root:base_root,gripper0_right_right_gripper:piece_2_root,piece_2_root:piece_1_root

# Needs manual effort current due to the way obj-obj end is defined: edge case is when another object is inside object we're manipulating
Kitchen # TODO; maybe update interaction end to not be dependent on motion planning? we can handle motion planning else where? or maybe not ... let's ignore kitchen for now
python scripts/segment/demo_seg.py --dataset datasets/source/kitchen.hdf5  --interaction-threshold 0.02 --interactions gripper0_right_rightfinger:Button1_main,gripper0_right_rightfinger:PotObject_root,PotObject_root:Stove1_main,gripper0_right_rightfinger:cube_bread_main,cube_bread_main:PotObject_root,gripper0_right_rightfinger:PotObject_root,PotObject_root:ServingRegionRed_main,gripper0_right_rightfinger:Button1_main

Coffee
python scripts/segment/demo_seg.py --dataset datasets/source/coffee.hdf5  --interaction-threshold 0.02 --interactions gripper0_right_right_gripper:coffee_pod_main,coffee_pod_main:coffee_machine_root,gripper0_right_right_gripper:coffee_machine_root

MugCleanup
python scripts/segment/demo_seg.py --dataset datasets/source/mug_cleanup.hdf5 --interaction-threshold 0.03 --interactions gripper0_right_right_gripper:DrawerObject_main,gripper0_right_right_gripper:cleanup_object_main,cleanup_object_main:DrawerObject_main,gripper0_right_right_gripper:DrawerObject_main

# Needs manual effort, esp in Symmetry and adding extra lift steps to enable motion planning collision-free-ness
HammerCleanup
python scripts/segment/demo_seg.py --dataset datasets/source/hammer_cleanup.hdf5  --interaction-threshold 0.04 --interaction-interaction-t-gap 10 --interactions  gripper0_right_right_gripper:CabinetObject_main,gripper0_right_right_gripper:hammer_root,hammer_root:CabinetObject_main,gripper0_right_right_gripper:CabinetObject_main

python scripts/segment/demo_seg.py --dataset datasets/source/hammer_cleanup_real.hdf5  --interaction-threshold 0.04 --interaction-interaction-t-gap 10 --interactions  gripper0_right_right_gripper:CabinetObject_main,gripper0_right_right_gripper:hammer_root,hammer_root:CabinetObject_main,gripper0_right_right_gripper:CabinetObject_main


New scripts:
python scripts/segment/demo_seg.py --dataset datasets/source/mug_cleanup.hdf5  --segmentation-type llm-e2e  --interaction-threshold 0.03 --interactions  gripper0_right_right_gripper:DrawerObject_main,gripper0_right_right_gripper:cleanup_object_main,cleanup_object_main:DrawerObject_main,gripper0_right_right_gripper:DrawerObject_main --lang-description "Open drawer, pick and place mug into drawer, close drawer"

python scripts/segment/demo_seg.py --dataset datasets/source/hammer_cleanup.hdf5  --segmentation-type llm-e2e  --interaction-threshold 0.03 --interactions  gripper0_right_right_gripper:CabinetObject_main,gripper0_right_right_gripper:cleanup_object_main,cleanup_object_main:CabinetObject_main,gripper0_right_right_gripper:CabinetObject_main --lang-description "Open drawer, pick hammer, place hammer into drawer, close drawer"

python scripts/segment/demo_seg.py --dataset datasets /source/stack_three.hdf5  --segmentation-type llm-success  --interaction-threshold 0.03 --lang-description "Pick red cube, place red cube, pick blue cube, place blue cube"


python scripts/segment/demo_seg.py --dataset datasets/source/threading.hdf5  --segmentation-type llm-success  --interaction-threshold 0.03 --lang-description "Pick needle, insert needle into tripod hole"

python scripts/segment/demo_seg.py --dataset datasets/source/coffee.hdf5  --segmentation-type llm-success  --interaction-threshold 0.03 --lang-description "Pick pod, place pod into coffee machine, close coffee machine lid"

python scripts/segment/demo_seg.py --dataset datasets/source/coffee.hdf5  --segmentation-type llm-success  --interaction-threshold 0.03 --lang-description "Pick pod, place pod into coffee machine, close coffee machine lid"

# 3pa
# gripper0_right_right_gripper:piece_1_root,piece_1_root:base_root,gripper0_right_right_gripper:piece_2_root,piece_2_root:piece_1_root
python scripts/segment/demo_seg.py --dataset datasets/source/three_piece_assembly.hdf5  --segmentation-type llm-success  --interaction-threshold 0.03 --lang-description "Pick piece 1, place piece 1, pick piece 2, place piece 2"

# kitchen
# gripper0_right_rightfinger:Button1_main,gripper0_right_rightfinger:PotObject_root,PotObject_root:Stove1_main,gripper0_right_rightfinger:cube_bread_main,cube_bread_main:PotObject_root,gripper0_right_rightfinger:PotObject_root,PotObject_root:ServingRegionRed_main,gripper0_right_rightfinger:Button1_main
python scripts/segment/demo_seg.py --dataset datasets/source/kitchen.hdf5  --segmentation-type llm-success  --interaction-threshold 0.03 --lang-description "Switch button, pick pot, place pot, pick bread, place bread, pick pot, place pot, switch button"

python scripts/segment/demo_seg.py --dataset datasets/source/pouring.hdf5  --segmentation-type llm-success  --interaction-threshold 0.03 --lang-description "Pick mug, pour ball into bowl, place mug"python scripts/segment/demo_seg.py --dataset datasets/source/pouring.hdf5  --segmentation-type llm-success  --interaction-threshold 0.03 --lang-description "Pick mug, pour ball into bowl, place mug"

python scripts/segment/demo_seg.py --dataset datasets/source/pouring.hdf5  --segmentation-type distance-based  --interaction-threshold 0.03 --lang-description "Pick mug, pour ball into bowl, place mug"python scripts/segment/demo_seg.py --interaction-threshold 0.03 --lang-description "Pick mug, pour ball into bowl, place mug"


"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import h5py
import imageio
import numpy as np
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robosuite
import cpgen_envs

# monkey patch robomimic's EnvBase.reset_to with the above function
from robomimic.envs.env_robosuite import EnvRobosuite

from demo_aug.utils.demo_segmentation_utils import (
    create_constraint,
    decompose_trajectory,
    get_interactions_from_llm,
    parse_interactions,
    run_llm_e2e_segmentation,
    run_llm_success_segmentation,
    run_llm_two_phase_single_prompt,
)
from demo_aug.utils.mujoco_utils import get_body_children, get_top_level_body_names
from demo_aug.utils.robosuite_utils import refactor_composite_controller_config


def reset_to(self, state, no_return_obs: bool = False):
    """
    Reset to a specific simulator state.

    Args:
        state (dict): current simulator state that contains one or more of:
            - states (np.ndarray): initial state of the mujoco environment
            - model (str): mujoco scene xml
        no_return_obs (bool): if True, do not return observation after setting the simulator state.
            Used to not waste computation when we don't need the observation.
            If False, return observation after setting the simulator state.
    Returns:
        observation (dict): observation dictionary after setting the simulator state (only
            if "states" is in @state)
    """
    should_ret = False
    if "model" in state:
        self.reset()
        robosuite_version_id = int(robosuite.__version__.split(".")[1])
        if robosuite_version_id <= 3:
            from robosuite.utils.mjcf_utils import postprocess_model_xml

            xml = postprocess_model_xml(state["model"])
        else:
            # v1.4 and above use the class-based edit_model_xml function
            xml = self.env.edit_model_xml(state["model"])

        from robosuite.utils.binding_utils import MjSim

        # first try to reset using MjSim
        try:
            MjSim.from_xml_string(xml)
            self.env.reset_from_xml_string(xml)
        except Exception as e:
            print(
                f"Error in reset_to; skip updating xml, since not possible to set to specified xml: {e}"
            )
        self.env.sim.reset()
        if not self._is_v1:
            # hide teleop visualization after restoring from model
            self.env.sim.model.site_rgba[self.env.eef_site_id] = np.array(
                [0.0, 0.0, 0.0, 0.0]
            )
            self.env.sim.model.site_rgba[self.env.eef_cylinder_id] = np.array(
                [0.0, 0.0, 0.0, 0.0]
            )
    if "states" in state:
        self.env.sim.set_state_from_flattened(state["states"])
        self.env.sim.forward()
        should_ret = True

    if "goal" in state:
        self.set_goal(**state["goal"])
    if not no_return_obs and should_ret:
        # only return obs if we've done a forward call - otherwise the observations will be garbage
        return self.get_observation()
    return None


EnvRobosuite.reset_to = reset_to


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to hdf5 dataset"
    )
    parser.add_argument(
        "--interaction-threshold",
        type=float,
        default=0.03,
        help="Distance threshold for all interactions",
    )
    #     interaction_interaction_t_gap: int = 3,
    parser.add_argument(
        "--interaction-interaction-t-gap",
        type=int,
        default=3,
        help="Minimum number of time steps between interactions",
    )  # heuristc to avoid one interaction happening way too close to the previous
    # due to how demo was performed / body distances
    parser.add_argument(
        "--interactions",
        type=str,
        required=False,
        help="Comma-separated list of interactions in the form 'entity1:entity2'. "
        "For robot-object, prefix the robot entity with 'gripper0_' (e.g., 'gripper0_eef:object').",
    )

    # language description
    parser.add_argument(
        "--lang-description",
        type=str,
        default="",
        help="Natural language description of the task. Used for llm-e2e segmentation.",
    )
    parser.add_argument(
        "--segments-output-dir",
        type=str,
        default="",
        help="Directory relative to dataset path's (parent directory / dataset_path's file name) to save segment renderings",
    )
    parser.add_argument(
        "--segmentation-type",
        type=str,
        choices=[
            "distance-based",
            "llm-e2e",
            "llm-success",
            "llm-success-start-single",
        ],
        default="distance-based",
        help="Type of segmentation to perform. Currently only 'distance-based' is supported. "
        "Note that distance based also uses knowledge of next interaction to determine "
        "when to end current interaction segment.",
    )
    parser.add_argument(
        "--no-save-segment-videos",
        action="store_true",
        help="Do not create and save videos for each segment",
    )
    parser.add_argument(
        "--print-bodies",
        action="store_true",
        help="Print all available body names in the environment",
    )
    parser.add_argument(
        "--force-overwrite",
        action="store_true",
        help="Force overwrite constraints file if it already exists",
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    segments_output_dir = (
        (dataset_path.parent / dataset_path.stem / args.segments_output_dir)
        if args.segments_output_dir
        else (
            dataset_path.parent
            / dataset_path.stem
            / f"{args.segmentation_type}-segments"
        )
    )

    # add time stamp to output directory
    segments_output_dir = segments_output_dir / datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )
    segments_output_dir.mkdir(exist_ok=True, parents=True)

    # create env and and get states
    with h5py.File(dataset_path, "r") as f:
        import robomimic.utils.obs_utils as ObsUtils

        dummy_spec = dict(
            obs=dict(
                low_dim=["robot0_eef_pos"],
                rgb=[],
            ),
        )
        ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=dummy_spec)

        env_meta = FileUtils.get_env_metadata_from_dataset(
            dataset_path=str(dataset_path)
        )
        if "env_name" in env_meta["env_kwargs"]:
            del env_meta["env_kwargs"]["env_name"]
        try:
            original_controller_configs = env_meta["env_kwargs"][
                "controller_configs"
            ].copy()
            env_meta["env_kwargs"]["controller_configs"] = (
                refactor_composite_controller_config(
                    env_meta["env_kwargs"]["controller_configs"],
                    robot_type="panda",
                    arms=["right"],
                )
            )
            env = EnvUtils.create_env_from_metadata(
                env_meta=env_meta,
                render=False,
                render_offscreen=True,
                use_image_obs=True,
            )
        except Exception as e:
            print(
                f"Error in creating env: {e} with refactored controller configs. Trying with original controller configs"
            )
            env_meta["env_kwargs"]["controller_configs"] = original_controller_configs
            env = EnvUtils.create_env_from_metadata(
                env_meta=env_meta,
                render=False,
                render_offscreen=True,
                use_image_obs=True,
            )
        demo = list(f["data"].keys())[0]
        states = f[f"data/{demo}/states"][()]
        update_original_demo = False
        if update_original_demo:
            states = np.concatenate(
                [states, np.repeat(states[-1][None], 10, axis=0)], axis=0
            )  # Extend for better viz
        initial_state = {
            "states": states[0],
            "model": f[f"data/{demo}"].attrs["model_file"],
        }
        env.reset()
        env.reset_to(initial_state)

    if args.print_bodies:
        print("Bodies in the environment:")
        for i, body in enumerate(env.env.sim.model.body_names):
            print(f"{i}: {body}")
        return

    # If language description is provided but no interactions, use LLM to generate interactions
    if args.lang_description and not args.interactions:
        print("Using LLM to generate interactions from language description...")
        # Get top level body names for the LLM to use
        NON_RELEVANT_TOP_LEVEL_BODY_NAME_PREFIXES = [
            "world",
            "left_eef_target",
            "right_eef_target",
            "table",
            "robot0",
        ]
        relevant_body_names = get_top_level_body_names(
            env.env.sim.model._model,
            exclude_prefixes=NON_RELEVANT_TOP_LEVEL_BODY_NAME_PREFIXES,
        )
        relevant_body_names += get_body_children(env.env.sim.model._model, "table")
        RELEVANT_ROBOT_BODY_NAMES = ["gripper0_right_right_gripper"]
        relevant_body_names += RELEVANT_ROBOT_BODY_NAMES

        interactions = get_interactions_from_llm(
            args.lang_description, relevant_body_names
        )

        # Parse the returned interactions
        if interactions:
            print(f"LLM generated interactions: {interactions}")
        else:
            print(
                "Failed to parse interactions pairs from language description. Please specify interactions manually."
            )
            print(f"Relevant body names: {relevant_body_names}")
            return
    else:
        interactions = args.interactions.split(",")

    segments = []
    if args.segmentation_type == "distance-based":
        # decompose trajectory into segments
        segments = decompose_trajectory(
            env,
            states,
            args.interaction_threshold,
            parse_interactions(",".join(interactions)),
            segments_output_dir,
            interaction_interaction_t_gap=args.interaction_interaction_t_gap,
        )
    elif args.segmentation_type == "llm-e2e":
        segments = run_llm_e2e_segmentation(
            env,
            states,
            args.lang_description,
            segments_output_dir,
            ",".join(interactions),
            llm_log_dir=segments_output_dir,
        )
    elif args.segmentation_type == "llm-success":
        segments = run_llm_success_segmentation(
            env,
            states,
            args.lang_description,
            segments_output_dir,
            ",".join(interactions),
            llm_log_dir=segments_output_dir,
        )
    elif args.segmentation_type == "llm-success-start-single":
        segments = run_llm_two_phase_single_prompt(
            env,
            states,
            args.lang_description,
            segments_output_dir,
            ",".join(interactions),
            llm_log_dir=segments_output_dir,
        )
    num_segments = len(segments)
    print(f"Number of segments: {num_segments}")

    # get robot gripper type from env
    gripper_type = env.env.robots[0].gripper["right"].__class__.__name__

    # create constraints for each segment
    constraints: List[Dict] = []
    for i, (label, time_range) in enumerate(segments):
        if label != "motion":
            constraints.append(
                create_constraint(label, time_range, gripper_type=gripper_type)
            )

    if not args.no_save_segment_videos:
        # save videos for each segment by loading src video
        source_video_path = (
            dataset_path.parent
            / dataset_path.stem
            / "videos"
            / f"{dataset_path.stem}_with_frame_number.mp4"
        )
        if not source_video_path.exists():
            print(f"Source video not found: {source_video_path}")
            return

        for i, (label, time_range) in enumerate(segments):
            if label != "motion":
                # load video
                video = imageio.get_reader(source_video_path, "ffmpeg")
                # save segment video
                segment_video_path = (
                    segments_output_dir
                    / f"{label}_{i}_{time_range[0]}-{time_range[1]}.mp4"
                )
                with imageio.get_writer(segment_video_path, fps=30) as writer:
                    for frame in range(time_range[0], time_range[1]):
                        img = video.get_data(frame)
                        writer.append_data(img)
                print(f"Saved segment video to {segment_video_path}")

    # save constraints to JSON
    # Add a timestamp to the constraints file to avoid overwriting previous files
    constraints_path = (
        dataset_path.parent
        / dataset_path.stem
        / f"{args.segmentation_type}-constraints.json"
    )

    # Check if the file exists and handle overwriting
    if constraints_path.exists() and not args.force_overwrite:
        overwrite = input(
            f"{constraints_path} already exists. Do you want to overwrite it? (y/n): "
        )
        if overwrite.lower() != "y":
            print("Constraints not saved.")
            return

    with open(constraints_path, "w") as f:
        json.dump(constraints, f, indent=4)

    print(f"Saved constraints to {constraints_path}")


if __name__ == "__main__":
    main()
