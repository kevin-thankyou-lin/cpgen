import json
import pathlib
from pathlib import Path
from string import Template
from typing import Dict, List, Tuple

import h5py
import moviepy.editor as mp
import mujoco
import numpy as np
import openai
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
from moviepy.video.VideoClip import ImageClip
from PIL import Image, ImageDraw, ImageFont
from robosuite.controllers import load_composite_controller_config

from demo_aug.generate import Constraint, CPEnv
from demo_aug.utils.fm_utils import LLMCache
from demo_aug.utils.mujoco_utils import (
    check_geom_collisions,
    get_all_body_names,
    get_body_contact_pairs,
    get_body_joints_recursive,
    get_body_pose,
    get_body_velocity,
    get_joint_name_to_qpos,
    get_joint_name_to_qvel,
    get_subtree_geom_ids_by_group,
    get_top_level_bodies,
)


def get_trajectory(
    env,
    states,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    obj_names: List[str],
    body_to_joints: Dict[str, List[str]],
) -> List[Dict]:
    """
    Extracts a trajectory-level tau list of dictionaries from MuJoCo's MjData and MjModel.

    Args:
        env: Simulation environment.
        states: List of states to reset the environment to.
        model (mujoco.MjModel): MuJoCo model containing the simulation information.
        data (mujoco.MjData): MuJoCo data structure that holds simulation state.
        obj_names (list[str]): List of object names to track poses and velocities.
        body_to_joints (Dict[str, List[str]]): Mapping from body names to joint names.

    Returns:
        List[Dict]: A list where each element is a dictionary containing trajectory information at a specific timestep.
    """
    tau_traj = []

    for state in states:
        env.reset_to({"states": state})
        joint_to_qpos = get_joint_name_to_qpos(model, data)
        joint_to_qvel = get_joint_name_to_qvel(model, data)

        tau_step = {
            "qpos": {},
            "qvel": {},
            "pose": {},
            "velocity": {},
            "contacts": [],
        }

        for body, joints in body_to_joints.items():
            for joint in joints:
                tau_step["qpos"][joint] = joint_to_qpos[joint]
                tau_step["qvel"][joint] = joint_to_qvel[joint]

        for object_name in obj_names:
            tau_step["pose"][object_name] = get_body_pose(model, data, object_name)
            tau_step["velocity"][object_name] = get_body_velocity(
                model, data, object_name
            )

        body_names = get_all_body_names(
            mj_model, exclude_keywords=["target", "mount", "table", "world", "robot"]
        )
        tau_step["contacts"] = get_body_contact_pairs(model, data, body_names)

        tau_traj.append(tau_step)

    return tau_traj


def load_env(file_path: str):
    dummy_spec = dict(
        obs=dict(
            low_dim=["robot0_eef_pos"],
            rgb=["agentview_image", "agentview"],
        ),
    )
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=file_path)
    ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=dummy_spec)
    controller_config = load_composite_controller_config(robot="Panda")
    env_meta["env_kwargs"]["controller_configs"] = controller_config
    src_env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        use_image_obs=True,
        render_offscreen=True,
        render=False,
    )

    src_env = CPEnv(src_env)
    return src_env


def create_text_clip(text, width, height, duration):
    """Create an ImageClip with text without relying on ImageMagick."""
    img = Image.new("RGB", (width, height), color="black")
    draw = ImageDraw.Draw(img)

    # Use a default font available on the system
    try:
        font = ImageFont.truetype("arial.ttf", 50)
    except IOError:
        font = ImageFont.load_default()

    text_size = draw.textbbox((0, 0), text, font=font)
    text_width = text_size[2] - text_size[0]
    text_position = ((width - text_width) // 2, 10)
    draw.text(text_position, text, fill="white", font=font)

    text_np = np.array(img)
    return ImageClip(text_np, duration=duration)


def crop_and_label_video(video_path: str, constraints: List[Dict], output_path: str):
    # Load the video
    video = mp.VideoFileClip(video_path)
    fps = video.fps

    width, height = video.size
    clips = []

    for constraint in constraints:
        start_time, end_time = (
            constraint["timesteps"][0] / fps,
            constraint["timesteps"][-1] / fps,
        )

        # Extract subclip
        subclip = video.subclip(start_time, end_time)

        # Create text overlay
        txt_clip = create_text_clip(
            constraint["name"]
            + f" t={constraint['timesteps'][0]} to t={constraint['timesteps'][-1]}",
            width,
            80,
            subclip.duration,
        )
        txt_clip = txt_clip.set_position(("center", "top"))

        # Composite the video with text overlay
        labeled_clip = mp.CompositeVideoClip([subclip, txt_clip])
        clips.append(labeled_clip)

    # Concatenate all clips
    final_video = mp.concatenate_videoclips(clips, method="compose")

    # Save the final concatenated video
    final_video.write_videofile(output_path, codec="libx264", fps=fps)

    print(f"Final video saved at {output_path}")


class ConstraintExtractor:
    def __init__(
        self,
        api_key,
        model="gpt-4",
        cache_db: pathlib.Path = pathlib.Path("cache/llm_cache.db"),
    ):
        """Initialize the ConstraintExtractor class with OpenAI API key."""
        try:
            self.client = openai.OpenAI(api_key=api_key)
        except Exception as e:
            print("Error initializing OpenAI client:", e)
            self.client = None
        self.model = model

        # create directory of cache_db if it doesn't exist
        cache_db.parent.mkdir(parents=True, exist_ok=True)
        self.cache = LLMCache(cache_db.as_posix())

    def get_constraints_prompt(
        self, instruction: str, body_names: list, joints_dict: dict, template_path: str
    ) -> str:
        """Generate the OpenAI prompt by filling in the template with task details."""
        with open(template_path, "r") as file:
            prompt_template = Template(file.read())
        return prompt_template.substitute(
            instruction=instruction,
            body_names=", ".join(body_names),
            json_joints_dict=json.dumps(joints_dict, indent=4),
        )

    def get_constraints(
        self,
        instruction,
        body_names,
        joints_dict,
        print_prompt: bool = False,
        allow_manual_input: bool = False,
    ):
        """Query OpenAI to convert natural language instructions into structured constraints using real object names."""
        prompt = self.get_constraints_prompt(
            instruction,
            body_names,
            joints_dict,
            "demo_aug/prompt_templates/constraint.md",
        )
        if print_prompt:
            print("Prompt:", prompt)

        # Check SQLite cache
        with self.cache as cache:
            cached_output = cache.get(prompt)
            if cached_output:
                return json.loads(cached_output)

            try:
                output = self.query_llm(prompt)
            except Exception as e:
                print("Error querying OpenAI:", e)
                print(f"Prompt used: {prompt}")
                if allow_manual_input:
                    output = input("Please manually paste the output here: ")
                else:
                    print(
                        "Prompt not in cache, API call failed, manual input not allowed. Exiting."
                    )
                    return None  # Fail silently if manual input is not allowed

            # Store in cache and return
            cache.set(prompt, output)
            return json.loads(output)

    def query_llm(self, prompt: str) -> str:
        """Sends a structured prompt to an LLM and retrieves generated Python code."""
        print("Querying OpenAI with prompt:\n", prompt)
        response = self.client.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    def get_constraint_timestep(
        self,
        constraints: List[Dict],
        observations: List[Dict],
        body_names: List[str],
        joints_dict: Dict,
        template_path: str = "demo_aug/prompt_templates/constraint-timestep-functions.md",
        save_code_path: str = "demo_aug/utils/constraint_segmentation.py",
        allow_manual_input: bool = False,
    ) -> List[Dict]:
        """Detect timesteps for each subtask in the constraints using observations."""
        # TODO(klin): add open-ai key when needed
        prompt = self.get_constraint_timestep_functions_prompt(
            constraints, template_path, body_names, joints_dict
        )
        # Check SQLite cache
        with self.cache as cache:
            cached_output = cache.get(prompt)
            if cached_output:
                return json.loads(cached_output)

            try:
                output = self.query_llm(prompt)
            except Exception as e:
                print("Error querying OpenAI:", e)
                print(f"Prompt used: {prompt}")
                if allow_manual_input:
                    output = input("Please manually paste the output here: ")
                else:
                    print(
                        "Prompt not in cache, API call failed, manual input not allowed. Exiting."
                    )
                    return []  # Fail silently if manual input is not allowed

        self.save_generated_code(output, save_code_path)
        constraints = self.load_and_run(
            save_code_path, tau=observations, subtask_description=constraints
        )
        return constraints

    def get_constraint_timestep_functions_prompt(
        self,
        constraints: List[Dict],
        template_path: str,
        body_names: List[str],
        joints_dict: Dict,
    ) -> str:
        """Generate the OpenAI prompt by filling in the template with task details."""
        with open(template_path, "r", encoding="utf-8") as file:
            prompt_template = Template(file.read())

        return prompt_template.substitute(
            subtask_description=constraints,
            body_names=", ".join(body_names),
            json_joints_dict=json.dumps(joints_dict, indent=4),
        )

    def save_generated_code(
        self, code_str: str, filename="demo_aug/utils/constraint_segmentation.py"
    ):
        """Saves generated Python code from the LLM to a file."""
        file_path = Path(filename)
        if file_path.exists():
            response = input(
                f"The file {filename} already exists. Do you want to overwrite it? (y/n): "
            )
            if response.lower() != "y":
                print("Aborting code saving.")
                return

        with open(filename, "a") as f:
            f.write(code_str)

    def load_and_run(
        self,
        filename: str,
        tau: List[Dict],
        subtask_description: List[Dict],
        verbose: bool = False,
    ):
        """Dynamically loads the generated functions and computes subtask start/stop times."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "demo_aug/utils/constraint_segmentation.py", filename
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        results = {}
        current_t = 0
        start_idx, end_idx = 0, len(tau)
        for subtask in subtask_description:
            if verbose:
                print("Processing subtask:", subtask["name"])
            subtask_name = subtask["name"].replace(" ", "_")  # Normalize key
            start_fn = getattr(module, f"is_{subtask_name}_start")
            end_fn = getattr(module, f"is_{subtask_name}_end")

            for t in range(current_t, len(tau)):
                if start_fn(tau[t], subtask["obj_names"]):
                    start_idx = t
                    current_t = t
                    break
            if verbose:
                print(f"Finding end of {subtask_name} from timestep {start_idx}")

            current_t = start_idx
            for t in range(current_t, len(tau)):
                if end_fn(tau[t], subtask["obj_names"]):
                    end_idx = t
                    current_t = end_idx
                    break

            if verbose:
                print(f"Found end of {subtask_name} at timestep {end_idx}")

            results[subtask_name] = {
                "start_function": f"is_{subtask_name}_start",
                "end_function": f"is_{subtask_name}_end",
                "timesteps": list(range(start_idx, end_idx)),
            }

        return results


def adjust_constraint_timesteps_for_collision_free_motion(
    constraints: List[Dict],
    env,
    states: np.ndarray,
    robot_body_name: str = "robot",
    verbose: bool = False,
    robot_weld_frame: str = "gripper0_right_eef",
) -> List[Dict]:
    """Adjusts the first and last timesteps of constraints to ensure they are collision-free.

    Args:
        constraints (List[Dict]): List of constraint dictionaries.
        env: MuJoCo environment object supporting reset_to({"states": state}).
        model: MuJoCo model object.
        states (np.ndarray): Array of environment states corresponding to timesteps.
        robot_body_name (str, optional): Root body name of the robot. Defaults to "robot".
        verbose (bool, optional): Whether to log debug info. Defaults to False.

    Returns:
        List[Dict]: Updated constraints with adjusted timestep ranges.
    """
    model = env.env.env.sim.model._model
    data = env.env.env.sim.data._data

    # Get robot geometries
    robot_geoms = get_subtree_geom_ids_by_group(
        model, model.body(robot_body_name).id, group=0
    )

    for i, constraint in enumerate(constraints):
        # Determine current constraint welded objects (by attachment frame)
        welded_bodies = set()
        if "obj_to_parent_attachment_frame" in constraint:
            for obj, parent in constraint["obj_to_parent_attachment_frame"].items():
                if (
                    parent == robot_weld_frame
                ):  # Assumption: welded objects are attached here
                    welded_bodies.add(obj)

        # Exclude default prefixes (always ignore collisions with these)
        default_excludes = ["robot", "gripper", "table"]
        # (They will be concatenated with welded objects later.)

        timesteps = constraint.get("timesteps", [])
        if not timesteps:
            continue

        # Adjust the start timestep: move earlier until collision free.
        new_start_idx = timesteps[0]
        while new_start_idx >= 0:
            env.reset_to({"states": states[new_start_idx]})
            # Check collisions between robot and non-robot bodies.
            body_ids = get_top_level_bodies(
                model, exclude_prefixes=default_excludes + list(welded_bodies)
            )
            non_robot_geoms = [
                geom_id
                for body_id in body_ids
                for geom_id in get_subtree_geom_ids_by_group(model, body_id, group=0)
            ]
            geom_pairs_to_check: List[Tuple] = [(robot_geoms, non_robot_geoms)]
            if (
                len(
                    check_geom_collisions(
                        model, data, geom_pairs_to_check, collision_activation_dist=0.03
                    )
                )
                == 0
            ):
                break
            new_start_idx -= 1

        # Adjust the last timestep: move later until collision free.
        new_end_idx = timesteps[-1]
        while new_end_idx < len(states) - 1:
            env.reset_to({"states": states[new_end_idx]})
            # For object-object constraints, add extra welded objects.
            object_object_welded = set()
            if constraint.get("constraint_type") == "object-object":
                object_object_welded.update(constraint.get("obj_names", []))
            # Build local exclusion set based on current and next constraint.
            current_attached = set(welded_bodies)
            if i < len(constraints) - 1:
                next_constraint = constraints[i + 1]
                next_welded = set()
                if "obj_to_parent_attachment_frame" in next_constraint:
                    for obj, parent in next_constraint[
                        "obj_to_parent_attachment_frame"
                    ].items():
                        if parent == robot_weld_frame:
                            next_welded.add(obj)
                    # If next constraint involves a welded object, add it.
                    exclude_set = current_attached.union(object_object_welded).union(
                        next_welded
                    )
                else:
                    # Otherwise, exclude only the originally attached bodies.
                    exclude_set = current_attached
            else:
                exclude_set = current_attached.union(object_object_welded)
            # Combine with default exclusions.
            exclude_prefixes = default_excludes + list(exclude_set)

            body_ids = get_top_level_bodies(model, exclude_prefixes=exclude_prefixes)
            non_robot_geoms = [
                geom_id
                for body_id in body_ids
                for geom_id in get_subtree_geom_ids_by_group(model, body_id, group=0)
            ]
            geom_pairs_to_check = [(robot_geoms, non_robot_geoms)]
            if len(check_geom_collisions(model, data, geom_pairs_to_check)) == 0:
                break
            new_end_idx += 1

        # Ensure new_end_idx is valid.
        new_end_idx = max(new_start_idx, new_end_idx)
        constraint["timesteps"] = list(range(new_start_idx, new_end_idx + 1))
        if verbose:
            print(f"original start and end timesteps: {timesteps[0], timesteps[-1]}")
            print(f"new start and end timesteps: {new_start_idx, new_end_idx}")
    return constraints


# Example usage
if __name__ == "__main__":
    api_key = "your-openai-api-key"
    # task_name = "mug_cleanup"
    # instruction = "open drawer, pick and place mug into drawer, close drawer"
    task_name = "hammer_cleanup"
    instruction = "open drawer, pick and place hammer into drawer, close drawer"
    source_dir = "datasets/source"

    demo_file = f"{source_dir}/{task_name}.hdf5"
    video_path = f"{source_dir}/{task_name}.mp4"
    output_path_non_collision_free = f"{source_dir}/{task_name}_labeled.mp4"
    output_path_collision_free = f"{source_dir}/{task_name}_labeled_collision_free.mp4"

    allow_manual_input = True
    env = load_env(demo_file)
    with h5py.File(demo_file, "r") as f:
        demo_key = list(f["data"].keys())[0]
        state_key = f"data/{demo_key}/states"
        states = np.array(f[state_key])
    mj_model, mj_data = env.env.env.sim.model._model, env.env.env.sim.data._data
    body_names = get_all_body_names(
        mj_model, exclude_keywords=["target", "mount", "table", "world", "robot"]
    )
    body_to_joints: Dict = {
        body: get_body_joints_recursive(mj_model, body) for body in body_names
    }
    tau = get_trajectory(env, states, mj_model, mj_data, body_names, body_to_joints)
    extractor = ConstraintExtractor(api_key)
    constraints: List[Dict] = extractor.get_constraints(
        instruction, body_names, body_to_joints, allow_manual_input=allow_manual_input
    )
    constraint_timesteps = extractor.get_constraint_timestep(
        constraints,
        tau,
        body_names,
        body_to_joints,
        allow_manual_input=allow_manual_input,
    )

    print("Final Constraints:", constraint_timesteps)
    # Add timesteps to the original constraints dictionary
    for i, constraint in enumerate(constraints):
        subtask_name = constraint["name"].replace(" ", "_")
        if subtask_name in constraint_timesteps:
            constraints[i]["timesteps"] = constraint_timesteps[subtask_name][
                "timesteps"
            ]
        else:
            constraints[i][
                "timesteps"
            ] = []  # Assign an empty list if no timesteps found
            breakpoint()

    val = Constraint.from_constraint_data_dict(constraints[0])
    crop_and_label_video(video_path, constraints, output_path_non_collision_free)

    adjust_constraint_timesteps_for_collision_free_motion(
        constraints=constraints,
        env=env,
        states=states,
        robot_body_name="robot0_base",
        verbose=False,
    )
    crop_and_label_video(video_path, constraints, output_path_collision_free)
    print(f"final constraints: {constraints}")
