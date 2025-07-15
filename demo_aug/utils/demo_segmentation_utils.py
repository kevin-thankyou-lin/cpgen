import datetime
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, TypedDict, Union

import imageio
import mujoco
import numpy as np
from openai import OpenAI
from robomimic.envs.env_base import EnvBase

from demo_aug.utils.mujoco_utils import get_top_level_body_names

# Default keypoints definitions
DEFAULT_OBJ_KEYPOINTS: List[List[List[float]]] = [
    [[0, 0, 0.01], [0, 0, -0.01], [0, -0.01, 0], [0, 0.01, 0]]
]

# for franka_umi_gripper
DEFAULT_ROBOT_KEYPOINTS: Dict[str, Dict[str, List[List[List[float]]]]] = {
    "PandaUmiGripper": {
        "gripper0_right_rightfinger": [[[0, 0, 0.12], [0, 0, 0.14]]],
        "gripper0_right_leftfinger": [[[0, 0, 0.12], [0, 0, 0.14]]],
    },
    "PandaGripper": {
        "gripper0_right_rightfinger": [[[0, 0, 0.00], [0, 0, 0.048]]],
        "gripper0_right_leftfinger": [[[0, 0, 0.00], [0, 0, 0.048]]],
    },
}

# for robotiq_85_gripper: have default keypoints on each given gripper
# and only some are activated?
# DEFAULT_ROBOT_KEYPOINTS: Dict[str, List[List[List[float]]]] = {
#     "gripper0_right_rightfinger": [[[0, 0, 0.0447], [0, 0, 0]]],
#     "gripper0_right_leftfinger": [[[0, 0, 0.0447], [0, 0, 0]]],
# }


def get_current_contacts(env: EnvBase) -> List[Tuple[str, str]]:
    contacts = []
    ncon = env.env.sim.data.ncon
    for i in range(ncon):
        contact = env.env.sim.data.contact[i]
        geom1 = contact.geom1
        geom2 = contact.geom2
        body1_id = env.env.sim.model.geom_bodyid[geom1]
        body2_id = env.env.sim.model.geom_bodyid[geom2]
        body1 = env.env.sim.model.body_names[body1_id]
        body2 = env.env.sim.model.body_names[body2_id]
        body1_name = body1.decode() if isinstance(body1, bytes) else body1
        body2_name = body2.decode() if isinstance(body2, bytes) else body2
        contacts.append((body1_name, body2_name))
    return contacts


def get_body_descendant_indices(model, body_idx: int) -> List[int]:
    """Recursively gathers indices for a body and all its descendant bodies."""
    descendants = [body_idx]
    for i in range(model.nbody):
        if model.body_parentid[i] == body_idx:
            descendants.extend(get_body_descendant_indices(model, i))
    return descendants


def get_body_and_children_geom_indices(model, body: str) -> List[int]:
    """Returns the geom indices for the specified body and all its children."""
    body_names = [n.decode() if isinstance(n, bytes) else n for n in model.body_names]
    body_idx = body_names.index(body)
    descendant_body_indices = get_body_descendant_indices(model, body_idx)
    geom_indices = []
    for b in descendant_body_indices:
        start = model.body_geomadr[b]
        num = model.body_geomnum[b]
        geom_indices.extend(range(start, start + num))
    return geom_indices


def compute_body_distance(
    env: EnvBase,
    body_name_1: str,
    body_name_2: str,
    collision_distance: float,
    mj_dist: bool = True,
) -> float:
    """
    Computes the minimum distance between all geoms of two bodies using MuJoCo.
    """

    if not mj_dist:
        pos1 = env.env.sim.data.get_body_xpos(body_name_1).copy()
        pos2 = env.env.sim.data.get_body_xpos(body_name_2).copy()
        d = np.linalg.norm(pos1 - pos2)

    sim = env.env.sim
    model = sim.model
    data = sim.data

    # Convert model body names (bytes) to strings.
    geoms1 = get_body_and_children_geom_indices(model, body_name_1)
    geoms2 = get_body_and_children_geom_indices(model, body_name_2)

    min_dist = float("inf")
    fromto = np.empty(6)
    for g1 in geoms1:
        for g2 in geoms2:
            d = mujoco.mj_geomDistance(
                model._model, data._data, g1, g2, collision_distance, fromto
            )
            if d < min_dist:
                min_dist = d
    return min_dist


def is_robot_object_interaction_end(
    env: EnvBase,
    robot: str,
    obj: str,
    next_interaction: Optional[Tuple[str, str]],
    interaction_threshold: float = 0.03,
) -> bool:
    """
    Checks if the robot has ended its interaction the object.
    Ending is defined by one of two conditions:

    - If the next interaction involves the object (i.e., obj is in next_interaction),
      ending is when the object has no contacts with any non-robot bodies.
    - If the next interaction does not involve the object,
      ending is when the robot stops colliding with the object.
    """
    contacts = get_current_contacts(env)
    if next_interaction is None or obj in next_interaction:
        contact_bodies = {
            b2 if b1 == obj else b1 for b1, b2 in contacts if obj in (b1, b2)
        }
        non_robot_contacts = [
            body for body in contact_bodies if not body.startswith("gripper0_")
        ]
        if len(non_robot_contacts) == 0:
            bodies_to_check = get_top_level_body_names(
                env.env.sim.model._model,
                exclude_prefixes=["gripper0_", "robot0", "world", obj],
            )
            # Check if the object is far enough from other things in the world to do motion planning and consider the interaction ended
            distance = (
                min(
                    compute_body_distance(
                        env, obj, body, collision_distance=interaction_threshold
                    )
                    for body in bodies_to_check
                )
                if bodies_to_check
                else np.inf
            )
            return distance >= interaction_threshold
        return False
    else:
        # Explicitly determine if there is an ongoing collision between the robot and the object.
        robot_obj_contacts = [
            (body1, body2)
            for body1, body2 in contacts
            if (body1 == robot and body2 == obj) or (body1 == obj and body2 == robot)
        ]
        collision_exists = len(robot_obj_contacts) > 0
        if not collision_exists:
            bodies_to_check = get_top_level_body_names(
                env.env.sim.model._model,
                exclude_prefixes=["gripper0_", "robot0", "world"],
            )
            # Check if robot is far enough from the object to be able to do motion planning and consider the interaction ended
            distance = min(
                compute_body_distance(
                    env, robot, obj, collision_distance=interaction_threshold
                )
                for body in bodies_to_check
            )
            return distance >= interaction_threshold
    return not collision_exists


def is_obj_obj_interaction_end(
    env: EnvBase,
    obj1: str,
    obj2: str,
    next_interaction: Optional[Tuple[str, str]],
    interaction_threshold: float = 0.03,
) -> bool:
    """
    Determines if the current interaction between an obj1 and obj2 has ended.

    We assume obj1 is the object that's being manipulated by the gripper, and obj2 is the other object.

    Thus, we have two cases to consider from the perspective of obj1 (which is directly 'actuated'):
        1. next_interaction involves obj1: We still need to hold onto obj1 ---> we want to do some collision free motion planning with obj1.
            Thus, current interaction ends if obj1 is not in contact with obj1 and has some distance from obj2.
        2. next_interaction does not involve obj1: we want to let go of obj1. We assume that we want to do some motion planning with the robot afterwards.
            Thus, current interaction ends once robot is no longer in contact with obj1.

    Args:
        env (EnvBase): The environment instance to check for collisions.

    Returns:
        bool: True if there are no collisions involving the gripper, False otherwise.
    """
    contacts = get_current_contacts(env)
    if next_interaction is None or obj1 in next_interaction:
        contact_bodies = {
            b2 if b1 == obj1 else b1 for b1, b2 in contacts if obj1 in (b1, b2)
        }
        if obj2 not in contact_bodies:
            distance = compute_body_distance(
                env, obj1, obj2, collision_distance=interaction_threshold
            )
            return (
                distance >= interaction_threshold
            )  # Assuming interaction_threshold as the minimum distance threshold need for motion planning
        return False
    else:
        # next interaction does not involve obj1, so we end when robot lets go of obj1
        # technically the above is true, however to enable motion planning,
        # also need to ensure robot is far enough away from other objects too. Note, however,
        # that during cpgen time, we use keypoints relative to the target object to get actions.
        # this means that increasing the timerange of the constraint doesn't help us move out of collision.
        # instead we use heuristic of extra "lift" actions to get into collision free zone.
        robot_contacts = [
            (b1, b2)
            for b1, b2 in contacts
            if (b1 == "gripper0_right_right_gripper" and b2 == obj1)
            or (b1 == obj1 and b2 == "gripper0_right_right_gripper")
        ]
        if len(robot_contacts) == 0:
            distance = compute_body_distance(
                env,
                "gripper0_right_right_gripper",
                obj1,
                collision_distance=interaction_threshold,
            )
            return (
                distance >= interaction_threshold
            )  # Assuming interaction_threshold as the minimum distance threshold need for motion planning
        return len(robot_contacts) == 0


def decompose_trajectory(
    env: EnvBase,
    states: np.ndarray,
    interaction_threshold: float,
    interactions: Sequence[Tuple[str, str]],
    segments_output_dir: Path = Path("segments"),
    interaction_interaction_t_gap: int = 10,
) -> List[Tuple[str, Tuple[int, int]]]:
    """
    Decomposes a trajectory into segments.
    For each state, distances for each (non-completed) interaction pair are computed.

    For determining the start and end of an interaction segment, we use interaction threshold.
    A state is labeled with the pair having the smallest distance below its threshold or as "motion".
    Once an interaction segment is complete, it is ignored in later classifications.
    Collision conditions can force an early end of an interaction segment.
    Returns:
        List of tuples: (segment_label, (start_index, end_index))
    """
    segments = []
    current_seg_label = None
    seg_start = 0
    completed_interactions: List[str] = []
    completed_interactions_t: List[int] = []

    for t, state in enumerate(states):
        env.reset_to({"states": state}, no_return_obs=True)
        active_interactions = []
        curr_segment_start_t = segments[-1][1][-1] if segments else 0
        # Skip if we're too close to the last completed interaction
        if (
            completed_interactions_t
            and t < completed_interactions_t[-1] + interaction_interaction_t_gap
        ) or curr_segment_start_t + interaction_interaction_t_gap > t:
            print(
                f"Skipping {t} as it's too close to last completed interaction; last is {t}"
            )
            continue

        if len(completed_interactions) != len(interactions):
            curr_interaction = interactions[len(completed_interactions)]
            name1, name2 = curr_interaction
            label = f"{name1}-{name2}"

            d = compute_body_distance(
                env,
                name1,
                name2,
                collision_distance=interaction_threshold,
                mj_dist=True,
            )
            if d < interaction_threshold:
                active_interactions.append((label, d))

        if active_interactions:
            # find earliest active interaction that's not completed
            seg_label = min(active_interactions, key=lambda x: x[1])[0]
        else:
            seg_label = "motion"

        # Detect termination of current interaction segments
        if seg_label != "motion":
            entity1, entity2 = seg_label.split("-")
            curr_interaction_idx = len(completed_interactions)
            next_interaction_idx = curr_interaction_idx + 1
            next_interaction_label = (
                interactions[next_interaction_idx]
                if next_interaction_idx < len(interactions)
                else None
            )
            if entity1.startswith("gripper0_") and not entity2.startswith("gripper0_"):
                if is_robot_object_interaction_end(
                    env,
                    entity1,
                    entity2,
                    next_interaction=next_interaction_label,
                    interaction_threshold=interaction_threshold,
                ):
                    seg_label = "motion"
            elif entity2.startswith("gripper0_") and not entity1.startswith(
                "gripper0_"
            ):
                if is_robot_object_interaction_end(
                    env,
                    entity2,
                    entity1,
                    next_interaction=next_interaction_label,
                    interaction_threshold=interaction_threshold,
                ):
                    seg_label = "motion"
                if is_obj_obj_interaction_end(
                    env,
                    entity1,
                    entity2,
                    next_interaction=next_interaction_label,
                    interaction_threshold=interaction_threshold,
                ):
                    seg_label = "motion"
            else:
                if is_obj_obj_interaction_end(
                    env,
                    entity1,
                    entity2,
                    next_interaction=next_interaction_label,
                    interaction_threshold=interaction_threshold,
                ):
                    seg_label = "motion"

        # Segment change detection.
        if current_seg_label is None:
            current_seg_label = seg_label
            seg_start = t
        elif seg_label != current_seg_label:
            segments.append((current_seg_label, (seg_start, t - 1)))
            if current_seg_label != "motion":
                completed_interactions.append(current_seg_label)
                completed_interactions_t.append(t)
            current_seg_label = seg_label
            seg_start = t

    segments.append((current_seg_label, (seg_start, len(states) - 1)))

    # save segment renderings
    # for seg_label, (start, end) in segments:
    #     filename = segments_output_dir / f"{seg_label}_segment_{start}_{end}.mp4"
    #     save_segment_rendering(env, states, (start, end), filename)
    #     print(f"Saved {seg_label} segment rendering to {filename}")

    return segments


def save_segment_rendering(
    env: EnvBase,
    states: np.ndarray,
    segment: Tuple[int, int],
    filename: Path,
    fps: int = 20,
) -> None:
    frames = []
    start, end = segment
    print(f"Saving segment {start} to {end}")
    for t in range(start, end + 1):
        # assumes env has been monkey patched to take in this no_return_obs argument
        env.reset_to({"states": states[t]}, no_return_obs=True)
        cam_name = env.env.camera_names[0]
        frame = env.render(mode="rgb_array", height=90, width=160, camera_name=cam_name)
        frames.append(frame)
    imageio.mimsave(filename.as_posix(), frames, fps=fps)


def parse_interactions(interactions_str: str) -> List[Tuple[str, str]]:
    pairs = []
    for pair_str in interactions_str.split(","):
        pair_str = pair_str.strip()
        if not pair_str:
            continue
        tokens = pair_str.split(":")
        if len(tokens) != 2:
            raise ValueError(
                f"Invalid interaction format; expected 'entity1:entity2'. Got: {interactions_str}"
            )
        pairs.append((tokens[0].strip(), tokens[1].strip()))
    return pairs


def unparse_interactions(interactions: List[Tuple[str, str]]) -> str:
    """Convert list of interaction pairs back to string format."""
    return ",".join(f"{entity1}:{entity2}" for entity1, entity2 in interactions)


def create_constraint(
    label: str, time_range: Tuple[int, int], gripper_type: Literal["PandaGripper"]
) -> Dict[str, Any]:
    parts = label.split("-")
    if len(parts) != 2:
        raise ValueError(f"Unexpected segment label format: {label}")
    obj1, obj2 = parts[0], parts[1]

    robot_keypoints = DEFAULT_ROBOT_KEYPOINTS.get(gripper_type, {})
    constraint: Dict[str, Any] = {
        "name": label,
        "obj_names": [obj for obj in [obj1, obj2] if not obj.startswith("gripper0_")],
        "timesteps": list(range(time_range[0], time_range[1])),
        "keypoints_obj_frame_annotation": {obj1: DEFAULT_OBJ_KEYPOINTS}
        if not obj1.startswith("gripper0_")
        else {obj2: DEFAULT_OBJ_KEYPOINTS},
        # These keys are part of the data format
        "symmetries": [],
        "during_constraint_behavior": None,
        "reset_near_random_constraint_state": None,
        "src_model_file": None,
    }

    # Decide constraint type and add additional fields
    if "gripper0" in obj1 or "gripper0" in obj2:
        constraint["constraint_type"] = "robot-object"
        constraint["keypoints_robot_link_frame_annotation"] = robot_keypoints
        # Not used for robot-obj, so leave obj_to_parent_attachment_frame unset (or None)
        constraint["obj_to_parent_attachment_frame"] = None
    else:
        constraint["constraint_type"] = "object-object"
        constraint["obj_to_parent_attachment_frame"] = {obj1: "gripper0_right_eef"}
        # Not used for object-object, so leave keypoints_robot_link_frame_annotation unset (or None)
        constraint["keypoints_robot_link_frame_annotation"] = None

    if constraint["constraint_type"] == "robot-object":
        constraint["post_constraint_behavior"] = [
            "lift",
            "lift",
        ]  # for motion planning clearance
        # (should really be "move away" for motion-planability)
    if constraint["constraint_type"] == "object-object":
        # if there's no next interaction with the current object, we can likely open gripper
        # TODO: this motion-planning enabler can be improved, I think (esp if object isn't let go of in next step)
        constraint["post_constraint_behavior"] = ["open_gripper", "lift", "lift"]
    return constraint


if __name__ == "__main__":
    # Simulation states + LLM demo segmentation

    # --- OpenAI API Client ---
    client = OpenAI()

# --- Type Definitions ---
SegmentType = Literal["motion", "skill"]


class Segment(TypedDict):
    type: SegmentType
    start: int
    end: int


# --- State Extraction Functions from MuJoCo ---
def quat_to_mat(quat):
    from scipy.spatial.transform import Rotation as R

    return R.from_quat(quat[[1, 2, 3, 0]]).as_matrix()  # MuJoCo uses w, x, y, z


def get_pose(data, model, body_name):
    body_id = model.body(body_name).id
    pos = data.xpos[body_id]
    quat = data.xquat[body_id]
    rot_mat = quat_to_mat(quat)
    pose = np.eye(4)
    pose[:3, :3] = rot_mat
    pose[:3, 3] = pos
    return pose


def get_velocity(data, model, body_name):
    body_id = model.body(body_name).id
    lin_vel = data.cvel[body_id][:3]
    ang_vel = data.cvel[body_id][3:]
    return np.concatenate([lin_vel, ang_vel])


def get_contact_info(data, model):
    contacts = []
    for i in range(data.ncon):
        contact = data.contact[i]
        geom1_id = contact.geom1
        geom2_id = contact.geom2
        body1_id = model.geom_bodyid[geom1_id]
        body2_id = model.geom_bodyid[geom2_id]
        body1_name = model.body(body1_id).name
        body2_name = model.body(body2_id).name

        # Get contact position and normal (approximate)
        contact_pos = contact.pos.copy()
        contact_normal = contact.frame[:3].copy()

        # Optional: approximate contact force if solver computed it
        force = None
        if data.efc_force is not None and contact.efc_address >= 0:
            force = data.efc_force[contact.efc_address]

        contacts.append(
            {
                "body1": body1_name,
                "body2": body2_name,
                "position": contact_pos,
                "normal": contact_normal,
                "force": force,
            }
        )
    return contacts


def extract_robot_and_object_state(
    data, model, eef_body_name, gripper_joint_names, object_names
):
    robot_state = {
        "eef_pose": get_pose(data, model, eef_body_name),
        "eef_velocity": get_velocity(data, model, eef_body_name),
        "gripper_state": np.array(
            [data.joint(name).qpos for name in gripper_joint_names]
        ),
    }

    object_state = {}
    for obj_name in object_names:
        obj_info = {
            "pose": get_pose(data, model, obj_name),
            "velocity": get_velocity(data, model, obj_name),
        }
        joint_names = []
        for j in range(model.njnt):
            joint = model.joint(j)
            if joint.type == mujoco.mjtJoint.mjJNT_FREE:
                # ignore free joints since we'll get their pose anyways
                continue
            if joint.bodyid == model.body(obj_name).id:
                joint_names.append(joint.name)
            # might fail if we get another joint type, but we'll ignore that for now

        if joint_names:
            obj_info["joint_pos"] = np.array(
                [data.joint(name).qpos for name in joint_names]
            )
            obj_info["joint_vel"] = np.array(
                [data.joint(name).qvel for name in joint_names]
            )
        object_state[obj_name] = obj_info

    contact_info = get_contact_info(data, model)

    return {"robot": robot_state, "objects": object_state, "contacts": contact_info}


# --- Compressor with Smart Sampling ---
def compress_state_sequence(
    state_sequence: List[Dict],
    sample_every: int = 5,
    include_first_last: bool = True,
    verbose: bool = False,
    debug_skipped: bool = False,
) -> List[Dict]:
    compressed = []
    prev_contacts = set()

    for t, state in enumerate(state_sequence):
        contacts = state.get("contacts", [])
        contact_pairs = set((c["body1"], c["body2"]) for c in contacts)
        contact_change = contact_pairs != prev_contacts

        is_sample = t % sample_every == 0
        should_keep = is_sample or contact_change

        if should_keep or (
            include_first_last and (t == 0 or t == len(state_sequence) - 1)
        ):
            robot = state["robot"]
            objects = state["objects"]

            gripper_state_mean = float(np.mean(robot["gripper_state"]))
            # discretize in increments of 0.01
            gripper_state_mean = round(gripper_state_mean * 100) / 100
            robot_summary = {
                "eef_pos": robot["eef_pose"][:3, 3].tolist(),
                "eef_vel": float(np.linalg.norm(robot["eef_velocity"])),
                "gripper_state": gripper_state_mean,
            }

            object_summary = {}
            for name, obj in objects.items():
                obj_summary = {
                    "pos": obj["pose"][:3, 3].tolist(),
                    "vel_norm": float(np.linalg.norm(obj["velocity"])),
                }
                if "joint_pos" in obj:
                    obj_summary["joint_pos"] = obj["joint_pos"].tolist()
                if "joint_vel" in obj:
                    obj_summary["joint_vel"] = obj["joint_vel"].tolist()
                object_summary[name] = obj_summary

            contact_summary = {
                # "n_contacts": len(contacts),
                "contact_pairs": list(contact_pairs)
            }

            compressed.append(
                {
                    "t": t,
                    "robot": robot_summary,
                    "objs": object_summary,
                    "contacts": contact_summary,
                }
            )
        else:
            if debug_skipped:
                print(f"[Skip] t={t}: not a sample step and no contact change")

        prev_contacts = contact_pairs

    if verbose:
        print(
            f"[Compressor] Kept {len(compressed)} / {len(state_sequence)} timesteps "
            f"({100 * len(compressed) / len(state_sequence):.1f}%)"
        )

    return compressed


def get_relevant_body_names(
    model: mujoco.MjModel, exclude_prefixes: List[str]
) -> List[str]:
    relevant_names = set()
    for i in range(model.nbody):
        body_name = model.body(i).name
        try:
            # Condition 1: Body name is not excluded.
            excluded = any(body_name.startswith(prefix) for prefix in exclude_prefixes)
        except Exception:
            import ipdb

            ipdb.set_trace()
        # Condition 2: Body has at least one non-fixed joint.
        jntadr = model.body_jntadr[i]
        jntnum = model.body_jntnum[i]
        has_non_fixed = (
            False
            if jntnum == 0
            else any(model.jnt_type[j] != -1 for j in range(jntadr, jntadr + jntnum))
        )

        def has_prefix(name: str) -> bool:
            return any(name.startswith(prefix) for prefix in exclude_prefixes or [])

        # Condition 3: Body is top-level (i.e., has no parent body).
        is_top_level = model.body_parentid[i] == 0 and has_prefix(
            model.names[model.name_bodyadr[i] :].decode()
        )
        if (not excluded) and (has_non_fixed or is_top_level):
            relevant_names.add(body_name)

    print(relevant_names)
    return list(relevant_names)


def choose_object_names(
    lang_description: str, top_level_body_names: List[str]
) -> Union[List[str], str]:
    prompt = f"""
    You are an expert in robotics simulation analysis. Your task is to identify which of the following MuJoCo top-level body names correspond to **objects** in the scene, given a task description.

    ### Task description:
    "{lang_description}"

    ### Top-level body names:
    {json.dumps(top_level_body_names, indent=2)}

    ### Instructions:
    Return only a **JSON-compatible Python list of strings** representing the object names, like:

    ["obj1", "obj2", "obj3"]

    ### Format constraints:
    - Do NOT wrap the result in markdown or code blocks.
    - Do NOT add any explanation, notes, or extra text.
    - The output must be directly parsable using `json.loads(output)`.

    *Only* return the list. Begin.
    """
    response = client.responses.create(model="gpt-4o", input=prompt)
    output_text = response.output[0].content[0].text
    try:
        return json.loads(output_text)
    except json.JSONDecodeError:
        print("[Warning] Failed to parse JSON for object names, returning raw output.")
        return output_text


def build_prompt(
    lang_description: str, interactions: str, n_interactions: int, compressed: str
) -> str:
    return f"""
You are an expert with great judgement in robotic task segmentation. Your job is to segment a robot demonstration into alternating 'motion' and 'skill' segments.

Each state in the demonstration contains:
- Robot end-effector information: position, velocity norm, and gripper state.
- Object information: positions, velocity norms, and, if applicable, joint positions and velocities (for articulated objects).
- Contact information: number of contacts and body pairs in contact.

Definitions:
- A 'motion' segment is when the robot is moving or when the robot is tranporting an object that it's holding.
- A 'skill' segment is when the robot is actively manipulating, contacting, or interacting with objects.
- Skill segments must be connected by collision-free motion (i.e., motion segments). Each skill is necessary for task success. A skill segment end_t
is the time when the robot successfully completes the subtask. A motion segment's end_t is when the transfer and transit motion ends.

**Guidance on imperfect demonstrations**:
Demonstrations may contain small imperfections, such as momentary breaks in contact or inconsistent contact readings.
Do not treat brief, noisy losses of contact as the end of a skill segment. Instead, use your judgment to maintain temporal consistency and recognize the robot's high-level intent.
Only switch between skill and motion segments when there's a meaningful, sustained change in behavior.


**Task description**: "{lang_description}"

**Interactions**:
{json.dumps(interactions, indent=2)}

These interactions are the pairs of robot-object or object-object interactions representing the skill segments in the demonstration. Thus, there are
{n_interactions} skill segments in the demonstration.

**Trajectory summary**:
{compressed}


Reason about what is happening in the trajectory, then return your final output.

Include the full list of segments as a JSON array, inside a Markdown code block like this:

```json
[
  {{"label": "motion", "start": start_t, "end": end_t}},
  {{"label": "skill", "start": start_t, "end": end_t}},
  ...
]```
This final JSON block should appear after all reasoning steps and must be valid JSON."""


def round_floats(obj: Any) -> Any:
    if isinstance(obj, float):
        # Convert float to a new float rounded to 3 decimal places.
        return float(f"{obj:.3f}")
    if isinstance(obj, list):
        return [round_floats(item) for item in obj]
    if isinstance(obj, dict):
        return {key: round_floats(val) for key, val in obj.items()}
    return obj


def call_segmenter_api_llm_e2e(
    lang_description: str,
    interactions: str,
    state_sequence: List,
    sample_every: int = 5,
    llm_log_dir: Optional[str] = None,
):
    # Step 1: Build compressed summary
    compressed_state_sequence = compress_state_sequence(
        state_sequence, sample_every=sample_every, include_first_last=True, verbose=True
    )
    interactions_lst = parse_interactions(interactions)
    compressed_state_sequence = round_floats(compressed_state_sequence)
    prompt = build_prompt(
        lang_description,
        interactions,
        len(interactions_lst),
        str(compressed_state_sequence),
    )

    model = "o3-mini"
    # Step 2: LLM call
    response = client.responses.create(
        model=model,
        input=prompt,
        # temperature=0.25,  # recommended for code-generation
    )

    if model == "o3-mini":
        output_text = response.output[1].content[0].text
    else:
        output_text = response.output[0].content[0].text.strip()
    print("Full LLM output:\n", output_text)

    # Log prompt and response if log_dir is provided
    if llm_log_dir:
        log_dir_path = Path(llm_log_dir)
        log_dir_path.mkdir(exist_ok=True, parents=True)

        timestamp = datetime.datetime.now().isoformat()
        log_file = log_dir_path / "segmenter_logs.json"

        log_entry = {"timestamp": timestamp, "prompt": prompt, "response": output_text}

        # Read existing log if present
        if log_file.exists():
            with open(log_file, "r") as f:
                try:
                    logs = json.load(f)
                except json.JSONDecodeError:
                    logs = []
        else:
            logs = []

        # Append new log and save
        logs.append(log_entry)
        with open(log_file, "w") as f:
            json.dump(logs, f, indent=2)

    # Step 3: Extract final JSON segment list using regex
    json_match = re.search(r"```json(.*?)```", output_text, re.DOTALL)
    if not json_match:
        print("❌ Could not find a valid JSON code block in LLM output.")
        return output_text, None

    json_text = json_match.group(1).strip()
    try:
        segments = json.loads(json_text)
        # Convert segments from the LLM output format to the expected format:
        # From: [{"type": "motion", "start": start_idx, "end": end_idx}, ...]
        # To: [("motion", (start_idx, end_idx)), ...]
        formatted_segments = []
        for segment in segments:
            segment_type = segment["label"]
            start_idx = segment["start"]
            end_idx = segment["end"]
            formatted_segments.append((segment_type, (start_idx, end_idx)))

        # First, ensure the number of skill segments matches the number of interactions
        skill_segments = [seg for seg in segments if seg["label"] == "skill"]
        if len(skill_segments) != len(interactions_lst):
            print(
                f"⚠️ Warning: Number of skill segments ({len(skill_segments)}) "
                f"does not match number of interactions ({len(interactions_lst)})"
            )
            return None

        new_interactions = interactions_lst.copy()
        # convert all "skill" segments to the corresponding interaction labels
        for i, (seg_type, (start, end)) in enumerate(formatted_segments):
            if seg_type == "skill":
                formatted_segments[i] = (
                    "-".join(new_interactions.pop(0)),
                    (start, end),
                )
        segments = formatted_segments
        return segments
    except json.JSONDecodeError as e:
        print("❌ JSON parsing failed:", e)
        return None


def run_llm_e2e_segmentation(
    env: Any,
    states: List[Any],
    lang_description: str,
    segments_output_dir: Path,
    interactions: str,
    llm_log_dir: Optional[str] = None,
):
    """
    Runs the language-model-based segmentation logic and returns trajectory segments.
    """
    # Acquire model handles (type ignored if mismatch: depends on your environment definitions)
    model, data = env.env.sim.model._model, env.env.sim.data._data  # type: ignore

    NON_RELEVANT_OBJECT_BODY_PREFIXES: List[str] = [
        "gripper",
        "table",
        "left_eef_target",
        "right_eef_target",
        "robot",
    ]
    ROBOT_GRIPPER_JOINT_NAMES: List[str] = [
        "gripper0_right_finger_joint1",
        "gripper0_right_finger_joint2",
    ]
    EEF_BODY_NAME: str = "gripper0_right_right_gripper"
    relevant_body_names = get_relevant_body_names(
        model, exclude_prefixes=NON_RELEVANT_OBJECT_BODY_PREFIXES
    )
    # objects for which we should query for state information
    task_relevant_object_names = choose_object_names(
        lang_description, relevant_body_names
    )

    # Build the state sequence
    state_sequence = []
    for t in range(10, len(states)):
        env.reset_to({"states": states[t]}, no_return_obs=True)
        step_state = extract_robot_and_object_state(
            data=data,
            model=model,
            eef_body_name=EEF_BODY_NAME,
            gripper_joint_names=ROBOT_GRIPPER_JOINT_NAMES,
            object_names=task_relevant_object_names
            if isinstance(task_relevant_object_names, list)
            else [],
        )
        state_sequence.append(step_state)

    # Calculate sample_every based on wanting ~60 state descriptions
    # We want to sample overall states evenly
    sample_every = max(1, len(state_sequence) // 60)
    print(
        f"Using sample_every={sample_every} to get approximately 60 state descriptions"
    )

    # Call your language-model segmenter
    segments = call_segmenter_api_llm_e2e(
        lang_description=lang_description,
        interactions=interactions,
        state_sequence=state_sequence,
        sample_every=sample_every,
        llm_log_dir=llm_log_dir,
    )

    if segments is None:
        print("❌ Segmentation failed.")
        return None

    return segments


def get_interactions_from_llm(
    lang_description: str, relevant_body_names: List[str]
) -> List[str]:
    # Craft a prompt for the LLM to generate interactions
    prompt = f"""
You are an assistant that translates a natural language task description into a list of chronological interactions.
Each interaction is between the robot gripper and an object (e.g., "gripper0_right_right_gripper:object_name"),
or between two objects (e.g., "object1_name:object2_name").

Task description: "{lang_description}". There are {len(lang_description.split(',')) if ',' in lang_description else 1} interactions.

Objects/entities in the environment: {', '.join(relevant_body_names)}

Return only the interactions as a single list of strings that is JSON loadable.
Once parsed, the JSON content should be a comma-separated list of interactions in chronological order.

Example:

Language description: "Pick and insert the nut into the peg."
Objects/entities in the environment: gripper0_right_right_gripper, SquareNut_main, peg0

Should return: ["gripper0_right_right_gripper:SquareNut_main", "SquareNut_main:peg0"]

### Format constraints:
- Do NOT wrap the result in markdown or code blocks.
- Do NOT add any explanation, notes, or extra text.
- The output must be directly parsable using `json.loads(output)`.

Only return the list. Begin.
"""
    response = client.responses.create(model="gpt-4o", input=prompt)
    output_text = response.output[0].content[0].text
    try:
        output_text = json.loads(output_text)
    except json.JSONDecodeError:
        print("[Warning] Failed to parse JSON for object names, returning raw output.")
    return output_text


def log_llm_interaction(llm_log_dir: str, prompt: str, output_text: str) -> None:
    """
    Logs the LLM prompt and response to a JSON file.

    Args:
        llm_log_dir: Directory to save the log file
        prompt: The prompt sent to the LLM
        output_text: The response received from the LLM
    """
    log_dir_path = Path(llm_log_dir)
    log_dir_path.mkdir(exist_ok=True, parents=True)

    timestamp = datetime.datetime.now().isoformat()
    log_file = log_dir_path / "segmenter_logs.json"

    log_entry = {"timestamp": timestamp, "prompt": prompt, "response": output_text}

    # Read existing log if present
    if log_file.exists():
        with open(log_file, "r") as f:
            try:
                logs = json.load(f)
            except json.JSONDecodeError:
                logs = []
    else:
        logs = []

    # Append new log and save
    logs.append(log_entry)
    with open(log_file, "w") as f:
        json.dump(logs, f, indent=2)


def call_segmenter_api_llm_success(
    lang_description: str,
    interactions: str,
    state_sequence: List,
    sample_every: int = 5,
    llm_log_dir: Optional[str] = None,
):
    # Step 1: Build compressed summary
    compressed_state_sequence = compress_state_sequence(
        state_sequence, sample_every=sample_every, include_first_last=True, verbose=True
    )
    interactions_lst = parse_interactions(interactions)
    compressed_state_sequence = round_floats(compressed_state_sequence)
    prompt = build_prompt_llm_success(
        lang_description,
        interactions,
        len(interactions_lst),
        str(compressed_state_sequence),
    )

    model = "o3-mini"
    # model = "gpt-4o-mini-2024-07-18"
    # Step 2: LLM call
    response = client.responses.create(
        model=model,
        input=prompt,
        reasoning={"effort": "medium"},
        # temperature=0.25,  # recommended for code-generation
    )

    if model == "o3-mini":
        output_text = response.output[1].content[0].text
    else:
        output_text = response.output[0].content[0].text.strip()
    print("Full LLM output:\n", output_text)

    # Log prompt and response if log_dir is provided
    if llm_log_dir:
        log_llm_interaction(
            llm_log_dir=llm_log_dir,
            prompt=prompt,
            output_text=output_text,
        )

    # Step 3: Extract final JSON segment list using regex
    json_match = re.search(r"```json(.*?)```", output_text, re.DOTALL)
    if not json_match:
        print("❌ Could not find a valid JSON code block in LLM output.")
        return output_text, None

    json_text = json_match.group(1).strip()
    try:
        segments = json.loads(json_text)
        # Convert segments from the LLM output format to the expected format:
        # From: [{"type": "motion", "start": start_idx, "end": end_idx}, ...]
        # To: [("motion", (start_idx, end_idx)), ...]
        formatted_segments = []
        for i, segment in enumerate(segments):
            segment_type = segment["label"]
            # let start_idx be the end of the previous segment for now
            if i > 0:
                prev_segment = segments[i - 1]
                start_idx = prev_segment["end"] + 20
            else:
                start_idx = 20
            end_idx = segment["end"]
            formatted_segments.append((segment_type, (start_idx, end_idx)))

        # First, ensure the number of skill segments matches the number of interactions
        if len(formatted_segments) != len(interactions_lst):
            print(
                f"⚠️ Warning: Number of subtask segments ({len(formatted_segments)}) "
                f"does not match number of interactions ({len(interactions_lst)})"
            )
            return None

        new_interactions = interactions_lst.copy()
        # convert all "skill" segments to the corresponding interaction labels
        for i, (_, (start, end)) in enumerate(formatted_segments):
            formatted_segments[i] = ("-".join(new_interactions.pop(0)), (start, end))
        segments = formatted_segments
        return segments
    except json.JSONDecodeError as e:
        print("❌ JSON parsing failed:", e)
        return None


def build_prompt_llm_success(
    lang_description: str, interactions: str, n_interactions: int, compressed: str
) -> str:
    return f"""
You are an expert with great judgement in robotic task segmentation. Your job is to classify the timesteps in a robot demonstration
containing the timesteps the robot successfully and convincingly completes subtasks.

Each state in the demonstration contains:
- Robot end-effector information: position, velocity norm, and gripper state.
- Object information: positions, velocity norms, and, if applicable, joint positions and velocities (for articulated objects).
- Contact information: number of contacts and body pairs in contact.

Definitions: A subtask end_t is the timestep when the robot successfully completes the subtask.
Examples: for picking up objects, end_t might be when object successfully leaves the original surface and is held by the robot.
For opening objects, make sure the end_t is when the object is fully opened or at least
the point where the object is the most opened.

**Guidance on imperfect demonstrations**:
Demonstrations may contain small imperfections, such as momentary breaks in contact or inconsistent contact readings.
Do not treat brief, noisy losses of contact as the success completion of a subtask, since the robot might  have more actions where they more fully complete the task.
Instead, use your judgement to maintain temporal consistency and recognize the robot's high-level intent.

**Subtasks**: "{lang_description}"

There are {n_interactions} subtasks in the demonstration. Please return the timesteps only for the each of the given subtasks.

**Trajectory summary**:
{compressed}


Reason about what is happening in the trajectory, then return your final output.

Include the full list of segments as a JSON array, inside a Markdown code block like this:

```json
[
  {{"label": "<subtask-name>", "end": end_t}},
  ...
]```
This final JSON block should appear after all reasoning steps and must be valid JSON."""


def run_llm_success_segmentation(
    env: Any,
    states: List[Any],
    lang_description: str,
    segments_output_dir: Path,
    interactions: List[str],
    llm_log_dir: Optional[str] = None,
):
    """
    Runs the language-model-based segmentation logic and returns trajectory segments.
    """
    # Acquire model handles (type ignored if mismatch: depends on your environment definitions)
    model, data = env.env.sim.model._model, env.env.sim.data._data  # type: ignore

    NON_RELEVANT_OBJECT_BODY_PREFIXES: List[str] = [
        "gripper",
        "left_eef_target",
        "right_eef_target",
        "robot",
    ]
    ROBOT_GRIPPER_JOINT_NAMES: List[str] = [
        "gripper0_right_finger_joint1",
        "gripper0_right_finger_joint2",
    ]
    EEF_BODY_NAME: str = "gripper0_right_right_gripper"
    relevant_body_names = get_relevant_body_names(
        model, exclude_prefixes=NON_RELEVANT_OBJECT_BODY_PREFIXES
    )
    # objects for which we should query for state information
    task_relevant_object_names = choose_object_names(
        lang_description, relevant_body_names
    )

    # Build the state sequence
    state_sequence = []
    for t in range(len(states)):
        env.reset_to({"states": states[t]}, no_return_obs=True)
        step_state = extract_robot_and_object_state(
            data=data,
            model=model,
            eef_body_name=EEF_BODY_NAME,
            gripper_joint_names=ROBOT_GRIPPER_JOINT_NAMES,
            object_names=task_relevant_object_names
            if isinstance(task_relevant_object_names, list)
            else [],
        )
        state_sequence.append(step_state)

    # Calculate sample_every based on wanting ~60 state descriptions
    # We want to sample overall states evenly
    sample_every = max(1, len(state_sequence) // 60)
    print(
        f"Using sample_every={sample_every} to get approximately 60 state descriptions"
    )

    # Call your language-model segmenter
    segments = call_segmenter_api_llm_success(
        lang_description=lang_description,
        interactions=interactions,
        state_sequence=state_sequence,
        sample_every=sample_every,
        llm_log_dir=llm_log_dir,
    )

    if segments is None:
        print("❌ Segmentation failed.")
        return None

    return segments

# not yet working ...
def build_prompt_llm_ends_then_starts(
    lang_description: str, interactions: str, n_interactions: int, compressed: str
) -> str:
    """
    A single prompt that instructs the LLM:
      - Step 1: find subtask ends
      - Step 2: find subtask starts, referencing step 1
      - Return a final JSON listing each subtask label, start, end
    """
    return f"""
You are an expert in robotic subtask segmentation. We have {n_interactions} subtasks described as follows:
"{lang_description}"

We provide a compressed summary of the trajectory below. Each line includes relevant features like the robot end-effector positions, object states, and contacts:

{compressed}

**Task**:
1. First, identify the timestep `end` for each subtask, i.e., when the subtask is successfully completed.
2. After deciding these ends, identify the timestep `start` for each subtask, referencing the ends you determined. The start is the earliest time when the robot transitions from prior motion (or idle) into the new subtask.

**Important**:
 - The `start` must come before the `end` for each subtask.
 - We have {n_interactions} subtasks. They should appear in the same order you believe they are actually completed in the data.
Please return the timesteps only for the each of the given subtasks.

 - If the demonstration is imperfect or has small breaks, use your best judgment for start/end.
 - The start of a subtask should be after the pure transportation/transit motion has ended.
 - "Transit/transport" means collision-free movement or simply holding an object without active manipulation.
 - The `end' subtask timestep, for certain subtasks where this applies, should end once the robot is no longer in control of the object.

**Final Output**:
At the end of your reasoning, output one code block containing valid JSON. Each subtask must have:
  - "label": subtask label (a short string)
  - "start": integer timestep
  - "end": integer timestep

For example:

```json
[
  {{"label": "pick-object", "start": 10, "end": 45}},
  {{"label": "open-lid",    "start": 46, "end": 90}}
]
That is the only code block you should include. Provide all your reasoning before the code block, but do not include it in the JSON. We only want the final JSON in the code fence. Use the subtask ordering that you deduce from the demonstration. """


def call_segmenter_api_llm_ends_then_starts(
    lang_description: str,
    interactions: str,
    state_sequence: List,
    sample_every: int = 5,
    llm_log_dir: Optional[str] = None,
) -> Optional[List[Tuple[str, Tuple[int, int]]]]:
    """
    Prompts for subtask ends first, then subtask starts, in one shot.
    Returns a list of (subtask_label, (start_t, end_t)) for each subtask.
    """
    # Step 1: Compress
    compressed_state_sequence = compress_state_sequence(
        state_sequence, sample_every=sample_every, include_first_last=True, verbose=True
    )
    compressed_state_sequence = round_floats(compressed_state_sequence)
    interactions_lst = parse_interactions(interactions)

    # Step 2: Build the single prompt
    prompt = build_prompt_llm_ends_then_starts(
        lang_description=lang_description,
        interactions=interactions,
        n_interactions=len(interactions_lst),
        compressed=str(compressed_state_sequence),
    )

    # Step 3: Call the LLM
    model = "o3-mini"
    response = client.responses.create(
        model=model,
        input=prompt,
        reasoning={"effort": "medium"},
    )

    if model == "o3-mini":
        output_text = response.output[1].content[0].text
    else:
        output_text = response.output[0].content[0].text.strip()

    print("Full LLM output:\n", output_text)

    # (Optional) Log the entire conversation
    if llm_log_dir:
        log_llm_interaction(
            llm_log_dir=llm_log_dir,
            prompt=prompt,
            output_text=output_text,
        )

    # Step 4: Extract JSON from code block
    json_match = re.search(r"```json(.*?)```", output_text, re.DOTALL)
    if not json_match:
        print("❌ Could not find a valid JSON code block in LLM output.")
        return None

    json_text = json_match.group(1).strip()
    try:
        segments_data = json.loads(json_text)
        # segments_data should look like:
        # [
        #   {"label": "pick-object", "start": 10, "end": 45},
        #   {"label": "open-lid", "start": 46, "end": 90}
        # ]

        formatted_segments = []
        for seg in segments_data:
            label = seg["label"]
            st = seg["start"]
            en = seg["end"]
            formatted_segments.append((label, (st, en)))

        # Check count matches your subtask count
        if len(formatted_segments) != len(interactions_lst):
            print(
                f"⚠️ # of segments {len(formatted_segments)} != # interactions {len(interactions_lst)}"
            )

        return formatted_segments

    except json.JSONDecodeError as e:
        print("❌ JSON parsing failed:", e)
        return None


def run_llm_two_phase_single_prompt(
    env: Any,
    states: List[Any],
    lang_description: str,
    segments_output_dir: Path,
    interactions: List[str],
    llm_log_dir: Optional[str] = None,
):
    """
    Demonstration of how to do the single-prompt, two-phase (end -> start)
    in one shot, then parse the final JSON.
    """
    # 1) Build state_sequence
    model, data = env.env.sim.model._model, env.env.sim.data._data
    # ... do your environment setup, relevant bodies, etc. ...
    NON_RELEVANT_OBJECT_BODY_PREFIXES: List[str] = [
        "gripper",
        "table",
        "left_eef_target",
        "right_eef_target",
        "robot",
    ]
    relevant_body_names = get_relevant_body_names(
        model, exclude_prefixes=NON_RELEVANT_OBJECT_BODY_PREFIXES
    )
    task_relevant_object_names = choose_object_names(
        lang_description, relevant_body_names
    )
    ROBOT_GRIPPER_JOINT_NAMES: List[str] = [
        "gripper0_right_finger_joint1",
        "gripper0_right_finger_joint2",
    ]
    EEF_BODY_NAME: str = "gripper0_right_right_gripper"
    state_sequence = []
    for t in range(len(states)):
        env.reset_to({"states": states[t]}, no_return_obs=True)
        step_state = extract_robot_and_object_state(
            data=data,
            model=model,
            eef_body_name=EEF_BODY_NAME,
            gripper_joint_names=ROBOT_GRIPPER_JOINT_NAMES,
            object_names=task_relevant_object_names
            if isinstance(task_relevant_object_names, list)
            else [],
        )
        state_sequence.append(step_state)

    # 2) Call the single-prompt approach
    sample_every = max(1, len(state_sequence) // 60)
    segments = call_segmenter_api_llm_ends_then_starts(
        lang_description=lang_description,
        interactions=interactions,
        state_sequence=state_sequence,
        sample_every=sample_every,
        llm_log_dir=llm_log_dir,
    )

    if segments is None:
        print("❌ Could not parse subtask segmentation.")
        return None

    # # 3) (Optional) Save segment videos from (start, end)
    # for (label, (start_t, end_t)) in segments:
    #     filename = segments_output_dir / f"{label}_segment_{start_t}_{end_t}.mp4"
    #     save_segment_rendering(env, states, (start_t, end_t), filename)
    #     print(f"Saved {label} from {start_t}..{end_t} in {filename}")

    return segments
