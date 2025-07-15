You are an expert in analyzing robot demonstration trajectories. Your task is to generate Python functions
that detect interaction start and stop times for the following subtasks.

### Subtasks:
$subtask_description

### Data Available:
- body_names: $body_names
- body_to_joint_info (body to joint info, joint info contains joint name, joint type, dofs and joint limits): $json_joints_dict

- `tau`: Timestep info dict, containing:
  - `tau["qpos"]`: Joint positions. Keys must be joint names from Joint Dictionary.
  - `tau["qvel"]`: Joint velocities. Keys must be joint names from Joint Dictionary.
  - `tau["pose"]`: Body poses `{body_name: (x, y, z, qw, qx, qy, qz)}`.
  - `tau["velocity"]`: Body velocities `{body_name: (x, y, z, angular_x, angular_y, angular_z)}`.
  - `tau["contacts"]`: List of active contacts between provided mujoco bodies at current timestep.


For each subtask, generate two functions similar to the example below for detecting start and end times, using the naming convention:
is_<subtask_name>_start is_<subtask_name>_end
where <subtask_name> is normalized by replacing spaces with underscores.


### Example Function for "Pick up Nut":

body_names: robot0_base, Apple_main, gripper0_right_leftfinger, nut_main, drawer

import numpy as np

def is_pick_up_nut_start(tau: dict, body_names: list[str]) -> int:
    """Returns true when robot begins pick_up_nut."""
    nut = "nut_main"
    is_contact = any("gripper" in body1 and nut in body2 or "gripper" in body2 and nut in body1 for body1, body2 in contacts)
    return is_contact

def is_pick_up_nut_end(tau: dict, body_names: list[str]) -> int:
    """Returns true when the robot stops pick_up_nut.
    In this case, picking start and end have the same criteria, because picking is binary."""
    return pick_up_nut_start(tau, body_names)

Additional Requirements:

1. Use contact information (contacts) for robot-object constraints to check when interaction starts.
2. For place subtasks, assume placement occurs placement using stabilization (low velocity) and object placement starts when gripper opens.
3. For subtasks with articulated objects, try to use joint information if possible. For drawer, 0 is closed.
4. Return only Python code (including docstrings and code-level comments). No explanations, notes, or additional formatting.
