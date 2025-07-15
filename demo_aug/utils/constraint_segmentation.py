import numpy as np


def is_open_drawer_start(tau: dict, object_names: list[str]) -> int:
    """Returns true when the robot begins opening the drawer."""
    drawer = object_names[0]
    contacts = tau["contacts"]

    # Check for contact between the gripper fingers and the drawer handle.
    is_contact = any(
        ("gripper0_right" in body1 and drawer in body2)
        or ("gripper0_right" in body2 and drawer in body1)
        for body1, body2 in contacts
    )

    return is_contact


def is_open_drawer_end(tau: dict, object_names: list[str]) -> int:
    """Returns true when the drawer has been opened, determined by joint position."""
    drawer_joint_name = "DrawerObject_goal_slidey"
    drawer_pos = tau["qpos"].get(drawer_joint_name, None)

    if drawer_pos is None:
        return False

    # Consider drawer opened if the position is near the upper limit
    return drawer_pos <= -0.13  # Empirical threshold for significant opening


def is_pick_up_mug_start(tau: dict, object_names: list[str]) -> int:
    """Returns true when the robot starts picking up the mug."""
    mug = object_names[0]
    contacts = tau["contacts"]

    # Detect initial grasping contact
    is_contact = any(
        ("gripper0_right" in body1 and mug in body2)
        or ("gripper0_right" in body2 and mug in body1)
        for body1, body2 in contacts
    )

    return is_contact


def is_pick_up_mug_end(tau: dict, object_names: list[str]) -> int:
    """Returns true when the robot successfully lifts the mug off the surface."""
    mug_pose = tau["pose"].get(object_names[0], None)

    if mug_pose is None:
        return False

    # Consider the mug lifted if the z position has increased significantly
    return mug_pose[2] > 0.82 + 0.04  # Assuming threshold height for lifting


def is_place_mug_into_drawer_start(tau: dict, object_names: list[str]) -> int:
    """Returns true when the mug is placed into the drawer."""
    mug, drawer = object_names
    contacts = tau["contacts"]

    # Check contact between the mug and the drawer interior
    is_contact = any(
        (mug in body1 and drawer in body2) or (mug in body2 and drawer in body1)
        for body1, body2 in contacts
    )

    return is_contact


def is_place_mug_into_drawer_end(tau: dict, object_names: list[str]) -> int:
    """Returns true when the mug is stabilized in the drawer, determined by low velocity."""
    mug_velocity = tau["qvel"].get(object_names[0], None)
    object_names = ["cleanup_object_joint0"]
    mug_velocity = tau["qvel"].get(object_names[0], None)
    if mug_velocity is None:
        return False

    # Consider placement complete when the object's velocity is near zero
    return np.linalg.norm(mug_velocity) < 0.01  # Small velocity threshold


def is_close_drawer_start(tau: dict, object_names: list[str]) -> int:
    """Returns true when the robot begins closing the drawer."""
    drawer = object_names[0]
    contacts = tau["contacts"]

    # Check if the gripper is in contact with the drawer handle
    is_contact = any(
        ("gripper0_right" in body1 and drawer in body2)
        or ("gripper0_right" in body2 and drawer in body1)
        for body1, body2 in contacts
    )
    # Check if drawer joint starts moving
    drawer_joint_name = "DrawerObject_goal_slidey"
    drawer_vel = tau["qvel"].get(drawer_joint_name, None)
    if drawer_vel is None:
        return False

    # Consider the drawer closing when the joint velocity is positive
    return is_contact and drawer_vel > 0.01  # Empirical threshold for movement


def is_close_drawer_end(tau: dict, object_names: list[str]) -> int:
    """Returns true when the drawer is fully closed, determined by joint position."""
    drawer_joint_name = "DrawerObject_goal_slidey"
    drawer_pos = tau["qpos"].get(drawer_joint_name, None)

    if drawer_pos is None:
        return False

    # Consider the drawer closed when the position reaches near the lower limit
    is_closed = drawer_pos >= -0.005  # Empirical threshold for closure

    # Also check if the drawer's velocity is near zero to ensure it's not just bouncing
    drawer_vel = tau["qvel"].get(drawer_joint_name, None)
    if drawer_vel is None:
        return is_closed  # If no velocity data, rely on position

    is_stable = np.abs(drawer_vel) < 0.01  # Velocity threshold for stability

    return is_closed and is_stable


def is_open_drawer_start(tau: dict, body_names: list[str]) -> int:
    """Returns true when robot begins opening the drawer."""
    drawer = "CabinetObject_drawer_link"
    print(tau["contacts"])
    breakpoint()
    is_contact = any(
        "gripper" in body1 and drawer in body2 or "gripper" in body2 and drawer in body1
        for body1, body2 in tau["contacts"]
    )
    print(is_contact)
    return is_contact


def is_open_drawer_end(tau: dict, body_names: list[str]) -> int:
    """Returns true when the drawer is sufficiently open."""
    drawer_joint = "CabinetObject_goal_slidey"
    drawer_pos = tau["qpos"].get(drawer_joint, 0)
    return drawer_pos < -0.8  # Assuming -1.0 is fully open


def is_pick_up_hammer_start(tau: dict, body_names: list[str]) -> int:
    """Returns true when robot begins picking up the hammer."""
    hammer = "hammer_root"
    is_contact = any(
        "gripper" in body1 and hammer in body2 or "gripper" in body2 and hammer in body1
        for body1, body2 in tau["contacts"]
    )
    return is_contact


def is_pick_up_hammer_end(tau: dict, body_names: list[str]) -> int:
    """Returns true when the hammer is lifted off its initial position."""
    hammer = "hammer_root"
    hammer_z = tau["pose"].get(hammer, (0, 0, 0, 0, 0, 0, 0))[2]
    return hammer_z > 0.1  # Assuming a threshold for lifting


def is_place_hammer_into_drawer_start(tau: dict, body_names: list[str]) -> int:
    """Returns true when the hammer begins placement into the drawer."""
    drawer = "CabinetObject_drawer_link"
    hammer = "hammer_root"
    is_contact = any(
        hammer in body1 and drawer in body2 or hammer in body2 and drawer in body1
        for body1, body2 in tau["contacts"]
    )
    return is_contact


def is_place_hammer_into_drawer_end(tau: dict, body_names: list[str]) -> int:
    """Returns true when the hammer is placed into the drawer and stabilized."""
    hammer = "hammer_root"
    hammer_vel = np.linalg.norm(tau["velocity"].get(hammer, (0, 0, 0, 0, 0, 0))[:3])
    gripper_open = (
        tau["qpos"].get("gripper0_right_finger_joint1", 0) > 0.03
    )  # Assuming open threshold
    return hammer_vel < 0.01 and gripper_open


def is_close_drawer_start(tau: dict, body_names: list[str]) -> int:
    """Returns true when robot begins closing the drawer."""
    drawer = "CabinetObject_drawer_link"
    is_contact = any(
        "gripper" in body1 and drawer in body2 or "gripper" in body2 and drawer in body1
        for body1, body2 in tau["contacts"]
    )
    return is_contact


def is_close_drawer_end(tau: dict, body_names: list[str]) -> int:
    """Returns true when the drawer is fully closed."""
    drawer_joint = "CabinetObject_goal_slidey"
    drawer_pos = tau["qpos"].get(drawer_joint, 0)
    return drawer_pos >= -0.05  # Assuming near zero means closed
