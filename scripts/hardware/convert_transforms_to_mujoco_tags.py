import json

import numpy as np


def pose_matrix_to_mujoco_camera(pose_matrix):
    """
    Convert a 4x4 pose matrix to MuJoCo camera pos and quat.

    Args:
    - pose_matrix (np.array): 4x4 pose matrix.

    Returns:
    - dict: Contains 'pos' and 'quat' for the MuJoCo camera tag.
    """
    # Extract position
    pos = pose_matrix[:3, 3]

    # Extract rotation matrix
    R = pose_matrix[:3, :3]

    # Calculate quaternion
    qw = np.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2]) / 2.0
    qx = (R[2, 1] - R[1, 2]) / (4 * qw)
    qy = (R[0, 2] - R[2, 0]) / (4 * qw)
    qz = (R[1, 0] - R[0, 1]) / (4 * qw)

    quat = np.array([qw, qx, qy, qz])

    return {"pos": pos, "quat": quat}


# Function to generate MuJoCo camera tag from a 4x4 pose matrix
def generate_mujoco_camera_tag(
    name, pose_matrix, fovy="90", position_scale: float = 0.1
):
    camera_info = pose_matrix_to_mujoco_camera(np.array(pose_matrix))
    pos_str = " ".join(map(str, camera_info["pos"] * position_scale))
    quat_str = " ".join(map(str, camera_info["quat"]))
    return f'<camera name="{name}" pos="{pos_str}" quat="{quat_str}" fovy="{fovy}" />'


if __name__ == "__main__":
    # load transform data from a path
    path = "/home/thankyou/autom/demo-aug/models/two_cams_description/transforms.json"
    with open(path, "r") as f:
        transform_data = json.load(f)

    # Loop through each item in transform_data, convert to MuJoCo camera tag for those with a matrix
    mujoco_camera_tags = [
        generate_mujoco_camera_tag(item["name"], item["matrix"])
        for item in transform_data
        if item["matrix"] is not None
    ]

    for t in mujoco_camera_tags:
        print(t)
