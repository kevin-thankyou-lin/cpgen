import json
from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R


def euler_xyz_to_rpy(euler_xyz: np.ndarray) -> Tuple[float, float, float]:
    """
    URDF files use rpy, while camera calibration often saves as certain euler angles.
    """
    # https://web.mit.edu/2.05/www/Handout/HO2.PDF
    # Create a rotation object from 'XYZ' (extrinsic) Euler angles
    rotation = R.from_euler("xyz", euler_xyz, degrees=False)

    # Convert the rotation object to 'ZYX' (intrinsic) Euler angles
    ypr = rotation.as_euler("ZYX", degrees=False)

    # Roll (φ), Pitch (θ), Yaw (ψ)
    return [ypr[2], ypr[1], ypr[0]]


def euler_xyz_to_quat_wxyz(euler_xyz: np.ndarray) -> Tuple[float, float, float]:
    # https://web.mit.edu/2.05/www/Handout/HO2.PDF
    # Create a rotation object from 'XYZ' (extrinsic) Euler angles
    rotation = R.from_euler("xyz", euler_xyz, degrees=False)

    # Convert the rotation object to a quaternion
    quat_xyzw = rotation.as_quat()
    quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
    return quat_wxyz


def rpy_to_quat_wxyz(rpy: np.ndarray) -> Tuple[float, float, float, float]:
    # Create a rotation object from RPY (intrinsic ZYX) Euler angles
    rotation = R.from_euler("ZYX", rpy, degrees=False)

    # Convert the rotation object to a quaternion
    quat_xyzw = rotation.as_quat()
    quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
    return tuple(quat_wxyz)


if __name__ == "__main__":
    path = "scripts/real_world/camera_calibration_info/calibration_info.json"
    camera_name = "12391924"
    with open(path, "r") as file:
        calibration_info = json.load(file)

    left_camera_pose = calibration_info[camera_name + "_left"]["pose"]
    right_camera_pose = calibration_info[camera_name + "_right"]["pose"]
    left_camera_pose = np.array(left_camera_pose)
    right_camera_pose = np.array(right_camera_pose)

    left_camera_pos, left_camera_euler_xyz = left_camera_pose[:3], left_camera_pose[3:]
    right_camera_pos, right_camera_euler_xyz = (
        right_camera_pose[:3],
        right_camera_pose[3:],
    )
    center_pos = (left_camera_pos + right_camera_pos) / 2

    # set printoptions
    np.set_printoptions(precision=6, suppress=True)

    print("left pos: ", left_camera_pos)
    print("right pos: ", right_camera_pos)
    print("center pos: ", center_pos)
    print("center pos: ", center_pos)

    left_camera_rpy = euler_xyz_to_rpy(left_camera_euler_xyz)
    print("left rpy: ", left_camera_rpy)
    right_camera_rpy = euler_xyz_to_rpy(right_camera_euler_xyz)
    print("right rpy: ", right_camera_rpy)

    left_camera_quat_wxyz = euler_xyz_to_quat_wxyz(left_camera_euler_xyz)
    print("left quat: ", left_camera_quat_wxyz)
    right_camera_quat_wxyz = euler_xyz_to_quat_wxyz(right_camera_euler_xyz)
    print("right quat: ", right_camera_quat_wxyz)
