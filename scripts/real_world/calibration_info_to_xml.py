import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.spatial.transform import Rotation as R

from demo_aug.utils.camera_utils import (
    convert_opencv_to_opengl,
    make_pose,
    pos_euler_opencv_to_pos_quat_opengl,
)


def read_json(file_path: Path) -> Dict:
    with file_path.open("r") as file:
        data = json.load(file)
    return data


def euler_to_quat_wyxz(euler_angles: List[float]) -> List[float]:
    rotation = R.from_euler("xyz", euler_angles)
    quat_xyzw = rotation.as_quat()
    quat_wyxz = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
    return quat_wyxz


def extract_pose(data: Dict) -> Dict[str, Dict[str, List[float]]]:
    result = {}
    for key, value in data.items():
        pose = value["pose"]
        pos = pose[:3]
        euler_angles = pose[3:]
        quat_wyxz = euler_to_quat_wyxz(euler_angles)
        result[key] = {"pos": pos, "quat_wyxz": quat_wyxz}
    return result


def process_pose(
    pos: List[float], quat_wyxz: List[float], convert_to_opengl: bool = False
) -> Dict[str, List[float]]:
    position = np.array(pos)
    quat_xyzw = [quat_wyxz[1], quat_wyxz[2], quat_wyxz[3], quat_wyxz[0]]
    rotation_matrix = R.from_quat(quat_xyzw).as_matrix()

    # Creating pose matrix
    pose_matrix = make_pose(position, rotation_matrix)

    # Converting from OpenCV to OpenGL convention if needed
    if convert_to_opengl:
        pose_matrix = convert_opencv_to_opengl(pose_matrix)

    # Convert pose to pos quat_wxyz
    pos = pose_matrix[:3, 3]
    quat_xyzw = R.from_matrix(pose_matrix[:3, :3]).as_quat()
    quat_wxyz = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]

    return {"pos": pos.tolist(), "quat_wxyz": quat_wxyz}


def main():
    file_path = Path("demo_aug/hardware/camera_extrinsics/calibration_info.json")
    data = read_json(file_path)
    poses = extract_pose(data)
    for key, value in poses.items():
        processed_pose = process_pose(
            value["pos"], value["quat_wyxz"], convert_to_opengl=True
        )
        # convert lists to strings w/o the commas
        formatted_pos = " ".join(f"{p:.5f}" for p in processed_pose["pos"])
        formatted_quat_wxyz = " ".join(f"{q:.5f}" for q in processed_pose["quat_wxyz"])
        template_str = '{key}: pos="{pos}" quat="{quat_wxyz}"'
        print(
            template_str.format(
                key=key, pos=formatted_pos, quat_wxyz=formatted_quat_wxyz
            )
        )
        out = pos_euler_opencv_to_pos_quat_opengl(
            data["12391924_left"]["pose"][:3], data["12391924_left"]["pose"][3:]
        )
        print("pos: ", out["pos"])
        print("quat_wxyz_opengl: ", out["quat_wxyz"])


if __name__ == "__main__":
    main()
