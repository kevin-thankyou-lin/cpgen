# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Camera transformation helper code.
"""

import math
from typing import Dict, List, Literal, Optional, Union

import numpy as np
import torch
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation as R

_EPS = np.finfo(float).eps * 4.0


def make_pose(translation, rotation):
    """
    Makes a homogeneous pose matrix from a translation vector and a rotation matrix.

    Args:
        translation (np.array): (x,y,z) translation value
        rotation (np.array): a 3x3 matrix representing rotation

    Returns:
        pose (np.array): a 4x4 homogeneous matrix
    """
    pose = np.zeros((4, 4))
    pose[:3, :3] = rotation
    pose[:3, 3] = translation
    pose[3, 3] = 1.0
    return pose


def get_camera_intrinsic_matrix(
    sim, camera_name: str, camera_height: int, camera_width: int
) -> np.ndarray:
    """
    Obtains camera intrinsic matrix.

    Args:
        sim (MjSim): simulator instance
        camera_name (str): name of camera
        camera_height (int): height of camera images in pixels
        camera_width (int): width of camera images in pixels
    Return:
        K (np.array): 3x3 camera matrix
    """
    cam_id = sim.model.camera_name2id(camera_name)
    fovy = sim.model.cam_fovy[cam_id]
    f = 0.5 * camera_height / np.tan(fovy * np.pi / 360)
    K = np.array([[f, 0, camera_width / 2], [0, f, camera_height / 2], [0, 0, 1]])
    return K


def compute_fov_y(
    fx: float, fy: float, im_w: int, im_h: int, degrees: bool = False
) -> float:
    # Compute FOV in the y-direction
    fov_y = 2 * np.arctan(im_h / (2 * fy))
    if degrees:
        fov_y = np.rad2deg(fov_y)
    return fov_y


def get_camera_extrinsic_matrix(
    sim, camera_name: str, convention: Literal["opengl", "opencv"] = "opengl"
) -> np.ndarray:
    """
    Returns a 4x4 homogenous matrix corresponding to the camera pose in the
    world frame. MuJoCo has a weird convention for how it sets up the
    camera body axis, so we also apply a correction so that the x and y
    axis are along the camera view and the z axis points along the
    viewpoint.
    Normal camera convention:
    https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html

    Args:
        sim (MjSim): simulator instance
        camera_name (str): name of camera
    Return:
        R (np.array): 4x4 camera extrinsic matrix

    """
    cam_id = sim.model.camera_name2id(camera_name)
    camera_pos = sim.data.cam_xpos[cam_id]
    camera_rot = sim.data.cam_xmat[cam_id].reshape(3, 3)
    R = make_pose(camera_pos, camera_rot)

    # TODO(klin): double check e.g. via viser if need to correct the transforms:
    # think I don't want to convert to opencv?
    if convention == "opencv":
        # IMPORTANT! This is a correction so that the camera axis is set up along the viewpoint correctly.
        camera_axis_correction = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        R = R @ camera_axis_correction
    elif convention == "opengl":
        # assumes that mujoco uses opengl convention
        R = R
    return R


def get_real_depth_map(sim, depth_map: np.ndarray) -> np.ndarray:
    """
    By default, MuJoCo will return a depth map that is normalized in [0, 1]. This
    helper function converts the map so that the entries correspond to actual z-depths.

    (see https://github.com/deepmind/dm_control/blob/master/dm_control/mujoco/engine.py#L742)

    Args:
        sim (MjSim): simulator instance
        depth_map (np.array): depth map with values normalized in [0, 1] (default depth map
            returned by MuJoCo)
    Return:
        depth_map (np.array): depth map that corresponds to actual z-depth
    """
    # Make sure that depth values are normalized
    assert np.all(depth_map >= 0.0) and np.all(depth_map <= 1.0)
    extent = sim.model.stat.extent
    far = sim.model.vis.map.zfar * extent
    near = sim.model.vis.map.znear * extent
    return near / (1.0 - depth_map * (1.0 - near / far))


def get_real_distance_map(
    depth_image: Union[np.ndarray, torch.Tensor],
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """
    Convert depth image to distance image.

    Parameters:
    depth_image (np.ndarray): Depth image
    fx (float): Focal length in x direction
    fy (float): Focal length in y direction
    cx (float): X coordinate of principal point
    cy (float): Y coordinate of principal point

    Returns:
    np.ndarray: Distance image

    Taken from: https://github.com/DLR-RM/BlenderProc/blob/b5195737639f3ff476c93b18be309cf349babd2c/blenderproc/python/postprocessing/PostProcessingUtility.py#L44
    """
    # Determine if input is a tensor or a numpy array
    is_tensor = torch.is_tensor(depth_image)

    if is_tensor:
        device = depth_image.device
        lib = torch
    else:
        lib = np

    # Create meshgrid
    ys, xs = lib.meshgrid(
        lib.arange(depth_image.shape[0]),
        lib.arange(depth_image.shape[1]),
        indexing="ij",
    )
    xs, ys = xs.to(device) if is_tensor else xs, ys.to(device) if is_tensor else ys

    # Coordinate distances to principal point
    x_opt = lib.abs(xs - cx)
    y_opt = lib.abs(ys - cy)

    # Calculate distance
    if fx == fy:
        distance_image = depth_image * lib.sqrt(x_opt**2 + y_opt**2 + fx**2) / fx
    else:
        distance_image = depth_image * lib.sqrt(
            (x_opt**2 / fx**2) + (y_opt**2 / fy**2) + 1
        )

    return distance_image


def unit_vector(data: ArrayLike, axis: Optional[int] = None) -> np.ndarray:
    """Return ndarray normalized by length, i.e. Euclidean norm, along axis.

    Args:
        axis: the axis along which to normalize into unit vector
        out: where to write out the data to. If None, returns a new np ndarray
    """
    data = np.array(data, dtype=np.float64, copy=True)
    if data.ndim == 1:
        data /= math.sqrt(np.dot(data, data))
        return data
    length = np.atleast_1d(np.sum(data * data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    return data


def quaternion_from_matrix(matrix: ArrayLike, isprecise: bool = False) -> np.ndarray:
    """Return quaternion from rotation matrix.

    Args:
        matrix: rotation matrix to obtain quaternion
        isprecise: if True, input matrix is assumed to be precise rotation matrix and a faster algorithm is used.
    """
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    if isprecise:
        q = np.empty((4,))
        t = np.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 1, 2, 3
            if M[1, 1] > M[0, 0]:
                i, j, k = 2, 3, 1
            if M[2, 2] > M[i, i]:
                i, j, k = 3, 1, 2
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        # symmetric matrix K
        K = np.array(
            [
                [m00 - m11 - m22, 0.0, 0.0, 0.0],
                [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
                [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
                [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22],
            ]
        )
        K /= 3.0
        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        q = V[np.array([3, 0, 1, 2]), np.argmax(w)]
    if q[0] < 0.0:
        np.negative(q, q)
    return q


def quaternion_slerp(
    quat0: ArrayLike,
    quat1: ArrayLike,
    fraction: float,
    spin: int = 0,
    shortestpath: bool = True,
) -> np.ndarray:
    """Return spherical linear interpolation between two quaternions.
    Args:
        quat0: first quaternion
        quat1: second quaternion
        fraction: how much to interpolate between quat0 vs quat1 (if 0, closer to quat0; if 1, closer to quat1)
        spin: how much of an additional spin to place on the interpolation
        shortestpath: whether to return the short or long path to rotation
    """
    q0 = unit_vector(quat0[:4])
    q1 = unit_vector(quat1[:4])
    if q0 is None or q1 is None:
        raise ValueError("Input quaternions invalid.")
    if fraction == 0.0:
        return q0
    if fraction == 1.0:
        return q1
    d = np.dot(q0, q1)
    if abs(abs(d) - 1.0) < _EPS:
        return q0
    if shortestpath and d < 0.0:
        # invert rotation
        d = -d
        np.negative(q1, q1)
    angle = math.acos(d) + spin * math.pi
    if abs(angle) < _EPS:
        return q0
    isin = 1.0 / math.sin(angle)
    q0 *= math.sin((1.0 - fraction) * angle) * isin
    q1 *= math.sin(fraction * angle) * isin
    q0 += q1
    return q0


def quaternion_matrix(quaternion: ArrayLike) -> np.ndarray:
    """Return homogeneous rotation matrix from quaternion.

    Args:
        quaternion: value to convert to matrix
    """
    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array(
        [
            [1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0], 0.0],
            [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0], 0.0],
            [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2], 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def get_interpolated_poses(
    pose_a: ArrayLike, pose_b: ArrayLike, steps: int = 10
) -> List[float]:
    """Return interpolation of poses with specified number of steps.
    Args:
        poseA: first pose
        poseB: second pose
        steps: number of steps the interpolated pose path should contain
    """

    quat_a = quaternion_from_matrix(pose_a[:3, :3])
    quat_b = quaternion_from_matrix(pose_b[:3, :3])

    ts = np.linspace(0, 1, steps)
    quats = [quaternion_slerp(quat_a, quat_b, t) for t in ts]
    trans = [(1 - t) * pose_a[:3, 3] + t * pose_b[:3, 3] for t in ts]

    poses_ab = []
    for quat, tran in zip(quats, trans):
        pose = np.identity(4)
        pose[:3, :3] = quaternion_matrix(quat)[:3, :3]
        pose[:3, 3] = tran
        poses_ab.append(pose[:3])
    return poses_ab


def get_transform_matrix(pos: np.ndarray, euler_xyz: np.ndarray) -> np.ndarray:
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = R.from_euler("xyz", euler_xyz).as_matrix()
    transform_matrix[:3, 3] = pos
    return transform_matrix


def convert_opencv_to_opengl(pose_matrix: np.ndarray) -> np.ndarray:
    x_old, y_old, z_old = pose_matrix[:3, 0], pose_matrix[:3, 1], pose_matrix[:3, 2]
    new_x = x_old
    new_y = -y_old
    new_z = -z_old
    pose_matrix[:3, 0] = new_x
    pose_matrix[:3, 1] = new_y
    pose_matrix[:3, 2] = new_z
    return pose_matrix


def convert_opengl_to_opencv(pose_matrix: np.ndarray) -> np.ndarray:
    x_old, y_old, z_old = pose_matrix[:3, 0], pose_matrix[:3, 1], pose_matrix[:3, 2]
    new_x = x_old
    new_y = -y_old
    new_z = -z_old
    pose_matrix[:3, 0] = new_x
    pose_matrix[:3, 1] = new_y
    pose_matrix[:3, 2] = new_z
    return pose_matrix


def lookat_matrix(
    position: np.ndarray,
    target: np.ndarray,
    up: np.ndarray,
    output_camera_convention: Literal["opencv", "opengl"] = "opencv",
) -> np.ndarray:
    # Normalize the vectors
    forward = target - position
    forward /= np.linalg.norm(forward)

    right = np.cross(up, forward)
    right /= np.linalg.norm(right)

    true_up = np.cross(forward, right)

    # Create rotation matrix
    rotation_matrix = np.eye(4)
    rotation_matrix[:3, 0] = right
    rotation_matrix[:3, 1] = true_up
    rotation_matrix[:3, 2] = forward
    rotation_matrix[:3, 3] = position

    if output_camera_convention == "opengl":
        x_old, y_old, z_old = (
            rotation_matrix[:3, 0],
            rotation_matrix[:3, 1],
            rotation_matrix[:3, 2],
        )
        new_x = x_old
        new_y = -y_old
        new_z = -z_old
        rotation_matrix[:3, 0] = new_x
        rotation_matrix[:3, 1] = new_y
        rotation_matrix[:3, 2] = new_z
        rotation_matrix = convert_opencv_to_opengl(rotation_matrix)

    return rotation_matrix


def pos_euler_opencv_to_pos_quat_opengl(
    pos: List[float], euler_xyz: List[float]
) -> Dict[str, List[float]]:
    position = np.array(pos)
    rotation_matrix = R.from_euler("xyz", euler_xyz).as_matrix()

    # Creating pose matrix
    pose_matrix = make_pose(position, rotation_matrix)

    # Converting from OpenCV to OpenGL convention
    pose_matrix = convert_opencv_to_opengl(pose_matrix)

    # Convert pose to pos quat_wxyz
    pos = pose_matrix[:3, 3]
    quat_xyzw = R.from_matrix(pose_matrix[:3, :3]).as_quat()
    quat_wxyz = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]

    return {"pos": pos.tolist(), "quat_wxyz": quat_wxyz}
