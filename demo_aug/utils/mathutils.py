"""A collection of math functions."""

import math
from typing import List, Tuple, Union

import numpy as np
import scipy.spatial.transform as st
import torch
from scipy.spatial.transform import Rotation as R
from torch import Tensor

PI = np.pi
EPS = np.finfo(float).eps * 4.0


def random_rotation_matrix_zyx(
    z_angle_bounds: Tuple[float, float],
    y_angle_bounds: Tuple[float, float],
    x_angle_bounds: Tuple[float, float],
) -> np.ndarray:
    z_angle_deg = np.random.uniform(*z_angle_bounds)
    y_angle_deg = np.random.uniform(*y_angle_bounds)
    x_angle_deg = np.random.uniform(*x_angle_bounds)
    rotation = R.from_euler(
        "zyx", [z_angle_deg, y_angle_deg, x_angle_deg], degrees=True
    )
    return rotation.as_matrix()


def random_translation(
    x_bounds: Tuple[float, float],
    y_bounds: Tuple[float, float],
    z_bounds: Tuple[float, float],
) -> np.ndarray:
    x = np.random.uniform(*x_bounds)
    y = np.random.uniform(*y_bounds)
    z = np.random.uniform(*z_bounds)
    return np.array([x, y, z])


def random_axis_angle(angle_limit=None, random_state=None):
    """
    Samples an axis-angle rotation by first sampling a random axis
    and then sampling an angle. If @angle_limit is provided, the size
    of the rotation angle is constrained.

    If @random_state is provided (instance of np.random.RandomState), it
    will be used to generate random numbers.

    Args:
        angle_limit (None or float): If set, determines magnitude limit of angles to generate
        random_state (None or RandomState): RNG to use if specified

    Raises:
        AssertionError: [Invalid RNG]

    Note: taken from robosuite repo
    """
    if angle_limit is None:
        angle_limit = 2.0 * np.pi

    if random_state is not None:
        assert isinstance(random_state, np.random.RandomState)
        npr = random_state
    else:
        npr = np.random

    # sample random axis using a normalized sample from spherical Gaussian.
    # see (http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/)
    # for why it works.
    random_axis = npr.randn(3)
    random_axis /= np.linalg.norm(random_axis)
    random_angle = npr.uniform(low=0.0, high=angle_limit)
    return random_axis, random_angle


def random_pose(
    pos_bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
    max_angle: float,
) -> np.ndarray:
    pos = random_translation(*pos_bounds)
    random_axis, random_angle = random_axis_angle(max_angle)
    rot = quat2mat(axisangle2quat(random_axis * random_angle))
    pose = np.eye(4)
    pose[:3, :3] = rot
    pose[:3, 3] = pos
    return pose


def make_pose(pos: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = pos
    return pose


def biased_sampling(
    scale_range: Union[List, Tuple], alpha: float = 0.7, beta: float = 0.7
):
    """
    Sample from the given range with a bias towards the bounds.

    Parameters:
    - scale_range: tuple indicating the range.
    - alpha, beta: parameters of the beta distribution. When both are less than 1,
      the sampling will be biased towards the bounds.

    Returns:
    A sampled value from scale_range with bias towards its bounds.
    """
    # Sample from beta distribution
    sample = np.random.beta(alpha, beta)

    # Rescale to the desired range
    return scale_range[0] + sample * (scale_range[1] - scale_range[0])


def axis_angle_to_quaternion(axis: Tensor, angle: Tensor) -> Tensor:
    """
    Convert axis-angle to quaternion for a batch of samples.
    """
    w = torch.cos(angle / 2.0)
    xyz = axis * torch.sin(angle / 2.0)
    return torch.cat([w.unsqueeze(-1), xyz], dim=-1)


def rotmat_to_quat(rotmat: torch.Tensor) -> torch.Tensor:
    """Converts a batch of rotation matrices to quaternions in wxyz format."""
    K = rotmat.shape[0]
    q = torch.empty((K, 4), dtype=rotmat.dtype, device=rotmat.device)
    m00, m01, m02 = rotmat[:, 0, 0], rotmat[:, 0, 1], rotmat[:, 0, 2]
    m10, m11, m12 = rotmat[:, 1, 0], rotmat[:, 1, 1], rotmat[:, 1, 2]
    m20, m21, m22 = rotmat[:, 2, 0], rotmat[:, 2, 1], rotmat[:, 2, 2]
    q[:, 0] = (
        torch.sqrt(
            torch.maximum(
                torch.tensor(0.0, device=rotmat.device), 1.0 + m00 + m11 + m22
            )
        )
        / 2.0
    )
    q[:, 1] = torch.copysign(
        torch.sqrt(
            torch.maximum(
                torch.tensor(0.0, device=rotmat.device), 1.0 + m00 - m11 - m22
            )
        )
        / 2.0,
        m21 - m12,
    )
    q[:, 2] = torch.copysign(
        torch.sqrt(
            torch.maximum(
                torch.tensor(0.0, device=rotmat.device), 1.0 - m00 + m11 - m22
            )
        )
        / 2.0,
        m02 - m20,
    )
    q[:, 3] = torch.copysign(
        torch.sqrt(
            torch.maximum(
                torch.tensor(0.0, device=rotmat.device), 1.0 - m00 - m11 + m22
            )
        )
        / 2.0,
        m10 - m01,
    )
    return q


def sample_random_rotation(
    num_samples: int, theta_max: Tensor
) -> Tuple[Tensor, Tensor]:
    """
    Sample random rotation axis and angle for a batch of samples.
    """
    axis = torch.randn(num_samples, 3)
    axis /= torch.norm(axis, dim=1, keepdim=True)  # Normalize to unit vector

    angle = (
        2 * torch.rand(num_samples) - 1
    ) * theta_max  # Uniformly between [-theta_max, theta_max]

    return axis, angle


def apply_perturbation(
    source_quaternion: Tensor, num_samples: int, theta_max: Tensor
) -> Tensor:
    axis, angle = sample_random_rotation(num_samples, theta_max)
    perturb_quaternions = axis_angle_to_quaternion(axis, angle)

    # Quaternion multiplication for all samples in parallel
    w1, x1, y1, z1 = source_quaternion.split(1, dim=-1)
    w2, x2, y2, z2 = perturb_quaternions.split(1, dim=-1)

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    rotated_quaternions = torch.cat([w, x, y, z], dim=-1)

    return rotated_quaternions


def compute_quaternion_distances(quat: List[np.ndarray]) -> List[float]:
    """Given a list of quaternions, compute the distance between each neighboring pair of quaternions."""
    quat_distances = []
    for i in range(len(quat) - 1):
        # Normalize the quaternions
        q1 = quat[i] / np.linalg.norm(quat[i])
        q2 = quat[i + 1] / np.linalg.norm(quat[i + 1])

        if np.allclose(q1, q2):
            quat_distances.append(0.0)
            continue

        dot_product = np.dot(q1, q2)

        # Compute angular distance
        angle_distance = 2 * np.arccos(abs(dot_product))
        quat_distances.append(angle_distance)

        # check if angle_distance is nan
        assert not np.isnan(angle_distance), "angle_distance is nan"

    return quat_distances


def rotation_matrix_distance(R1: np.ndarray, R2: np.ndarray) -> float:
    """
    Compute the angle between two rotation matrices.

    Parameters:
    - R1: First rotation matrix (3x3).
    - R2: Second rotation matrix (3x3).

    Returns:
    - Angle in radians between the two rotation matrices.
    """
    R = np.dot(R1, R2.T)
    # Handle edge case where trace(R) is close to 3
    theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    return theta


def eval_error(
    pose_src: np.ndarray,
    pose_tgt: np.ndarray,
) -> Tuple[float, float]:
    """Compute the distance and angle difference between two 4x4 poses."""
    # Compute distance between positions
    offset = pose_src[:3, 3] - pose_tgt[:3, 3]
    dist = np.linalg.norm(offset)

    # Convert quaternions to rotation matrices
    rot_src = pose_src[:3, :3]
    rot_tgt = pose_tgt[:3, :3]

    # Compute angle difference between rotations
    angle_diff = rotation_matrix_distance(rot_src, rot_tgt)
    angle_diff = np.degrees(angle_diff)

    return dist, angle_diff


def interpolate_poses(X_1: np.ndarray, X_2: np.ndarray, num_poses: int) -> np.ndarray:
    """Interpolate between two 4x4 poses.

    TODO(klin): Move ot motion_planning_space?
    """
    # Extract translation and rotation components from the poses
    T1 = X_1[:3, 3]
    T2 = X_2[:3, 3]
    R1 = X_1[:3, :3]
    R2 = X_2[:3, :3]

    factors = np.linspace(0, 1, num_poses)
    # Interpolate translation using linear interpolation
    Ts = (1 - factors[:, np.newaxis]) * T1 + factors[:, np.newaxis] * T2

    # Interpolate rotation using spherical linear interpolation (slerp)
    rotation_slerp = st.Slerp([0, 1], R.from_matrix([R1, R2]))
    Rs = rotation_slerp(factors).as_matrix()

    # Combine interpolated translation and rotation into a 4x4 pose
    interpolated_poses = np.eye(4)
    interpolated_poses = np.tile(interpolated_poses, (num_poses, 1, 1))

    interpolated_poses[:, :3, 3] = Ts
    interpolated_poses[:, :3, :3] = Rs

    return interpolated_poses


def multiply_with_X_transf(
    X_transf: torch.Tensor, x: Union[torch.Tensor, np.ndarray]
) -> Union[torch.Tensor, np.ndarray]:
    """Multiplies a tensor with up to 3 dims with a transformation matrix (4, 4)."""
    # assert shape of last entry of x is 4
    assert x.shape[-1] == 4, "Last dimension of X_transf must be 4."
    assert X_transf.shape == (4, 4), "Last dimension of X_transf must be 4."

    convert_output_to_numpy = False
    if isinstance(x, np.ndarray):
        if isinstance(X_transf, torch.Tensor):
            X_transf = (
                X_transf.clone()
                .detach()
                .to(device=torch.device("cpu"), dtype=torch.float32)
            )
        convert_output_to_numpy = True
        x = torch.tensor(x, device=X_transf.device, dtype=X_transf.dtype)

    # Get the dimensions of x
    x_dims = x.dim()

    # Reshape X_transf to match the dimensions of x
    if x_dims == 2:
        X_transf = X_transf.unsqueeze(0)
    elif x_dims == 3:
        X_transf = X_transf.unsqueeze(0).unsqueeze(0)

    # Perform matrix multiplication
    result = torch.matmul(X_transf, x.unsqueeze(-1))

    # Remove the extra dimension
    result = result.squeeze(-1)

    # Convert to numpy if necessary
    if convert_output_to_numpy:
        result = result.detach().cpu().numpy()

    return result


def random_z_rotation(
    theta_min: float,
    theta_max: float,
    use_biased_sampling: bool = False,
    alpha: float = 0.7,
    beta: float = 0.7,
) -> np.ndarray:
    if use_biased_sampling:
        theta = biased_sampling([theta_min, theta_max], alpha, beta)
    else:
        theta = np.random.uniform(theta_min, theta_max)

    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array(
        [[cos_theta, -sin_theta, 0], [sin_theta, cos_theta, 0], [0, 0, 1]]
    )
    return rotation_matrix


def get_timesteps(total_time: float, interval_length: float) -> np.ndarray:
    """
    Returns a list of timesteps for a given total time and interval length between timesteps.

    If the remainder of total_time / interval_length is greater than interval_length / 4, then
    the total_time is added as a timestep.
    """
    num_intervals = total_time / interval_length
    remainder = total_time % interval_length
    timesteps = np.arange(num_intervals) * interval_length

    if remainder > interval_length / 4:
        timesteps = np.append(timesteps, total_time)

    return timesteps


def random_rotation_matrix(max_angle_deg: float) -> np.ndarray:
    """Generates a random 3x3 rotation matrix with angle of rotation
    at most max_angle_deg."""
    from scipy.spatial.transform import Rotation

    # Convert maximum angle to radians
    max_angle_rad = np.radians(max_angle_deg)

    # Generate random rotation vector
    random_vector = np.random.randn(3)
    random_vector /= np.linalg.norm(random_vector)

    # Generate random rotation angle
    random_angle = np.random.uniform(-max_angle_rad, max_angle_rad)

    # Create rotation object and retrieve rotation matrix
    rotation = Rotation.from_rotvec(random_angle * random_vector)
    rot_matrix = rotation.as_matrix()

    return rot_matrix


def change_target_coordinate_frame_of_transformation_matrix(
    matrix, new_frame: List[str]
) -> np.ndarray:
    """Changes the coordinate frame the given transformation matrix is mapping to.

    Given a matrix $T_A^B$ that maps from A to B, this function can be used
    to change the axes of B into B' and therefore end up with $T_A^B'$.

    :param matrix: The matrix to convert in form of a np.ndarray or mathutils.Matrix
    :param new_frame: An array containing three elements, describing each axis of the new coordinate frame
                      based on the axes of the current frame. Available: ["X", "Y", "Z", "-X", "-Y", "-Z"].
    :return: The converted matrix is in form of a np.ndarray
    """
    tmat = MathUtility.build_coordinate_frame_changing_transformation_matrix(new_frame)

    # Apply transformation matrix
    output = np.matmul(tmat, matrix)
    return output


def change_source_coordinate_frame_of_transformation_matrix(
    matrix, new_frame: list
) -> np.ndarray:
    """Changes the coordinate frame the given transformation matrix is mapping from.

    Given a matrix $T_A^B$ that maps from A to B, this function can be used
    to change the axes of A into A' and therefore end up with $T_A'^B$.

    :param matrix: The matrix to convert in form of a np.ndarray or mathutils.Matrix
    :param new_frame: An array containing three elements, describing each axis of the new coordinate frame
                      based on the axes of the current frame. Available: ["X", "Y", "Z", "-X", "-Y", "-Z"].
    :return: The converted matrix is in form of a np.ndarray
    """
    tmat = MathUtility.build_coordinate_frame_changing_transformation_matrix(new_frame)
    tmat = np.linalg.inv(tmat)

    # Apply transformation matrix
    output = np.matmul(matrix, tmat)
    return output


class MathUtility:
    """
    Math utility class
    """

    @staticmethod
    def build_coordinate_frame_changing_transformation_matrix(
        destination_frame: List[str],
    ) -> np.ndarray:
        """Builds a transformation matrix that switches the coordinate frame.

        :param destination_frame: An array containing three elements, describing each axis of the destination
                                  coordinate frame based on the axes of the source frame.
                                  Available: ["X", "Y", "Z", "-X", "-Y", "-Z"].
        :return: The transformation matrix
        """
        assert (
            len(destination_frame) == 3
        ), f"The specified coordinate frame has more or less than tree axes: {destination_frame}"

        # Build transformation matrix that maps the given matrix to the specified coordinate frame.
        tmat = np.zeros((4, 4))
        for i, axis in enumerate(destination_frame):
            axis = axis.upper()

            if axis.endswith("X"):
                tmat[i, 0] = 1
            elif axis.endswith("Y"):
                tmat[i, 1] = 1
            elif axis.endswith("Z"):
                tmat[i, 2] = 1
            else:
                raise Exception("Invalid axis: " + axis)

            if axis.startswith("-"):
                tmat[i] *= -1
        tmat[3, 3] = 1
        return tmat


def mat2quat(rmat: np.ndarray):
    """
    Converts given rotation matrix to quaternion.

    Args:
        rmat (np.array): 3x3 rotation matrix

    Returns:
        np.array: (x,y,z,w) float quaternion angles
    """
    M = np.asarray(rmat).astype(np.float32)[:3, :3]

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
            [m00 - m11 - m22, np.float32(0.0), np.float32(0.0), np.float32(0.0)],
            [m01 + m10, m11 - m00 - m22, np.float32(0.0), np.float32(0.0)],
            [m02 + m20, m12 + m21, m22 - m00 - m11, np.float32(0.0)],
            [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22],
        ]
    )
    K /= 3.0
    # quaternion is Eigen vector of K that corresponds to largest eigenvalue
    w, V = np.linalg.eigh(K)
    inds = np.array([3, 0, 1, 2])
    q1 = V[inds, np.argmax(w)]
    if q1[0] < 0.0:
        np.negative(q1, q1)
    inds = np.array([1, 2, 3, 0])
    return q1[inds]


# https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/utils/transform_utils.py
def axisangle2quat(vec: np.ndarray):
    """
    Converts scaled axis-angle to quat.
    Args:
        vec (np.array): (ax,ay,az) axis-angle exponential coordinates
    Returns:
        np.array: (x,y,z,w) vec4 float angles
    """
    # Grab angle
    angle = np.linalg.norm(vec)

    # handle zero-rotation case
    if math.isclose(angle, 0.0):
        return np.array([0.0, 0.0, 0.0, 1.0])

    # make sure that axis is a unit vector
    axis = vec / angle

    q = np.zeros(4)
    q[3] = np.cos(angle / 2.0)
    q[:3] = axis * np.sin(angle / 2.0)
    return q


def mat2axisangle(rmat: np.ndarray) -> np.ndarray:
    """
    Converts given rotation matrix to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        rmat (np.array): 3x3 rotation matrix

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    quat = mat2quat(rmat)
    return quat2axisangle(quat)


def quat2axisangle(quat: np.ndarray):
    """
    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


# @jit_decorator
def quat2mat(quaternion):
    """
    Converts given quaternion to matrix.
    Args:
        quaternion (np.array): (x,y,z,w) vec4 float angles
    Returns:
        np.array: 3x3 rotation matrix
    """
    # awkward semantics for use with numba
    inds = np.array([3, 0, 1, 2])
    q = np.asarray(quaternion).copy().astype(np.float32)[inds]

    n = np.dot(q, q)
    if n < EPS:
        return np.identity(3)
    q *= math.sqrt(2.0 / n)
    q2 = np.outer(q, q)
    return np.array(
        [
            [1.0 - q2[2, 2] - q2[3, 3], q2[1, 2] - q2[3, 0], q2[1, 3] + q2[2, 0]],
            [q2[1, 2] + q2[3, 0], 1.0 - q2[1, 1] - q2[3, 3], q2[2, 3] - q2[1, 0]],
            [q2[1, 3] - q2[2, 0], q2[2, 3] + q2[1, 0], 1.0 - q2[1, 1] - q2[2, 2]],
        ]
    )
