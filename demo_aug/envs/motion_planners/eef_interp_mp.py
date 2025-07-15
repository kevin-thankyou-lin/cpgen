import pathlib
from typing import Optional, Tuple

import mink
import numpy as np
import scipy.spatial.transform as st
from mink import Configuration
from scipy.spatial.transform import Rotation as R

from demo_aug.envs.motion_planners.base_mp import MotionPlanner


class LongPathSlerp:
    def __init__(self, times, rotations):
        self.times = np.asarray(times)
        if isinstance(rotations, R):
            self.rots = rotations
        else:
            self.rots = R.from_quat(rotations)

        if len(self.times) != len(self.rots):
            raise ValueError("Number of rotations must match number of times")

    def __call__(self, factors):
        """
        Interpolate rotations.

        Args:
            factors: Array of interpolation factors between 0 and 1

        Returns:
            Rotation object containing interpolated rotations
        """
        factors = np.asarray(factors)
        if factors.ndim == 0:
            factors = np.array([factors])

        r0 = self.rots[0]
        r1 = self.rots[1]

        q0 = r0.as_quat()
        q1 = r1.as_quat()

        dot = np.sum(q0 * q1)

        if dot > 0:
            q1 = -q1

        theta = np.arccos(np.abs(dot))
        if theta == 0:
            return st.Rotation.from_quat(np.tile(q0, (len(factors), 1)))

        sin_theta = np.sin(theta)
        w0 = np.sin((1.0 - factors) * theta) / sin_theta
        w1 = np.sin(factors * theta) / sin_theta

        # Broadcasting to handle array of times
        q_interp = w0[:, np.newaxis] * q0 + w1[:, np.newaxis] * q1
        q_interp = q_interp / np.linalg.norm(q_interp, axis=1)[:, np.newaxis]

        return st.Rotation.from_quat(q_interp)


def interpolate_poses(
    X_1: np.ndarray, X_2: np.ndarray, num_poses: int, use_long_slerp: bool = False
) -> np.ndarray:
    """Interpolate between two 4x4 poses.

    Long slerp takes the 'longer version' of the shortest path between two quaternions.
    Useful for avoiding joint limits of robot hand (esp. robots like Panda).

    TODO(klin): Move ot motion_planning_space?
    """
    # Extract translation and rotation components from the poses
    T1 = X_1[:3, 3]
    T2 = X_2[:3, 3]
    R1 = X_1[:3, :3]
    R2 = X_2[:3, :3]

    if num_poses == 0:
        return np.array([X_1, X_2])

    factors = np.linspace(0, 1, num_poses)
    # Interpolate translation using linear interpolation
    Ts = (1 - factors[:, np.newaxis]) * T1 + factors[:, np.newaxis] * T2

    if use_long_slerp:
        rotation_slerp = LongPathSlerp([0, 1], R.from_matrix([R1, R2]))
    else:
        rotation_slerp = st.Slerp([0, 1], R.from_matrix([R1, R2]))
    Rs = rotation_slerp(factors).as_matrix()

    # Combine interpolated translation and rotation into a 4x4 pose
    interpolated_poses = np.eye(4)
    interpolated_poses = np.tile(interpolated_poses, (num_poses, 1, 1))

    interpolated_poses[:, :3, 3] = Ts
    interpolated_poses[:, :3, :3] = Rs

    return interpolated_poses


class EEFInterpMotionPlanner(MotionPlanner):
    name: str = "eef_interp"

    def __init__(
        self,
        env,
        save_dir: Optional[pathlib.Path] = None,
        num_steps: Optional[int] = 32,
        max_eef_vel: float = 0.2,
    ):
        super().__init__(env, save_dir)
        self.num_steps = num_steps
        self.dof = 7
        self.max_eef_vel: float = max_eef_vel

    def plan(
        self,
        X_start: np.ndarray,
        X_goal: np.ndarray,
        num_steps: Optional[int] = None,
        use_long_slerp: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # TODO: need to have a joint limit aware interpolation, especially w.r.t the last robot link
        # current fix is to try both and see which one can be tracked by the IK tracker decently. Prefer shorter interp.
        l2_dist = np.linalg.norm(X_goal[:3, 3] - X_start[:3, 3])
        if num_steps is not None:
            num_steps = num_steps
        elif self.num_steps is None:
            num_steps = int(l2_dist / self.max_eef_vel * 20)
        else:
            num_steps = self.num_steps
        interp_poses = interpolate_poses(
            X_start, X_goal, num_steps, use_long_slerp=use_long_slerp
        )
        interp_pos = [pose[:3, 3] for pose in interp_poses]
        interp_quat_xyzw = [
            R.from_matrix(pose[:3, :3]).as_quat() for pose in interp_poses
        ]
        return interp_pos, interp_quat_xyzw

    def visualize_motion(self, configs: np.ndarray):
        return


class EEFInterpMinkMotionPlanner(EEFInterpMotionPlanner):
    name: str = "eef_interp_mink"

    def __init__(
        self,
        env,
        save_dir: Optional[pathlib.Path] = None,
        num_steps: Optional[int] = None,
        max_eef_vel: float = 0.2,
    ):
        self.solver: str = "quadprog"
        self.velocity_limits = (
            np.array([150, 150, 150, 150, 180, 180, 180]) * np.pi / 180
        )
        self.velocity_limits = {
            f"robot0_joint{i + 1}": v for i, v in enumerate(self.velocity_limits)
        }
        self.mink_dt = 1 / 20

        super().__init__(env, save_dir, num_steps=num_steps, max_eef_vel=max_eef_vel)

    def plan(
        self,
        X_start: np.ndarray,
        X_goal: np.ndarray,
        q_start: np.ndarray,
        robot_configuration: Configuration,
        q_retract: np.ndarray,
        eef_frame_name: str = "gripper0_right_grip_site",
        eef_frame_type: str = "site",
        retract_q_weights: Optional[np.ndarray] = None,
        verbose: bool = False,
        num_steps: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Plan a path from X_start to X_goal using mink IK solver.

        Returns:
            robot_qs: List of robot configurations
            eef_wxyz_xyzs: List of eef poses in world frame. These eef poses are not the result of IK, but rather the poses
                that the IK solver is trying to reach. We select between "long" and "short" slerp based on which one has the
                smallest sum of position and quaternion differences.
        """
        tasks = [
            r_ee_task := mink.FrameTask(
                frame_name=eef_frame_name,
                frame_type=eef_frame_type,
                position_cost=1.0,
                orientation_cost=1.0,
                lm_damping=1.0,
            ),
            posture_task := mink.PostureTask(
                model=robot_configuration.model,
                cost=1e-3 if retract_q_weights is None else retract_q_weights / 1e3,
            ),
        ]
        limits = [
            mink.ConfigurationLimit(model=robot_configuration.model),
            mink.VelocityLimit(robot_configuration.model, self.velocity_limits),
            # collision_avoidance_limit,
        ]
        if q_retract is not None:
            posture_task.set_target(q_retract)

        robot_qs_lst = []
        eef_wxyz_xyzs_lst = []

        for use_long_slerp in [False, True]:
            robot_configuration.update(q_start)
            interp_pos, interp_quat_xyzw = super().plan(
                X_start, X_goal, use_long_slerp=use_long_slerp, num_steps=num_steps
            )
            interp_quat_xyzw = np.array(interp_quat_xyzw)
            interp_quat_wxyz = np.roll(interp_quat_xyzw, 1, axis=1)
            interp_wxyz_xyz = np.concatenate([interp_quat_wxyz, interp_pos], axis=1)

            pos_diff_sum = 0
            quat_diff_sum = 0

            robot_qs = []
            eef_wxyz_xyzs = interp_wxyz_xyz
            for i in range(len(interp_pos)):
                pos = interp_pos[i]
                quat_wxyz = np.roll(interp_quat_xyzw[i], 1)
                # set target
                r_ee_task.set_target(mink.SE3(np.concatenate([quat_wxyz, pos])))
                vel = mink.solve_ik(
                    robot_configuration,
                    tasks,
                    self.mink_dt,
                    self.solver,
                    limits=limits,
                    damping=1e-3,
                )
                robot_configuration.integrate_inplace(vel, self.mink_dt)
                # check distance to eef pos and quat_wxyz
                wxyz_xyz = robot_configuration.get_transform_frame_to_world(
                    eef_frame_name, eef_frame_type
                ).wxyz_xyz
                pos_diff = np.linalg.norm(pos - wxyz_xyz[4:])
                quat_diff = np.linalg.norm(quat_wxyz - wxyz_xyz[:4])
                pos_diff_sum += pos_diff
                quat_diff_sum += quat_diff
                if verbose:
                    print(f"pos_diff: {pos_diff}, quat_diff: {quat_diff}")
                robot_qs.append(robot_configuration.q.copy())
                # eef_wxyz_xyzs.append(wxyz_xyz)
            robot_qs_lst.append((robot_qs, pos_diff_sum + quat_diff_sum))
            eef_wxyz_xyzs_lst.append((eef_wxyz_xyzs, pos_diff_sum + quat_diff_sum))

        robot_qs, _ = min(robot_qs_lst, key=lambda x: x[1])
        eef_wxyz_xyzs, _ = min(eef_wxyz_xyzs_lst, key=lambda x: x[1])
        return robot_qs, eef_wxyz_xyzs
