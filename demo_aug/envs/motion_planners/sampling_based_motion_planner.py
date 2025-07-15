import logging
import time
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import numpy as np
import scipy.interpolate as interpolate
from klampt.plan.cspace import CSpace, MotionPlan
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from demo_aug.envs.motion_planners.base_motion_planner import (
    BaseMotionPlanner,
    MotionPlanningSpace,
)
from demo_aug.envs.motion_planners.motion_planning_space import (
    DrakeMotionPlanningSpace,
    IKType,
)
from demo_aug.utils.mathutils import interpolate_poses, random_z_rotation


@dataclass
class SamplingBasedMotionPlanningCfg:
    type: Literal["lazyprm*", "rrtconnect"] = "lazyprm*"


class RobotConfigSampler:
    def __init__(
        self,
        start_conf: np.ndarray,
        goal_conf: np.ndarray,
        min_values: np.ndarray,
        max_values: np.ndarray,
        numpy_random: np.random.RandomState,
        init_samples: Optional[List[np.ndarray]] = None,
        use_1D_gripper: bool = False,
    ):
        self.start_conf = start_conf
        self.goal_conf = goal_conf
        self.min_values = min_values
        self.max_values = max_values
        self.value_range = self.max_values - self.min_values
        self.numpy_random = numpy_random
        self.init_samples = init_samples
        self.curr_sample_idx = 0
        self.use_1D_gripper = use_1D_gripper

        if use_1D_gripper:
            assert (
                self.start_conf[-1] == self.start_conf[-2]
                and self.goal_conf[-1] == self.goal_conf[-2]
            ), "2D gripper qpos must be at the same value for both gripper q values to use 1D gripper sampling"
            self.start_conf = self.start_conf[:-1]
            self.goal_conf = self.goal_conf[:-1]

    def __call__(self):
        if self.init_samples is not None and self.curr_sample_idx < len(
            self.init_samples
        ):
            self.curr_sample_idx += 1
            return self.init_samples[self.curr_sample_idx - 1]
        return self.numpy_random.uniform(low=self.min_values, high=self.max_values)


class NearJointsNormalSampler(RobotConfigSampler):
    def __init__(self, bias: float, **kwargs):
        self.bias = bias
        super().__init__(**kwargs)

    def __call__(self):
        if self.numpy_random.random() > 0.5:
            return super().__call__()
        center = self.goal_conf if self.numpy_random.random() > 0.5 else self.start_conf
        sample = (
            center
            + self.numpy_random.randn(len(center))
            * (self.max_values - self.min_values)
            * self.bias
        )
        return np.clip(sample, a_min=self.min_values, a_max=self.max_values)


class NearJointsUniformSampler(RobotConfigSampler):
    def __init__(self, bias: float, **kwargs):
        self.bias = bias
        super().__init__(**kwargs)

    def __call__(self):
        if self.numpy_random.random() > 0.5:
            return super().__call__()
        center = self.goal_conf if self.numpy_random.random() > 0.5 else self.start_conf
        sample = self.numpy_random.uniform(
            low=np.clip(
                center - self.bias * self.value_range,
                a_min=self.min_values,
                a_max=self.max_values,
            ),
            high=np.clip(
                center + self.bias * self.value_range,
                a_min=self.min_values,
                a_max=self.max_values,
            ),
        )
        return sample


class NearPoseUniformSampler(RobotConfigSampler):
    def __init__(
        self,
        bias: float,
        normalize_quat: bool,
        quat_start_idx: int,
        quat_end_idx: int,
        **kwargs,
    ):
        self.bias = bias
        self.normalize_quat = normalize_quat
        self.quat_start_idx = quat_start_idx
        self.quat_end_idx = quat_end_idx
        super().__init__(**kwargs)

    def __call__(self):
        if self.numpy_random.random() > 0.5:
            return super().__call__()
        center = self.goal_conf if self.numpy_random.random() > 0.5 else self.start_conf
        sample = self.numpy_random.uniform(
            low=np.clip(
                center - self.bias * self.value_range,
                a_min=self.min_values,
                a_max=self.max_values,
            ),
            high=np.clip(
                center + self.bias * self.value_range,
                a_min=self.min_values,
                a_max=self.max_values,
            ),
        )
        if self.normalize_quat:
            sample[self.quat_start_idx : self.quat_end_idx] /= np.linalg.norm(
                sample[self.quat_start_idx : self.quat_end_idx]
            )
        return sample


class SamplingBasedMotionPlannerCSpace(CSpace):
    """A wrapper for an underlying MotionPlanningSpace that implements the CSpace interface for use with klampt.plan.cspace.MotionPlan."""

    def __init__(
        self, motion_planning_space: MotionPlanningSpace, sampler: RobotConfigSampler
    ):
        # super().__init__()
        CSpace.__init__(self)

        self.motion_planning_space = motion_planning_space
        self.sampler = sampler
        self.interp_type = "configuration"

        self.bound = np.array(self.motion_planning_space.bounds).T

    def feasible(self, q: np.ndarray) -> bool:
        # check if q is in bounds
        if not np.all(np.logical_and(self.bound[:, 0] <= q, q <= self.bound[:, 1])):
            return False
        return not self.motion_planning_space.is_collision(q)

    def visible(self, q1: np.ndarray, q2: np.ndarray) -> bool:
        return self.motion_planning_space.is_visible(q1, q2)

    def sample(self) -> np.ndarray:
        return self.sampler()

    def sampleneighborhood(self, q: np.ndarray, r: float) -> np.ndarray:
        raise NotImplementedError(
            "sampleneighborhood not implemented (properly);that said, lazyprm* doesn't use samplenighborhood"
        )
        return self.motion_planning_space.sample_neighborhood(q, r)

    def distance(self, q1: np.ndarray, q2: np.ndarray) -> float:
        # check that all of q1[-2:] is close to 0.04
        if not np.all(np.isclose(q1[-2:], 0.04)) or not np.all(
            np.isclose(q2[-2:], 0.04)
        ):
            import ipdb

            ipdb.set_trace()
        return self.motion_planning_space.distance(
            q1,
            q2,
            env_dist_factor=self.motion_planning_space.env_dist_factor,
            env_influence_distance=self.motion_planning_space.env_influence_distance,
        )

    def interpolate(self, q1: np.ndarray, q2: np.ndarray, u: float) -> np.ndarray:
        """
        Note: think lazyprm* doesn't call interpolate if we've visible() already implemented
        """
        raise NotImplementedError(
            "interpolate not implemented (properly); unused by lazyprm* given visible()is implemented"
        )
        return self.motion_planning_space.interpolate(q1, q2, u, self.interp_type)


class SamplingBasedMotionPlanner(BaseMotionPlanner):
    def __init__(self, motion_planning_space: Optional[MotionPlanningSpace] = None):
        """
        Note: the only time we use an instance of this class would be when we want to re-use a road map.
        Currently not storing the roadmap due to Klampt's lack of support for that for lazyprm*
        """
        self.motion_planning_space = motion_planning_space

    @staticmethod
    def smooth_path(
        path: List[np.ndarray],
        motion_planning_space: DrakeMotionPlanningSpace,  # should really have a 'robot' motion planning space
        np_random: np.random.RandomState,
        iterations: int = 25,
    ):
        smoothed_path = path
        for _ in range(iterations):
            if len(smoothed_path) <= 2:
                return smoothed_path
            i = np_random.randint(0, len(smoothed_path) - 1)
            j = np_random.randint(0, len(smoothed_path) - 1)
            if abs(i - j) <= 1:
                continue
            if j < i:
                i, j = j, i
            shortcut = list(
                motion_planning_space.extend_configs(
                    smoothed_path[i], smoothed_path[j], extend_type="end_effector"
                )
            )
            if (len(shortcut) < (j - i)) and all(
                not motion_planning_space.is_collision(q) for q in shortcut
            ):
                smoothed_path = (
                    smoothed_path[: i + 1] + shortcut + smoothed_path[j + 1 :]
                )
        return smoothed_path

    @staticmethod
    def discretize_path(
        path: List[np.ndarray],
        motion_planning_space: MotionPlanningSpace,
        resolution: float = 0.004,
        interpolation_type: Literal["configuration", "end_effector"] = "end_effector",
    ) -> List[np.ndarray]:
        """
        Given a path of configurations, discretize the path so that consecutive configurations are at most resolution apart.

        TODO(klin): should check if things are in collisions post discretization.
        """
        new_path: List[np.ndarray] = [path[0]]
        for i in range(len(path) - 1):
            # TODO(klin) @ here fix
            new_path.extend(
                motion_planning_space.extend_configs(
                    path[i],
                    path[i + 1],
                    resolution=resolution,
                    extend_type=interpolation_type,
                )[1:]
            )
        return new_path

    def get_init_samples(
        self,
        start_cfg: np.ndarray,
        goal_cfg: np.ndarray,
        motion_planning_space: MotionPlanningSpace,
        n_samples: int = 100,
        sample_near_goal: bool = False,
        sample_near_pregrasp: bool = False,
        sample_along_eef_interp: bool = False,
        eef_interp_n_samples: int = 100,
        sample_along_sampled_eef_interp: bool = False,
        eef_interp_eef_n_samples: int = 20,
        check_collisions: bool = False,
        sample_top_down: bool = False,
    ) -> List[np.ndarray]:
        """
        Warm starting the motion planner with (not necessarily collision free) samples.
        """
        assert (
            motion_planning_space.robot_type == "hand"
        ), "get_init_samples() currently supports hand robot type"

        init_samples: List[np.ndarray] = []
        q_gripper_fixed = start_cfg[-2:]
        X_ee_start, X_ee_goal = np.eye(4), np.eye(4)
        X_ee_start[:3, 3] = start_cfg[4:7]
        X_ee_start[:3, :3] = R.from_quat(np.roll(start_cfg[0:4], shift=-1)).as_matrix()
        X_ee_goal[:3, 3] = goal_cfg[4:7]
        X_ee_goal[:3, :3] = R.from_quat(np.roll(goal_cfg[0:4], shift=-1)).as_matrix()

        if sample_along_eef_interp:
            q_hand_gripper_interp: List[np.ndarray] = []

            # interpolate between the two poses
            interpolated_poses = interpolate_poses(
                X_ee_start, X_ee_goal, eef_interp_n_samples
            )

            hand_pos = np.array([X_ee[:3, 3] for X_ee in interpolated_poses])
            hand_quats_xyzw = np.array(
                [R.from_matrix(X_ee[:3, :3]).as_quat() for X_ee in interpolated_poses]
            )
            hand_quats = np.roll(hand_quats_xyzw, shift=1, axis=1)

            # repeat the start gripper qpos
            q_gripper = np.tile(q_gripper_fixed, (eef_interp_n_samples, 1))
            # q_hand_gripper_interp = np.concatenate([hand_pos, hand_quats, q_gripper], axis=1)
            q_hand_gripper_interp = np.concatenate(
                [hand_quats, hand_pos, q_gripper], axis=1
            )

            init_samples.extend(q_hand_gripper_interp)

            if sample_along_sampled_eef_interp:
                init_samples.extend(
                    [
                        motion_planning_space.sample_neighborhood(q, 0.2)
                        for q in q_hand_gripper_interp
                        for _ in range(eef_interp_eef_n_samples)
                    ]
                )
            # check gripper is correct
            if not np.all(np.isclose(np.array(init_samples)[..., -2:], 0.04)):
                print(f"init_samples: {init_samples}")
                import ipdb

                ipdb.set_trace()

        if sample_top_down:
            for z_height in np.arange(0, 0.3 + 1e-4, 0.05):
                for z_angle in np.arange(-np.pi / 2, np.pi / 2 + 1e-4, np.pi / 4):
                    z_orn = random_z_rotation(z_angle, z_angle)
                    before_goal_pose = np.eye(4)
                    before_goal_pose[:3, 3] = X_ee_goal[:3, 3] + np.array(
                        [0, 0, z_height]
                    )
                    before_goal_pose[:3, :3] = z_orn @ X_ee_goal[:3, :3]

                    after_start_pose = np.eye(4)
                    after_start_pose[:3, 3] = X_ee_start[:3, 3] + np.array(
                        [0, 0, z_height]
                    )
                    after_start_pose[:3, :3] = z_orn @ X_ee_start[:3, :3]

                    intermediate_pose = np.eye(4)
                    intermediate_pose[:3, 3] = (
                        before_goal_pose[:3, 3] + after_start_pose[:3, 3]
                    ) / 2
                    intermediate_pose[:3, :3] = z_orn @ X_ee_start[:3, :3]

                    init_samples.append(
                        np.concatenate(
                            [
                                np.roll(
                                    R.from_matrix(before_goal_pose[:3, :3]).as_quat(),
                                    shift=1,
                                ),
                                before_goal_pose[:3, 3],
                                q_gripper_fixed,
                            ]
                        )
                    )
                    init_samples.append(
                        np.concatenate(
                            [
                                np.roll(
                                    R.from_matrix(after_start_pose[:3, :3]).as_quat(),
                                    shift=1,
                                ),
                                after_start_pose[:3, 3],
                                q_gripper_fixed,
                            ]
                        )
                    )
                    init_samples.append(
                        np.concatenate(
                            [
                                np.roll(
                                    R.from_matrix(intermediate_pose[:3, :3]).as_quat(),
                                    shift=1,
                                ),
                                intermediate_pose[:3, 3],
                                q_gripper_fixed,
                            ]
                        )
                    )

        if sample_near_goal:

            def perturb_pose(
                X_ee: np.ndarray,
                dx: float,
                dy: float,
                dz: float,
                dthetax: float,
                dthetay: float,
                dthetaz: float,
            ) -> np.ndarray:
                X_ee_perturbed = X_ee.copy()
                X_ee_perturbed[:3, 3] += np.array([dx, dy, dz])
                X_ee_perturbed[:3, :3] = (
                    R.from_euler("xyz", [dthetax, dthetay, dthetaz]).as_matrix()
                    @ X_ee_perturbed[:3, :3]
                )
                return X_ee_perturbed

            dx_values = np.arange(-0.04, 0.04 + 1e-4, 0.04)
            dy_values = np.arange(-0.04, 0.04 + 1e-4, 0.04)
            dz_values = np.arange(0, 0.04 + 1e-4, 0.02)
            dthetax_values = np.arange(-np.pi / 8, np.pi / 8 + 1e-8, np.pi / 12)
            dthetay_values = np.arange(-np.pi / 8, np.pi / 8 + 1e-8, np.pi / 12)
            dthetaz_values = np.arange(-np.pi / 8, np.pi / 8 + 1e-8, np.pi / 12)

            X_hand_near_goal: List[np.ndarray] = [
                perturb_pose(X_ee_goal, dx, dy, dz, dthetax, dthetay, dthetaz)
                for dx in dx_values
                for dy in dy_values
                for dz in dz_values
                for dthetax in dthetax_values
                for dthetay in dthetay_values
                for dthetaz in dthetaz_values
                # for X_ee in [X_ee_goal]
            ]
            X_hand_near_goal.append(X_ee_goal.copy())

            q_hand_gripper_near_goal: List[np.ndarray] = [
                np.concatenate(
                    [
                        np.roll(R.from_matrix(X_ee[:3, :3]).as_quat(), shift=1),
                        X_ee[:3, 3],
                        q_gripper_fixed,
                    ]
                )
                for X_ee in X_hand_near_goal
            ]

            init_samples.extend(q_hand_gripper_near_goal)
        # check the lengths of the samples, ensure length 9
        assert all(
            len(q) == 9 for q in init_samples
        ), "sample_along_sampled_eef_interp has wrong length"

        if not np.all(np.isclose(np.array(init_samples)[..., -2:], 0.04)):
            import ipdb

            ipdb.set_trace()

        # filter out all samples that aren't in bounds
        # init_samples = np.array(init_samples)
        # # original samples
        # n_original_samples = init_samples.shape[0]
        # lb, up = self.motion_planning_space.bounds
        # in_bounds = np.all((init_samples >= lb) & (init_samples <= up), axis=1)
        # init_samples = init_samples[in_bounds]
        # n_post_filtering_samples = init_samples.shape[0]
        # print(f"Filtered out {n_original_samples - n_post_filtering_samples} / {n_original_samples} samples")
        # TODO(klin): why not filter out samples here since we can parallelize here but not during PRM sampling?
        return init_samples

        assert not sample_along_eef_interp, "sample_along_eef_interp uses sample_neighborhoodwhich isn't implemented correctly yet"
        # TODO(klin): include gripper qpos too?
        # start line in end effector space: interpolate and get IKs
        X_ee_start = motion_planning_space.get_end_effector_pose(start_cfg)
        X_ee_goal = motion_planning_space.get_end_effector_pose(goal_cfg)

        init_samples = []

        eef_interp_configs: List[np.ndarray] = []
        if sample_along_eef_interp:
            # interpolate between the two poses
            interpolated_poses = interpolate_poses(
                X_ee_start, X_ee_goal, eef_interp_n_samples
            )
            for X_ee in interpolated_poses:
                config = motion_planning_space.inverse_kinematics(
                    X_ee,
                    start_cfg,
                    IKType.X_EE_TO_Q_ROBOT,
                    goal_cfg,
                    min_dist=0 if not check_collisions else 0.01,
                    min_dist_thresh=0 if not check_collisions else 0.03,
                )[0]
                if config is not None:
                    eef_interp_configs.append(config)
            init_samples.extend(eef_interp_configs)
            if sample_along_sampled_eef_interp:
                for config in eef_interp_configs:
                    for _ in range(eef_interp_eef_n_samples):
                        init_samples.extend(
                            [motion_planning_space.sample_neighborhood(config, 0.1)]
                        )

        if sample_top_down:
            for z_height in np.arange(0, 0.3 + 1e-4, 0.05):
                for z_angle in np.arange(-np.pi / 2, np.pi / 2 + 1e-4, np.pi / 4):
                    z_orn = random_z_rotation(z_angle, z_angle)
                    before_goal_pose = np.eye(4)
                    before_goal_pose[:3, 3] = X_ee_goal[:3, 3] + np.array(
                        [0, 0, z_height]
                    )
                    before_goal_pose[:3, :3] = z_orn @ X_ee_goal[:3, :3]
                    config = motion_planning_space.inverse_kinematics(
                        before_goal_pose,
                        ik_type=IKType.X_EE_TO_Q_ROBOT,
                        q_grip=np.array([0.04, 0.04]),
                    )[0]
                    if config is not None:
                        init_samples.append(config)
                        continue
                    after_start_pose = np.eye(4)
                    after_start_pose[:3, 3] = X_ee_start[:3, 3] + np.array(
                        [0, 0, z_height]
                    )
                    after_start_pose[:3, :3] = z_orn @ X_ee_start
                    config = motion_planning_space.inverse_kinematics(
                        after_start_pose,
                        q_grip=np.array([0.04, 0.04]),
                        ik_type=IKType.X_EE_TO_Q_ROBOT,
                    )[0]
                    if config is not None:
                        init_samples.append(config)
                    intermediate_pose = np.eye(4)
                    intermediate_pose[:3, 3] = (
                        before_goal_pose[:3, 3] + after_start_pose[:3, 3]
                    ) / 2
                    intermediate_pose[:3, :3] = z_orn @ X_ee_start

                    config = motion_planning_space.inverse_kinematics(
                        intermediate_pose,
                        q_grip=np.array([0.04, 0.04]),
                        ik_type=IKType.X_EE_TO_Q_ROBOT,
                    )[0]
                    if config is not None:
                        init_samples.append(config)

        if sample_near_goal:
            goal_samples = 0
            # loop has 15625 data points; potentially use fastIK to speed up
            for dx in np.arange(-0.1, 0.1 + 1e-4, 0.05):
                for dy in np.arange(-0.1, 0.1 + 1e-4, 0.05):
                    for dz in np.arange(-0.1, 0.1 + 1e-4, 0.05):
                        for dthetax in np.arange(
                            -np.pi / 4, np.pi / 4 + 1e-4, np.pi / 8
                        ):
                            for dthetay in np.arange(
                                -np.pi / 4, np.pi / 4 + 1e-4, np.pi / 8
                            ):
                                for dthetaz in np.arange(
                                    -np.pi / 4, np.pi / 4 + 1e-4, np.pi / 8
                                ):
                                    X_ee_goal_perturbed = X_ee_goal.copy()
                                    X_ee_goal_perturbed[:3, 3] += np.array([dx, dy, dz])
                                    X_ee_goal_perturbed[:3, :3] = (
                                        R.from_euler(
                                            "xyz", [dthetax, dthetay, dthetaz]
                                        ).as_matrix()
                                        @ X_ee_goal_perturbed[:3, :3]
                                    )
                                    config = motion_planning_space.inverse_kinematics(
                                        X_ee_goal_perturbed,
                                        ik_type=IKType.X_EE_TO_Q_ROBOT,
                                        q_grip=np.array([0.04, 0.04]),
                                        q_init=goal_cfg,
                                        debug=False,
                                        min_dist_thresh=0,
                                        save_to_file=False,
                                    )[0]
                                    if config is not None:
                                        init_samples.append(config)
                                        goal_samples += 1
                print(f"goal_samples: {goal_samples}")

        return init_samples

    # following two methods are specific to robot gripper
    def get_open_gripper_trajectory(
        self,
        start_cfg: np.ndarray,
        motion_planning_space: MotionPlanningSpace,
    ) -> Optional[List[np.ndarray]]:
        """
        Returns trajectory corresponding to keeping all else still up opening gripper.
        """
        # get the open gripper qpos
        open_gripper_qpos = motion_planning_space.get_open_gripper_qpos()

        # get open gripper q_arm_gripper
        open_gripper_q_arm_gripper = np.concatenate([start_cfg[:-2], open_gripper_qpos])

        # get the open gripper trajectory
        open_gripper_trajectory = motion_planning_space.extend_configs(
            start_cfg,
            open_gripper_q_arm_gripper,
            resolution=0.02,
            extend_type="configuration",
        )

        # check there're no collisions in the trajectory
        if any(motion_planning_space.is_collision(q) for q in open_gripper_trajectory):
            return None

        return open_gripper_trajectory

    def get_close_gripper_trajectory(
        self,
        start_cfg: np.ndarray,
        q_gripper_goal: np.ndarray,
        motion_planning_space: MotionPlanningSpace,
    ) -> Optional[List[np.ndarray]]:
        """
        Returns trajectory corresponding to keeping all else still up opening gripper.
        """
        min_gripper, max_gripper = motion_planning_space.q_gripper_bounds()

        # get close gripper q_arm_gripper
        close_gripper_q_arm_gripper = np.concatenate([start_cfg[:-2], q_gripper_goal])

        # get the close gripper trajectory
        close_gripper_trajectory = motion_planning_space.extend_configs(
            start_cfg,
            close_gripper_q_arm_gripper,
            resolution=0.02,
            extend_type="configuration",
        )

        # check there're no collisions in the trajectory
        if any(motion_planning_space.is_collision(q) for q in close_gripper_trajectory):
            return None

        return close_gripper_trajectory

    def get_optimal_trajectory(
        self,
        start_cfg: np.ndarray,
        goal_cfg: np.ndarray,
        motion_planning_space: MotionPlanningSpace,
        sampler: RobotConfigSampler,
        planner_type: Literal["lazyprm*", "rrtconnect"] = "lazyprm*",
        timeout: float = 20,
        increment: int = 1000,
        max_no_improvement_iters: int = 5,
        debug: bool = True,
        Q_near_goal: Optional[List[np.ndarray]] = None,
        use_manual_splining_hand: bool = True,
    ) -> Optional[Tuple[List[np.ndarray], float]]:
        """
        Returns a list of configurations from start_cfg to goal_cfg.
        start_cfg, goal_cfg both include the arm and gripper qpos.

        Args:
            Q_near_goal: set of configurations that are 'near' the goal configuration.

        Bug 0: current radius measurements for PRM is definitely off ... so basically running lazy prm (not *);
            alleviate by reducing gripper dim from sampling space
        """
        if debug:
            print(f"Begining motion planning from {start_cfg} to {goal_cfg}")

        self.cspace = SamplingBasedMotionPlannerCSpace(motion_planning_space, sampler)
        self.planner = MotionPlan(
            self.cspace, type=planner_type, connectionThreshold=0.1
        )
        self.planner.setEndpoints(start_cfg, goal_cfg)

        start_time = time.time()
        previous_cost = float("inf")

        curr_no_improvement_iters = 0
        while time.time() - start_time < timeout:
            self.planner.planMore(increment)
            path = self.planner.getPath()
            if not path:
                if debug:
                    print(self.planner.getStats())
                    print(
                        f"Running for {time.time() - start_time:.2f} / {timeout} seconds, no path found yet"
                    )
                    print(f"MotionPlan.planner.getStats(): {self.planner.getStats()}")
                    print(f"CSpace.getStats(): {CSpace.getStats(self)}")
                    print(f"Get roadmap: {len(self.planner.getRoadmap()[0])}")
                continue
            else:
                current_cost = self.planner.pathCost(path)

            if debug:
                print(self.planner.getStats())
                logging.info(f"Path: {path}")
                logging.info(
                    f"MotionPlan.planner.getStats(): {self.planner.getStats()}"
                )
                logging.info(f"MotionPlan.planner.pathCost(path): {current_cost:.2f}")
                print(f"MotionPlan.planner.getStats(): {self.planner.getStats()}")
                print(f"MotionPlan.planner.pathCost(path): {current_cost:.2f}")
                print(f"Get roadmap: {len(self.planner.getRoadmap()[0])}")

            if previous_cost <= current_cost:
                print(
                    f"Cost reduction is negligible: {previous_cost} -> {current_cost}; curr_no_improvement_iters:"
                    f" {curr_no_improvement_iters}"
                )
                curr_no_improvement_iters += 1
                if curr_no_improvement_iters >= max_no_improvement_iters:
                    print("Reached max num no improvement, using current path")
                    break
                print(
                    f"Cost reduction is negligible: {previous_cost} -> {current_cost}"
                )
            else:
                print(
                    f"Cost reduction is not negligible: {previous_cost} -> {current_cost}"
                )
                curr_no_improvement_iters = 0

            previous_cost = current_cost

        motion_planning_space.set_q(goal_cfg)

        if path is not None:
            print(
                f"Found a path of length {len(path)} in {time.time() - start_time:.2f} seconds"
            )
            X_ees = [motion_planning_space.get_end_effector_pose(q) for q in path]
            distances = [
                np.linalg.norm(X_ee[:3, 3] - X_ees[i + 1][:3, 3])
                for i, X_ee in enumerate(X_ees[:-1])
            ]
            logging.info(f"l2 distance between eef positions: {distances}")
            print(
                f"post smoothing: l2 distance between eef positions: {[round(d, 3) for d in distances]}"
            )
            path_cost = self.planner.pathCost(path)
            logging.info(f"MotionPlan.planner.pathCost(path): {path_cost}")
            print(f"MotionPlan.planner.pathCost(path): {path_cost}")
        else:
            return None, -np.inf

        if debug:
            if motion_planning_space.robot_type == "hand":
                path = np.array(path)

        if use_manual_splining_hand:
            distances = [
                motion_planning_space.distance(path[i], path[i + 1])
                for i in range(len(path) - 1)
            ]
            total_distance = sum(distances)

            # max_vel = 0.075
            max_vel = 0.1
            max_t = total_distance / max_vel

            times = np.concatenate([[0], np.cumsum(distances) / total_distance]) * max_t
            rotation_slerp = Slerp(
                times, R.from_quat([np.roll(q[:4], shift=-1) for q in path])
            )

            ts = np.arange(0, max_t, 0.2)
            ts = np.append(ts, max_t)  # include final time to include the final config

            interp_quats = rotation_slerp(ts)
            interp_positions = interpolate.interp1d(
                times, np.array(path)[:, 4:7], axis=0
            )(ts)
            interp_gripper = interpolate.interp1d(times, np.array(path)[:, 7:], axis=0)(
                ts
            )
            interp_positions = np.array(interp_positions)
            interp_quats = np.roll(interp_quats.as_quat(), shift=1, axis=1)
            new_path = np.concatenate(
                (interp_quats, interp_positions, interp_gripper), axis=1
            )

            path = new_path
            logging.info(f"new path {path}")

            for q in path:
                if motion_planning_space.is_collision(q):
                    logging.info(
                        "manually splined path is not collision free; skipping"
                    )
                    return None, -np.inf

            for q1, q2 in zip(path[:-1], path[1:]):
                if not motion_planning_space.is_visible(q1, q2):
                    logging.info(
                        "manually splined path hasn't have all 'visible edges'; skipping"
                    )
                    return None, -np.inf
            print(f"new path length: {len(path)}")
        return path, previous_cost
