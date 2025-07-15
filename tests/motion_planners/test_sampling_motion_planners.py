from typing import List

import numpy as np
import pytest

from demo_aug.envs.motion_planners.motion_planning_space import (
    DrakeMotionPlanningSpace,
    IKType,
)
from demo_aug.envs.motion_planners.sampling_based_motion_planner import (
    NearJointsUniformSampler,
    SamplingBasedMotionPlanner,
)


@pytest.fixture
def drake_mp_space() -> DrakeMotionPlanningSpace:
    return DrakeMotionPlanningSpace(
        drake_package_path="/juno/u/thankyou/autom/demo-aug/demo_aug/models/package.xml",
        robot_base_pos=np.array([-0.56, 0, 0.912]),
        robot_base_quat_wxyz=np.array([1, 0, 0, 0]),
        view_meshcat=True,
    )


@pytest.fixture
def planner_drake(
    drake_mp_space: DrakeMotionPlanningSpace,
) -> SamplingBasedMotionPlanner:
    return SamplingBasedMotionPlanner(drake_mp_space)


@pytest.fixture
def collision_free_q1():
    return np.array(
        [
            0.0229177,
            0.19946329,
            -0.01342641,
            -2.63559645,
            0.02568405,
            2.9339680,
            0.79548173,
            0.0,
            0,
        ]
    )


@pytest.fixture
def collision_free_q2(collision_free_q1: np.ndarray) -> np.ndarray:
    return collision_free_q1 + np.ones(9) * 0.05


@pytest.fixture
def q_grip_1() -> np.ndarray:
    return np.array([0.03, 0.03])


@pytest.fixture
def q_grip_2() -> np.ndarray:
    return np.array([0.03, 0.03])


@pytest.fixture
def X_ee_1() -> np.ndarray:
    return np.array(
        [
            [0.99479933, -0.02476076, 0.09879878, -0.1238769],
            [-0.02524072, -0.99967488, 0.00361078, 0.00222318],
            [0.09867725, -0.00608575, -0.99510088, 1.09940147],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


@pytest.fixture
def X_ee_2() -> np.ndarray:
    return np.array(
        [
            [0.99479933, -0.02476076, 0.09879878, 0.1],
            [-0.02524072, -0.99967488, 0.00361078, 0.05],
            [0.09867725, -0.00608575, -0.99510088, 0.98],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


# better name for specifying planner: also, planner is technically agnostic to the space right? that's right
def test_planning_no_collision_1(
    planner_drake: SamplingBasedMotionPlanner,
    drake_mp_space: DrakeMotionPlanningSpace,
    collision_free_q1: np.ndarray,
    collision_free_q2: np.ndarray,
):
    numpy_random = np.random.RandomState(42)
    q1, q2 = collision_free_q1, collision_free_q2
    init_samples = planner_drake.get_init_samples(
        collision_free_q1, collision_free_q2, drake_mp_space
    )
    sampler = NearJointsUniformSampler(
        bias=0.2,
        start_conf=q1,
        goal_conf=q2,
        numpy_random=numpy_random,
        min_values=drake_mp_space.bounds[0],
        max_values=drake_mp_space.bounds[1],
        init_samples=init_samples,
        use_1D_gripper=False,
    )
    path: List[np.ndarray] = planner_drake.get_optimal_trajectory(
        q1, q2, drake_mp_space, sampler, timeout=3
    )

    assert path is not None, (
        "Failed to find a path between two collision free configurations. Motion planner default arguments may need to"
        " be re-tuned."
    )


def test_planning_no_collision_2(
    planner_drake: SamplingBasedMotionPlanner,
    drake_mp_space: DrakeMotionPlanningSpace,
    X_ee_1: np.ndarray,
    X_ee_2: np.ndarray,
    q_grip_1: np.ndarray,
    q_grip_2: np.ndarray,
):
    numpy_random = np.random.RandomState(42)

    q1, is_success = drake_mp_space.inverse_kinematics(
        X_ee_1, q_grip_1, ik_type=IKType.X_EE_TO_Q_ROBOT
    )
    assert is_success, "Failed to find inverse kinematics: hard coded X_ee_1 won't work for the motion planning test"
    q2, is_success = drake_mp_space.inverse_kinematics(
        X_ee_2, q_grip_2, ik_type=IKType.X_EE_TO_Q_ROBOT, q_init=q1
    )
    assert is_success, "Failed to find inverse kinematics: hard coded X_ee_2 won't work for the motion planning test"

    init_samples = planner_drake.get_init_samples(q1, q2, drake_mp_space, n_samples=20)
    sampler = NearJointsUniformSampler(
        bias=0.2,
        start_conf=q1,
        goal_conf=q2,
        numpy_random=numpy_random,
        min_values=drake_mp_space.bounds[0],
        max_values=drake_mp_space.bounds[1],
        init_samples=init_samples,
        use_1D_gripper=False,
    )
    path: List[np.ndarray] = planner_drake.get_optimal_trajectory(
        q1, q2, drake_mp_space, sampler, timeout=3
    )

    assert path is not None, (
        "Failed to find a path between two collision free configurations. Motion planner default arguments may need to"
        " be re-tuned."
    )
