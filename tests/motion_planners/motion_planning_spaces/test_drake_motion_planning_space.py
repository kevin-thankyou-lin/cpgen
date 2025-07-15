import pathlib

import numpy as np
import pytest

import demo_aug
from demo_aug.envs.motion_planners.motion_planning_space import DrakeMotionPlanningSpace


@pytest.fixture
def drake_mp_space():
    return DrakeMotionPlanningSpace(
        drake_package_path=str(
            pathlib.Path(demo_aug.__file__).parent / "models/package.xml"
        ),
        robot_base_pos=np.array([-0.56, 0, 0.912]),
        robot_base_quat_wxyz=np.array([1, 0, 0, 0]),
    )


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
    return collision_free_q1 + np.ones(9) * 0.1


def test_default_joint_config_collisions(
    drake_mp_space: DrakeMotionPlanningSpace, collision_free_q1: np.ndarray
):
    assert not drake_mp_space.is_collision(collision_free_q1)


def test_default_edge_collisions(
    drake_mp_space: DrakeMotionPlanningSpace,
    collision_free_q1: np.ndarray,
    collision_free_q2: np.ndarray,
):
    assert drake_mp_space.is_visible(collision_free_q1, collision_free_q2)


def test_distance_between_two_configs(
    drake_mp_space: DrakeMotionPlanningSpace,
    collision_free_q1: np.ndarray,
    collision_free_q2: np.ndarray,
):
    dist = drake_mp_space.distance(
        collision_free_q1, collision_free_q2, "end_effector_l2"
    )
    print(f"Distance between two configs: {dist}")
    assert dist > 0, "Distance should be greater than 0"
