"""
Usage: pytest tests/utils/test_subsample_for_training.py
"""

from demo_aug.utils.obj_utils import subsample_timestamps_for_training_nerf


def test_subsample_combinations_case1():
    obj_timestamps_and_image_timestamps_for_training = [
        (5, [1, 2, 3]),
        (5, [5]),
        (10, [8, 9, 10]),
        (15, [13, 14, 15]),
    ]

    obj_timestamps_and_image_timestamps_for_training = [
        (obj_timestamp, tuple(image_timestamps))
        for obj_timestamp, image_timestamps in obj_timestamps_and_image_timestamps_for_training
    ]
    image_ts_to_trained_nerf_paths = {
        (1, 2, 3): "/path/for/previous/nerf",
        (5,): "/path/for/previous/nerf",
    }

    K = 10

    subsampled_combinations = subsample_timestamps_for_training_nerf(
        obj_timestamps_and_image_timestamps_for_training,
        image_ts_to_trained_nerf_paths,
        K,
    )
    assert subsampled_combinations == [(10, (8, 9, 10))]


def test_subsample_combinations_case2():
    obj_timestamps_and_image_timestamps_for_training = [
        (5, [1, 2, 3]),
        (5, [5]),
        (10, [8, 9, 10]),
        (15, [13, 14, 15]),
    ]

    obj_timestamps_and_image_timestamps_for_training = [
        (obj_timestamp, tuple(image_timestamps))
        for obj_timestamp, image_timestamps in obj_timestamps_and_image_timestamps_for_training
    ]
    image_ts_to_trained_nerf_paths = {(5,): "/path/for/previous/nerf"}

    K = 10

    subsampled_combinations = subsample_timestamps_for_training_nerf(
        obj_timestamps_and_image_timestamps_for_training,
        image_ts_to_trained_nerf_paths,
        K,
    )
    assert subsampled_combinations == [(5, (1, 2, 3)), (15, (13, 14, 15))]
