"""
Usage: pytest tests/utils/test_get_renderable_obj_ts.py
"""

from demo_aug.utils.obj_utils import get_renderable_obj_timestamps


def test_simple_case():
    obj_image_mappings = [(1, (10,)), (2, (11, 12)), (3, (13,))]
    nerf_paths = {
        (10,): "path/to/nerf/10",
        (11, 12): "path/to/nerf/11",
    }
    assert get_renderable_obj_timestamps(obj_image_mappings, nerf_paths) == [1, 2]


def test_missing_nerf_path():
    obj_image_mappings = [(1, (10,)), (2, (11, 12)), (3, (13,))]
    nerf_paths_missing = {
        (10,): "path/to/nerf/10",
        (11,): "path/to/nerf/11",
        (13,): "path/to/nerf/13",
    }
    assert get_renderable_obj_timestamps(obj_image_mappings, nerf_paths_missing) == [
        1,
        3,
    ]


def test_no_nerf_paths():
    obj_image_mappings = [(1, (10,)), (2, (11, 12)), (3, (13,))]
    nerf_paths_empty = {}
    assert get_renderable_obj_timestamps(obj_image_mappings, nerf_paths_empty) == []


def test_complex_case():
    obj_image_mappings_complex = [
        (1, (10, 11)),
        (2, (11, 12, 13)),
        (3, (13, 14)),
        (4, (15,)),
    ]
    nerf_paths_complex = {
        (10,): "path/to/nerf/10",
        (12,): "path/to/nerf/12",
        (13,): "path/to/nerf/13",
        (13, 14): "path/to/nerf/13_14",
    }
    assert get_renderable_obj_timestamps(
        obj_image_mappings_complex, nerf_paths_complex
    ) == [3]
