from pathlib import Path

import pytest

from demo_aug.utils.snopt_utils import (
    extract_feasiblity_optimality_values_from_snopt_log,
    find_last_header_line_idx,
)


def test_no_header():
    data = ["This is a random line", "Another random line", "Yet another random line"]
    assert find_last_header_line_idx(data) is None


def test_single_header():
    data = [
        "This is a random line",
        "   Itns ... Major ... Minors ... Feasible ... Optimal ... MeritFunction  sfsdfsdf ",
        "   Itns1231 ... Major ... Minors ... Feasible ... Optimal ... MeritFunction  sfsdfsdf ",
    ]
    assert find_last_header_line_idx(data) == 1


def test_multiple_headers():
    data = [
        "This is a random line",
        "   a   Itns ... Major ... Minors ... Feasible ... Optimal ... MeritFunction",
        "Another random line after first header",
        "Itns ... Major ... Minors ... Feasible ... Optimal ... MeritFunction",
        "Another random line after second header",
    ]
    assert find_last_header_line_idx(data) == 3


def test_data_with_empty_lines():
    data = [
        "This is a random line",
        "",
        "Itns ... Major ... Minors ... Feasible ... Optimal ... MeritFunction",
        "Another random line after header",
        "",
    ]
    assert find_last_header_line_idx(data) == 2


def test_empty_data():
    data = []
    assert find_last_header_line_idx(data) is None


@pytest.mark.parametrize(
    "data,expected",
    [
        (
            ["This is a random line", "Another random line", "Yet another random line"],
            None,
        ),
        (
            [
                "This is a random line",
                "Itns ... Major ... Minors ... Feasible ... Optimal ... MeritFunction",
                "Another random line after header",
            ],
            1,
        ),
        (
            [
                "This is a random line",
                "Itns ... Major ... Minors ... Feasible ... Optimal ... MeritFunction",
                "Another random line after first header",
                "Itns ... Major ... Minors ... Feasible ... Optimal ... MeritFunction",
                "Another random line after second header",
            ],
            3,
        ),
        (
            [
                "This is a random line",
                "",
                "Itns ... Major ... Minors ... Feasible ... Optimal ... MeritFunction",
                "Another random line after header",
                "",
            ],
            2,
        ),
        ([], None),
    ],
)
def test_find_last_header_line_idx(data, expected):
    assert find_last_header_line_idx(data) == expected


def create_log_file(content: str, tmp_path: Path) -> Path:
    """Helper function to create a dummy SNOPT log file."""
    file_path = tmp_path / "dummy_log.log"
    with file_path.open("w") as f:
        f.write(content)
    return file_path


def test_no_headers(tmp_path):
    content = """
    This is a dummy log file.
    There are no headers in this file.
    """
    file_path = create_log_file(content, tmp_path)
    feasible, optimal, is_acceptable = (
        extract_feasiblity_optimality_values_from_snopt_log(file_path)
    )
    assert feasible is None
    assert optimal is None
    assert not is_acceptable


def test_headers_no_data(tmp_path):
    content = """
    This is a dummy log file.
    Itns Major Minors Feasible Optimal MeritFunction L+U BSwap  nS condZHZ Penalty
    """
    file_path = create_log_file(content, tmp_path)
    feasible, optimal, is_acceptable = (
        extract_feasiblity_optimality_values_from_snopt_log(file_path)
    )
    assert feasible is None
    assert optimal is None
    assert not is_acceptable


def test_headers_with_acceptable_solution(tmp_path):
    content = """
    Itns Major Minors Feasible Optimal MeritFunction L+U BSwap     nS condZHZ Penalty
    1    2     3     4.5     6.7    8.9           1   2        3   4       5
    2    3     4     (5.5)   7.8    9.10        1   2        3   4       5
    """
    file_path = create_log_file(content, tmp_path)
    feasible, optimal, is_acceptable = (
        extract_feasiblity_optimality_values_from_snopt_log(file_path)
    )
    assert feasible == "(5.5)"
    assert optimal == 7.8
    assert is_acceptable


def test_headers_with_not_acceptable_solution(tmp_path):
    content = """
    Itns Major Minors Feasible Optimal MeritFunction L+U BSwap     nS condZHZ Penalty
    1    2     3     4.5     6.7    8.9              1   2        3   4       5
    2    3     4     5.5     7.8    9.10            1   2        3   4       5
    """
    file_path = create_log_file(content, tmp_path)
    feasible, optimal, is_acceptable = (
        extract_feasiblity_optimality_values_from_snopt_log(file_path)
    )
    assert feasible == "5.5"
    assert optimal == 7.8
    assert not is_acceptable
