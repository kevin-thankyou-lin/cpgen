import logging
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Optional, Tuple

# may need tuning: hyperparameter used to decide if an 'acceptably feasible' solution is optimal enough to be used

SNOPT_SOLVER_MAX_OPTIMALITY_VALUE = 1e-1


def check_is_header(line: str) -> bool:
    header_pattern = re.compile(
        r"\bItns\b.*\bMajor\b.*\bMinors\b.*\bFeasible\b.*\bOptimal\b.*\bMeritFunction\b"
    )
    return header_pattern.search(line) is not None


def find_last_header_line_idx(data: List[str]) -> Optional[int]:
    """Find the index of the last header line in the data."""
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(check_is_header, data))

    header_line_idxs = [idx for idx, is_header in enumerate(results) if is_header]
    return header_line_idxs[-1] if header_line_idxs else None


def extract_feasiblity_optimality_values_from_snopt_log(
    file_path: Path,
) -> Tuple[Optional[str], Optional[float], bool]:
    """Extract the feasibility and optimality values for the last iteration of optimization from a SNOPT log file.

    Args:
      max_optimality_value: The maximum optimality value for which the solution is considered acceptable.

    Returns:
        feasible_value: The feasibility value for the last iteration of optimization.
        optimal_value: The optimality value for the last iteration of optimization.
        is_acceptably_feasible: Whether the solution is acceptably feasible.
    """
    MIN_NUM_COLUMNS_IN_HEADER = 8

    # Read file content
    with file_path.open("r") as f:
        line_lst = f.readlines()

    last_header_idx = find_last_header_line_idx(line_lst)

    if last_header_idx is None:
        print("No header found")
        return None, None, False

    last_header = line_lst[last_header_idx]
    # From the index of the last header, get the last data row
    last_data_row = line_lst[last_header_idx + 1]
    remaining_lines = line_lst[last_header_idx + 1 :]

    # find indices by looking for the entry in the last header that contains "Feasible" and "Optimal"
    FEASIBLE_IDX = [
        idx for idx, entry in enumerate(last_header.split()) if "Feasible" in entry
    ][0]
    OPTIMAL_IDX = [
        idx for idx, entry in enumerate(last_header.split()) if "Optimal" in entry
    ][0]

    for row in remaining_lines:
        if len(row.split()) > MIN_NUM_COLUMNS_IN_HEADER:
            last_data_row = row
        else:
            break

    if len(last_data_row.split()) > MIN_NUM_COLUMNS_IN_HEADER:
        values = last_data_row.split()

        is_acceptably_feasible = (
            "(" in values[FEASIBLE_IDX]
        )  # if entry is contained in parentheses, solution is less than "major feasibility tolerance"
        feasible_value = values[FEASIBLE_IDX] if FEASIBLE_IDX < len(values) else None
        if "(" in values[OPTIMAL_IDX] or ")" in values[OPTIMAL_IDX]:
            optimal_value = values[OPTIMAL_IDX] if OPTIMAL_IDX < len(values) else None
        else:
            optimal_value = (
                float(values[OPTIMAL_IDX]) if OPTIMAL_IDX < len(values) else None
            )

        # TODO deal w case where it's acceptably feasible but optimal value also has parentheses or is not an entry
        return feasible_value, optimal_value, is_acceptably_feasible

    logging.info("No data found after the last header")
    return None, None, False
