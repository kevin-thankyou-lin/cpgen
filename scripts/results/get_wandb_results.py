import json
from typing import Callable

import numpy as np

import wandb


def get_relevant_runs(project_path, save_path: str = "relevant_runs.json"):
    """
    Retrieve all runs from a project and filter based on specific criteria hardcoded.
    Save the filtered runs to a JSON file.
    """
    # Set the project path to your project: "kevin-lin/adaflow"
    api = wandb.Api()
    runs = api.runs(project_path)

    filtered_runs = []

    for run in runs:
        # Check if run name starts with "E7"
        if not (
            run.name.startswith("E5")
            or run.name.startswith("E7")
            or run.name.startswith("E8")
        ):
            continue

        dataset_path = run.config["task"]["dataset_path"]
        hydra_run_dir = run.config["multi_run"]["run_dir"]
        duration = run.summary["_runtime"]
        # Filter out runs that lasted less than 3 hours (3 * 3600 seconds)
        if duration < 3 * 3600:
            continue

        # Retrieve the max test/mean_score metric from the run history (adjust if your metric key is different)
        df = run.history(keys=["test/mean_score"], pandas=False)
        max_test_mean_score = np.array(
            [df[i]["test/mean_score"] for i in range(len(df))]
        ).max()
        max_test_mean_score = float(max_test_mean_score)

        # Append a dictionary with the information for the filtered run
        filtered_runs.append(
            {
                "name": run.name,
                "max_test_mean_score": max_test_mean_score,
                "run_id": run.id,
                "dataset_path": dataset_path,
                "hydra_run_dir": hydra_run_dir,
                "duration_hours": duration / 3600,
                "wandb_url": f"https://wandb.ai/{project_path}/runs/{run.id}",
            }
        )

    # Print the filtered runs
    print("Filtered Runs:")
    for run_info in filtered_runs:
        print(run_info)

    # save to json
    with open(save_path, "w") as f:
        json.dump(filtered_runs, f, indent=4)


def filter_run(
    filter_criteria: Callable,
    source_path: str = "relevant_runs.json",
    save_path: str = "filtered_runs.json",
):
    # Load the existing filtered runs
    with open(source_path, "r") as f:
        runs = json.load(f)

    # Filter only runs that have "depth-seg" in the dataset_path
    # Apply the filter criteria if provided, otherwise include all runs
    filtered_runs = [run for run in runs if filter_criteria(run)]

    # Order by "name" field
    filtered_runs.sort(key=lambda x: x["name"])

    # Save to a new JSON file
    with open(save_path, "w") as f:
        json.dump(filtered_runs, f, indent=4)
        print(f"Saved {len(filtered_runs)} depth-seg runs to {save_path}")


if __name__ == "__main__":
    relevant_runs_path = "scripts/results/data/relevant_runs.json"
    # get_relevant_runs("kevin-lin/adaflow")

    # filter_criteria = lambda run: "depth-seg" in run["dataset_path"] and "E7" in run["name"] and not "E7." in run["name"]
    # filter_run(filter_criteria, relevant_runs_path, "E7X_wide_depth_seg_runs.json")
    # filter_criteria = lambda run: "depth-seg" not in run["dataset_path"] and "E7" in run["name"] and not "E7." in run["name"]
    # filter_run(filter_criteria, relevant_runs_path, "E7X_wide_rgb_runs.json")

    # # E5X_wide_depth_seg_runs.json
    # filter_criteria = lambda run: "depth-seg" in run["dataset_path"] and "E5" in run["name"] and not "E5." in run["name"]
    # filter_run(filter_criteria, relevant_runs_path, "E5X_wide_depth_seg_runs.json")
    # filter_criteria = lambda run: "depth-seg" not in run["dataset_path"] and "E5" in run["name"] and not "E5." in run["name"]
    # filter_run(filter_criteria, relevant_runs_path, "E5X_wide_rgb_runs.json")

    # # E8X_real_runs.json
    # filter_criteria = lambda run: "depth-seg" in run["dataset_path"] and "E8" in run["name"] and not "E8." in run["name"]
    # filter_run(filter_criteria, relevant_runs_path, "E8X_real_depth_seg_runs.json")
    def filter_criteria(run):
        return (
            "depth_seg" not in run["dataset_path"]
            and "E8" in run["name"]
            and "E8." not in run["name"]
        )

    filter_run(filter_criteria, relevant_runs_path, "E8X_real_rgb_runs.json")
