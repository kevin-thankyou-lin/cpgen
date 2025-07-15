#!/usr/bin/env python3
import csv
import json
from pathlib import Path
from typing import Any, Dict

# Optional mapping to prettify task names (adjust as needed)
TASK_NAME_MAP = {
    "StackThreeWide": "Stack Three - Wide",
    "SquareWide": "Square - Wide",
    "ThreadingWide": "Threading - Wide",
    "ThreePieceAssemblyWide": "3Pc. Assembly - Wide",
    "KitchenWide": "Kitchen - Wide",
    "CoffeeWide": "Coffee - Wide",
    "MugCleanupWide": "Mug Cleanup - Wide",
    "HammerCleanupWide": "Hammer Cleanup - Wide",
}


def get_task_name(dataset_path: str) -> str:
    """
    Extract the folder after 'generated/' and convert it using TASK_NAME_MAP.
    E.g., /.../generated/StackThreeWide/2025-... -> "Stack Three - Wide"
    """
    parts = dataset_path.split("/generated/")
    if len(parts) < 2:
        return "Unknown"
    folder = parts[1].split("/")[0]
    return TASK_NAME_MAP.get(folder, folder)


def get_modality(dataset_path: str) -> str:
    """
    Returns 'rgb' or 'depth_seg' based on the dataset_path field.
    """
    if "rgb" in dataset_path.lower():
        return "rgb"
    if "depth-seg" in dataset_path.lower():
        return "depth-seg"
    return "rgb"


def gather_data(json_files: list[Path]) -> Dict[str, Dict[str, Any]]:
    """
    Reads each JSON file, aggregates info by (task, modality).
    Returns a structure like:
      {
        "Stack Three - Wide": {
          "rgb": {
            "checkpoint": ...,
            "wandb": ...,
            "sr": ...,
            "data_gen_link": ...
          },
          "depth_seg": {...}
        },
        ...
      }
    """
    data = {}
    for jf in json_files:
        with jf.open("r") as f:
            items = json.load(f)
            for item in items:
                task = get_task_name(item["dataset_path"])
                mod = get_modality(item["dataset_path"])
                if task not in data:
                    data[task] = {}
                data[task].setdefault(mod, {})
                data[task][mod]["checkpoint"] = ""  # item["hydra_run_dir"]
                data[task][mod]["wandb"] = item["wandb_url"]
                data[task][mod]["sr"] = item["max_test_mean_score"]
                data[task][mod]["data_gen_link"] = ""  # item["dataset_path"]
    return data


def main():
    # Adjust file paths as needed
    json_files = [
        Path("/home/thankyou/autom/demo-aug/E5X_d1_rgb_runs.json"),
        Path("/home/thankyou/autom/demo-aug/E5X_d1_depth_seg_runs.json"),
        Path("/home/thankyou/autom/demo-aug/E7X_wide_rgb_runs.json"),
        Path("/home/thankyou/autom/demo-aug/E7X_wide_depth_seg_runs.json"),
    ]
    # for csv path, use stem of the first json file
    csv_path = json_files[0].stem + ".csv"
    data = gather_data(json_files)

    writer = csv.writer(open(csv_path, "w", newline=""))
    # Columns for a single row per task
    header = [
        "Task",
        "Data Generation Link",
        "RGB Policy Checkpoint",
        "RGB Wandb",
        "RGB SR",
        "Depth+Seg Checkpoint",
        "Depth+Seg Wandb",
        "Depth+Seg SR",
    ]
    writer.writerow(header)
    for task, info in data.items():
        # Gather columns. Fallback to empty string if missing.
        rgb = info.get("rgb", {})
        ds = info.get("depth-seg", {})

        row = [
            task,
            rgb.get("data_gen_link") or ds.get("data_gen_link", ""),
            rgb.get("checkpoint", ""),
            rgb.get("wandb", ""),
            rgb.get("sr", ""),
            ds.get("checkpoint", ""),
            ds.get("wandb", ""),
            ds.get("sr", ""),
        ]
        writer.writerow(row)

    print(f"Wrote {csv_path}. You can copy/paste into Google Sheets.")


if __name__ == "__main__":
    main()
