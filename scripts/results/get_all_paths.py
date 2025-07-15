import argparse
import json
from pathlib import Path
from typing import List


def process_json_file(json_path: Path, entry_key: str) -> List[str]:
    """
    Process a JSON file containing dataset run entries, extract the dataset_path,
    and upload the corresponding file or directory to Hugging Face.
    """
    result_paths = []
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON from {json_path}: {e}")
        return result_paths

    for entry in data:
        entry_val = entry.get(entry_key)
        if not entry_val:
            print(f"Missing '{entry_key}' in entry: {entry}")
            continue
        result_paths.append(entry_val)

    return result_paths


def main():
    parser = argparse.ArgumentParser(
        description="Upload datasets to Hugging Face from JSON files containing dataset paths."
    )
    parser.add_argument(
        "json_files",
        nargs="+",
        type=Path,
        help="List of JSON file paths containing dataset entries.",
    )
    parser.add_argument(
        "--entry-key",
        type=str,
        default="dataset_path",
        help="Key to extract from each entry in the JSON file.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default="all_paths.json",
        help="Path to save the extracted paths.",
    )
    args = parser.parse_args()

    all_paths = []
    for json_file in args.json_files:
        if not json_file.exists():
            print(f"JSON file not found: {json_file}")
            continue

        print(f"Processing JSON file: {json_file}")
        paths = process_json_file(json_file, entry_key=args.entry_key)
        all_paths.extend(paths)

    # save to json
    with open(args.output_path, "w") as f:
        json.dump(all_paths, f, indent=4)


if __name__ == "__main__":
    main()
