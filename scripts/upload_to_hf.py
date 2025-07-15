import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

from huggingface_hub import HfApi, login


def upload_to_huggingface(
    token: str,
    repo_id: str,
    file_path: Path,
    repo_type: str = "dataset",
    create_repo: bool = True,
    exclude_suffixes: Optional[List[str]] = None,
    include_suffixes: Optional[List[str]] = None,
):
    """
    Upload a file or all files in a folder to Hugging Face Hub.

    Args:
        token: HuggingFace access token
        repo_id: Repository ID (format: "username/repo-name")
        file_path: Path to file or folder to upload
        repo_type: Type of repository ("dataset" or "model")
        create_repo: Whether to create the repo if it doesn't exist
        exclude_suffixes: List of suffixes to exclude from upload
        include_suffixes: List of suffixes to include in upload (if specified, only these will be uploaded)
    """
    try:
        # Login to Hugging Face
        login(token=token)
        api = HfApi()

        # Create repo if it doesn't exist and create_repo is True
        if create_repo:
            try:
                api.create_repo(
                    repo_id=repo_id, repo_type=repo_type, private=True, exist_ok=True
                )
            except Exception as e:
                print(f"Note: Repository creation returned: {e}")

        # Function to upload individual files
        def upload_file(file: Path):
            try:
                print(f"Uploading {file} to {repo_id}...")
                api.upload_file(
                    path_or_fileobj=str(file),
                    path_in_repo=str(file),
                    repo_id=repo_id,
                    repo_type=repo_type,
                )
                print(f"Successfully uploaded {file.name} to {repo_id}")
            except Exception as e:
                print(f"Error uploading {file}: {e}")

        # If file_path is a directory, upload all files recursively
        if file_path.is_dir():
            for file in file_path.rglob("*"):
                if file.is_file():
                    # Skip files with specified suffixes
                    if exclude_suffixes and any(
                        file.name.endswith(suffix) for suffix in exclude_suffixes
                    ):
                        print(f"Skipping {file} due to matching exclude suffix.")
                        continue

                    # Only upload files with specified suffixes if include_suffixes is provided
                    if include_suffixes and not any(
                        file.name.endswith(suffix) for suffix in include_suffixes
                    ):
                        print(
                            f"Skipping {file} as it doesn't match any include suffix."
                        )
                        continue

                    upload_file(file)
        else:
            # If file_path is a single file, check if it should be included based on include_suffixes
            should_upload = True
            if include_suffixes and not any(
                file_path.name.endswith(suffix) for suffix in include_suffixes
            ):
                should_upload = False
                print(f"Skipping {file_path} as it doesn't match any include suffix.")

            # Check if it should be excluded
            if (
                should_upload
                and exclude_suffixes
                and any(file_path.name.endswith(suffix) for suffix in exclude_suffixes)
            ):
                should_upload = False
                print(f"Skipping {file_path} due to matching exclude suffix.")

            if should_upload:
                upload_file(file_path)

    except Exception as e:
        print(f"Error during upload process: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload files to Hugging Face Hub")
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help='Repository ID (format: "username/repo-name")',
    )
    parser.add_argument(
        "--paths-json",
        type=str,
        required=True,
        help="JSON file containing list of file paths to upload",
    )
    parser.add_argument(
        "--exclude-suffixes",
        type=str,
        nargs="*",
        default=[".mp4"],
        help="List of file suffixes to exclude from upload",
    )
    parser.add_argument(
        "--include-suffixes",
        type=str,
        nargs="*",
        help="List of file suffixes to include in upload (if specified, only these will be uploaded)",
    )
    parser.add_argument(
        "--create-repo",
        action="store_true",
        help="Create the repository if it does not exist",
    )
    parser.add_argument(
        "--repo-type",
        type=str,
        default="dataset",
        help='Type of repository ("dataset" or "model")',
    )
    args = parser.parse_args()

    # Get token from environment or prompt
    TOKEN = os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if TOKEN is None:
        TOKEN = input("Please enter your Hugging Face Hub token: ")

    # Load paths from JSON file
    with open(args.paths_json, "r") as f:
        file_paths = json.load(f)

    if not isinstance(file_paths, list):
        print("Error: JSON file must contain a list of file paths")
        sys.exit(1)

    # Upload each path in the list
    for file_path in file_paths:
        print(f"Processing: {file_path}")
        upload_to_huggingface(
            token=TOKEN,
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            file_path=Path(file_path),
            exclude_suffixes=args.exclude_suffixes,
            include_suffixes=args.include_suffixes,
            create_repo=args.create_repo,
        )
