import logging
import pathlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import cv2
import tyro
from moviepy.editor import VideoFileClip, clips_array


def png_to_video(
    input_dir: pathlib.Path, output_path: pathlib.Path, prefix: str = "agentview", fps=6
):
    image_files = sorted(
        [
            f
            for f in Path(input_dir).iterdir()
            if f.suffix == ".png" and prefix in f.name
        ],
        key=lambda x: int(x.stem.split("image")[-1]),
    )
    logging.info("image_files")

    # Check if any PNG images were found
    if not image_files:
        logging.info("No PNG images found in the directory.")
        return

    # Load the first image to get dimensions
    first_image = cv2.imread(str(image_files[0]))
    height, width, _ = first_image.shape

    # Define the output video codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )  # Use appropriate codec based on the output video file extension
    video = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # Iterate through the image files and write them to the video
    for image_file in image_files:
        image = cv2.imread(str(image_file))
        video.write(image)

    # Release the video writer and close any open windows
    video.release()
    cv2.destroyAllWindows()


def merge_mp4s(mp4_paths: List[Path], output_path: Path):
    # Load video files
    clips = [VideoFileClip(str(path)) for path in mp4_paths]

    # Get the maximum height and width for the final video
    widths, heights = zip(*(clip.size for clip in clips))
    max_width = max(widths)
    max_height = max(heights)

    # Resize all clips to the maximum height and width
    clips = [clip.resize(height=max_height, width=max_width) for clip in clips]

    # Merge videos side by side
    final_clip = clips_array([clips])

    # Save the final clip
    final_clip.write_videofile(str(output_path), codec="libx264")


@dataclass
class Config:
    image_prefixes: List[str] = field(
        default_factory=lambda: [
            "agentview",
            "robot0_eye_in_hand",
        ]
    )
    input_dir: pathlib.Path = pathlib.Path(
        "test_transf_future_actions_w_loading_nerfs_welded_nose3/0/"
    )
    output_dir: pathlib.Path = pathlib.Path(
        "test_transf_future_actions_w_loading_nerfs_welded_nose3/0/"
    )
    output_name: str = "welded"


def main(cfg: Config):
    for prefix in cfg.image_prefixes:
        png_to_video(
            cfg.input_dir, cfg.input_dir / f"{cfg.output_name}-{prefix}.mp4", prefix
        )

    merge_mp4s(
        [cfg.input_dir / f"{cfg.output_name}-{prefix}.mp4" for prefix in cfg.prefixes],
        cfg.input_dir / f"{cfg.output_name}_all.mp4",
    )


if __name__ == "__main__":
    tyro.cli(main)
