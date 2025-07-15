from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

import h5py
import imageio
import numpy as np
import tyro
from PIL import Image, ImageDraw, ImageFont


@dataclass
class Config:
    h5_file_path: str = ""
    demos: List[str] = field(
        default_factory=lambda: [
            "demo_0",
            "demo_1",
            "demo_2",
            "demo_3",
            "demo_4",
            "demo_198",
            "demo_199",
        ]
    )
    images: List[str] = field(
        default_factory=lambda: ["agentview_image", "robot0_eye_in_hand_image"]
    )
    output_dir: Optional[str] = None
    fps: int = 15
    all_demos: bool = False
    show_time_idx: bool = False

    def __post_init__(self):
        if self.all_demos:
            with h5py.File(self.h5_file_path, "r") as f:
                self.demos = [key for key in f["data"].keys()]
        if self.output_dir is None:
            self.output_dir = (Path(self.h5_file_path).parent / "videos").as_posix()


def generate_videos_from_hdf5(cfg: Config) -> None:
    """
    Generate and save MP4 videos from images in an HDF5 file for multiple demos and image datasets.

    Args:
        cfg (Config): Configuration dataclass containing the necessary parameters.
    """
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(cfg.h5_file_path, "r") as f:
        print(f"Dataset has {len(f['data'])} demos.")
        for demo_name in cfg.demos:
            for image_name in cfg.images:
                try:
                    # Navigate to the specified demo and image dataset
                    # import ipdb; ipdb.set_trace()
                    dataset = f[f"data/{demo_name}/obs/{image_name}"]
                except KeyError:
                    print(
                        f"Warning: Dataset 'data/{demo_name}/obs/{image_name}' not found in the HDF5 file. Skipping."
                    )
                    continue

                # Get the shape of the dataset
                num_frames, height, width, channels = dataset.shape

                output_path = (
                    output_dir / f"output_{demo_name}_{image_name}_{height}x{width}.mp4"
                )

                # Create a writer object
                with imageio.get_writer(output_path, fps=cfg.fps) as writer:
                    # Iterate through all frames
                    for i in range(num_frames):
                        # Read the frame
                        frame = dataset[i]

                        # Add time index to the frame if show_time_idx is True
                        if cfg.show_time_idx:
                            frame = add_time_index(frame, i)

                        # Write the frame
                        writer.append_data(frame)

                print(f"Video saved to {output_path}")


def add_time_index(frame: Any, time_idx: int) -> Any:
    """
    Add time index to the image frame.

    Args:
        frame (Any): Image frame, must be a PIL Image.
        time_idx (int): Time index.

    Returns:
        Any: Image frame with time index added.
    """
    # Convert to PIL Image if necessary
    if not isinstance(frame, Image.Image):
        frame = Image.fromarray(frame)

    # Initialize drawing context
    draw = ImageDraw.Draw(frame)

    # Optionally, specify font (you may need to provide a path to a .ttf file)
    try:
        font = ImageFont.load_default()
    except IOError:
        font = None  # Fallback if font can't be loaded
        print("Warning: Font not found. Using default font.")

    # Draw the time index on the frame
    text = f"Time: {time_idx}"
    draw.text((10, 10), text, font=font, fill=(0, 255, 0))

    return np.array(frame)


if __name__ == "__main__":
    tyro.cli(generate_videos_from_hdf5)
