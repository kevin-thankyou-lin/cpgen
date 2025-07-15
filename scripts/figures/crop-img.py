from pathlib import Path
from typing import Tuple

from PIL import Image


def crop_and_save_image(
    image_path: Path, top_left: Tuple[int, int], bottom_right: Tuple[int, int]
) -> Path:
    # Open the image
    img = Image.open(image_path)

    # Crop the image using the provided coordinates (left, upper, right, lower)
    cropped_img = img.crop((*top_left, *bottom_right))

    # Format coordinates nicely for the new file name
    coordinates_str = (
        f"_{top_left[0]}_{top_left[1]}_{bottom_right[0]}_{bottom_right[1]}"
    )

    # Create a new file name based on the original, including the coordinates
    new_image_name = f"{image_path.stem}_cropped{coordinates_str}{image_path.suffix}"
    new_image_path = image_path.parent / new_image_name

    # Save the cropped image
    cropped_img.save(new_image_path)

    print(new_image_path)
    return new_image_path


tasks = ["square", "lift", "can"]
distrs = ["narrow", "wide"]
for task in tasks:
    for distr in distrs:
        objs = ["cube", "peg"]
        if task == "square":
            objs = ["square", "peg"]
        if task == "can":
            objs = ["can"]
        if task == "lift":
            objs = ["cube"]
        for obj in objs:
            if task == "square":
                if obj == "square":
                    top_left = (330, 114)
                    bottom_right = (430, 200)
                    image_paths = [
                        f"resets/square-{distr}-canon-pose-resets_frame_0.png",
                        f"resets/square-{distr}-canon-pose-resets_frame_1.png",
                        f"resets/square-{distr}-canon-pose-resets_frame_2.png",
                    ]
                elif obj == "peg":
                    top_left = (337, 338)
                    bottom_right = (427, 480)
                    image_paths = [
                        f"resets/square-{distr}-canon-pose-resets_frame_0.png",
                        f"resets/square-{distr}-canon-pose-resets_frame_1.png",
                        f"resets/square-{distr}-canon-pose-resets_frame_2.png",
                    ]
            elif task == "lift":
                top_left = (220, 235)
                bottom_right = (290, 310)
                image_paths = [
                    f"resets/lift-{distr}-canon-pose-resets_frame_0.png",
                    f"resets/lift-{distr}-canon-pose-resets_frame_1.png",
                    f"resets/lift-{distr}-canon-pose-resets_frame_2.png",
                ]
            elif task == "can":
                top_left = (105, 217)
                bottom_right = (160, 276)
                image_paths = [
                    f"resets/{task}-{distr}-canon-pose-resets_frame_0.png",
                    f"resets/{task}-{distr}-canon-pose-resets_frame_1.png",
                    f"resets/{task}-{distr}-canon-pose-resets_frame_2.png",
                ]

            for image_path in image_paths:
                crop_and_save_image(Path(image_path), top_left, bottom_right)
