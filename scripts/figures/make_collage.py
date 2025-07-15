from pathlib import Path
from typing import List

from PIL import Image


def create_image_collage(
    top_images: List[Path],
    child_images: List[List[Path]],
    output_image_path: Path,
    top_image_size: tuple,
    child_image_size: tuple,
    child_image_rows_height: int,
    nominal_spacing: int,
):
    num_top_images = len(top_images)
    top_width, top_height = top_image_size

    # Calculate total width and height for the collage, accounting for nominal_spacing between top images
    collage_width = (top_width + nominal_spacing) * num_top_images - nominal_spacing
    collage_height = top_height + child_image_rows_height + nominal_spacing

    # Create a blank canvas for the collage with white background
    collage_image = Image.new("RGB", (collage_width, collage_height), (255, 255, 255))

    spacing = nominal_spacing
    # Load and place the top images in the collage
    for i, top_image_path in enumerate(top_images):
        top_image = Image.open(top_image_path).resize(top_image_size)
        x_offset_top = i * (top_width + spacing)
        collage_image.paste(top_image, (x_offset_top, 0))

        # Load and place the child images under each top image
        num_child_images = len(child_images[i])
        # num_rows = 2 if num_child_images > 3 else 1
        num_cols = 3  # Max 3 images per row
        # row_height = (child_image_rows_height - (num_rows - 1) * spacing) // num_rows
        child_img_width = (top_width - (num_cols - 1) * spacing) // num_cols
        child_img_height = child_image_size[0]

        if num_child_images == 3:
            child_img_width = 40
            child_img_height = 40
            # Center the 3 images in a single row with spacing
            padding = (
                top_width
                - (child_img_width * num_child_images)
                - (num_child_images - 1) * spacing
            ) // 2
            for j, child_image_path in enumerate(child_images[i]):
                child_image = Image.open(
                    child_image_path
                )  # .resize((child_img_width, row_height))
                child_image.thumbnail((child_img_width, child_img_height))

                x_offset = x_offset_top + padding + j * (child_img_width + spacing)
                y_offset = top_height + spacing
                collage_image.paste(child_image, (x_offset, y_offset))
        else:
            # child_img_width = 40
            # child_img_height = 40
            # # For 6 images, arrange them in two rows with spacing
            # for j, child_image_path in enumerate(child_images[i]):
            #     child_image = Image.open(child_image_path)# .resize((child_img_width, row_height))
            #     child_image.thumbnail((child_img_width, child_img_height))
            #     row_idx = j // num_cols
            #     col_idx = j % num_cols
            #     x_offset = x_offset_top + col_idx * (child_img_width + spacing)
            #     y_offset = top_height + row_idx * (row_height + spacing) + spacing
            #     collage_image.paste(child_image, (x_offset, y_offset))
            child_img_width = 40
            child_img_height = 40
            # num_cols = 3
            # num_rows = 2
            # Calculate total width of child images and horizontal padding
            total_child_width = num_cols * child_img_width + (num_cols - 1) * spacing
            padding = (top_width - total_child_width) // 2

            for j, child_image_path in enumerate(child_images[i]):
                child_image = Image.open(child_image_path)
                child_image.thumbnail((child_img_width, child_img_height))
                row_idx = j // num_cols
                col_idx = j % num_cols
                x_offset = (
                    x_offset_top + padding + col_idx * (child_img_width + spacing)
                )
                y_offset = top_height + row_idx * (child_img_height + spacing) + spacing
                collage_image.paste(child_image, (x_offset, y_offset))
    # Save the final collage image
    collage_image.save(output_image_path)


# Example usage:
# top_images_paths = [
#     Path("resets/lift-narrow-viz-plane-resets_frame_0.png"),
#     Path("resets/can-narrow-viz-plane-resets_frame_0.png"),
#     Path("resets/square-narrow-viz-plane-resets_frame_0.png"),
#     Path("resets/lift-wide-viz-plane-resets_frame_0.png"),
#     Path("resets/can-wide-viz-plane-resets_frame_0.png"),
#     Path("resets/square-wide-viz-plane-resets_frame_0.png"),
# ]
top_images_paths = [
    Path("resets/lift-narrow-viz-plane-resets_frame_0.png"),
    Path("resets/lift-wide-viz-plane-resets_frame_0.png"),
    Path("resets/can-narrow-viz-plane-resets_frame_0.png"),
    Path("resets/can-wide-viz-plane-resets_frame_0.png"),
    Path("resets/square-narrow-viz-plane-resets_frame_0.png"),
    Path("resets/square-wide-viz-plane-resets_frame_0.png"),
]

child_images_paths = [
    [
        Path(
            "resets/lift-narrow-canon-pose-resets_frame_0_cropped_220_235_290_310.png"
        ),
        Path(
            "resets/lift-narrow-canon-pose-resets_frame_1_cropped_220_235_290_310.png"
        ),
        Path(
            "resets/lift-narrow-canon-pose-resets_frame_2_cropped_220_235_290_310.png"
        ),
    ],
    [
        Path("resets/lift-wide-canon-pose-resets_frame_0_cropped_220_235_290_310.png"),
        Path("resets/lift-wide-canon-pose-resets_frame_1_cropped_220_235_290_310.png"),
        Path("resets/lift-wide-canon-pose-resets_frame_2_cropped_220_235_290_310.png"),
    ],
    [
        Path("resets/can-narrow-canon-pose-resets_frame_0_cropped_105_217_160_276.png"),
        Path("resets/can-narrow-canon-pose-resets_frame_1_cropped_105_217_160_276.png"),
        Path("resets/can-narrow-canon-pose-resets_frame_2_cropped_105_217_160_276.png"),
    ],
    [
        Path("resets/can-wide-canon-pose-resets_frame_0_cropped_105_217_160_276.png"),
        Path("resets/can-wide-canon-pose-resets_frame_1_cropped_105_217_160_276.png"),
        Path("resets/can-wide-canon-pose-resets_frame_2_cropped_105_217_160_276.png"),
    ],
    [
        Path(
            "resets/square-narrow-canon-pose-resets_frame_0_cropped_330_114_430_200.png"
        ),
        Path(
            "resets/square-narrow-canon-pose-resets_frame_1_cropped_330_114_430_200.png"
        ),
        Path(
            "resets/square-narrow-canon-pose-resets_frame_2_cropped_330_114_430_200.png"
        ),
        Path(
            "resets/square-narrow-canon-pose-resets_frame_0_cropped_337_338_427_480.png"
        ),
        Path(
            "resets/square-narrow-canon-pose-resets_frame_1_cropped_337_338_427_480.png"
        ),
        Path(
            "resets/square-narrow-canon-pose-resets_frame_2_cropped_337_338_427_480.png"
        ),
    ],
    [
        Path(
            "resets/square-wide-canon-pose-resets_frame_0_cropped_330_114_430_200.png"
        ),
        Path(
            "resets/square-wide-canon-pose-resets_frame_1_cropped_330_114_430_200.png"
        ),
        Path(
            "resets/square-wide-canon-pose-resets_frame_2_cropped_330_114_430_200.png"
        ),
        Path(
            "resets/square-wide-canon-pose-resets_frame_0_cropped_337_338_427_480.png"
        ),
        Path(
            "resets/square-wide-canon-pose-resets_frame_1_cropped_337_338_427_480.png"
        ),
        Path(
            "resets/square-wide-canon-pose-resets_frame_2_cropped_337_338_427_480.png"
        ),
    ],
    # Add more child images for each top image
]

tasks = ["lift", "can", "square"]
distrs = ["narrow", "wide"]

all_tasks = []
for task in tasks:
    for distr in distrs:
        all_tasks.append((task, distr))

for i, (task, distr) in enumerate(all_tasks):
    output_image_path = Path(f"{task}-{distr}.png")
    # create_image_collage(top_images_paths, child_images_paths, output_image_path, (200, 200), (100, 100))
    print(f"Creating collage {i + 1}/{len(all_tasks)}: {output_image_path}")
    create_image_collage(
        [top_images_paths[i]],
        [child_images_paths[i]],
        output_image_path,
        top_image_size=(200, 200),
        child_image_size=(30, 30),
        child_image_rows_height=80,  # Total height for child images below each top image
        nominal_spacing=2,  # Add spacing between images
    )
