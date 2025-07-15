"""Downstream to scripts/data_analysis/plot_eef_pos_relative_to_final.py script."""

import os

import imageio
import numpy as np

# Set the directory path
dir_path = "/scr/thankyou/autom/demo-aug/relative_eef"


# Function to combine images horizontally
def combine_images(images):
    return np.hstack(images)


# List to store combined images
combined_images = []

# Loop through the timesteps
for timestep in [24, 25, 26, 27, 28, 29, 30, 31, 32]:
    # Read the three views for each timestep
    main_view = imageio.imread(
        os.path.join(dir_path, f"relative_eef_poses_timestep_{timestep}.png")
    )
    side_view = imageio.imread(
        os.path.join(dir_path, f"relative_eef_poses_timestep_{timestep}_side_view.png")
    )
    top_view = imageio.imread(
        os.path.join(dir_path, f"relative_eef_poses_timestep_{timestep}_top_down.png")
    )

    # Combine the three views
    combined = combine_images([main_view, side_view, top_view])

    # Add the combined image to our list
    combined_images.append(combined)

# decrease the size of the images
combined_images = [image[::2, ::2] for image in combined_images]

# Create the video
imageio.mimsave("relative_eef_positions.mp4", combined_images, fps=1)

print("Video created successfully: relative_eef_positions.mp4")
