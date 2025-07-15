import logging
import math
import pathlib
from typing import List, Optional, Tuple

import h5py
import imageio
import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw

from demo_aug.utils.run_script_utils import retry_on_exception


@retry_on_exception(max_retries=15, retry_delay=1, exceptions=(BlockingIOError,))
def create_gifs_from_h5py_file(
    file_path: pathlib.Path, gif_dir: pathlib.Path, image_keys: List
) -> None:
    with h5py.File(file_path, "r") as f:
        demo_groups = [
            g for g in f["data"].values() if g.name.startswith("/data/demo_")
        ]
        for i, demo_group in enumerate(demo_groups):
            demo_name_prefix = demo_group.name.split("/")[-1]
            for image_key in image_keys:
                images = []
                if image_key in demo_group["obs"]:
                    obs_dataset = demo_group["obs"][image_key]
                    obs_images = np.array(obs_dataset)

                    for j in range(obs_images.shape[0]):
                        img = Image.fromarray(obs_images[j])
                        images.append(img)

                    gif_path = f"{gif_dir}/{demo_name_prefix}_{image_key}.gif"
                    print(f"Saving gif to {gif_path}")
                    imageio.mimsave(gif_path, images, duration=0.05, loop=0)

                    mp4_path = f"{gif_dir}/{demo_name_prefix}_{image_key}.mp4"
                    print(f"Saving video to {mp4_path}")
                    imageio.mimsave(mp4_path, images, fps=10)


def render_image_collage_from_image_list(
    image_list: List[Tuple[str, np.ndarray]], save_path: pathlib.Path
) -> None:
    """
    Renders and saves a collage of images from the provided list of name, image tuples.
    For visualization/data debugging purposes.

    image_list (list): List of tuples with the first element as image name and the second as image array.

    Returns:
    None
    """
    # Calculate the number of rows and columns for the collage grid
    num_images = len(image_list)
    num_cols = int(np.sqrt(num_images))  # Adjust the number of columns as desired
    num_rows = (num_images + num_cols - 1) // num_cols

    # Calculate the max width and height among the images
    max_width = max(image_array.shape[1] for image_name, image_array in image_list)
    max_height = max(image_array.shape[0] for image_name, image_array in image_list)

    # Create an empty white canvas for the collage
    collage = Image.new(
        "RGB", (num_cols * max_width, num_rows * max_height), (255, 255, 255)
    )

    # Create a draw object to add text
    draw = ImageDraw.Draw(collage)

    # Loop over the filtered image list and paste each image
    for i, (image_name, image_array) in enumerate(image_list):
        row_index = i // num_cols
        col_index = i % num_cols
        image = Image.fromarray(image_array)
        collage.paste(image, (col_index * max_width, row_index * max_height))

        # some bug with draw.textsize
        # text_width, text_height = draw.textsize(image_name)
        text_width = max_width

        # Calculate the centered text position
        text_position = (
            (col_index * max_width + (max_width - text_width) // 2),
            row_index * max_height,
        )

        # Overlay the image name on the image
        text_position = (col_index * max_width, row_index * max_height)
        draw.text(text_position, image_name, fill=(255, 0, 0))

    # Save the collage
    collage.save(str(save_path))
    logging.info(f"Saved collage to {save_path}")


def fibonacci_sphere(samples: int = 40):
    points = []
    phi = math.pi * (math.sqrt(5.0) - 1.0)  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        points.append((x, y, z))

    return points


def generate_camera_positions(
    K: int, r: float, center: torch.Tensor, z_min: float = 0, z_max: float = 1
) -> List[torch.Tensor]:
    """
    Generate K random camera positions on a hemisphere with radius r.

    z_min and z_max are used to filter out camera positions with z values outside of the range.
        Filtering is applied to the fibonacci sphere points (where sphere's radius is 1)
        before applying the radius r.
    """
    points: List[torch.Tensor] = []
    fib_points = fibonacci_sphere(samples=K * 30)
    # filter out camera positions with z values outside of the range
    fib_points = [p for p in fib_points if p[2] > z_min and p[2] <= z_max]
    # shuffle fibonacci points and take first K
    np.random.shuffle(fib_points)
    fib_points = fib_points[:K]

    # add center point and apply radius to each fibonacci point
    for p in fib_points:
        points.append(torch.tensor(p) * r + center)

    assert (
        len(points) == K
    ), "Generated {} points instead of {}; easiest fix: update fibonacci_sphere() samples above".format(
        len(points), K
    )
    return points


def get_lookat(
    eye: np.ndarray, target: np.ndarray, up: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the lookat matrix
    https://ksimek.github.io/2012/08/22/extrinsic/.

    Args:
        eye (numpy.ndarray): The camera eye position
        target (numpy.ndarray): The target position to look towards
        up (numpy.ndarray): The up vector indicating which direction is up

    Returns:
        A tuple containing the rotation and translation matrices
        rotation (numpy.ndarray): The rotation matrix
        translation (numpy.ndarray): The translation matrix
    """
    L = target - eye
    L /= np.linalg.norm(L)
    s = np.cross(L, up)
    s /= np.linalg.norm(s)
    u_prime = np.cross(s, L)

    rotation = np.array(
        [
            [s[0], s[1], s[2]],
            [u_prime[0], u_prime[1], u_prime[2]],
            [-L[0], -L[1], -L[2]],
        ]
    )

    translation = -np.dot(rotation, eye)

    return rotation, translation


def plot_image_diff_map(
    image1: np.ndarray,
    image2: np.ndarray,
    title: str,
    save_dir: Optional[pathlib.Path] = None,
    print_max_diff: bool = False,
) -> None:
    diff = np.abs(image1.astype(np.int16) - image2.astype(np.int16))

    if print_max_diff:
        print(f"Max difference: {np.max(diff)}")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title, fontsize=16)

    im1 = ax1.imshow(image1)
    ax1.set_title("Image 1")
    plt.colorbar(im1, ax=ax1, label="Pixel Value")

    im2 = ax2.imshow(image2)
    ax2.set_title("Image 2")
    plt.colorbar(im2, ax=ax2, label="Pixel Value")

    im3 = ax3.imshow(diff, cmap="hot")
    ax3.set_title("Difference Map")
    plt.colorbar(im3, ax=ax3, label="Absolute Difference")

    plt.tight_layout()
    if save_dir is None:
        save_dir = pathlib.Path.cwd()

    save_path = save_dir / f"{title.replace(' ', '_').lower()}.png"
    plt.savefig(save_path)
    print(f"Saved image diff map to {save_path}")
