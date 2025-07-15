"""
Script that takes in images, queries user for segmentation info (such as
natural language, points, etc.), and then runs the segmentation algorithm (e.g. SAM)
and save the segmentation masks and updated segmented images (with white background)
to a chosen directory.

Thus, output images should be similar to the nerf synthetic images."""

import argparse
import json
import logging
import pathlib
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from lang_sam import LangSAM
from PIL import Image
from tqdm import tqdm

from demo_aug.utils.segmentation_utils import get_mask_for_image
from demo_aug.utils.viz_utils import render_image_collage_from_image_list


def segment_images_from_json(
    phrase: str,
    src_json_path: pathlib.Path,
    dest_json_name: str,
    save_as_blender: bool = True,
):
    """
    Segments images in a json file by creating a mask for the specified phrase and saves the segmented images
    with a new name (under the same directory).

    :param save_as_blender: If true, save images + transforms in NeRF blender format (i.e. nerf-synthetic)
                            see data from https://www.matthewtancik.com/nerf.
    """
    all_image_infos: List[Dict] = []
    all_images: List[Tuple[str, np.ndarray]] = []

    model = LangSAM()
    with open(src_json_path, "r") as f:
        image_json = json.load(f)

    new_json = image_json.copy()
    frames = new_json["frames"]
    # update the file_path name to point to the segmented images
    for i, frame in enumerate(tqdm(frames)):
        src_img_full_path = src_json_path.parent / frames[i]["file_path"]
        if "." not in src_img_full_path.name:
            src_img_full_path = src_img_full_path.with_suffix(".png")

        src_img = Image.open(src_img_full_path)
        src_img = np.array(src_img)

        # save segmented image
        mask = get_mask_for_image(src_img, phrase, model, threshold=0.5)
        # use masks to white out the background of image
        if mask is None:
            logging.info(f"No mask found for image {src_img_full_path}. Skipping ...")
            continue

        segmented_image = np.copy(src_img)
        # Set non-masked pixels to white
        segmented_image[np.logical_not(mask)] = [255, 255, 255]

        # following line assumes file_path doesn't use the .png extension
        orig_file_path = frames[i]["file_path"]
        if ".png" in orig_file_path:
            orig_file_path = orig_file_path[:-4]
        segmented_file_path = f"{str(orig_file_path)}_segmented.png"

        success = cv2.imwrite(
            str(src_json_path.parent / segmented_file_path),
            cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB),
        )
        if not success:
            logging.warning(f"Failed to save image: {segmented_file_path}")

        if save_as_blender:
            frames[i]["file_path"] = segmented_file_path[:-4]
        else:
            frames[i]["file_path"] = segmented_file_path

        all_image_infos.append(frames[i])
        all_images.append((str(Path(segmented_file_path).stem), segmented_image))

    new_json["frames"] = all_image_infos
    # Save all image infos to a json file
    with open(Path(src_json_path.parent) / dest_json_name, "w") as f:
        json.dump(new_json, f)

    render_image_collage_from_image_list(
        all_images, Path(src_json_path).parent / f"{dest_json_name[:-5]}_seg_all.png"
    )


if __name__ == "__main__":
    """
    python scripts/segment_images.py  --save-dir data/robomimic/lift/
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save-as-blender",
        type=bool,
        default=False,
        help="Save the segmented images as blender data format (c.f. nerf-synthetic data)",
    )
    parser.add_argument(
        "--segmentation-phrase",
        type=str,
        help="Phrase to use for segmentation via LangSAM",
    )
    parser.add_argument(
        "--src-json-path",
        type=str,
        help="Path to the json file that contains information about image locations",
    )
    parser.add_argument(
        "--dest-json-name",
        type=str,
        help=(
            "Name of the json file that contains information about segmented image locations; same directory as"
            " src_json_path"
        ),
    )

    args = parser.parse_args()
    segment_images_from_json(
        args.segmentation_phrase, pathlib.Path(args.src_json_path), args.dest_json_name
    )
