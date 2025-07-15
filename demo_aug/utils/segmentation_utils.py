import logging
import random
from typing import Any, Union

import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image

try:
    from lang_sam import LangSAM
except ImportWarning:
    logging.warning(
        "lang_sam required for segmenting images: see https://github.com/luca-medeiros/lang-segment-anything"
    )


def get_masks_for_images(
    img: Union[np.ndarray, torch.Tensor],
    phrase: str,
    model: LangSAM,
    threshold: float = 0.5,
) -> Any:
    """Generates SAM masks for images: mainly for speed by parallelizing inference."""
    raise NotImplementedError("get_masks_for_images not implemented")


def show_mask(mask: Union[torch.Tensor, np.ndarray]) -> None:
    # convert to numpy  array
    if isinstance(mask, torch.Tensor):
        mask = mask.numpy()

    # Convert the boolean matrix to a numeric matrix
    numeric_matrix = np.array(mask, dtype=int)

    for mask in numeric_matrix:
        # Create a heatmap using matplotlib
        plt.imshow(mask, cmap="binary", interpolation="nearest")
        plt.colorbar()
        plt.show()


def get_mask_for_image(
    img: np.ndarray,
    phrase: str,
    model: LangSAM,
    threshold: float = 0.75,
    is_show_mask: bool = False,
) -> np.ndarray:
    """
    Returns a binary mask for a given input image using the LangSAM model and a specified phrase.
    Applies thresholding to the predicted logits to select the best mask.
    """
    img_pil = Image.fromarray(img)
    masks, boxes, phrases, logits = model.predict(img_pil, phrase)

    selected_mask = None
    max_logit = -float("inf")

    # Iterate over each mask and apply threshold individually
    for i in range(masks.shape[0]):
        mask = masks[i]
        logit = logits[i]

        if logit > threshold and logit > max_logit:
            max_logit = logit
            selected_mask = mask

    if selected_mask is None:
        return None

    if is_show_mask:
        show_mask(selected_mask, f"mask_{random.randint(0, 100000)}.png")

    return selected_mask.cpu().numpy()
