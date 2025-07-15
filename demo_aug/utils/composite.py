from typing import Optional, Tuple

import torch


def alpha_composite(
    rgb_list: torch.Tensor,
    disp_list: torch.Tensor,
    acc_list: Optional[torch.Tensor] = None,
    gamma: Optional[float] = 1.0,
    white_background: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Composites a list of RGBA images with alpha values, given their corresponding disparity values.
    The algorithm sorts the images based on their disparity values and composites them from front to back.

    Args:
    - rgb_list (torch.Tensor): A tensor of shape (N, H, W, 3) containing RGB images to composite.
    - disp_list (torch.Tensor): A tensor of shape (N, H, W) containing the disparity values of
            each pixel in each image in rgb_list.
    - acc_list (Optional[torch.Tensor]): A tensor of shape (N, H, W) containing the alpha values of
            each pixel in each image in rgb_list.
    - gamma (Optional[float]): A float value to control the intensity of the colors during compositing.
    - white_background (bool): A boolean value to subtract the white background from each image,
            if acc_list is not None.

    Returns:
    - Tuple[torch.Tensor, torch.Tensor]: A tuple containing the final composited RGB image
        (torch.Tensor of shape (H, W, 3)) and its alpha values (torch.Tensor of shape (H, W)).
    """
    # Sort disparity from highest (nearest) to lowest across objects to separate foreground objects
    # from background
    _, indices = torch.sort(disp_list, dim=0, descending=True)

    if acc_list is not None:
        if white_background is True:
            # Subtract out the white background from each image
            rgb_list = rgb_list - (1.0 - acc_list[..., None])
        # add an extra dimension to indices using torch.unsqueeze()
        # indices = torch.unsqueeze(indices, dim=-1)
        rgbs = torch.gather(rgb_list, 0, indices[..., None].expand((*indices.shape, 3)))
        alphas = torch.gather(acc_list, 0, indices)
    else:
        rgbs = torch.gather(rgb_list, 0, indices[..., None].expand((*indices.shape, 4)))
        alphas = rgbs[..., -1]
        rgbs = rgbs[..., :3]

    # Calculate alpha accumulation (a0 = a1 + a2(1-a1))
    colors = rgbs[0]
    alphs = alphas[0]

    for i in range(disp_list.shape[0] - 1):
        colors = (
            colors ** (1.0 / gamma)
            + (rgbs[i + 1] ** (1.0 / gamma)) * (1.0 - alphs[..., None])
        ) ** (gamma)
        alphs = alphs + alphas[i + 1] * (1.0 - alphs)

    if white_background is True:
        colors = colors + (1.0 - alphs[..., None])

    # clamp the colors to [0, 1]: hack/less principled for now
    # https://chatgpt.com/share/ddb88745-0851-4567-bdd3-64245e03c710
    # correct intuition if the representations are physically put together
    # if we stop the blendering once we reach alpha = 1, then the colors will be correct
    # however, in our case we are blendering different things --- I think the background might be
    # straight alpha?
    colors = torch.clamp(colors, 0.0, 1.0)

    return colors, alphs
