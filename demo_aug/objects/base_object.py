from typing import Dict

import torch


class BaseObject:
    """
    Base object class that implements are render method and a collision checking method (for RRT*).
    """

    def __init__(
        self,
    ):
        pass

    def render(self, c2w: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Renders an object in the given world coordinate system.

        Args:
            c2w (torch.Tensor): A tensor of shape (4, 4) representing the camera to
                                world transformation matrix.
            Q: what is the world coordinate frame?
            A: If we use a physics simulator, there'd need to be some conversion.
            A simulator's z axis would be parallel to gravity but a NeRF frame mightn't.

        Returns:
            outputs (Dict[str, torch.Tensor]): A dictionary of outputs.
        """
        raise NotImplementedError
