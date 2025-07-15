from typing import Dict

import numpy as np
from mink import Configuration


### Monkey patch the Configuration class to add and transform keypoints ###
def transform_keypoints(keypoints: np.ndarray, transform: np.ndarray) -> np.ndarray:
    # Step 1: Convert keypoints to homogeneous coordinates (add a column of 1s)
    num_keypoints = keypoints.shape[0]
    homogeneous_keypoints = np.hstack(
        [keypoints, np.ones((num_keypoints, 1))]
    )  # shape (N, 4)

    # Step 2: Apply the transformation matrix (shape (4, 4)) to each keypoint
    transformed_keypoints = (transform @ homogeneous_keypoints.T).T  # shape (N, 4)
    return transformed_keypoints[:, :3]  # shape (N, 3)


def set_keypoints(
    self, keypoints: Dict[str, np.ndarray], flip_eef: bool = False
) -> None:
    """
    Add keypoints to the Configuration. Keypoints are specified in the link's frame.
    """
    self._keypoints = keypoints
    self._flip_eef = flip_eef


def get_keypoints(self) -> np.ndarray:
    """
    Get the keypoints in the world frame.
    """
    P_W = []
    for link_name, P_link in self._keypoints.items():
        X_W_link = self.get_transform_frame_to_world(link_name, "body").as_matrix()
        P_W_curr = transform_keypoints(P_link, X_W_link)
        P_W.append(P_W_curr)

    if self._flip_eef:
        assert (
            len(self._keypoints.keys()) == 2
        ), "Expected exactly two link_names when flipping end effector."
        P_W = [P_W[1], P_W[0]]  # Swap the order of the keypoints

    return np.concatenate(P_W, axis=0)


Configuration.set_keypoints = set_keypoints
Configuration.get_keypoints = get_keypoints
