from typing import Optional

import mujoco
import numpy as np
from mink import Configuration


class IndexedConfiguration(Configuration):
    """
    A configuration class extending the base Configuration to allow selective updating
    of joint indices for kinematic calculations and frame updates.
    This class adds more control to update specific joints, providing flexibility for use cases
    where only a subset of joints needs to be modified.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        q: Optional[np.ndarray] = None,
        robot_idxs: Optional[np.ndarray] = None,
    ):
        """Constructor.
        Args:
            model: Mujoco model.
            q: Configuration to initialize from. If None, the configuration is
                initialized to the default configuration `qpos0`.
            robot_idxs: indices of `data.qpos` to be updated.
                If not specified, default to updating all qpos.
        """
        self.model = model
        self.data = mujoco.MjData(model)
        if robot_idxs is None:
            robot_idxs = np.arange(model.nq)
        self.robot_idxs = robot_idxs
        self.update(q=q, update_qpos_idxs=robot_idxs)

    def update(
        self,
        q: Optional[np.ndarray] = None,
        model: Optional[mujoco.MjModel] = None,
        update_qpos_idxs: Optional[np.ndarray] = None,
    ) -> None:
        """Run forward kinematics.
        Args:
            q: Optional configuration vector to override internal `data.qpos` with.
            update_qpos_idxs: indices of `data.qpos` to be updated.
                If not, default to updating self.robot_idxs.
        """
        if model is not None:
            self.model = model
            self.data = mujoco.MjData(model)
        if update_qpos_idxs is None:
            update_qpos_idxs = self.robot_idxs
        if q is not None:
            self.data.qpos[update_qpos_idxs] = q

        # The minimal function call required to get updated frame transforms is
        # mj_kinematics. An extra call to mj_comPos is required for updated Jacobians.
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

    def get_robot_qpos(self) -> np.ndarray:
        """Get the configuration vector."""
        return self.data.qpos[self.robot_idxs]
