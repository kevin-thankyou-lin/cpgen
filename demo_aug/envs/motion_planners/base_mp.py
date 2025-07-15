import pathlib
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np


class MotionPlanner(ABC):
    name: str = "base"

    def __init__(
        self,
        env: Optional[Any] = None,
        save_dir: Optional[pathlib.Path] = None,
        **kwargs,
    ):
        self.env = env
        self.save_dir = save_dir

    @abstractmethod
    def plan(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        motion_planner_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        pass

    def update_env(self, env: Optional[Any] = None, **kwargs):
        self.env = env

    def visualize_plan(self, plan: np.ndarray):
        pass
