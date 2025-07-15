import logging
import pathlib

import cv2
import numpy as np

from demo_aug.configs.base import RobotConfig
from demo_aug.objects.robot_object import RobotObject

# TODO(klin): what tests to write? also, convert to actual tests
if __name__ == "__main__":
    save_dir = "test-robot-object"
    robot_obj = RobotObject(cfg=RobotConfig())
    joint_qpos = robot_obj.sample_valid_joint_qpos()
    cam_names = ["robot0_eye_in_hand", "agentview"]
    # pure eye in hand is pretty similar depending on if gripper moves
    for i in range(10):
        for cam_name in cam_names:
            obs = robot_obj.get_observation(
                joint_qpos["robot_joint_qpos"],
                joint_qpos["robot_gripper_qpos"],
                np.eye(4),
                cam_name,
            )
            rgb = obs["rgb"].cpu().numpy()
            depth = real_depth = obs["depth"].cpu().numpy()
            vis_depth = real_depth.copy()
            vis_depth = cv2.normalize(vis_depth, None, 0, 255, cv2.NORM_MINMAX).astype(
                "uint8"
            )

            # TODO(klin): unclear why the images are reversed
            file_path = f"images/frame_{str(i).zfill(5)}_{cam_name}_rgb.png"
            full_file_path = pathlib.Path(save_dir) / file_path
            depth_file_path = f"images/frame_{str(i).zfill(5)}_{cam_name}_depth.png"
            full_depth_file_path = pathlib.Path(save_dir) / depth_file_path
            visible_depth_file_path = (
                f"images/frame_{str(i).zfill(5)}_{cam_name}_depth_vis.png"
            )
            full_visible_depth_file_path = (
                pathlib.Path(save_dir) / visible_depth_file_path
            )

            # create directory if it doesn't exist
            full_file_path.parent.mkdir(parents=True, exist_ok=True)

            # save normal image
            success = cv2.imwrite(
                str(full_file_path), cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            )
            if not success:
                logging.warning(f"Failed to save image to {full_file_path}")
            success = cv2.imwrite(str(full_depth_file_path), real_depth)
            if not success:
                logging.warning(f"Failed to save depth image to {full_depth_file_path}")
            success = cv2.imwrite(str(full_visible_depth_file_path), vis_depth)
            if not success:
                logging.warning(
                    f"Failed to save depth image to {full_visible_depth_file_path}"
                )
