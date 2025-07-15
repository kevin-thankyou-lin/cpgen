import pathlib
from typing import Dict

import tyro

import demo_aug
from demo_aug.configs.robot_configs import robot_configs
from demo_aug.envs.base_env import (
    CameraConfig,
    EnvConfig,
    MultiCameraConfig,
)

all_env_configs: Dict[str, EnvConfig] = {
    "fr3-real-env": EnvConfig(
        robot_cfg=robot_configs["fr3-real"],
        multi_camera_cfg=MultiCameraConfig(
            camera_intrinsics_path=pathlib.Path(demo_aug.__file__).parent
            / "hardware/camera_intrinsics/intrinsics_desired.json",
            camera_extrinsics_path=pathlib.Path(demo_aug.__file__).parent
            / "hardware/camera_extrinsics/calibration_info.json",
        ),
    ),
    "panda-sim-env": EnvConfig(
        robot_cfg=robot_configs["panda-sim"],
        multi_camera_cfg=MultiCameraConfig(
            # env cfg from robomimic doesn't give the intrinsic params
            # so ran other script and pasting values here
            camera_cfgs=[
                CameraConfig(
                    name="agentview",
                    height=84,
                    width=84,
                    fx=101.3969696,
                    fy=101.3969696,
                    cx=42.0,
                    cy=42.0,
                ),
                CameraConfig(
                    name="robot0_eye_in_hand",
                    height=84,
                    width=84,
                    fx=54.73546566,
                    fy=54.73546566,
                    cx=42.0,
                    cy=42.0,
                ),
            ]
            # [
            #     CameraConfig(
            #         name="agentview",
            #         height=256,
            #         width=256,
            #         fx=309.01933598375615,
            #         fy=309.01933598375615,
            #         cx=128,
            #         cy=128,
            #     ),
            #     CameraConfig(
            #         name="robot0_eye_in_hand",
            #         height=256,
            #         width=256,
            #         fx=166.81284772367434,
            #         fy=166.81284772367434,
            #         cx=128,
            #         cy=128,
            #     ),
            # ]
            # [
            #     CameraConfig(
            #         name="agentview",
            #         height=512,
            #         width=512,
            #         fx=618.0386719675123,
            #         fy=618.0386719675123,
            #         cx=256.0,
            #         cy=256.0,
            #     ),
            #     CameraConfig(
            #         name="robot0_eye_in_hand",
            #         height=512,
            #         width=512,
            #         fx=333.62569545,
            #         fy=333.62569545,
            #         cx=256.0,
            #         cy=256.0,
            #     ),
            # ]
        ),
    ),
}

EnvConfigUnion = tyro.extras.subcommand_type_from_defaults(
    all_env_configs,
    prefix_names=False,  # Omit prefixes in subcommands themselves.
)

AnnotatedEnvConfigUnion = tyro.conf.OmitSubcommandPrefixes[
    EnvConfigUnion
]  # Omit prefixes of flags in subcommands.
