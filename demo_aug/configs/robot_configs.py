from __future__ import annotations

import json
import math
import pathlib
from dataclasses import dataclass, field
from functools import cached_property
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import tyro
from nerfstudio.cameras.cameras import CameraType

import demo_aug

# TODO(klin): find better way to specify these constants
# lets move these out to robot config??? wait no ... offsetframeinfo probably should be here?? oh actually doesn't have to be?
DRAKE_TO_ROBOMIMIC_GRIPPER_OFFSET_POS = [0, 0, 0.097]
ROBOMIMIC_HAND_GRIPPER_OFFSET_QUAT_WXYZ = [0.707107, 0, 0, -0.707107]
ROBOMIMIC_EE_SITE_FRAME_NAME = "panda_hand_eef_site_robomimic"
ROBOMIMIC_GRIPPER_SITE_FRAME_NAME = (
    "panda_hand_gripper_site_robomimic"  # used by robomimic actions
)
GRIPPER_SRC_FRAME = "panda_hand"
ROBOT_BASE_FRAME_NAME = "panda_link0"


@dataclass
class CameraConfig:
    name: str = "default_camera"
    cx: float = 256.0
    cy: float = 256.0
    fx: float = 618.0386719675123
    fy: float = 618.0386719675123
    k1: Optional[float] = None
    k2: Optional[float] = None
    k3: Optional[float] = None
    p1: Optional[float] = None
    p2: Optional[float] = None
    height: Optional[int] = None
    width: Optional[int] = None
    camera_extrinsics_type: Literal["fixed", "hand_camera"] = "hand_camera"
    camera_extrinsics_params: Optional[Dict[str, float]] = None
    camera_extrinsics_in_robot_base_frame: Optional[List[float]] = None
    camera_type: CameraType = CameraType.PERSPECTIVE
    target: Optional[Tuple[float, float, float]] = field(
        default_factory=lambda: [0.0, 0.0, 0.0]
    )
    up: Optional[Tuple[float, float, float]] = field(
        default_factory=lambda: [0.0, 0.0, 1.0]
    )
    center: Optional[Tuple[float, float, float]] = field(
        default_factory=lambda: [-6.0, 0.0, 0.0]
    )
    near: float = 0.1
    far: float = 10.0
    rendered_output_names: List[str] = field(
        default_factory=lambda: ["rgb", "depth", "accumulation"]
    )

    @cached_property
    def fovy(self) -> float:
        return 2 * np.arctan(self.height / (2 * self.fy))


def _get_camera_params(
    file_path: pathlib.Path,
) -> Tuple[int, int, np.ndarray, np.ndarray]:
    import cv2 as cv

    assert file_path.exists()
    fs = cv.FileStorage(str(file_path), cv.FILE_STORAGE_READ)
    image_width = int(fs.getNode("image_width").real())
    image_height = int(fs.getNode("image_height").real())
    camera_matrix = fs.getNode("camera_matrix").mat()
    dist_coeffs = fs.getNode("distortion_coefficients").mat()
    fs.release()
    return image_width, image_height, camera_matrix, dist_coeffs


@dataclass
class MultiCameraConfig:
    camera_cfgs: List[CameraConfig] = field(default_factory=list)
    camera_intrinsics_dir: Optional[pathlib.Path] = None
    camera_intrinsics_path: Optional[pathlib.Path] = None
    camera_extrinsics_path: Optional[pathlib.Path] = None

    def get_cam_cfg(self, name: str) -> CameraConfig:
        for cam_cfg in self.camera_cfgs:
            if cam_cfg.name == name:
                return cam_cfg

        raise ValueError(f"No camera named {name}")

    @cached_property
    def camera_names(self) -> List[str]:
        return [cam_cfg.name for cam_cfg in self.camera_cfgs]

    # method to load camera configs from directory
    def load_configs_from_files(self) -> None:
        assert (
            self.camera_intrinsics_path.exists() or self.camera_intrinsics_dir.exists()
        ), "camera intrinsics does not exist"
        assert (
            self.camera_extrinsics_path.exists()
        ), "camera extrinsics file does not exist"
        assert (
            self.camera_extrinsics_path.suffix == ".json"
        ), "Assuming camera extrinsics file must be .json"
        with open(self.camera_extrinsics_path) as f:
            camera_extrinsics = json.load(f)

        camera_cfgs: List[CameraConfig] = []
        if self.camera_intrinsics_path.exists():
            # Assuming the JSON file path is stored in a variable called json_file_path
            with open(self.camera_intrinsics_path, "r") as f:
                camera_intrinsics_list = json.load(f)

            for camera_intrinsics in camera_intrinsics_list:
                camera_id = camera_intrinsics["camera_id"]
                camera_cfgs.append(
                    CameraConfig(
                        name=camera_id,
                        cx=camera_intrinsics["cx"],
                        cy=camera_intrinsics["cy"],
                        fx=camera_intrinsics["fx"],
                        fy=camera_intrinsics["fy"],
                        height=camera_intrinsics["h"],
                        width=camera_intrinsics["w"],
                        camera_extrinsics_params=camera_extrinsics[camera_id]["pose"],
                        camera_extrinsics_type="hand_camera",
                    )
                )
        else:
            for camera_intrinsics_file in self.camera_intrinsics_dir.iterdir():
                assert (
                    camera_intrinsics_file.suffix == ".yml"
                ), "Assumes camera intrinsics files are .yml if using dir"
                camera_name = camera_intrinsics_file.stem
                (
                    image_width,
                    image_height,
                    opencv_calibration_camera_matrix,
                    dist_coeffs,
                ) = _get_camera_params(camera_intrinsics_file)
                # TODO(klin): figure out what the camera extrinsics looks like
                camera_cfgs.append(
                    CameraConfig(
                        name=camera_name,
                        cx=opencv_calibration_camera_matrix[0, 2],
                        cy=opencv_calibration_camera_matrix[1, 2],
                        fx=opencv_calibration_camera_matrix[0, 0],
                        fy=opencv_calibration_camera_matrix[1, 1],
                        height=image_height,
                        width=image_width,
                        k1=dist_coeffs[0][0],
                        k2=dist_coeffs[0][1],
                        p1=dist_coeffs[0][2],
                        p2=dist_coeffs[0][3],
                        k3=dist_coeffs[0][4],
                        camera_extrinsics_params=camera_extrinsics[camera_name]["pose"],
                        camera_extrinsics_type="hand_camera",
                    )
                )

        self.camera_cfgs = camera_cfgs
        print(
            f"Loaded camera configs from {self.camera_intrinsics_dir} and {self.camera_extrinsics_path}"
        )


# Q: only diff between obs space i.e. ee site and action space i.e. grip site eef pose is the rotation?
@dataclass
class OffsetFrameInfo:
    offset_quat_wxyz: List
    offset_pos: List
    src_frame: str


@dataclass
class PandaGripperNameToFrameInfo:
    name_to_frame_info: Dict[str, OffsetFrameInfo] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Dict[str, List]]:
        return self.name_to_frame_info

    def get_frame_info(self, name: str) -> OffsetFrameInfo:
        return self.name_to_frame_info[name]

    @staticmethod
    def default() -> PandaGripperNameToFrameInfo:
        default_data = {
            "ee_obs_frame": OffsetFrameInfo(
                offset_quat_wxyz=[1, 0, 0, 0],
                offset_pos=DRAKE_TO_ROBOMIMIC_GRIPPER_OFFSET_POS,
                src_frame=GRIPPER_SRC_FRAME,
            ),
            "ee_action_frame": OffsetFrameInfo(
                offset_quat_wxyz=ROBOMIMIC_HAND_GRIPPER_OFFSET_QUAT_WXYZ,
                offset_pos=DRAKE_TO_ROBOMIMIC_GRIPPER_OFFSET_POS,
                src_frame=GRIPPER_SRC_FRAME,
            ),
            # Add more default frames here as required...
        }
        return PandaGripperNameToFrameInfo(name_to_frame_info=default_data)


@dataclass
class PandaRobotiqNameToFrameInfo:
    """
    For real world panda + robotiq gripper setup, obs and action frames are the same: both panda_link8.
    """

    OBS_FRAME: str = "ee_obs_frame"
    ACTION_FRAME: str = "ee_action_frame"
    name_to_frame_info: Dict[str, OffsetFrameInfo] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Dict[str, List]]:
        return self.name_to_frame_info

    def get_frame_info(self, name: str) -> OffsetFrameInfo:
        return self.name_to_frame_info[name]

    @staticmethod
    def default() -> PandaRobotiqNameToFrameInfo:
        default_data = {
            "ee_obs_frame": OffsetFrameInfo(
                offset_quat_wxyz=np.array([1, 0, 0, 0]),
                offset_pos=np.array([0, 0, 0]),
                src_frame="panda_link8",
            ),
            "ee_action_frame": OffsetFrameInfo(
                offset_quat_wxyz=np.array([1, 0, 0, 0]),
                offset_pos=np.array([0, 0, 0]),
                src_frame="panda_link8",
            ),
        }
        return PandaRobotiqNameToFrameInfo(name_to_frame_info=default_data)


@dataclass
class RobotConfig:
    """
    Ideal way to get robot working in mujoco:

    1. Pass in the URDF file, extrinsics and intrinsics and convert to mujoco xml

    In this case, the URDF file is incomplete because we're calibrating cameras which may move.
    Ideally, we go from URDF + extrinsics to mujoco xml. We'll still probably need to pass in intrinsics separately.
    Not a big deal.

    2. However, we're not quite doing so: instead, we're going to assume access to the mujoco xml file
    that contains the robot arm and hand where things are consistent with the URDF
    """

    urdf_file_path: pathlib.Path = (
        pathlib.Path(demo_aug.__file__).parent
        / "hardware/franka_description/urdf/panda_arm_hand.urdf"
    )

    chain_dtype: Literal["float16", "float32", "int8"] = "float32"
    robot_type: Literal["xarm", "panda", "panda_arm_hand"] = "panda_arm_hand"
    end_link_name: Literal[
        "xarm6link1",
        "xarm6link2",
        "xarm6link3",
        "xarm6link4",
        "xarm6link5",
        "xarm6link6",
        "xarm6link_eef",
        "panda_rightfinger",
    ] = "panda_rightfinger"
    renderer_type: Literal["robomimic"] = "robomimic"
    kinematics_type: Literal["robomimic"] = "robomimic"

    update_last_link_pos: bool = False
    last_link_xml_body_name: str = "robot0_right_hand"
    last_link_pos: Optional[Tuple[float, float, float]] = None
    last_link_quat_wxyz: Optional[Tuple[float, float, float, float]] = None

    update_robot_base_pos: bool = False
    robot_base_xml_body_name: str = "robot0_base"
    robot_base_pos: Optional[Tuple[float, float, float]] = None

    mujoco_xml_file: Optional[pathlib.Path] = None

    remove_table: bool = False
    # convert robomimic to real world URDF
    # strategy:
    # have a separate xml/URDF for real world experiments --> need to ensure all robot stuff uses correct values
    # and a separate xml/URDF for sim experiments in order to re-use the demos
    # "real_robot_model"
    add_fr3_cameras_to_xml: bool = False  # only relevant for mujoco/robomimic rendering
    camera_parent_body: Optional[str] = None
    # Need to specify camera parameters in robot config
    # because rendering is related to the robot / xml
    robot_model_type: Literal["real_FR3_robotiq", "sim_panda_arm_hand"] = (
        "sim_panda_arm_hand"
    )

    # for mujoco/robomimic rendering
    dataset: pathlib.Path = (
        pathlib.Path(demo_aug.__file__).parent.parent.parent
        / "diffusion_policy/data/robomimic/datasets/lift/ph/1_demo.hdf5"
    )

    frame_name_to_frame_info: Union[
        PandaGripperNameToFrameInfo, PandaRobotiqNameToFrameInfo
    ] = PandaGripperNameToFrameInfo.default()
    curobo_robot_file_path: pathlib.Path = (
        pathlib.Path(demo_aug.__file__).parent
        / "envs/motion_planners/robot_models/curobo_franka.yml"
    )
    curobo_robot_ee_link: Optional[str] = "panda_hand"
    default_joint_qpos: List[float] = field(
        default_factory=lambda: [
            0,
            math.pi / 16.0,
            0.00,
            -math.pi / 2.0 - math.pi / 3.0,
            0.00,
            math.pi - 0.2,
            math.pi / 4,
        ]
    )

    robot_background_xml_file: Optional[str] = None

    @cached_property
    def num_joints(self) -> int:
        with open(self.urdf_file_path) as f:
            xml_robot = f.read()

        import pytorch_kinematics as pk

        chain = pk.build_serial_chain_from_urdf(xml_robot, self.end_link_name)
        return len(chain.get_joint_parameter_names())


robot_configs: Dict[str, RobotConfig] = {
    "fr3-real": RobotConfig(
        update_last_link_pos=True,
        last_link_pos=[0, 0, 0.107],
        update_robot_base_pos=True,
        robot_base_pos=[0, 0, 0],
        add_fr3_cameras_to_xml=True,
        frame_name_to_frame_info=PandaRobotiqNameToFrameInfo(),
        camera_parent_body="robot0_right_hand",
        remove_table=True,
        curobo_robot_file_path=pathlib.Path(demo_aug.__file__).parent
        / "envs/motion_planners/robot_models/curobo_franka_robotiq_85.yml",
        curobo_robot_ee_link="panda_link8",
        robot_model_type="real_FR3_robotiq",  # TODO: check what this does --- unclear as hardcoded mjc xml in robot.py
        default_joint_qpos=[
            0,
            -1 / 5 * np.pi,
            0,
            -4 / 5 * np.pi,
            0,
            3 / 5 * np.pi,
            0.0,
        ],
        robot_background_xml_file=(
            "/scr/thankyou/autom/demo-aug/demo_aug/models/panda_robotiq_85/panda_robotiq_85_w_cameras.xml"
        ),
    ),
    "panda-sim": RobotConfig(
        default_joint_qpos=[
            0,
            math.pi / 16.0,
            0.00,
            -math.pi / 2.0 - math.pi / 3.0,
            0.00,
            math.pi - 0.2,
            math.pi / 4,
        ],
    ),
}

RobotConfigUnion = tyro.extras.subcommand_type_from_defaults(
    robot_configs,
    prefix_names=False,  # Omit prefixes in subcommands themselves.
)

AnnotatedRobotConfigUnion = tyro.conf.OmitSubcommandPrefixes[
    RobotConfigUnion
]  # Omit prefixes of flags in subcommands.
