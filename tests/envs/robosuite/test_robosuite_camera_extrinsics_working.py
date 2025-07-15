import pathlib
from typing import Literal

import h5py
import matplotlib.pyplot as plt
import numpy as np
from mujoco_py import MjSim, load_model_from_xml
from robosuite.utils.mjcf_utils import postprocess_model_xml
from scipy.spatial.transform import Rotation as R

from demo_aug.objects.robot_object import MultiCameraConfig
from demo_aug.utils.camera_utils import (
    compute_fov_y,
    get_camera_extrinsic_matrix,
    get_camera_intrinsic_matrix,
)
from demo_aug.utils.mujoco_utils import add_camera_to_xml, update_xml_body_pos

"""
Findings:

To get the equivalent robot0_right_hand body pose in robosuite, set quat="1 0 0 0" and pos="0 0 0.107" for robot0_right_hand body in the xml file.
"""

# multi_camera_config = MultiCameraConfig(
#     camera_intrinsics_dir=pathlib.Path("demo_aug/hardware/example_camera_data/camera_params/"),
#     camera_extrinsics_path=pathlib.Path(
#         "demo_aug/hardware/example_camera_data/camera_calibration/calibration_info.json"
#     ),
# )
multi_camera_config = MultiCameraConfig(
    camera_intrinsics_dir=pathlib.Path(
        "/juno/u/thankyou/pi_cameras/camera_params_lens_pos_1/"
    ),
    camera_extrinsics_path=pathlib.Path(
        "/juno/u/thankyou/autom/R2D2/r2d2/calibration/calibration_info.json"
    ),
)
multi_camera_config.load_configs_from_files()

demo_path = "/juno/u/thankyou/autom/R2D2/data/success/2023-12-04/Mon_Dec__4_23:49:10_2023/trajectory.h5"
# CAMERA_NAME: str = "iprl-raspberrypi-13"
CAMERA_NAME: str = "iprl-raspberrypi-12"

manual_rotate_camera_about_x_axis: bool = True

# create mujoco env and FK to get camera extrinsics
xml_with_camera = "demo_aug/hardware/example_camera_data/franka_agentview_camera.xml"
with open(xml_with_camera, "r") as f:
    xml_string = f.read()


def get_parent_to_camera_transform(
    camera_setup_type: Literal["FR3_URDF_TO_PANDA_XML_CORRECTED_POS"],
    camera_params: np.ndarray,
):
    """
    Returns the transformation matrix from the parent body to the camera.
    """
    if camera_setup_type == "FR3_URDF_TO_PANDA_XML_CORRECTED_POS":
        assert (
            len(camera_params) == 6
        ), f"for {camera_setup_type}, assuming camera_params.shape = (6, ). Got {camera_params.shape} instead."
        # hardcoded data
        camera_offset_pos = camera_params[:3]
        camera_offset_euler = camera_params[3:]

        X_panda_link_8_to_right_hand = np.eye(4)
        panda_link_8_to_right_hand_quat_wxyz = np.array(
            [0.923785, 0, 0, -0.382911]
        )  # taken from robosuite xml's robot0_right_hand body
        # TODO: Pass in the whole RobotConfig or CameraConfig. Leave for now, since then have dependency on xml and urdf.
        # Ugly dependency on xml and urdf, but better to have explicitly. For reproducibility, version control both the xml and urdf.
        X_panda_link_8_to_right_hand[:3, :3] = R.from_quat(
            np.roll(panda_link_8_to_right_hand_quat_wxyz, shift=-1)
        ).as_matrix()
        X_panda_link_8_to_right_hand_inv = np.linalg.inv(X_panda_link_8_to_right_hand)

        X_panda_link_8_to_camera = np.eye(4)
        X_panda_link_8_to_camera[:3, 3] = camera_offset_pos
        X_panda_link_8_to_camera[:3, :3] = R.from_euler(
            "xyz", camera_offset_euler
        ).as_matrix()

        X_right_hand_to_camera = np.dot(
            X_panda_link_8_to_right_hand_inv, X_panda_link_8_to_camera
        )

        # convert to pos and quat_wxyz in string format
        camera_pos = X_right_hand_to_camera[:3, 3]
        camera_quat_wxyz = np.roll(
            R.from_matrix(X_right_hand_to_camera[:3, :3]).as_quat(), shift=1
        )
        camera_pos = " ".join([str(x) for x in camera_pos])
        camera_quat_wxyz = " ".join([str(x) for x in camera_quat_wxyz])

        return X_right_hand_to_camera
    else:
        raise NotImplementedError(f"Unknown camera_setup_type: {camera_setup_type}")


def add_cameras_to_xml(
    xml: str,
    camera_parent_body: str,
    multi_camera_config: MultiCameraConfig,
    remove_default_cameras: bool = False,
    manual_rotate_camera_about_x_axis: bool = False,
) -> str:
    """
    Adds a camera to the given xml string.
    """
    for camera_cfg in multi_camera_config.camera_cfgs:
        assert (
            camera_cfg.camera_extrinsics_type == "hand_camera"
        ), "Only hand_camera supported for now"

        X_parent_to_camera = get_parent_to_camera_transform(
            camera_setup_type="FR3_URDF_TO_PANDA_XML_CORRECTED_POS",
            camera_params=camera_cfg.camera_extrinsics_params,
        )

        if manual_rotate_camera_about_x_axis:
            # rotate about x axis by 180 degrees; seems to point camera in the right direction ...
            X_parent_to_camera[:3, :3] = np.dot(
                X_parent_to_camera[:3, :3],
                np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]),
            )

        camera_pos = X_parent_to_camera[:3, 3]
        camera_quat_wxyz = np.roll(
            R.from_matrix(X_parent_to_camera[:3, :3]).as_quat(), shift=1
        )
        camera_pos = " ".join([str(x) for x in camera_pos])
        camera_quat_wxyz = " ".join([str(x) for x in camera_quat_wxyz])

        fov_y: float = compute_fov_y(
            camera_cfg.fx,
            camera_cfg.fy,
            camera_cfg.width,
            camera_cfg.height,
            degrees=True,
        )
        xml = add_camera_to_xml(
            xml,
            camera_cfg.name,
            camera_pos,
            camera_quat_wxyz,
            parent_body_name=camera_parent_body,
            is_eye_in_hand_camera=camera_cfg.camera_extrinsics_type == "hand_camera",
            fovy=str(fov_y),
        )

    return xml


joint_position = None
cartesian_pose = None
camera_extrinsics_stored = None
extrinsics_offset = None

with h5py.File(demo_path, "r") as file:
    # List all groups
    joint_position = file["observation"]["robot_state"]["joint_positions"][0]
    cartesian_pose = file["observation"]["robot_state"]["cartesian_position"][0]
    extrinsics_offset = file["observation"]["camera_extrinsics"][
        CAMERA_NAME + "_gripper_offset"
    ][0]
    camera_extrinsics_stored = file["observation"]["camera_extrinsics"][CAMERA_NAME][0]
    # image = file["observation"]["camera"]["image"][f"{CAMERA_NAME}:8000"][0]
    image = file["observation"]["image"][f"{CAMERA_NAME}"][0]
    multi_camera_config.camera_cfgs[0].camera_extrinsics_params = extrinsics_offset

# save the above variables in the same data format as the original h5 demo_path in a new h5 file
with h5py.File("demo_path.h5", "w") as file:
    file.create_group("observation")
    file["observation"].create_group("robot_state")
    file["observation"]["robot_state"].create_dataset(
        "joint_positions", data=[joint_position]
    )
    file["observation"]["robot_state"].create_dataset(
        "cartesian_position", data=[cartesian_pose]
    )
    file["observation"].create_group("camera_extrinsics")
    file["observation"]["camera_extrinsics"].create_dataset(
        CAMERA_NAME + "_gripper_offset", data=[extrinsics_offset]
    )
    file["observation"]["camera_extrinsics"].create_dataset(
        CAMERA_NAME, data=[camera_extrinsics_stored]
    )
    file["observation"].create_group("camera")
    file["observation"]["camera"].create_group("image")
    file["observation"]["camera"]["image"].create_dataset(
        f"{CAMERA_NAME}:8000", data=[image]
    )


# post-process xml_string
xml_string = postprocess_model_xml(xml_string)
xml_string = update_xml_body_pos(xml_string, [0, 0, 0], "robot0_base")
xml_string = update_xml_body_pos(xml_string, [0, 0, 0.107], "robot0_right_hand")
xml_string = update_xml_body_pos(
    xml_string, camera_extrinsics_stored[:3], "gripper0_right_gripper"
)

quat_xyzw = R.from_euler("xyz", extrinsics_offset[3:], degrees=False).as_quat()
quat_wxyz = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
# convert to string with spaces
quat_wxyz = " ".join([str(x) for x in quat_wxyz])
pos = " ".join([str(x) for x in extrinsics_offset[:3]])


# Compare FR3 camera extrinsics with robosuite camera extrinsics given the same joint position
X_panda_link_8_to_right_hand = np.eye(4)
panda_link_8_to_right_hand_quat_wxyz = np.array([0.923785, 0, 0, -0.382911])
X_panda_link_8_to_right_hand[:3, :3] = R.from_quat(
    np.roll(panda_link_8_to_right_hand_quat_wxyz, shift=-1)
).as_matrix()
X_panda_link_8_to_right_hand_inv = np.linalg.inv(X_panda_link_8_to_right_hand)

X_panda_link_8_to_camera = np.eye(4)
X_panda_link_8_to_camera[:3, 3] = extrinsics_offset[:3]
X_panda_link_8_to_camera[:3, :3] = R.from_euler(
    "xyz", extrinsics_offset[3:]
).as_matrix()

X_right_hand_to_camera = np.dot(
    X_panda_link_8_to_right_hand_inv, X_panda_link_8_to_camera
)

# convert to pos and quat_wxyz in string format
camera_pos = X_right_hand_to_camera[:3, 3]
camera_quat_wxyz = np.roll(
    R.from_matrix(X_right_hand_to_camera[:3, :3]).as_quat(), shift=1
)
camera_pos = " ".join([str(x) for x in camera_pos])
camera_quat_wxyz = " ".join([str(x) for x in camera_quat_wxyz])


camera_parent_body = "robot0_right_hand"
xml_string = add_cameras_to_xml(
    xml_string,
    camera_parent_body,
    multi_camera_config,
    manual_rotate_camera_about_x_axis=manual_rotate_camera_about_x_axis,
    remove_default_cameras=False,
)

mjpy_model = load_model_from_xml(xml_string)
sim = MjSim(mjpy_model)
sim.forward()
sim.data.qpos[:7] = joint_position
sim.forward()


robosuite_cam_extr = get_camera_extrinsic_matrix(sim, CAMERA_NAME)
robosuite_cam_extr_pos = robosuite_cam_extr[:3, 3]
robosuite_cam_extr_rot = robosuite_cam_extr[:3, :3]
robosuite_cam_extr_euler = R.from_matrix(robosuite_cam_extr_rot).as_euler(
    "xyz", degrees=False
)

camera_extrinsics_stored_quat_wxyz = np.roll(
    R.from_euler("xyz", camera_extrinsics_stored[3:]).as_quat(), shift=1
)

# compare quaternions
robosuite_cam_extr_quat_wxyz = np.roll(
    R.from_matrix(robosuite_cam_extr_rot).as_quat(), shift=1
)
# ensure the first element is positive
if robosuite_cam_extr_quat_wxyz[0] < 0:
    robosuite_cam_extr_quat_wxyz = np.array([-x for x in robosuite_cam_extr_quat_wxyz])
if camera_extrinsics_stored_quat_wxyz[0] < 0:
    camera_extrinsics_stored_quat_wxyz = np.array(
        [-x for x in camera_extrinsics_stored_quat_wxyz]
    )

# camera_extrinsics_stored
assert np.allclose(robosuite_cam_extr_pos, camera_extrinsics_stored[:3], atol=1e-5), (
    "robosuite cam extrinsics does not match stored camera extrinsics. May have updated xml related to camera"
    " incorrectly"
)
assert (
    np.allclose(
        robosuite_cam_extr_quat_wxyz, camera_extrinsics_stored_quat_wxyz, atol=1e-5
    )
    or manual_rotate_camera_about_x_axis
), (
    "robosuite cam extrinsics quat does not match stored camera extrinsics quat. May have updated xml related to camera"
    " incorrectly"
)

for cam_name in [CAMERA_NAME, "agentview"]:
    im_width = multi_camera_config.camera_cfgs[0].width
    im_height = multi_camera_config.camera_cfgs[0].height
    im = sim.render(im_width, im_height, camera_name=f"{cam_name}", device_id=0)[::-1]
    plt.imsave(f"{cam_name}.png", im)
    print(f"saved {cam_name}.png")
    if cam_name == CAMERA_NAME:
        # save original image too
        plt.imsave(f"{cam_name}_original.png", image)
        print(f"saved {cam_name}_original.png")

matrix = get_camera_intrinsic_matrix(sim, CAMERA_NAME, im_height, im_width)
print(f"matrix: {matrix}")


# check cartesian position corresponds to link8's position
