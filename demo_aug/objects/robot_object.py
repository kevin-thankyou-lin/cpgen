import logging
import pathlib
from collections import defaultdict
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import h5py
import numpy as np
import torch

# ideally remove as dependency. However, looks necessary for camera rendering. Hack setup.py file instead.
from robosuite.utils.mjcf_utils import postprocess_model_xml
from scipy.spatial.transform import Rotation as R

from demo_aug.configs.robot_configs import MultiCameraConfig, RobotConfig
from demo_aug.objects.nerf_object import MeshObject, NeRFObject
from demo_aug.utils.camera_utils import (
    compute_fov_y,
    get_camera_extrinsic_matrix,
    get_camera_intrinsic_matrix,
    get_real_depth_map,
    get_real_distance_map,
    pos_euler_opencv_to_pos_quat_opengl,
)
from demo_aug.utils.mathutils import random_rotation_matrix_zyx
from demo_aug.utils.mujoco_utils import (
    add_camera_to_xml,
    remove_body_from_xml,
    update_camera_in_xml,
    update_model_xml,
    update_xml_body_pos,
)
from demo_aug.utils.run_script_utils import retry_on_exception


@retry_on_exception(max_retries=10, retry_delay=1, exceptions=(BlockingIOError,))
def get_first_eps_key_states_model(dataset: str) -> Tuple:
    """
    Returns a dict of episode keys to states and model files.
    """
    # check dataset exists first
    assert pathlib.Path(dataset).exists(), f"dataset {dataset} does not exist"

    with h5py.File(dataset, "r") as f:
        first_eps_key = list(f["data"].keys())[0]
        states = f[f"data/{first_eps_key}/states"][()]
        model = f[f"data/{first_eps_key}"].attrs["model_file"]
    return first_eps_key, states, model


class RobotObject:
    """
    Base object class that implements a render method for a given robot joint_qpos and gripper_qpos.

    However, this class has also leaked functionality to store information suchs as robot_base_pos and quat ...
    Non ideal?
    """

    def __init__(
        self, cfg: RobotConfig, multi_camera_cfg: Optional[MultiCameraConfig] = None
    ):
        self.cfg = cfg
        self.multi_camera_cfg = multi_camera_cfg

        if self.cfg.renderer_type == "robomimic":
            import robomimic.utils.env_utils as EnvUtils
            import robomimic.utils.file_utils as FileUtils
            import robomimic.utils.obs_utils as ObsUtils

            dummy_spec = dict(
                obs=dict(
                    low_dim=["robot0_eef_pos"],
                    rgb=[],
                ),
            )
            ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=dummy_spec)
            env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=cfg.dataset)
            if self.cfg.robot_model_type == "real_FR3_robotiq":
                env_meta["env_kwargs"]["gripper_types"] = "Robotiq85Gripper"
                env_meta["env_kwargs"]["mount_types"] = None
            elif self.cfg.robot_model_type == "sim_panda_arm_hand":
                env_meta["env_kwargs"]["gripper_types"] = "default"
            else:
                raise NotImplementedError(
                    f"Unknown robot_model_type: {self.cfg.robot_model_type}"
                )

            self.env = EnvUtils.create_env_from_metadata(
                env_meta=env_meta, render_offscreen=True, use_image_obs=False
            )

            first_eps_key, states, model = get_first_eps_key_states_model(
                str(cfg.dataset)
            )
            self.states = states
            self.model = model

            self.robot_jnt_pos_idxs_in_mjc_state_vec = (
                np.array(self.env.env.robots[0]._ref_joint_pos_indexes) + 1
            )
            self.gr_jnt_pos_idxs_in_mjc_state_vec = (
                np.array(self.env.env.robots[0]._ref_gripper_joint_pos_indexes) + 1
            )

            if self.cfg.robot_model_type == "real_FR3_robotiq":
                # HACK: need to set the robotiq gripper state
                states = np.concatenate(
                    [
                        [0],
                        self.env.env.sim.get_state().qpos,
                        self.env.env.sim.get_state().qvel,
                    ]
                )
                states = states[None, ...]
                self.ROBOTIQ_CLOSED_LEN = 0.8
                self.ROBOTIQ_OPEN_LEN = 0
                states[0][self.gr_jnt_pos_idxs_in_mjc_state_vec] = (
                    np.array([1, -1, 1, 1, -1, 1]) * self.ROBOTIQ_CLOSED_LEN
                )
                self.states = states

                self.model = self.env.env.sim.model.get_xml()
                self.model = update_xml_body_pos(
                    self.model,
                    self.cfg.robot_base_pos,
                    self.cfg.robot_base_xml_body_name,
                )
                self.model = update_xml_body_pos(
                    self.model, self.cfg.last_link_pos, self.cfg.last_link_xml_body_name
                )
                self.model = self._add_cameras_to_xml(
                    self.model, self.cfg.camera_parent_body, self.multi_camera_cfg
                )
                self.model = remove_body_from_xml(self.model, "table")

                # replacing old cameras; add camera names to self.env.env.camera_names
                num_new_cameras = len(self.multi_camera_cfg.camera_cfgs)
                self.env.env.num_cameras = num_new_cameras
                self.env.env.camera_depths = [True] * num_new_cameras
                self.env.env.camera_names = [
                    camera_cfg.name for camera_cfg in self.multi_camera_cfg.camera_cfgs
                ]
                self.env.env.camera_heights = [
                    camera_cfg.height
                    for camera_cfg in self.multi_camera_cfg.camera_cfgs
                ]
                self.env.env.camera_widths = [
                    camera_cfg.width for camera_cfg in self.multi_camera_cfg.camera_cfgs
                ]
                # temp: use the manually updated xml model containing
                # TODO: correct this code to e.g. modify a frontview camera if it's 3pv + general refactor of these methods/data!
                if self.cfg.robot_background_xml_file is not None:
                    with open(self.cfg.robot_background_xml_file, "r") as f:
                        self.model = f.read()

                for camera_cfg in self.multi_camera_cfg.camera_cfgs:
                    camera_name = camera_cfg.name
                    camera_extrinsics_params = camera_cfg.camera_extrinsics_params
                    pos = camera_extrinsics_params[:3]
                    euler_xyz = camera_extrinsics_params[3:]
                    out = pos_euler_opencv_to_pos_quat_opengl(pos, euler_xyz)
                    self.model = update_camera_in_xml(
                        self.model, camera_name, out["pos"], out["quat_wxyz"]
                    )

            self.initial_state_dict = dict(
                states=self.states[0], model=self.model
            )  # TODO(klin): this magic number should come from somewhere else
            # unimportant for the purpose of this class (render object); but good for testing / debugging

            self.joint_qpos_ndim = len(self.robot_jnt_pos_idxs_in_mjc_state_vec)
            self.gripper_qpos_ndim = len(self.gr_jnt_pos_idxs_in_mjc_state_vec)

            self.env.reset()
            self.env.reset_to(self.initial_state_dict)

            self.base_pos = self.env.env.robots[0].base_pos.copy()
            self.base_quat_xyzw = self.env.env.robots[
                0
            ].base_ori.copy()  # robosuite uses xyzw; drake uses wxyz
            self.base_quat_wxyz = np.array(
                [
                    self.base_quat_xyzw[3],
                    self.base_quat_xyzw[0],
                    self.base_quat_xyzw[1],
                    self.base_quat_xyzw[2],
                ]
            )

            # TODO(klin): no way to check if values correspond to correct physical structure
            self.kinematic_ranges = {
                "robot_joint_qpos": self.env.env.sim.model.jnt_range[
                    : self.joint_qpos_ndim
                ].copy(),
                "robot_gripper_qpos": self.env.env.sim.model.jnt_range[
                    self.joint_qpos_ndim : self.joint_qpos_ndim + self.gripper_qpos_ndim
                ].copy(),
            }

            self.default_joint_qpos = torch.FloatTensor(self.cfg.default_joint_qpos)

            self.transform_to_xml: Dict[str, str] = {}
            self.prev_up_to_last_non_se3_transf_hash: Optional[str] = (
                None  # determines if we need to reset env's xml
            )

            self.original_model_geom_rgba = self.env.env.sim.model.geom_rgba.copy()
        else:
            import pytorch_kinematics as pk

            with open(self.cfg.urdf_file_path) as f:
                xml_robot = f.read()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.chain: pk.Chain = pk.build_serial_chain_from_urdf(
                xml_robot, self.cfg.end_link_name
            ).to(dtype=torch.float32, device=device)
            self.num_joints: int = len(self.chain.get_joint_parameter_names())
            self.hardware_dir = pathlib.Path(self.cfg.urdf_file_path).parent

    def randomize_robot_texture(self):
        """
        Randomizes the robot's texture. Currently, randomizes textures for all geoms.
        """
        if self.cfg.renderer_type == "robomimic":
            self.env.env.sim.model.geom_rgba[:, :3] = self.original_model_geom_rgba[
                :, :3
            ] + np.random.uniform(
                -0.3, 0.3, size=(self.original_model_geom_rgba.shape[0], 3)
            )
            self.env.env.sim.model.geom_rgba[:] = np.clip(
                self.env.env.sim.model.geom_rgba, 0, 1
            )

    def sample_object_initial_state(
        self,
    ) -> Dict[str, Tuple[Tuple[float], np.ndarray, Any]]:
        """
        Samples an initial state for the object.

        For a given object, returns xyz, quat_wxyz and the robosuite object instance.
        Note: the z value tends to be much higher than where the object will land. Thus, may need to manually adjust z value.

        robosuite task's reset() does the following, and sim.data assumes wxyz:
            self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))
        """
        return self.env.env.placement_initializer.sample()

    @staticmethod
    def _get_parent_to_camera_transform(
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
            X_panda_link_8_to_right_hand_inv = np.linalg.inv(
                X_panda_link_8_to_right_hand
            )

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

    @staticmethod
    def _add_cameras_to_xml(
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

            X_parent_to_camera = RobotObject._get_parent_to_camera_transform(
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
                is_eye_in_hand_camera=camera_cfg.camera_extrinsics_type
                == "hand_camera",
                fovy=str(fov_y),
                # COMMENT for now: TODO: try these values once camera xml is fixed
                # fx_pixel=str(camera_cfg.fx),
                # fy_pixel=str(camera_cfg.fy),
                # cx_pixel=str(camera_cfg.cx),
                # cy_pixel=str(camera_cfg.cy),
            )

        return xml

    def get_camera_extrinsics(
        self, camera_name: str, robot_joint_qpos: Union[np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        """
        Returns camera extrinsics for the given camera name.

        TODO(klin): should add a test here to make sure that the camera extrinsics
        match results from if I used env.reset() instead of updating sim.data and then sim.forward.
        """
        if self.cfg.kinematics_type == "robomimic":
            reset_states = self.initial_state_dict["states"].copy()
            reset_states[self.robot_jnt_pos_idxs_in_mjc_state_vec] = (
                robot_joint_qpos  # .cpu().numpy()
            )
            reset_qpos = reset_states[1:][: len(self.env.env.sim.data.qpos[:])]
            reset_qvel = reset_states[1:][len(self.env.env.sim.data.qpos[:]) :]
            self.env.env.sim.data.qpos[:] = reset_qpos
            self.env.env.sim.data.qvel[:] = reset_qvel

            self.env.env.sim.forward()

            c2w = get_camera_extrinsic_matrix(self.env.env.sim, camera_name)

            return torch.tensor(c2w, dtype=torch.float32)
        else:
            raise NotImplementedError

    def forward_kinematics(
        self,
        robot_joint_qpos: Union[np.ndarray, torch.Tensor],
        body_name: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the forward kinematics of the robot for the given joint qpos and gripper qpos.
        """
        if self.cfg.kinematics_type == "robomimic":
            reset_states = self.initial_state_dict["states"].copy()
            reset_states[self.robot_jnt_pos_idxs_in_mjc_state_vec] = robot_joint_qpos
            reset_qpos = reset_states[1:][: len(self.env.env.sim.data.qpos[:])]
            reset_qvel = reset_states[1:][len(self.env.env.sim.data.qpos[:]) :]
            self.env.env.sim.data.qpos[:] = reset_qpos
            self.env.env.sim.data.qvel[:] = reset_qvel
            self.env.env.sim.forward()

            if body_name is None:
                # copied from single_arm.py
                # important to copy the field for the correct FK value because otherwise might change
                fk_pos = (
                    self.env.env.robots[0]
                    .sim.data.site_xpos[self.env.env.robots[0].eef_site_id]
                    .copy()
                )
                fk_quat_wxyz = (
                    self.env.env.robots[0]
                    .sim.data.get_body_xquat("robot0_right_hand")
                    .copy()
                )
            else:
                fk_pos = self.env.env.sim.data.get_body_xpos(body_name)
                fk_quat_wxyz = self.env.env.sim.data.get_body_xquat(body_name)
            # ensure first element is positive
            if fk_quat_wxyz[0] < 0:
                fk_quat_wxyz = -fk_quat_wxyz
        return fk_pos, fk_quat_wxyz

    def randomize_camera_pose(self):
        """Randomizes camera extrinsics for the given camera names."""
        if self.cfg.renderer_type == "robomimic":
            if self.cfg.robot_model_type == "real_FR3_robotiq":
                for camera_name in self.env.env.camera_names:
                    # temp: use the manually updated xml model containing
                    # TODO: correct this code to e.g. modify a frontview camera if it's 3pv + general refactor of these methods/data!
                    if self.cfg.robot_background_xml_file is not None:
                        with open(self.cfg.robot_background_xml_file, "r") as f:
                            self.model = f.read()

                    for camera_cfg in self.multi_camera_cfg.camera_cfgs:
                        camera_name = camera_cfg.name
                        camera_extrinsics_params = camera_cfg.camera_extrinsics_params
                        pos = camera_extrinsics_params[:3]
                        euler_xyz = camera_extrinsics_params[3:]
                        # randomize camera extrinsics
                        pos = pos + np.random.uniform(-0.01, 0.01, size=3)
                        euler_xyz = euler_xyz + np.random.uniform(-0.03, 0.03, size=3)
                        out = pos_euler_opencv_to_pos_quat_opengl(pos, euler_xyz)
                        self.model = update_camera_in_xml(
                            self.model, camera_name, out["pos"], out["quat_wxyz"]
                        )

                self.initial_state_dict = dict(
                    states=self.states[0], model=self.model
                )  # TODO(klin): this magic number should come from somewhere else
                # unimportant for the purpose of this class (render object); but good for testing / debugging
                # reset env to new xml
                self.env.reset()
                self.env.reset_to(self.initial_state_dict)
                return

        raise NotImplementedError("randomize_camera_extrinsics() not implemented")

    def get_observation(
        self,
        robot_joint_qpos: Union[np.ndarray, torch.Tensor],
        robot_gripper_qpos: Optional[Union[np.ndarray, torch.Tensor]] = None,
        task_relev_obj: Optional[Dict[str, NeRFObject]] = None,
        task_relev_obj_mesh: Optional[Dict[str, MeshObject]] = None,
        cube_pos_transf: Optional[Union[np.ndarray, torch.Tensor]] = None,
        # TODO(klin): shouldn't be just cube!
        # perhaps pass in whole c_objs that contains all transforms
        cube_rot_transf: Optional[Union[np.ndarray, torch.Tensor]] = None,
        c2w: Optional[torch.Tensor] = None,
        camera_names: Optional[List[str]] = None,
        device: Optional[torch.device] = None,
        heights: Optional[List[int]] = None,
        widths: Optional[List[int]] = None,
        render_task_relev_obj: bool = False,
        debug_robot_obs: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns observations for the given robot joint qpos and gripper qpos.
        """
        assert (
            camera_names is not None or c2w is not None
        ), "either camera_names or c2w must be provided in get_observation()"

        cam_name_to_outputs: Dict[str, Dict[str, torch.Tensor]] = {}

        if self.cfg.renderer_type == "robomimic":
            """
            In robomimic, it's easier to specify camera_names to avoid explicit computation of camera extrinsics.
            """
            assert (
                camera_names is not None
            ), "camera_names must be provided in get_observation() for robomimic"
            if isinstance(robot_joint_qpos, torch.Tensor):
                robot_joint_qpos = robot_joint_qpos.cpu().numpy()
            if isinstance(robot_gripper_qpos, torch.Tensor):
                robot_gripper_qpos = robot_gripper_qpos.cpu().numpy()

            if self.env.name == "Lift" or self.env.name == "NutAssemblySquare":
                task_relevant_obj_pos_idxs = np.array(
                    [
                        self.gr_jnt_pos_idxs_in_mjc_state_vec[-1] + i
                        for i in range(1, 4)
                    ],
                    dtype=np.uint8,
                )
                task_relevant_obj_quat_wxyz_idxs = np.array(
                    [task_relevant_obj_pos_idxs[-1] + i for i in range(1, 5)],
                    dtype=np.uint8,
                )
            elif self.env.name == "PickPlaceCan":
                task_relevant_obj_pos_idxs = np.array([31, 32, 33], dtype=np.uint8)
                task_relevant_obj_quat_wxyz_idxs = np.array(
                    [34, 35, 36, 37], dtype=np.uint8
                )
                # state comes from compressing sim.get_state(), which contains 1D timestep, ND qpos, N?D qvel
            else:
                raise NotImplementedError(
                    f"Unknown env name: {self.env.name}: figure out what the task relevant obj idx for state vector is"
                )

            reset_state_dict = dict(
                states=self.initial_state_dict["states"].copy(),
                model=self.initial_state_dict["model"],
            )

            reset_state_dict["states"][self.robot_jnt_pos_idxs_in_mjc_state_vec] = (
                robot_joint_qpos
            )
            if robot_gripper_qpos is not None:
                if self.cfg.robot_model_type == "real_FR3_robotiq":
                    robot_gripper_qpos = (
                        robot_gripper_qpos[0]
                        * np.array([1, -1, 1, 1, -1, 1])
                        * self.ROBOTIQ_CLOSED_LEN
                    )

                reset_state_dict["states"][self.gr_jnt_pos_idxs_in_mjc_state_vec] = (
                    robot_gripper_qpos
                )

            # resetting takes a long time ... robosuite has this 'black screen of death' that forces a reset
            # dug into the reset and took the required parts: need _initialize_sim() which takes 0.4s ...
            # check if we need to reset
            rgb = self.env.env.sim.render(
                camera_name=camera_names[0],
                height=16,
                width=16,
            )
            if np.all(rgb == 0):
                import ipdb

                ipdb.set_trace()
                self.env.env._load_model()
                self.env.env._postprocess_model()
                self.env.env._initialize_sim()
                self.env.env._reset_internal()
                self.env.env.sim.forward()
                self.env.env.visualize(
                    vis_settings={vis: False for vis in self.env.env._visualizations}
                )

            self.env.env.sim.data.qpos[:] = reset_state_dict["states"][1:][
                : len(self.env.env.sim.data.qpos[:])
            ]
            self.env.env.sim.forward()

            if render_task_relev_obj:
                # roughly copied from tasks' _reset_internal()
                # TODO: refactor this into a modular function --- lots of repeated code
                if self.env.name == "Door":
                    # assuming state post reset is correct?
                    door_body_id = self.env.env.sim.model.body_name2id(
                        self.env.env.door.root_body
                    )
                    door_pos = self.env.env.sim.model.body_pos[door_body_id]
                    door_quat = self.env.env.sim.model.body_quat[door_body_id]

                    new_door_rot = cube_rot_transf @ R.from_quat(door_quat).as_matrix()
                    new_door_pos = cube_rot_transf @ door_pos + cube_pos_transf
                    new_door_quat_xyzw = R.from_matrix(new_door_rot).as_quat()
                    np.array(
                        [
                            new_door_quat_xyzw[3],
                            new_door_quat_xyzw[0],
                            new_door_quat_xyzw[1],
                            new_door_quat_xyzw[2],
                        ]
                    )
                    self.env.env.sim.model.body_pos[door_body_id] = new_door_pos
                    self.env.env.sim.model.body_quat[door_body_id] = (
                        new_door_quat_xyzw  # xyzw is correct
                    )
                elif self.env.name == "Lift":
                    (
                        obj_to_transf_params_seq,
                        obj_to_transf_name_seq,
                        obj_to_transf_type_seq,
                    ) = (
                        defaultdict(dict),
                        defaultdict(list),
                        defaultdict(list),
                    )
                    (
                        obj_to_transf_params_seq_postnonse3,
                        obj_to_transf_name_seq_postnonse3,
                        obj_to_transf_type_seq_postnonse3,
                    ) = (defaultdict(dict), defaultdict(list), defaultdict(list))

                    # TODO(klin): fix this bug - last_non_se3_idx is always -1 b/c
                    # obj_to_transf_type_seq is empty!

                    # TODO(klin): maybe something still buggy w/ caching
                    last_non_se3_idx = -1
                    for obj_name, obj in task_relev_obj.items():
                        transf_params_seq = obj.transform_params_seq
                        transf_name_seq = obj.transform_name_seq
                        transf_type_seq = obj.transform_type_seq

                        for idx, transf_type in enumerate(transf_type_seq):
                            if transf_type.name != "SE3":
                                last_non_se3_idx = idx

                    for obj_name, obj in task_relev_obj.items():
                        transf_params_seq = obj.transform_params_seq
                        transf_name_seq = obj.transform_name_seq
                        transf_type_seq = obj.transform_type_seq

                        obj_to_transf_params_seq[obj_name] = {}
                        for k in range(len(transf_params_seq)):
                            transf_params = transf_params_seq[k]
                            transf_name = transf_name_seq[k]
                            transf_type = transf_type_seq[k]

                            # apply up to the last non-SE3 transform idx
                            if k <= last_non_se3_idx:
                                for sub_key, sub_value in transf_params.items():
                                    sub_key_with_seq = f"{k}:{sub_key}"
                                    obj_to_transf_params_seq[obj_name][
                                        sub_key_with_seq
                                    ] = sub_value
                                    obj_to_transf_name_seq[obj_name].append(transf_name)

                                obj_to_transf_type_seq[obj_name].append(
                                    transf_type.value
                                )
                            else:
                                for sub_key, sub_value in transf_params.items():
                                    sub_key_with_seq = f"{k}:{sub_key}"
                                    obj_to_transf_params_seq_postnonse3[obj_name][
                                        sub_key_with_seq
                                    ] = sub_value
                                obj_to_transf_name_seq_postnonse3[obj_name].append(
                                    transf_name
                                )
                                obj_to_transf_type_seq_postnonse3[obj_name].append(
                                    transf_type.value
                                )

                    obj_to_xml_body_name = {
                        "red_cube": "cube_main",
                    }

                    obj_to_original_joint_qpos: Dict[str, np.ndarray] = {}
                    for obj_name in task_relev_obj.keys():
                        orig_pos = self.states[obj.original_timestep][
                            task_relevant_obj_pos_idxs
                        ]
                        orig_quat_wxyz = self.states[obj.original_timestep][
                            task_relevant_obj_quat_wxyz_idxs
                        ]

                        orig_joint_qpos = np.concatenate([orig_pos, orig_quat_wxyz])
                        obj_to_original_joint_qpos = {obj_name: orig_joint_qpos}

                    xml = postprocess_model_xml(self.model)
                    xml = update_model_xml(
                        xml,
                        obj_to_xml_body_name,
                        obj_to_transf_type_seq,
                        obj_to_transf_params_seq,
                        remove_body_free_joint=False,
                        apply_all_transforms=True,
                        obj_to_original_joint_qpos=obj_to_original_joint_qpos,
                        set_obj_collision_free=True,  # for pure rendering purposes, disable collisions
                    )

                    # get pure se3 transforms that occur after the last non-se3 transform
                    post_non_se3_transf = np.eye(4)
                    for obj_name, obj in task_relev_obj.items():
                        for k in range(last_non_se3_idx + 1, len(transf_params_seq)):
                            X_SE3 = obj_to_transf_params_seq_postnonse3[obj_name][
                                f"{k}:X_SE3"
                            ]

                            # X_SE3_origin = obj_to_transf_params_seq[obj_name][f"{k}:X_SE3_origin"]
                            post_non_se3_transf = X_SE3 @ post_non_se3_transf

                    state = reset_state_dict["states"].copy()
                    post_non_se3_transf_pos = post_non_se3_transf[:3, 3]
                    post_non_se3_transf_quat_xyzw = R.from_matrix(
                        post_non_se3_transf[:3, :3]
                    ).as_quat()
                    post_non_se3_transf_quat_wxyz = np.concatenate(
                        [
                            [post_non_se3_transf_quat_xyzw[3]],
                            post_non_se3_transf_quat_xyzw[:3],
                        ]
                    )

                    state[task_relevant_obj_pos_idxs] = post_non_se3_transf_pos
                    state[task_relevant_obj_quat_wxyz_idxs] = (
                        post_non_se3_transf_quat_wxyz
                    )
                    # TODO(klin): more robust way is to also store object name (to distinguish between objects);
                    # unlikely that two objs have same transfs?
                    current_xml_up_to_last_non_se3_transf_hash: str = repr(
                        obj_to_transf_params_seq
                    )  # repr should be fine if repr'ing on same np object
                    if (
                        self.prev_up_to_last_non_se3_transf_hash is None
                        or current_xml_up_to_last_non_se3_transf_hash
                        != self.prev_up_to_last_non_se3_transf_hash
                    ):
                        # implementation assumes xml is only ever updated here
                        self.prev_up_to_last_non_se3_transf_hash = (
                            current_xml_up_to_last_non_se3_transf_hash
                        )
                        self.env.env.reset_from_xml_string(xml)
                    self.env.env.sim.data.qpos[:] = state[1:][
                        : len(self.env.env.sim.data.qpos)
                    ]
                    self.env.env.sim.forward()

                elif self.env.name == "NutAssemblySquare":
                    obj_to_xml_body_name = {
                        "square_peg": "peg1",
                        "square_nut": "SquareNut_main",
                    }
                    # what was reason for the post non se3 meshes? save time on saving to disk?
                    (
                        obj_to_transf_params_seq,
                        obj_to_transf_name_seq,
                        obj_to_transf_type_seq,
                    ) = (
                        defaultdict(dict),
                        defaultdict(list),
                        defaultdict(list),
                    )
                    (
                        obj_to_transf_params_seq_postnonse3,
                        obj_to_transf_name_seq_postnonse3,
                        obj_to_transf_type_seq_postnonse3,
                    ) = (defaultdict(dict), defaultdict(list), defaultdict(list))

                    last_non_se3_idx = -1
                    for obj_name, obj in task_relev_obj.items():
                        transf_params_seq = obj.transform_params_seq
                        transf_name_seq = obj.transform_name_seq
                        transf_type_seq = obj.transform_type_seq

                        for idx, transf_type in enumerate(transf_type_seq):
                            if transf_type.name != "SE3":
                                last_non_se3_idx = idx

                    for obj_name, obj in task_relev_obj.items():
                        transf_params_seq = obj.transform_params_seq
                        transf_name_seq = obj.transform_name_seq
                        transf_type_seq = obj.transform_type_seq

                        obj_to_transf_params_seq[obj_name] = {}
                        for k in range(len(transf_params_seq)):
                            transf_params = transf_params_seq[k]
                            transf_name = transf_name_seq[k]
                            transf_type = transf_type_seq[k]

                            # apply up to the last non-SE3 transform idx
                            if k <= last_non_se3_idx:
                                for sub_key, sub_value in transf_params.items():
                                    sub_key_with_seq = f"{k}:{sub_key}"
                                    obj_to_transf_params_seq[obj_name][
                                        sub_key_with_seq
                                    ] = sub_value
                                    obj_to_transf_name_seq[obj_name].append(transf_name)

                                obj_to_transf_type_seq[obj_name].append(
                                    transf_type.value
                                )
                            else:
                                for sub_key, sub_value in transf_params.items():
                                    sub_key_with_seq = f"{k}:{sub_key}"
                                    obj_to_transf_params_seq_postnonse3[obj_name][
                                        sub_key_with_seq
                                    ] = sub_value
                                obj_to_transf_name_seq_postnonse3[obj_name].append(
                                    transf_name
                                )
                                obj_to_transf_type_seq_postnonse3[obj_name].append(
                                    transf_type.value
                                )

                    obj_to_original_joint_qpos: Dict[str, np.ndarray] = {}
                    for obj_name in task_relev_obj.keys():
                        orig_pos = self.states[obj.original_timestep][
                            task_relevant_obj_pos_idxs
                        ]
                        orig_quat_wxyz = self.states[obj.original_timestep][
                            task_relevant_obj_quat_wxyz_idxs
                        ]

                        orig_joint_qpos = np.concatenate([orig_pos, orig_quat_wxyz])
                        obj_to_original_joint_qpos = {obj_name: orig_joint_qpos}

                    xml = postprocess_model_xml(self.model)
                    # this format works currently b/c peg doesn't get se3 augs
                    xml = update_model_xml(
                        xml,
                        obj_to_xml_body_name,
                        obj_to_transf_type_seq,
                        obj_to_transf_params_seq,
                        remove_body_free_joint=False,
                        apply_all_transforms=True,
                        obj_to_original_joint_qpos=obj_to_original_joint_qpos,
                        set_obj_collision_free=True,  # for pure rendering purposes, disable collisions
                    )

                    # get pure se3 transforms that occur after the last non-se3 transform
                    post_non_se3_transf = np.eye(4)
                    for obj_name, obj in task_relev_obj.items():
                        if obj_name == "square_peg":
                            print("hardcoding to skip square peg for now")
                            continue
                        for k in range(last_non_se3_idx + 1, len(transf_params_seq)):
                            X_SE3 = obj_to_transf_params_seq_postnonse3[obj_name][
                                f"{k}:X_SE3"
                            ]
                            X_SE3_origin = obj_to_transf_params_seq_postnonse3[
                                obj_name
                            ][f"{k}:X_SE3_origin"]
                            post_non_se3_transf = X_SE3_origin @ (
                                X_SE3
                                @ (np.linalg.inv(X_SE3_origin) @ post_non_se3_transf)
                            )

                    state = reset_state_dict["states"].copy()
                    post_non_se3_transf_pos = post_non_se3_transf[:3, 3]
                    post_non_se3_transf_quat_xyzw = R.from_matrix(
                        post_non_se3_transf[:3, :3]
                    ).as_quat()
                    post_non_se3_transf_quat_wxyz = np.concatenate(
                        [
                            [post_non_se3_transf_quat_xyzw[3]],
                            post_non_se3_transf_quat_xyzw[:3],
                        ]
                    )

                    state[task_relevant_obj_pos_idxs] = post_non_se3_transf_pos
                    state[task_relevant_obj_quat_wxyz_idxs] = (
                        post_non_se3_transf_quat_wxyz
                    )
                    # TODO(klin): more robust way is to also store object name (to distinguish between objects);
                    # unlikely that two objs have same transfs?
                    # We have this up to nonse3 transform to avoid needing to reset the env many times
                    current_xml_up_to_last_non_se3_transf_hash: str = repr(
                        obj_to_transf_params_seq
                    )  # repr should be fine if repr'ing on same np object
                    if (
                        self.prev_up_to_last_non_se3_transf_hash is None
                        or current_xml_up_to_last_non_se3_transf_hash
                        != self.prev_up_to_last_non_se3_transf_hash
                    ):
                        # implementation assumes xml is only ever updated here
                        self.prev_up_to_last_non_se3_transf_hash = (
                            current_xml_up_to_last_non_se3_transf_hash
                        )
                        self.env.env.reset_from_xml_string(xml)

                    self.env.env.sim.data.qpos[:] = state[1:][
                        : len(self.env.env.sim.data.qpos)
                    ]
                    self.env.env.sim.forward()
                elif self.env.name == "PickPlaceCan":
                    obj_to_xml_body_name = {
                        "can": "Can_main",  # TODO(klin): check this please
                    }
                    # what was reason for the post non se3 meshes? save time on saving to disk?
                    (
                        obj_to_transf_params_seq,
                        obj_to_transf_name_seq,
                        obj_to_transf_type_seq,
                    ) = (
                        defaultdict(dict),
                        defaultdict(list),
                        defaultdict(list),
                    )
                    (
                        obj_to_transf_params_seq_postnonse3,
                        obj_to_transf_name_seq_postnonse3,
                        obj_to_transf_type_seq_postnonse3,
                    ) = (defaultdict(dict), defaultdict(list), defaultdict(list))

                    last_non_se3_idx = -1
                    for obj_name, obj in task_relev_obj.items():
                        transf_params_seq = obj.transform_params_seq
                        transf_name_seq = obj.transform_name_seq
                        transf_type_seq = obj.transform_type_seq

                        for idx, transf_type in enumerate(transf_type_seq):
                            if transf_type.name != "SE3":
                                last_non_se3_idx = idx

                    for obj_name, obj in task_relev_obj.items():
                        transf_params_seq = obj.transform_params_seq
                        transf_name_seq = obj.transform_name_seq
                        transf_type_seq = obj.transform_type_seq

                        obj_to_transf_params_seq[obj_name] = {}
                        for k in range(len(transf_params_seq)):
                            transf_params = transf_params_seq[k]
                            transf_name = transf_name_seq[k]
                            transf_type = transf_type_seq[k]

                            # apply up to the last non-SE3 transform idx
                            if k <= last_non_se3_idx:
                                for sub_key, sub_value in transf_params.items():
                                    sub_key_with_seq = f"{k}:{sub_key}"
                                    obj_to_transf_params_seq[obj_name][
                                        sub_key_with_seq
                                    ] = sub_value
                                    obj_to_transf_name_seq[obj_name].append(transf_name)

                                obj_to_transf_type_seq[obj_name].append(
                                    transf_type.value
                                )
                            else:
                                for sub_key, sub_value in transf_params.items():
                                    sub_key_with_seq = f"{k}:{sub_key}"
                                    obj_to_transf_params_seq_postnonse3[obj_name][
                                        sub_key_with_seq
                                    ] = sub_value
                                obj_to_transf_name_seq_postnonse3[obj_name].append(
                                    transf_name
                                )
                                obj_to_transf_type_seq_postnonse3[obj_name].append(
                                    transf_type.value
                                )

                    obj_to_original_joint_qpos: Dict[str, np.ndarray] = {}
                    for obj_name in task_relev_obj.keys():
                        orig_pos = self.states[obj.original_timestep][
                            task_relevant_obj_pos_idxs
                        ]
                        orig_quat_wxyz = self.states[obj.original_timestep][
                            task_relevant_obj_quat_wxyz_idxs
                        ]

                        orig_joint_qpos = np.concatenate([orig_pos, orig_quat_wxyz])
                        obj_to_original_joint_qpos = {obj_name: orig_joint_qpos}

                    # set print options
                    np.set_printoptions(precision=3, suppress=True)
                    xml = postprocess_model_xml(self.model)
                    # this format works currently b/c peg doesn't get se3 augs
                    # removing this thing works kinda... why?
                    # however, doesn't translate the guy appropriately .. why?
                    xml = update_model_xml(
                        xml,
                        obj_to_xml_body_name,
                        obj_to_transf_type_seq,
                        obj_to_transf_params_seq,
                        remove_body_free_joint=False,
                        apply_all_transforms=True,
                        obj_to_original_joint_qpos=obj_to_original_joint_qpos,
                        set_obj_collision_free=True,  # for pure rendering purposes, disable collisions
                    )

                    # get pure se3 transforms that occur after the last non-se3 transform
                    post_non_se3_transf = np.eye(4)
                    for obj_name, obj in task_relev_obj.items():
                        for k in range(last_non_se3_idx + 1, len(transf_params_seq)):
                            X_SE3 = obj_to_transf_params_seq_postnonse3[obj_name][
                                f"{k}:X_SE3"
                            ]
                            X_SE3_origin = obj_to_transf_params_seq_postnonse3[
                                obj_name
                            ][f"{k}:X_SE3_origin"]

                            post_non_se3_transf = X_SE3_origin @ (
                                X_SE3
                                @ (np.linalg.inv(X_SE3_origin) @ post_non_se3_transf)
                            )

                    state = reset_state_dict["states"].copy()
                    post_non_se3_transf_pos = post_non_se3_transf[:3, 3]
                    post_non_se3_transf_quat_xyzw = R.from_matrix(
                        post_non_se3_transf[:3, :3]
                    ).as_quat()
                    post_non_se3_transf_quat_wxyz = np.concatenate(
                        [
                            [post_non_se3_transf_quat_xyzw[3]],
                            post_non_se3_transf_quat_xyzw[:3],
                        ]
                    )

                    state[task_relevant_obj_pos_idxs] = post_non_se3_transf_pos
                    state[task_relevant_obj_quat_wxyz_idxs] = (
                        post_non_se3_transf_quat_wxyz
                    )
                    # TODO(klin): more robust way is to also store object name (to distinguish between objects);
                    # unlikely that two objs have same transfs?
                    # We have this up to nonse3 transform to avoid needing to reset the env many times
                    current_xml_up_to_last_non_se3_transf_hash: str = repr(
                        obj_to_transf_params_seq
                    )  # repr should be fine if repr'ing on same np object
                    if (
                        self.prev_up_to_last_non_se3_transf_hash is None
                        or current_xml_up_to_last_non_se3_transf_hash
                        != self.prev_up_to_last_non_se3_transf_hash
                    ):
                        # implementation assumes xml is only ever updated here
                        self.prev_up_to_last_non_se3_transf_hash = (
                            current_xml_up_to_last_non_se3_transf_hash
                        )
                        self.env.env.reset_from_xml_string(xml)

                    self.env.env.sim.data.qpos[:] = state[1:][
                        : len(self.env.env.sim.data.qpos)
                    ]
                    self.env.env.sim.forward()
                else:
                    raise NotImplementedError(
                        f"env {self.env.name} not implemented for rendering task relevant objects with mujoco rendering"
                    )
            else:
                # roughly copied from tasks' _reset_internal()
                if self.env.name == "Door":
                    # could also just reset robot stuff above but update background stuff here
                    door_pos = np.array([100, 0, 0])
                    door_quat_xyzw = np.array([0, 0, 0, 1])  # wxyz
                    door_body_id = self.env.env.sim.model.body_name2id(
                        self.env.env.door.root_body
                    )
                    self.env.env.sim.model.body_pos[door_body_id] = door_pos
                    self.env.env.sim.model.body_quat[door_body_id] = door_quat_xyzw
                elif self.env.name == "Lift":
                    obj_pos = np.array([100, 0, 0])
                    obj_quat = np.array([1, 0, 0, 0])  # wxyz
                    # TODO(klin): not sure if wxyz or xyzw is correct here ...
                    self.env.env.sim.data.set_joint_qpos(
                        self.env.env.cube.joints[0],
                        np.concatenate([np.array(obj_pos), np.array(obj_quat)]),
                    )
                    self.env.env.sim.forward()
                elif self.env.name == "NutAssemblySquare":
                    obj_pos = np.array([100, 0, 0])
                    obj_quat_xyzw = np.array([0, 0, 0, 1])  # wxyz

                    self.peg1_body_id = self.env.env.sim.model.body_name2id("peg1")
                    self.peg2_body_id = self.env.env.sim.model.body_name2id("peg2")
                    # TODO(klin): unclear what this stuff actually
                    keep_peg1_fixed = False
                    if not keep_peg1_fixed:
                        self.env.env.sim.model.body_pos[self.peg1_body_id] = obj_pos
                        self.env.env.sim.model.body_quat[self.peg1_body_id] = (
                            obj_quat_xyzw
                        )

                    for nut in self.env.env.nuts:
                        self.env.env.sim.data.set_joint_qpos(
                            nut.joints[0],
                            np.concatenate(
                                [
                                    np.array(obj_pos) + np.random.uniform(size=(3,)),
                                    np.array(obj_quat_xyzw),
                                ]
                            ),
                        )
                    pre_step = self.env.env.sim.get_state()
                    self.env.env.sim.forward()
                    post_step = self.env.env.sim.get_state()

                    if not np.allclose(pre_step.qpos, post_step.qpos, atol=1e-6):
                        print("pre_step and pos_step are different")
                        import ipdb

                        ipdb.set_trace()
                elif self.env.name == "PickPlaceCan":
                    obj_pos = np.array([100, 0, 0])
                    obj_quat = np.array([1, 0, 0, 0])
                    obj_joint_name = "Can_joint0"
                    self.env.env.sim.data.set_joint_qpos(
                        obj_joint_name,
                        np.concatenate([np.array(obj_pos), np.array(obj_quat)]),
                    )
                    self.env.env.sim.forward()
                else:
                    import ipdb

                    ipdb.set_trace()
                    raise NotImplementedError(f"env {self.env.name} not implemented")

            for idx, cam_name in enumerate(self.env.env.camera_names):
                if cam_name in camera_names:
                    rgb, depth = self.env.env.sim.render(
                        camera_name=cam_name,
                        height=heights[idx]
                        if heights is not None
                        else self.env.env.camera_heights[idx],
                        width=widths[idx]
                        if widths is not None
                        else self.env.env.camera_widths[idx],
                        depth=True,
                    )
                    rgb = rgb[::-1]
                    depth = depth[::-1]
                    real_depth = get_real_depth_map(self.env.env.sim, depth)
                    cam_intrinsic_matrix = get_camera_intrinsic_matrix(
                        self.env.env.sim,
                        cam_name,
                        heights[idx]
                        if heights is not None
                        else self.env.env.camera_heights[idx],
                        widths[idx]
                        if widths is not None
                        else self.env.env.camera_widths[idx],
                    )
                    real_distance = get_real_distance_map(
                        real_depth,
                        cam_intrinsic_matrix[0, 0],
                        cam_intrinsic_matrix[1, 1],
                        cam_intrinsic_matrix[0, 2],
                        cam_intrinsic_matrix[1, 2],
                    )

                    if (real_depth == 0).sum() > 0:
                        import ipdb

                        ipdb.set_trace()
                        # I think mujoco just gives an actual depth though!
                        logging.warning(
                            "Need to handle zero depth case (if mujoco renders zero depth for inf depth)                "
                            "                     for alpha compositing"
                        )

                    cam_name_to_outputs[cam_name] = {
                        "rgb": torch.tensor(
                            rgb.copy() / 255, dtype=torch.float32, device=device
                        ),
                        "depth": torch.tensor(
                            real_distance.copy(), dtype=torch.float32, device=device
                        ),
                        "accumulation": torch.ones(
                            size=real_distance.shape, device=device
                        ),
                    }

            if debug_robot_obs:
                import datetime

                import imageio

                now = datetime.datetime.now()
                now_day_str = now.strftime("%d")
                now_str = now.strftime("%H-%M-%S")

                robot_obs_dir = pathlib.Path("robot_obs") / now_day_str
                robot_obs_dir.mkdir(parents=True, exist_ok=True)

                # save robot rgb as images for all cameras
                for name, outputs in cam_name_to_outputs.items():
                    rgb = outputs["rgb"].cpu().numpy()
                    # convert to uint8
                    rgb = (rgb * 255).astype(np.uint8)
                    imageio.imwrite(
                        str(robot_obs_dir / f"robot_rgb_{name}_{now_str}.png"), rgb
                    )
                    print(
                        f"saved images to {robot_obs_dir / f'robot_rgb_{name}_{now_str}.png'}"
                    )
            return cam_name_to_outputs

    def sample_valid_joint_qpos(
        self,
        output_convention: str = "drake",
        sample_near_default_qpos: bool = False,
        near_qpos_scaling: float = 0.1,
        sample_near_eef_pose: bool = False,
        center_eef_pos: Optional[np.ndarray] = None,
        center_eef_quat_xyzw: Optional[np.ndarray] = None,
        sample_pos_x_bound: Optional[Tuple[float, float]] = None,
        sample_pos_y_bound: Optional[Tuple[float, float]] = None,
        sample_pos_z_bound: Optional[Tuple[float, float]] = None,
        sample_min_height: Optional[float] = None,
        sample_rot_angle_z_bound: Optional[Tuple[float, float]] = None,
        sample_rot_angle_y_bound: Optional[Tuple[float, float]] = None,
        sample_rot_angle_x_bound: Optional[Tuple[float, float]] = None,
        forward_kinematics_body_name: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Samples a valid joint qpos for the given robot.

        sample_pos_bound: a bound on the distance from the center_eef_pos to sample from

        # TODO(klin): add collision checking
        """
        sampled_values = {}
        if self.cfg.kinematics_type == "robomimic":
            from scipy.spatial.transform import Rotation as R

            if sample_near_eef_pose:
                assert (
                    center_eef_pos is not None
                    and center_eef_quat_xyzw is not None
                    and sample_pos_x_bound is not None
                    and sample_pos_y_bound is not None
                    and sample_pos_z_bound is not None
                    and sample_min_height is not None
                ), "must provide center_eef_pos, center_eef_quat_xyzw, sample_pos_xyz_bound, sample_min_height"
                if isinstance(center_eef_pos, np.ndarray):
                    center_eef_pos = torch.tensor(center_eef_pos)
                if isinstance(center_eef_quat_xyzw, np.ndarray):
                    center_eef_quat_xyzw = torch.tensor(center_eef_quat_xyzw)

                new_eef_pos = center_eef_pos + torch.tensor(
                    [
                        torch.empty(1).uniform_(*sample_pos_x_bound),
                        torch.empty(1).uniform_(*sample_pos_y_bound),
                        torch.empty(1).uniform_(*sample_pos_z_bound),
                    ]
                )
                while new_eef_pos[2] < sample_min_height:
                    new_eef_pos = center_eef_pos + torch.tensor(
                        [
                            torch.empty(1).uniform_(*sample_pos_x_bound),
                            torch.empty(1).uniform_(*sample_pos_y_bound),
                            torch.empty(1).uniform_(*sample_pos_z_bound),
                        ]
                    )

                rand_rot = random_rotation_matrix_zyx(
                    sample_rot_angle_z_bound,
                    sample_rot_angle_y_bound,
                    sample_rot_angle_x_bound,
                )
                rand_rot = R.from_matrix(rand_rot)
                orig_rot = R.from_quat(center_eef_quat_xyzw)
                new_eef_quat_xyzw = torch.tensor((rand_rot * orig_rot).as_quat())
                # I guess we throw away rotations that can't physically motion plan to the goal

                if self.cfg.robot_model_type == "real_FR3_robotiq":
                    new_gripper_qpos = (
                        torch.FloatTensor([1, -1, 1, 1, -1, 1])
                        * np.random.uniform(
                            self.ROBOTIQ_OPEN_LEN, self.ROBOTIQ_CLOSED_LEN
                        )  # uniform between open and closed
                    )
                else:
                    gripper_qpos_range = self.kinematic_ranges["robot_gripper_qpos"]
                    # Generate random values within the specified range
                    new_gripper_qpos = (
                        torch.FloatTensor(len(gripper_qpos_range)).uniform_()
                        * (gripper_qpos_range[:, 1] - gripper_qpos_range[:, 0])
                        + gripper_qpos_range[:, 0]
                    )
                if output_convention == "drake":
                    new_gripper_qpos[1] = -new_gripper_qpos[1]

                sampled_values["robot_ee_pos"] = new_eef_pos
                sampled_values["robot_ee_quat_xyzw"] = new_eef_quat_xyzw
                sampled_values["robot_gripper_qpos"] = new_gripper_qpos
                sampled_values["robot_ee_quat_wxyz"] = torch.tensor(
                    [
                        new_eef_quat_xyzw[3],
                        new_eef_quat_xyzw[0],
                        new_eef_quat_xyzw[1],
                        new_eef_quat_xyzw[2],
                    ]
                )
            elif sample_near_default_qpos:
                for key, value in self.kinematic_ranges.items():
                    if key == "robot_joint_qpos":
                        sampled_values[key] = (
                            self.default_joint_qpos
                            + torch.randn(len(self.default_joint_qpos))
                            * near_qpos_scaling
                        )
                    else:
                        # Generate random values within the specified range
                        sampled_values[key] = (
                            torch.FloatTensor(len(value)).uniform_()
                            * (value[:, 1] - value[:, 0])
                            + value[:, 0]
                        )

                    if self.cfg.robot_model_type == "real_FR3_robotiq":
                        new_gripper_qpos = (
                            torch.FloatTensor([1, -1, 1, 1, -1, 1])
                            * np.random.uniform(
                                self.ROBOTIQ_OPEN_LEN, self.ROBOTIQ_CLOSED_LEN
                            )  # uniform between open and closed
                        )
                        sampled_values["robot_gripper_qpos"] = new_gripper_qpos

                    if output_convention == "drake" and key == "robot_gripper_qpos":
                        sampled_values[key][1] = -sampled_values[key][1]
                # apply forward kinematics to get ee pos and quat
                sampled_values["robot_ee_pos"], sampled_values["robot_ee_quat_wxyz"] = (
                    self.forward_kinematics(
                        sampled_values["robot_joint_qpos"],
                        body_name=forward_kinematics_body_name,
                    )
                )
            else:
                raise NotImplementedError("Unknown sampling method for robomimic")
        return sampled_values
