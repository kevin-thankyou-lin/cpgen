import copy
import pathlib
import re
import xml.etree.ElementTree as ET
from functools import cached_property
from typing import Dict, List, Union

import numpy as np
import open3d as o3d
import pytorch_kinematics as pk
import torch

from demo_aug.configs.base import RobotConfig


class Robot:
    def __init__(self, hardware_cfg: RobotConfig, device: str) -> None:
        self.cfg = hardware_cfg
        self.device = device

        with open(self.cfg.urdf_file_path) as f:
            xml_robot = f.read()

        # Remove encoding declaration from the XML string
        xml_robot = re.sub(r"<\?xml.*\?>", "", xml_robot)

        if self.cfg.robot_type == "panda_arm_hand":
            # use two chains to control both left and right fingers
            self.chain_left_finger: pk.Chain = pk.build_serial_chain_from_urdf(
                xml_robot, "panda_leftfinger"
            ).to(dtype=torch.float32, device=device)

            # self.chain_right_finger: pk.Chain = pk.build_serial_chain_from_urdf(
            #     xml_robot, "panda_rightfinger"
            # ).to(dtype=torch.float32, device=device)

            self.num_joints: int = len(
                self.chain_left_finger.get_joint_parameter_names()
            )
            # self.num_joints += 1

            self.chain = self.chain_left_finger
        else:
            self.chain: pk.Chain = pk.build_serial_chain_from_urdf(
                xml_robot, self.cfg.end_link_name
            ).to(dtype=torch.float32, device=device)
            self.num_joints: int = len(self.chain.get_joint_parameter_names())

        self.hardware_dir = pathlib.Path(self.cfg.urdf_file_path).parent

    @cached_property
    def joint_limits(self) -> Dict[str, Dict[str, float]]:
        # Load and parse the URDF file
        tree = ET.parse(self.cfg.urdf_file_path)
        root = tree.getroot()

        joint_limits: Dict[str, Dict[str, float]] = {}

        # Iterate over the joints in the URDF
        for joint in root.findall(".//joint"):
            joint_name = joint.attrib["name"]
            limit_element = joint.find("limit")

            # Retrieve the joint limits
            lower_limit = float(limit_element.attrib["lower"])
            upper_limit = float(limit_element.attrib["upper"])

            joint_limits[joint_name] = {
                "lower_limit": lower_limit,
                "upper_limit": upper_limit,
            }
        return joint_limits

    def sample_valid_joint_configuration(self) -> torch.Tensor:
        """
        Sample valid joint configuration using joint limits and
        TODO(klin) self-collision-free sampling
        (how does drake do it with their bubble approximations?).
        """
        joint_angles: torch.Tensor = torch.rand(
            self.num_joints, dtype=torch.float32, device=self.device
        )
        joint_angles = (
            joint_angles
            * (self.joint_limits["upper_limit"] - self.joint_limits["lower_limit"])
            + self.joint_limits["lower_limit"]
        )
        return joint_angles

    def update_joint_configuration_mesh(
        self,
        joint_angles: Union[np.ndarray, torch.Tensor, List[float]],
        entire_mesh,
        mesh_list: List,
    ):
        """
        Updates mesh's joint configuration by updating the mesh's vertices.

        :param joint_angles: A numpy array, torch tensor, or list of floats representing the joint angles.
        :param entire_mesh: The mesh to update.
        :param mesh_list: A list of meshes corresponding to each joint.
        """
        chain_joint_angles = joint_angles
        # if self.cfg.robot_type == "panda_arm_hand":
        #     # remove the last joint angle for panda_arm_hand
        #     # as chain 1 only includes up to left finger
        #     chain_joint_angles = joint_angles[:-1]

        forward_kinematics = self.chain.forward_kinematics(
            chain_joint_angles, end_only=False
        )
        vertices = []
        for frame, mesh in zip(self.chain._serial_frames, mesh_list):
            # remove '_frame' from the end of the frame name
            frame_name = frame.name[:-6]
            transform = forward_kinematics[frame_name]
            mesh_points = np.asarray(mesh.vertices)
            mesh_points_torch = torch.tensor(
                mesh_points, device=self.device, dtype=torch.float32
            )
            points_transformed = (
                transform.transform_points(mesh_points_torch).cpu().numpy()
            )
            mesh_points = np.asarray(points_transformed)
            vertices.append(mesh_points)

        # if self.cfg.robot_type == "panda_arm_hand":
        #     chain_joint_angles = np.concatenate(
        #         [joint_angles[:-2], joint_angles[-1:]]
        #     )
        #     forward_kinematics = self.chain_right_finger.forward_kinematics(chain_joint_angles, end_only=False)
        #     # include the last frame for the right finger
        #     frame_name = self.chain_right_finger._serial_frames[-1].name[:-6]
        #     transform = forward_kinematics[frame_name]
        #     mesh_points = np.asarray(mesh.vertices)
        #     mesh_points_torch = torch.tensor(
        #         mesh_points, device=self.device, dtype=torch.float32
        #     )
        #     points_transformed = (
        #         transform.transform_points(mesh_points_torch).cpu().numpy()
        #     )
        #     mesh_points = np.asarray(points_transformed)
        #     vertices.append(mesh_points)

        entire_mesh.vertices = o3d.utility.Vector3dVector(np.concatenate(vertices))

    def get_mesh(self):
        """
        Get robot concatenated mesh and mesh list.

        I think I need specific logic for the franka (esp. for the fingers)
        """
        meshes: List = []
        for frame in self.chain._serial_frames:
            geom_file = frame.link.visuals[0].geom_param
            if geom_file is None:
                continue
            geom_path = self.hardware_dir / geom_file
            mesh = o3d.io.read_triangle_mesh(str(geom_path))
            meshes.append(mesh)

        # if self.cfg.robot_type == "panda_arm_hand":
        #     frame = self.chain_right_finger._serial_frames[-1]
        #     geom_file = frame.link.visuals[0].geom_param
        #     if geom_file is None:
        #         logging.error("No right hand geom file")
        #     geom_path = self.hardware_dir / geom_file
        #     mesh = o3d.io.read_triangle_mesh(str(geom_path))
        #     meshes.append(mesh)

        # concatenate all the meshes
        entire_mesh = copy.deepcopy(meshes[0])
        for mesh in meshes[1:]:
            entire_mesh += mesh

        return entire_mesh, meshes


if __name__ == "__main__":
    # TODO(klin): maybe check the output of the original
    # panda_arm_hand.urdf.xacro urdf?
    # first try not include the right hand thing?
    # try viser for sanity check
    hardware_cfg = RobotConfig()
    robot = Robot(hardware_cfg, "cuda")

    entire_mesh, mesh_list = robot.get_mesh()
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(entire_mesh)
    for i in range(100):
        print(f"iteration {i}")
        # robot.update_joint_configuration_mesh(
        #     np.ones(9) * -0.01 * i, entire_mesh, mesh_list
        # )
        robot.update_joint_configuration_mesh(
            np.ones(8) * -0.01 * i, entire_mesh, mesh_list
        )
        entire_mesh.compute_vertex_normals()
        vis.update_geometry(entire_mesh)
        vis.poll_events()
        vis.update_renderer()
        vis.run()
    vis.destroy_window()
