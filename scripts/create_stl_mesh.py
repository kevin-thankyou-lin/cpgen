# TODO(klin): this is a script to create stl meshes from xml files containing simple shapes

import pathlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh
import tyro

import demo_aug
from demo_aug.utils.mujoco_utils import convert_xml_to_transformed_mesh


@dataclass
class Config:
    # filepath isn't used at the moment: hardcoding the relevant xml string instead
    filepath: str = "/juno/u/thankyou/autom/diffusion_policy/data/robomimic/datasets/square/ph/demo_0-nut_step_80_original.xml"
    output_dir: Optional[pathlib.Path] = (
        Path(demo_aug.__file__).parent.parent / "models/assets/objects/meshes"
    )
    object_name: str = "square_nut"


def main(cfg: Config) -> None:
    xml_str = None
    if cfg.obj_name == "square_nut":
        xml_str = """<body name="SquareNut_main" pos="0 0 0">
                <inertial pos="0.0109551 0 0" quat="0 0.707107 0 0.707107"
                mass="0.0158068" diaginertia="2.91963e-05 1.87672e-05 1.14829e-05" />
                <joint name="SquareNut_joint0" type="free" damping="0.0005" />
                <geom name="SquareNut_g0" size="0.0105 0.04375 0.01" pos="-0.03325 0 0" type="box"
                condim="4" friction="0.95 0.3 0.2" solimp="0.998 0.998 0.001 0.5 2" rgba="0.5 0 0 1" />
                <geom name="SquareNut_g1" size="0.03125 0.0105 0.01" pos="0 0.03325 0" type="box"
                condim="4" friction="0.95 0.3 0.1" solimp="0.998 0.998 0.001 0.5 2" rgba="0.5 0 0 1" />
                <geom name="SquareNut_g2" size="0.03125 0.0105 0.01" pos="0 -0.03325 0" type="box"
                condim="4" friction="0.95 0.3 0.1" solimp="0.998 0.998 0.001 0.5 2" rgba="0.5 0 0 1" />
                <geom name="SquareNut_g3" size="0.0105 0.04375 0.01" pos="0.03325 0 0" type="box"
                condim="4" friction="0.95 0.3 0.1" solimp="0.998 0.998 0.001 0.5 2" rgba="0.5 0 0 1" />
                <geom name="SquareNut_g4" size="0.02525 0.015875 0.01" pos="0.054 0 0" type="box"
                condim="4" friction="0.95 0.3 0.1" solimp="0.998 0.998 0.001 0.5 2" rgba="0.5 0 0 1" />
            </body>"""
    elif cfg.obj_name == "cube":
        print("Creating cube mesh: hardcoded code implementation")
        # Define the dimensions of the cube (X, Y, Z)
        dimensions = [0.0212431, 0.020076, 0.0214241]
        dimensions = np.array(dimensions) * 2

        # Create the cube mesh
        cube_mesh = trimesh.creation.box(extents=dimensions)

        # Export the cube mesh as an STL file
        # stl_file_path = "/juno/u/thankyou/autom/demo-aug/models/assets/objects/meshes/cube_mesh.stl"
        stl_file_path = (
            Path(demo_aug.__file__).parent.parent
            / "models/assets/objects/meshes/cube_mesh.stl"
        )

        cube_mesh.export(stl_file_path)

        print(f"Cube mesh has been exported to {stl_file_path}")
        # print out vertex locations
        print(cube_mesh.vertices)
    else:
        raise NotImplementedError(f"Object {cfg.obj_name} not supported")

    convert_xml_to_transformed_mesh(xml_str, cfg.output_dir)


if __name__ == "__main__":
    tyro.cli(main)
