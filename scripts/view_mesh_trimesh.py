"""Meshes
Visualize a mesh. To get the demo data, see `./assets/download_dragon_mesh.sh`.
"""

from pathlib import Path

import numpy as onp
import trimesh

# mesh_file2 = Path(
#     "/scr/thankyou/autom/demo-aug/demo_aug/models/robotiq_85_description/meshes/visual/robotiq_85_base_link.dae"
# )


# TODOKL commit this!
# Create or load your main mesh/scene
scene = trimesh.Scene()

mesh_file1 = Path(
    "/scr/thankyou/autom/demo-aug/demo_aug/models/robotiq_85_description/camera_mount/hand_camera_part_1.stl"
)
mesh_file3 = Path(
    "/scr/thankyou/autom/demo-aug/models/franka_panda_polymetis_real_world/meshes/visual/link7.dae"
)
mesh_file2 = Path(
    "/scr/thankyou/autom/demo-aug/demo_aug/models/robotiq_85_description/camera_mount/hand_camera_part_2.stl"
)

mesh_file2 = Path("/scr/thankyou/autom/demo-aug/point_cloud_new.obj")
# shrink mesh 1
for mesh_file in [mesh_file1, mesh_file2, mesh_file3]:
    mesh = trimesh.load_mesh(str(mesh_file))
    scene.add_geometry(mesh)
    if mesh_file == mesh_file1:
        mesh.apply_scale(0.001)
        # move in +y direction
        mesh.apply_translation([0, 0.32, 0])
        # mesh.apply_translation([0, 0.275, 0])
        # move in +z direction
        mesh.apply_translation([0, 0, 0.098])
        # rotate 180 degree around z axis
        mesh.apply_transform(trimesh.transformations.rotation_matrix(onp.pi, [0, 0, 1]))
    if mesh_file == mesh_file2:
        mesh.apply_scale(0.001)
        # move in +y direction
        mesh.apply_translation([0, 0.28, 0])
        # move in +z direction
        mesh.apply_translation([0, 0, 0.098])
        # rotate 180 degree around z axis
        mesh.apply_transform(trimesh.transformations.rotation_matrix(onp.pi, [0, 0, 1]))

# Create an axis object
axis = trimesh.creation.axis(origin_size=0.01, axis_length=1.0)

# Add the axis to the scene
scene.add_geometry(axis)

# Visualize the scene
scene.show()
