import pathlib
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Literal

import numpy as np
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    ConstantVectorSource,
    CoulombFriction,
    DiagramBuilder,
    Quaternion,
    RandomGenerator,
    RigidTransform,
    Simulator,
)
from pydrake.geometry import (
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    Role,
    StartMeshcat,
)
from pydrake.multibody.parsing import Parser
from pydrake.multibody.tree import FixedOffsetFrame, RevoluteJoint

import demo_aug  # for drake models

DRAKE_TO_ROBOMIMIC_GRIPPER_OFFSET = [0, 0, 0.097]
ROBOMIMIC_EE_FRAME_NAME = "panda_hand_gripper_site_robomimic"


def get_obj_file_paths_from_sdf(
    sdf_file_path, parent_tag: Literal["visual", "collision"] = "collision"
):
    # Convert the input path to a Path object for easier manipulation
    sdf_file_path = Path(sdf_file_path).resolve()

    # Get the directory of the .sdf file
    sdf_directory = sdf_file_path.parent

    # Load and parse the .sdf file
    tree = ET.parse(sdf_file_path)
    root = tree.getroot()

    # Initialize a list to hold full paths of .obj files
    obj_file_paths = []
    # import ipdb;ipdb.set_trace()
    print(f'root.findall(".//mesh/uri"): {root.findall(".//mesh/uri")}')
    # Search for all 'uri' elements within 'mesh' elements
    for mesh in root.findall(f".//{parent_tag}/geometry/mesh/uri"):
        # Extract the text content of the uri element
        uri = mesh.text

        # Check if the uri references an .obj file
        if uri and uri.endswith(".obj"):
            # Construct the full path using pathlib's / operator for path concatenation
            full_path = sdf_directory / uri
            obj_file_paths.append(full_path.resolve())

    return obj_file_paths

    # Define the namespace map to handle the XML namespaces properly
    namespace = {"sdf": "http://sdformat.org/schema/sdf/1.8"}
    # Search for all 'uri' elements within 'mesh' elements, considering the namespace
    for mesh in root.findall(
        ".//sdf:visual/sdf:geometry/sdf:mesh/sdf:uri", namespaces=namespace
    ):
        uri_text = mesh.text
        # Check if the uri_text is not None and ends with '.obj'
        if uri_text and uri_text.endswith(".obj"):
            # Construct the full path using pathlib
            full_path = (sdf_file_path.parent / uri_text).resolve()
            obj_file_paths.append(full_path)

    import ipdb

    ipdb.set_trace()

    return obj_file_paths


def main():
    rng = np.random.default_rng(135)  # this is for python
    RandomGenerator(rng.integers(0, 1000))  # this is for c++

    # Set time step for the simulation.
    time_step = 0.0001

    # Create a simple block diagram containing both the plant and the scene graph.
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step)

    # New #############################################################################
    parser = Parser(plant)
    drake_package_path = str(
        pathlib.Path(demo_aug.__file__).parent / "models/package.xml"
    )
    parser.package_map().AddPackageXml(filename=drake_package_path)
    task_irrelev_obj_url: str = "package://models/assets/arenas/table_arena.xml"
    task_irrelev_obj = parser.AddModelsFromUrl(task_irrelev_obj_url)

    franka = parser.AddModelsFromUrl(
        "package://models/franka_description/urdf/panda_arm_hand.urdf"
    )[0]
    body_world_pose = None
    # body_world_pose = RigidTransform([0, 0.5, 1.4])
    base_pos = np.array(
        [-0.56, 0.0, 0.912]
    )  # nerf_robomimic_env.robot_obj.base_pos  querying env causes segfault
    base_quat_wxyz = np.array(
        [1, 0, 0, 0]
    )  # nerf_robomimic_env.robot_obj.base_quat_wxyz
    if body_world_pose is None:
        body_world_pose = RigidTransform(Quaternion(base_quat_wxyz), base_pos)
    plant.WeldFrames(
        plant.world_frame(), plant.GetFrameByName("panda_link0"), body_world_pose
    )

    # task_relev_obj_url = "package://models/assets/task_relevant/mesh-outputs/mesh.obj"
    # task_relev_obj = parser.AddModelsFromUrl(task_relev_obj_url)[0]
    task_relev_obj_url = "demo_aug/models/assets/task_relevant/square_nut_85/2024-02-04-23-04-44-015977/square_nut_85_2024-02-04-23-04-44-015977square_nut-t105-watertight-mesh-convex-decomp0_transformed-multi.sdf"
    # parser.AddModelFromFile(task_relev_obj_url)
    # object_frame = plant.GetFrameByName("square_nut-t105-watertight-mesh-convex-decomp0_transformed")  # Adjust "object_frame_name" as necessary

    # plant.WeldFrames(plant.GetFrameByName("panda_link7"), plant.GetFrameByName("mesh"), RigidTransform(
    #     # Quaternion([[ 0.4610, -0.8793,  0.1198],
    #     # [-0.8767, -0.4722, -0.0922],
    #     # [ 0.1376, -0.0625, -0.9885]]),
    #     # [-0.0950,  0.1132,  1.0169]
    #     )
    # )
    # Set default positions:
    # q0 = [0.0229177, 0.19946329, -0.01342641, -2.63559645, 0.02568405, 2.93396808, 0.79548173]
    q0 = [0.0229177, 2, -0.01342641, -2.63559645, 0.02568405, 2.93396808, 0.79548173]
    # q0 = np.array([0, np.pi / 16.0, 0.00, -np.pi / 2.0 - np.pi / 3.0, 0.00, np.pi - 0.2, np.pi/4])
    q0 = [
        0.1040641,
        0.93151709,
        -0.07581167,
        -1.95661668,
        -0.11392898,
        3.00559677,
        1.26353959,
    ]
    index = 0
    for joint_index in plant.GetJointIndices(franka):
        joint = plant.get_mutable_joint(joint_index)
        if isinstance(joint, RevoluteJoint):
            joint.set_default_angle(q0[index])
            index += 1

    # Add gravity to the plant.
    # plant.mutable_gravity_field().set_gravity_vector([0, 0, 9.81])

    # Add a falling sphere into the plant.

    hand = plant.GetBodyByName("panda_hand")
    X_Hand_Grippersite = RigidTransform(DRAKE_TO_ROBOMIMIC_GRIPPER_OFFSET)
    plant.AddFrame(
        FixedOffsetFrame(ROBOMIMIC_EE_FRAME_NAME, hand.body_frame(), X_Hand_Grippersite)
    )

    # task_relev_obj_url = "package://models/assets/task_relevant/mesh-outputs/mesh.obj"
    # parser.AddModelsFromUrl(task_relev_obj_url)[0]

    # plant.WeldFrames(
    #     plant.GetFrameByName("panda_link7"),
    #     plant.GetFrameByName("square_nut-t105-watertight-mesh-convex-decomp0_transformed"),
    #     RigidTransform(
    #         Quaternion([[0.4610, -0.8793, 0.1198], [-0.8767, -0.4722, -0.0922], [0.1376, -0.0625, -0.9885]]),
    #         [-0.0950, 0.1132, 1.0169],
    #     ),
    # )
    # import the Mesh class
    from pydrake.geometry import Mesh

    # parse the .sdf to get list of .obj files paths
    task_relev_obj_url_sdf = "/scr/thankyou/autom/demo-aug/demo_aug/models/assets/task_relevant/square_nut_85/2024-02-04-23-04-44-015977/square_nut-t105-watertight-mesh-convex-decomp0_transformed.sdf"
    task_relev_obj_url_sdf = "/scr/thankyou/autom/demo-aug/demo_aug/models/assets/task_relevant/square_nut_85/2024-02-04-23-04-44-015977/square_nut_85_2024-02-04-23-04-44-015977square_nut-t105-watertight-mesh-convex-decomp0_transformed-multi.sdf"
    task_relev_obj_urls = get_obj_file_paths_from_sdf(
        task_relev_obj_url_sdf, "collision"
    )
    # task_relev_obj_urls = []
    X_parentframe_obj = np.array(
        [
            [0.99439438, -0.02257143, 0.10329742, -0.24024431],
            [-0.02018599, -0.99950621, -0.02408042, 0.13177443],
            [0.10378995, 0.02186028, -0.99435898, 0.9266631],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    hand_to_object = X_Hand_Grippersite.GetAsMatrix4() @ X_parentframe_obj
    # TODO(klin: Feb 4)
    print(f"task_relev_obj_urls: {task_relev_obj_urls}")
    for i, task_relev_obj_url in enumerate(task_relev_obj_urls):
        print(1)
        visual_mesh = Mesh(str(task_relev_obj_url), 1.0)
        plant.RegisterVisualGeometry(
            hand,
            RigidTransform(hand_to_object),
            visual_mesh,
            f"new_object_visual{i}",
            np.array([0.5, 0.5, 0.5, 1.0]),
        )
        material = CoulombFriction(
            static_friction=0.5, dynamic_friction=0.3
        )  # Adjust friction properties as needed

        collision_mesh = Mesh(str(task_relev_obj_url), 1.0)
        plant.RegisterCollisionGeometry(
            hand,
            RigidTransform(hand_to_object),
            collision_mesh,
            f"new_object_collision{i}",
            material,
        )
    # add box
    # box = plant.AddRigidBody(
    #     "Box",
    #     task_irrelev_obj[0],
    #     SpatialInertia(mass=mass, p_PScm_E=np.array([0.0, 0.0, 0.0]), G_SP_E=UnitInertia.SolidBox(0.1, 0.1, 0.1)),
    # )
    # shape = Box(0.1, 0.1, 0.1)
    # plant.RegisterVisualGeometry(hand, RigidTransform(), shape, "BoxVisual", np.array([0.5, 0.5, 0.5, 1.0]))  # RGBA color
    # plant.RegisterCollisionGeometry(
    #     hand, RigidTransform(), shape, "BoxCollision", CoulombFriction(0.9, 0.8)
    # )  # Friction parameters
    print("finalizing")
    plant.Finalize()

    GetCollisionGeometriesForBody = plant.GetCollisionGeometriesForBody(
        plant.GetBodyByName("panda_hand")
    )
    print(f"GetCollisionGeometriesForBody: {GetCollisionGeometriesForBody}")
    import ipdb

    ipdb.set_trace()
    body_indices = plant.GetBodyIndices(franka)
    bodies = [plant.get_body(index) for index in body_indices]
    # print names of all bodies
    print(f"body names: {[body.name() for body in bodies]}")
    print(f"body_indices: {body_indices}")
    # import ipdb;ipdb.set_trace()
    print(f"finalized plant with {plant.num_bodies()} bodies")
    plant.GetBodyByName("panda_")
    import ipdb

    ipdb.set_trace()
    plant.GetBodyByName("panda_hand")
    all_bodies = plant.bodies()

    # Get the name of each body
    body_names = [body.name() for body in all_bodies]
    print(f"body_names: {body_names}")
    meshcat = StartMeshcat()
    # Connect the visualizer.
    visualizer = MeshcatVisualizer.AddToBuilder(
        builder,
        scene_graph,
        meshcat,
        MeshcatVisualizerParams(role=Role.kIllustration),
    )
    collision_visualizer = MeshcatVisualizer.AddToBuilder(
        builder,
        scene_graph,
        meshcat,
        MeshcatVisualizerParams(prefix="collision", role=Role.kProximity),
    )

    torques = builder.AddSystem(ConstantVectorSource(np.zeros(plant.num_actuators())))
    builder.Connect(torques.get_output_port(), plant.get_actuation_input_port())

    # Create a simulator and set the initial state.
    diagram = builder.Build()
    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()
    plant_context = plant.GetMyContextFromRoot(context)

    wrist_frame = plant.GetFrameByName("panda_link7")
    X_W_EE1 = wrist_frame.CalcPoseInWorld(plant_context)
    print(f"X_W_EE1: {X_W_EE1}")

    # mesh_frame = plant.GetFrameByName("square_nut-t105-watertight-mesh-convex-decomp0_transformed")
    # X_W_nerf1 = mesh_frame.CalcPoseInWorld(plant_context)
    # print(f"X_W_nerf1: {X_W_nerf1}")

    # box = plant.GetJointByName('$world_Box')
    # sphere.set_random_position_distribution([0, 0, 0.5])
    # box.set_position(plant_context, [0, 0, 1.4])
    # plant.GetBodyByName("Box")

    q_orig = plant.GetPositions(plant_context)
    q_res = [
        0.1040641,
        0.93151709,
        -0.07581167,
        -1.95661668,
        -0.11392898,
        3.00559677,
        1.26353959,
        0.04,
        0.04,
    ]
    q_res[5] += np.pi
    q_orig[: len(q_res)] = q_res
    # q_all = q_res + [0 for _ in range(plant.num_positions() - len(q_res))]
    # import ipdb;ipdb.set_trace()
    import time

    a = time.time()
    plant.SetPositions(plant_context, q_orig)
    wrist_frame = plant.GetFrameByName("panda_link7")
    X_W_EE2 = wrist_frame.CalcPoseInWorld(plant_context)
    b = time.time()
    print(f"took {b - a} seconds")
    print(f"X_W_EE2: {X_W_EE2}")

    # mesh_frame = plant.GetFrameByName("square_nut-t105-watertight-mesh-convex-decomp0_transformed")
    # X_W_nerf2 = mesh_frame.CalcPoseInWorld(plant_context)
    # print(f"X_W_nerf2: {X_W_nerf2}")

    # # compute the transform between X_W_EE1 and X_W_EE2
    # X_EE1_EE2 = X_W_EE1.inverse().multiply(X_W_EE2)
    # print(f"X_EE1_EE2: {X_EE1_EE2}")
    # compute the transform between X_W_nerf1 and X_W_nerf2
    # X_nerf1_nerf2 = X_W_nerf1.inverse().multiply(X_W_nerf2)
    # print(f"X_nerf1_nerf2: {X_nerf1_nerf2}")
    # # compute the transform between X_EE1_EE2 and X_nerf1_nerf2
    # X_EE1_EE2_nerf1_nerf2 = X_EE1_EE2.inverse().multiply(X_nerf1_nerf2)
    # print(f"X_EE1_EE2_nerf1_nerf2: {X_EE1_EE2_nerf1_nerf2}")

    visualizer_context = visualizer.GetMyContextFromRoot(context)
    visualizer.ForcedPublish(visualizer_context)
    # meshcat.StartRecording(frames_per_second=10)
    # plant.SetFreeBodyPose(context=plant_context, body=box_body, X_WB=RigidTransform([0, 0, 2.4]))
    # import ipdb

    # ipdb.set_trace()
    # z = 0.94
    z = 0.1
    for body_index in plant.GetFloatingBaseBodies():
        tf = RigidTransform(
            # # UniformlyRandomRotationMatrix(generator),
            [rng.uniform(-0.15, 0.15), rng.uniform(-0.3, 0.3), z],
        )
        plant.SetFreeBodyPose(plant_context, plant.get_body(body_index), tf)
        # z += 0.1
    # get current pose

    collision_visualizer.ForcedPublish(
        collision_visualizer.GetMyContextFromRoot(context)
    )

    # Set the simulation to run in real time.
    simulator.set_target_realtime_rate(1.0)
    simulator.set_publish_every_time_step(False)
    simulator.Initialize()
    print("advancing to 0.5")
    # Run the simulation.
    simulator.AdvanceTo(0.5)
    # simulator.AdvanceTo(5.0)
    meshcat.StopRecording()
    import ipdb

    ipdb.set_trace()

    # Create a QueryObject.
    meshcat.PublishRecording()
    import ipdb

    ipdb.set_trace()

    context = scene_graph.CreateDefaultContext()
    query_object = scene_graph.get_query_output_port().Eval(context)

    task_irrelev_obj_id = plant.GetBodyIndices(task_irrelev_obj[0])[0]
    franka_id = plant.GetBodyIndices(franka)[0]
    # Compute the signed distance between the closest points on the sphere and the cylinder.
    query_object.ComputeSignedDistancePairClosestPoints(task_irrelev_obj_id, franka_id)

    # The sphere should have fallen under the influence of gravity.
    final_state = plant.GetPositions(plant_context)
    print("Final state:", final_state)


if __name__ == "__main__":
    main()
