import json
import pathlib

import numpy as np
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    Box,
    CollisionFilterDeclaration,
    ConstantVectorSource,
    CoulombFriction,
    DiagramBuilder,
    GeometrySet,
    RandomGenerator,
    RigidTransform,
    Simulator,
    SpatialInertia,
    UnitInertia,
)
from pydrake.common.eigen_geometry import Quaternion
from pydrake.geometry import (
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    Role,
    StartMeshcat,
)
from pydrake.multibody.parsing import Parser
from pydrake.multibody.tree import FixedOffsetFrame, RevoluteJoint

import demo_aug  # for drake models
from demo_aug.envs.motion_planners.drake_motion_planner import DrakeMotionPlanner

# import demo_aug.envs.motion_planners.drake_motion_planner as DrakeMotionPlanner

DRAKE_TO_ROBOMIMIC_GRIPPER_OFFSET = [0, 0, 0.097]
ROBOMIMIC_EE_FRAME_NAME = "panda_hand_gripper_site_robomimic"


def main():
    rng = np.random.default_rng(135)  # this is for python
    RandomGenerator(rng.integers(0, 1000))  # this is for c++

    # Set time step for the simulation.
    time_step = 0.00001

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

    robot_base_pos = np.array(
        [-0.56, 0.0, 0.912]
    )  # nerf_robomimic_env.robot_obj.robot_base_pos  querying env causes segfault
    robot_base_quat_wxyz = np.array(
        [1, 0, 0, 0]
    )  # nerf_robomimic_env.robot_obj.base_qurobot_at_wxyz

    X_WRobotBase = RigidTransform(Quaternion(robot_base_quat_wxyz), robot_base_pos)
    franka = DrakeMotionPlanner.AddFranka(parser, plant, body_world_pose=X_WRobotBase)
    # task_relev_obj_url = "package://models/assets/task_relevant/mesh-outputs-welded/mesh.sdf"
    task_relev_obj_url = "package://models/assets/task_relevant/mesh-outputs/mesh.obj"
    parser.AddModelsFromUrl(task_relev_obj_url)[0]

    plant.WeldFrames(
        plant.GetFrameByName("panda_link7"),
        plant.GetFrameByName("mesh"),
        RigidTransform(
            Quaternion(
                [
                    [0.4610, -0.8793, 0.1198],
                    [-0.8767, -0.4722, -0.0922],
                    [0.1376, -0.0625, -0.9885],
                ]
            ),
            [-0.0950, 0.1132, 1.0169],
        ),
    )

    X_Hand_Grippersite = RigidTransform(DRAKE_TO_ROBOMIMIC_GRIPPER_OFFSET)
    plant.AddFrame(
        FixedOffsetFrame(
            ROBOMIMIC_EE_FRAME_NAME,
            plant.GetFrameByName("panda_link7"),
            X_Hand_Grippersite,
        )
    )

    with open(
        "demo_aug/models/assets/task_relevant/mesh-outputs-welded/robot_qpos.json", "r"
    ) as f:
        robot_qpos = json.load(f)

    q0 = robot_qpos["robot_qpos"]
    index = 0
    for joint_index in plant.GetJointIndices(franka):
        joint = plant.get_mutable_joint(joint_index)
        if isinstance(joint, RevoluteJoint):
            joint.set_default_angle(q0[index])
            index += 1
        print(f"index: {index}")

    # Add gravity to the plant.
    plant.mutable_gravity_field().set_gravity_vector([0, 0, -9.81])

    # Add a falling sphere into the plant.
    mass = 1.0  # kg
    # add box
    box = plant.AddRigidBody(
        "Box",
        task_irrelev_obj[0],
        SpatialInertia(
            mass=mass,
            p_PScm_E=np.array([0.0, 0.0, 0.0]),
            G_SP_E=UnitInertia.SolidBox(0.1, 0.1, 0.1),
        ),
    )
    shape = Box(0.1, 0.1, 0.1)
    plant.RegisterVisualGeometry(
        box, RigidTransform(), shape, "BoxVisual", np.array([0.5, 0.5, 0.5, 1.0])
    )  # RGBA color
    plant.RegisterCollisionGeometry(
        box, RigidTransform(), shape, "BoxCollision", CoulombFriction(0.9, 0.8)
    )  # Friction parameters

    # Create a QueryObject.
    context = scene_graph.CreateDefaultContext()
    query_object = scene_graph.get_query_output_port().Eval(context)
    inspector = query_object.inspector()

    hand_geoms = []
    obj_collision_geoms = []
    fids = inspector.GetAllFrameIds()
    for f in fids:
        name = inspector.GetName(f)
        print(f"name: {name}")
        if (
            "hand" in name
            or "finger" in name
            or "link8" in name
            or "link7" in name
            or "link6" in name
            or "link" in name
        ):
            hand_geoms.append(f)
        elif "mesh" in name:
            obj_collision_geoms.append(f)

    ftip_set = GeometrySet(hand_geoms)
    obj_set = GeometrySet(obj_collision_geoms)

    cfm = scene_graph.collision_filter_manager()

    cfm.Apply(CollisionFilterDeclaration().ExcludeBetween(ftip_set, obj_set))

    # get EE pose in world frame
    plant.Finalize()

    meshcat = StartMeshcat()
    # Connect the visualizer.
    MeshcatVisualizer.AddToBuilder(
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
    finger_joint1 = plant.GetJointByName("panda_finger_joint1")
    finger_joint2 = plant.GetJointByName("panda_finger_joint2")
    finger_joint1.set_translation(plant_context, q0[-2])
    finger_joint2.set_translation(plant_context, q0[-1])

    plant.GetBodyByName("Box")
    meshcat.StartRecording(frames_per_second=10)
    collision_visualizer.ForcedPublish(
        collision_visualizer.GetMyContextFromRoot(context)
    )

    # Set the simulation to run in real time.
    simulator.set_target_realtime_rate(1.0)
    simulator.set_publish_every_time_step(False)
    simulator.Initialize()

    # Run the simulation.
    simulator.AdvanceTo(1)
    # simulator.AdvanceTo(5.0)
    meshcat.StopRecording()

    # https://stackoverflow.com/questions/75104234/pydrake-collisionfiltermanager-not-applying-filter
    # READ the above KL F chatgpt!
    plant.GetBodyIndices(task_irrelev_obj[0])[0]
    plant.GetBodyIndices(franka)[0]

    # The sphere should have fallen under the influence of gravity.
    final_state = plant.GetPositions(plant_context)
    print("Final state:", final_state)

    meshcat.PublishRecording()
    import ipdb

    ipdb.set_trace()


if __name__ == "__main__":
    main()
