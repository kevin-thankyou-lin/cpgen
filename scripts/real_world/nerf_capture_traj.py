import time
import xml.etree.ElementTree as ET

import numpy as np
from pydrake.all import (
    CoulombFriction,
    DiagramBuilder,
    Parser,
    RigidTransform,
    SpatialInertia,
    UnitInertia,
)
from pydrake.geometry import (
    CollisionFilterDeclaration,
    Cylinder,
    GeometrySet,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    Role,
    Sphere,
    StartMeshcat,
)
from pydrake.math import BsplineBasis
from pydrake.multibody.inverse_kinematics import (
    AngleBetweenVectorsConstraint,
    DistanceConstraint,
    GazeTargetConstraint,
    MinimumDistanceConstraint,
    PositionConstraint,
)
from pydrake.multibody.meshcat import JointSliders
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.planning import KinematicTrajectoryOptimization
from pydrake.solvers import Solve
from pydrake.trajectories import BsplineTrajectory

"""This script generates a trajectory for the collection of images of an object on a
tabletop with which we will train a NeRF.
"""

DEBUG = True  # set to true to use joint sliders

# ################# #
# BSPLINE FUNCTIONS #
# ################# #


def deBoor(
    s: float,
    knots: np.ndarray,
    control_points: np.ndarray,
    deg: int,
) -> np.ndarray:
    """Evaluates spline(s).

    See: en.wikipedia.org/wiki/De_Boor%27s_algorithm.

    Arguments
    ---------
    s : float
        Parameter value 0 <= s <= 1.
    knots : np.ndarray, shape=(n + 1 + order)
        Padded array of knots.
    control_points : np.ndarray, shape=(n + 1, dimension)
        Array of control points.
    deg : int
        Degree of B-spline. Is one less than the order.

    Returns
    -------
    spline(s) : np.ndarray
        The spline evaluated at parameter value s.
    """
    # preprocessing
    assert knots.shape == (control_points.shape[0] + deg + 1,)
    s = np.clip(s, 0.0, 1.0)  # ensures 0.0 <= s <= 1.0

    # setting up algorithm
    i_knot = (s >= knots).nonzero()[0][-1]  # index of interval containing s
    d = [control_points[j + i_knot - deg] for j in range(0, deg + 1)]

    # executing recursion
    for r in range(1, deg + 1):
        for j in range(deg, r - 1, -1):
            alpha = (s - knots[j + i_knot - deg]) / (
                knots[j + 1 + i_knot - r] - knots[j + i_knot - deg]
            )
            d[j] = (1.0 - alpha) * d[j - 1] + alpha * d[j]
    return d[deg]


def deBoorDerivative(
    s: float,
    knots: np.ndarray,
    control_points: np.ndarray,
    deg: int,
) -> np.ndarray:
    """
    Evaluates spline'(x), the derivative of the B-spline.

    See: stackoverflow.com/questions/57507696.

    Parameters
    ----------
    s : float
        Parameter value 0 <= s <= 1.
    knots : np.ndarray, shape=(n + 1 + order)
        Padded array of knots.
    control_points : np.ndarray, shape=(n + 1, dimension)
        Array of control points.
    deg : int
        Degree of B-spline. Is one less than the order.

    Returns
    -------
    spline'(s) : np.ndarray
        The spline derivative evaluated at parameter value s.
    """
    # preprocessing
    assert knots.shape == (control_points.shape[0] + deg + 1,)
    s = np.clip(s, 0.0, 1.0)  # ensures 0.0 <= s <= 1.0

    # setting up algorithm
    i_knot = (s >= knots).nonzero()[0][-1]  # index of interval containing s
    q = [
        deg
        * (control_points[j + i_knot - deg + 1] - control_points[j + i_knot - deg])
        / (knots[j + i_knot + 1] - knots[j + i_knot - deg + 1])
        for j in range(0, deg)
    ]

    # executing recursion
    for r in range(1, deg):
        for j in range(deg - 1, r - 1, -1):
            right = j + 1 + i_knot - r
            left = j + i_knot - (deg - 1)
            alpha = (s - knots[left]) / (knots[right] - knots[left])
            q[j] = (1.0 - alpha) * q[j - 1] + alpha * q[j]

    return q[deg - 1]


# ############################# #
# COLLISION FILTERING FUNCTIONS #
# ############################# #


def get_collision_geometries(plant, body_name):
    try:
        return plant.GetCollisionGeometriesForBody(plant.GetBodyByName(body_name))
    except RuntimeError:
        print(f"Could not find {body_name}")
        return


def disable_collision(plant, collision_filter_manager, allowed_collision_pair):
    declaration = CollisionFilterDeclaration()
    set1 = GeometrySet()
    set2 = GeometrySet()
    set1_geometries = get_collision_geometries(plant, allowed_collision_pair[0])
    if set1_geometries is None:
        return
    set2_geometries = get_collision_geometries(plant, allowed_collision_pair[1])
    if set2_geometries is None:
        return
    set1.Add(set1_geometries)
    set2.Add(set2_geometries)
    declaration.ExcludeBetween(set1, set2)
    collision_filter_manager.Apply(declaration)


def load_srdf_disabled_collisions(srdf_file, plant, collision_filter_manager):
    """A function that disables collisions based on an SRDF file.

    From: https://stackoverflow.com/questions/76783635/how-to-allow-collisions-based-on-srdf-file
    """
    tree = ET.parse(srdf_file)
    robot = tree.getroot()
    for disable_collisions in robot.iter("disable_collisions"):
        allowed_collision_pair = [
            disable_collisions.get("link1"),
            disable_collisions.get("link2"),
        ]
        disable_collision(plant, collision_filter_manager, allowed_collision_pair)


# ##### #
# UTILS #
# ##### #


def PublishPositionTrajectory(
    trajectory, root_context, plant, visualizer, time_step=1.0 / 33.0
):
    """
    https://github.com/RussTedrake/manipulation/blob/8c3e596528c439214d63926ba011522fdf25c04a/manipulation/meshcat_utils.py#L454
    """
    plant_context = plant.GetMyContextFromRoot(root_context)
    visualizer_context = visualizer.GetMyContextFromRoot(root_context)

    visualizer.StartRecording(False)

    for t in np.append(
        np.arange(trajectory.start_time(), trajectory.end_time(), time_step),
        trajectory.end_time(),
    ):
        root_context.SetTime(t)
        plant.SetPositions(plant_context, trajectory.value(t))
        visualizer.ForcedPublish(visualizer_context)

    visualizer.StopRecording()
    visualizer.PublishRecording()


DEBUG = False


def load_scene(
    object_name: str = "glass",
    robot_file: str = "models/franka_panda_polymetis_real_world/panda_arm.urdf",
    package_file: str = "models/package.xml",
    robot_base_link: str = "panda_link0",
):
    # boilerplate loading the scene
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
    parser = Parser(plant, scene_graph)
    parser.package_map().AddPackageXml(filename=package_file)

    robot = parser.AddModelFromFile(robot_file)
    body_world_pose = RigidTransform()
    plant.WeldFrames(
        plant.world_frame(), plant.GetFrameByName(robot_base_link), body_world_pose
    )

    if object_name == "glass":
        shape_name = "capsule"

        # Create a model instance for the sphere
        sphere_model_instance = plant.AddModelInstance("sphere_model")

        # Add a rigid body to the plant with the specific model instance
        body = plant.AddRigidBody(
            f"{shape_name}_body",
            sphere_model_instance,
            SpatialInertia(
                mass=0.1, p_PScm_E=[0.0, 0.0, 0.0], G_SP_E=UnitInertia(0.01, 0.01, 0.01)
            ),
        )

        # Define the sphere geometry
        sphere_radius = 0.1
        obj_geometry = Sphere(sphere_radius)

        # Define friction properties
        default_friction = CoulombFriction(static_friction=0.9, dynamic_friction=0.5)

        # weld_location = RigidTransform([0.4, 0.18, 0.1])  # specify the desired location
        weld_location = RigidTransform([0.4, 0.18, 0.1])  # specify the desired location

    elif object_name == "wine_glass_holder":
        shape_name = "cylinder"
        # Create a model instance for the Capsule
        model_instance = plant.AddModelInstance(f"{shape_name}_model")

        # Add a rigid body to the plant with the specific model instance
        body = plant.AddRigidBody(
            f"{shape_name}_body",
            model_instance,
            SpatialInertia(
                mass=0.1, p_PScm_E=[0.0, 0.0, 0.0], G_SP_E=UnitInertia(0.01, 0.01, 0.01)
            ),
        )

        # Define the capsule geometry
        cylinder_radius = 0.27 / 2
        cylinder_length = 0.32
        cylinder_radius /= 1
        cylinder_length /= 1
        # cylinder_radius = 0.04 / 2
        # cylinder_length = 0.02
        obj_geometry = Cylinder(cylinder_radius, cylinder_length)

        weld_location = RigidTransform(
            [0.52, -0.18, cylinder_length / 2]
        )  # specify the desired location

        # note the third person view camera is 0.45m to the right i.e. negative y i.e. [x, -0.45, x]
    else:
        raise ValueError(f"Unknown object name: {object_name}")

    # Define friction properties
    default_friction = CoulombFriction(static_friction=0.9, dynamic_friction=0.5)

    # Add the visual and collision geometries to the capsule body
    plant.RegisterVisualGeometry(
        body,
        RigidTransform(),
        obj_geometry,
        f"{shape_name}_visual",
        [0.9, 0.1, 0.1, 1.0],  # RGBA color
    )

    plant.RegisterCollisionGeometry(
        body,
        RigidTransform(),
        obj_geometry,
        f"{shape_name}_collision",
        default_friction,
    )

    # Weld the sphere to a specific location in the world frame
    plant.WeldFrames(plant.world_frame(), body.body_frame(), weld_location)

    plant.Finalize()

    return plant, scene_graph, builder, robot


if __name__ == "__main__":
    target_object = "wine_glass_holder"
    target_object = "glass"
    if target_object == "glass":
        gaze_target = np.array([0.4, 0.15, 0.2])
        p_offset = np.array([0.4, 0.15, 0.2])  # offset of dome center wrt robot base
        r_cam = 0.33  # radius of camera sphere
        n_revs = 4  # number of revolutions to make
        h0 = 0.8  # initial height
        hf = 0.17  # final height
        num_ctrl_pts = 20
        # tol_position = 0.03  # tolerance for position error on the trajectory
        tol_position = 0.1  # tolerance for position error on the trajectory
        tol_ang = np.pi / 4  # tolerance for angular error
        tol_ang = np.pi / 8  # tolerance for angular error
        q_home = np.array(
            [
                1,
                0.0,
                0,
                0,
                0,
                0,
                0,
                np.pi / 4,
                np.pi / 16.0,
                0.00,
                -np.pi / 2.0 - np.pi / 3.0,
                0.00,
                np.pi - 0.2,
                np.pi / 4,
                0,
                0,
            ]
        )
    elif target_object == "wine_glass_holder":
        gaze_target = np.array([0.52, -0.15, 0.2])
        p_offset = np.array([0.4, 0.25, 0.2])  # offset of dome center wrt robot base
        r_cam = 0.33  # radius of camera sphere
        n_revs = 4  # number of revolutions to make
        h0 = 0.8  # initial height
        hf = 0.17  # final height
        num_ctrl_pts = 20
        # tol_position = 0.03  # tolerance for position error on the trajectory
        tol_position = 0.18  # tolerance for position error on the trajectory
        tol_ang = np.pi / 4  # tolerance for angular error
        tol_ang = np.pi / 8  # tolerance for angular error
        q_home = np.array(
            [
                1,
                0.0,
                0,
                0,
                0,
                0,
                0,
                np.pi / 4,
                np.pi / 16.0,
                0.00,
                -np.pi / 2.0 - np.pi / 3.0,
                0.00,
                np.pi - 0.2,
                np.pi / 4,
                0,
                0,
            ]
        )
    else:
        raise ValueError(f"Unknown target object: {target_object}")

    plant, scene_graph, builder, robot = load_scene(target_object)

    # starting meshcat
    meshcat = StartMeshcat()
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
        MeshcatVisualizerParams(
            prefix="collision",
            role=Role.kProximity,
            visible_by_default=True,
        ),
    )
    if DEBUG:
        sliders = builder.AddSystem(
            JointSliders(meshcat, plant, step=1e-4)
        )  # [DEBUG] this will pause the system and show sliders for it so you can mess with it

    # building diagram and retrieving plant context
    diagram = builder.Build()
    diag_context = diagram.CreateDefaultContext()

    # filtering collisions excluded in the SRDF file
    plant_context = plant.GetMyMutableContextFromRoot(diag_context)
    sg_context = scene_graph.GetMyMutableContextFromRoot(diag_context)
    cfm = scene_graph.collision_filter_manager(sg_context)
    # load_srdf_disabled_collisions(ROOT + "/models/fr3_algr.srdf", plant, cfm)

    q_home = q_home[7:]
    q_home = q_home[: plant.num_positions()]
    plant.SetPositions(plant_context, q_home)

    n_positions = plant.num_positions()

    if DEBUG:
        sliders.SetPositions(q_home)
        sliders.Run(diagram, None)  # [DEBUG]

    # defining the optimization program
    cam_frame = plant.GetFrameByName("zedm_left_cam_link", robot)
    trajopt = KinematicTrajectoryOptimization(plant.num_positions(), num_ctrl_pts)
    prog = trajopt.get_mutable_prog()

    # initial guess for the optimizer
    q_guess = np.tile(
        q_home.reshape((n_positions, 1)), (1, trajopt.num_control_points())
    )
    path_guess = BsplineTrajectory(trajopt.basis(), q_guess)
    trajopt.SetInitialGuess(path_guess)

    # adding joint limit constraints
    trajopt.AddPositionBounds(
        plant.GetPositionLowerLimits(), plant.GetPositionUpperLimits()
    )
    trajopt.AddPathVelocityConstraint(
        np.zeros((n_positions, 1)), np.zeros((n_positions, 1)), 0
    )  # start/end with 0 velocity
    trajopt.AddPathVelocityConstraint(
        np.zeros((n_positions, 1)), np.zeros((n_positions, 1)), 1
    )

    # solve two programs: the first without collision avoidance and the second with,
    # where the initial guess into the second program is the first solution
    qo_port = scene_graph.get_query_output_port()
    sg_context = scene_graph.GetMyMutableContextFromRoot(diag_context)
    query_object = qo_port.Eval(sg_context)
    inspector = query_object.inspector()
    col_cands = list(inspector.GetCollisionCandidates())

    for k in range(2):
        for i in range(num_ctrl_pts):
            s = i / (num_ctrl_pts - 1)
            # parametric time in [0, 1]

            # constraints on the camera frame (position and z axis)
            path_start = 0.25  # offsetting the path to get more side views
            path_end = 1.0
            arg = 2 * np.pi * n_revs * s + np.pi / 2
            coeff = r_cam * (path_start * (1 - s) + s * path_end)
            p_cam = p_offset + np.array(
                [
                    coeff * np.cos(arg),
                    1.5 * coeff * np.sin(arg),
                    # h0 - s * (h0 - hf),
                    hf + (h0 - hf) * np.exp(-2.0 * s),
                ]
            )
            p_offset_cam = p_offset + np.array([0.0, 0.0, 0.225])
            cam_x_axis = -(p_cam - p_offset_cam) / np.linalg.norm(p_cam - p_offset_cam)

            # defining the constraint
            pos_constraint = PositionConstraint(
                plant,
                plant.world_frame(),
                p_cam - tol_position,  # lower bounds on the frame
                p_cam + tol_position,  # upper bounds on the frame
                cam_frame,
                np.zeros(3),  # which point we're constraining on the cam_frame
                plant_context,
            )
            ang_constraint_type = "target"
            if ang_constraint_type == "original":
                ang_constraint = AngleBetweenVectorsConstraint(
                    plant,
                    plant.world_frame(),
                    cam_x_axis,
                    cam_frame,
                    np.array([1.0, 0.0, 0.0]),
                    0.0,
                    tol_ang,  # tolerance of angular constraint in radians
                    plant_context,
                )
            elif ang_constraint_type == "z_down":
                # Define the downward direction in the world frame
                world_down = np.array([0.0, 0.0, -1.0])

                # Camera's z-axis in its own frame
                cam_z_axis = np.array([0.0, 0.0, 1.0])

                # Define the angular constraint for the camera's z-axis to point downwards
                ang_constraint = AngleBetweenVectorsConstraint(
                    plant,
                    plant.world_frame(),
                    world_down,
                    cam_frame,
                    cam_z_axis,
                    0.0,
                    tol_ang,  # tolerance of angular constraint in radians
                    plant_context,
                )
            elif ang_constraint_type == "target":
                # Position of the cone's source point (camera position) in the camera frame
                p_AS = np.array(
                    [0.0, 0.0, 0.0]
                )  # Assuming the source is at the camera origin

                # Directional vector representing the center ray of the cone, expressed in the camera frame
                n_A = np.array(
                    [0.0, 0.0, 1.0]
                )  # Assuming the camera's forward direction is -z

                target_point = gaze_target
                # Position of the target point in the target frame
                p_BT = target_point  # Assuming the target point is in the world frame

                # Define the cone half-angle (e.g., 10 degrees)
                cone_half_angle = np.deg2rad(10)  # Convert degrees to radians

                # Create the GazeTargetConstraint
                ang_constraint = GazeTargetConstraint(
                    plant=plant,
                    frameA=cam_frame,
                    p_AS=p_AS,
                    n_A=n_A,
                    frameB=plant.world_frame(),
                    p_BT=p_BT,
                    cone_half_angle=cone_half_angle,
                    plant_context=plant_context,
                )
            trajopt.AddPathPositionConstraint(pos_constraint, s)
            trajopt.AddPathPositionConstraint(ang_constraint, s)

            # adding cost term
            Q = np.eye(n_positions)
            Q[6, 6] = 10.0
            prog.AddQuadraticErrorCost(Q, q_home, trajopt.control_points()[:, i])

            if k == 1:
                for c in col_cands:
                    geometry_id1 = c[0]
                    geometry_id2 = c[1]
                    name1 = plant.GetBodyFromFrameId(
                        inspector.GetFrameId(geometry_id1)
                    ).name()
                    name2 = plant.GetBodyFromFrameId(
                        inspector.GetFrameId(geometry_id2)
                    ).name()
                    if ("algr" in name1 and "table" in name2) or (
                        "table" in name1 and "algr" in name2
                    ):
                        for i in range(num_ctrl_pts):
                            # qi = trajopt.control_points()[i, :]
                            cons = DistanceConstraint(
                                plant,
                                (geometry_id1, geometry_id2),
                                plant_context,
                                0.1,
                                np.inf,
                            )
                            # prog.AddConstraint(cons, qi)
                            trajopt.AddPathPositionConstraint(cons, s)

                min_dist = 0.01
                min_dist_thresh = 0.1
                collision_constraint = MinimumDistanceConstraint(
                    plant, min_dist, plant_context, None, min_dist_thresh
                )
                evaluate_at_s = np.linspace(0, 1, 60)
                for s in evaluate_at_s:
                    trajopt.AddPathPositionConstraint(collision_constraint, s)

        # solving the program
        print("Solving...")
        start = time.time()
        result = Solve(prog)
        end = time.time()
        print(f"Solver exited! Solve time: {end - start}")
        if not result.is_success():
            print("Trajectory optimization failed")
            breakpoint()
            break
        else:
            print("Trajectory optimization succeeded!")
            path_guess = trajopt.ReconstructTrajectory(result)
            trajopt.SetInitialGuess(path_guess)
    ###################################################################################

    if result.is_success():
        viz = collision_visualizer
        PublishPositionTrajectory(
            trajopt.ReconstructTrajectory(result), diag_context, plant, viz
        )

        # solution
        # get num_times random points on the spline
        num_times = 200
        spline_solution = trajopt.ReconstructTrajectory(result)
        rand_times = np.sort(np.random.rand(num_times))
        q_sol = spline_solution.vector_values(rand_times).T  # (num_times, 23)

        # getting the essential elements of the bspline of order k (default=4, deg k-1)
        control_points = np.array(spline_solution.control_points()).squeeze(
            -1
        )  # (num_times, 23)
        basis = spline_solution.basis()
        knots = np.array(basis.knots())  # (num_times + k + 1)
        k = basis.order()

        # save basis.knots() to pkl
        import pickle

        with open("knots.pkl", "wb") as f:
            pickle.dump(basis.knots(), f)
        with open("control_points.pkl", "wb") as f:
            pickle.dump(spline_solution.control_points(), f)

        # # testing
        # s = 0.0
        # print("Spline Value")
        # print(deBoor(s, knots, control_points, k - 1))
        # print(spline_solution.vector_values([s]).flatten())

        # print("Spline Derivative Value")
        # print(deBoorDerivative(s, knots, control_points, k - 1))
        # print(spline_solution.MakeDerivative().vector_values([s]).flatten())

        # Ensure the loaded data dimensions are correct
        num_control_points = len(control_points)
        num_basis_functions = len(knots) - k
        assert num_control_points == num_basis_functions, (
            f"Number of control points ({num_control_points}) does not match the number of basis functions"
            f" ({num_basis_functions})"
        )
        np.save("control_points", control_points)
        np.save("knots", knots)

        with open("knots.pkl", "rb") as f:
            knots = pickle.load(f)
        with open("control_points.pkl", "rb") as f:
            control_points = pickle.load(f)

        speed_factor = 32
        print(f"Round 2: Scaling the knots by {speed_factor}")
        # Ensure the loaded data dimensions are correct
        num_control_points = len(control_points)
        num_basis_functions = len(knots) - k
        assert num_control_points == num_basis_functions, (
            f"Number of control points ({num_control_points}) does not match the number of basis functions"
            f" ({num_basis_functions})"
        )
        # update the knots by scaling by 2x then visualize via PublishPositionTrajectory
        knots = [speed_factor * knot for knot in knots]
        basis = BsplineBasis(4, knots)
        spline = BsplineTrajectory(basis, control_points)
        import ipdb

        ipdb.set_trace()

        knots_path = "knots.pkl"
        control_points_path = "control_points.pkl"
        control_freq = 15

        def get_full_calibration_traj_from_drake(
            knots_path: str,
            control_points_path: str,
            control_freq: int = 15,
            speed_factor: int = 32,
        ) -> np.ndarray:
            import pickle

            from pydrake.math import BsplineBasis
            from pydrake.trajectories import BsplineTrajectory

            with open("knots.pkl", "rb") as f:
                knots = pickle.load(f)
            with open("control_points.pkl", "rb") as f:
                control_points = pickle.load(f)

            knots = [speed_factor * knot for knot in knots]
            basis = BsplineBasis(4, knots)
            spline = BsplineTrajectory(basis, control_points)

            end_time = spline.end_time()
            # times using control freq and end time
            times = np.arange(0, end_time, 1 / control_freq)
            full_calibration_traj = np.array([spline.vector_values(times)]).squeeze(0).T

            return full_calibration_traj

        full_calibration_traj = get_full_calibration_traj_from_drake(
            knots_path, control_points_path, control_freq
        )
        print(full_calibration_traj.shape)
        PublishPositionTrajectory(spline, diag_context, plant, viz)

    breakpoint()
