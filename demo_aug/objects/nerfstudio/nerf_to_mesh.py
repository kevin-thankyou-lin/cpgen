import pathlib
import time
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh
import viser
from nerfstudio.fields.base_field import Field

# from nerf_grasping.grasp_utils import load_nerf_field, get_nerf_configs
from nerfstudio.models.base_model import Model
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils import eval_utils
from skimage.measure import marching_cubes
from typing_extensions import Unpack


def get_nerf_configs(nerf_checkpoints_path: str) -> List[pathlib.Path]:
    return list(pathlib.Path(nerf_checkpoints_path).rglob("nerfacto/*/config.yml"))


def load_nerf_model(cfg_path: pathlib.Path) -> Model:
    return load_nerf_pipeline(cfg_path).model


def load_nerf_field(cfg_path: pathlib.Path) -> Field:
    return load_nerf_model(cfg_path).field


def load_nerf_pipeline(cfg_path: pathlib.Path, test_mode="inference") -> Pipeline:
    _, pipeline, _, _ = eval_utils.eval_setup(cfg_path, test_mode=test_mode)
    return pipeline


def sdf_to_mesh(
    sdf: Callable[[torch.Tensor], float],
    npts: int = 31,
    lb: np.ndarray = -np.ones(3),
    ub: np.ndarray = np.ones(3),
) -> Unpack[Tuple[np.ndarray, ...]]:
    """Converts an SDF to a mesh using marching cubes.

    Parameters
    ----------
    sdf : Callable[[torch.Tensor], float]
        A signed distance field. The input into this function has shape (B, 3).
    npts : int, default=31
        Number of points used to grid space in each dimension for marching cubes.
    lb : np.ndarray, default=-np.ones(3)
        Lower bound for marching cubes.
    ub : np.ndarray, default=np.ones(3)
        Upper bound for marching cubes.


    Returns
    -------
    verts : np.ndarray, shape=(nverts, 3)
        The vertices of the mesh.
    faces : np.ndarray, shape=(nfaces, 3), type=int
        The vertex indices associated with the corners of each face.
    normals : np.ndarray, shape=(nfaces, 3)
        The surface normals associated with each face.
    """
    # running marching cubes to extract the isosurface
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, Y, Z = np.mgrid[
        lb[0] : ub[0] : npts * 1j,
        lb[1] : ub[1] : npts * 1j,
        lb[2] : ub[2] : npts * 1j,
    ]
    pts_plot = np.stack((X, Y, Z), axis=-1)  # (npts, npts, npts, 3)
    pts_plot_flat = torch.tensor(pts_plot.reshape((-1, 3)), device=device)  # (B, 3)
    vol = np.array(sdf(pts_plot_flat).reshape(pts_plot.shape[:-1]))

    min_density, max_density = vol.min(), vol.max()

    threshold = 0.0
    if not (min_density <= threshold <= max_density):
        raise ValueError(
            f"Threshold must be within the volume data range: [{min_density}, {max_density}]."
        )

    # import ipdb; ipdb.set_trace()
    _verts, faces, normals, _ = marching_cubes(vol, 0.0, allow_degenerate=False)
    verts = (ub - lb) * _verts / (np.array(X.shape) - 1) + lb  # scaling verts properly
    return verts, faces, normals


def nerf_to_mesh(
    field,
    level: float,
    npts: int = 31,
    lb: np.ndarray = -np.ones(3),
    ub: np.ndarray = np.ones(3),
    scale: float = 1.0,
    min_len: Optional[float] = 200,  # Default 200 to get rid of floaters
    flip_faces: bool = True,
    save_path: Optional[Path] = None,
) -> trimesh.Trimesh:
    """Takes a nerfstudio pipeline field and plots or saves a mesh.

    Parameters
    ----------
    field
        The nerfstudio field.
    level : float
        The density level to treat as the 0-level set.
    npts : int, default=31
        Number of points used to grid space in each dimension for marching cubes.
    lb : np.ndarray, default=-np.ones(3)
        Lower bound for marching cubes.
    ub : np.ndarray, default=np.ones(3)
        Upper bound for marching cubes.
    scale : float, default=1.0
        The scale to apply to the mesh.
    min_len : Optional[float], default=None
        Minimum number of edges to be considered a relevant component, used to remove floaters
    flip_faces : bool, default=True
        Whether to flip the faces, helps get correct signed distance values
        (it appears that the faces are flipped inside out by default)
    save_path : Optional[Path], default=None
        The save path. If None, shows a plot instead.

    Returns
    -------
    mesh : trimesh.Trimesh
        The mesh.
    """

    # marching cubes
    def sdf(x):
        return field.density_fn(x).cpu().detach().numpy() - level

    verts, faces, normals = sdf_to_mesh(
        sdf,
        npts=npts,
        lb=lb,
        ub=ub,
    )

    if flip_faces:
        faces = np.fliplr(faces)

    # making a trimesh mesh
    mesh = trimesh.Trimesh(
        vertices=verts, faces=faces, vertex_normals=normals, process=False
    )
    mesh.apply_transform(trimesh.transformations.scale_matrix(scale))
    if False and min_len is not None:
        cc = trimesh.graph.connected_components(mesh.face_adjacency, min_len=min_len)
        ONLY_LARGEST_COMPONENT = False
        if ONLY_LARGEST_COMPONENT:
            # Identify the largest component by the number of faces
            largest_component = max(cc, key=len)

            # Create a mask for the largest component
            mask = np.zeros(len(mesh.faces), dtype=bool)
            mask[largest_component] = True

            # Update the mesh to keep only the largest component
            mesh.update_faces(mask)
            mesh.remove_unreferenced_vertices()

        else:
            mask = np.zeros(len(mesh.faces), dtype=bool)
            mask[np.concatenate(cc)] = True
            mesh.update_faces(mask)

    # saving/visualizing
    if save_path is None:
        assert isinstance(mesh, trimesh.Trimesh)

        # load mesh from path
        # path = "demo_aug/models/assets/task_relevant/nerf_outputs/nerfacto-/scr/thankyou/autom/demo-aug/data/real_world_recordings/2024-06-12-21:26:48-glass-good/nerfacto/2024-06-14_165522/meshes/watertight-mesh.obj"
        # path = Path(path)
        # mesh = trimesh.load_mesh(path)
        vertices = mesh.vertices
        faces = mesh.faces
        print(f"Loaded mesh with {vertices.shape} vertices, {faces.shape} faces")

        server = viser.ViserServer()
        server.add_mesh_simple(
            name="/simple",
            vertices=vertices,
            faces=faces,
            wxyz=(1.0, 0.0, 0.0, 0.0),
            position=(0.0, 0.0, 0.0),
        )
        server.add_mesh_trimesh(
            name="/trimesh",
            mesh=mesh.smoothed(),
            wxyz=(1.0, 0.0, 0.0, 0.0),
            position=(0.0, 0.0, 0.0),
        )

        while True:
            time.sleep(10.0)

        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        _ = ax.plot_trisurf(
            verts[:, 0],
            verts[:, 1],
            faces,
            verts[:, 2],
        )
        ax.set_aspect("equal")
        plt.savefig(f"mesh_{level}.png")
        print(f"mesh_{level}.png")
    else:
        # create directory if it doesn't exist
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        mesh.export(save_path)
        print(f"Saved mesh extracted from nerf to {save_path}")

    return mesh


if __name__ == "__main__":
    NERF_CHECKPOINTS_PATH = Path("data/2023-09-11_20-52-40/nerfcheckpoints")
    # /scr/thankyou/autom/demo-aug/nerf_outputs/nerfacto-/scr/thankyou/autom/demo-aug/data/real_world_recordings/2024-06-07-for-nerf/wine-glass-holder-more-texture/nerfstudio/nerfacto/2024-06-08_165908/nerfstudio_models
    nerf_checkpoints_path = Path(
        "/scr/thankyou/autom/demo-aug/nerf_outputs/nerfacto-/scr/thankyou/autom/demo-aug/data/real_world_recordings/2024-06-07-for-nerf/wine-glass-holder-more-texture/nerfstudio/nerfacto/2024-06-08_165908/nerfstudio_models"
    )
    nerf_checkpoints_path = "/scr/thankyou/autom/demo-aug/nerf_outputs/nerfacto-/scr/thankyou/autom/demo-aug/data/real_world_recordings/2024-06-07-for-nerf/wine-glass-holder-more-texture/nerfstudio/"
    nerf_checkpoints_path = "/scr/thankyou/autom/demo-aug/nerf_outputs/nerfacto-/scr/thankyou/autom/demo-aug/data/real_world_recordings/2024-06-07-for-nerf/wine-glass-holder-more-texture/nerfstudio/"
    nerf_checkpoints_path = "nerf_outputs/nerfacto-/scr/thankyou/autom/demo-aug/data/real_world_recordings/2024-06-12-21:26:48-glass-good/"
    # nerf_checkpoints_path = Path("/scr/thankyou/autom/demo-aug/nerf_outputs/nerfacto-data/robomimic/lift/newest/35")
    # nerf_checkpoints_path = Path("outputs/nerfacto-v1.1lift/")
    IDX = 0
    RADIUS = 1
    # LEVEL = 50.0
    LEVEL = 15

    nerf_configs = get_nerf_configs(
        nerf_checkpoints_path=str(nerf_checkpoints_path),
    )
    nerf_config = nerf_configs[IDX]
    # nerf_config = "nerf_outputs/nerfacto-/scr/thankyou/autom/demo-aug/data/real_world_recordings/2024-06-12-21:26:48-glass-good/nerfacto/2024-06-13_192056/config.yml"
    nerf_config = "nerf_outputs/nerfacto-/scr/thankyou/autom/demo-aug/data/real_world_recordings/2024-06-13-00:51:14-glass-holder-start-near-agentview-good/nerfacto/2024-06-14_152057/nerfstudio_models/config.yml"
    nerf_config = "nerf_outputs/nerfacto-/scr/thankyou/autom/demo-aug/data/real_world_recordings/2024-06-13-00:51:14-glass-holder-start-near-agentview-good/nerfacto/2024-06-14_152057/config.yml"
    nerf_config = "nerf_outputs/nerfacto-/scr/thankyou/autom/demo-aug/data/real_world_recordings/2024-06-12-21:26:48-glass-good/nerfacto/2024-06-14_151443/config.yml"
    nerf_config = "outputs/unnamed/splatfacto/2024-07-10_225215/config.yml"
    nerf_config = "/scr/thankyou/autom/demo-aug/outputs/unnamed/nerfacto/2024-07-10_230511/config.yml"
    nerf_config = "nerf_outputs/nerfacto-agent_w_hand_views_2024-07-06-23:44:47/nerfacto/2024-07-12_221120/config.yml"
    nerf_config = Path(nerf_config)
    field = load_nerf_field(nerf_config)

    lb = -RADIUS * np.ones(3)
    ub = RADIUS * np.ones(3)

    # obb_center = np.array([0.53, -0.12, 0.17])
    # obb_scale = np.array([0.1, 0.1, 0.2])
    # obb_rotation = np.array([0.00, 0.06, 0.0])

    obb_center = np.array([0, 0, 0.81])
    obb_scale = np.array([0.06, 0.06, 0.06])
    obb_rotation = np.array([0.00, 0.0, 0.0])

    obb_center = np.array([0.42, 0.17, 0.12])
    obb_scale = np.array([0.1, 0.1, 0.2])
    obb_rotation = np.array([0.00, 0.0, 0.0])

    lb = obb_center - obb_scale / 2
    ub = obb_center + obb_scale / 2
    print(f"lb: {lb}, ub: {ub}")

    if "glass-good" in str(nerf_config):
        lb = np.array([0.37, 0.12, 0.0])
        ub = np.array([0.47, 0.22, 0.24])
    elif "glass-holder" in str(nerf_config):
        # obb_center = np.array([0.52, -0.14, 0.17])
        # obb_scale = np.array([0.28, 0.24, 0.32])
        # obb_rotation = np.array([0.00, 0.06, 0.0])
        obb_center = np.array([0.53, -0.14, 0.17])
        obb_scale = np.array([0.27, 0.32, 0.33])
        obb_rotation = np.array([0.00, 0.0, -0.15])

        lb = obb_center - obb_scale / 2
        ub = obb_center + obb_scale / 2
        print(f"lb: {lb}, ub: {ub}")
    else:
        print("No bounds set for this object.")
        lb = -np.array([0.1, 0.1, 0.1])
        ub = np.array([0.0, 0.0, 0.0])

    print(Path(f"./{nerf_config.parent}/mesh.obj"))
    LEVEL = 1
    # lb[2] = 0
    # up[2] = 0.3
    print(lb, ub)
    # [EXAMPLE] seeing a plot of the mesh
    nerf_to_mesh(
        field,
        # level=10.0,  # VERY IMPORTANT TO ADJUST!
        level=LEVEL,
        npts=31,
        # lb=np.array([-0.1, -0.1, 0.8]),  # VERY IMPORTANT TO ADJUST!
        # ub=np.array([0.1, 0.1, 0.87]),  # VERY IMPORTANT TO ADJUST!
        # lb=np.array([-0.4, -0.4, 0.0]),  # VERY IMPORTANT TO ADJUST!
        # ub=np.array([0.4, 0.4, 0.4]),  # VERY IMPORTANT TO ADJUST!
        lb=lb,
        ub=ub,
        # save_path=None,
        save_path=Path(
            "demo_aug/models/assets/task_relevant/nerf_outputs/nerfacto-agent_w_hand_views_2024-07-06-23:44:47/nerfacto/2024-07-12_221120/meshes/watertight-mesh.obj"
        ),
        # save_path=Path(f"./{nerf_config.parent}.obj"),
    )

    # [EXAMPLE] saving the mesh
    nerf_to_mesh(
        field,
        # level=10.0,
        level=LEVEL,
        npts=31,
        # lb=np.array([-0.1, -0.1, 0.0]),
        # ub=np.array([0.1, 0.1, 0.3]),
        lb=lb,
        ub=ub,
        save_path=Path(f"./{nerf_config.stem}.obj"),
    )
