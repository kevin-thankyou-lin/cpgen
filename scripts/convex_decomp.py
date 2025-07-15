"""
Approximate Convex Decomposition for 3D Meshes with Collision-Aware Concavity and Tree Search

https://github.com/SarahWeiii/CoACD/blob/main/python/py_example.py
"""

import pathlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

try:
    import coacd
except ImportError:
    raise ImportError("Please install CoACD via pip install coacd")
import numpy as np
import trimesh
import tyro


@dataclass
class CoacdConfig:
    threshold: float = 0.25
    preprocess_resolution: int = 75


@dataclass
class Config:
    input_file: pathlib.Path = pathlib.Path(
        "~/autom/demo-aug/demo_aug/models/assets/task_relevant/mesh-outputs/mesh.obj"
    ).expanduser()
    output_dir: Optional[pathlib.Path] = None
    output_file_prefix: str = "mesh-convex-decomp"
    output_file_suffixes: List[str] = field(default_factory=lambda: [".obj", ".stl"])

    show_decomposed_meshes: bool = False
    coacd_cfg: CoacdConfig = CoacdConfig()

    def __post_init__(self):
        if self.output_dir is None:
            self.output_dir = self.input_file.parent


def main(cfg: Config) -> Dict[str, List[pathlib.Path]]:
    """
    Performs an Approximate Convex Decomposition for 3D Meshes with Collision-Aware Concavity and Tree Search

    Returns a Dict mapping suffix str to list of paths to the decomposed meshes.
    """
    decomposed_mesh_paths: Dict[str, List[pathlib.Path]] = defaultdict(list)

    mesh = trimesh.load(cfg.input_file, force="mesh")
    coacd_mesh = coacd.Mesh(mesh.vertices, mesh.faces)
    # low fidelty results mightn't work well for very fine-grained motion and collision free waypoint heuristic ...
    result: List = coacd.run_coacd(
        coacd_mesh,
        threshold=cfg.coacd_cfg.threshold,
        preprocess_resolution=cfg.coacd_cfg.preprocess_resolution,
    )

    mesh_parts = []
    for vs, fs in result:
        mesh_parts.append(trimesh.Trimesh(vs, fs))

    scene = trimesh.Scene()
    np.random.seed(0)
    for p in mesh_parts:
        p.visual.vertex_colors[:, :3] = (np.random.rand(3) * 255).astype(np.uint8)
        scene.add_geometry(p)

    # Iterate over each mesh in the result and
    for i, (vs, fs) in enumerate(result):
        mesh = trimesh.Trimesh(vertices=vs, faces=fs)
        for output_file_suffix in cfg.output_file_suffixes:
            export_path = cfg.output_dir / (
                cfg.output_file_prefix + str(i) + output_file_suffix
            )
            mesh.export(export_path)
            decomposed_mesh_paths[output_file_suffix].append(export_path)

    for output_file_suffix in cfg.output_file_suffixes:
        scene.export(cfg.output_dir / (cfg.output_file_prefix + output_file_suffix))
        print(
            f"Exported to {cfg.output_dir / (cfg.output_file_prefix + output_file_suffix)}"
        )

    if cfg.show_decomposed_meshes:
        scene.show()

    return decomposed_mesh_paths


if __name__ == "__main__":
    tyro.cli(main)
