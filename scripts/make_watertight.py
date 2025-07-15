try:
    import point_cloud_utils as pcu
except ImportError:
    # TODO(klin): use https://github.com/hjwdzh/Manifold instead
    # https://github.com/PRBonn/manifold_python for python version instead
    # moving on for now
    print(
        "Could not import point_cloud_utils. Please install pip install point-cloud-utils"
    )

v, f, c = pcu.load_mesh_vfc(
    "demo_aug/models/assets/task_relevant/mesh-outputs-welded/mesh.obj"
)

vw, fw = pcu.make_mesh_watertight(v, f)

pcu.save_mesh_vf(
    "demo_aug/models/assets/task_relevant/mesh-outputs-welded/watertight_mesh.obj",
    vw,
    fw,
)
