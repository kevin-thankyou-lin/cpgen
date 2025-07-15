import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    MeshRasterizer,
    PerspectiveCameras,
    RasterizationSettings,
    SoftPhongShader,
)
from torch import nn

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# obj_filename = "/scr/thankyou/autom/demo-aug/data/cow_mesh/cow.obj"
obj_filename = "point_cloud_test.obj"

mesh = load_objs_as_meshes([obj_filename], device=device)


obj_filename = "point_cloud_test.obj"
mesh1 = load_objs_as_meshes([obj_filename], device=device)

# Define the camera parameters
R = torch.eye(3).unsqueeze(0).to(device)
T = torch.zeros(1, 3).to(device)
T[0, 2] = 5

fl_x = 731.4708862304688
fl_y = 731.4708862304688
cx = 646.266357421875
cy = 355.9967956542969
w = 1280
h = 720

K = (
    torch.tensor(
        [[fl_x, 0, cx, 0], [0, fl_y, cy, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=torch.float32,
    )
    .unsqueeze(0)
    .to(device)
)

# Create the camera
cameras = PerspectiveCameras(
    image_size=((h, w),), in_ndc=False, K=K, R=R, T=T, device=device
)

# Define the rasterization settings
raster_settings = RasterizationSettings(
    image_size=(h, w),
    blur_radius=0.0,
    faces_per_pixel=1,
)


class MeshRendererWithDepth(nn.Module):
    def __init__(self, rasterizer, shader):
        super().__init__()
        self.rasterizer = rasterizer
        self.shader = shader

    def forward(self, meshes_world, **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(meshes_world, **kwargs)
        images = self.shader(fragments, meshes_world, **kwargs)
        return images, fragments.zbuf


# Create the renderer
renderer = MeshRendererWithDepth(
    rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
    shader=SoftPhongShader(cameras=cameras, device=device),
).to(device)


# Render the mesh
images, zbuf = renderer(mesh)


# Extract RGB image and depth
rgb = images[0, ..., :3].cpu().numpy()
depth = zbuf[0, ..., 0].cpu().numpy()

print(f"zbuf min: {zbuf.min().item()}, zbuf max: {zbuf.max().item()}")

# Convert depth to absolute distance
principal_point = (256, 256)
height, width = depth.shape
x, y = np.meshgrid(np.arange(width), np.arange(height))
x = (x - principal_point[0]) / fl_x
y = (y - principal_point[1]) / fl_y
absolute_distance = depth * np.sqrt(1 + x**2 + y**2)

print(f"a: {absolute_distance.min()}, b: {absolute_distance.max()}")

# Plotting
plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.imshow(rgb)
plt.title("RGB Render")
plt.subplot(132)
plt.imshow(depth)
plt.title("Depth")
plt.subplot(133)
plt.imshow(absolute_distance)
plt.title("Absolute Distance")
plt.savefig("output.png")
