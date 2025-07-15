import numpy as np
import pytorch_kinematics as pk
import torch

path = "demo_aug/models/assets/robots/panda/robot.xml"

# KL: needed to update robot.xml's file paths and add joint type = "hinge" (== "revolute")
chain = pk.build_chain_from_mjcf(open(path).read())

qpos = [
    0.11059413,
    0.93231301,
    -0.07452845,
    -1.9616505,
    -0.1056378,
    3.02557832,
    1.25656168,
]
ret = chain.forward_kinematics(qpos, end_only=False)
print(ret.keys())
tg = ret["link7"]
m = tg.get_matrix()[0]
base_offset = torch.FloatTensor([-0.56, 0.0, 0.912])
m[:3, 3] += base_offset
print(f"matrix: {m}")
print(f"pos: {m[:3, 3]}")

R = m[:3, :3]
T = m[:3, 3]

R_inv = R.T
T_inv = -R.T @ T
print(f"R_inv: {R_inv}")
print(f"T_inv: {T_inv}")

# convert R_inv to quaternion
q_wxyz = pk.matrix_to_quaternion(R_inv)
print(f"q_wxyz: {q_wxyz}")
# noramlize
q_wxyz /= np.linalg.norm(q_wxyz)
print(f"q_wxyz: {q_wxyz}")

# inverse of this transform is
# R^-1 * T

# sdf = pv.RobotSDF(chain, path_prefix=path)

# # visualize
# query_range = np.array([
#     [-0.15, 0.2],
#     [0, 0],
#     [-0.1, 0.2],
# ])

# pv.draw_sdf_slice(sdf, query_range)
