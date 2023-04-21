import pybullet as p
import numpy as np
import pybullet_data
import matplotlib.pyplot as plt
import pyvista as pv
from scipy.spatial import ConvexHull
import trimesh

mesh = trimesh.load_mesh('/home/frog/courses/498_project/HW3/assets/objects/medium_clamp/textured.obj')

point_cloud = mesh.sample(int(1e5))

mean_z = np.mean(point_cloud[:, 2])

# Get all the points that are close to the mean z value
z_value = mean_z
z_tolerance = 0.005
z_min = z_value - z_tolerance
z_max = z_value + z_tolerance
z_mask = np.logical_and(z_min <= point_cloud[:, 2], point_cloud[:, 2] <= z_max)
z_points = point_cloud[z_mask]

plt.scatter(z_points[:, 0], z_points[:, 1])
plt.show()

# # Find boundary points at fixed z-value
# boundary_points = []
# for i in range(len(hull_points)):
#     i_next = (i + 1) % len(hull_points)
#     p1 = hull_points[i]
#     p2 = hull_points[i_next]
#     if p1[2] <= z_value <= p2[2] or p2[2] <= z_value <= p1[2]:
#         t = (z_value - p1[2]) / (p2[2] - p1[2])
#         boundary_points.append((1 - t) * p1[:2] + t * p2[:2])

# plt.scatter([p[0] for p in boundary_points], [p[1] for p in boundary_points])
# plt.show()
# Print boundary points

# Perform surface reconstruction
# point_cloud_pv = pv.PolyData(indices)
# surface = point_cloud_pv.delaunay_2d(alpha=2.0)

# # Extract boundary vertices
# boundary = surface.extract_feature_edges().points

# z_values = boundary[:, 2]
# middle_z = np.median(z_values)
# projection = np.copy(boundary)
# projection[:, 2] = middle_z

# # Draw outline of projected points
# x_values = projection[:, 0]
# y_values = projection[:, 1]
# fig, ax = plt.subplots()
# ax.plot(x_values, y_values, linestyle='-', marker='o', color='b', linewidth=2)
# ax.set_aspect('equal', 'box')
# plt.show()

# # ax.scatter(indices[:, 0], indices[:, 1], indices[:, 2], c='r', marker='o')

# middle_z = np.mean(indices[:, 2])
# projection = indices.copy()
# projection[:, 2] = middle_z

# ax.scatter(projection[:, 0], projection[:, 1], projection[:, 2], c='b', marker='o')
# plt.show()

# x_values = projection[:, 0]
# y_values = projection[:, 1]
# fig, ax = plt.subplots()
# ax.plot(x_values, y_values, linestyle='-', marker='o', color='b', linewidth=2)
# ax.set_aspect('equal', 'box')
# plt.show()
