import numpy as np
import sys
import os
from matplotlib import pyplot as plt

from utils.visualizers import *

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(
    __file__)), 'logs', 'linear_collision')
# Randomly select a file from the LOG_DIR
files = os.listdir(LOG_DIR)
files = [os.path.join(LOG_DIR, f) for f in files if f.endswith('.npz')]
file = np.random.choice(files)
data = np.load(file, allow_pickle=True)
print(data.files)

plot_all(
    data['moving_position'],
    data['moving_orientation'],
    data['moving_velocity'],
    data['moving_angular_velocity'],
    data['static_position'],
    data['static_orientation'],
    data['static_velocity'],
    data['static_angular_velocity'],
    data['simulation_time'],
    data['contact_times']
)

# contact_pts_moving = data['contact_points_moving']
# contact_pts_static = data['contact_points_static']
# contact_normal = data['contact_normal']
# contact_normal_force = data['contact_normal_force']
# contact_times = data['contact_times']

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# for ii in range(contact_times.shape[0]):
#     # 3D figure
#     if ii == 0:
#         ax.scatter(contact_pts_moving[ii, 0], contact_pts_moving[ii, 1],
#                    contact_pts_moving[ii, 2], color='red', label='Moving Object')
#         ax.scatter(contact_pts_static[ii, 0], contact_pts_static[ii, 1],
#                    contact_pts_static[ii, 2], color='blue', label='Static Object')
#     else:
#         ax.scatter(contact_pts_moving[ii, 0], contact_pts_moving[ii, 1],
#                    contact_pts_moving[ii, 2], color='red')
#         ax.scatter(contact_pts_static[ii, 0], contact_pts_static[ii, 1],
#                    contact_pts_static[ii, 2], color='blue')
#     # Draw arrow with length of the normal force
#     ax.quiver(contact_pts_moving[ii, 0],
#               contact_pts_moving[ii, 1],
#               contact_pts_moving[ii, 2],
#               contact_normal[ii, 0] / 2,
#               contact_normal[ii, 1] / 2,
#               contact_normal[ii, 2] / 2,
#               color='green')
# ax.legend()
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# # Set lims
# ax.set_xlim([0.5, 1.5])
# ax.set_ylim([-0.5, 0.5])
# ax.set_zlim([0.5, 1.5])
# fig.suptitle('Contact Points and Normals')
# plt.show()
