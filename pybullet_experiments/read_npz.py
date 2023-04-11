import numpy as np
import sys
from matplotlib import pyplot as plt

data = np.load(f"{sys.path[0]}/data.npz", allow_pickle=True)
print(data.files)
link1_pos = data['link1_pos']
link1_ori = data['link1_ori']
link1_vel = data['link1_vel']
link1_angvel = data['link1_angvel']
link0_pos = data['link0_pos']
link0_ori = data['link0_ori']
link0_vel = data['link0_vel']
link0_angvel = data['link0_angvel']
times = data['times']
contacts = data['contact_points']
contact_times = data['contact_times']

# Plot the data
fig, axs = plt.subplots(2, 1, sharex=True)
axs[0].plot(times, link1_pos[:, 0], label='x')
axs[0].plot(times, link1_pos[:, 1], label='y')
axs[0].plot(times, link1_pos[:, 2], label='z')
axs[0].legend()
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('Position (m)')

# Plot a vertical line at the time of the first contact
axs[0].axvline(contact_times[0], color='r', linestyle='--')

axs[1].plot(times, link0_pos[:, 0], label='x')
axs[1].plot(times, link0_pos[:, 1], label='y')
axs[1].plot(times, link0_pos[:, 2], label='z')
axs[1].legend()
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Position (m)')

# Plot a vertical line at the time of the first contact
axs[1].axvline(contact_times[0], color='r', linestyle='--')

plt.show()
