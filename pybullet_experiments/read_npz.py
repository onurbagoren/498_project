import numpy as np
import sys
import os
from matplotlib import pyplot as plt

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(
    __file__)), 'logs', 'linear_collision')

# Randomly select a file from the LOG_DIR
files = os.listdir(LOG_DIR)
files = [os.path.join(LOG_DIR, f) for f in files if f.endswith('.npz')]
file = np.random.choice(files)

# Load the file
data = np.load(file, allow_pickle=True)
collision_times = data['contact_times']
simulation_times = data['simulation_time']
plt.plot(simulation_times[1:], data['moving_velocity'][1:, 0], label='x')
plt.plot(simulation_times[1:], data['moving_velocity'][1:, 1], label='y')
plt.plot(simulation_times[1:], data['moving_velocity'][1:, 2], label='z')
# Vertical lines at the collision times
plt.axvline(x=collision_times[0], color='r')
plt.legend()
plt.show()

plt.plot(simulation_times, data['moving_position'][:, 0], label='x')
plt.plot(simulation_times, data['moving_position'][:, 1], label='y')
plt.plot(simulation_times, data['moving_position'][:, 2], label='z')
# Vertical lines at the collision times
plt.axvline(x=collision_times[0], color='r')
plt.legend()
plt.show()

plt.plot(simulation_times, data['static_position'][:, 0], label='x')
plt.plot(simulation_times, data['static_position'][:, 1], label='y')
plt.plot(simulation_times, data['static_position'][:, 2], label='z')
# Vertical lines at the collision times
plt.axvline(x=collision_times[0], color='r')
plt.legend()
plt.show()