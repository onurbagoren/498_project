import numpy as np
import sys
import os
from matplotlib import pyplot as plt

from utils.visualizers import *

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(
    __file__)), 'logs', 'data')

# Randomly select a file from the LOG_DIR
files = os.listdir(LOG_DIR)
files = [os.path.join(LOG_DIR, f) for f in files if f.endswith('.npz')]
file = np.random.choice(files)
data = np.load(file)
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