import numpy as np
import sys
import os
from matplotlib import pyplot as plt

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(
    __file__)), 'logs', 'data')

# Randomly select a file from the LOG_DIR
files = os.listdir(LOG_DIR)
files = [os.path.join(LOG_DIR, f) for f in files if f.endswith('.npz')]
file = np.random.choice(files)

# Load the file
data = np.load(file, allow_pickle=True)
print(data.files)