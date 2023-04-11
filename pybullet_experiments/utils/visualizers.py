import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


def plot_positions(urdf1_pos, urdf2_pos, collision_times):
    '''
    Plot the positions of the URDFs over time
    '''
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    axs[0].plot(urdf1_pos[:, 0], label=r'$URDF1_x$')
    axs[0].plot(urdf1_pos[:, 1], label=r'$URDF2_y$')
    axs[0].plot(urdf1_pos[:, 2], label=r'$URDF2_z$')
    axs[1].plot(urdf2_pos[:, 0], label=r'$URDF2_x$')
    axs[1].plot(urdf2_pos[:, 1], label=r'$URDF2_y$')
    axs[1].plot(urdf2_pos[:, 2], label=r'$URDF2_z$')
    
