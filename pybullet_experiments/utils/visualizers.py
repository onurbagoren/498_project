import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


def plot_positions(urdf1_pos, urdf2_pos, simulation_time, collision_times):
    '''
    Plot the positions of the URDFs over time
    '''
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    axs[0].plot(simulation_time, urdf1_pos[:, 0], label=r'$URDF1_x$')
    axs[0].plot(simulation_time, urdf1_pos[:, 1], label=r'$URDF2_y$')
    axs[0].plot(simulation_time, urdf1_pos[:, 2], label=r'$URDF2_z$')
    axs[1].plot(simulation_time, urdf2_pos[:, 0], label=r'$URDF2_x$')
    axs[1].plot(simulation_time, urdf2_pos[:, 1], label=r'$URDF2_y$')
    axs[1].plot(simulation_time, urdf2_pos[:, 2], label=r'$URDF2_z$')
    axs[0].axvline(x=collision_times[0], color='r')
    axs[1].axvline(x=collision_times[0], color='r')

    axs[0].set_ylabel('Position (m)')
    axs[1].set_ylabel('Position (m)')
    axs[1].set_xlabel('Time (s)')
    axs[0].legend()
    axs[1].legend()
    plt.show()


def plot_velocities(urdf1_vel, urdf2_vel, simulation_time, collision_times):
    '''
    Plot the velocities of the URDFs over time
    '''
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    axs[0].plot(simulation_time, urdf1_vel[:, 0], label=r'$URDF1_x$')
    axs[0].plot(simulation_time, urdf1_vel[:, 1], label=r'$URDF2_y$')
    axs[0].plot(simulation_time, urdf1_vel[:, 2], label=r'$URDF2_z$')
    axs[1].plot(simulation_time, urdf2_vel[:, 0], label=r'$URDF2_x$')
    axs[1].plot(simulation_time, urdf2_vel[:, 1], label=r'$URDF2_y$')
    axs[1].plot(simulation_time, urdf2_vel[:, 2], label=r'$URDF2_z$')
    axs[0].axvline(x=collision_times[0], color='r')
    axs[1].axvline(x=collision_times[0], color='r')

    axs[0].set_ylabel('Velocity (m/s)')
    axs[1].set_ylabel('Velocity (m/s)')
    axs[1].set_xlabel('Time (s)')
    axs[0].legend()
    axs[1].legend()
    plt.show()


def plot_all(urdf1_pos, urdf1_ori, urdf1_vel, urdf1_angvel,
             urdf2_pos, urdf2_ori, urdf2_vel, urdf2_angvel, sim_time, collision_time):
    '''
    Plot all the data
    '''
    fig, axs = plt.subplots(4, 2, figsize=(10, 10))
    axs[0, 0].plot(sim_time, urdf1_pos[:, 0], label=r'$URDF1_x$')
    axs[0, 0].plot(sim_time, urdf1_pos[:, 1], label=r'$URDF1_y$')
    axs[0, 0].plot(sim_time, urdf1_pos[:, 2], label=r'$URDF1_z$')
    axs[0, 1].plot(sim_time, urdf2_pos[:, 0], label=r'$URDF2_x$')
    axs[0, 1].plot(sim_time, urdf2_pos[:, 1], label=r'$URDF2_y$')
    axs[0, 1].plot(sim_time, urdf2_pos[:, 2], label=r'$URDF2_z$')
    axs[0, 0].axvline(x=collision_time[0], color='r')
    axs[0, 1].axvline(x=collision_time[0], color='r')
    axs[0,0].set_title('Moving URDF')
    axs[0,1].set_title('Static URDF')


    axs[1, 0].plot(sim_time, urdf1_vel[:, 0], label=r'$URDF1_x$')
    axs[1, 0].plot(sim_time, urdf1_vel[:, 1], label=r'$URDF1_y$')
    axs[1, 0].plot(sim_time, urdf1_vel[:, 2], label=r'$URDF1_z$')
    axs[1, 1].plot(sim_time, urdf2_vel[:, 0], label=r'$URDF2_x$')
    axs[1, 1].plot(sim_time, urdf2_vel[:, 1], label=r'$URDF2_y$')
    axs[1, 1].plot(sim_time, urdf2_vel[:, 2], label=r'$URDF2_z$')
    axs[1, 0].axvline(x=collision_time[0], color='r')
    axs[1, 1].axvline(x=collision_time[0], color='r')

    axs[2, 0].plot(sim_time, urdf1_angvel[:, 0], label=r'$URDF1_x$')
    axs[2, 0].plot(sim_time, urdf1_angvel[:, 1], label=r'$URDF1_y$')
    axs[2, 0].plot(sim_time, urdf1_angvel[:, 2], label=r'$URDF1_z$')
    axs[2, 1].plot(sim_time, urdf2_angvel[:, 0], label=r'$URDF2_x$')
    axs[2, 1].plot(sim_time, urdf2_angvel[:, 1], label=r'$URDF2_y$')
    axs[2, 1].plot(sim_time, urdf2_angvel[:, 2], label=r'$URDF2_z$')
    axs[2, 0].axvline(x=collision_time[0], color='r')
    axs[2, 1].axvline(x=collision_time[0], color='r')

    axs[3, 0].plot(sim_time, urdf1_ori[:, 0], label=r'$URDF1_x$')
    axs[3, 0].plot(sim_time, urdf1_ori[:, 1], label=r'$URDF1_y$')
    axs[3, 0].plot(sim_time, urdf1_ori[:, 2], label=r'$URDF1_z$')
    axs[3, 0].plot(sim_time, urdf1_ori[:, 3], label=r'$URDF1_w$')
    axs[3, 1].plot(sim_time, urdf2_ori[:, 0], label=r'$URDF2_x$')
    axs[3, 1].plot(sim_time, urdf2_ori[:, 1], label=r'$URDF2_y$')
    axs[3, 1].plot(sim_time, urdf2_ori[:, 2], label=r'$URDF2_z$')
    axs[3, 1].plot(sim_time, urdf2_ori[:, 3], label=r'$URDF2_w$')
    axs[3, 0].axvline(x=collision_time[0], color='r')
    axs[3, 1].axvline(x=collision_time[0], color='r')

    axs[0, 0].set_ylabel('Position (m)')
    axs[1, 0].set_ylabel('Velocity (m/s)')
    axs[2, 0].set_ylabel('Angular Velocity (rad/s)')
    axs[3, 0].set_ylabel('Orientation (quaternion)')
    axs[3, 0].set_xlabel('Time (s)')
    axs[3, 1].set_xlabel('Time (s)')
    axs[0, 0].legend()
    axs[0, 1].legend()
    axs[1, 0].legend()
    axs[1, 1].legend()
    axs[2, 0].legend()
    axs[2, 1].legend()
    axs[3, 0].legend()
    axs[3, 1].legend()
    plt.show()


