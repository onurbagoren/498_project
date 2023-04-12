import os
import sys

import numpy as np

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(
    __file__)), '..', 'logs', 'data')


def get_urdf_path():
    urdf_dir = os.path.join(os.path.dirname(os.path.abspath(
        __file__)), '..', 'assets', 'franka_panda', 'components')
    return urdf_dir


def random_urdfs(fingers=True):
    '''
    Function to get two URDFs. The first one will be used as the one that is "thrown around" and the second one will be the static one
    '''
    urdf_dir = get_urdf_path()
    urdf_files = os.listdir(urdf_dir)
    urdf_files = [os.path.join(urdf_dir, f)
                  for f in urdf_files if f.endswith('.urdf')]
    urdf_files = np.random.choice(urdf_files, size=2, replace=False)
    # Get the names of the URDFs, which is the text after the last/ and before the .urdf
    moving_name = urdf_files[0].split('/')[-1].split('.')[0]
    static_name = urdf_files[1].split('/')[-1].split('.')[0]
    if 'finger' in static_name and fingers:
        # Pick a new random URDF
        return random_urdfs()
    return urdf_files[0], urdf_files[1], moving_name, static_name


def write_npy(moving_name: str,
              static_name: str,
              simulation_time: np.array,
              moving_position: np.array,
              moving_orientation: np.array,
              moving_velocity: np.array,
              moving_angular_velocity: np.array,
              static_position: np.array,
              static_orientation: np.array,
              static_velocity: np.array,
              static_angular_velocity: np.array,
              contact_points_moving: np.array,
              contact_points_static: np.array,
              contact_normal: np.array,
              contact_normal_force: np.array,
              contact_times: np.array,
              log_dir=LOG_DIR
              ):
    '''
    Write the npy file between two URDFs
    '''
    # Check if LOGDIR exists
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Create the filename
    ii = 0
    filename = os.path.join(log_dir, f'{moving_name}_{static_name}_{ii}.npz')
    # Check if file exists
    while os.path.isfile(filename):
        ii += 1
        filename = os.path.join(
            log_dir, f'{moving_name}_{static_name}_{ii}.npz')
    # Write the file
    np.savez(file=filename[:-4],
             simulation_time=simulation_time,
             moving_position=moving_position,
             moving_orientation=moving_orientation,
             moving_velocity=moving_velocity,
             moving_angular_velocity=moving_angular_velocity,
             static_position=static_position,
             static_orientation=static_orientation,
             static_velocity=static_velocity,
             static_angular_velocity=static_angular_velocity,
             contact_points_moving=contact_points_moving,
             contact_points_static=contact_points_static,
             contact_normal=contact_normal,
             contact_normal_force=contact_normal_force,
             contact_times=contact_times,
             allow_pickle=True)


def main():
    moving_urdf, static_urdf, moving_name, static_name = random_urdfs()


if __name__ == '__main__':
    main()
