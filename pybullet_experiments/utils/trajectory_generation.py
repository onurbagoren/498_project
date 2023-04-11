import numpy as np
from scipy.interpolate import interp1d
import scipy


def interpolate_3d_traj(duration: float,
                        num_points: int,
                        xlim: list = [-0.4, 0.4],
                        ylim: list = [-0.4, 0.4],
                        zlim: list = [0.4, 0.6]):
    '''
    Generate a trajectory for the object to follow in a 3D space

    Parameters
    ----------
    duration : float
        The duration of the trajectory in seconds
    num_points : int
        The number of points on the trajectory
    xlim : list, optional
        The x limits of the trajectory, by default [-0.4, 0.4]
    ylim : list, optional
        The y limits of the trajectory, by default [-0.4, 0.4]
    zlim : list, optional
        The z limits of the trajectory, by default [0.5, 0.5]

    Returns
    -------
    trajectory : np.ndarray
        The trajectory in the form of a numpy array
    interp_x : scipy.interpolate._interpolate.interp1d
        The x interpolation function
    interp_y : scipy.interpolate._interpolate.interp1d
        The y interpolation function
    interp_z : scipy.interpolate._interpolate.interp1d
        The z interpolation function
    '''
    t = np.linspace(0, duration, num_points)
    x = np.random.uniform(xlim[0], xlim[1], size=num_points)
    y = np.random.uniform(ylim[0], ylim[1], size=num_points)
    z = np.random.uniform(zlim[0], zlim[1], size=num_points)
    trajectory = np.vstack((t, x, y, z)).T

    # Interpolate the positions along the trajectory
    interp_x = interp1d(trajectory[:, 0], trajectory[:, 1], kind='cubic')
    interp_y = interp1d(trajectory[:, 0], trajectory[:, 2], kind='cubic')
    interp_z = interp1d(trajectory[:, 0], trajectory[:, 3], kind='cubic')

    return trajectory, interp_x, interp_y, interp_z


def generate_positions(t_start: float,
                       t_end: float,
                       time_step: float,
                       interp_x: scipy.interpolate._interpolate.interp1d,
                       interp_y: scipy.interpolate._interpolate.interp1d,
                       interp_z: scipy.interpolate._interpolate.interp1d):
    '''
    For a slice of time, generate the positions of the object

    Parameters
    ----------
    t_start : float
        The start time of the slice
    t_end : float
        The end time of the slice
    time_step : float
        The time step of the slice
    interp_x : scipy.interpolate._interpolate.interp1d
        The x interpolation function
    interp_y : scipy.interpolate._interpolate.interp1d
        The y interpolation function
    interp_z : scipy.interpolate._interpolate.interp1d
        The z interpolation function

    Returns
    -------
    positions : np.ndarray
        The positions of the object in the form of a numpy array
    '''
    t = np.arange(t_start, t_end, time_step)
    x = interp_x(t)
    y = interp_y(t)
    z = interp_z(t)
    positions = np.vstack((x, y, z)).T

    return positions
