import pybullet as p
import pybullet_data
import random
from scipy.interpolate import interp1d
import numpy as np
import time
import sys

from utils.trajectory_generation import *
from utils.file_operations import *
from utils.trajectory_generation import *

def run_single_simulation():
    # Set up the simulation
    physicsClient = p.connect(p.GUI) 
    p.setTimeStep(1/24000)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, 0)
    planeId = p.loadURDF("plane.urdf")
    # Set camera position
    p.resetDebugVisualizerCamera(
        cameraDistance=3.0, cameraYaw=0, cameraPitch=-45, cameraTargetPosition=[0, 0, 0])
    # Get camera data


    moving_urdf, static_urdf, moving_name, static_name = random_urdfs()

    # Load the URDF
    startPos = [0, 0, 1]
    startOrientation = p.getQuaternionFromEuler([random.uniform(
        0, 2 * np.pi), random.uniform(0, 2 * np.pi), random.uniform(0, 2 * np.pi)])
    movingId = p.loadURDF(moving_urdf, startPos, startOrientation)

    # Add a static object
    staticPos = [0, 0, 0.5]
    staticOrientation = p.getQuaternionFromEuler([random.uniform(
        0, 2 * np.pi), random.uniform(0, 2 * np.pi), random.uniform(0, 2 * np.pi)])
    staticId = p.loadURDF(static_urdf, staticPos, staticOrientation)

    # Define a nonlinear trajectory
    duration = 10.0  # seconds
    num_points = 10  # number of points on the trajectory

    trajectory, interp_x, interp_y, interp_z = interpolate_3d_traj(
        duration, num_points)


    # Define the simulation parameters
    timeStep = 1.0 / 24000.0

    prevPos = None
    prevTime = None
    times = []
    moving_pos = []
    moving_ori = []
    moving_vel = []
    moving_angvel = []
    static_pos = []
    static_vel = []
    static_ori = []
    static_angvel = []
    collision = False
    curr_pos = None
    contact_points = []
    contact_times = []
    for i in range(num_points - 1):
        p.getCameraImage(320, 200)

        positions = generate_positions(
            trajectory[i, 0],
            trajectory[i+1, 0],
            timeStep,
            interp_x,
            interp_y,
            interp_z
        )

        for ii, pos in enumerate(positions):
            p.stepSimulation(physicsClientId=physicsClient)
            _, ori = p.getBasePositionAndOrientation(movingId)
            newPos = [pos[0], pos[1], pos[2]]
            p.resetBasePositionAndOrientation(movingId, newPos, ori)

            # Check for collisions
            contacts = p.getContactPoints(movingId, staticId)
            movingPos, movingOrn = p.getBasePositionAndOrientation(movingId)
            movingLinVel, movingAngVel = p.getBaseVelocity(movingId)
            staticPos, staticOrn = p.getBasePositionAndOrientation(staticId)
            staticLinVel, staticAngVel = p.getBaseVelocity(staticId)

            # Compute the current time
            curr_time = trajectory[i, 0] + ii * timeStep
            times.append(curr_time)
            moving_pos.append(movingPos)
            moving_ori.append(movingOrn)
            moving_vel.append(movingLinVel)
            moving_angvel.append(movingAngVel)
            static_pos.append(staticPos)
            static_ori.append(staticOrn)
            static_vel.append(staticLinVel)
            static_angvel.append(staticAngVel)
            if len(contacts) > 0:
                contact_points.append(contacts)
                contact_times.append(curr_time)
                curr_pos = ii
                collision = True
            if curr_pos is not None:
                if ii == curr_pos + 1000:
                    break
        if collision:
            break

    write_npy(
        moving_name=moving_name,
        static_name=static_name,
        simulation_time=np.array(times),
        moving_position=np.array(moving_pos),
        moving_orientation=np.array(moving_ori),
        moving_velocity=np.array(moving_vel),
        moving_angular_velocity=np.array(moving_angvel),
        static_position=np.array(static_pos),
        static_orientation=np.array(static_ori),
        static_velocity=np.array(static_vel),
        static_angular_velocity=np.array(static_angvel),
        contact_points=np.array(contact_points),
        contact_times=np.array(contact_times)
    )

    # Disconnect from the simulation
    p.disconnect()


def main():
    for i in range(100):
        run_single_simulation()

if __name__ == "__main__":
    main()