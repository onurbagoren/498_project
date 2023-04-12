import pybullet as p
import pybullet_data
import numpy as np
import time
import os


physicsClient = p.connect(p.GUI)
p.setTimeStep(1/200)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

TABLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets','objects')

planeId = p.loadURDF("plane.urdf")
table = p.loadURDF(f"{TABLE_DIR}/table/table.urdf", [0, 0, 0], useFixedBase=True)

OBJECTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets','objects')

# Pick a random folder in the objects directory
cube = p.loadURDF(f"{OBJECTS_DIR}/cube_lightweight/cube.urdf", [0, 0, 2], useFixedBase=False)

while True:
    p.stepSimulation()

    pos, ori = p.getBasePositionAndOrientation(cube)
    vel, ang_vel = p.getBaseVelocity(cube)
    vel = np.linalg.norm(vel)
    if vel < 1e-6:
        # Apply a random impulse
        impulse = np.array([[1, 0, 0],[0, 1, 0], [0, 0, 0.0001]]) @ np.random.randn(3)
        direction = impulse / np.linalg.norm(impulse)
        force = 1e7 * direction
        p.applyExternalForce(cube, -1, impulse, pos, p.WORLD_FRAME)

    time.sleep(1./240.)