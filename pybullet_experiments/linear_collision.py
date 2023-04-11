import pybullet as p
import pybullet_data
import time

from utils.trajectory_generation import *
from utils.file_operations import *
from utils.trajectory_generation import *
from tqdm import tqdm
def run_single_simulation(num_frames):

    # Set up PyBullet physics simulation
    physicsClient = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, 0)

    moving_urdf, static_urdf, moving_name, static_name = random_urdfs(fingers=False)

    # Load URDF objects
    planeId = p.loadURDF("plane.urdf")

    # Random start position and orientation
    startPos_moving = np.array([0, 0, 1])
    startPos_moving[0] = np.random.randn() * 0.15
    startOrientation_moving = p.getQuaternionFromEuler(np.random.randn(3))
    movingId = p.loadURDF(moving_urdf, startPos_moving, startOrientation_moving)

    # Random start position and orientation
    startPos_static = np.array([1, 0, 1])
    startPos_static[0] += np.random.randn() * 0.15 + 0.5
    startPos_static[2] = startPos_moving[2]
    startOrientation_static = p.getQuaternionFromEuler(np.random.randn(3))
    staticId = p.loadURDF(static_urdf, startPos_static, startOrientation_static)

    # Set up constant velocity and force magnitude
    velocity = np.random.randn() * 100 + 750

    # Get mass of object1
    object1Mass = p.getDynamicsInfo(movingId, -1)[0]
    forceMagnitude = object1Mass * velocity

    # Get initial position and orientation of object1
    pos1, _ = p.getBasePositionAndOrientation(movingId)

    # Get position and orientation of object2
    pos2, _ = p.getBasePositionAndOrientation(staticId)

    # Calculate direction and magnitude from object1 to object2
    direction = [pos2[0] - pos1[0], pos2[1] - pos1[1], pos2[2] - pos1[2]]
    distance = np.linalg.norm(direction)
    normalizedDirection = [direction[0]/distance,
                        direction[1]/distance, direction[2]/distance]
    # Apply force to object1 in the direction of object2
    force = [forceMagnitude * normalizedDirection[0], forceMagnitude *
            normalizedDirection[1], forceMagnitude * normalizedDirection[2]]
    p.applyExternalForce(movingId, -1, force, startPos_moving, p.WORLD_FRAME)

    # Run physics simulation at constant velocity
    ii = 0
    collided = False

    times = []
    moving_pos = []
    moving_ori = []
    moving_vel = []
    moving_angvel = []
    static_pos = []
    static_ori = []
    static_vel = []
    static_angvel = []
    contact_times = []
    contacts_list = []
    contact_times = []
    t = 0
    while True:
        p.getCameraImage(320, 200)
        p.stepSimulation()
        # Get simulation time
        pos1, ori1 = p.getBasePositionAndOrientation(movingId) # m, quaternion
        pos2, ori2 = p.getBasePositionAndOrientation(staticId)
        vel1, angvel1 = p.getBaseVelocity(movingId) # m/s, rad/s
        vel2, angvel2 = p.getBaseVelocity(staticId)
        t += p.getPhysicsEngineParameters()["fixedTimeStep"]
        times.append(t) # s
        moving_pos.append(pos1)
        moving_ori.append(ori1)
        moving_vel.append(vel1)
        moving_angvel.append(angvel1)
        static_pos.append(pos2)
        static_ori.append(ori2)
        static_vel.append(vel2)
        static_angvel.append(angvel2)
        # Check if objects have collided
        contacts = p.getContactPoints(movingId, staticId)
        if len(contacts) > 0:
            collided = True
            contact_times.append(t)
            contacts_list.append(contacts)
        if collided:
            ii += 1
            if ii == num_frames:
                break
        time.sleep(1.0/240.0)

    # Disconnect from PyBullet physics simulation
    LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(
        __file__)), 'logs', 'linear_collision')
    # Save data
    write_npy(
        moving_name=moving_name,
        static_name=static_name,
        simulation_time=np.array(times)[-num_frames*2:],
        moving_position=np.array(moving_pos)[-num_frames*2:],
        moving_orientation=np.array(moving_ori)[-num_frames*2:],
        moving_velocity=np.array(moving_vel)[-num_frames*2:],
        moving_angular_velocity=np.array(moving_angvel)[-num_frames*2:],
        static_position=np.array(static_pos)[-num_frames*2:],
        static_orientation=np.array(static_ori)[-num_frames*2:],
        static_velocity=np.array(static_vel)[-num_frames*2:],
        static_angular_velocity=np.array(static_angvel)[-num_frames*2:],
        contact_points=contacts_list,
        contact_times=np.array(contact_times)[-num_frames*2:],
        log_dir=LOG_DIR
    )
    p.disconnect()

def main():
    for i in tqdm(range(int(1e5))):
        run_single_simulation(25)


if __name__ == "__main__":
    main()
