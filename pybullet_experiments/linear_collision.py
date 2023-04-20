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
    p.setTimeStep(1/200)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, 0)

    moving_urdf, static_urdf, moving_name, static_name = random_urdfs(
        fingers=False)

    # Load URDF objects
    planeId = p.loadURDF("plane.urdf")

    # p.resetDebugVisualizerCamera(
    #     cameraDistance=3.0, cameraYaw=0, cameraPitch=-45, cameraTargetPosition=[0, 0, 0])

    # Random start position and orientation
    startPos_moving = np.array([0, 0, 1])
    startPos_moving[0] = np.random.randn() * 0.15
    startOrientation_moving = p.getQuaternionFromEuler(np.random.randn(3))
    movingId = p.loadURDF(moving_urdf, startPos_moving,
                          startOrientation_moving)

    # Random start position and orientation
    startPos_static = np.array([1, 0, 1])
    startPos_static[0] += np.random.randn() * 0.15 + 0.5
    startPos_static[2] = startPos_moving[2]
    startOrientation_static = p.getQuaternionFromEuler(np.random.randn(3))
    staticId = p.loadURDF(static_urdf, startPos_static,
                          startOrientation_static)

    # Set up constant velocity and force magnitude
    velocity = np.random.randn() * 10 + 200

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
    try:
        normalizedDirection = [direction[0]/distance,
                            direction[1]/distance, direction[2]/distance]
    except ZeroDivisionError:
        print("ZeroDivisionError")
        return
    # Apply force to object1 in the direction of object2
    force = [forceMagnitude * normalizedDirection[0], forceMagnitude *
             normalizedDirection[1], forceMagnitude * normalizedDirection[2]]
    p.applyExternalForce(movingId, -1, force, startPos_moving, p.WORLD_FRAME)

    # Run physics simulation at constant velocity
    ii = 0
    collided = False

    # Get the mass of the moving object
    mass = p.getDynamicsInfo(movingId, -1)[0]
    # Get the inertia of the moving object
    inertia = p.getDynamicsInfo(movingId, -1)[2]
    # Get the friction of the moving object
    friction = p.getDynamicsInfo(movingId, -1)[1]
    # Get the restitution of the moving object

    mass_static = p.getDynamicsInfo(staticId, -1)[0]
    inertia_static = p.getDynamicsInfo(staticId, -1)[2]
    friction_static = p.getDynamicsInfo(staticId, -1)[1]

    times = []
    moving_pos = []
    moving_ori = []
    moving_vel = []
    moving_angvel = []
    static_pos = []
    static_ori = []
    static_vel = []
    static_angvel = []
    contact_points_moving = []
    contact_points_static = []
    contact_normal = []
    contact_force = []
    contact_times = []
    lateral_friction_force1 = []
    lateral_friction_force2 = []
    lateral_friction_dir1 = []
    lateral_friction_dir2 = []
    t = 0
    while True:
        p.stepSimulation()
        # Get simulation time
        pos1, ori1 = p.getBasePositionAndOrientation(movingId)  # m, quaternion
        pos2, ori2 = p.getBasePositionAndOrientation(staticId)
        vel1, angvel1 = p.getBaseVelocity(movingId)  # m/s, rad/s
        vel2, angvel2 = p.getBaseVelocity(staticId)
        t += p.getPhysicsEngineParameters()["fixedTimeStep"]
        times.append(t)  # s
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
            if len(contacts) > 1:
                for ii, contact_pts in enumerate(contacts):
                    contact_times.append(t)
                    contact_points_moving.append(contacts[ii][5])
                    contact_points_static.append(contacts[ii][6])
                    contact_normal.append(contacts[ii][7])
                    contact_force.append(contacts[ii][9])
                    lateral_friction_force1.append(contacts[ii][10])
                    lateral_friction_force2.append(contacts[ii][12])
                    lateral_friction_dir1.append(contacts[ii][11])
                    lateral_friction_dir2.append(contacts[ii][13])
            else:
                contact_times.append(t)
                contact_points_moving.append(contacts[0][5])
                contact_points_static.append(contacts[0][6])
                contact_normal.append(contacts[0][7])
                contact_force.append(contacts[0][9])
                lateral_friction_force1.append(contacts[0][10])
                lateral_friction_force2.append(contacts[0][12])
                lateral_friction_dir1.append(contacts[0][11])
                lateral_friction_dir2.append(contacts[0][13])
        if collided:
            ii += 1
            if ii == num_frames:
                break
        # time.sleep(1./200.)
        if t > 5.0:
            return

    # Disconnect from PyBullet physics simulation
    LOG_DIR = os.path.join('/media/frog/DATA/Datasets/498_project', 'logs', 'linear_collision_ycb_extra_massinfo')
    # Save data
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
        contact_points_moving=contact_points_moving,
        contact_points_static=contact_points_static,
        contact_normal=contact_normal,
        contact_normal_force=contact_force,
        contact_times=np.array(contact_times),
        lateral_friction_force1=np.array(lateral_friction_force1),
        lateral_friction_force2=np.array(lateral_friction_force2),
        lateral_friction_dir1=np.array(lateral_friction_dir1),
        lateral_friction_dir2=np.array(lateral_friction_dir2),
        moving_mass=mass,
        moving_inertia=np.array(inertia),
        moving_friction=friction,
        static_mass=mass_static,
        static_inertia=np.array(inertia_static),
        static_friction=friction_static,
        log_dir=LOG_DIR
    )
    p.disconnect()

def main():
    for i in tqdm(range(int(2e5))):
        run_single_simulation(25)


if __name__ == "__main__":
    main()
