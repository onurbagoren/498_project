import pybullet as pyb
import pybullet_data

def load_environment(client_id):
    pyb.setAdditionalSearchPath(
        pybullet_data.getDataPath(), physicsClientId=client_id
    )

    # ground plane
    ground_id = pyb.loadURDF(
        "plane.urdf",
        [0, 0, 0],
        useFixedBase=True,
        physicsClientId=client_id,
    )

    # KUKA iiwa robot arm
    kuka_id = pyb.loadURDF(
        "kuka_iiwa/model.urdf",
        [0, 0, 0],
        useFixedBase=True,
        physicsClientId=client_id,
    )

    # some cubes for obstacles
    cube1_id = pyb.loadURDF(
        "cube.urdf",
        [1, 1, 0.5],
        useFixedBase=True,
        physicsClientId=client_id,
    )
    cube2_id = pyb.loadURDF(
        "cube.urdf",
        [-1, -1, 0.5],
        useFixedBase=True,
        physicsClientId=client_id,
    )
    cube3_id = pyb.loadURDF(
        "cube.urdf",
        [1, -1, 0.5],
        useFixedBase=True,
        physicsClientId=client_id,
    )

    # store body indices in a dict with more convenient key names
    bodies = {
        "robot": kuka_id,
        "ground": ground_id,
        "cube1": cube1_id,
        "cube2": cube2_id,
        "cube3": cube3_id,
    }
    return bodies


# start the main physics server and load the environment
gui_id = pyb.connect(pyb.GUI)
bodies = load_environment(gui_id)

col_id = pyb.connect(pyb.DIRECT)

# collision simulator has the same objects as the main one
collision_bodies = load_environment(col_id)

# NamedCollisionObjects contain the name of the body, and optionally
# the name of the link on the body to check for collisions
ground = NamedCollisionObject("ground")
cube1 = NamedCollisionObject("cube1")
cube2 = NamedCollisionObject("cube2")
cube3 = NamedCollisionObject("cube3")
link7 = NamedCollisionObject("robot", "lbr_iiwa_link_7")  # last link

# then we set up collision detection for desired pairs of objects
col_detector = CollisionDetector(
    col_id,  # client ID for collision physics server
    collision_bodies,  # bodies in the simulation
    # these are the pairs of objects to compute distances between
    [(link7, ground), (link7, cube1), (link7, cube2), (link7, cube3)],
)