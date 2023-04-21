import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from panda_pushing_env import TARGET_POSE_FREE, TARGET_POSE_OBSTACLES, OBSTACLE_HALFDIMS, OBSTACLE_CENTRE, BOX_SIZE
import matplotlib.pyplot as plt
TARGET_POSE_FREE_TENSOR = torch.as_tensor(
    TARGET_POSE_FREE, dtype=torch.float32)
TARGET_POSE_OBSTACLES_TENSOR = torch.as_tensor(
    TARGET_POSE_OBSTACLES, dtype=torch.float32)
OBSTACLE_CENTRE_TENSOR = torch.as_tensor(
    OBSTACLE_CENTRE, dtype=torch.float32)[:2]
OBSTACLE_HALFDIMS_TENSOR = torch.as_tensor(
    OBSTACLE_HALFDIMS, dtype=torch.float32)[:2]


def collect_data_random(env, nerf_manager, num_trajectories=1000, trajectory_length=10):
    """
    Collect data from the provided environment using uniformly random exploration.
    :param env: Gym Environment instance.
    :param num_trajectories: <int> number of data to be collected.
    :param trajectory_length: <int> number of state transitions to be collected
    :return: collected data: List of dictionaries containing the state-action trajectories.
    Each trajectory dictionary should have the following structure:
        {'states': states,
        'actions': actions}
    where
        * states is a numpy array of shape (trajectory_length+1, state_size) containing the states [x_0, ...., x_T]
        * actions is a numpy array of shape (trajectory_length, actions_size) containing the actions [u_0, ...., u_{T-1}]
    Each trajectory is:
        x_0 -> u_0 -> x_1 -> u_1 -> .... -> x_{T-1} -> u_{T_1} -> x_{T}
        where x_0 is the state after resetting the environment with env.reset()
    All data elements must be encoded as np.float32.
    """
    collected_data = None
    # --- Your code here
    collected_data = []
    state_size = 3
    action_size = 3
    states = np.zeros((num_trajectories, trajectory_length +
                      1, state_size), dtype=np.float32)
    actions = np.zeros((num_trajectories, trajectory_length,
                       action_size), dtype=np.float32)
    
    deltas_list = []

    for traj_idx in tqdm(range(num_trajectories)):
        states[traj_idx, 0, :] = env.reset()
        for step_idx in tqdm(range(trajectory_length), leave=False):
            tf_homogenous = nerf_manager.get_object_tf(env)
            action_i = env.action_space.sample()
            action_rand_xy, deltas, phi = nerf_manager.get_rand_action_xy_with_transform(tf_homogenous, env.lower_z * 2)
            deltas_list.append(deltas.cpu().numpy())

            action_i[0] = action_rand_xy[0]
            action_i[1] = action_rand_xy[1]

            action = np.zeros(3)
            action[0] = phi
            action[1] = action_i[2]
            action[2] = action_i[3]
            next_state, _, _, _ = env.step(action_i)
            states[traj_idx, step_idx+1, :] = next_state.astype(np.float32)
            actions[traj_idx, step_idx, :] = action.astype(np.float32)
        traj_data = {
            'states': states[traj_idx],
            'actions': actions[traj_idx]
        }
        collected_data.append(traj_data)
    # cat the deltas and average
    deltas_list = np.array(deltas_list)
    deltas_list = np.mean(deltas_list, axis=0)
    # ---
    return collected_data, deltas_list


def process_data_single_step(collected_data, batch_size=500):
    """
    Process the collected data and returns a DataLoader for train and one for validation.
    The data provided is a list of trajectories (like collect_data_random output).
    Each DataLoader must load dictionary as {'state': x_t,
     'action': u_t,
     'next_state': x_{t+1},
    }
    where:
     x_t: torch.float32 tensor of shape (batch_size, state_size)
     u_t: torch.float32 tensor of shape (batch_size, action_size)
     x_{t+1}: torch.float32 tensor of shape (batch_size, state_size)

    The data should be split in a 80-20 training-validation split.
    :param collected_data:
    :param batch_size: <int> size of the loaded batch.
    :return:

    Hints:
     - Pytorch provides data tools for you such as Dataset and DataLoader and random_split
     - You should implement SingleStepDynamicsDataset below.
        This class extends pytorch Dataset class to have a custom data format.
    """
    train_loader = None
    val_loader = None
    # --- Your code here

    dataset = SingleStepDynamicsDataset(collected_data)

    split_dataset = random_split(
        dataset, [int(len(dataset)*0.8), int(len(dataset)*0.2)])
    train_loader = DataLoader(
        split_dataset[0], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(
        split_dataset[1], batch_size=batch_size, shuffle=True)

    # ---
    return train_loader, val_loader


def process_data_multiple_step(collected_data, batch_size=500, num_steps=4):
    """
    Process the collected data and returns a DataLoader for train and one for validation.
    The data provided is a list of trajectories (like collect_data_random output).
    Each DataLoader must load dictionary as
    {'state': x_t,
     'action': u_t, ..., u_{t+num_steps-1},
     'next_state': x_{t+1}, ... , x_{t+num_steps}
    }
    where:
     state: torch.float32 tensor of shape (batch_size, state_size)
     next_state: torch.float32 tensor of shape (batch_size, num_steps, action_size)
     action: torch.float32 tensor of shape (batch_size, num_steps, state_size)

    Each DataLoader must load dictionary dat
    The data should be split in a 80-20 training-validation split.
    :param collected_data:
    :param batch_size: <int> size of the loaded batch.
    :param num_steps: <int> number of steps to load the multistep data.
    :return:

    Hints:
     - Pytorch provides data tools for you such as Dataset and DataLoader and random_split
     - You should implement MultiStepDynamicsDataset below.
        This class extends pytorch Dataset class to have a custom data format.
    """
    train_loader = None
    val_loader = None
    # --- Your code here

    dataset = MultiStepDynamicsDataset(
        collected_data=collected_data,
        num_steps=num_steps
    )

    split_dataset = random_split(
        dataset, [int(len(dataset)*0.8), int(len(dataset)*0.2)])
    train_loader = DataLoader(
        split_dataset[0], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(
        split_dataset[1], batch_size=batch_size, shuffle=True)

    # ---
    return train_loader, val_loader


class SingleStepDynamicsDataset(Dataset):
    """
    Each data sample is a dictionary containing (x_t, u_t, x_{t+1}) in the form:
    {'state': x_t,
     'action': u_t,
     'next_state': x_{t+1},
    }
    where:
     x_t: torch.float32 tensor of shape (state_size,)
     u_t: torch.float32 tensor of shape (action_size,)
     x_{t+1}: torch.float32 tensor of shape (state_size,)
    """

    def __init__(self, collected_data):
        self.data = collected_data
        self.trajectory_length = self.data[0]['actions'].shape[0]

    def __len__(self):
        return len(self.data) * self.trajectory_length

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def __getitem__(self, item):
        """
        Return the data sample corresponding to the index <item>.
        :param item: <int> index of the data sample to produce.
            It can take any value in range 0 to self.__len__().
        :return: data sample corresponding to encoded as a dictionary with keys (state, action, next_state).
        The class description has more details about the format of this data sample.
        """
        sample = {
            'state': None,
            'action': None,
            'next_state': None,
        }
        # --- Your code here

        sample = {}

        trajectory_idx = item // self.trajectory_length
        step_idx = item % self.trajectory_length

        sample['state'] = torch.from_numpy(
            self.data[trajectory_idx]['states'][step_idx]).to(torch.float32)
        sample['action'] = torch.from_numpy(
            self.data[trajectory_idx]['actions'][step_idx]).to(torch.float32)
        sample['next_state'] = torch.from_numpy(
            self.data[trajectory_idx]['states'][step_idx+1]).to(torch.float32)

        # ---
        return sample


class MultiStepDynamicsDataset(Dataset):
    """
    Dataset containing multi-step dynamics data.

    Each data sample is a dictionary containing (state, action, next_state) in the form:
    {'state': x_t, -- initial state of the multipstep torch.float32 tensor of shape (state_size,)
     'action': [u_t,..., u_{t+num_steps-1}] -- actions applied in the muli-step.
                torch.float32 tensor of shape (num_steps, action_size)
     'next_state': [x_{t+1},..., x_{t+num_steps} ] -- next multiple steps for the num_steps next steps.
                torch.float32 tensor of shape (num_steps, state_size)
    }
    """

    def __init__(self, collected_data, num_steps=4):
        self.data = collected_data
        self.trajectory_length = self.data[0]['actions'].shape[0] - \
            num_steps + 1
        self.num_steps = num_steps

    def __len__(self):
        return len(self.data) * (self.trajectory_length)

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def __getitem__(self, item):
        """
        Return the data sample corresponding to the index <item>.
        :param item: <int> index of the data sample to produce.
            It can take any value in range 0 to self.__len__().
        :return: data sample corresponding to encoded as a dictionary with keys (state, action, next_state).
        The class description has more details about the format of this data sample.
        """
        # sample = {
        #     'state': None,
        #     'action': None,
        #     'next_state': None
        # }
        # --- Your code here
        sample = {}

        trajectory_idx = item // self.trajectory_length
        step_idx = item % self.trajectory_length

        sample['state'] = torch.from_numpy(
            self.data[trajectory_idx]['states'][step_idx]).to(torch.float32)
        actions = []
        next_states = []
        for ii in range(self.num_steps):
            actions.append(self.data[trajectory_idx]['actions'][step_idx + ii])
            next_states.append(
                self.data[trajectory_idx]['states'][step_idx + ii + 1])

        sample['action'] = torch.Tensor(np.array(actions)).to(torch.float32)
        sample['next_state'] = torch.Tensor(
            np.array(next_states)).to(torch.float32)
        # ---
        return sample


class SE2PoseLoss(nn.Module):
    """
    Compute the SE2 pose loss based on the object dimensions (block_width, block_length).
    Need to take into consideration the different dimensions of pose and orientation to aggregate them.

    Given a SE(2) pose [x, y, theta], the pose loss can be computed as:
        se2_pose_loss = MSE(x_hat, x) + MSE(y_hat, y) + \
                            rg * MSE(theta_hat, theta)
    where rg is the radious of gyration of the object.
    For a planar rectangular object of width w and length l, the radius of gyration is defined as:
        rg = ((l^2 + w^2)/12)^{1/2}

    """

    def __init__(self, block_width, block_length):
        super().__init__()
        self.w = block_width
        self.l = block_length

    def forward(self, pose_pred, pose_target):
        se2_pose_loss = None
        # --- Your code here
        r_g = np.sqrt((self.l**2 + self.w**2)/12)
        x_mse = nn.functional.mse_loss(pose_pred[:, 0], pose_target[:, 0])
        y_mse = nn.functional.mse_loss(pose_pred[:, 1], pose_target[:, 1])
        theta_mse = nn.functional.mse_loss(
            pose_pred[:, 2], pose_target[:, 2]) * r_g
        se2_pose_loss = x_mse + y_mse + theta_mse
        # ---
        return se2_pose_loss


class SingleStepLoss(nn.Module):

    def __init__(self, loss_fn):
        super().__init__()
        self.loss = loss_fn

    def forward(self, model, state, action, target_state):
        """
        Compute the single step loss resultant of querying model with (state, action) and comparing the predictions with target_state.
        """
        single_step_loss = None
        # --- Your code here

        out = model(state, action)
        single_step_loss = self.loss(out, target_state)

        # ---
        return single_step_loss


class MultiStepLoss(nn.Module):

    def __init__(self, loss_fn, discount=0.99):
        super().__init__()
        self.loss = loss_fn
        self.discount = discount

    def forward(self, model, state, actions, target_states):
        """
        Compute the multi-step loss resultant of multi-querying the model from (state, action) and comparing the predictions with targets.
        """
        multi_step_loss = 0
        # --- Your code here
        prev_state = state
        num_steps = actions.shape[1]
        for ii in range(num_steps):
            action = actions[:, ii]
            next_state = target_states[:, ii]
            pred_state = model(prev_state, action)
            multi_step_loss += (self.discount ** ii) * \
                                self.loss(pred_state, next_state)
            prev_state = pred_state
        # multi_step_loss /= num_steps
        # ---
        return multi_step_loss


class AbsoluteDynamicsModel(nn.Module):
    """
    Model the absolute dynamics x_{t+1} = f(x_{t},a_{t})
    """

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        # --- Your code here

        self.model = nn.Sequential(
            nn.Linear(self.state_dim + self.action_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, self.state_dim)
        )

        # ---

    def forward(self, state, action):
        """
        Compute next_state resultant of applying the provided action to provided state
        :param state: torch tensor of shape (..., state_dim)
        :param action: torch tensor of shape (..., action_dim)
        :return: next_state: torch tensor of shape (..., state_dim)
        """
        next_state = None
        # --- Your code here

        next_state = self.model(torch.cat((state, action), dim=-1))

        # ---
        return next_state


class ResidualDynamicsModel(nn.Module):
    """
    Model the residual dynamics s_{t+1} = s_{t} + f(s_{t}, u_{t})

    Observation: The network only needs to predict the state difference as a function of the state and action.
    """

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        # --- Your code here

        self.model = nn.Sequential(
            nn.Linear(self.state_dim + self.action_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, self.state_dim)
        )

        # ---

    def forward(self, state, action):
        """
        Compute next_state resultant of applying the provided action to provided state
        :param state: torch tensor of shape (..., state_dim)
        :param action: torch tensor of shape (..., action_dim)
        :return: next_state: torch tensor of shape (..., state_dim)
        """
        next_state = None
        # --- Your code here
        next_state = state + self.model(torch.concat([state, action], dim=-1))
        # ---
        return next_state


def free_pushing_cost_function(state, action):
    """
    Compute the state cost for MPPI on a setup without obstacles.
    :param state: torch tensor of shape (B, state_size)
    :param action: torch tensor of shape (B, state_size)
    :return: cost: torch tensor of shape (B,) containing the costs for each of the provided states
    """
    target_pose = TARGET_POSE_FREE_TENSOR  # torch tensor of shape (3,) containing (pose_x, pose_y, pose_theta)
    cost = None
    # --- Your code here

    Q = torch.eye(3)
    Q[-1, -1] = 0.1

    cost = torch.sum((state - target_pose) @ Q @
                     (state - target_pose).T, dim=1)

    # ---
    return cost


def collision_detection(state):
    """
    Checks if the state is in collision with the obstacle.
    The obstacle geometry is known and provided in obstacle_centre and obstacle_halfdims.
    :param state: torch tensor of shape (B, state_size)
    :return: in_collision: torch tensor of shape (B,) containing 1 if the state is in collision and 0 if not.
    """
    obstacle_centre = OBSTACLE_CENTRE_TENSOR  # torch tensor of shape (2,) consisting of obstacle centre (x, y)
    # torch tensor of shape (2,) consisting of (w_obs, l_obs)
    obstacle_dims = 2 * OBSTACLE_HALFDIMS_TENSOR
    box_size = BOX_SIZE  # scalar for parameter w
    in_collision = None
    # --- Your code here

    num_objects = state.shape[0]
    in_collision = torch.zeros(num_objects)
    obstacle_state = torch.cat((obstacle_centre, torch.zeros((1)))).unsqueeze(0)
    starting_corners = find_starting_corners((box_size, box_size), state)
    obstacle_corners = find_starting_corners(obstacle_dims, obstacle_state)
    in_collision = collides(starting_corners, obstacle_corners)
    return in_collision

    obstacle = Rectangle(obstacle_centre, obstacle_dims, torch.zeros((1)))
    for ii in range(num_objects):
        obj = Rectangle(state[ii, :2], torch.tensor([box_size, box_size]), state[ii, 2])
        rect_comp = RectangleCompare(obstacle, obj)
        in_collision[ii] = rect_comp.collides()

    # ---
    return in_collision


def obstacle_avoidance_pushing_cost_function(state, action):
    """
    Compute the state cost for MPPI on a setup with obstacles.
    :param state: torch tensor of shape (B, state_size)
    :param action: torch tensor of shape (B, state_size)
    :return: cost: torch tensor of shape (B,) containing the costs for each of the provided states
    """
    target_pose = TARGET_POSE_OBSTACLES_TENSOR  # torch tensor of shape (3,) containing (pose_x, pose_y, pose_theta)
    cost = None
    # --- Your code here

    Q = torch.eye(3)
    Q[-1, -1] = 0.1

    cost = torch.sum((state - target_pose) @ Q @ (state -
                     target_pose).T, dim=1) + 100 * collision_detection(state)

    # ---
    return cost


class PushingController(object):
    """
    MPPI-based controller
    Since you implemented MPPI on HW2, here we will give you the MPPI for you.
    You will just need to implement the dynamics and tune the hyperparameters and cost functions.
    """

    def __init__(self, env, model, cost_function, num_samples=100, horizon=10):
        self.env = env
        self.model = model
        self.target_state = None
        # MPPI Hyperparameters:
        # --- You may need to tune them
        state_dim = env.observation_space.shape[0]
        u_min = torch.from_numpy(env.action_space.low)
        u_max = torch.from_numpy(env.action_space.high)
        noise_sigma = 0.5 * torch.eye(env.action_space.shape[0])
        lambda_value = 0.01
        # ---
        from mppi import MPPI
        self.mppi = MPPI(self._compute_dynamics,
                         cost_function,
                         nx=state_dim,
                         num_samples=num_samples,
                         horizon=horizon,
                         noise_sigma=noise_sigma,
                         lambda_=lambda_value,
                         u_min=u_min,
                         u_max=u_max)

    def _compute_dynamics(self, state, action):
        """
        Compute next_state using the dynamics model self.model and the provided state and action tensors
        :param state: torch tensor of shape (B, state_size)
        :param action: torch tensor of shape (B, action_size)
        :return: next_state: torch tensor of shape (B, state_size) containing the predicted states from the learned model.
        """
        next_state = None
        # --- Your code here

        next_state = self.model(state, action)

        # ---
        return next_state

    def control(self, state):
        """
        Query MPPI and return the optimal action given the current state <state>
        :param state: numpy array of shape (state_size,) representing current state
        :return: action: numpy array of shape (action_size,) representing optimal action to be sent to the robot.
        TO DO:
         - Prepare the state so it can be send to the mppi controller. Note that MPPI works with torch tensors.
         - Unpack the mppi returned action to the desired format.
        """
        action = None
        state_tensor = None
        # --- Your code here
        state_tensor = torch.from_numpy(state).to(torch.float32)

        # ---
        action_tensor = self.mppi.command(state_tensor)
        # --- Your code here
        action = action_tensor.detach().cpu().numpy()

        # ---
        return action

# =========== AUXILIARY FUNCTIONS AND CLASSES HERE ===========
# --- Your code here


# ---
# ============================================================
class Rectangle:
    def __init__(self, center, dims, angle):
        self.center = center
        self.dims = dims
        self.angle = angle
        self.get_corners()
        self.get_sides()

    def get_corners(self):
        tr = torch.tensor([self.dims[0] / 2, self.dims[1] / 2])
        br = torch.tensor([self.dims[0] / 2, -self.dims[1] / 2])
        bl = torch.tensor([-self.dims[0] / 2, -self.dims[1] / 2])
        tl = torch.tensor([-self.dims[0] / 2, self.dims[1] / 2])

        rotmat = torch.tensor([[torch.cos(self.angle), -torch.sin(self.angle)],
                                 [torch.sin(self.angle), torch.cos(self.angle)]])
        pts = torch.stack([tr, br, bl, tl])
        pts = rotmat @ pts.T
        pts = pts.T + self.center
        self.corners = pts
        return pts

    def draw_corners(self):
        plt.scatter(self.corners[:, 0], self.corners[:, 1], c='r')
        plt.axis('equal')
        plt.show()

    def get_sides(self):
        corners = self.get_corners()
        tr = corners[0]
        br = corners[1]
        bl = corners[2]
        tl = corners[3]

        self.top_side = torch.stack([tl, tr])
        self.bottom_side = torch.stack([bl, br])
        self.left_side = torch.stack([tl, bl])
        self.right_side = torch.stack([tr, br])

    def get_max_x(self):
        return torch.max(self.corners[:, 0])
    
    def get_min_x(self):
        return torch.min(self.corners[:, 0])
    
    def get_max_y(self):
        return torch.max(self.corners[:, 1])
    
    def get_min_y(self):
        return torch.min(self.corners[:, 1])

    def draw_sides(self):
        plt.plot([self.top_side[0, 0], self.top_side[1, 0]], [self.top_side[0, 1], self.top_side[1, 1]], 'b')
        plt.plot([self.bottom_side[0, 0], self.bottom_side[1, 0]], [self.bottom_side[0, 1], self.bottom_side[1, 1]], 'b')
        plt.plot([self.left_side[0, 0], self.left_side[1, 0]], [self.left_side[0, 1], self.left_side[1, 1]], 'b')
        plt.plot([self.right_side[0, 0], self.right_side[1, 0]], [self.right_side[0, 1], self.right_side[1, 1]], 'b')
        plt.axis('equal')
        plt.show()        


class RectangleCompare:
    def __init__(self, rect1, rect2):
        self.rect1 = rect1
        self.rect2 = rect2

    def collides(self):
        rect1_max_x = self.rect1.get_max_x()
        rect1_min_x = self.rect1.get_min_x()
        rect1_max_y = self.rect1.get_max_y()
        rect1_min_y = self.rect1.get_min_y()

        rect2_max_x = self.rect2.get_max_x()
        rect2_min_x = self.rect2.get_min_x()
        rect2_max_y = self.rect2.get_max_y()
        rect2_min_y = self.rect2.get_min_y()

        if rect1_max_x <= rect2_min_x:
            return 0.0
        if rect1_min_x >= rect2_max_x:
            return 0.0
        if rect1_max_y <= rect2_min_y:
            return 0.0
        if rect1_min_y >= rect2_max_y:
            return 0.0
        return 1.0

def draw_corners(ax, corners):
    tr = corners[..., 0]
    br = corners[..., 1]
    bl = corners[..., 2]
    tl = corners[..., 3]
    ax.scatter(tr[..., 0], tr[..., 1], c='r')
    ax.scatter(tl[..., 0], tl[..., 1], c='k')
    ax.scatter(bl[..., 0], bl[..., 1], c='b')
    ax.scatter(br[..., 0], br[..., 1], c='g')
    return ax

def get_min_max(corners):
    tr = corners[..., 0]
    br = corners[..., 1]
    bl = corners[..., 2]
    tl = corners[..., 3]
    max_x = torch.max(torch.stack([tr[..., 0], br[..., 0], bl[..., 0], tl[..., 0]]), dim=0)[0]
    min_x = torch.min(torch.stack([tr[..., 0], br[..., 0], bl[..., 0], tl[..., 0]]), dim=0)[0]
    max_y = torch.max(torch.stack([tr[..., 1], br[..., 1], bl[..., 1], tl[..., 1]]), dim=0)[0]
    min_y = torch.min(torch.stack([tr[..., 1], br[..., 1], bl[..., 1], tl[..., 1]]), dim=0)[0]
    return min_x, max_x, min_y, max_y


def collides(corners1, obstacle_corners):
    min_x1, max_x1, min_y1, max_y1 = get_min_max(corners1)
    min_x2, max_x2, min_y2, max_y2 = get_min_max(obstacle_corners)
    collides = torch.zeros((corners1.shape[0]))
    collides = collides + (min_x1 <= max_x2).float()
    collides = collides + (max_x1 >= min_x2).float()
    collides = collides + (min_y1 <= max_y2).float()
    collides = collides + (max_y1 >= min_y2).float()
    collides = collides == 4.0
    collides = collides.float()
    return collides


def find_starting_corners(box_size, state):
    '''
    Find the points of the objects in the worls
    '''
    num_objects = state.shape[0]
    starting_corners = torch.ones((num_objects, 2, 4))
    starting_corners[:, 0, 0] = box_size[0]/2
    starting_corners[:, 0, 1] = box_size[0]/2
    starting_corners[:, 0, 2] = -box_size[0]/2
    starting_corners[:, 0, 3] = -box_size[0]/2

    starting_corners[:, 1, 0] = box_size[1]/2
    starting_corners[:, 1, 1] = -box_size[1]/2
    starting_corners[:, 1, 2] = -box_size[1]/2
    starting_corners[:, 1, 3] = box_size[1]/2


    rotmat = torch.zeros((num_objects, 2, 2))
    rotmat[:, 0, 0] = torch.cos(state[:, 2])
    rotmat[:, 0, 1] = -torch.sin(state[:, 2])
    rotmat[:, 1, 0] = torch.sin(state[:, 2])
    rotmat[:, 1, 1] = torch.cos(state[:, 2])

    starting_corners = torch.bmm(rotmat, starting_corners) + state[:, :2].unsqueeze(2)

    return starting_corners


