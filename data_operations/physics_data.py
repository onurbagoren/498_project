import os
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
import torch


class PhysicsDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_list = os.listdir(data_dir)
        self.file_list = [f for f in self.file_list if f.endswith('.npz')]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        file_path = os.path.join(self.data_dir, file_name)

        # Load data from npz file
        data = np.load(file_path, allow_pickle=True)

        # Convert numpy arrays to PyTorch tensors
        simulation_time = torch.from_numpy(data['simulation_time']).float()
        moving_position = torch.from_numpy(data['moving_position']).float()
        moving_orientation = torch.from_numpy(
            data['moving_orientation']).float()
        moving_velocity = torch.from_numpy(data['moving_velocity']).float()
        moving_angular_velocity = torch.from_numpy(
            data['moving_angular_velocity']).float()
        static_position = torch.from_numpy(data['static_position']).float()
        static_orientation = torch.from_numpy(
            data['static_orientation']).float()
        static_velocity = torch.from_numpy(data['static_velocity']).float()
        static_angular_velocity = torch.from_numpy(
            data['static_angular_velocity']).float()
        contact_points_moving = torch.from_numpy(
            data['contact_points_moving']).float()
        contact_points_static = torch.from_numpy(
            data['contact_points_static']).float()
        contact_normal = torch.from_numpy(data['contact_normal']).float()
        contact_normal_force = torch.from_numpy(
            data['contact_normal_force']).float()
        contact_times = torch.from_numpy(data['contact_times']).float()
        lateral_friction_force1 = torch.from_numpy(
            data['lateral_friction_force1']).float()
        lateral_friction_force2 = torch.from_numpy(
            data['lateral_friction_force2']).float()
        lateral_friction_dir1 = torch.from_numpy(
            data['lateral_friction_dir1']).float()
        lateral_friction_dir2 = torch.from_numpy(
            data['lateral_friction_dir2']).float()
        moving_mass = torch.tensor(data['moving_mass']).float()
        moving_inertia = torch.from_numpy(data['moving_inertia']).float()
        moving_friction = torch.from_numpy(data['moving_friction']).float()
        static_mass = torch.tensor(data['static_mass']).float()
        static_inertia = torch.from_numpy(data['static_inertia']).float()
        static_friction = torch.from_numpy(data['static_friction']).float()

        # Combine data into a dictionary
        sample = {'simulation_time': simulation_time, # (1, N)
                  'moving_position': moving_position, # (1, N, 3)
                  'moving_orientation': moving_orientation, # (1, N, 4) QUATERNION
                  'moving_velocity': moving_velocity, # (1, N, 3)
                  'moving_angular_velocity': moving_angular_velocity, # (1, N, 3)
                  'static_position': static_position, # (1, N, 3)
                  'static_orientation': static_orientation, # (1, N, 4) QUATERNION
                  'static_velocity': static_velocity, # (1, N, 3)
                  'static_angular_velocity': static_angular_velocity, # (1, N, 3)
                  'contact_points_moving': contact_points_moving, # (1, M, 3) NOTICE DIFFERENT SIZE
                  'contact_points_static': contact_points_static, # (1, M, 3) NOTICE DIFFERENT SIZE
                  'contact_normal': contact_normal, # (1, M, 3) NOTICE DIFFERENT SIZE
                  'contact_normal_force': contact_normal_force, # (1, M) NOTICE DIFFERENT SIZE
                  'contact_times': contact_times, # (1, M) NOTICE DIFFERENT SIZE
                  'lateral_friction_force': lateral_friction_force1, # (1, M) NOTICE DIFFERENT SIZE
                  'lateral_friction_force2': lateral_friction_force2, # (1, M) NOTICE DIFFERENT SIZE
                  'lateral_friction_dir': lateral_friction_dir1, # (1, M, 3) NOTICE DIFFERENT SIZE
                  'lateral_friction_dir2': lateral_friction_dir2, # (1, M, 3) NOTICE DIFFERENT SIZE
                  'moving_mass': moving_mass, # (1)
                  'moving_inertia': moving_inertia, # (1, 3)
                  'moving_friction': moving_friction, # (1, 3)
                  'static_mass': static_mass, # (1)
                  'static_inertia': static_inertia, # (1, 3)
                  'static_friction': static_friction # (1, 3)
                  }

        return sample


def main():
    dataset = PhysicsDataset(
        '/media/frog/DATA/Datasets/498_project/logs/linear_collision_ycb_extra_massinfo')
    split_dataset = random_split(
        dataset, [int(len(dataset)*0.8), int(len(dataset) * 0.2)]
    )

    train_data = split_dataset[0]
    test_data = split_dataset[1]

    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=True)
    for i_batch, sample_batched in enumerate(train_loader):
        print(i_batch)
        print('\t Simulation time:', sample_batched['moving_inertia'].shape)
        print('\t Moving position:', sample_batched['moving_position'].shape)
        print('\t Moving orientation', sample_batched['moving_orientation'].shape)
        print('\t Contact points moving:', sample_batched['contact_points_moving'].shape)

        # observe 4th batch and stop.
        if i_batch == 3:
            break


if __name__ == '__main__':
    main()
