import torch
import argparse

from nerf.provider import NeRFDataset
from nerf.gui import NeRFGUI
from nerf.utils import *

from functools import partial
from loss import huber_loss

from nerf.network import NeRFNetwork

import cv2 as cv
from scipy.spatial.transform import Rotation as R

class ObjectNerfManager(object):
    def __init__(self, pth_checkpoint, device = "cuda:0", bound = 2) -> None:
        self.model = NeRFNetwork(
            encoding="hashgrid",
            bound=bound,
            cuda_ray=True,
            density_scale=1,
            min_near=0.2,
            density_thresh=10,
            bg_radius=-1,
        )
        self.bound = bound
        checkpoint_dict = torch.load(pth_checkpoint)
        missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint_dict["model"], strict=False)

        self.model.eval()
        self.device = device
        self.model = self.model.to(device)

    def get_contour_with_transform(self, obj_transform, z_height, grid_len = 1, grid_res = 100):
        object_pos = obj_transform[:3, 3]
        # generate a set of samples in world coordinates that are at the provided z height
        # and are in a grid around the object's x/y position
        xs = torch.linspace(object_pos[0] - grid_len/2, object_pos[0] + grid_len/2, grid_res)
        ys = torch.linspace(object_pos[1] - grid_len/2, object_pos[1] + grid_len/2, grid_res)
        X_grid, Y_grid = torch.meshgrid(xs, ys)
        samples_world = torch.stack([X_grid, Y_grid, torch.ones_like(X_grid) * z_height], dim=-1).reshape(-1, 3).to(device=self.device)

        # need to now transform those samples to the object's frame
        # make the samples homogenous
        samples_world = torch.cat([samples_world, torch.ones_like(samples_world[:, :1])], dim=-1)
        samples_obj = torch.matmul(samples_world, obj_transform.to(self.device).T)
        samples_obj = samples_obj[:, :3]
        samples_world = samples_world[:, :3]

        # execute the model on the samples
        with torch.no_grad():
            results = self.model.density(samples_obj)['sigma'].detach()

            # check if any of the samples were outside the box
            # if so, set them to 0
            bbox_min = -1 * self.bound
            bbox_max = self.bound
            outside_bbox = (samples_obj[:, 0] < bbox_min) |\
                (samples_obj[:, 0] > bbox_max) |\
                (samples_obj[:, 1] < bbox_min) |\
                (samples_obj[:, 1] > bbox_max) |\
                (samples_obj[:, 2] < bbox_min) |\
                (samples_obj[:, 2] > bbox_max)
            results[outside_bbox] = 0

            # reshape the results
            results = results.reshape(grid_res, grid_res)

            # # threshold the results
            # thresh = 1.
            # results[results < thresh] = 0.
            # results[results > thresh] = 1.

        results = results.cpu().numpy()
        # get the contours
        contours, _ = cv.findContours(results.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)

        samples_world = samples_world.reshape(grid_res, grid_res, 3)
        samples_obj = samples_obj.reshape(grid_res, grid_res, 3)

        max_x = np.max(largest_contour[:, :, 0])
        min_x = np.min(largest_contour[:, :, 0])
        max_y = np.max(largest_contour[:, :, 1])
        min_y = np.min(largest_contour[:, :, 1])
        max_point = samples_world[max_x, max_y]
        min_point = samples_world[min_x, min_y]
        deltas = max_point - min_point

        return largest_contour, results, samples_obj, samples_world, deltas
    
    def get_rand_action_xy_with_transform(self, obj_transform, z_height, grid_len = 1, grid_res = 100):
        contour, results_raw, samples_obj, samples_world, deltas = self.get_contour_with_transform(obj_transform, z_height, grid_len, grid_res)

        # select a random phi from -pi to pi
        rand_phi = np.random.uniform(-np.pi, np.pi)

        # for each point on the contour, get the phi
        contour_phis = np.arctan2(contour[:, :, 1] - (grid_res / 2), contour[:, :, 0] - (grid_res / 2))

        # find the closest phi to the random phi
        closest_phis = np.abs(contour_phis - rand_phi)
        closest_phis = closest_phis.reshape(-1)
        closest_idx = np.argmin(closest_phis)

        # select a random point on the contour
        rand_point = contour[closest_idx]
        # get its neighbor points for normal estimation
        neighbor_1 = contour[closest_idx - 1 if closest_idx - 1 >= 0 else len(contour) - 1]
        neighbor_2 = contour[0 if closest_idx + 1 >= len(contour) else closest_idx + 1]

        return samples_obj[rand_point[0,0], rand_point[0,1]] / 2, deltas / 2, rand_phi
    
    def get_object_tf(self, env):
        # get the object's pose
        object_pose = env.get_object_pose()
        # convert the pose to a transform
        tf_homogenous = torch.eye(4)
        # set the translation
        tf_homogenous[0, :3] = torch.from_numpy(object_pose[:3])
        # set the rotation
        quat = object_pose[3:]
        r = R.from_quat(quat)
        tf_homogenous[:3, :3] = torch.from_numpy(r.as_matrix()).float()
        return tf_homogenous