#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import least_squares

def bundle_adjustment(x0, points_3d, points_2d, camera_poses):
    '''
    x0: combined array of camera poses (6DoF) and 3D points (3DoF).
        Set appropriate initial value of bundle adjustment.
    
    '''
    num_cameras = len(camera_poses)
    num_points = points_3d.shape[0]
    num_params = num_cameras * 6 + num_points * 3
    parameters = np.zeros(num_params)

    # Set initial camera parameters
    for i, camera_pose in enumerate(camera_poses):
        parameters[i * 6:(i + 1) * 6] = camera_pose.flatten()

    # Set initial 3D point parameters
    parameters[num_cameras * 6:] = points_3d.flatten()

    def fun(params, n_cams, n_pts, pts_3d, pts_2d):
        cam_params = params[:n_cams * 6].reshape((n_cams, 6))
        pt_params = params[n_cams * 6:].reshape((n_pts, 3))

        error = []
        for i, camera_pose in enumerate(cam_params):
            projected_points = project_points(camera_pose, pt_params)
            error.append((projected_points - pts_2d[i]).ravel())

        return np.concatenate(error)

    # adjust poses to minimize projection error
    result = least_squares(fun, parameters, args=(num_cameras, num_points, points_3d, points_2d))
    return result.x.reshape((-1, 6)), result.cost

def project_points(camera_pose, points_3d):
    # convert 3D points to 2D image by projection
    camera_matrix = np.zeros((3, 4))
    camera_matrix[:3, :3] = camera_pose[:, :3]
    camera_matrix[:3, 3] = camera_pose[:, 3]

    projected_points = camera_matrix @ np.hstack((points_3d, np.ones((points_3d.shape[0], 1)))).T
    projected_points = projected_points[:2, :] / projected_points[2, :]
    return projected_points.T