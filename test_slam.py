#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import time
import os
import math
import transforms3d
import scipy.io
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from util import *

cv2.useOptimized()

def esimate_pose(kp0, kp1, cam_mtx):
    E, mask = cv2.findEssentialMat(kp0, kp1, cam_mtx, method=cv2.RANSAC, prob=0.999, threshold=0.4, mask=None)

    kp0 = kp0[mask.ravel() == 1]
    kp1 = kp1[mask.ravel() == 1]
    
    # recover pose
    _, Rmat, tvec, mask = cv2.recoverPose(E, kp0, kp1, cam_mtx)
    print("Estimate pose from Emat: R: \n{}, \nt: \n{}".format(Rmat, tvec))

    return Rmat, tvec

def triangulate_points(Rmat, tvec, kp0, kp1, K):
    # triangulate points to caocluate 3d coordinates
    if np.linalg.norm(tvec) > 0:
        # create projection matrix
        P0 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
        P0 = K.dot(P0)
        P1 = np.hstack((Rmat, tvec))
        P1 = K.dot(P1)
        # triangulate to get 3d points in homogeneous coordinate (x,y,z,w)
        points1 = np.float32(kp0.reshape(-1, 1, 2))
        points2 = np.float32(kp1.reshape(-1, 1, 2))
        cloud = cv2.triangulatePoints(P0, P1, points1, points2)
        # get Euclidian coordinate (x/w,y/w,z/w)
        cloud = cloud.T
        cloud /= cloud[:, 3:]
        cloud = np.delete(cloud, 3, 1)
    else:
        cloud = None
        print("norm of t is zero, so skip triangulation")

    return cloud

def relative_scale(X0, X1):
    # calculate relative scale between 2 point clouds
	min_idx = min([X1.shape[0], X0.shape[0]])
	p_X1 = X1[:min_idx]
	f_X1 = np.roll(p_X1, shift = -3)
	p_X0 = X0[:min_idx]
	f_X0 = np.roll(p_X0, shift = -3)
	d_ratio = (np.linalg.norm(p_X0 - f_X0, axis = -1))/(np.linalg.norm(p_X1 - f_X1, axis = -1))

	return np.median(d_ratio)

def translate_points(X1, R_cam, t_cam):
    # translate camera coordinate to world coordinate
    Rt = np.identity(4)
    Rt[:3, 3] = np.array(-t_cam[:3, 0])
    Rt[:3, :3] = np.array(R_cam.T.reshape([3, 3]))
    ones = np.ones(shape=(X1.shape[0], 1))
    X1_homo = np.hstack([X1, ones])
    X1_w = X1_homo @ Rt.T
    X1_w = X1_w[:, :3]

    return X1_w

def main():
    # dataset can be downloaded from
    # https://www.robots.ox.ac.uk/~vgg/data/mview/

    videopath = "/home/spiral/share/dataset/Dinosaur/images/"
    start_frame = 0
    end_frame = 36

    # define ORB feature and descriptor
    orb = cv2.ORB_create()
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    # get camera(projection) matrix as ground truth
    mat = scipy.io.loadmat('/home/spiral/share/dataset/Dinosaur/dino_Ps.mat')
    P_gts = mat['P'][0]
    K_gts = []
    R_gts = []
    t_gts = []
    for i, P in enumerate(P_gts):
        # Decompose [K,R,t] from projection matrix
        out = cv2.decomposeProjectionMatrix(P)
        K, R, t = out[:3]
        K_gts.append(K)
        R_gts.append(R)
        t_gts.append(t)
        print("P[{}]= {}".format(i, P))
        print("K= {}, \nR= {}, \nt= {}".format(K, R, t))

    mtx = K_gts[0]
    print("Intrinsic matrix: {}".format(mtx))

    # get tracked 2D points as ground truth
    # array size: 4984 x (36 x 2)
    # row1: x1, y1, x2, y2, ..., x36, y36 for tracked points 1
    # row2: x1, y1, x2, y2, ..., x36, y36 for tracked points 2
    # ...
    # there are all 4984 tracked points
    points_list = []
    with open('/home/spiral/share/dataset/Dinosaur/viff.xy', 'r') as f:
        # read 4984 lines
        for line in f.readlines():
            # divide into 36 x 2 data
            rows = line.split()
            points = []
            for i_row in range(int(len(rows)/2)):
                point = [float(rows[i_row*2]), float(rows[i_row*2 + 1])]
                points.append(point)
            points_list.append(points)
        points_list = np.array(points_list)
        print("points_list.shape: {}".format(points_list.shape))

    X1 = None
    P1 = None

    cloud = []
    cam_pos = []
    cam_pos_gt = []

    R_cam = np.array([[1,0,0],[0,1,0],[0,0,1]])
    t_cam = np.array([[0],[0],[0]])

    frame_id = start_frame
    while True:
        # get file path
        filepath = videopath + '/viff.{:03d}.ppm'.format(frame_id)
        frame_id += 1

        if frame_id > end_frame:
            break
        elif frame_id == start_frame + 1:
            # process first (train) frame
            img1 = cv2.imread(filepath)
            if img1 is None:
                print('Cannot open file: ', filepath)
                break
            print("Read image id={} shape={}".format(frame_id, img1.shape))

            # calculate keypoint and descriptor
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            kp1, des1 = orb.detectAndCompute(gray1, None)
            continue

        else:
            # save parameter as train data
            img0 = img1
            gray0 = gray1
            kp0 = kp1
            des0 = des1
            X0 = X1
            P0 = P1

            # read second (query) image
            img1 = cv2.imread(filepath)
            if img1 is None:
                raise IOError('Cannot open file: ', filepath)
            print("Read image id={} shape={}".format(frame_id, img1.shape))

            # calculate keypoint and descriptor
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            kp1, des1 = orb.detectAndCompute(gray1, None)

            # calculate matches
            matches = matcher.match(des0, des1)
            print("Found matches n= {}".format(len(matches)))

            # sort matches according to distance
            matches = sorted(matches, key = lambda x:x.distance)

            kp0_np = np.array([kp0[idx].pt for idx in range(len(kp0))], dtype=np.float32)
            kp1_np = np.array([kp1[idx].pt for idx in range(len(kp1))], dtype=np.float32)

            # draw keypoints
            num_keypoints = 100
            disp = gray1.copy()
            disp = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)
            for i, m in enumerate(matches):
                if i > num_keypoints:
                    break
                distance = m.distance
                idx0 = m.queryIdx
                idx1 = m.trainIdx
                pt0 = kp0_np[idx0].astype(np.int32)
                pt1 = kp1_np[idx1].astype(np.int32)

                # change radius of circle depending on descriptor distance
                radius = int(math.exp(distance/15))
                cv2.line(disp, (pt0[0], pt0[1]), (pt1[0], pt1[1]), [0,255,255], 1)
                cv2.circle(disp, (pt1[0], pt1[1]), radius, [0,255,0], thickness=2)
            
            # draw 2d points of ground truth
            for i_point, pts in enumerate(points_list):
                pt = pts[frame_id - 1]
                x, y = pt[:2]
                if x > 0 and y > 0:
                    cv2.circle(disp, (int(x), int(y)), 2, [0,0,255], thickness=2)
            
            # draw matches
            matches = cv2.drawMatches(img0, kp0, img1, kp1, matches[:num_keypoints], None)

            # estimate pose
            R_rel, t_rel = esimate_pose(kp0_np, kp1_np, mtx)

            # triangulate points
            X1 = triangulate_points(R_rel, t_rel, kp0_np, kp1_np, mtx)

            # calculate 4x4 homogeneous matrix
            t_scale = 1.0
            P1 = np.hstack((R_rel, t_rel))
            P1 = np.vstack((P1, [0, 0, 0, 1]))
            # print("P1: {}".format(P1))

            if P0 is None or X0 is None:
                # first 2 frames
                X1_w = translate_points(X1, R_cam, t_cam)
                cloud.extend(X1_w)
                continue
            else:
                t_scale = relative_scale(X0, X1)

            # accumulate translation and rotation
            t_cam = t_cam - t_scale * R_cam @ t_rel
            R_cam = R_cam @ R_rel.T

            # collect estimated camera pose
            cam_pos.append([R_cam, t_cam])

            # collect ground truth
            cam_pos_gt.append([R_gts[frame_id-1], t_gts[frame_id-1]])

            # translate 3D points from camera coordinate to world coordinate
            print("X1: {}, relative_scale: {}".format(X1.shape, t_scale))
            X1_w = translate_points(X1 / t_scale, R_cam, t_cam)
            cloud.extend(X1_w)

            plot3D(cloud, cam_pos, cam_pos_gt)

            cv2.imshow("Matches", matches)
            cv2.imshow("Keypoints", disp)
            
            key = cv2.waitKey(0)
            if key & 0xFF == ord('q'):
                break
    print("Finish processing")

    while True:
        plot3D(cloud, cam_pos, cam_pos_gt)
        key = cv2.waitKey(30)
        if key & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()