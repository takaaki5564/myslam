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

    print("return 3d points: {}".format(cloud))
    return cloud


def project_points(K, R, t, points):
    Rt = np.hstack((R, t))
    P = np.dot(K, Rt)
    points_homo = np.hstack((points, np.ones((points.shape[0], 1))))
    points_homo_projected = np.dot(P, points_homo.T).t
    points_projected = points_homo_projected[:, :2] / points_homo_projected[:, 2:]
    return points_projected

def esimate_scale(K, P0, P1, X0, X1):
    R0, t0 = P0[:, :3], P0[:, 3]
    R1, t1 = P1[:, :3], P1[:, 3]

    # calculate projected points
    x0 = projected_points(K, R0, t0, X0)
    x1 = projected_points(K, R1, t1, X1)
    print("point projected x0: {}".format(x0))
    print("point projected x1: {}".format(x1))

    scale = np.mean(np.linalg.norm(x1 - x0, axis=1) / np.linalg.norm(X1 - X0, axis=1))
    print("scale: {}".format(scale))
    return scale


img_w = 720
img_h = 576
f = 500

mtx = np.array([[f, 0, img_w/2],[0, f, img_h/2], [0, 0, 1]])

def main():
    # for KITTI
    videopath = "/home/spiral/share/dataset/gozzila/images/"
    start_frame = 0
    end_frame = 36

    orb = cv2.ORB_create()
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    X1 = None
    P1 = None

    frame_id = start_frame
    while True:
        if frame_id > end_frame:
            frame_id = start_frame
            
        # get file path
        filepath = videopath + '/viff.{:03d}.ppm'.format(frame_id)
        frame_id += 1

        if frame_id == start_frame + 1:
            # process first (train) frame
            img1 = cv2.imread(filepath)
            if img1 is None:
                raise IOError('Cannot open file: ', filepath)
            print("Read image id={} shape={}".format(frame_id, img1.shape))

            # calculate keypoint and descriptor
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            kp1, des1 = orb.detectAndCompute(gray1, None)
            continue

        else:
            # take parameter as train data
            img0 = img1
            gray0 = gray1
            kp0 = kp1
            des0 = des1
            X0 = X1
            P0 = P1

            # read second (query) image
            img1 = cv2.imread(filepath)
            if img1 is None:
                print('Cannot open file: ', filepath)
                break
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
            num_keypoints = 200
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

                radius = int(math.exp(distance/20))
                # print("i={}, radius={}".format(i, radius))
                cv2.line(disp, (pt0[0], pt0[1]), (pt1[0], pt1[1]), [0,255,255], 1)
                cv2.circle(disp, (pt1[0], pt1[1]), radius, [0,255,0], thickness=2)
            
            # draw matches
            matches = cv2.drawMatches(img0, kp0, img1, kp1, matches[:num_keypoints], None)

            # estimate pose
            R, t = esimate_pose(kp0_np, kp1_np, mtx)

            # triangulate points
            X1 = triangulate_points(R, t, kp0_np, kp1_np, mtx)

            plot3D(X1)

            cv2.imshow("Matches", matches)
            cv2.imshow("Keypoints", disp)
            
            key = cv2.waitKey(0)
            if key & 0xFF == ord('q'):
                break




if __name__ == '__main__':
    main()