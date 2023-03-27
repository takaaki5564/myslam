#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pangolin
import OpenGL.GL as gl

pangolin.CreateWindowAndBind('Main', 640, 480)
gl.glEnable(gl.GL_DEPTH_TEST)
scam = pangolin.OpenGlRenderState(
    pangolin.ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 1000),
    pangolin.ModelViewLookAt(-2.0, 2.0, 2.0, 0, 0, 0, pangolin.AxisDirection.AxisNegZ))
dcam = pangolin.CreateDisplay()
dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -640/480.0)
dcam.SetHandler(pangolin.Handler3D(scam))

#def plot3D(points3d, R_cam=None, t_cam=None):
def plot3D(points3d, cam_pos=None, cam_pos_gt=None):
    # print("points3d: {} len: {}".format(points3d, len(points3d)))
    draw_scale=1.0
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    gl.glClearColor(1.0, 1.0, 1.0, 1.0)
    dcam.Activate(scam)
 
    # draw axes
    axis_x = [[-1.1,0,0],[1.1,0,0]]
    axis_y = [[0,-1.1,0],[0,1.1,0]]
    axis_z = [[0,0,-1.1],[0,0,1.1]]
    gl.glLineWidth(2)
    gl.glColor3f(0.0, 0.0, 0.0)
    pangolin.DrawLine(axis_x)
    gl.glColor3f(0.0, 0.0, 0.0)
    pangolin.DrawLine(axis_y)
    gl.glColor3f(1.0, 0.0, 0.0)
    pangolin.DrawLine(axis_z)
    
    # draw camera pose
    if cam_pos is not None:
        for (R_cam, t_cam) in cam_pos:
            pose = np.identity(4)
            pose[:3, 3] = np.array(t_cam[:3, 0]*draw_scale)
            pose[:3, :3] = np.array(R_cam.reshape([3, 3]))
            gl.glColor3f(0.0, 0.0, 1.0) # blue
            gl.glLineWidth(2)
            pangolin.DrawCamera(pose, 0.2, 0.2, 0.2)

    # draw camera pose (ground truth)
    if cam_pos_gt is not None:
        for (R_cam, t_cam) in cam_pos_gt:
            pose = np.identity(4)
            pose[:3, 3] = np.array(t_cam[:3, 0]*draw_scale)
            pose[:3, :3] = np.array(R_cam.reshape([3, 3]))
            gl.glColor3f(1.0, 0.0, 0.0) # blue
            gl.glLineWidth(2)
            pangolin.DrawCamera(pose, 0.2, 0.4, 0.4)

    # draw 3d points
    gl.glPointSize(2)
    points = np.array(points3d)
    colors = np.zeros((len(points3d), 3))
    colors[:, :] = 0.4
    pangolin.DrawPoints(points*draw_scale, colors)
    pangolin.FinishFrame()