import sys
from math import pi, sin, cos, tan, acos, atan2

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def eular2rot(theta):
    Rx = np.array([
        [1, 0, 0],
        [0, cos(theta[0]), -sin(theta[0])],
        [0, sin(theta[0]), cos(theta[0])]
    ])
    Ry = np.array([
        [cos(theta[1]), 0, sin(theta[1])],
        [0, 1, 0],
        [-sin(theta[1]), 0, cos(theta[1])]
    ])
    Rz = np.array([
        [cos(theta[2]), -sin(theta[2]), 0],
        [sin(theta[2]), cos(theta[2]), 0],
        [0, 0, 1]
    ])
    
    return Rx @ Ry @ Rz
    
def rotate_pixel(in_vec, rot_mat, w, h):
    vec_rad = (pi*in_vec[0] / h, 2*pi*in_vec[1]/w)
    vec_cartesian = (
        -sin(vec_rad[0]) * cos(vec_rad[1]),
        sin(vec_rad[0]) * sin(vec_rad[1]),
        cos(vec_rad[0])
    )
    
    vec_cartesian_rot = (
        rot_mat[0, 0]*vec_cartesian[0] + rot_mat[0, 1]*vec_cartesian[1] + rot_mat[0, 2]*vec_cartesian[2],
        rot_mat[1, 0]*vec_cartesian[0] + rot_mat[1, 1]*vec_cartesian[1] + rot_mat[1, 2]*vec_cartesian[2],
        rot_mat[2, 0]*vec_cartesian[0] + rot_mat[2, 1]*vec_cartesian[1] + rot_mat[2, 2]*vec_cartesian[2]
    )
    
    vec_rot = [
        acos(vec_cartesian_rot[2]), 
        atan2(vec_cartesian_rot[1], -vec_cartesian_rot[0])
    ]
    if vec_rot[1] < 0:
        vec_rot[1] += pi * 2
    
    vec_pixel = (
        h * vec_rot[0] / pi,
        w * vec_rot[1] / (2*pi)
    )
    return vec_pixel

def RAD(x):
    return pi * x / 180.0

def DEGREE(x):
    return 180.0 * x / pi

def rotate_image(im, theta):
    h, w = im.shape[:2]
    
    srci = np.zeros((h, w), dtype=np.float32)
    srcj = np.zeros((h, w), dtype=np.float32)
    
    rot_mat = eular2rot(theta)
    for i in range(h):
        for j in range(w):
            vec_pixel = rotate_pixel((i, j), rot_mat, w, h)
            srci[i, j] = vec_pixel[0]
            srcj[i, j] = vec_pixel[1]
            
    im_out = cv2.remap(im, srcj, srci, cv2.INTER_LINEAR)
    return im_out
