#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 15:54:33 2021

@author: maximilianvanamerongen
"""

import cv2
from cv2 import aruco
import numpy as np
from matplotlib import image
from matplotlib import pyplot as plt
import time

# Load image for testing
#input_image = image.imread('/Users/maximilianvanamerongen/Documents/Master/EPFL/ITMR/Project/ThymioProject/Pictures/testImage.jpg')

# Caclulate pix2mm_y pix2mm_x
#cal_pix2mm(input_image)

# Detect marker and calculate the position:
#x_center, y_center,theta = camera_pose(input_image)
  
# Plot picture with a marker at the calculated center:
#plt.plot(x_center, y_center, marker='v', color="white")
#plt.imshow(input_image)
#plt.show()

def cal_pix2mm(input_image):
    # Input: input_image := image of your map
    # Transform pixel to mm 
    
    height = input_image.shape[0] 
    width  = input_image.shape[1]
    
    pix2m_y = 0.891/height
    pix2m_x = 1.260/width
    
    return pix2m_y, pix2m_x


def pose(input_image, doPath = False):
    # input: input_image := image of your map
    # output: x_center, y_center, theta = absolute coordinates of the detected marker
    dictionary = cv2.aruco.Dictionary_get(aruco.DICT_4X4_250)
    marker_parameters = cv2.aruco.DetectorParameters_create()
    markerCorners, markerIDs, rejectedImgPoints = cv2.aruco.detectMarkers(input_image,dictionary,parameters = marker_parameters)
    
    markerCorners = np.array(markerCorners)

    if markerIDs is not None:
        
        height = input_image.shape[0]
        
        #markerCorners[0,0,:,1] = height - markerCorners[0,0,:,1] # Flipping y axis to put origin of the global frame at bottom left corner
        
        x_corner  = markerCorners[0,0,0:4,0]
        y_corner  = markerCorners[0,0,0:4,1]
        
        x_center  = np.mean(x_corner)
        y_center  = np.mean(y_corner)
        
        # Calculate orientation: 
        orient_vec = (markerCorners[0,0,1,:]+markerCorners[0,0,0,:])/2 - (markerCorners[0,0,2,:]+markerCorners[0,0,3,:])/2
        
        theta = np.arctan2(orient_vec[0],orient_vec[1])
    
    else:
        x_center = -1
        y_center = -1
        theta    = -1000
        return x_center, y_center, theta
    
    if doPath:
        return x_center, y_center, theta  # height-y_center
    else:
        pix2m_y, pix2m_x = cal_pix2mm(input_image)
        return [x_center * pix2m_x, y_center * pix2m_y, theta]  
    
        