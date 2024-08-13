# Setup Enviorment and Imports
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import cv2
import matplotlib.pyplot as plt
import open3d as o3d

import os
import random
from pathlib import Path
import glob

# Load Sample Image

imagePath = 'images/roomSample0.png'
image = cv2.imread(imagePath)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Import Depth Anything Model

processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-small-hf")
model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-small-hf").to('cuda')

# Pre-Process Images

input = processor(images=image, return_tensors='pt').to('cuda')

#Inference
with torch.no_grad():
    outputs = model(**input)
    depth = outputs.predicted_depth

depth = depth.squeeze().cpu().numpy()
imageSample = [image, depth]

# Visualize Depth Image and Refrence Image
"""
fig, axs = plt.subplots(1, 2)
axs[0].imshow(imageSample[0])
axs[0].set_title('Original Image')
axs[1].imshow(imageSample[1])
axs[1].set_title('Depth Image')
plt.show()
"""
# Important Functions

def get_intrinsics(H,W, fov=73.0):
    """
    Returns Camera's Intrinsic Matrix
    FOV of IMX219-83 is 73 degrees (horizontal)
    """
    f = 0.5 * W / np.tan(0.5 * fov * np.pi / 180)
    cx = 0.5 * W
    cy = 0.5 * H
    return np.array([[f, 0, cx],
                     [0, f, cy],
                     [0, 0, 1]])


def pixel_to_point(depth_image, camera_intrinsics=None):
    """
    Convert depth iamge to 3D points
    FOV of 73 degrees
    """

    height, width = depth_image.shape
    if camera_intrinsics is None:
        camera_intrinsics = get_intrinsics(height, width)
    
    # Create u, v meshgrid and precompute projection triangle ratios
    fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
    cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]

    x = np.linspace(0, width -1, width)
    y = np.linspace(0, height -1, height)
    u, v = np.meshgrid(x, y)

    x_over_z = (u - cx) / (fx)
    y_over_z = (v - cy) / (fy)

    # 3-D Pythagoras re-arranged to solve for 2
    z = depth_image / np.sqrt(1. + x_over_z**2 + y_over_z**2)
    x = x_over_z
    y = y_over_z

    return x, y, z

def create_point_cloud(depth_image, color_image, camera_intrinsics=None, scale_ratio=10.0):

    height, width = depth_image.shape
    if camera_intrinsics is None:
        camera_intrinsics = get_intrinsics(height, width)

        color_image = cv2.resize(color_image, (width, height))


        #Make sure depth_image doesn't contain any zeros
        depth_image = np.maximum(depth_image, 1e-5)

        depth_image = scale_ratio /depth_image
        x, y, z = pixel_to_point(depth_image, camera_intrinsics)
        point_image = np.stack((x, y, z), axis=-1)

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(point_image.reshape(-1, 3))
        cloud.colors = o3d.utility.Vector3dVector(color_image.reshape(-1, 3) / 255)

        return cloud

def filter_pointcloud(cloud, y_threshold, z_threshold):
    points = np.asarray(cloud.points)
    mask = points[:, 1] <= y_threshold
    filteredPoints = points[mask]
    mask = filteredPoints[:, 2] <= z_threshold
    filteredPoints = filteredPoints[mask]
    filteredPoints[:, [1, 2]] = filteredPoints[:, [2, 1]]
    filteredPoints[:, 2] = filteredPoints[:, 2] * -1
    return o3d.utility.Vector3dVector(filteredPoints)


# Output Point Cloud


output_path = './pointclouds'

cloud = create_point_cloud(imageSample[1], imageSample[0])
cloud.points = filter_pointcloud(cloud, 0.2, 3)
print(type(cloud))
o3d.io.write_point_cloud(os.path.join(output_path, f'testPC.ply'), cloud)


