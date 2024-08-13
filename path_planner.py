import cv2
import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np

pc = o3d.io.read_point_cloud("pointclouds/testPC.ply")
print(pc)