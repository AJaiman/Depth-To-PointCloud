import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import cv2

# Load the point cloud
point_cloud = o3d.io.read_point_cloud("pointclouds/testPC.ply")
points = np.asarray(point_cloud.points)

# Get depth values from points
depth_values = points[:, 2]  # Assuming z represents depth in your case

# Determine grid dimensions
x_min, x_max = points[:, 0].min(), points[:, 0].max()
y_min, y_max = points[:, 1].min(), points[:, 1].max()

# Define the resolution of the grid (e.g., 100x100)
resolution = 100
x_range = np.linspace(x_min, x_max, resolution)
y_range = np.linspace(y_min, y_max, resolution)

# Initialize the depth grid with the minimum depth value
min_depth = depth_values.min()
depth_grid = np.full((resolution, resolution), min_depth)

# Map (x, y) coordinates to grid indices
x_indices = np.digitize(points[:, 0], x_range) - 1
y_indices = np.digitize(points[:, 1], y_range) - 1

# Place depth values into the grid
for i in range(len(points)):
    xi, yi = x_indices[i], y_indices[i]
    depth_grid[yi, xi] = points[i, 2]

# Normalize depth grid for visualization
depth_grid_normalized = (depth_grid - depth_grid.min()) / (depth_grid.max() - depth_grid.min())
depth_grid_visual = (depth_grid_normalized * 255).astype(np.uint8)

# Save the depth grid to a file
np.save('depth_grid.npy', depth_grid)

# Create and save the depth map image
cv2.imwrite('depth_map.png', depth_grid_visual)

# Visualize the depth map using matplotlib
plt.imshow(depth_grid_visual, cmap='gray')
plt.colorbar()
plt.title('Depth Map')
plt.show()
