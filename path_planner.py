import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import cv2

# Load the point cloud
point_cloud = o3d.io.read_point_cloud("pointclouds/testPC.ply")
points = np.asarray(point_cloud.points)

# Get depth values from points
depth_values = points[:, 2]  # Assuming z represents depth in your case

# Define the resolution of the grid and the number of sub-grids
resolution = 100
x = 50  # Number of smaller grids along each axis
x_min, x_max = points[:, 0].min(), points[:, 0].max()
y_min, y_max = points[:, 1].min(), points[:, 1].max()

# Define the range for x and y
x_range = np.linspace(x_min, x_max, resolution)
y_range = np.linspace(y_min, y_max, resolution)

# Initialize the depth grid
depth_grid = np.full((resolution, resolution), np.nan)

# Map (x, y) coordinates to grid indices
x_indices = np.digitize(points[:, 0], x_range) - 1
y_indices = np.digitize(points[:, 1], y_range) - 1

# Place depth values into the grid
for i in range(len(points)):
    xi, yi = x_indices[i], y_indices[i]
    depth_grid[yi, xi] = points[i, 2]

# Define the number of sub-grids
sub_grid_size = resolution // x

# Process each sub-grid
for i in range(x):
    for j in range(x):
        # Define the bounds of the current sub-grid
        start_row, end_row = i * sub_grid_size, (i + 1) * sub_grid_size
        start_col, end_col = j * sub_grid_size, (j + 1) * sub_grid_size

        # Extract the sub-grid
        sub_grid = depth_grid[start_row:end_row, start_col:end_col]

        # Compute the average depth value of the sub-grid
        avg_depth = np.nanmean(sub_grid)

        # Fill the sub-grid with the average depth value
        depth_grid[start_row:end_row, start_col:end_col] = avg_depth

# Normalize depth grid for visualization
depth_grid_normalized = (depth_grid - np.nanmin(depth_grid)) / (np.nanmax(depth_grid) - np.nanmin(depth_grid))
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
