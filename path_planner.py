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

# Compute the minimum depth value
min_depth = np.nanmin(depth_grid)

# Define the size of the final depth grid
final_grid_size = x

# Initialize the final depth grid
final_depth_grid = np.full((final_grid_size, final_grid_size), min_depth)

# Define the size of each sub-grid in the original grid
sub_grid_size = resolution // final_grid_size

# Process each sub-grid
for i in range(final_grid_size):
    for j in range(final_grid_size):
        # Define the bounds of the current sub-grid
        start_row, end_row = i * sub_grid_size, (i + 1) * sub_grid_size
        start_col, end_col = j * sub_grid_size, (j + 1) * sub_grid_size

        # Extract the sub-grid
        sub_grid = depth_grid[start_row:end_row, start_col:end_col]

        # Check if the sub-grid is empty (contains only NaNs)
        if np.all(np.isnan(sub_grid)):
            # Set the depth value to the minimum depth value
            final_depth_grid[i, j] = min_depth
        else:
            # Compute the average depth value of the sub-grid
            avg_depth = np.nanmean(sub_grid)
            # Fill the final grid with the average depth value
            final_depth_grid[i, j] = avg_depth

# Normalize final depth grid for visualization
final_depth_grid_normalized = (final_depth_grid - np.nanmin(final_depth_grid)) / (np.nanmax(final_depth_grid) - np.nanmin(final_depth_grid))
final_depth_grid_visual = (final_depth_grid_normalized * 255).astype(np.uint8)

# Save the final depth grid to a file
np.save('final_depth_grid.npy', final_depth_grid)

# Create and save the final depth map image
cv2.imwrite('final_depth_map.png', final_depth_grid_visual)

# Visualize the final depth map using matplotlib
plt.imshow(final_depth_grid_visual, cmap='gray')
plt.colorbar()
plt.title('Final Depth Map')
plt.show()
print(len(final_depth_grid))