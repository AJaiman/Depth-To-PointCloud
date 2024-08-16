import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import cv2
from GridBox import GridBox

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

# Define the size of each GridBox sub-grid
l = 2  # This variable can be edited

# Calculate the number of GridBox objects along each axis
grid_box_rows = final_grid_size // l
grid_box_cols = final_grid_size // l

# Function to calculate the angle using least squares method
def calculate_angle(sub_grid):
    y, x = np.mgrid[0:sub_grid.shape[0], 0:sub_grid.shape[1]]
    A = np.column_stack((x.flatten(), y.flatten(), np.ones_like(x.flatten())))
    b = sub_grid.flatten()
    coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    a, b, _ = coeffs
    return np.degrees(np.arctan(np.sqrt(a**2 + b**2)))

# Initialize the 2D array of GridBox objects
grid_boxes = []

# Padding for the sliding window
pad_size = l

# Pad the final_depth_grid
padded_grid = np.pad(final_depth_grid, pad_size, mode='edge')

# Create GridBox objects by slicing the final_depth_grid
for i in range(grid_box_rows):
    row = []
    for j in range(grid_box_cols):
        # Define the bounds of the current GridBox sub-grid
        start_row, end_row = i * l, (i + 1) * l
        start_col, end_col = j * l, (j + 1) * l

        # Define the bounds for the 3l x 3l sliding window
        window_start_row, window_end_row = start_row, end_row + 2*l
        window_start_col, window_end_col = start_col, end_col + 2*l

        # Slice the padded grid to get the 3l x 3l window
        window = padded_grid[window_start_row:window_end_row, window_start_col:window_end_col]

        # Calculate the angle for the 3l x 3l window
        theta = calculate_angle(window)

        # Slice the final_depth_grid to get the l x l sub-array
        sub_grid = final_depth_grid[start_row:end_row, start_col:end_col]

        # Create a GridBox object with the calculated angle and add it to the row
        grid_box = GridBox(sub_grid, theta)
        row.append(grid_box)

    # Add the row of GridBox objects to the grid_boxes array
    grid_boxes.append(row)

# Set neighbors for each GridBox
for i in range(grid_box_rows):
    for j in range(grid_box_cols):
        n = grid_boxes[i-1][j] if i > 0 else None
        s = grid_boxes[i+1][j] if i < grid_box_rows-1 else None
        e = grid_boxes[i][j+1] if j < grid_box_cols-1 else None
        w = grid_boxes[i][j-1] if j > 0 else None
        ne = grid_boxes[i-1][j+1] if i > 0 and j < grid_box_cols-1 else None
        nw = grid_boxes[i-1][j-1] if i > 0 and j > 0 else None
        se = grid_boxes[i+1][j+1] if i < grid_box_rows-1 and j < grid_box_cols-1 else None
        sw = grid_boxes[i+1][j-1] if i < grid_box_rows-1 and j > 0 else None
        
        grid_boxes[i][j].set_neighbors(n, s, e, w, ne, nw, se, sw)

# Convert grid_boxes to a numpy array for easier manipulation
grid_boxes = np.array(grid_boxes)

print(grid_boxes[19, 5].theta)



# Uncomment for Visualization

#  Normalize final depth grid for visualization
final_depth_grid_normalized = (final_depth_grid - np.nanmin(final_depth_grid)) / (np.nanmax(final_depth_grid) - np.nanmin(final_depth_grid))
final_depth_grid_visual = (final_depth_grid_normalized * 255).astype(np.uint8)
# Show Overhead Depth Map
plt.imshow(final_depth_grid_visual, cmap='gray')
plt.colorbar()
plt.title('Final Depth Map')
plt.show()
