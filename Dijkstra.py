# Dijkstra.py
import numpy as np
import heapq

def dijkstra(array, start, end):
    """Perform Dijkstra's algorithm to find the shortest path from start to end in the given array."""
    rows, cols = array.shape
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}

    while open_set:
        current = heapq.heappop(open_set)[1]

        if current == end:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        for i, j in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            neighbor = (current[0] + i, current[1] + j)
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                if array[neighbor] == 0:
                    continue  # Skip if the neighbor is an impassable area
                tentative_g_score = g_score[current] + 1

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    heapq.heappush(open_set, (g_score[neighbor], neighbor))

    return None  # Return None if no path is found

def update_array_with_path(array, path):
    """Update the array with the path found by setting path points to 0.5."""
    for coord in path:
        array[coord] = 0.5

# Example usage:
if __name__ == "__main__":
    map_array = np.array([
        [1, 1, 1, 0, 1],
        [1, 0, 1, 1, 1],
        [1, 1, 1, 0, 1],
        [0, 0, 1, 1, 1],
        [1, 1, 1, 1, 1]
    ])

    start = (0, 0)  # Starting coordinate
    end = (4, 4)    # Ending coordinate

    path = dijkstra(map_array, start, end)

    if path:
        update_array_with_path(map_array, path)
        print("Path found and updated:")
        print(map_array)
    else:
        print("No path found.")
