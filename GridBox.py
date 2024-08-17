import numpy as np

class GridBox:
    def __init__(self, elevation_grid, theta):
        self.elevation_grid = elevation_grid  # This is a 2D array
        self.theta = theta
        
        # References to surrounding GridBox objects
        self.north = None
        self.south = None
        self.east = None
        self.west = None
        self.northeast = None
        self.northwest = None
        self.southeast = None
        self.southwest = None

        # Constants (these should be set to appropriate values)
        self.M = 1000  # Mass of the rover in kg
        self.g_lun = 1.62  # Lunar gravitational acceleration in m/s^2
        self.l = 1  # Grid size in meters
        self.theta_max = 1  # Maximum allowable slope angle in degrees
        self.l_max = 0.5  # Maximum surmountable obstacle height in meters

    def set_neighbors(self, n, s, e, w, ne, nw, se, sw):
        self.north = n
        self.south = s
        self.east = e
        self.west = w
        self.northeast = ne
        self.northwest = nw
        self.southeast = se
        self.southwest = sw

    def slope_climbing_cost(self):
        if 0 <= self.theta <= self.theta_max:
            theta_rad = np.radians(self.theta)
            l_z = self.l * np.sin(theta_rad)
            return self.M * self.g_lun * l_z
        else:
            return float('inf')

    def obstacle_crossing_cost(self):
        # Calculate l_obs (maximum obstacle height)
        l_ele = self.calculate_l_ele()
        l_obs = max(l_ele) - min(l_ele)
        
        if l_obs <= self.l_max:
            return self.M * self.g_lun * l_obs
        else:
            return float('inf')

    def calculate_l_ele(self):
        # Fit a plane to the elevation grid
        y, x = np.mgrid[0:self.elevation_grid.shape[0], 0:self.elevation_grid.shape[1]]
        A = np.column_stack((x.flatten(), y.flatten(), np.ones_like(x.flatten())))
        b = self.elevation_grid.flatten()
        coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        a, b, c = coeffs
        
        # Calculate distances from points to the fitted plane
        l_ele = []
        for i in range(self.elevation_grid.shape[0]):
            for j in range(self.elevation_grid.shape[1]):
                z = self.elevation_grid[i, j]
                d = abs(a*j + b*i + c - z) / np.sqrt(a**2 + b**2 + 1)
                l_ele.append(d)
        
        return l_ele
    
    def total_cost(self):
        return self.obstacle_crossing_cost() + self.slope_climbing_cost()