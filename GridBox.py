class GridBox:
    def __init__(self, points):
        self.points = points
    
    def obstacle_crossing_cost(self):
        return 0
    
    def slope_climbing_cost(self):
        return 0
    
    def total_cost(self):
        return self.obstacle_crossing_cost() + self.slope_climbing_cost()