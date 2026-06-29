"""A* Global Path Planner with Line-of-Sight Pruning and Gradient Smoothing."""

import numpy as np
import heapq
import math

class AStarPlanner:
    def __init__(self):      
        self.resolution = 0.1
        self.grid_size = 100 
        self.robot_radius = 0.3

    def plan(self, start_xy, goal_xy, obstacles):
        grid = np.zeros((self.grid_size, self.grid_size))
        for obs in obstacles:
            ox, oy = obs[0], obs[1]
            ix = int(ox / self.resolution)
            iy = int(oy / self.resolution)
            inf_cells = int(math.ceil(self.robot_radius / self.resolution))
            for dx in range(-inf_cells, inf_cells + 1):
                for dy in range(-inf_cells, inf_cells + 1):
                    nx, ny = ix + dx, iy + dy
                    if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                        grid[ny][nx] = 1

        start_cell = (max(0, min(99, int(start_xy[1] / self.resolution))),
                      max(0, min(99, int(start_xy[0] / self.resolution))))
        goal_cell = (max(0, min(99, int(goal_xy[1] / self.resolution))),
                     max(0, min(99, int(goal_xy[0] / self.resolution))))

        path_cells = self._a_star_search(grid, start_cell, goal_cell)
        if path_cells is None: return [goal_xy]

        waypoints = [[c[1] * self.resolution, c[0] * self.resolution] for c in path_cells]
        pruned = self._prune_path(waypoints, grid)
        smoothed = self._smooth_path(pruned)
        return smoothed

    def _a_star_search(self, grid, start, goal):
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal: return self._reconstruct_path(came_from, current)
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)]:
                nb = (current[0] + dx, current[1] + dy)
                if 0 <= nb[0] < 100 and 0 <= nb[1] < 100 and grid[nb[0]][nb[1]] == 0:
                    cost = 1.414 if dx != 0 and dy != 0 else 1.0
                    tg = g_score[current] + cost
                    if tg < g_score.get(nb, float('inf')):
                        came_from[nb] = current; g_score[nb] = tg
                        heapq.heappush(open_set, (tg + math.hypot(goal[0]-nb[0], goal[1]-nb[1]), nb))
        return None

    def _reconstruct_path(self, cf, cur):
        p = [cur]
        while cur in cf: cur = cf[cur]; p.append(cur)
        return p[::-1]

    def _is_line_clear(self, p1, p2, grid):
        steps = int(max(abs(p2[0]-p1[0]), abs(p2[1]-p1[1]) / 0.1)) + 1
        for i in range(steps + 1):
            t = i / max(steps, 1)
            x = p1[0] + t * (p2[0]-p1[0]); y = p1[1] + t * (p2[1]-p1[1])
            ix, iy = int(x/0.1), int(y/0.1)
            if 0 <= iy < 100 and 0 <= ix < 100 and grid[iy][ix] == 1: return False
        return True

    def _prune_path(self, path, grid):
        if len(path) <= 2: return path
        pruned = [path[0]]; i = 0
        while i < len(path) - 2:
            if self._is_line_clear(path[i], path[i+2], grid): i += 1
            else: pruned.append(path[i+1]); i += 1
        pruned.append(path[-1]); return pruned

    def _smooth_path(self, path):
        """main13 Optimization: Enhanced Gradient Smoothing with more iterations."""
        if len(path) <= 2: return path
        path = np.array(path); sm = path.copy()
        # Increased iterations and weight for better convergence to a smooth path
        for _ in range(80): 
            for i in range(1, len(path)-1):
                # Gradient descent towards neighbours and original point
                sm[i] += 0.4 * (path[i] - sm[i]) + 0.4 * (sm[i-1] + sm[i+1] - 2.0*sm[i])
        return sm.tolist()
