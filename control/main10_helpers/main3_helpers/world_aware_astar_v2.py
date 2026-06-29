"""World-aware A* on [-5, 5] x [-5, 5].

This is a safer second version of the planner used by the simulation.
It keeps the original coordinate handling but rejects any smoothed path that
re-enters an occupied grid cell, which matters in tight corridor layouts.

Public API identical to AStarPlanner:
    plan(start_xy, goal_xy, obstacles) -> List[(x, y)]
where each obstacle is (x, y, radius).
"""

import heapq
import math
import numpy as np


class WorldAwareAStarV2:
    def __init__(self,
                 x_min=-5.0, x_max=5.0,
                 y_min=-5.0, y_max=5.0,
                 resolution=0.05,
                 robot_radius=0.30):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.resolution = resolution
        self.robot_radius = robot_radius
        self.cols = int(round((x_max - x_min) / resolution))
        self.rows = int(round((y_max - y_min) / resolution))
        self._warned_oob = False

    def _world_to_grid(self, x, y, clamp=True):
        col = int((x - self.x_min) / self.resolution)
        row = int((y - self.y_min) / self.resolution)
        if clamp:
            col = max(0, min(self.cols - 1, col))
            row = max(0, min(self.rows - 1, row))
        return col, row

    def _grid_to_world(self, col, row):
        x = self.x_min + (col + 0.5) * self.resolution
        y = self.y_min + (row + 0.5) * self.resolution
        return x, y

    def _in_bounds(self, x, y):
        return (self.x_min <= x <= self.x_max
                and self.y_min <= y <= self.y_max)

    def plan(self, start_xy, goal_xy, obstacles):
        if not self._in_bounds(*goal_xy) and not self._warned_oob:
            print(f"[WorldAwareAStarV2] warning: goal ({goal_xy[0]:.2f},{goal_xy[1]:.2f}) "
                  f"outside grid [{self.x_min},{self.x_max}]; clamped.")
            self._warned_oob = True

        grid = np.zeros((self.rows, self.cols), dtype=np.uint8)

        for obs in obstacles:
            ox, oy = obs[0], obs[1]
            o_r = obs[2] if len(obs) > 2 else 0.0
            total_r = self.robot_radius + o_r
            inf_cells = int(math.ceil(total_r / self.resolution))
            cc, cr = self._world_to_grid(ox, oy, clamp=False)
            if (cc < -inf_cells or cc >= self.cols + inf_cells
                    or cr < -inf_cells or cr >= self.rows + inf_cells):
                continue
            for dc in range(-inf_cells, inf_cells + 1):
                for dr in range(-inf_cells, inf_cells + 1):
                    if dc * dc + dr * dr > inf_cells * inf_cells:
                        continue
                    nc = cc + dc
                    nr = cr + dr
                    if 0 <= nc < self.cols and 0 <= nr < self.rows:
                        grid[nr, nc] = 1

        sc, sr = self._world_to_grid(start_xy[0], start_xy[1])
        gc, gr = self._world_to_grid(goal_xy[0], goal_xy[1])

        if grid[sr, sc] == 1:
            grid[max(0, sr - 1):min(self.rows, sr + 2),
                 max(0, sc - 1):min(self.cols, sc + 2)] = 0
        if grid[gr, gc] == 1:
            grid[max(0, gr - 1):min(self.rows, gr + 2),
                 max(0, gc - 1):min(self.cols, gc + 2)] = 0

        path_cells = self._a_star_search(grid, (sc, sr), (gc, gr))
        if path_cells is None:
            return [tuple(goal_xy)]

        waypoints = [self._grid_to_world(c, r) for (c, r) in path_cells]
        pruned = self._prune_path(waypoints, grid)
        smoothed = self._smooth_path(pruned)
        if self._path_clear(smoothed, grid):
            return smoothed
        return pruned

    def _a_star_search(self, grid, start, goal):
        if start == goal:
            return [start]
        if grid[start[1], start[0]] == 1 or grid[goal[1], goal[0]] == 1:
            return None

        open_set = []
        heapq.heappush(open_set, (0.0, start))
        came_from = {}
        g_score = {start: 0.0}

        neighbours = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                      (1, 1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (-1, -1, 1.414)]

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                return self._reconstruct(came_from, current)
            cx, cy = current
            for dc, dr, cost in neighbours:
                nc, nr = cx + dc, cy + dr
                if not (0 <= nc < self.cols and 0 <= nr < self.rows):
                    continue
                if grid[nr, nc] == 1:
                    continue
                tentative = g_score[current] + cost
                if tentative < g_score.get((nc, nr), float("inf")):
                    came_from[(nc, nr)] = current
                    g_score[(nc, nr)] = tentative
                    h = math.hypot(goal[0] - nc, goal[1] - nr)
                    heapq.heappush(open_set, (tentative + h, (nc, nr)))
        return None

    def _reconstruct(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]

    def _line_clear(self, p1, p2, grid):
        n = int(max(abs(p2[0] - p1[0]), abs(p2[1] - p1[1])) / self.resolution) + 1
        for i in range(n + 1):
            t = i / max(n, 1)
            x = p1[0] + t * (p2[0] - p1[0])
            y = p1[1] + t * (p2[1] - p1[1])
            c, r = self._world_to_grid(x, y)
            if grid[r, c] == 1:
                return False
        return True

    def _prune_path(self, path, grid):
        if len(path) <= 2:
            return list(path)
        pruned = [path[0]]
        i = 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1:
                if self._line_clear(path[i], path[j], grid):
                    break
                j -= 1
            pruned.append(path[j])
            i = j
        return pruned

    def _path_clear(self, path, grid):
        if not path or len(path) < 2:
            return True
        for p1, p2 in zip(path[:-1], path[1:]):
            if not self._line_clear(p1, p2, grid):
                return False
        return True

    def _smooth_path(self, path, iterations=20, alpha=0.3, beta=0.3):
        if len(path) <= 2:
            return [list(p) for p in path]
        arr = np.array(path, dtype=float)
        sm = arr.copy()
        for _ in range(iterations):
            for k in range(1, len(arr) - 1):
                sm[k] += alpha * (arr[k] - sm[k]) + beta * (sm[k - 1] + sm[k + 1] - 2.0 * sm[k])
        return sm.tolist()
