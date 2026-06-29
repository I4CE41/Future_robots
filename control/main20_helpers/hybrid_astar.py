"""Kinodynamic Hybrid-A* global planner.

Grid A* plans on cell centres and ignores the robot's turning radius, so its
paths contain sharp corners that the local layer must smooth away. Hybrid-A*
searches in continuous ``(x, y, theta)`` space by expanding a small set of
constant-curvature motion primitives (left / straight / right arcs, forward
only), so the returned path is kinematically feasible for a differential-drive
base: smoother, with fewer pinch points feeding the local planner.

It exposes ``plan(start_xy, goal_xy, obstacles)`` returning a list of
``(x, y)`` waypoints (theta stripped), matching the WorldAwareAStar interface
so it is a drop-in for the global-plan step.
"""

import heapq
import math


class HybridAStarPlanner:
    def __init__(self, x_bounds=(-5.0, 5.0), y_bounds=(-5.0, 5.0),
                 robot_radius=0.30, safety_margin=0.18,
                 arc_length=0.35, steer_angles=(-0.6, -0.3, 0.0, 0.3, 0.6),
                 xy_resolution=0.20, theta_resolution=math.radians(15.0),
                 goal_tol=0.30, max_expansions=20000):
        self.x_min, self.x_max = x_bounds
        self.y_min, self.y_max = y_bounds
        self.robot_radius = float(robot_radius)
        self.safety_margin = float(safety_margin)
        self.arc_length = float(arc_length)
        self.steer_angles = tuple(steer_angles)
        self.xy_res = float(xy_resolution)
        self.theta_res = float(theta_resolution)
        self.goal_tol = float(goal_tol)
        self.max_expansions = int(max_expansions)

    def _disc(self, x, y, theta):
        ix = int(round((x - self.x_min) / self.xy_res))
        iy = int(round((y - self.y_min) / self.xy_res))
        ith = int(round(theta / self.theta_res)) % int(round(2 * math.pi / self.theta_res))
        return (ix, iy, ith)

    def _in_bounds(self, x, y):
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max

    def _collision(self, x, y, obstacles):
        if not self._in_bounds(x, y):
            return True
        for ox, oy, orad in obstacles:
            if math.hypot(x - ox, y - oy) <= orad + self.robot_radius + self.safety_margin:
                return True
        return False

    def _expand(self, x, y, theta, obstacles):
        """Yield (nx, ny, ntheta, step_cost) for each feasible primitive."""
        results = []
        n_sub = 4
        ds = self.arc_length / n_sub
        for steer in self.steer_angles:
            cx, cy, cth = x, y, theta
            ok = True
            for _ in range(n_sub):
                cth = cth + steer * ds
                cx = cx + ds * math.cos(cth)
                cy = cy + ds * math.sin(cth)
                if self._collision(cx, cy, obstacles):
                    ok = False
                    break
            if not ok:
                continue
            # Penalise turning and (mildly) keep arcs uniform.
            step_cost = self.arc_length + 0.5 * abs(steer) * self.arc_length
            results.append((cx, cy, cth, step_cost))
        return results

    def plan(self, start_xy, goal_xy, obstacles):
        obstacles = list(obstacles) if obstacles else []
        sx, sy = float(start_xy[0]), float(start_xy[1])
        gx, gy = float(goal_xy[0]), float(goal_xy[1])
        start_theta = math.atan2(gy - sy, gx - sx)

        start_key = self._disc(sx, sy, start_theta)
        open_heap = []
        g_score = {start_key: 0.0}
        came_from = {}
        node_state = {start_key: (sx, sy, start_theta)}
        h0 = math.hypot(gx - sx, gy - sy)
        heapq.heappush(open_heap, (h0, 0.0, start_key))

        closed = set()
        expansions = 0
        best_key = start_key
        best_dist = h0

        while open_heap and expansions < self.max_expansions:
            _, g, key = heapq.heappop(open_heap)
            if key in closed:
                continue
            closed.add(key)
            x, y, theta = node_state[key]

            d_goal = math.hypot(gx - x, gy - y)
            if d_goal < best_dist:
                best_dist = d_goal
                best_key = key
            if d_goal <= self.goal_tol:
                best_key = key
                break

            expansions += 1
            for nx, ny, nth, step_cost in self._expand(x, y, theta, obstacles):
                nkey = self._disc(nx, ny, nth)
                if nkey in closed:
                    continue
                tentative = g + step_cost
                if tentative < g_score.get(nkey, float("inf")):
                    g_score[nkey] = tentative
                    came_from[nkey] = key
                    node_state[nkey] = (nx, ny, nth)
                    f = tentative + math.hypot(gx - nx, gy - ny)
                    heapq.heappush(open_heap, (f, tentative, nkey))

        # Reconstruct from best_key.
        path = []
        key = best_key
        while key in came_from:
            x, y, _ = node_state[key]
            path.append((x, y))
            key = came_from[key]
        path.append((sx, sy))
        path.reverse()

        # Always make sure the goal is the final waypoint.
        if not path or math.hypot(path[-1][0] - gx, path[-1][1] - gy) > 1e-3:
            path.append((gx, gy))
        return path
