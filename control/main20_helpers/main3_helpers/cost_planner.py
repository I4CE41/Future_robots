"""Cost-aware A* wrapper with event-driven replanning.

Plans on CURRENT observed obstacles only. Dynamic obstacles are handled by the
local reaction layer (the FSM in main3.py), not by inflating future positions
into the global plan — that approach carpeted the workspace with phantom
obstacles and forced the planner to pick wrong-direction paths.

Wraps any planner with a `.plan(start, goal, obstacles)` interface — works with
both the legacy AStarPlanner and the new WorldAwareAStar.
"""

import math


class CostAwarePlanner:
    def __init__(self, base_planner, energy_policy, config,
                 dynamic_tracker=None, commit_seconds=1.5):
        self.base = base_planner
        self.policy = energy_policy
        self.cfg = config
        self.dyn = dynamic_tracker  # kept only for needs_replan conflict check
        self.commit_seconds = float(commit_seconds)
        self.v_ref = float(config.get("capp", {}).get("v_ref", 0.5))

        self.current_path = []
        self.last_obstacle_count = 0
        self.last_cost = float("inf")
        self.last_goal = None
        self.last_phase = None
        self.last_sim_time = 0.0
        self.commit_until = 0.0

    # ---------------------------------------------------------- scoring
    @staticmethod
    def _path_length(path):
        if not path or len(path) < 2:
            return 0.0
        total = 0.0
        for i in range(len(path) - 1):
            total += math.hypot(path[i + 1][0] - path[i][0],
                                path[i + 1][1] - path[i][1])
        return total

    @staticmethod
    def _min_clearance(point, obstacles):
        if not obstacles:
            return 5.0
        best = float("inf")
        for ox, oy, r in obstacles:
            d = math.hypot(point[0] - ox, point[1] - oy) - r
            if d < best:
                best = d
        return max(best, 0.01)

    def _path_risk(self, path, obstacles):
        if not path:
            return 0.0
        s = 0.0
        for wp in path:
            s += 1.0 / (self._min_clearance(wp, obstacles) + 0.05)
        return s / max(1, len(path))

    def score(self, path, obstacles, soc):
        if not path:
            return float("inf")
        w = self.policy.planner_weights(soc)
        length = self._path_length(path)
        time_cost = length / max(0.05, self.v_ref)
        energy_cost = self.policy.estimate_energy_for(length)
        risk = self._path_risk(path, obstacles)
        return w["w_t"] * time_cost + w["w_e"] * energy_cost * 10.0 + w["w_r"] * risk

    # ---------------------------------------------------------- planning
    def _candidate(self, start_xy, goal_xy, obstacles, inflate):
        if inflate > 0.0:
            inflated = [(ox, oy, r + inflate) for ox, oy, r in obstacles]
        else:
            inflated = obstacles
        try:
            path = self.base.plan(start_xy, goal_xy, inflated)
        except Exception:
            path = None
        if not path or len(path) < 2:
            return [tuple(goal_xy)]
        return [(float(x), float(y)) for x, y in path]

    def plan(self, start_xy, goal_xy, obstacles, soc, sim_time=None,
             num_candidates=3, min_inflate=0.0):
        """Pick the lowest-cost path among candidates with varying static inflation.

        `min_inflate` shifts every candidate inflation UP — used when the progress
        monitor reports the robot is stuck. Forces wider berths around obstacles.
        """
        base_inflations = [0.0, 0.10, 0.20][:max(1, num_candidates)]
        inflations = [i + min_inflate for i in base_inflations]
        candidates = [self._candidate(start_xy, goal_xy, obstacles, dr)
                      for dr in inflations]
        scored = [(self.score(c, obstacles, soc), c) for c in candidates]
        scored.sort(key=lambda x: x[0])
        best_cost, best_path = scored[0]

        self.current_path = best_path
        self.last_cost = best_cost
        self.last_goal = tuple(goal_xy)
        self.last_obstacle_count = len(obstacles)
        self.last_sim_time = sim_time if sim_time is not None else 0.0
        if sim_time is not None:
            self.commit_until = sim_time + self.commit_seconds
        return best_path

    # ---------------------------------------------------------- replan trigger
    def needs_replan(self, state, goal_xy, obstacles, soc, phase, sim_time=None):
        if not self.current_path:
            return True
        if phase != self.last_phase:
            self.last_phase = phase
            return True
        if self.last_goal is None or math.hypot(goal_xy[0] - self.last_goal[0],
                                                goal_xy[1] - self.last_goal[1]) > 0.4:
            return True
        end = self.current_path[-1]
        if math.hypot(state["x"] - end[0], state["y"] - end[1]) < 0.25:
            return True
        within_commit = (sim_time is not None and sim_time < self.commit_until)
        if within_commit:
            return False
        if abs(len(obstacles) - self.last_obstacle_count) >= 2:
            return True
        live_cost = self.score(self.current_path, obstacles, soc)
        if live_cost > self.last_cost * 2.0 + 1e-3:
            return True
        return False
