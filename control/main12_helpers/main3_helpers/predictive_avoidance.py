"""Predictive collision avoidance + anti-oscillation guard.

Implements the four collision-avoidance principles cited in production-grade
mobile-robotics literature:

  (1) Environmental sensors      -- already done in perception_fusion.py
  (2) Sensor fusion & AI         -- already done in perception_fusion.py
  (3) Motion planning evaluating MULTIPLE candidate paths
      -> `MultiPathPlanner`: tries direct + lateral-detour candidates and
         picks the lowest-energy + lowest-risk one.
  (4) Predictive 3D collision check by forward-simulating the robot footprint
      -> `TrajectoryPredictor`: integrates the unicycle model 1.5 s ahead at
         the commanded (v, omega) and reports whether the swept volume hits
         any static or predicted-dynamic obstacle. Brakes the command before
         the collision happens, not after.

Plus the anti-oscillation guard the dashboard screenshot motivated:

  `OscillationDetector`: flags when v or omega flip sign back-and-forth more
  than `flip_threshold` times within a 1 s window, then commits to the
  median sign for `commit_s` seconds. Eliminates the dithering that burns
  energy without making progress.

Everything is pure Python + math; no extra dependencies.
"""

import math


# ============================================================================
# Trajectory prediction (principle #4)
# ============================================================================
class TrajectoryPredictor:
    """Forward-simulates the robot under the commanded (v, omega) and reports
    whether the swept footprint will collide with any obstacle within
    `horizon_s` seconds. Pure unicycle kinematics, no actuator dynamics.

    Usage:
        ok, t_hit, _ = predictor.check(state, v_cmd, w_cmd, obstacles)
        if not ok:
            v_cmd, w_cmd = predictor.brake(v_cmd, w_cmd, t_hit)
    """

    def __init__(self, robot_radius=0.30, horizon_s=1.5, dt=0.1,
                 safety_margin=0.05):
        self.robot_radius = float(robot_radius)
        self.horizon_s = float(horizon_s)
        self.dt = float(dt)
        self.safety_margin = float(safety_margin)

    def rollout(self, state, v_cmd, w_cmd):
        """Yield (t, x, y, theta) along the predicted trajectory."""
        x, y, th = state["x"], state["y"], state["theta"]
        t = 0.0
        yield t, x, y, th
        steps = max(1, int(self.horizon_s / self.dt))
        for _ in range(steps):
            x += v_cmd * math.cos(th) * self.dt
            y += v_cmd * math.sin(th) * self.dt
            th += w_cmd * self.dt
            t += self.dt
            yield t, x, y, th

    def check(self, state, v_cmd, w_cmd, obstacles):
        """Returns (collision_free, t_hit_or_None, hit_obs_or_None)."""
        if not obstacles:
            return True, None, None
        margin = self.robot_radius + self.safety_margin
        for t, x, y, _ in self.rollout(state, v_cmd, w_cmd):
            if t == 0.0:
                continue
            for (ox, oy, orad) in obstacles:
                if math.hypot(x - ox, y - oy) < orad + margin:
                    return False, t, (ox, oy, orad)
        return True, None, None

    def time_to_collision(self, state, v_cmd, w_cmd, obstacles):
        """Returns t (seconds) until first collision under (v,w), or
        horizon_s if no collision."""
        ok, t_hit, _ = self.check(state, v_cmd, w_cmd, obstacles)
        return self.horizon_s if ok else (t_hit or 0.0)

    @staticmethod
    def brake(v_cmd, w_cmd, t_hit, t_safe=0.6):
        """Scale (v, w) so the trajectory needs `t_safe` instead of `t_hit`
        seconds to reach the same point.  At t_hit=0 we stop completely.
        """
        if t_hit <= 0.05:
            return 0.0, 0.0
        scale = max(0.0, min(1.0, t_hit / t_safe))
        return v_cmd * scale, w_cmd * scale


# ============================================================================
# Multi-candidate path planner (principle #3)
# ============================================================================
def _path_length(path):
    if not path or len(path) < 2:
        return 0.0
    return sum(math.hypot(path[i + 1][0] - path[i][0],
                          path[i + 1][1] - path[i][1])
               for i in range(len(path) - 1))


def _min_clearance(path, obstacles):
    if not path or not obstacles:
        return float("inf")
    best = float("inf")
    for (wx, wy) in path:
        for (ox, oy, orad) in obstacles:
            d = math.hypot(wx - ox, wy - oy) - orad
            if d < best:
                best = d
    return best


class MultiPathPlanner:
    """Wraps an A*-like planner. For each plan request, generates several
    candidate goals (direct, lateral-left detour, lateral-right detour) and
    keeps the one with the best (length × risk_penalty) cost.

    cost = length + α · penalty(min_clearance)
    where penalty(c) = (c_target - c)² · w if c < c_target else 0.
    """

    def __init__(self, base_planner, robot_radius=0.30,
                 detour_offset=0.6, c_target=0.30,
                 weight_clearance=4.0):
        self.planner = base_planner
        self.robot_radius = float(robot_radius)
        self.detour_offset = float(detour_offset)
        self.c_target = float(c_target)
        self.w_clearance = float(weight_clearance)

    def _candidates(self, start, goal):
        """Generate 3 intermediate-waypoint candidates: direct, +lateral
        detour, -lateral detour. Each is a planner call with that point as
        an intermediate via-target."""
        gx, gy = goal
        sx, sy = start
        dx, dy = gx - sx, gy - sy
        L = math.hypot(dx, dy)
        if L < 1e-3:
            return [None]
        # unit perpendicular to start->goal direction
        px, py = -dy / L, dx / L
        midx, midy = (sx + gx) / 2.0, (sy + gy) / 2.0
        cands = [None,
                 (midx + self.detour_offset * px, midy + self.detour_offset * py),
                 (midx - self.detour_offset * px, midy - self.detour_offset * py)]
        return cands

    def plan(self, start, goal, obstacles):
        """Returns (best_path, info_dict)."""
        best_path = None
        best_cost = float("inf")
        best_info = {"candidate": None,
                     "length": float("inf"),
                     "clearance": -float("inf")}

        for cand in self._candidates(start, goal):
            try:
                if cand is None:
                    path = self.planner.plan(start, goal, obstacles)
                else:
                    # Two-segment plan: start -> cand -> goal
                    p1 = self.planner.plan(start, cand, obstacles)
                    p2 = self.planner.plan(cand, goal, obstacles)
                    if not p1 or not p2:
                        continue
                    path = list(p1) + list(p2)[1:]   # avoid duplicate
            except Exception:
                continue
            if not path or len(path) < 2:
                continue
            length = _path_length(path)
            clr = _min_clearance(path, obstacles)
            penalty = 0.0
            if clr < self.c_target:
                penalty = self.w_clearance * (self.c_target - clr) ** 2
            cost = length + penalty
            if cost < best_cost:
                best_cost = cost
                best_path = path
                best_info = {"candidate": cand,
                             "length": length,
                             "clearance": clr,
                             "cost": cost}

        return best_path or [tuple(goal)], best_info


# ============================================================================
# Oscillation detector (anti-dither)
# ============================================================================
class OscillationDetector:
    """Detects when (v, omega) flip sign rapidly — the energy-wasting
    "ping-pong" failure mode visible in the dashboard.

    Strategy:
      - Track the sign history of v and omega over a sliding window.
      - If v flips sign >= flip_threshold_v times in `window_s`, commit
        to the median sign for `commit_s` seconds (suppress further flips).
      - Same for omega independently.

    Reports v_committed and w_committed back via filter().
    """

    def __init__(self, window_s=1.0, dt=0.1,
                 flip_threshold_v=3, flip_threshold_w=4,
                 commit_s=0.8):
        self.window = max(2, int(window_s / dt))
        self.commit_ticks = max(1, int(commit_s / dt))
        self.dt = float(dt)
        self.flip_v = int(flip_threshold_v)
        self.flip_w = int(flip_threshold_w)
        self.v_hist = []
        self.w_hist = []
        self.v_lock = 0     # ticks remaining; 0 = not locked
        self.w_lock = 0
        self.v_lock_sign = 0
        self.w_lock_sign = 0
        self.events_v = 0
        self.events_w = 0

    def _sign(self, x, eps=0.02):
        if x > eps:  return 1
        if x < -eps: return -1
        return 0

    def _count_flips(self, hist):
        flips = 0
        prev = 0
        for s in hist:
            if s == 0: continue
            if prev != 0 and s != prev:
                flips += 1
            prev = s
        return flips

    def filter(self, v_cmd, w_cmd):
        """Returns (v_out, w_out, log_dict).  Caller passes the controller's
        proposed (v, omega); we return a (possibly committed) version."""
        # Decrement locks
        if self.v_lock > 0:  self.v_lock -= 1
        if self.w_lock > 0:  self.w_lock -= 1

        # Update history
        self.v_hist.append(self._sign(v_cmd))
        self.w_hist.append(self._sign(w_cmd))
        if len(self.v_hist) > self.window: self.v_hist.pop(0)
        if len(self.w_hist) > self.window: self.w_hist.pop(0)

        # Detect new oscillation
        if self.v_lock == 0:
            flips_v = self._count_flips(self.v_hist)
            if flips_v >= self.flip_v:
                # Commit to the MAJORITY non-zero sign in the window
                ones = sum(1 for s in self.v_hist if s == 1)
                neg = sum(1 for s in self.v_hist if s == -1)
                if ones > neg:
                    self.v_lock_sign = 1
                elif neg > ones:
                    self.v_lock_sign = -1
                else:
                    self.v_lock_sign = self._sign(v_cmd) or 1
                self.v_lock = self.commit_ticks
                self.events_v += 1

        if self.w_lock == 0:
            flips_w = self._count_flips(self.w_hist)
            if flips_w >= self.flip_w:
                ones = sum(1 for s in self.w_hist if s == 1)
                neg = sum(1 for s in self.w_hist if s == -1)
                if ones > neg:
                    self.w_lock_sign = 1
                elif neg > ones:
                    self.w_lock_sign = -1
                else:
                    self.w_lock_sign = self._sign(w_cmd) or 1
                self.w_lock = self.commit_ticks
                self.events_w += 1

        # Apply commit: if locked, force the magnitude to the same sign
        v_out, w_out = v_cmd, w_cmd
        if self.v_lock > 0:
            # Force v to be at least 0 with the locked sign — if user asked
            # for the OPPOSITE sign, suppress to 0 to avoid further flip.
            if self._sign(v_cmd) == -self.v_lock_sign:
                v_out = 0.0
            elif self._sign(v_cmd) == 0:
                # Coast forward in the committed direction at low speed
                v_out = 0.15 * self.v_lock_sign
        if self.w_lock > 0:
            if self._sign(w_cmd) == -self.w_lock_sign:
                w_out = 0.0
            elif self._sign(w_cmd) == 0:
                w_out = 0.6 * self.w_lock_sign

        return v_out, w_out, {
            "v_locked": self.v_lock > 0,
            "w_locked": self.w_lock > 0,
            "events_v": self.events_v,
            "events_w": self.events_w,
        }
