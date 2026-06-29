"""Preemptive corridor-aware path scoring.

The original main4 control loop only knew a path was dangerous AFTER the robot
got stuck inside the corridor (Stuck monitor -> EscalatingRecovery). This
module scores the planned path *before* execution and forces an early replan
with bigger inflation when:

  - Any path point is closer than `min_clearance` to an obstacle
  - Two obstacles flank the path with a gap narrower than `gap_min`
  - The lookahead window (next `lookahead_m` of the path) has falling
    clearance worse than `min_clearance`

This is purely geometric — no NMPC predictions, no Kalman, just precomputed
inflation analysis on the existing A* output.
"""

import math


def path_min_clearance(path, obstacles):
    """Return (min_clearance, min_index) over the whole path."""
    if not path or not obstacles:
        return float("inf"), -1
    best_d = float("inf")
    best_i = -1
    for i, (wx, wy) in enumerate(path):
        for ox, oy, orad in obstacles:
            d = math.hypot(wx - ox, wy - oy) - orad
            if d < best_d:
                best_d = d
                best_i = i
    return best_d, best_i


def path_clearance_profile(path, obstacles):
    """Return list of per-waypoint clearance values.

    Each value is the distance from that waypoint to the *surface* of the
    nearest obstacle (negative -> inside obstacle).
    """
    if not path:
        return []
    out = []
    for wx, wy in path:
        best = float("inf")
        for ox, oy, orad in obstacles:
            d = math.hypot(wx - ox, wy - oy) - orad
            if d < best:
                best = d
        out.append(best)
    return out


def lookahead_min_clearance(path, obstacles, robot_xy, lookahead_m=1.5):
    """Walk forward from the path-point nearest `robot_xy` for `lookahead_m`
    meters and return the worst clearance in that window.
    """
    if not path or not obstacles:
        return float("inf")
    # Find nearest path index to the robot
    rx, ry = robot_xy
    best_i = 0
    best_d = float("inf")
    for i, (wx, wy) in enumerate(path):
        d = math.hypot(wx - rx, wy - ry)
        if d < best_d:
            best_d = d
            best_i = i
    # Walk forward accumulating distance, take the worst clearance over it
    acc = 0.0
    worst = float("inf")
    prev = path[best_i]
    for i in range(best_i, len(path)):
        wx, wy = path[i]
        acc += math.hypot(wx - prev[0], wy - prev[1])
        for ox, oy, orad in obstacles:
            d = math.hypot(wx - ox, wy - oy) - orad
            if d < worst:
                worst = d
        if acc >= lookahead_m:
            break
        prev = path[i]
    return worst


def detect_pinch_points(path, obstacles, gap_min=0.80):
    """Return list of (index, gap_width) where the path passes between two
    obstacles whose surface-to-surface gap is below gap_min.
    """
    pinches = []
    if not path or len(obstacles) < 2:
        return pinches
    for i, (wx, wy) in enumerate(path):
        # Find two closest obstacles to this point
        dists = []
        for ox, oy, orad in obstacles:
            d = math.hypot(wx - ox, wy - oy) - orad
            dists.append((d, ox, oy, orad))
        if len(dists) < 2:
            continue
        dists.sort(key=lambda t: t[0])
        d1, x1, y1, r1 = dists[0]
        d2, x2, y2, r2 = dists[1]
        # The two obstacles must be on opposite sides of the path point.
        # Approximate by: the centerline-to-centerline distance minus radii.
        gap = math.hypot(x1 - x2, y1 - y2) - r1 - r2
        if gap < gap_min and d1 < 0.5 and d2 < 0.5:
            pinches.append((i, gap))
    return pinches


def plan_with_clearance(astar, start_xy, goal_xy, obstacles,
                        min_clearance=0.25, gap_min=0.80,
                        max_attempts=3, inflation_step=0.15):
    """Plan a path, then iteratively re-inflate obstacles until per-point
    clearance >= min_clearance and no pinch points worse than gap_min.

    Parameters:
        inflation_step: How much (in meters) to inflate obstacles on each
                        retry attempt. Default 0.15. Increase for tighter
                        corridors.

    Returns (path, info_dict) where info_dict reports attempt count,
    inflation_used, min_clearance, and detected_pinches.
    """
    attempts = 0
    inflation_bonus = 0.0   # how much extra we virtually inflate obstacles
    final_path = None
    final_clearance = float("-inf")
    final_pinches = []

    # Each attempt: virtually inflate obstacles by `inflation_bonus` so A*
    # routes wider. We DO NOT permanently modify the obstacle list — only
    # this attempt sees inflated radii.
    while attempts < max_attempts:
        inflated = [(ox, oy, orad + inflation_bonus) for (ox, oy, orad) in obstacles]
        try:
            path = astar.plan(start_xy, goal_xy, inflated)
        except Exception:
            path = None

        if not path or len(path) < 2:
            # planner failed; fall back to last good attempt or direct
            break

        clearance, _ = path_min_clearance(path, obstacles)   # against ORIGINAL obs
        pinches = detect_pinch_points(path, obstacles, gap_min=gap_min)

        final_path = path
        final_clearance = clearance
        final_pinches = pinches

        if clearance >= min_clearance and not pinches:
            return path, {
                "attempts": attempts + 1,
                "inflation_used": inflation_bonus,
                "min_clearance": clearance,
                "pinches": pinches,
            }

        attempts += 1
        inflation_bonus += inflation_step   # next attempt: `inflation_step` more buffer

    return (final_path or [tuple(goal_xy)]), {
        "attempts": attempts,
        "inflation_used": inflation_bonus,
        "min_clearance": final_clearance,
        "pinches": final_pinches,
    }


class CorridorPredictor:
    """Anticipatory replan trigger.

    `should_replan_now(state, path, obstacles)` returns True when the robot's
    next `lookahead_m` of the planned path drops below the safety threshold.
    Use this BEFORE the robot enters the dangerous segment, so the new path
    is ready by the time it would have been needed.
    """

    def __init__(self, lookahead_m=1.5, min_clearance=0.22, cooldown_s=1.0,
                 dt=0.1):
        self.lookahead_m = float(lookahead_m)
        self.min_clearance = float(min_clearance)
        self.cooldown_ticks = max(1, int(cooldown_s / dt))
        self._last_trigger_age = self.cooldown_ticks + 1
        self.trigger_count = 0

    def tick(self):
        self._last_trigger_age += 1

    def should_replan_now(self, state, path, obstacles):
        if not path:
            return False
        if self._last_trigger_age < self.cooldown_ticks:
            return False
        worst = lookahead_min_clearance(path, obstacles,
                                        (state["x"], state["y"]),
                                        self.lookahead_m)
        if worst < self.min_clearance:
            self._last_trigger_age = 0
            self.trigger_count += 1
            return True
        return False


# ============================================================================
# Dynamic-obstacle prediction layer
# ============================================================================
def predicted_obstacle_sweep(tracker, sim_time, horizon_s=1.5, dt_sample=0.2,
                             inflate=0.10):
    """Return a list of (x, y, r) virtual obstacles that cover the predicted
    trajectory of every dynamic obstacle over the next `horizon_s` seconds.

    Each sample is treated as a static obstacle with radius +inflate added
    so A* routes wider around moving objects. With dt_sample=0.2 and
    horizon=1.5s we add 8 samples per dynamic obstacle.
    """
    if tracker is None:
        return []
    out = []
    steps = max(1, int(horizon_s / dt_sample))
    for k in range(steps + 1):
        dt_ahead = k * dt_sample
        for x, y, r in tracker.predict(sim_time, dt_ahead):
            out.append((x, y, r + inflate))
    return out


class DynamicConflictPredictor:
    """Detects when a moving obstacle's predicted trajectory will cross the
    planned path. Triggers a replan BEFORE the obstacle reaches the path,
    not after the robot gets stopped by it.

    Key difference vs. CorridorPredictor: CorridorPredictor scores static
    geometry. This one scores STATIC path vs. MOVING obstacle predictions.
    """

    def __init__(self, lookahead_path_m=1.5, horizon_s=1.5, dt_sample=0.2,
                 conflict_dist=0.45, cooldown_s=1.2, dt=0.1):
        self.lookahead_path_m = float(lookahead_path_m)
        self.horizon_s = float(horizon_s)
        self.dt_sample = float(dt_sample)
        self.conflict_dist = float(conflict_dist)
        self.cooldown_ticks = max(1, int(cooldown_s / dt))
        self._last_trigger_age = self.cooldown_ticks + 1
        self.trigger_count = 0
        self.last_conflict = None    # (t_ahead, x, y) of detected conflict

    def tick(self):
        self._last_trigger_age += 1

    def should_replan_now(self, state, path, tracker, sim_time):
        """Return True if any predicted dynamic obstacle will close to within
        `conflict_dist` of the robot's lookahead path within `horizon_s`.
        """
        if not path or tracker is None:
            return False
        if self._last_trigger_age < self.cooldown_ticks:
            return False

        # Build the upcoming path window (next lookahead_path_m meters)
        rx, ry = state["x"], state["y"]
        # nearest path index to robot
        best_i, best_d = 0, float("inf")
        for i, (wx, wy) in enumerate(path):
            d = math.hypot(wx - rx, wy - ry)
            if d < best_d:
                best_d, best_i = d, i
        # collect waypoints over the lookahead window
        window = []
        acc = 0.0
        prev = path[best_i]
        for i in range(best_i, len(path)):
            wx, wy = path[i]
            acc += math.hypot(wx - prev[0], wy - prev[1])
            window.append((wx, wy))
            if acc >= self.lookahead_path_m:
                break
            prev = path[i]
        if not window:
            return False

        # Sample dynamic-obstacle positions over the prediction horizon and
        # check if any predicted position is within conflict_dist of any
        # upcoming waypoint.
        steps = max(1, int(self.horizon_s / self.dt_sample))
        for k in range(steps + 1):
            dt_ahead = k * self.dt_sample
            for x, y, r in tracker.predict(sim_time, dt_ahead):
                threshold = self.conflict_dist + r
                for wx, wy in window:
                    if math.hypot(wx - x, wy - y) < threshold:
                        self._last_trigger_age = 0
                        self.trigger_count += 1
                        self.last_conflict = (dt_ahead, x, y)
                        return True
        return False


def los_safe(state, goal_xy, obstacles, robot_radius=0.30, safety=0.20,
             samples_per_m=8):
    """Strict line-of-sight check. The straight segment robot->goal must keep
    every sample point at least (robot_radius + safety) from any obstacle.

    Used to gate the A* LOS shortcut — previously the shortcut used a fixed
    0.4 m buffer that was just barely larger than the obstacle radius, so the
    robot would dive into gaps it couldn't actually fit through.
    """
    sx, sy = state["x"], state["y"]
    gx, gy = goal_xy
    seg = math.hypot(gx - sx, gy - sy)
    if seg < 1e-3:
        return True
    n = max(4, int(seg * samples_per_m))
    needed = robot_radius + safety
    for k in range(0, n + 1):
        t = k / n
        px = sx + t * (gx - sx)
        py = sy + t * (gy - sy)
        for ox, oy, orad in obstacles:
            if math.hypot(px - ox, py - oy) < orad + needed:
                return False
    return True
