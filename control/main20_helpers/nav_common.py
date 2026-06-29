"""main20 navigation infrastructure and pluggable local-planner interface.

This module replicates the navigation infrastructure that lived inside
``main12.py`` (LocalCostmap, corridor memories, path scoring) so that
``main20.py`` is fully isolated from main12, and adds a small
``LocalPlanner`` dispatch layer so the navigation control tick can call any
of several interchangeable local planners:

  * ``BaselinePursuitPlanner`` -- the main12 nav behaviour (EnergyAwarePursuit
    + ReactiveAvoider + clearance/corridor speed scaling). Reproduces the
    baseline without importing main12.
  * ``DWAPlannerAdapter``       -- wraps the Dynamic Window planner.
  * (MPPI and NMPC-nav planners live in their own modules and subclass
    ``LocalPlanner`` from here.)

The scoring / costmap / memory helpers are copied verbatim from main12 so the
baseline path-selection policy is identical.
"""

import json
import math
import os
import time

import numpy as np

from main9 import EnergyAwarePursuit, ReactiveAvoider, densify
from robot_rescue.control.main20_helpers.main3_helpers.path_clearance import (
    detect_pinch_points,
    lookahead_min_clearance,
    path_min_clearance,
)


# ---------------------------------------------------------------------------
# Local costmap + corridor cross-section (copied from main12)
# ---------------------------------------------------------------------------
class LocalCostmap:
    """Local robot-centric occupancy grid built from sensed and predicted obstacles."""

    def __init__(self, robot_radius=0.30, safety_margin=0.18, radius=2.6, resolution=0.10):
        self.robot_radius = float(robot_radius)
        self.safety_margin = float(safety_margin)
        self.inflate_radius = self.robot_radius + self.safety_margin
        self.radius = float(radius)
        self.resolution = float(resolution)
        self.size = int(math.ceil((2.0 * self.radius) / self.resolution)) + 1
        self.grid = np.zeros((self.size, self.size), dtype=np.uint8)
        self.origin = (0.0, 0.0)
        self.inflated_obstacles = []

    def build(self, center_xy, obstacles):
        self.origin = (float(center_xy[0]) - self.radius, float(center_xy[1]) - self.radius)
        self.grid.fill(0)
        self.inflated_obstacles = [(ox, oy, orad + self.inflate_radius) for ox, oy, orad in obstacles]
        for ox, oy, orad in self.inflated_obstacles:
            x0 = max(0, int(math.floor((ox - orad - self.origin[0]) / self.resolution)))
            x1 = min(self.size - 1, int(math.ceil((ox + orad - self.origin[0]) / self.resolution)))
            y0 = max(0, int(math.floor((oy - orad - self.origin[1]) / self.resolution)))
            y1 = min(self.size - 1, int(math.ceil((oy + orad - self.origin[1]) / self.resolution)))
            for ix in range(x0, x1 + 1):
                px = self.origin[0] + ix * self.resolution
                for iy in range(y0, y1 + 1):
                    py = self.origin[1] + iy * self.resolution
                    if math.hypot(px - ox, py - oy) <= orad:
                        self.grid[iy, ix] = 1
        return self

    def occupied_at(self, x, y):
        ix = int(round((float(x) - self.origin[0]) / self.resolution))
        iy = int(round((float(y) - self.origin[1]) / self.resolution))
        if ix < 0 or iy < 0 or ix >= self.size or iy >= self.size:
            return True
        return bool(self.grid[iy, ix])

    def corridor_cross_section(self, center_xy, direction_xy, span=2.0):
        dx, dy = float(direction_xy[0]), float(direction_xy[1])
        norm = math.hypot(dx, dy)
        if norm < 1e-6:
            return None
        dx /= norm
        dy /= norm
        perp = (-dy, dx)
        samples = np.arange(-span, span + self.resolution, self.resolution)
        free = []
        for offset in samples:
            px = float(center_xy[0]) + perp[0] * offset
            py = float(center_xy[1]) + perp[1] * offset
            free.append(not self.occupied_at(px, py))

        intervals = []
        start = None
        for idx, is_free in enumerate(free):
            offset = float(samples[idx])
            if is_free and start is None:
                start = offset
            elif not is_free and start is not None:
                intervals.append((start, float(samples[idx - 1])))
                start = None
        if start is not None:
            intervals.append((start, float(samples[-1])))

        if not intervals:
            return None

        chosen = None
        for interval in intervals:
            if interval[0] <= 0.0 <= interval[1]:
                chosen = interval
                break
        if chosen is None:
            chosen = min(intervals, key=lambda interval: abs((interval[0] + interval[1]) * 0.5))

        left_boundary, right_boundary = chosen
        width = right_boundary - left_boundary
        center_offset = 0.5 * (left_boundary + right_boundary)
        left_clearance = max(0.0, -left_boundary)
        right_clearance = max(0.0, right_boundary)
        confidence = min(1.0, max(0.0, width / max(span * 2.0, self.resolution)))
        return {
            "left_boundary": left_boundary,
            "right_boundary": right_boundary,
            "left_clearance": left_clearance,
            "right_clearance": right_clearance,
            "width": width,
            "center_offset": center_offset,
            "confidence": confidence,
        }


class CorridorMemory:
    """Hysteresis for corridor choice so we do not oscillate between openings."""

    def __init__(self, hysteresis=0.20, smooth=0.35):
        self.hysteresis = float(hysteresis)
        self.smooth = float(smooth)
        self.center_offset = 0.0
        self.width = None
        self.last_width = None

    def update(self, report):
        if not report:
            return self.center_offset
        self.last_width = self.width
        new_width = float(report.get("width", 0.0))
        new_offset = float(report.get("center_offset", 0.0))
        if self.width is None:
            self.center_offset = new_offset
            self.width = new_width
            return self.center_offset
        if new_width > self.width + self.hysteresis:
            self.center_offset = new_offset
            self.width = new_width
            return self.center_offset
        self.center_offset = (1.0 - self.smooth) * self.center_offset + self.smooth * new_offset
        self.width = max(self.width * 0.98, new_width)
        return self.center_offset

    def off_center(self, report):
        if not report:
            return False
        width = float(report.get("width", 0.0))
        center_offset = abs(float(report.get("center_offset", 0.0)))
        if width < 0.90:
            return True
        return center_offset > max(0.18, 0.20 * width)


class CorridorBlacklist:
    """Short-term memory for corridors that have just caused contact or recovery."""

    def __init__(self, ttl_s=18.0, dt=0.1, offset_bucket=0.15, width_bucket=0.20):
        self.ttl_ticks = max(1, int(ttl_s / dt))
        self.offset_bucket = float(offset_bucket)
        self.width_bucket = float(width_bucket)
        self.entries = {}

    def tick(self):
        expired = []
        for key in list(self.entries):
            self.entries[key] -= 1
            if self.entries[key] <= 0:
                expired.append(key)
        for key in expired:
            self.entries.pop(key, None)

    def _key(self, report):
        if not report:
            return None
        offset = float(report.get("center_offset", 0.0))
        width = float(report.get("width", 0.0))
        return (
            round(offset / self.offset_bucket),
            round(width / self.width_bucket),
        )

    def mark_bad(self, report):
        key = self._key(report)
        if key is None:
            return
        self.entries[key] = self.ttl_ticks

    def is_bad(self, report):
        key = self._key(report)
        return key in self.entries if key is not None else False

    def penalty(self, report):
        return 6.0 if self.is_bad(report) else 0.0


class PersistentCorridorMemory:
    def __init__(self, memory_path, offset_bucket=0.15, width_bucket=0.20):
        self.memory_path = memory_path
        self.offset_bucket = float(offset_bucket)
        self.width_bucket = float(width_bucket)
        self.data = {"bad_corridors": []}
        self.load()

    def load(self):
        if not os.path.exists(self.memory_path):
            self.save()
            return
        try:
            with open(self.memory_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self.data = data
            else:
                self.data = {"bad_corridors": []}
        except Exception:
            self.data = {"bad_corridors": []}
        self.data.setdefault("bad_corridors", [])

    def save(self):
        try:
            with open(self.memory_path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2)
        except Exception:
            pass

    def reset(self):
        self.data = {"bad_corridors": []}
        self.save()

    def _key(self, report):
        if not report:
            return None
        offset = float(report.get("center_offset", 0.0))
        width = float(report.get("width", 0.0))
        return (
            round(offset / self.offset_bucket),
            round(width / self.width_bucket),
        )

    def mark_bad(self, report, context="collision"):
        key = self._key(report)
        if key is None:
            return
        if any(entry.get("key") == key for entry in self.data["bad_corridors"]):
            return
        entry = {
            "key": key,
            "offset": float(report.get("center_offset", 0.0)),
            "width": float(report.get("width", 0.0)),
            "confidence": float(report.get("confidence", 0.0)),
            "context": context,
            "timestamp": int(time.time()),
        }
        self.data["bad_corridors"].append(entry)
        self.save()

    def is_bad(self, report):
        key = self._key(report)
        if key is None:
            return False
        return any(entry.get("key") == key for entry in self.data["bad_corridors"])

    def penalty(self, report):
        return 8.0 if self.is_bad(report) else 0.0


# ---------------------------------------------------------------------------
# Path scoring helpers (copied from main12)
# ---------------------------------------------------------------------------
def path_length(path):
    if not path or len(path) < 2:
        return 0.0
    total = 0.0
    prev = path[0]
    for point in path[1:]:
        total += math.hypot(point[0] - prev[0], point[1] - prev[1])
        prev = point
    return total


def path_turn_cost(path):
    if not path or len(path) < 3:
        return 0.0
    total = 0.0
    prev_vec = None
    for i in range(1, len(path)):
        vx = path[i][0] - path[i - 1][0]
        vy = path[i][1] - path[i - 1][1]
        norm = math.hypot(vx, vy)
        if norm < 1e-6:
            continue
        vec = (vx / norm, vy / norm)
        if prev_vec is not None:
            dot = max(-1.0, min(1.0, prev_vec[0] * vec[0] + prev_vec[1] * vec[1]))
            total += math.acos(dot)
        prev_vec = vec
    return total


def _estimate_path_direction(path, samples=3):
    if not path or len(path) < 2:
        return None
    end_idx = min(len(path) - 1, samples)
    dx = path[end_idx][0] - path[0][0]
    dy = path[end_idx][1] - path[0][1]
    norm = math.hypot(dx, dy)
    if norm < 1e-6:
        return None
    return (dx / norm, dy / norm)


def candidate_corridor_report(path, costmap, robot_xy, span=2.4):
    direction = _estimate_path_direction(path)
    if direction is None:
        return None
    return costmap.corridor_cross_section(robot_xy, direction, span=span)


def score_candidate_path(path, obstacles, robot_xy, goal_xy, corridor_report=None,
                         perception=None):
    if not path:
        return float("-inf"), {}
    min_clearance, min_idx = path_min_clearance(path, obstacles)
    lookahead_clearance = lookahead_min_clearance(path, obstacles, robot_xy, lookahead_m=1.6)
    pinch_count = len(detect_pinch_points(path, obstacles, gap_min=0.95))
    length = path_length(path)
    turn_cost = path_turn_cost(path)
    final_goal_distance = math.hypot(path[-1][0] - goal_xy[0], path[-1][1] - goal_xy[1])
    corridor_bonus = 0.0
    if corridor_report:
        width = float(corridor_report.get("width", 0.0))
        center = abs(float(corridor_report.get("center_offset", 0.0)))
        confidence = float(corridor_report.get("confidence", 0.0))
        corridor_bonus = max(0.0, width - 0.95) * 1.20
        corridor_bonus += confidence * 0.18
        corridor_bonus -= min(0.9, center) * 0.90
        corridor_bonus -= max(0.0, float(corridor_report.get("bad_penalty", 0.0)))
        if width < 0.95:
            corridor_bonus -= 0.95
        if center > 0.18:
            corridor_bonus -= 0.75
    sensor_bonus = 0.0
    if perception is not None:
        front_clearance = float(getattr(perception, "front_clearance", 1.0))
        left_clearance = float(getattr(perception, "left_clearance", 1.0))
        right_clearance = float(getattr(perception, "right_clearance", 1.0))
        sensor_bonus += max(0.0, front_clearance - 0.35) * 0.45
        sensor_bonus -= max(0.0, abs(left_clearance - right_clearance) - 0.15) * 0.35
    score = (
        7.0 * min_clearance +
        3.5 * lookahead_clearance +
        corridor_bonus -
        0.08 * length -
        1.05 * turn_cost -
        2.3 * pinch_count -
        5.0 * final_goal_distance +
        sensor_bonus
    )
    return score, {
        "min_clearance": min_clearance,
        "min_idx": min_idx,
        "lookahead_clearance": lookahead_clearance,
        "pinch_count": pinch_count,
        "length": length,
        "turn_cost": turn_cost,
        "goal_distance": final_goal_distance,
        "score": score,
    }


def choose_best_path(current_path, candidate_paths, obstacles, robot_xy, goal_xy,
                     costmap=None, persistent_memory=None, perception=None,
                     hysteresis=0.18, span=2.4):
    current_corridor = candidate_corridor_report(current_path, costmap, robot_xy) if costmap else None
    if current_corridor and persistent_memory:
        current_corridor = dict(current_corridor)
        current_corridor["bad_penalty"] = persistent_memory.penalty(current_corridor)
    current_score, current_details = score_candidate_path(
        current_path, obstacles, robot_xy, goal_xy, current_corridor, perception)
    best = (current_path, current_score, current_details, "hold")
    for label, path in candidate_paths:
        report = candidate_corridor_report(path, costmap, robot_xy) if costmap else None
        if report and persistent_memory:
            report = dict(report)
            report["bad_penalty"] = persistent_memory.penalty(report)
        score, details = score_candidate_path(path, obstacles, robot_xy, goal_xy, report, perception)
        if score > best[1] + hysteresis:
            best = (path, score, details, label)
    if current_path and current_score >= best[1] - hysteresis:
        return current_path, current_score, current_details, "hold"
    return best


def clearance_speed_scale(min_clearance):
    if min_clearance < 0.30:
        return 0.55
    if min_clearance < 0.45:
        return 0.70
    if min_clearance < 0.60:
        return 0.85
    return 1.0


def corridor_escape_control(state, corridor_state):
    theta = float(state.get("theta", 0.0))
    hx = math.cos(theta)
    hy = math.sin(theta)
    if corridor_state is not None and corridor_state.centerline:
        pts = [(float(p[0]), float(p[1])) for p in corridor_state.centerline]
        if len(pts) >= 2:
            dx = pts[-1][0] - pts[0][0]
            dy = pts[-1][1] - pts[0][1]
            if math.hypot(dx, dy) < 1e-6:
                dx, dy = hx, hy
            dot = dx * hx + dy * hy
            if dot >= 0.0:
                ex, ey = -dx, -dy
            else:
                ex, ey = dx, dy
            norm = math.hypot(ex, ey)
            if norm > 1e-6:
                ex /= norm
                ey /= norm
        else:
            ex, ey = -hx, -hy
    else:
        ex, ey = -hx, -hy
    local_x = ex * hx + ey * hy
    local_y = -hx * ey + hy * ex
    v = 0.22 * local_x
    w = 0.80 * local_y
    if corridor_state is not None:
        w -= 0.70 * float(getattr(corridor_state, "center_offset", 0.0))
    return np.array([max(-0.25, min(0.05, v)), max(-1.8, min(1.8, w)), 0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# Pluggable local planner interface
# ---------------------------------------------------------------------------
class NavContext:
    """Bundle of per-tick navigation signals passed to a LocalPlanner."""

    __slots__ = ("state", "path", "obstacles", "corridor_state", "corridor_report",
                 "corridor_offset", "goal_xy", "soc", "perc", "bearing",
                 "sensor_front", "sensor_side_bias", "nmpc_dt")

    def __init__(self, state, path, obstacles, corridor_state, corridor_report,
                 corridor_offset, goal_xy, soc, perc, bearing,
                 sensor_front, sensor_side_bias, nmpc_dt):
        self.state = state
        self.path = path
        self.obstacles = obstacles
        self.corridor_state = corridor_state
        self.corridor_report = corridor_report
        self.corridor_offset = corridor_offset
        self.goal_xy = goal_xy
        self.soc = soc
        self.perc = perc
        self.bearing = bearing
        self.sensor_front = sensor_front
        self.sensor_side_bias = sensor_side_bias
        self.nmpc_dt = nmpc_dt


class LocalPlanner:
    """Base class. Returns (v, w, info_dict).

    info_dict may carry {"nmpc_solve_t", "nmpc_traj"} for diagnostics.
    """

    name = "base"

    def compute(self, ctx):
        raise NotImplementedError


class BaselinePursuitPlanner(LocalPlanner):
    """main12's nav behaviour: EnergyAwarePursuit + corridor bias + ReactiveAvoider."""

    name = "baseline"

    def __init__(self, v_ref=0.6, turn_gate=0.6, w_turn=1.8, d_safe=0.45):
        self.pursuit = EnergyAwarePursuit(v_ref=v_ref, turn_gate=turn_gate, w_turn=w_turn)
        self.avoider = ReactiveAvoider(d_safe=d_safe)
        self.avoider_was_active = False
        self.switches = 0

    def compute(self, ctx):
        v, w, _ = self.pursuit.compute(ctx.state, ctx.path, ctx.soc,
                                       bearing_override=ctx.bearing)
        current_min_clearance, _ = path_min_clearance(ctx.path, ctx.obstacles)
        v *= clearance_speed_scale(current_min_clearance)
        v *= float(getattr(ctx.corridor_state, "velocity_scale", 1.0))
        if ctx.sensor_front < 0.55:
            v *= 0.75
        if ctx.corridor_state is not None or ctx.corridor_report:
            if ctx.corridor_state is not None:
                offset = float(getattr(ctx.corridor_state, "center_offset", ctx.corridor_offset))
            else:
                offset = ctx.corridor_offset
            lateral_bias = max(-0.25, min(0.25, -0.8 * offset))
            w += lateral_bias
        if ctx.sensor_front < 0.55:
            w += max(-0.20, min(0.20, 0.18 * ctx.sensor_side_bias))
        v, w, act = self.avoider.correct(v, w, ctx.state, ctx.obstacles)
        if act and not self.avoider_was_active:
            self.switches += 1
        self.avoider_was_active = act
        return v, w, {}


class DWAPlannerAdapter(LocalPlanner):
    """Wraps the Dynamic Window planner as a LocalPlanner."""

    name = "dwa"

    def __init__(self, config):
        from robot_rescue.control.main20_helpers.dwa_planner import DWAPlanner
        self.dwa = DWAPlanner(config)
        self._last_v = 0.0
        self._last_w = 0.0

    def compute(self, ctx):
        # Steer toward a lookahead carrot on the path, not the far goal, so the
        # DWA respects the corridor-aware global plan instead of cutting across.
        target = carrot_on_path(ctx.path, ctx.state, lookahead=1.0) or ctx.goal_xy
        ir = getattr(ctx.perc, "ir", {}) or {}
        v, w = self.dwa.plan(ctx.state, target, ctx.obstacles, ir,
                             current_v=self._last_v, current_w=self._last_w)
        self._last_v, self._last_w = v, w
        return v, w, {}


def carrot_on_path(path, state, lookahead=1.0):
    """Return the point ~lookahead metres ahead of the robot along the path."""
    if not path:
        return None
    rx, ry = float(state["x"]), float(state["y"])
    nearest_i = min(range(len(path)),
                    key=lambda i: math.hypot(path[i][0] - rx, path[i][1] - ry))
    acc = 0.0
    prev = path[nearest_i]
    for pt in path[nearest_i + 1:]:
        acc += math.hypot(pt[0] - prev[0], pt[1] - prev[1])
        prev = pt
        if acc >= lookahead:
            return (float(pt[0]), float(pt[1]))
    return (float(path[-1][0]), float(path[-1][1]))
