"""main12.py -- corridor-aware robot rescue benchmark.

This file keeps the main10 structure, but adds the corridor-awareness and
collision-avoidance improvements we discussed:

  * predictive obstacle inflation using the dynamic tracker,
  * stricter corridor scoring before A* paths are accepted,
  * cooldown-gated replan triggers for narrow passages and moving obstacles,
  * speed reduction when the current path has low clearance.

The goal is not to rewrite the controller stack; it reuses the main9 logic
and strengthens the path selection and replan policy with a local costmap,
corridor tracking, clearance-aware scoring, and early dynamic replanning.
"""

import argparse
import csv
import json
import math
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pybullet as p

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
PATH_MEMORY_FILE = os.path.join(_ROOT, "path_memory.json")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _HERE not in sys.path:
    sys.path.append(_HERE)

from robot.mobile_manipulator import MobileManipulator
from simulation.environment import SimEnvironment
from perception.neural_detector import NeuralObstacleDetector
from perception.sensor_fusion import SensorFusion
from perception.camera_vision import CameraVision
from perception.occupancy_grid import OccupancyGrid
from visualization.dashboard import Dashboard

from main9 import (
    EnergyAwarePursuit,
    DockingController,
    ReactiveAvoider,
    ProgressWatchdog,
    RecoveryAction,
    densify,
)

from robot_rescue.control.main12_helpers.nmpc_controller import NMPCController
from robot_rescue.control.main12_helpers.safety import SafetyFilter
from robot_rescue.control.main12_helpers.main3_helpers.shims import apply_shims
from robot_rescue.control.main12_helpers.main3_helpers.dynamic_tracker import DynamicTracker
from robot_rescue.control.main12_helpers.main3_helpers.world_aware_astar import WorldAwareAStar
from robot_rescue.control.main12_helpers.main3_helpers.energy_paper import PhysicsEnergyManager
from robot_rescue.control.main12_helpers.main3_helpers.perception_fusion import PerceptionFusion
from robot_rescue.control.main12_helpers.main3_helpers.path_clearance import (
    CorridorPredictor,
    DynamicConflictPredictor,
    detect_pinch_points,
    lookahead_min_clearance,
    path_min_clearance,
    plan_with_clearance,
    los_safe,
    predicted_obstacle_sweep,
)
from robot_rescue.control.main12_helpers.main3_helpers.corridor_geometry import (
    CorridorGeometryExtractor,
    is_wedged,
)

from robot_rescue.main10 import (
    variant_settings,
    CorridorWarehouseMission,
    CollisionMonitor,
    load_config,
    seed_everything,
)


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
        self.entries = {}  # key -> ticks remaining

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


class EnhancedCorridorWarehouseMission(CorridorWarehouseMission):
    """Keeps the main10 layout, but records a few corridor hints for clarity."""

    def _spawn_corridor_field(self):
        super()._spawn_corridor_field()
        try:
            p1 = np.asarray(self.pick_pos, float)
            p2 = np.asarray(self.place_pos, float)
            vec = p2 - p1
            dist = float(np.linalg.norm(vec))
            if dist > 1e-6:
                unit = vec / dist
                perp = np.array([-unit[1], unit[0]])
                self.corridor_centerline = (tuple(p1[:2]), tuple(p2[:2]))
                self.corridor_direction = tuple(unit[:2])
                self.corridor_perp = tuple(perp[:2])
        except Exception:
            self.corridor_centerline = None
            self.corridor_direction = None
            self.corridor_perp = None


def _clearance_speed_scale(min_clearance):
    if min_clearance < 0.30:
        return 0.55
    if min_clearance < 0.45:
        return 0.70
    if min_clearance < 0.60:
        return 0.85
    return 1.0


def _corridor_escape_control(state, corridor_state):
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

def run_mission(args):
    config = load_config(args.config)
    vs = variant_settings(args.variant)

    if not vs["energy_weight"]:
        config.setdefault("nmpc", {})["R_energy"] = 0.0

    # --- main12 real-time NMPC tuning -------------------------------------
    # The Chapter-4 analysis flagged the ~360 ms NMPC solve as the headline
    # latency. The dominant cost is the NLP size, which scales with the
    # prediction horizon N (states 7*(N+1), controls 5*N, plus per-step
    # obstacle/corridor constraints), and the IPOPT iteration cap. Warm-start
    # is already on, so shrinking the horizon and capping iterations is the
    # cheap, reversible lever. We override ONLY main12's own config copy here
    # (the shared config.yaml is untouched), and expose both as CLI flags so
    # the horizon/latency trade-off can be swept for the thesis table.
    nmpc_cfg = config.setdefault("nmpc", {})
    # Tier-1 default: horizon 8 (measured ~132 ms mean / 172 ms p99 vs ~240/326
    # at N=20, AND it delivers 3/3 where N=20 timed out -- a short horizon tracks
    # the tight slalom better). 0 means "use this default"; >0 overrides.
    nmpc_cfg["horizon"] = int(args.nmpc_horizon) if args.nmpc_horizon > 0 else 8
    if args.nmpc_max_iter > 0:
        nmpc_cfg["max_iter"] = int(args.nmpc_max_iter)
    # Tier-2: C codegen / JIT (CasADi -> gcc). Off by default; --nmpc-jit turns
    # it on. Guarded inside the controller so a missing compiler falls back.
    nmpc_cfg["jit"] = bool(getattr(args, "nmpc_jit", False))

    client = p.connect(p.GUI if args.gui else p.DIRECT)
    p.setRealTimeSimulation(0, physicsClientId=client)
    if args.gui:
        p.resetDebugVisualizerCamera(6, 0, -75, [0.25, 0.75, 0], physicsClientId=client)

    env = SimEnvironment(client, config)
    start = [0.0, 0.0, 0.0]
    robot = MobileManipulator(client, config, start_pos=[0.0, 0.0, 0.10],
                              start_orn=p.getQuaternionFromEuler([0, 0, 0]))

    spd = vs["speed_scale"]
    pursuit = EnergyAwarePursuit(v_ref=0.6 * spd, turn_gate=0.6, w_turn=1.8)
    avoider = ReactiveAvoider(d_safe=0.45)
    docker = DockingController(engage_r=0.75, face_tol=0.35, v_max=0.35)
    nmpc = NMPCController(config)
    safety = SafetyFilter(config)
    astar = WorldAwareAStar()
    occ = OccupancyGrid()

    detector = NeuralObstacleDetector(client, config, robot_id=robot.robot_id,
                                      ground_id=env.plane_id)
    fusion = SensorFusion(config)
    camera = CameraVision(client, config, robot.robot_id)
    energy = PhysicsEnergyManager(config,
                                  enable_regen=vs["enable_regen"],
                                  enable_ekf=vs["enable_ekf"],
                                  enable_balance=vs["enable_balance"],
                                  enable_dispersion=vs["enable_dispersion"])
    apply_shims(energy)
    tracker = DynamicTracker(env)
    perception = PerceptionFusion(detector, fusion, camera, env, tracker)
    costmap = LocalCostmap(robot_radius=0.30, safety_margin=0.18, radius=2.8, resolution=0.10)
    corridor_extractor = CorridorGeometryExtractor(robot_radius=0.30, safety_margin=0.12,
                                                   ray_range=2.8, ray_resolution=0.05,
                                                   path_spacing=0.22, profile_span=2.4)
    corridor_memory = CorridorMemory(hysteresis=0.20, smooth=0.35)

    sim_dt = config["simulation"]["timestep"]
    nmpc_dt = config.get("nmpc", {}).get("dt", 0.1)
    ctrl_every = max(1, int(nmpc_dt / sim_dt))
    astar_period = ctrl_every * 10

    corridor_predictor = CorridorPredictor(lookahead_m=1.9, min_clearance=0.30,
                                            cooldown_s=1.2, dt=nmpc_dt)
    conflict_predictor = DynamicConflictPredictor(lookahead_path_m=2.2,
                                                  horizon_s=2.5, dt_sample=0.05,
                                                  conflict_dist=0.65, cooldown_s=1.2,
                                                  dt=nmpc_dt)
    watchdog = ProgressWatchdog(window_s=3.0, min_gain=0.15, dt=nmpc_dt)
    recovery = RecoveryAction(dt=nmpc_dt)
    corridor_blacklist = CorridorBlacklist(ttl_s=18.0, dt=nmpc_dt)
    persistent_corridor_memory = PersistentCorridorMemory(PATH_MEMORY_FILE)
    corridor_state = None
    corridor_escape_ticks = 0
    corridor_escape_limit = int(1.2 / nmpc_dt)
    corridor_escape_cool = 0
    wedge_escapes = 0
    corridor_narrow_s = 0.0
    corridor_class = "none"
    if getattr(args, "reset_path_memory", False):
        persistent_corridor_memory.reset()
        print(f"[main12] reset path memory: {PATH_MEMORY_FILE}")

    mission = EnhancedCorridorWarehouseMission(config, env, robot, energy,
                                               initial_soc=args.initial_soc,
                                               num_boxes=args.num_boxes,
                                               start=start, dyn_count=args.dyn_count,
                                               dynamic=bool(getattr(args, "dynamic", False)))
    collision = CollisionMonitor(client, robot.robot_id, env)

    dashboard = None
    if args.gui and not args.no_dashboard:
        try:
            dashboard = Dashboard(config)
            plt.show(block=False)
        except Exception:
            dashboard = None

    box_colors = getattr(camera, "BOX_COLORS",
                         [(255, 255, 0), (255, 102, 0), (77, 153, 255)])
    soc_throttle = vs["soc_throttle"]

    for _ in range(50):
        try:
            p.stepSimulation(physicsClientId=client)
        except Exception:
            break

    state = robot.get_state()
    current_path = []
    last_astar = -astar_period
    prev_phase = mission.phase
    final_control = np.zeros(5)
    nmpc_prev = np.zeros(5)
    sim_time = 0.0
    path_len = 0.0
    prev_xy = (state["x"], state["y"])
    nmpc_solves = 0
    last_nmpc_solve_time = 0.0   # nmpc.solve() return value was being discarded (`u, _, _ = ...`)
    last_predicted_traj = None   # X_sol from nmpc.solve() -- also discarded before
    power = 0.0                  # energy.update() return value -- also discarded before
    dock_stall = 0
    dock_stall_limit = int(4.0 / nmpc_dt)
    astar_replans = 0
    switches = 0
    avoider_was_active = False
    collision_hits = 0
    failure = None

    try:
        for step in range(args.max_steps):
            try:
                p.getConnectionInfo(physicsClientId=client)
            except Exception:
                break

            state = robot.get_state()
            path_len += math.hypot(state["x"] - prev_xy[0], state["y"] - prev_xy[1])
            prev_xy = (state["x"], state["y"])
            mission.update_goal(state)
            goal = mission.goal
            goal_xy = (goal[0], goal[1])

            if mission.phase != prev_phase:
                current_path = []
                watchdog.reset()
                dock_stall = 0
                corridor_escape_ticks = 0
                corridor_escape_cool = 0
                prev_phase = mission.phase
                collision_hits = 0

            soc = energy.get_soc()
            if corridor_escape_cool > 0:
                corridor_escape_cool -= 1
            if corridor_escape_ticks > 0:
                corridor_escape_ticks -= 1

            if step % ctrl_every == 0:
                # BUG FIX: CorridorPredictor, DynamicConflictPredictor and
                # CorridorBlacklist all gate themselves with a tick-based
                # cooldown/TTL (`_last_trigger_age` / `entries[key]`), but
                # nothing in this loop ever called their `.tick()` methods.
                # `_last_trigger_age` was set to 0 on first trigger and then
                # stayed 0 forever (only tick() advances it), so
                # `should_replan_now()` returned False for the rest of the
                # run after firing exactly once -- the early-warning corridor
                # replan that is supposed to fire repeatedly was effectively
                # single-shot for the whole mission. Likewise corridor_blacklist
                # entries never expired (ttl_s=18.0 was decoration only), so a
                # corridor that caused one contact stayed blacklisted forever
                # instead of cooling off after 18s. Call tick() once per
                # control step (matching the dt=nmpc_dt they were configured
                # with) to restore the intended periodic behavior.
                corridor_predictor.tick()
                conflict_predictor.tick()
                corridor_blacklist.tick()

                want_vision_for = None
                try:
                    if args.use_camera and mission.phase == "nav_to_pick":
                        want_vision_for = ["yellow", "orange", "blue"][mission.pick_idx % 3]
                except Exception:
                    want_vision_for = None

                perc = perception.perceive(robot, state, sim_time, want_vision_for=want_vision_for)
                obstacles = list(perc.obstacles)
                ir = perc.ir

                try:
                    occ.update_from_obstacles(obstacles)
                    occ.update_from_ir(state["x"], state["y"], state["theta"], ir)
                    occ.decay()
                    for so in occ.get_obstacles_list():
                        if not any(math.hypot(so[0] - o[0], so[1] - o[1]) < 0.3 for o in obstacles):
                            obstacles.append(so)
                except Exception:
                    pass

                if energy.is_emergency():
                    failure = "battery"
                    break

                rx, ry = state["x"], state["y"]
                obstacles = [(ox, oy, r) for ox, oy, r in obstacles
                             if math.hypot(ox - rx, oy - ry) > 0.35
                             and math.hypot(ox - goal_xy[0], oy - goal_xy[1]) > 0.40]

                dynamic_sweep = predicted_obstacle_sweep(tracker, sim_time,
                                                         horizon_s=1.8,
                                                         dt_sample=0.25,
                                                         inflate=0.12)
                augmented_obstacles = obstacles + dynamic_sweep
                costmap.build((rx, ry), augmented_obstacles)
                corridor_state = corridor_extractor.extract(
                    state, goal_xy, current_path or densify([(rx, ry), goal_xy], 0.15),
                    augmented_obstacles)
                if (corridor_state.is_narrow or corridor_state.is_tight or
                        corridor_state.is_dead_end or
                        corridor_state.classification == "constrained"):
                    dynamic_sweep = predicted_obstacle_sweep(tracker, sim_time,
                                                             horizon_s=2.5,
                                                             dt_sample=0.05,
                                                             inflate=0.16)
                    augmented_obstacles = obstacles + dynamic_sweep
                    costmap.build((rx, ry), augmented_obstacles)
                    corridor_state = corridor_extractor.extract(
                        state, goal_xy, current_path or densify([(rx, ry), goal_xy], 0.15),
                        augmented_obstacles)

                corridor_class = getattr(corridor_state, "classification", "none")
                if (corridor_state.is_narrow or corridor_state.is_tight or
                        corridor_state.is_dead_end or
                        corridor_state.classification == "constrained"):
                    corridor_narrow_s += sim_dt

                if current_path and len(current_path) > 1:
                    nearest_i = min(range(len(current_path)),
                                    key=lambda i: math.hypot(current_path[i][0] - rx,
                                                             current_path[i][1] - ry))
                    ref_pt = current_path[min(nearest_i + 1, len(current_path) - 1)]
                else:
                    ref_pt = goal_xy
                ref_dir = (ref_pt[0] - rx, ref_pt[1] - ry)
                corridor_report = costmap.corridor_cross_section((rx, ry), ref_dir, span=2.2)
                corridor_offset = corridor_memory.update(corridor_report)
                if corridor_report:
                    corridor_report = dict(corridor_report)
                    corridor_report["bad_penalty"] = corridor_blacklist.penalty(corridor_report)

                current_lookahead_clearance = float("inf")
                if current_path:
                    current_lookahead_clearance = lookahead_min_clearance(
                        current_path, augmented_obstacles, (rx, ry), lookahead_m=1.8)

                sensor_front = float(getattr(perc, "front_clearance", 1.0))
                sensor_left = float(getattr(perc, "left_clearance", 1.0))
                sensor_right = float(getattr(perc, "right_clearance", 1.0))
                sensor_side_bias = sensor_left - sensor_right

                if mission.is_arm_phase:
                    final_control = np.zeros(5)
                    try:
                        corridor_for_nmpc = corridor_extractor.to_nmpc_corridor(
                            corridor_state, state, nmpc.N + 1, nmpc_dt)
                        u, solve_t, X_sol = nmpc.solve(state["full_q"], goal, obstacles, soc,
                                             u_prev=nmpc_prev,
                                             corridor=corridor_for_nmpc)
                        nmpc_prev = u.copy()
                        nmpc_solves += 1
                        last_nmpc_solve_time = solve_t
                        last_predicted_traj = np.asarray(X_sol)[:, :2]
                    except Exception:
                        pass
                else:
                    need = (not current_path or
                            (step - last_astar) >= astar_period or
                            (current_path and math.hypot(current_path[-1][0] - goal_xy[0],
                                                         current_path[-1][1] - goal_xy[1]) > 0.25))

                    if corridor_state.is_dead_end:
                        need = True
                    if current_lookahead_clearance < 0.30:
                        need = True
                    if sensor_front < 0.42:
                        need = True
                    if corridor_report and corridor_report["width"] < 1.00:
                        need = True
                    if corridor_memory.off_center(corridor_report):
                        need = True
                    if abs(corridor_offset) > 0.22:
                        need = True
                    if corridor_blacklist.is_bad(corridor_report) or persistent_corridor_memory.is_bad(corridor_report):
                        need = True

                    if current_path and not need:
                        ci = min(range(len(current_path)),
                                 key=lambda i: math.hypot(current_path[i][0] - rx,
                                                          current_path[i][1] - ry))
                        for wx, wy in current_path[ci:]:
                            if any(math.hypot(wx - ox, wy - oy) < r + 0.35 for ox, oy, r in augmented_obstacles):
                                need = True
                                break

                    if corridor_predictor.should_replan_now(state, current_path, augmented_obstacles):
                        need = True
                    if conflict_predictor.should_replan_now(state, current_path, tracker, sim_time):
                        need = True

                    if need:
                        candidate_paths = []
                        if los_safe(state, goal_xy, augmented_obstacles,
                                    robot_radius=0.3, safety=0.45):
                            direct_path = densify([(rx, ry), goal_xy], 0.15)
                            candidate_paths.append(("direct", direct_path))

                        plan_a, info_a = plan_with_clearance(
                            astar, (rx, ry), goal_xy, augmented_obstacles,
                            min_clearance=0.30, gap_min=0.95,
                            max_attempts=6, inflation_step=0.25,
                        )
                        candidate_paths.append(("clear", densify(plan_a if plan_a else [(rx, ry), goal_xy], 0.15)))

                        plan_b, info_b = plan_with_clearance(
                            astar, (rx, ry), goal_xy, augmented_obstacles,
                            min_clearance=0.38, gap_min=1.05,
                            max_attempts=4, inflation_step=0.30,
                        )
                        candidate_paths.append(("conservative", densify(plan_b if plan_b else [(rx, ry), goal_xy], 0.15)))

                        best_path, best_score, best_details, best_label = choose_best_path(
                            current_path, candidate_paths, augmented_obstacles, (rx, ry), goal_xy,
                            costmap=costmap,
                            persistent_memory=persistent_corridor_memory,
                            perception=perc,
                            hysteresis=0.20)
                        if best_path is not current_path:
                            current_path = best_path
                            if best_label != "hold":
                                astar_replans += 1
                        else:
                            current_path = best_path
                        if best_details.get("min_clearance", float("inf")) < 0.34:
                            astar_replans += 1
                        last_astar = step

                    bearing = perc.target_bearing
                    dist_goal = math.hypot(goal_xy[0] - rx, goal_xy[1] - ry)
                    if args.use_camera and mission.phase == "nav_to_pick" and dist_goal < 2.6:
                        bearing = perc.target_bearing if perc.target_bearing is not None else bearing

                    docking_now = False
                    geometric_wedge = (
                        corridor_state is not None
                        and (corridor_state.is_narrow or
                             corridor_state.is_tight or
                             corridor_state.is_dead_end)
                    )
                    wedge_now = (
                        geometric_wedge
                        and is_wedged(perc)
                        and corridor_escape_cool <= 0
                        and not docker.in_range(state, goal_xy)
                    )
                    if wedge_now:
                        final_control = _corridor_escape_control(state, corridor_state)
                        corridor_escape_ticks = corridor_escape_limit
                        corridor_escape_cool = int(1.4 / nmpc_dt)
                        current_path = []
                        wedge_escapes += 1
                        corridor_blacklist.mark_bad(corridor_report)
                        persistent_corridor_memory.mark_bad(corridor_report, context="wedge")
                    elif recovery.active:
                        bv, bw, _ = recovery.step()
                        final_control = np.array([bv, bw, 0, 0, 0])
                    elif docker.in_range(state, goal_xy):
                        docking_now = True
                        v, w = docker.compute(state, goal_xy)
                        final_control = np.array([v, w, 0, 0, 0])
                        dock_stall += 1
                        if dock_stall > dock_stall_limit:
                            recovery.trigger(state, goal_xy)
                            current_path = []
                            dock_stall = 0
                            bv, bw, _ = recovery.step()
                            final_control = np.array([bv, bw, 0, 0, 0])
                    elif watchdog.update(state, goal_xy):
                        recovery.trigger(state, goal_xy)
                        corridor_blacklist.mark_bad(corridor_report)
                        persistent_corridor_memory.mark_bad(corridor_report, context="watchdog")
                        current_path = []
                        bv, bw, _ = recovery.step()
                        final_control = np.array([bv, bw, 0, 0, 0])
                    else:
                        dock_stall = 0
                        soc_in = soc if soc_throttle else 1.0
                        v, w, _ = pursuit.compute(state, current_path, soc_in,
                                                  bearing_override=bearing)
                        current_min_clearance, _ = path_min_clearance(current_path, augmented_obstacles)
                        v *= _clearance_speed_scale(current_min_clearance)
                        v *= float(getattr(corridor_state, "velocity_scale", 1.0))
                        if sensor_front < 0.55:
                            v *= 0.75
                        if corridor_state is not None or corridor_report:
                            offset = float(getattr(corridor_state, "center_offset", corridor_offset)) if corridor_state is not None else corridor_offset
                            lateral_bias = max(-0.25, min(0.25, -0.8 * offset))
                            w += lateral_bias
                        if sensor_front < 0.55:
                            w += max(-0.20, min(0.20, 0.18 * sensor_side_bias))
                        v, w, act = avoider.correct(v, w, state, augmented_obstacles)
                        if act and not avoider_was_active:
                            switches += 1
                        avoider_was_active = act
                        final_control = np.array([v, w, 0, 0, 0])

                    try:
                        safe_obs = [] if docking_now else augmented_obstacles
                        final_control = safety.filter_control(final_control, state, safe_obs, soc)
                    except Exception:
                        final_control = np.zeros(5)
                        recovery.trigger(state, goal_xy)

            if collision.check():
                collision_hits += 1
                corridor_blacklist.mark_bad(corridor_report)
                persistent_corridor_memory.mark_bad(corridor_report, context="collision")
                recovery.trigger(state, goal_xy)
                current_path = []
                if collision_hits >= 3:
                    failure = "collision"
                    break
                try:
                    bv, bw, _ = recovery.step()
                    final_control = np.array([bv, bw, 0, 0, 0])
                except Exception:
                    final_control = np.zeros(5)

            if mission.is_arm_phase:
                robot.set_base_velocity(0.0, 0.0)
            else:
                try:
                    robot.apply_control(final_control)
                except Exception:
                    robot.set_base_velocity(0.0, 0.0)

            try:
                power = energy.update(final_control, sim_dt, state["arm_q"])
            except Exception:
                pass

            try:
                p.stepSimulation(physicsClientId=client)
            except Exception:
                break
            env.update_dynamic_obstacles(sim_time)
            sim_time += sim_dt

            # Feed the dashboard at the control-tick cadence (nmpc_dt), not
            # every physics sub-step -- matches the cadence everything else
            # in this loop (perception/planning) already runs at, and keeps
            # the per-call cost (mostly list appends) an order of magnitude
            # cheaper than calling it every sim_dt. Dashboard.update() itself
            # only redraws every `visualization.update_interval` calls (see
            # config.yaml) -- raise that value if the GUI is still too heavy
            # for this machine; no code change needed for that knob.
            # Wrapped in try/except so a plotting hiccup can never kill the
            # mission run.
            if dashboard is not None and step % ctrl_every == 0:
                try:
                    dashboard.update(
                        state, power, soc, last_nmpc_solve_time,
                        augmented_obstacles, goal_xy,
                        predicted_traj=last_predicted_traj,
                        t=sim_time, energy_mgr=energy,
                    )
                except Exception:
                    pass

            if mission.is_done(state):
                break

    except KeyboardInterrupt:
        pass
    except Exception as e:
        failure = failure or f"error:{e}"

    try:
        boxes = mission.box_idx
    except Exception:
        boxes = 0
    delivered_all = (boxes >= mission.num_boxes)
    done = mission.is_done(state)
    success = bool(delivered_all and done and failure is None)
    kicks = watchdog.kicks

    st = nmpc.get_solve_time_stats()
    metrics = {
        "variant":        args.variant,
        "seed":           args.seed,
        "success":        int(success),
        "boxes":          boxes,
        "num_boxes":      mission.num_boxes,
        "failure":        failure or ("none" if success else "timeout"),
        "mission_time_s": round(sim_time, 2),
        "path_m":         round(path_len, 3),
        "energy_wh":      round(float(getattr(energy, "energy_consumed", 0.0)), 4),
        "regen_wh":       round(float(getattr(energy, "energy_regenerated", 0.0)), 4),
        "final_soc":      round(energy.get_soc() * 100.0, 2),
        "terminal_v":     round(float(energy.get_voltage()), 3),
        "max_cell_temp":  round(float(energy.max_cell_temp()), 2),
        "soc_spread":     round(float(energy.soc_spread()) * 100.0, 3),
        "nmpc_mean_ms":   round(float(st["mean"]), 2) if st["count"] else 0.0,
        "nmpc_p99_ms":    round(float(st.get("p99", 0.0)), 2) if st["count"] else 0.0,
        "nmpc_solves":    nmpc_solves,
        "switches":       switches,
        "astar_replans":  astar_replans,
        "recoveries":     kicks,
        "collisions":     collision.count,
        "wedge_escapes":  wedge_escapes,
        "corridor_narrow_s": round(corridor_narrow_s, 2),
        "corridor_class": corridor_class,
    }

    try:
        p.disconnect(client)
    except Exception:
        pass
    return metrics


def print_summary(m):
    print("\n" + "=" * 60)
    print(" Chapter-4 Warehouse Mission Summary (main12)")
    print("=" * 60)
    print(f"  Variant:               {m['variant']}")
    print(f"  Boxes delivered:       {m['boxes']} / {m['num_boxes']}")
    print(f"  Total sim time:        {m['mission_time_s']:6.1f} s")
    print(f"  Path length:           {m['path_m']:6.2f} m")
    print(f"  Energy consumed:       {m['energy_wh']:6.3f} Wh")
    print(f"  Energy regenerated:    {m['regen_wh']:6.4f} Wh")
    print(f"  Final SoC:             {m['final_soc']:5.1f} %")
    print(f"  Terminal voltage:      {m['terminal_v']:6.2f} V")
    print(f"  Max cell temp:         {m['max_cell_temp']:5.2f} C")
    print(f"  SoC spread:            {m['soc_spread']:5.2f} %")
    print(f"  NMPC solve time:       mean={m['nmpc_mean_ms']:.1f}ms p99={m['nmpc_p99_ms']:.1f}ms n={m['nmpc_solves']}")
    print(f"  CAPP->DWA switches:    {m['switches']}")
    print(f"  A* replans:            {m['astar_replans']}")
    print(f"  Recovery kicks:        {m['recoveries']}")
    print(f"  Wedge corridor escapes:{m['wedge_escapes']}")
    print(f"  Narrow corridor time:  {m['corridor_narrow_s']:5.1f} s ({m['corridor_class']})")
    print(f"  Collisions:            {m['collisions']}")
    print(f"  Failure mode:          {m['failure']}")
    print(f"  SUCCESS:               {'YES' if m['success'] else 'NO'}")
    print("=" * 60)


def run_batch(args):
    import re
    import subprocess
    from concurrent.futures import ThreadPoolExecutor

    fields = ["variant", "seed", "success", "boxes", "num_boxes", "failure",
              "mission_time_s", "path_m", "energy_wh", "regen_wh", "final_soc",
              "terminal_v", "max_cell_temp", "soc_spread", "nmpc_mean_ms",
              "nmpc_p99_ms", "nmpc_solves", "switches", "astar_replans",
              "recoveries", "collisions", "wedge_escapes", "corridor_narrow_s",
              "corridor_class"]
    out_path = (args.batch_out if os.path.isabs(args.batch_out)
                else os.path.join(_HERE, args.batch_out))

    # Worker clamp: each replicate runs a single-threaded IPOPT solve (BLAS is
    # pinned to 1 thread for reproducibility). Oversubscribing cores makes
    # individual solves stall, which starves the controller, snowballs A* replans,
    # and trips the 240s watchdog. Verified on this machine: workers=4 -> 2
    # timeouts (same seeds that pass at workers=2); workers=2 -> 8/8. Each solve
    # wants a full core, so cap at (physical cores - 2), leaving headroom for the
    # OS + PyBullet. psutil gives true physical count; fall back to logical//2.
    try:
        import psutil
        phys = psutil.cpu_count(logical=False) or ((os.cpu_count() or 4) // 2)
    except Exception:
        phys = max(1, (os.cpu_count() or 4) // 2)
    safe_workers = max(1, phys - 2)
    if args.workers > safe_workers:
        print(f"[batch] clamping workers {args.workers} -> {safe_workers} "
              f"({phys} physical cores; IPOPT solves are CPU-bound, "
              f"oversubscription causes timeouts). Override is intentional only.",
              flush=True)
        args.workers = safe_workers

    def one(seed):
        cmd = [sys.executable, os.path.abspath(__file__), "--no-gui",
               "--variant", args.variant, "--seed", str(seed),
               "--initial-soc", str(args.initial_soc),
               "--num-boxes", str(args.num_boxes),
               "--dyn-count", str(args.dyn_count),
               "--max-steps", str(args.max_steps),
               "--nmpc-horizon", str(args.nmpc_horizon),
               "--nmpc-max-iter", str(args.nmpc_max_iter),
               "--config", str(args.config), "--emit-csv-row"]
        if args.nmpc_jit:
            cmd.append("--nmpc-jit")
        if getattr(args, "dynamic", False):
            cmd.append("--dynamic")
        env = dict(os.environ, PYTHONHASHSEED=str(seed), OMP_NUM_THREADS="1",
                   OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   NUMEXPR_NUM_THREADS="1")
        try:
            r = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=args.timeout, cwd=_HERE, env=env)
            out = (r.stdout or "") + "\n" + (r.stderr or "")
        except subprocess.TimeoutExpired:
            out = ""
        m = re.search(r"CSVROW\|(.*)", out)
        if not m:
            return {**{k: 0 for k in fields}, "variant": args.variant,
                    "seed": seed, "failure": "timeout", "success": 0,
                    "num_boxes": args.num_boxes}
        vals = m.group(1).split("|")
        row = dict(zip(fields, vals))
        for k in ("success", "boxes", "num_boxes", "nmpc_solves", "switches",
                  "astar_replans", "recoveries", "collisions", "wedge_escapes"):
            row[k] = int(float(row[k])) if row[k] else 0
        for k in ("mission_time_s", "path_m", "energy_wh", "regen_wh",
                  "final_soc", "terminal_v", "max_cell_temp", "soc_spread",
                  "nmpc_mean_ms", "nmpc_p99_ms", "corridor_narrow_s"):
            row[k] = float(row[k]) if row[k] else 0.0
        print(f"[batch] {args.variant:11s} seed={seed} success={row['success']} "
              f"boxes={row['boxes']}/{row['num_boxes']} E={row['energy_wh']:.1f}Wh "
              f"SoC={row['final_soc']:.0f}% t={row['mission_time_s']:.0f}s "
              f"fail={row['failure']}", flush=True)
        return row

    seeds = [args.seed_base + k for k in range(args.batch)]
    if args.workers <= 1:
        rows = [one(s) for s in seeds]
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            rows = list(pool.map(one, seeds))

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    ok = [r for r in rows if int(r["success"]) == 1]
    n = len(rows)

    def mean(key, src=ok):
        vals = [float(r[key]) for r in src]
        return sum(vals) / len(vals) if vals else 0.0

    print("\n" + "=" * 64)
    print(f" main12 batch: variant={args.variant}  {len(ok)}/{n} success "
          f"({100 * len(ok) / max(n, 1):.1f}%)")
    if ok:
        print(f"  energy   {mean('energy_wh'):.2f} Wh   regen {mean('regen_wh'):.3f} Wh")
        print(f"  final SoC {mean('final_soc'):.1f}%   term V {mean('terminal_v'):.2f}")
        print(f"  max cell T {mean('max_cell_temp'):.2f} C   SoC spread {mean('soc_spread'):.2f}%")
        print(f"  mission   {mean('mission_time_s'):.1f} s   path {mean('path_m'):.2f} m")
        print(f"  NMPC mean {mean('nmpc_mean_ms'):.1f} ms   switches {mean('switches'):.1f}  "
              f"replans {mean('astar_replans'):.1f}  recoveries {mean('recoveries'):.2f}")
    from collections import Counter
    fc = Counter(r["failure"] for r in rows if int(r["success"]) == 0)
    if fc:
        print(f"  failures: {dict(fc)}")
    print(f"  CSV: {out_path}")
    print("=" * 64)
    return 0


def parse_args():
    ap = argparse.ArgumentParser(description="main12 corridor-aware benchmark")
    ap.add_argument("--gui", action="store_true", default=True)
    ap.add_argument("--no-gui", dest="gui", action="store_false")
    ap.add_argument("--config", type=str, default=os.path.join(_ROOT, "config.yaml"))
    ap.add_argument("--max-steps", type=int, default=24000)
    ap.add_argument("--no-dashboard", action="store_true", default=False)
    ap.add_argument("--initial-soc", type=float, default=0.70)
    ap.add_argument("--num-boxes", type=int, default=3)
    ap.add_argument("--dyn-count", type=int, default=2)
    ap.add_argument("--dynamic", action="store_true", default=False,
                    help="Activate the dynamic-obstacle layer (sinusoidal "
                         "AGV/pedestrian sweep). Without this flag the corridor "
                         "field is static slalom only; --dyn-count then has no "
                         "effect. Used by the Sec 4.9 stress sweep.")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--variant", type=str, default="full",
                    choices=["full", "pack_level", "no_energy", "no_regen", "speed_only"])
    ap.add_argument("--use-camera", action="store_true", default=True)
    ap.add_argument("--no-camera", dest="use_camera", action="store_false")
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--batch-out", type=str, default="main12_batch.csv")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seed-base", type=int, default=1000)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--nmpc-horizon", type=int, default=0,
                    help="Override NMPC prediction horizon N (0 = use config, "
                         "currently 20). Lower N -> faster solve. Try 10-12 for "
                         "real-time-grade latency.")
    ap.add_argument("--nmpc-max-iter", type=int, default=30,
                    help="IPOPT max iterations (default 30). Measured: 30 gives "
                         "~30 ms mean / ~63 ms p99 and 8/8 success vs uncapped "
                         "~73 ms / 132 ms -- capping forces a good-enough solve "
                         "fast, which also cuts A* replan cascades. Set 0 to use "
                         "the config value (50).")
    ap.add_argument("--nmpc-jit", action="store_true", default=False,
                    help="Tier-2: compile the NMPC functions to C via CasADi JIT "
                         "(gcc). Targets 40-90 ms. Falls back to no-JIT if no "
                         "compiler is found.")
    ap.add_argument("--reset-path-memory", action="store_true", default=False,
                    help="Reset the persistent path/corridor memory file before running.")
    ap.add_argument("--emit-csv-row", action="store_true", default=False,
                    help="Print a CSVROW|... line for the batch parent to parse.")
    return ap.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    if args.batch and args.batch > 0:
        return run_batch(args)

    m = run_mission(args)
    if args.emit_csv_row:
        fields = ["variant", "seed", "success", "boxes", "num_boxes", "failure",
                  "mission_time_s", "path_m", "energy_wh", "regen_wh", "final_soc",
                  "terminal_v", "max_cell_temp", "soc_spread", "nmpc_mean_ms",
                  "nmpc_p99_ms", "nmpc_solves", "switches", "astar_replans",
                  "recoveries", "collisions", "wedge_escapes", "corridor_narrow_s",
                  "corridor_class"]
        print("CSVROW|" + "|".join(str(m[k]) for k in fields), flush=True)
    print_summary(m)
    return 0


if __name__ == "__main__":
    sys.exit(main())