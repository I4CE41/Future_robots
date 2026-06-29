"""Priority-aware perception fusion for real-world deployment.

Reads sensors with a deterministic priority order and produces a unified
navigation context. On a real robot, sensors fail at different rates and
have different latencies; this layer encodes the priority a roboticist
would actually use:

  1) IR proximity (highest priority, lowest latency, can't be fooled by lighting)
  2) RGB camera   (target identification, bearing-to-goal correction)
  3) Lidar/Neural detector (geometric obstacle map, longer range)
  4) Ground-truth env list (sim only, used to fill any gaps)
  5) Dynamic tracker (sim only; on real robot replace with EKF/multi-object tracker)

Returns a single `PerceptionResult` per tick that the controller consumes.
"""

import math
import time


class PerceptionResult:
    __slots__ = ("obstacles", "ir", "target_bearing", "target_color",
                 "dynamic_obstacles", "sensors_used", "latency_ms",
                 "front_clearance", "left_clearance", "right_clearance")

    def __init__(self):
        self.obstacles = []          # list of (x, y, r) world-frame
        self.ir = {}                 # dict of {sensor_name: distance_m}
        self.target_bearing = None   # float (rad) | None
        self.target_color = None     # str | None
        self.dynamic_obstacles = []  # list of (x, y, r, vx, vy)
        self.sensors_used = []       # list of sensor names that contributed
        self.latency_ms = 0.0
        self.front_clearance = 1.0   # min of front IR sensors (m)
        self.left_clearance = 1.0
        self.right_clearance = 1.0


class PerceptionFusion:
    """Combines multiple sensor streams into a single PerceptionResult."""

    def __init__(self, detector, fusion, camera, env, tracker,
                 ir_max_range=1.0,
                 vision_target_colors=None):
        self.detector = detector
        self.fusion = fusion
        self.camera = camera
        self.env = env
        self.tracker = tracker
        self.ir_max_range = float(ir_max_range)
        # In order of priority — first match wins for target bearing
        self.target_colors = vision_target_colors or [
            ("yellow", (255, 255, 0)),
            ("orange", (255, 102, 0)),
            ("blue",   (77, 153, 255)),
        ]

    # ------------------------------------------------------------------
    def perceive(self, robot, state, sim_time, want_vision_for=None):
        """Run all sensors and fuse. want_vision_for: color name to look for,
        or None to skip vision (vision is expensive)."""
        t0 = time.time()
        res = PerceptionResult()

        # 1) IR — cheapest, always run
        try:
            res.ir = robot.read_ir_sensors()
            res.sensors_used.append("ir")
        except Exception:
            res.ir = {}

        res.front_clearance = min(
            res.ir.get("front_center", self.ir_max_range),
            res.ir.get("front_left",   self.ir_max_range),
            res.ir.get("front_right",  self.ir_max_range),
        )
        res.left_clearance = res.ir.get("left", self.ir_max_range)
        res.right_clearance = res.ir.get("right", self.ir_max_range)

        # 2) Neural / lidar detector + ground-truth env list
        try:
            detected = self.detector.detect(state)
            res.sensors_used.append("neural_detector")
        except Exception:
            detected = []
        env_obs = self.env.get_all_obstacles(sim_time)
        res.sensors_used.append("env_gt")
        try:
            res.obstacles = self.fusion.fuse(detected, env_obs)
            res.sensors_used.append("sensor_fusion")
        except Exception:
            res.obstacles = list(env_obs)

        # 3) Dynamic-obstacle tracker (analytic in sim, would be EKF on real bot)
        try:
            res.dynamic_obstacles = self.tracker.get_at(sim_time)
            if res.dynamic_obstacles:
                res.sensors_used.append("dynamic_tracker")
        except Exception:
            res.dynamic_obstacles = []

        # 4) RGB camera — only if caller asked (real cameras at 10Hz are
        # ~30ms per frame, too expensive to run every tick)
        if want_vision_for is not None:
            color_rgb = None
            color_name = None
            for name, rgb in self.target_colors:
                if name == want_vision_for:
                    color_name = name
                    color_rgb = rgb
                    break
            if color_rgb is not None:
                try:
                    bearing = self.camera.get_bearing_to_object(state, color_rgb)
                    if bearing is not None:
                        res.target_bearing = float(bearing)
                        res.target_color = color_name
                        res.sensors_used.append(f"camera_{color_name}")
                except Exception:
                    pass

        res.latency_ms = (time.time() - t0) * 1000.0
        return res

    # ------------------------------------------------------------------
    def is_critical_block(self, perception, threshold=0.20):
        """Highest-priority signal: hard imminent collision via IR."""
        return perception.front_clearance < threshold

    def lateral_clearance(self, perception):
        """Returns (best_side, best_clearance). best_side in {-1, +1}."""
        if perception.left_clearance >= perception.right_clearance:
            return +1, perception.left_clearance
        return -1, perception.right_clearance
