"""main9.py -- Robot Rescue: a clean, scientifically-grounded rebuild of the
warehouse navigation control loop.

WHY THIS FILE EXISTS
--------------------
main8 produced "dumb" behaviour -- the robot overshot goals, orbited, drove
into walls and backed out, and ran a 230 ms NMPC solve on every navigation tick
while the arm sat tucked and idle. Three root causes, each fixed here:

  1. OVERSHOOT / ORBIT.  main8's SmoothCAPP kept a forward "creep" while an
     angular slew-limit throttled the turn, so the base swept wide arcs that
     blew past the goal; the final-approach docking radius then never engaged
     and the robot circled. FIX: a proper unicycle pursuit law with a
     TURN-IN-PLACE GATE -- if the heading error exceeds a threshold the base
     pivots with zero forward speed; it only drives forward once roughly
     aligned. Geometrically it cannot orbit.

  2. DUELLING RECOVERY.  An orbit-detector and a 3-tier recovery fired against
     each other (14 stuck kicks + 10 orbit replans in one mission). FIX: a
     SINGLE ProgressWatchdog that measures closing speed toward the goal and
     issues one unambiguous recovery (reverse-and-reface), with a cooldown.

  3. 230 ms NMPC ON THE NAV HOT PATH.  The 7-state IPOPT solve ran every nav
     tick even though the arm was tucked and motionless. FIX: NMPC is solved
     ONLY during manipulation phases (robot stationary), so the navigation loop
     is a fast analytic controller. The energy-aware behaviour the thesis
     attributes to the NMPC speed law is preserved cheaply during navigation
     through the SoC weighting phi(z-bar) (thesis Eq. 3.25/3.31), which scales
     the reference speed -- identical equation, no solver in the loop.

PERCEPTION.  The RGB camera (perception/camera_vision.py) is actually USED here:
on final pick approach the colour-blob bearing refines the heading toward the
target box, instead of trusting odometry alone.

This file is SELF-CONTAINED for control logic -- all new controller classes live
here. It reuses only the proven, non-control infrastructure (robot, environment,
mission FSM, A*, camera, energy model) by importing the parent package. Nothing
in the parent tree is modified.

Run:
    python robot_rescue/main9.py            # GUI
    python robot_rescue/main9.py --no-gui   # headless
    python robot_rescue/main9.py --batch 20 # reliability sweep
"""

import argparse
import math
import os
import sys

import numpy as np
import yaml
import pybullet as p
import matplotlib.pyplot as plt

# Make the parent project importable when run as robot_rescue/main9.py.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# --- proven infrastructure (NOT control logic) -- reused unchanged ----------
from robot.mobile_manipulator import MobileManipulator
from simulation.environment import SimEnvironment
from perception.neural_detector import NeuralObstacleDetector
from perception.sensor_fusion import SensorFusion
from perception.camera_vision import CameraVision
from perception.occupancy_grid import OccupancyGrid
from visualization.dashboard import Dashboard
from visualization.advanced_dashboard import AdvancedDashboard

from control.nmpc_controller import NMPCController
from control.safety import SafetyFilter
from control.main3_helpers.shims import apply_shims, guard_array
from control.main3_helpers.warehouse_mission import WarehouseMission
from control.main3_helpers.dynamic_tracker import DynamicTracker
from control.main3_helpers.world_aware_astar import WorldAwareAStar
from control.main3_helpers.energy_paper import PhysicsEnergyManager
from control.main3_helpers.perception_fusion import PerceptionFusion
from control.main3_helpers.path_clearance import plan_with_clearance, los_safe


# ===========================================================================
#  GEOMETRY HELPERS
# ===========================================================================
def wrap_angle(a):
    """Wrap to [-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


def densify(path, spacing=0.15):
    """Resample a polyline so consecutive points are <= spacing apart, giving
    the pursuit law a well-sampled reference (no 3-point curl)."""
    if not path or len(path) < 2:
        return list(path) if path else []
    out = [tuple(path[0])]
    for (x0, y0), (x1, y1) in zip(path[:-1], path[1:]):
        seg = math.hypot(x1 - x0, y1 - y0)
        n = max(1, int(math.ceil(seg / max(spacing, 1e-3))))
        for k in range(1, n + 1):
            t = k / n
            out.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))
    return out


def nearest_obstacle_distance(state, obstacles):
    """Surface distance to the closest obstacle, or +inf if none."""
    d = float("inf")
    for ox, oy, r in obstacles:
        d = min(d, math.hypot(state["x"] - ox, state["y"] - oy) - r)
    return d


# ===========================================================================
#  ENERGY-AWARE PURSUIT  (thesis CAPP, implemented correctly)
# ===========================================================================
class EnergyAwarePursuit:
    """Curvature-adaptive pure pursuit with a turn-in-place gate.

    Thesis equations retained exactly:
      * adaptive lookahead  L_d = L_max - (L_max-L_min)*tanh(beta*|kappa|)  (3.27)
      * curvature speed law  v   = v_ref*(1 - gamma*tanh(beta*|kappa|))     (3.31)
      * SoC speed weighting  phi(z) = 1/(0.5 + 0.5 z + eps)                 (3.25)
        applied as v <- v / phi  so the robot slows as the battery depletes.

    The decisive change from main8's SmoothCAPP: there is NO forward "creep"
    during large turns and NO angular slew-limit. The controller is a clean
    two-mode unicycle law:

      |alpha| > turn_gate :  PIVOT  (v = 0, w = sign(alpha)*w_turn)
      |alpha| <= turn_gate:  DRIVE  (v from curvature law * cos(alpha) taper,
                                     w = pure-pursuit curvature command)

    Pivoting in place to acquire the heading, then driving when aligned, makes
    overshoot and orbiting geometrically impossible while keeping motion smooth
    (cos(alpha) tapers v->0 as the error grows, so the two modes blend).
    """

    def __init__(self, L_max=0.9, L_min=0.30, beta=8.0, v_ref=0.6, gamma=0.4,
                 w_max=2.5, turn_gate=0.6, w_turn=1.8, eps=0.01):
        self.L_max = float(L_max)
        self.L_min = float(L_min)
        self.beta = float(beta)
        self.v_ref = float(v_ref)
        self.gamma = float(gamma)
        self.w_max = float(w_max)
        self.turn_gate = float(turn_gate)   # rad; above this -> pivot in place
        self.w_turn = float(w_turn)         # pivot angular speed
        self.eps = float(eps)

    @staticmethod
    def menger_curvature(a, b, c):
        ab = (b[0] - a[0], b[1] - a[1])
        bc = (c[0] - b[0], c[1] - b[1])
        ca = (a[0] - c[0], a[1] - c[1])
        cross = ab[0] * bc[1] - ab[1] * bc[0]
        denom = math.hypot(*ab) * math.hypot(*bc) * math.hypot(*ca)
        if denom < 1e-9:
            return 0.0
        return 2.0 * abs(cross) / denom

    def _local_curvature(self, path, idx):
        if not (0 < idx < len(path) - 1):
            return 0.0
        ks = []
        for off in (-1, 0, 1):
            i = idx + off
            if 0 < i < len(path) - 1:
                ks.append(self.menger_curvature(path[i - 1], path[i], path[i + 1]))
        return sum(ks) / len(ks) if ks else 0.0

    def compute(self, state, path, soc=1.0, bearing_override=None):
        """Return (v, w, mode). `bearing_override` (rad, robot-frame) lets the
        camera steer the final approach; when given it replaces the path
        heading error."""
        if not path:
            return 0.0, 0.0, "idle"

        rx, ry, rth = state["x"], state["y"], state["theta"]

        # closest waypoint
        ci = min(range(len(path)),
                 key=lambda i: math.hypot(path[i][0] - rx, path[i][1] - ry))
        kappa = self._local_curvature(path, ci)
        L_d = self.L_max - (self.L_max - self.L_min) * math.tanh(self.beta * abs(kappa))

        # lookahead point: first point >= L_d ahead, else the goal
        look = path[-1]
        for i in range(ci, len(path)):
            if math.hypot(path[i][0] - rx, path[i][1] - ry) >= L_d:
                look = path[i]
                break

        if bearing_override is not None:
            alpha = wrap_angle(bearing_override)
        else:
            alpha = wrap_angle(math.atan2(look[1] - ry, look[0] - rx) - rth)

        # energy-aware reference speed (thesis 3.31 * 3.25)
        v_curve = self.v_ref * (1.0 - self.gamma * math.tanh(self.beta * abs(kappa)))
        phi = 1.0 / (0.5 + 0.5 * float(soc) + self.eps)
        v_ref = v_curve / max(phi, 1e-6)

        # --- two-mode unicycle law ---
        if abs(alpha) > self.turn_gate:
            # PIVOT: acquire heading first, no forward motion -> cannot arc/orbit
            return 0.0, math.copysign(self.w_turn, alpha), "pivot"

        # DRIVE: pure-pursuit curvature, speed tapered by heading alignment
        kappa_cmd = 2.0 * math.sin(alpha) / max(L_d, 1e-3)
        w = max(-self.w_max, min(self.w_max, kappa_cmd * v_ref))
        v = v_ref * max(0.0, math.cos(alpha))
        return v, w, "drive"


# ===========================================================================
#  REACTIVE AVOIDER  (CBF angular bias, thesis Eq. 3.33)
# ===========================================================================
class DockingController:
    """Precise final-approach. Within `engage_r` of the goal the mission
    spawner guarantees a clear ring (>=1.2 m kept clear around pick/place), so
    we drop all reactive layers and run a clean point-then-shoot law:

        |alpha| > face_tol : pivot to face the goal (v=0)
        else               : drive straight in, speed tapered to the arrival
                             radius so the base settles instead of overshooting.

    This is what closes the last half-metre the watchdog kept fighting. It also
    OWNS the goal region, so the livelock-prone watchdog/avoider are disabled
    here (see main loop)."""

    def __init__(self, engage_r=0.75, face_tol=0.35, v_max=0.35,
                 w_max=2.5, arrive_r=0.10):
        self.engage_r = float(engage_r)
        self.face_tol = float(face_tol)
        self.v_max = float(v_max)
        self.w_max = float(w_max)
        self.arrive_r = float(arrive_r)

    def in_range(self, state, goal_xy):
        return math.hypot(goal_xy[0]-state["x"], goal_xy[1]-state["y"]) < self.engage_r

    def compute(self, state, goal_xy):
        dx, dy = goal_xy[0]-state["x"], goal_xy[1]-state["y"]
        dist = math.hypot(dx, dy)
        if dist < self.arrive_r:
            return 0.0, 0.0
        alpha = wrap_angle(math.atan2(dy, dx) - state["theta"])
        if abs(alpha) > self.face_tol:
            return 0.0, max(-self.w_max, min(self.w_max, 2.0 * alpha))
        v = max(0.05, self.v_max * min(1.0, dist / self.engage_r))
        return v, 1.2 * alpha


class ReactiveAvoider:
    """Lightweight reactive layer. When the nearest obstacle is within d_safe
    AND ahead of the robot, it adds an angular bias steering away from the
    obstacle's lateral side (thesis CBF Eq. 3.33) and caps forward speed by an
    admissible braking limit (Eq. 3.32). No velocity-space search, no 5-degree
    quantisation -- a smooth analytic correction layered on the pursuit output."""

    def __init__(self, d_safe=0.45, k_cbf=2.2, a_max=1.5, w_max=2.5):
        self.d_safe = float(d_safe)
        self.k_cbf = float(k_cbf)
        self.a_max = float(a_max)
        self.w_max = float(w_max)

    def correct(self, v, w, state, obstacles):
        if not obstacles:
            return v, w, False
        rx, ry, rth = state["x"], state["y"], state["theta"]
        # nearest obstacle
        nearest, d_min = None, float("inf")
        for ox, oy, r in obstacles:
            d = math.hypot(ox - rx, oy - ry) - r
            if d < d_min:
                d_min, nearest = d, (ox, oy)
        if nearest is None or d_min >= self.d_safe:
            return v, w, False

        dx, dy = nearest[0] - rx, nearest[1] - ry
        c, s = math.cos(rth), math.sin(rth)
        x_r = c * dx + s * dy        # forward offset
        y_r = -s * dx + c * dy       # lateral offset
        active = False
        if x_r > 0.0:                # obstacle ahead
            # steer away from its lateral side; tanh smooths dead-ahead chatter
            w -= self.k_cbf * max(0.0, self.d_safe - d_min) * math.tanh(y_r / 0.10)
            # admissible braking speed (Eq. 3.32)
            v_adm = math.sqrt(max(0.0, 2.0 * self.a_max * max(0.0, d_min)))
            v = min(v, max(0.05, v_adm))
            active = True
        w = max(-self.w_max, min(self.w_max, w))
        return v, w, active


# ===========================================================================
#  PROGRESS WATCHDOG  (single, unambiguous recovery)
# ===========================================================================
class ProgressWatchdog:
    """Detects genuine livelock by tracking CLOSING distance to the goal over a
    window. Unlike main8's raw-motion StuckMonitor (which a circling robot
    fooled), this fires only when the robot is NOT getting closer. One recovery
    action -- reverse briefly and re-face the goal -- with a cooldown so it
    never fights itself."""

    def __init__(self, window_s=3.0, min_gain=0.15, dt=0.1, cooldown_s=2.0):
        self.window = max(3, int(window_s / dt))
        self.min_gain = float(min_gain)
        self.dt = float(dt)
        self.cooldown_steps = int(cooldown_s / dt)
        self.hist = []
        self.cool = 0
        self.kicks = 0

    def reset(self):
        self.hist.clear()
        self.cool = 0

    def update(self, state, goal_xy):
        if self.cool > 0:
            self.cool -= 1
            return False
        d = math.hypot(goal_xy[0] - state["x"], goal_xy[1] - state["y"])
        self.hist.append(d)
        if len(self.hist) > self.window:
            self.hist.pop(0)
        if len(self.hist) < self.window:
            return False
        if self.hist[0] - min(self.hist) < self.min_gain:
            self.hist.clear()
            self.cool = self.cooldown_steps
            self.kicks += 1
            return True
        return False


class RecoveryAction:
    """Time-boxed reverse-and-reface. Returns (v, w, done)."""

    def __init__(self, dt=0.1, reverse_s=0.6, v_back=-0.25):
        self.dt = float(dt)
        self.reverse_steps = int(reverse_s / dt)
        self.v_back = float(v_back)
        self.t = 0
        self.active = False
        self.turn_dir = 1.0

    def trigger(self, state, goal_xy):
        self.active = True
        self.t = 0
        alpha = wrap_angle(math.atan2(goal_xy[1] - state["y"],
                                      goal_xy[0] - state["x"]) - state["theta"])
        self.turn_dir = 1.0 if alpha >= 0 else -1.0

    def step(self):
        if not self.active:
            return 0.0, 0.0, True
        self.t += 1
        if self.t <= self.reverse_steps:
            return self.v_back, 0.8 * self.turn_dir, False
        self.active = False
        return 0.0, 0.0, True


# ===========================================================================
#  CHALLENGE MISSION  (same obstacle layout as main8, standalone)
# ===========================================================================
class ChallengeWarehouseMission(WarehouseMission):
    def __init__(self, config, env, robot, energy_mgr,
                 initial_soc=0.8, num_boxes=2, start=(0.0, 0.0, 0.0)):
        super().__init__(config, env, robot, energy_mgr,
                         initial_soc=initial_soc, num_boxes=num_boxes,
                         num_dynamic_obs=0, num_static_obs=0,
                         randomize=False, start=start)
        self._spawn_challenge_obstacles()

    def _place_target(self):
        """Reachable place standoff.

        The parent puts the place target at the CENTRE of a 0.6 m-wide storage
        platform and advances the FSM only at d < 0.55 m. A 0.3 m-radius base
        physically cannot bring its centre within 0.55 m of the platform centre
        without penetrating the platform -- it floors at ~0.6 m, one hair
        outside the gate, and the mission hangs forever (the earlier 'freeze').

        A mobile MANIPULATOR does not need to stand on the target: the arm has
        the reach. So we pull the placement point ~0.22 m back along the
        approach direction (place -> pick). The point still lands ON the
        platform (well within its 0.3 m half-extent), but now the robot can
        satisfy d < 0.55 m from a body-clear standoff. Geometry, not luck."""
        base = super()._place_target()
        try:
            approach = np.asarray(self.pick_pos, float)[:2] - np.asarray(self.place_pos, float)[:2]
            n = float(np.linalg.norm(approach))
            if n > 1e-6:
                base[:2] = base[:2] + (approach / n) * 0.22
        except Exception:
            pass
        return base

    def _spawn_challenge_obstacles(self):
        p1 = np.asarray(self.pick_pos, dtype=float)
        p2 = np.asarray(self.place_pos, dtype=float)
        vec = p2 - p1
        dist = float(np.linalg.norm(vec))
        if dist < 2.5:
            return
        unit = vec / dist
        perp = np.array([-unit[1], unit[0]])
        hard = [(self.start[0], self.start[1], 1.0),
                (self.pick_pos[0], self.pick_pos[1], 1.2),
                (self.place_pos[0], self.place_pos[1], 1.2)]

        def clear(x, y, m=0.3):
            return all(math.hypot(x - cx, y - cy) >= cr + m for cx, cy, cr in hard)

        for along, sign in ((0.35, +1), (0.65, -1)):
            sx = p1[0] + along * vec[0] + 0.85 * sign * perp[0]
            sy = p1[1] + along * vec[1] + 0.85 * sign * perp[1]
            if clear(sx, sy):
                self.env.add_obstacle(sx, sy, radius=0.22, height=0.5)
        print("[Challenge] strategic obstacles deployed.")


# ===========================================================================
#  MAIN LOOP
# ===========================================================================
def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def parse_args():
    ap = argparse.ArgumentParser(description="Robot Rescue (main9) clean nav rebuild")
    ap.add_argument("--gui", action="store_true", default=True)
    ap.add_argument("--no-gui", dest="gui", action="store_false")
    ap.add_argument("--config", type=str, default=os.path.join(_ROOT, "config.yaml"))
    ap.add_argument("--max-steps", type=int, default=30000)
    ap.add_argument("--no-dashboard", action="store_true", default=False)
    ap.add_argument("--initial-soc", type=float, default=0.8)
    ap.add_argument("--num-boxes", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use-camera", action="store_true", default=True,
                    help="Use RGB camera bearing to refine final pick approach.")
    ap.add_argument("--no-camera", dest="use_camera", action="store_false")
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--batch-out", type=str, default="main9_batch.csv")
    return ap.parse_args()


def seed_everything(seed):
    import random
    random.seed(int(seed))
    try:
        np.random.seed(int(seed))
    except Exception:
        pass
    os.environ["PYTHONHASHSEED"] = str(int(seed))


def run_batch(n, out, base):
    import subprocess, sys as _s, re, csv, time
    fields = ["seed", "success", "boxes", "num_boxes", "time_s", "path_m",
              "energy_wh", "final_soc", "nmpc_mean_ms", "recoveries"]
    out_path = out if os.path.isabs(out) else os.path.join(_HERE, out)

    def one(seed):
        cmd = [_s.executable, os.path.abspath(__file__), "--no-gui", "--no-dashboard",
               "--seed", str(seed), "--initial-soc", str(base.initial_soc),
               "--num-boxes", str(base.num_boxes), "--max-steps", str(base.max_steps),
               "--config", str(base.config)]
        env = dict(os.environ, PYTHONHASHSEED=str(seed), OMP_NUM_THREADS="1",
                   OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=900,
                               cwd=_HERE, env=env)
            o = (r.stdout or "") + "\n" + (r.stderr or "")
        except subprocess.TimeoutExpired:
            o = ""
        tail = o.rfind("Rescue Mission Summary")
        sc = o[tail:] if tail >= 0 else o
        def g(pat, d="0"):
            m = re.search(pat, sc)
            return m.group(1) if m else d
        row = {"seed": seed,
               "success": int("YES" in g(r"SUCCESS:\s+(YES|NO)", "NO")),
               "boxes": g(r"Boxes delivered:\s+(\d+)"),
               "num_boxes": base.num_boxes,
               "time_s": g(r"Total sim time:\s+([\d.]+)"),
               "path_m": g(r"Path length:\s+([\d.]+)"),
               "energy_wh": g(r"Energy consumed:\s+([\d.]+)"),
               "final_soc": g(r"Final SoC:\s+([\d.]+)"),
               "nmpc_mean_ms": g(r"NMPC solve.*mean=([\d.]+)"),
               "recoveries": g(r"Recoveries:\s+(\d+)")}
        print(f"[batch] seed={seed} success={row['success']} path={row['path_m']}m "
              f"t={row['time_s']}s nmpc={row['nmpc_mean_ms']}ms", flush=True)
        return row

    seeds = [1000 + k for k in range(n)]
    t0 = time.time()
    rows = [one(s) for s in seeds]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    ok = sum(int(r["success"]) for r in rows)
    def mean(k):
        vals = [float(r[k]) for r in rows if int(r["success"]) and r[k]]
        return sum(vals) / len(vals) if vals else 0.0
    print("=" * 60)
    print(f" main9 batch: {ok}/{n} success ({100*ok/max(n,1):.0f}%)  {time.time()-t0:.0f}s")
    print(f"  mean path {mean('path_m'):.2f} m | time {mean('time_s'):.1f} s | "
          f"nmpc {mean('nmpc_mean_ms'):.1f} ms")
    print(f"  CSV: {out_path}")
    print("=" * 60)
    return 0


def main():
    args = parse_args()
    config = load_config(args.config)
    seed_everything(args.seed)

    if args.batch and args.batch > 0:
        return run_batch(args.batch, args.batch_out, args)

    client = p.connect(p.GUI if args.gui else p.DIRECT)
    p.setRealTimeSimulation(0, physicsClientId=client)
    if args.gui:
        p.resetDebugVisualizerCamera(6, 0, -60, [0.5, 0.5, 0], physicsClientId=client)

    env = SimEnvironment(client, config)
    start = [0.0, 0.0, 0.0]
    robot = MobileManipulator(client, config, start_pos=[0.0, 0.0, 0.10],
                              start_orn=p.getQuaternionFromEuler([0, 0, 0]))

    # Controllers (control logic = new; everything else reused).
    pursuit = EnergyAwarePursuit(v_ref=0.6, turn_gate=0.6, w_turn=1.8)
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
    energy = PhysicsEnergyManager(config)
    apply_shims(energy)
    tracker = DynamicTracker(env)
    perception = PerceptionFusion(detector, fusion, camera, env, tracker)

    sim_dt = config["simulation"]["timestep"]
    nmpc_dt = config.get("nmpc", {}).get("dt", 0.1)
    ctrl_every = max(1, int(nmpc_dt / sim_dt))
    astar_period = ctrl_every * 12

    watchdog = ProgressWatchdog(window_s=3.0, min_gain=0.15, dt=nmpc_dt)
    recovery = RecoveryAction(dt=nmpc_dt)

    mission = ChallengeWarehouseMission(config, env, robot, energy,
                                        initial_soc=args.initial_soc,
                                        num_boxes=args.num_boxes, start=start)

    dashboard = None
    if not args.no_dashboard:
        try:
            dashboard = Dashboard(config)
            plt.show(block=False)
        except Exception as e:
            print(f"[main9] dashboard init failed: {e}")

    # BOX_COLORS index per picked box, for camera bearing.
    box_colors = getattr(camera, "BOX_COLORS", [(255, 255, 0), (255, 102, 0), (77, 153, 255)])

    print(f"[main9] Robot Rescue active. pursuit turn-gate={pursuit.turn_gate:.2f} rad, "
          f"NMPC arm-phase-only, camera={'on' if args.use_camera else 'off'}")
    print("-" * 60)

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
    arm_control = np.zeros(3)
    nmpc_prev = np.zeros(5)
    solve_time = 0.0
    predicted_traj = None
    sim_time = 0.0
    path_len = 0.0
    prev_xy = (state["x"], state["y"])
    nmpc_solves = 0
    dock_stall = 0
    dock_stall_limit = int(4.0 / nmpc_dt)   # ~4 s in-range without arriving

    try:
        for step in range(args.max_steps):
            try:
                p.getConnectionInfo(physicsClientId=client)
            except Exception:
                print("[Exit] GUI closed.")
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
                prev_phase = mission.phase

            soc = energy.get_soc()

            if step % ctrl_every == 0:
                perc = perception.perceive(robot, state, sim_time)
                obstacles = list(perc.obstacles)
                ir = perc.ir

                try:
                    occ.update_from_obstacles(obstacles)
                    occ.update_from_ir(state["x"], state["y"], state["theta"], ir)
                    occ.decay()
                    for so in occ.get_obstacles_list():
                        if not any(math.hypot(so[0]-o[0], so[1]-o[1]) < 0.3 for o in obstacles):
                            obstacles.append(so)
                except Exception:
                    pass

                if energy.is_emergency():
                    print(f"[Safety] battery emergency SoC={soc:.2%} -- stop.")
                    break

                # Filter self + goal-region detections (target box / stack).
                rx, ry = state["x"], state["y"]
                obstacles = [(ox, oy, r) for ox, oy, r in obstacles
                             if math.hypot(ox-rx, oy-ry) > 0.35
                             and math.hypot(ox-goal_xy[0], oy-goal_xy[1]) > 0.40]

                if mission.is_arm_phase:
                    # Manipulation: base stationary; the mission FSM drives the
                    # arm via move_arm_to. NMPC solves HERE ONLY (robot still),
                    # exercising the energy-aware manipulation model. This is the
                    # only place the 7-state IPOPT runs -> nav loop stays fast.
                    final_control = np.zeros(5)
                    try:
                        u, solve_time, pred = nmpc.solve(state["full_q"], goal,
                                                         obstacles, soc, u_prev=nmpc_prev)
                        nmpc_prev = u.copy()
                        predicted_traj = pred[:, :6]
                        nmpc_solves += 1
                    except Exception:
                        pass
                else:
                    # ---- NAVIGATION: fast analytic control (no NMPC) ----
                    # Replan only when path stale (empty / goal moved / blocked).
                    need = (not current_path or
                            (step - last_astar) >= astar_period or
                            (current_path and math.hypot(current_path[-1][0]-goal_xy[0],
                                                         current_path[-1][1]-goal_xy[1]) > 0.25))
                    if not need and current_path:
                        # blocked-ahead check
                        ci = min(range(len(current_path)),
                                 key=lambda i: math.hypot(current_path[i][0]-rx,
                                                          current_path[i][1]-ry))
                        for wx, wy in current_path[ci:]:
                            if any(math.hypot(wx-ox, wy-oy) < r+0.30 for ox, oy, r in obstacles):
                                need = True
                                break

                    if need:
                        if los_safe(state, goal_xy, obstacles, robot_radius=0.3, safety=0.35):
                            current_path = densify([(rx, ry), goal_xy], 0.15)
                        else:
                            np_, _ = plan_with_clearance(astar, (rx, ry), goal_xy, obstacles,
                                                         min_clearance=0.22, gap_min=0.80,
                                                         max_attempts=5, inflation_step=0.20)
                            current_path = densify(np_ if np_ else [(rx, ry), goal_xy], 0.15)
                        last_astar = step

                    # Camera RGB bearing override on final approach to a box.
                    bearing = None
                    dist_goal = math.hypot(goal_xy[0]-rx, goal_xy[1]-ry)
                    if (args.use_camera and mission.phase == "nav_to_pick"
                            and dist_goal < 1.5):
                        try:
                            idx = mission.pick_idx % len(box_colors)
                            det = camera.detect_object(state, box_colors[idx])
                            if det and det.get("detected"):
                                bearing = det["bearing"]
                        except Exception:
                            bearing = None

                    # CONTROL ARBITRATION (priority order):
                    #   1. active recovery  -> finish it
                    #   2. docking range    -> point-then-shoot; watchdog &
                    #      avoider are DISABLED here (clear ring guaranteed),
                    #      which is what stops the goal-region livelock.
                    #   3. cruise           -> pursuit + reactive avoider, with
                    #      the watchdog guarding against genuine livelock.
                    docking_now = False
                    if recovery.active:
                        bv, bw, _ = recovery.step()
                        final_control = np.array([bv, bw, 0, 0, 0])
                    elif docker.in_range(state, goal_xy):
                        # Docking owns the goal ring (provably clear). Skip
                        # reactive obstacle braking -- otherwise the carried box /
                        # place platform reads as an obstacle dead-ahead and the
                        # safety CBF freezes the approach.
                        docking_now = True
                        v, w = docker.compute(state, goal_xy)
                        final_control = np.array([v, w, 0, 0, 0])
                        # Bounded stall guard: if docking sits in-range without
                        # arriving for too long, fire ONE recovery. (The old code
                        # reset the watchdog every tick here, which removed the
                        # only escape exactly where stalls happen.)
                        dock_stall += 1
                        if dock_stall > dock_stall_limit:
                            recovery.trigger(state, goal_xy)
                            current_path = []
                            dock_stall = 0
                            print(f"[Recovery] docking stall -> reverse&reface", flush=True)
                            bv, bw, _ = recovery.step()
                            final_control = np.array([bv, bw, 0, 0, 0])
                    elif watchdog.update(state, goal_xy):
                        recovery.trigger(state, goal_xy)
                        current_path = []
                        print(f"[Recovery] livelock -> reverse&reface "
                              f"(kick {watchdog.kicks})")
                        bv, bw, _ = recovery.step()
                        final_control = np.array([bv, bw, 0, 0, 0])
                    else:
                        dock_stall = 0
                        v, w, mode = pursuit.compute(state, current_path, soc,
                                                     bearing_override=bearing)
                        v, w, _ = avoider.correct(v, w, state, obstacles)
                        final_control = np.array([v, w, 0, 0, 0])

                    try:
                        safe_obs = [] if docking_now else obstacles
                        final_control = safety.filter_control(final_control, state, safe_obs, soc)
                    except Exception:
                        pass

            # Apply
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
                power = 0.0

            try:
                p.stepSimulation(physicsClientId=client)
            except Exception:
                print("[Exit] step failed.")
                break
            env.update_dynamic_obstacles(sim_time)
            sim_time += sim_dt

            if dashboard and step % ctrl_every == 0:
                state["v"], state["omega"] = final_control[0], final_control[1]
                try:
                    dashboard.update(state, power, energy.get_soc(), solve_time,
                                     env.get_all_obstacles(sim_time), goal,
                                     predicted_traj, sim_time, energy_mgr=energy)
                except Exception:
                    pass

            if mission.is_done(state):
                break
            if step % 200 == 0:
                print(f"step {step:5d} | pos ({state['x']:+.2f},{state['y']:+.2f}) "
                      f"| SoC {soc*100:4.1f}% | phase {mission.phase}")

    except KeyboardInterrupt:
        print("\n[Interrupted]")
    except Exception as e:
        print(f"[Error] {e}")

    # --- summary ---
    try:
        success = mission.is_done(state) and mission.box_idx >= mission.num_boxes
        boxes = mission.box_idx
    except Exception:
        success, boxes = False, 0
    st = nmpc.get_solve_time_stats()
    print("\n" + "=" * 60)
    print(" Rescue Mission Summary (main9)")
    print("=" * 60)
    print(f"  Total sim time:        {sim_time:6.1f} s")
    print(f"  Boxes delivered:       {boxes} / {mission.num_boxes}")
    print(f"  Path length:           {path_len:6.2f} m")
    try:
        print(f"  Energy consumed:       {energy.energy_consumed:6.3f} Wh")
        print(f"  Final SoC:             {energy.get_soc()*100:5.1f}%")
        print(f"  Max cell temp:         {energy.max_cell_temp():5.2f} C")
    except Exception:
        pass
    print(f"  Recoveries:            {watchdog.kicks}")
    print(f"  NMPC solves (arm only):{nmpc_solves}")
    if st["count"] > 0:
        print(f"  NMPC solve time:       mean={st['mean']:.1f}ms  p99={st['p99']:.1f}ms  n={st['count']}")
    print(f"  SUCCESS:               {'YES' if success else 'NO'}")
    print("=" * 60)

    try:
        p.disconnect(client)
    except Exception:
        pass


if __name__ == "__main__":
    main()
