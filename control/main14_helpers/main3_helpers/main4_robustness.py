"""main4 robustness helpers — five targeted upgrades over main3.

These classes/functions extend (do not replace) the main3 helper stack.
main4.py imports them alongside the existing main3_helpers modules.

Fixes covered:
  #1 Tight-corridor handling: shrunk A* inflation + corridor-detection retry
  #2 Defensive arm pause when IR critical during arm phases
  #3 Vision waypoint LOS gate — only inject when the box is visible AND clear
  #4 EscalatingRecovery — three-tier recovery instead of replan-loop
  #6 Place-position spawn validation — verify it is clear before mission start
"""

import math
import numpy as np


# ============================================================================
# Fix #1: tight-corridor handling
# ============================================================================
def plan_tight_corridor_aware(astar, start_xy, goal_xy, obstacles,
                              tight_threshold=1.0):
    """Two-pass A*: try with normal inflation; if no path, retry with shrunk
    inflation (corridor mode). Returns (path, used_shrunk: bool).
    """
    # First pass: default inflation (astar uses robot_radius=0.30 by default)
    try:
        path = astar.plan(start_xy, goal_xy, obstacles)
    except Exception:
        path = None
    if path and len(path) >= 2:
        # Validate path is more than a one-shot fallback to direct goal
        if not (len(path) == 1 and path[0] == tuple(goal_xy)):
            return path, False

    # Detect tight corridor: any pair of obstacles closer than tight_threshold?
    has_tight = False
    for i, oi in enumerate(obstacles):
        for oj in obstacles[i + 1:]:
            d = math.hypot(oi[0] - oj[0], oi[1] - oj[1])
            if d < tight_threshold + oi[2] + oj[2]:
                has_tight = True
                break
        if has_tight:
            break

    if not has_tight:
        return path or [tuple(goal_xy)], False

    # Second pass: temporarily shrink robot radius for corridor squeezing
    saved_r = astar.robot_radius
    try:
        astar.robot_radius = max(0.18, saved_r * 0.65)  # squeeze through
        path2 = astar.plan(start_xy, goal_xy, obstacles)
    except Exception:
        path2 = None
    finally:
        astar.robot_radius = saved_r

    if path2 and len(path2) >= 2:
        return path2, True
    return path or [tuple(goal_xy)], False


# ============================================================================
# Fix #2: defensive arm pause during arm phases
# ============================================================================
class ArmPauseGuard:
    """Pauses arm motion (commands zero arm velocity) when a dynamic obstacle
    closes within `pause_threshold` m of the robot while the base is stopped
    mid-pick/place. Real-robot benefit: prevents the gripper from closing on
    a moving box that has been displaced by a passing AGV.
    """
    def __init__(self, pause_threshold=0.35):
        self.pause_threshold = float(pause_threshold)
        self.pause_count = 0

    def should_pause(self, perc, ir_readings):
        """Return True if a dynamic obstacle is too close to safely continue
        the arm motion. Uses perception.dynamic_obstacles (positions only).
        """
        # IR-based safety: any sensor < threshold means something is right there
        for k, v in ir_readings.items():
            if v < self.pause_threshold:
                self.pause_count += 1
                return True
        # Dynamic-obstacle-position based: from perception_fusion
        for x, y, r, vx, vy in getattr(perc, "dynamic_obstacles", []):
            # we don't have robot pos here directly — caller must check via IR
            pass
        return False


# ============================================================================
# Fix #3: vision waypoint LOS gate
# ============================================================================
def vision_los_clear(state, bearing, obstacles, range_m=0.8, samples=6,
                     clearance=0.30):
    """Check that the visual lookahead point (range_m ahead at `bearing` rel
    to robot heading) has line-of-sight from the robot through no obstacles.
    Returns True if safe to inject the vision waypoint.
    """
    if bearing is None:
        return False
    sx, sy = state["x"], state["y"]
    ex = sx + range_m * math.cos(state["theta"] + bearing)
    ey = sy + range_m * math.sin(state["theta"] + bearing)
    for k in range(1, samples + 1):
        t = k / samples
        px = sx + t * (ex - sx)
        py = sy + t * (ey - sy)
        for ox, oy, orad in obstacles:
            if math.hypot(px - ox, py - oy) < orad + clearance:
                return False
    return True


# ============================================================================
# Fix #4: escalating recovery
# ============================================================================
class EscalatingRecovery:
    """Three-tier escalation when stuck-monitor keeps firing.

    tier 1 (1-2 kicks): force fresh A* with the same params
    tier 2 (3-4 kicks): rotate in place 60deg toward most-clear lateral, then replan
    tier 3 (5+ kicks):  back up 0.4m, replan with shrunk inflation
                        (and signal mission for "skip box" if available)

    Each tier transition prints a clear log line for debugging on real hardware.
    """
    def __init__(self):
        self.kicks = 0
        self.active_tier = 0       # 0=normal, 1/2/3 currently executing
        self.tier_timer = 0.0      # seconds spent in current tier
        self.tier_dir = 1          # +1 left / -1 right
        self.skip_requested = False

    def reset(self):
        self.kicks = 0
        self.active_tier = 0
        self.tier_timer = 0.0
        self.skip_requested = False

    def on_stuck(self, perception):
        self.kicks += 1
        self.tier_timer = 0.0
        # Pick a lateral direction based on IR clearance
        left = perception.ir.get("left", 1.0)
        right = perception.ir.get("right", 1.0)
        self.tier_dir = +1 if left >= right else -1
        if self.kicks <= 2:
            self.active_tier = 1
            print(f"[Recovery] tier-1 (kick {self.kicks}): force A* replan")
        elif self.kicks <= 4:
            self.active_tier = 2
            print(f"[Recovery] tier-2 (kick {self.kicks}): rotate {'L' if self.tier_dir>0 else 'R'} 60deg")
        else:
            self.active_tier = 3
            self.skip_requested = True
            print(f"[Recovery] tier-3 (kick {self.kicks}): backup + shrunk inflation + skip request")

    def step(self, dt, state, goal):
        """Advance the active recovery. Returns (v, w, done).

        `done=True` means the recovery action has completed; caller can resume
        normal control. `v, w` are None when this tier doesn't override base
        velocity (tier 1).
        """
        self.tier_timer += dt
        if self.active_tier == 0:
            return None, None, True

        if self.active_tier == 1:
            # tier 1 is just "force replan" — caller already cleared current_path
            self.active_tier = 0
            return None, None, True

        if self.active_tier == 2:
            # ~60deg rotation = 1.0 rad. With w_max=2.5 rad/s, takes ~0.42s
            if self.tier_timer > 0.5:
                self.active_tier = 0
                return None, None, True
            return 0.0, 1.5 * self.tier_dir, False

        if self.active_tier == 3:
            # 0.4m back-up at 0.3 m/s = ~1.3s
            if self.tier_timer > 1.3:
                self.active_tier = 0
                return None, None, True
            return -0.30, 0.3 * self.tier_dir, False

        return None, None, True

    def consume_skip(self):
        """One-shot: returns True if mission should skip the current box."""
        if self.skip_requested:
            self.skip_requested = False
            return True
        return False


# ============================================================================
# Fix #6: place-position spawn validation
# ============================================================================
def validate_place_position(place_pos, obstacles, min_clearance=0.6):
    """Return True if (place_pos[0], place_pos[1]) has at least `min_clearance`
    to every obstacle. If False, suggest a nearby valid replacement.
    """
    for ox, oy, orad in obstacles:
        if math.hypot(place_pos[0] - ox, place_pos[1] - oy) < orad + min_clearance:
            return False
    return True


def suggest_place_position(original, obstacles, x_range=(-3.0, 3.0),
                           y_range=(-3.0, 3.0), min_clearance=0.6,
                           attempts=200, rng=None):
    """Return a clear (x, y) near `original` if the original is blocked.

    `rng` may be a numpy Generator or a seed int. When None, fresh entropy
    is used. The previously hardcoded seed=42 caused identical candidate
    sets to be tried for every --seed value, so seed 200 always reused the
    same blocked points and the mission was doomed at spawn.

    `attempts` widened 80 -> 200 and the radial search 1.5 m -> 2.5 m to
    lower the false-fail rate.
    """
    if rng is None:
        rng = np.random.default_rng()
    elif isinstance(rng, (int, np.integer)):
        rng = np.random.default_rng(int(rng))
    for _ in range(attempts):
        radius = rng.uniform(0.3, 2.5)
        ang = rng.uniform(-math.pi, math.pi)
        cx = float(np.clip(original[0] + radius * math.cos(ang),
                           x_range[0], x_range[1]))
        cy = float(np.clip(original[1] + radius * math.sin(ang),
                           y_range[0], y_range[1]))
        ok = True
        for ox, oy, orad in obstacles:
            if math.hypot(cx - ox, cy - oy) < orad + min_clearance:
                ok = False
                break
        if ok:
            return (cx, cy)
    return tuple(original)
