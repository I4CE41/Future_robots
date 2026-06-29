"""Patched watchdog and recovery for corridor robustness.

CorridorAwareWatchdog (root cause: main9.py ProgressWatchdog:300-335)
---------------------------------------------------------------------
The base ProgressWatchdog fires when closing-distance gain over a 3 s window
is below a FIXED 0.15 m. In a narrow corridor the robot is legitimately
throttled (velocity_scale -> 0.25), so it can make < 0.15 m / 3 s while still
making honest progress, producing a FALSE livelock kick that then triggers a
blind reverse into a wall. The fix scales the required gain by the corridor
velocity_scale and skips the kick when the robot is still physically moving.

WallAwareRecovery (root cause: main9.py RecoveryAction:338-363)
---------------------------------------------------------------
The base RecoveryAction always reverses at v=-0.25 for 0.6 s with no check of
what is BEHIND the robot -- in a corridor that backs the rear straight into a
wall. The fix measures rear clearance along -heading and, when the rear is
blocked, pivots in place toward the freer side instead of reversing.
"""

import math

from robot_rescue.main9 import ProgressWatchdog, RecoveryAction, wrap_angle


class CorridorAwareWatchdog(ProgressWatchdog):
    def __init__(self, window_s=3.0, min_gain=0.15, dt=0.1, cooldown_s=2.0,
                 min_scale=0.35, motion_eps=0.04):
        super().__init__(window_s=window_s, min_gain=min_gain, dt=dt,
                         cooldown_s=cooldown_s)
        # Never relax the required gain below min_scale * min_gain, so a truly
        # stalled robot is still caught even in the tightest corridor.
        self.min_scale = float(min_scale)
        # If the robot still moves at least this much per step (raw xy), it is
        # not livelocked -- skip the kick regardless of goal-closing rate.
        self.motion_eps = float(motion_eps)
        self._prev_xy = None

    def update(self, state, goal_xy, velocity_scale=1.0):
        # Track raw displacement so legitimate slow crawling is not a "stall".
        xy = (state["x"], state["y"])
        moved = 0.0
        if self._prev_xy is not None:
            moved = math.hypot(xy[0] - self._prev_xy[0], xy[1] - self._prev_xy[1])
        self._prev_xy = xy

        if self.cool > 0:
            self.cool -= 1
            return False
        d = math.hypot(goal_xy[0] - state["x"], goal_xy[1] - state["y"])
        self.hist.append(d)
        if len(self.hist) > self.window:
            self.hist.pop(0)
        if len(self.hist) < self.window:
            return False

        scale = max(self.min_scale, min(1.0, float(velocity_scale)))
        effective_gain = self.min_gain * scale
        gained = self.hist[0] - min(self.hist)

        # Only declare livelock if BOTH: goal-closing gain is too small AND the
        # robot is barely moving in raw xy. A slow-but-moving robot threading a
        # corridor satisfies the first but not the second, so it is spared.
        if gained < effective_gain and moved < self.motion_eps:
            self.hist.clear()
            self.cool = self.cooldown_steps
            self.kicks += 1
            return True
        return False


class WallAwareRecovery(RecoveryAction):
    def __init__(self, dt=0.1, reverse_s=0.6, v_back=-0.25,
                 rear_block_dist=0.40, pivot_w=1.2):
        super().__init__(dt=dt, reverse_s=reverse_s, v_back=v_back)
        self.rear_block_dist = float(rear_block_dist)
        self.pivot_w = float(pivot_w)
        self._mode = "reverse"   # or "pivot"

    @staticmethod
    def _clearance_along(state, ux, uy, obstacles, max_d=1.2):
        """Min surface distance to any obstacle along a ray from the robot."""
        rx, ry = state["x"], state["y"]
        best = max_d
        for ox, oy, r in obstacles:
            vx, vy = ox - rx, oy - ry
            proj = vx * ux + vy * uy
            if proj < 0:
                continue
            perp = abs(vx * uy - vy * ux)
            if perp > r + 0.30:           # ray misses this inflated obstacle
                continue
            along = proj - math.sqrt(max(0.0, (r + 0.30) ** 2 - perp ** 2))
            if 0.0 <= along < best:
                best = along
        return best

    def trigger(self, state, goal_xy, obstacles=None):
        self.active = True
        self.t = 0
        alpha = wrap_angle(math.atan2(goal_xy[1] - state["y"],
                                      goal_xy[0] - state["x"]) - state["theta"])
        self.turn_dir = 1.0 if alpha >= 0 else -1.0
        # Decide reverse vs pivot up front based on rear clearance.
        self._mode = "reverse"
        if obstacles:
            theta = state["theta"]
            rear = self._clearance_along(state, -math.cos(theta), -math.sin(theta),
                                         obstacles)
            if rear < self.rear_block_dist:
                self._mode = "pivot"
                # Pivot toward whichever side has more room.
                left = self._clearance_along(state, -math.sin(theta), math.cos(theta),
                                             obstacles)
                right = self._clearance_along(state, math.sin(theta), -math.cos(theta),
                                              obstacles)
                self.turn_dir = 1.0 if left >= right else -1.0

    def step(self):
        if not self.active:
            return 0.0, 0.0, True
        self.t += 1
        if self.t <= self.reverse_steps:
            if self._mode == "pivot":
                # Rear is blocked: rotate in place to find a new heading instead
                # of backing into the wall.
                return 0.0, self.pivot_w * self.turn_dir, False
            return self.v_back, 0.8 * self.turn_dir, False
        self.active = False
        return 0.0, 0.0, True
