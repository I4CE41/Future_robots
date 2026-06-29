"""Ground-truth dynamic-obstacle tracker.

Reads `env.dynamic_obstacles` directly. Each obstacle moves as
    pos(t) = origin + amp * sin(2*pi*freq*t) along its direction axis.
So position AND velocity are analytic — no Kalman filter, no estimator.
This is correct because we are in simulation and the env owns the motion model.
"""

import math
import numpy as np


class DynamicTracker:
    def __init__(self, env):
        self.env = env

    def _omega(self, freq):
        return 2.0 * math.pi * freq

    def get_at(self, t):
        """Return list of (x, y, r, vx, vy) for each dynamic obstacle at world time t."""
        out = []
        for d in getattr(self.env, "dynamic_obstacles", []):
            amp = d["amp"]
            freq = d["freq"]
            r = d["radius"]
            w = self._omega(freq)
            phase = w * t
            if d["dir"] == "x":
                x = d["x0"] + amp * math.sin(phase)
                y = d["y0"]
                vx = amp * w * math.cos(phase)
                vy = 0.0
            else:
                x = d["x0"]
                y = d["y0"] + amp * math.sin(phase)
                vx = 0.0
                vy = amp * w * math.cos(phase)
            out.append((x, y, r, vx, vy))
        return out

    def predict(self, t, dt_ahead):
        """Position-only list of (x, y, r) at t + dt_ahead. Used for planning."""
        out = []
        for d in getattr(self.env, "dynamic_obstacles", []):
            amp = d["amp"]
            freq = d["freq"]
            r = d["radius"]
            phase = self._omega(freq) * (t + dt_ahead)
            if d["dir"] == "x":
                x = d["x0"] + amp * math.sin(phase)
                y = d["y0"]
            else:
                x = d["x0"]
                y = d["y0"] + amp * math.sin(phase)
            out.append((x, y, r))
        return out

    def sweep(self, t, horizon=1.5, dt=0.3):
        """Return a flat list of (x, y, r) samples along the predicted trajectory
        of every dynamic obstacle over [t, t+horizon].
        """
        samples = []
        steps = max(1, int(horizon / dt))
        for k in range(steps + 1):
            samples.extend(self.predict(t, k * dt))
        return samples

    def closest_to(self, point, t):
        """Return (dx, dy, r, vx, vy, dist) for the dynamic obstacle nearest to `point`,
        or None if no dynamic obstacles exist.
        """
        best = None
        for x, y, r, vx, vy in self.get_at(t):
            d = math.hypot(point[0] - x, point[1] - y)
            if best is None or d < best[5]:
                best = (x, y, r, vx, vy, d)
        return best

    def time_to_clear(self, point, t, clearance=0.45, max_lookahead=3.0, dt=0.1):
        """How many seconds until the nearest dynamic obstacle is more than
        `clearance` away from `point`? Returns max_lookahead if it never clears in window,
        or 0.0 if there are no dynamics.
        """
        if not getattr(self.env, "dynamic_obstacles", []):
            return 0.0
        steps = int(max_lookahead / dt)
        for k in range(steps + 1):
            dt_ahead = k * dt
            blocked = False
            for x, y, r in self.predict(t, dt_ahead):
                if math.hypot(point[0] - x, point[1] - y) - r < clearance:
                    blocked = True
                    break
            if not blocked:
                return dt_ahead
        return max_lookahead
