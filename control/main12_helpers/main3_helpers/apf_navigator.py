"""Artificial Potential Field navigator with anti-stuck watchdog.

Replaces pure_pursuit + reaction FSM with a single proven approach:

  attractive force: pulls robot toward goal (linear in distance, capped)
  repulsive force:  pushes from every obstacle, growing as 1/d
  resulting force:  becomes desired velocity in world frame
  desired heading:  atan2(Fy, Fx)
  output:          (v_linear, v_angular) in robot frame

Plus a deterministic anti-stuck watchdog:
  if not moved 0.15m in 1.5s -> rotate 90 deg in a chosen direction for 1s,
  then push along that lateral until cleared.

Why this works where pure-pursuit + planner failed:
  - No path commitment, no replanning hesitation.
  - Continuous gradient -> no discrete states to ping-pong between.
  - Dynamic obstacles automatically push the robot every tick (they don't need
    to be predicted; they ARE the field).
  - The watchdog gives the robot memory: if it hasn't moved, try something
    qualitatively different.
"""

import math


class APFNavigator:
    def __init__(self,
                 k_att=1.2,          # attractive gain
                 k_rep=0.40,         # repulsive gain
                 d_influence=1.2,    # obstacle influence radius (m)
                 d_safe=0.30,        # obstacle hard buffer (m)
                 v_max=0.55,         # max linear speed (m/s)
                 w_max=1.6,          # max angular speed (rad/s)
                 nmpc_dt=0.1):
        self.k_att = float(k_att)
        self.k_rep = float(k_rep)
        self.d_influence = float(d_influence)
        self.d_safe = float(d_safe)
        self.v_max = float(v_max)
        self.w_max = float(w_max)
        self.dt = float(nmpc_dt)

        # Watchdog state
        self.stuck_hist = []           # (sim_time, x, y)
        self.stuck_window = 1.5        # seconds
        self.stuck_threshold = 0.15    # meters of progress required
        self.escape_until = 0.0        # sim_time
        self.escape_w = 0.0            # rotation sign during escape
        self.escape_v = 0.0            # forward speed during escape
        self.escape_phase = "idle"     # "rotate" | "push" | "idle"
        self.escape_count = 0

        # Heading-priority threshold (rotate in place if goal more than this deg behind)
        self.rotate_in_place_deg = 100.0

    # ---------------------------------------------------------------------
    def _compute_force(self, state, goal, obstacles):
        """Return world-frame (Fx, Fy) — attractive minus repulsive."""
        rx, ry = state["x"], state["y"]
        gx, gy = goal[0], goal[1]

        # Attractive (linear, capped)
        dx, dy = gx - rx, gy - ry
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            ax = ay = 0.0
        else:
            mag = self.k_att * min(dist, 2.5)
            ax = mag * dx / dist
            ay = mag * dy / dist

        # Repulsive: sum of contributions from each obstacle in influence range
        rx_f = ry_f = 0.0
        for obs in obstacles:
            ox, oy = obs[0], obs[1]
            o_r = obs[2] if len(obs) > 2 else 0.0
            ddx, ddy = rx - ox, ry - oy
            d = math.hypot(ddx, ddy) - o_r
            d_eff = max(d, 0.05)
            if d_eff >= self.d_influence:
                continue
            # Gradient of (1/d - 1/d_inf)^2 -> magnitude k_rep * (1/d - 1/d_inf) / d^2
            mag = self.k_rep * (1.0 / d_eff - 1.0 / self.d_influence) / (d_eff * d_eff)
            if d > 1e-6:
                ux = ddx / max(math.hypot(ddx, ddy), 1e-6)
                uy = ddy / max(math.hypot(ddx, ddy), 1e-6)
            else:
                # Inside obstacle - push outward in any direction
                ux, uy = 1.0, 0.0
            rx_f += mag * ux
            ry_f += mag * uy

        return ax + rx_f, ay + ry_f

    # ---------------------------------------------------------------------
    def _update_watchdog(self, sim_time, state):
        """Append current position, prune old, decide if stuck."""
        self.stuck_hist.append((sim_time, state["x"], state["y"]))
        # Prune entries older than the window
        cutoff = sim_time - self.stuck_window
        while self.stuck_hist and self.stuck_hist[0][0] < cutoff:
            self.stuck_hist.pop(0)
        if len(self.stuck_hist) < 5:
            return False
        # Check displacement across window
        _, ox, oy = self.stuck_hist[0]
        moved = math.hypot(state["x"] - ox, state["y"] - oy)
        return moved < self.stuck_threshold

    def _enter_escape(self, sim_time, state, ir_readings):
        """Pick a lateral direction with more clearance, rotate then push."""
        left = ir_readings.get("left", 1.0)
        right = ir_readings.get("right", 1.0)
        rear_left = ir_readings.get("rear_left", 1.0)
        rear_right = ir_readings.get("rear_right", 1.0)
        # Prefer the side with most combined clearance
        left_score = left + 0.5 * rear_left
        right_score = right + 0.5 * rear_right
        self.escape_w = 1.0 if left_score >= right_score else -1.0
        # If both sides blocked, try going backward
        if max(left_score, right_score) < 0.5:
            self.escape_v = -0.30
        else:
            self.escape_v = 0.20
        self.escape_until = sim_time + 1.5
        self.escape_phase = "rotate"
        self.escape_count += 1
        print(f"[APF] STUCK #{self.escape_count} - escape: rotate {'L' if self.escape_w > 0 else 'R'} "
              f"v={self.escape_v:.2f}")
        self.stuck_hist.clear()

    # ---------------------------------------------------------------------
    def compute(self, state, goal, obstacles, sim_time, ir_readings):
        """Return (v_linear, v_angular). Always non-paralyzing - never returns (0, 0)
        unless the robot is exactly at the goal.
        """
        # 1) Active escape maneuver?
        if sim_time < self.escape_until:
            if self.escape_phase == "rotate":
                # Spin in place for first 0.5s
                if sim_time > self.escape_until - 1.0:
                    self.escape_phase = "push"
                return 0.0, self.w_max * self.escape_w * 0.8
            else:
                # Push laterally - small forward + curving
                return self.escape_v, self.w_max * self.escape_w * 0.4
        else:
            if self.escape_phase != "idle":
                self.escape_phase = "idle"
                print("[APF] escape done -> resume APF")
                self.stuck_hist.clear()  # give APF a fresh window

        # 2) Stuck watchdog
        if self._update_watchdog(sim_time, state):
            self._enter_escape(sim_time, state, ir_readings)
            return 0.0, self.w_max * self.escape_w * 0.8

        # 3) Compute APF force
        fx, fy = self._compute_force(state, goal, obstacles)
        f_mag = math.hypot(fx, fy)

        if f_mag < 1e-3:
            # Already at goal
            return 0.0, 0.0

        # Desired heading from force vector
        desired_theta = math.atan2(fy, fx)
        heading_err = math.atan2(
            math.sin(desired_theta - state["theta"]),
            math.cos(desired_theta - state["theta"]))

        # 4) Heading priority: rotate in place if goal is far behind
        if abs(heading_err) > math.radians(self.rotate_in_place_deg):
            w = math.copysign(self.w_max, heading_err)
            return 0.0, w

        # 5) Combined linear + angular
        # Linear velocity: f_mag scaled, but reduced when heading is misaligned
        align = math.cos(heading_err)  # in [-1, 1]
        # Only allow forward motion if reasonably aligned
        v = self.v_max * max(0.0, align) * min(1.0, f_mag / 2.0)
        # IR hard slowdown if front is very close (defensive layer)
        front_min = min(
            ir_readings.get("front_center", 1.0),
            ir_readings.get("front_left", 1.0),
            ir_readings.get("front_right", 1.0),
        )
        if front_min < 0.25:
            v *= max(0.0, (front_min - 0.10) / 0.15)

        # Angular velocity: P-control on heading error
        w = 2.0 * heading_err
        w = max(-self.w_max, min(self.w_max, w))

        return v, w

    # ---------------------------------------------------------------------
    def reset_for_new_goal(self):
        """Call when mission phase changes - clear stuck history."""
        self.stuck_hist.clear()
        self.escape_until = 0.0
        self.escape_phase = "idle"
