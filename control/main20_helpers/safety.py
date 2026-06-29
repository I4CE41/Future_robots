"""Safety filter: CBF constraints, joint limits, emergency stop."""

import numpy as np


class SafetyFilter:
    def __init__(self, config):
        self.cfg = config["safety"]
        self.robot_cfg = config["robot"]
        self.energy_cfg = config["energy"]
        self.alpha = self.cfg["cbf_alpha"]
        self.min_dist = self.cfg["min_obstacle_distance"]
        self.emergency_soc = self.cfg["emergency_soc_threshold"]
        self.joint_margin = self.cfg["joint_limit_margin"]

    def filter_control(self, u, state, obstacles, soc):
        """Apply safety constraints to control input.
        u: [v, omega, dq1, dq2, dq3]
        state: dict from robot.get_state()
        obstacles: list of (x, y, r)
        soc: battery state of charge
        Returns filtered control vector.
        """
        u = np.array(u, dtype=float)

        # Emergency stop
        if soc < self.emergency_soc:
            return np.zeros(5)

        # Velocity clamping
        v_lim = self.robot_cfg["velocity_limits"]
        u[0] = np.clip(u[0], -v_lim["base_linear"], v_lim["base_linear"])
        u[1] = np.clip(u[1], -v_lim["base_angular"], v_lim["base_angular"])
        for i in range(3):
            u[2 + i] = np.clip(u[2 + i], -v_lim["arm_joints"], v_lim["arm_joints"])

        # Low SoC speed reduction
        if soc < self.energy_cfg["low_soc_threshold"]:
            factor = self.energy_cfg["low_soc_speed_factor"]
            u *= factor

        # CBF-based obstacle avoidance.
        #
        # BUG FIX (root cause of repeated corridor-wall contact): this used to
        # mutate u[0] obstacle-by-obstacle inside the loop. Two problems:
        #
        #   1. `u[0] = max(0, u[0])` was meant to stop the robot driving
        #      FORWARD into something it's already touching, but max() with a
        #      floor of 0 instead clamps a NEGATIVE (backward/escape) command
        #      UP to zero -- i.e. it did the opposite of what the comment
        #      says, and actively cancelled any backing-away velocity that
        #      RecoveryAction / the wedge-escape controller had just issued.
        #      In a corridor the robot is squeezed by walls on both sides, so
        #      this ran for every flanking obstacle in the same tick, re-zeroing
        #      the escape velocity each time -- the robot could press against
        #      a wall but could never command itself back off it.
        #   2. The `dist <= 1e-6` fallback set push_dir to the robot's own
        #      heading, which makes push_local == 1.0 (maximum FORWARD push)
        #      regardless of which side the obstacle is actually on.
        #
        # Fix: scan all obstacles once against the ORIGINAL commanded u[0],
        # aggregate the worst-case correction, and apply it once. Forward
        # motion into something already violated is capped at zero; backward
        # (escape) motion is always preserved and reinforced, never blocked.
        x, y = state["x"], state["y"]
        theta = state["theta"]
        heading = np.array([np.cos(theta), np.sin(theta)])
        u0_cmd = float(u[0])  # commanded forward velocity before correction

        in_hard_violation = False
        escape_push = 0.0       # depth-weighted, aggregated escape contribution
        soft_scale = 1.0        # most restrictive CBF scale across all obstacles

        for obs in obstacles:
            ox, oy, r = obs[0], obs[1], obs[2]
            dx, dy = x - ox, y - oy
            dist = float(np.sqrt(dx ** 2 + dy ** 2))
            h = dist - r - self.min_dist  # barrier function

            if h < 0:
                in_hard_violation = True
                depth = min(-h, 0.5)  # cap a single obstacle's influence
                if dist > 1e-6:
                    # World-frame unit vector pointing obstacle -> robot
                    # (i.e. "away from the obstacle").
                    push_dir = np.array([dx / dist, dy / dist])
                    push_local = float(push_dir @ heading)
                else:
                    # Obstacle center ~coincides with robot center: no usable
                    # direction. Contribute nothing rather than guessing
                    # "forward" (which used to drive the robot further into
                    # whatever it's centered on).
                    push_local = 0.0
                escape_push += depth * push_local
            elif h < self.min_dist and dist > 1e-6:
                # Soft CBF band: Lfh + alpha*h >= 0, evaluated against the
                # ORIGINAL command so obstacle order doesn't matter.
                dh_dx, dh_dy = dx / dist, dy / dist
                Lfh = (dh_dx * heading[0] + dh_dy * heading[1]) * u0_cmd
                if Lfh + self.alpha * h < 0:
                    scale = max(0.0, 1.0 + (Lfh + self.alpha * h) / (abs(u0_cmd) + 1e-6))
                    soft_scale = min(soft_scale, scale)

        if in_hard_violation:
            # Never add forward motion into something we're already
            # touching/penetrating, but always allow -- and reinforce --
            # backing away from it.
            u[0] = min(0.0, u0_cmd) + 0.3 * escape_push
        else:
            u[0] = u0_cmd * soft_scale

        # Re-clip after the CBF/escape correction: the escape push above is
        # an additive term and can otherwise exceed the platform's velocity
        # limit when several obstacles are in violation at once.
        u[0] = np.clip(u[0], -v_lim["base_linear"], v_lim["base_linear"])

        # Joint limit enforcement
        j_lim = self.robot_cfg["joint_limits"]
        limits = [j_lim["shoulder"], j_lim["elbow"], j_lim["wrist"]]
        arm_q = state["arm_q"]
        for i, (lo, hi) in enumerate(limits):
            q = arm_q[i]
            if q >= hi - self.joint_margin and u[2 + i] > 0:
                u[2 + i] = 0
            elif q <= lo + self.joint_margin and u[2 + i] < 0:
                u[2 + i] = 0

        return u