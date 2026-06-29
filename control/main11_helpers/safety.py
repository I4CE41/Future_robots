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

        # CBF-based obstacle avoidance
        x, y = state["x"], state["y"]
        for obs in obstacles:
            ox, oy, r = obs[0], obs[1], obs[2]
            dx, dy = x - ox, y - oy
            dist = np.sqrt(dx ** 2 + dy ** 2)
            h = dist - r - self.min_dist  # barrier function

            if h < 0:
                # Already in violation - push away
                if dist > 1e-6:
                    push_dir = np.array([dx / dist, dy / dist])
                    u[0] = max(0, u[0])  # no forward into obstacle
                else:
                    push_dir = np.array([1.0, 0.0])
                u[0] += 0.3 * push_dir[0]
            elif h < self.min_dist:
                # CBF constraint: Lfh + alpha*h >= 0
                if dist > 1e-6:
                    dh_dx = dx / dist
                    dh_dy = dy / dist
                    theta = state["theta"]
                    # Lie derivative: dh/dt = dh_dx * v*cos(theta) + dh_dy * v*sin(theta)
                    Lfh = dh_dx * u[0] * np.cos(theta) + dh_dy * u[0] * np.sin(theta)
                    if Lfh + self.alpha * h < 0:
                        # Reduce forward velocity to satisfy constraint
                        scale = max(0, 1 + (Lfh + self.alpha * h) / (abs(u[0]) + 1e-6))
                        u[0] *= scale

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
