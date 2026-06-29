"""PatchedSafetyFilter -- corridor-aware CBF filter.

ROOT CAUSE (original safety.py:104-108)
---------------------------------------
The original filter set `in_hard_violation = True` for ANY obstacle with
barrier h = dist - r - min_dist < 0, then forced

    u[0] = min(0.0, u0_cmd) + 0.3 * escape_push

A corridor wall sits BESIDE the robot, so its world->robot push vector is
nearly perpendicular to the heading: push_local = push_dir . heading ~= 0.
With walls on both sides the two contributions also cancel, so
escape_push ~= 0 and u[0] collapses to min(0, u0_cmd) == 0. Forward velocity
is zeroed even though the path AHEAD is clear; omega stays free, so the robot
spins/creeps in place against the wall -- the "stuck near corridor" symptom.

FIX
---
Split violating obstacles into FRONTAL (inside a forward cone) vs LATERAL
(off to the side). Only a frontal block should stop forward motion. When the
squeeze is purely lateral, keep a forward creep so the robot drives THROUGH
the corridor, and add a gentle centering steer toward the freer side. The
barrier math is unchanged; only the post-loop decision differs.
"""

import math

import numpy as np

from robot_rescue.control.main12_helpers.safety import SafetyFilter


class PatchedSafetyFilter(SafetyFilter):
    def __init__(self, config, front_cos=0.30, creep_v=0.12):
        super().__init__(config)
        # An obstacle counts as "frontal" when the unit vector from it to the
        # robot points back toward the robot's heading by less than this much,
        # i.e. the robot is driving into it. push_local = (obs->robot) . heading;
        # for something dead ahead push_local ~ -1, for something beside ~ 0.
        self.front_cos = float(front_cos)
        self.creep_v = float(creep_v)

    def filter_control(self, u, state, obstacles, soc):
        u = np.array(u, dtype=float)

        # Emergency stop (identical to base).
        if soc < self.emergency_soc:
            return np.zeros(5)

        v_lim = self.robot_cfg["velocity_limits"]
        u[0] = np.clip(u[0], -v_lim["base_linear"], v_lim["base_linear"])
        u[1] = np.clip(u[1], -v_lim["base_angular"], v_lim["base_angular"])
        for i in range(3):
            u[2 + i] = np.clip(u[2 + i], -v_lim["arm_joints"], v_lim["arm_joints"])

        if soc < self.energy_cfg["low_soc_threshold"]:
            factor = self.energy_cfg["low_soc_speed_factor"]
            u *= factor

        x, y = state["x"], state["y"]
        theta = state["theta"]
        heading = np.array([np.cos(theta), np.sin(theta)])
        u0_cmd = float(u[0])

        frontal_violation = False
        lateral_violation = False
        escape_push = 0.0          # forward-projected escape (frontal obstacles)
        lateral_signed = 0.0       # +left / -right squeeze imbalance for centering
        soft_scale = 1.0

        for obs in obstacles:
            ox, oy, r = obs[0], obs[1], obs[2]
            dx, dy = x - ox, y - oy
            dist = float(math.hypot(dx, dy))
            h = dist - r - self.min_dist

            if h < 0:
                depth = min(-h, 0.5)
                if dist > 1e-6:
                    push_dir = np.array([dx / dist, dy / dist])
                    push_local = float(push_dir @ heading)   # ~ -1 frontal, ~0 lateral
                    # lateral component in robot frame (+left, -right)
                    lat = float(-heading[1] * push_dir[0] + heading[0] * push_dir[1])
                else:
                    push_local = 0.0
                    lat = 0.0
                # Frontal: obstacle ahead, robot driving into it
                # (obs->robot points backward along heading => push_local < 0).
                if push_local < -self.front_cos:
                    frontal_violation = True
                    escape_push += depth * push_local
                else:
                    lateral_violation = True
                    lateral_signed += depth * lat
            elif h < self.min_dist and dist > 1e-6:
                dh_dx, dh_dy = dx / dist, dy / dist
                Lfh = (dh_dx * heading[0] + dh_dy * heading[1]) * u0_cmd
                if Lfh + self.alpha * h < 0:
                    scale = max(0.0, 1.0 + (Lfh + self.alpha * h) / (abs(u0_cmd) + 1e-6))
                    soft_scale = min(soft_scale, scale)

        if frontal_violation:
            # Something is genuinely ahead: do not add forward motion into it,
            # but preserve/reinforce backing away (escape_push is negative for
            # frontal obstacles, so this pushes u[0] below zero).
            u[0] = min(0.0, u0_cmd) + 0.3 * escape_push
        elif lateral_violation:
            # Pure corridor squeeze: KEEP MOVING FORWARD through the gap rather
            # than freezing. Hold at least a creep, never exceed the command.
            if u0_cmd >= 0.0:
                u[0] = max(self.creep_v, soft_scale * u0_cmd)
            else:
                u[0] = u0_cmd  # respect an intentional reverse (recovery)
            # Steer toward the freer side: lateral_signed > 0 means the left
            # wall pushes harder (robot hugging left) -> steer right (negative w).
            u[1] = float(u[1]) - 0.8 * lateral_signed
            u[1] = float(np.clip(u[1], -v_lim["base_angular"], v_lim["base_angular"]))
        else:
            u[0] = u0_cmd * soft_scale

        u[0] = np.clip(u[0], -v_lim["base_linear"], v_lim["base_linear"])

        # Joint limit enforcement (identical to base).
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
