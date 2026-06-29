"""Curvature-Adaptive Pure Pursuit (CAPP) Controller."""

import math
import numpy as np

class PurePursuitController:
    def __init__(self, config):
        capp_cfg = config.get("capp", {})
        self.ld_max = capp_cfg.get("ld_max", 0.7)
        self.ld_min = capp_cfg.get("ld_min", 0.25)
        self.beta = capp_cfg.get("beta", 8.0)
        self.kappa_ref = capp_cfg.get("kappa_ref", 1.5)
        self.v_ref = capp_cfg.get("v_ref", 0.5)
        self.gamma = capp_cfg.get("gamma", 0.4)

    def compute(self, state, waypoints, goal, ir_readings, obstacles=None):
        if not waypoints: return 0.0, 0.0
        rx, ry, rtheta = state["x"], state["y"], state["theta"]
        
        min_dist = float('inf'); closest_idx = 0
        for i, wp in enumerate(waypoints):
            d = math.hypot(rx-wp[0], ry-wp[1])
            if d < min_dist: min_dist = d; closest_idx = i

        kappa = 0.0
        if 0 < closest_idx < len(waypoints)-1:
            A = np.array(waypoints[closest_idx-1]); B = np.array(waypoints[closest_idx]); C = np.array(waypoints[closest_idx+1])
            AB = B-A; BC = C-B; AC = C-A
            cross = abs(AB[0]*BC[1] - AB[1]*BC[0])
            denom = np.linalg.norm(AB) * np.linalg.norm(BC) * np.linalg.norm(AC)
            kappa = cross / denom if denom > 1e-6 else 0.0

        L_d = self.ld_min + (self.ld_max - self.ld_min) * 0.5 * (1 - math.tanh(self.beta * (kappa - self.kappa_ref)))
        
        lookahead_point = None
        for i in range(closest_idx, len(waypoints)):
            if math.hypot(waypoints[i][0]-rx, waypoints[i][1]-ry) >= L_d:
                lookahead_point = waypoints[i]; break
        if lookahead_point is None: lookahead_point = waypoints[-1]

        dx = lookahead_point[0]-rx; dy = lookahead_point[1]-ry
        local_x = dx*math.cos(rtheta) + dy*math.sin(rtheta)
        local_y = -dx*math.sin(rtheta) + dy*math.cos(rtheta)
        if abs(local_x) < 1e-6 and abs(local_y) < 1e-6: return 0.0, 0.0
        
        curvature = 2 * local_y / (L_d ** 2)
        omega = curvature * self.v_ref
        v = self.v_ref * math.exp(-self.gamma * kappa)

        if obstacles:
            for obs in obstacles:
                od = math.hypot(rx-obs[0], ry-obs[1])
                if od < 0.6: v *= max(0.2, od/0.6)
        return v, omega
