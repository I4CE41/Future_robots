"""Dynamic Window Approach local planner.

Adapted from AtsushiSakai/PythonRobotics with key fixes:
- Cost includes BOTH heading angle AND distance-to-goal (prevents spiraling)
- Higher goal cost weight so robot drives straight when path is clear
- Proper IR safety integration
"""

import math
import numpy as np


class DWAPlanner:
    def __init__(self, config):
        v_lim = config["robot"]["velocity_limits"]

        # Robot params
        self.max_speed = v_lim["base_linear"]
        self.min_speed = -0.3
        self.max_yaw_rate = v_lim["base_angular"]
        self.max_accel = 3.0
        self.max_delta_yaw_rate = 5.0

        # Sampling
        self.v_resolution = 0.05
        self.yaw_rate_resolution = 5.0 * math.pi / 180.0
        self.dt = 0.1
        self.predict_time = 1.5

        # Cost weights — heading + distance + speed + obstacle
        self.heading_cost_gain = 1.0       # face the goal
        self.dist_cost_gain = 0.6          # get closer to goal
        self.speed_cost_gain = 0.5         # prefer moving fast (higher = less stopping)
        self.obstacle_cost_gain = 1.0      # avoid obstacles

        # Collision radius
        self.robot_radius = 0.30

        # Stuck escape
        self.stuck_count = 0
        self.stuck_threshold = 15

    def plan(self, state, goal, obstacles, ir_readings,
             current_v=0.0, current_w=0.0):
        x = np.array([state["x"], state["y"], state["theta"],
                       current_v, current_w])
        goal_xy = np.array([goal[0], goal[1]])

        ob_list = [[ox, oy] for ox, oy, r in obstacles] if obstacles else [[999, 999]]
        ob = np.array(ob_list)
        radii = np.array([r for _, _, r in obstacles]) if obstacles else np.array([0.0])

        dw = self._calc_dynamic_window(x)

        # Thesis Eq. (3.32): admissible braking velocity v <= sqrt(2*d_min*a_max)
        # so the robot can always stop before the nearest obstacle.
        if obstacles:
            d_min = min(math.hypot(x[0] - ox, x[1] - oy) - r
                        for ox, oy, r in obstacles)
            v_adm = math.sqrt(max(0.0, 2.0 * d_min * self.max_accel))
            dw[1] = min(dw[1], max(v_adm, 0.05))

        best_u, best_traj = self._calc_control(x, dw, goal_xy, ob, radii, ir_readings)

        # Stuck detection: if nearly stopped for many cycles, force rotation
        if abs(best_u[0]) < 0.01 and abs(current_v) < 0.02:
            self.stuck_count += 1
            if self.stuck_count > self.stuck_threshold:
                best_u[0] = 0.0
                best_u[1] = self.max_yaw_rate * 0.5
                self.stuck_count = 0
        else:
            self.stuck_count = 0

        # IR safety: stop forward if front blocked
        fl = ir_readings.get("front_left", ir_readings.get("fl", 1.0))
        fr = ir_readings.get("front_right", ir_readings.get("fr", 1.0))
        front_min = min(fl, fr)
        if best_u[0] > 0 and front_min < 0.25:
            best_u[0] = 0.0
            best_u[1] = self.max_yaw_rate * 0.5

        return best_u[0], best_u[1]

    def _calc_dynamic_window(self, x):
        Vs = [self.min_speed, self.max_speed,
              -self.max_yaw_rate, self.max_yaw_rate]
        Vd = [x[3] - self.max_accel * self.dt,
              x[3] + self.max_accel * self.dt,
              x[4] - self.max_delta_yaw_rate * self.dt,
              x[4] + self.max_delta_yaw_rate * self.dt]
        return [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
                max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]

    def _predict_trajectory(self, x_init, v, yaw_rate):
        x = np.array(x_init, dtype=float)
        traj = [x.copy()]
        t = 0.0
        while t <= self.predict_time:
            x[2] += yaw_rate * self.dt
            x[0] += v * math.cos(x[2]) * self.dt
            x[1] += v * math.sin(x[2]) * self.dt
            x[3] = v
            x[4] = yaw_rate
            traj.append(x.copy())
            t += self.dt
        return np.array(traj)

    def _calc_control(self, x, dw, goal, ob, radii, ir_readings):
        min_cost = float("inf")
        best_u = [0.0, 0.0]
        best_traj = np.array([x])

        for v in np.arange(dw[0], dw[1] + self.v_resolution * 0.5, self.v_resolution):
            for yaw_rate in np.arange(dw[2], dw[3] + self.yaw_rate_resolution * 0.5,
                                       self.yaw_rate_resolution):
                traj = self._predict_trajectory(x, v, yaw_rate)

                ob_cost = self.obstacle_cost_gain * self._obstacle_cost(traj, ob, radii)
                if ob_cost == float("inf"):
                    continue

                heading_cost = self.heading_cost_gain * self._heading_cost(traj, goal)
                dist_cost = self.dist_cost_gain * self._distance_cost(traj, goal)
                speed_cost = self.speed_cost_gain * (self.max_speed - abs(traj[-1, 3]))

                final_cost = heading_cost + dist_cost + speed_cost + ob_cost

                if final_cost < min_cost:
                    min_cost = final_cost
                    best_u = [v, yaw_rate]
                    best_traj = traj

        return best_u, best_traj

    def _heading_cost(self, traj, goal):
        """Angle between robot heading and goal direction at trajectory end."""
        dx = goal[0] - traj[-1, 0]
        dy = goal[1] - traj[-1, 1]
        goal_angle = math.atan2(dy, dx)
        heading_err = goal_angle - traj[-1, 2]
        return abs(math.atan2(math.sin(heading_err), math.cos(heading_err)))

    def _distance_cost(self, traj, goal):
        """Euclidean distance from trajectory endpoint to goal (normalized)."""
        dx = goal[0] - traj[-1, 0]
        dy = goal[1] - traj[-1, 1]
        return math.hypot(dx, dy)

    def _obstacle_cost(self, traj, ob, radii):
        """Inf if collision, else 1/min_clearance."""
        dx = traj[:, 0, None] - ob[None, :, 0] if ob.ndim == 2 else traj[:, 0, None] - ob[:, 0]
        dy = traj[:, 1, None] - ob[None, :, 1] if ob.ndim == 2 else traj[:, 1, None] - ob[:, 1]

        # Handle array shapes
        if ob.ndim == 1:
            ob = ob.reshape(1, -1)
        dx = traj[:, 0:1] - ob[:, 0:1].T
        dy = traj[:, 1:2] - ob[:, 1:2].T
        r = np.hypot(dx, dy)

        clearance = r - radii[None, :]
        if np.any(clearance <= self.robot_radius):
            return float("inf")

        min_c = np.min(clearance)
        if min_c <= 0:
            return float("inf")
        return 1.0 / min_c
