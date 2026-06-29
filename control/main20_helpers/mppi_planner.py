"""Model Predictive Path Integral (MPPI) local planner for the diff-drive base.

MPPI is a sampling-based, gradient-free stochastic optimal controller. Each
control tick it:

  1. Perturbs a warm-started nominal control sequence ``U_nom`` (length H) with
     K Gaussian noise rollouts.
  2. Rolls every sample forward through the unicycle model (the same kinematics
     the NMPC uses) -- fully vectorised in numpy.
  3. Scores each rollout with a cost that blends obstacle avoidance, corridor
     centering, path/goal tracking, control effort and an optional SoC-weighted
     energy term.
  4. Updates ``U_nom`` with the softmax (information-theoretic) weighting
     ``w_k ~ exp(-(1/lambda)(S_k - min S))``.

Unlike the greedy pure-pursuit baseline, MPPI evaluates whole trajectories
against the non-convex clustered-obstacle field, so it threads the tight slalom
without wedging. It returns the first control of the optimised sequence and
shifts ``U_nom`` forward for next-tick warm start.
"""

import math

import numpy as np

from robot_rescue.control.main20_helpers.nav_common import LocalPlanner, carrot_on_path


class MPPIPlanner(LocalPlanner):
    name = "mppi"

    def __init__(self, config, samples=600, horizon=20, lam=0.30,
                 sigma_v=0.30, sigma_w=0.60, seed=0):
        v_lim = config["robot"]["velocity_limits"]
        self.v_max = float(v_lim["base_linear"])
        self.w_max = float(v_lim["base_angular"])
        self.K = int(samples)
        self.H = int(horizon)
        self.lam = float(lam)
        self.sigma = np.array([float(sigma_v), float(sigma_w)], dtype=float)
        self.dt = float(config.get("nmpc", {}).get("dt", 0.1))
        self.robot_radius = 0.30
        self.safety_margin = 0.18

        # Cost weights.
        self.w_obs = 60.0          # collision / proximity
        self.w_goal = 6.0          # terminal distance to carrot
        self.w_track = 2.5         # running distance to carrot
        self.w_corridor = 8.0      # deviation from corridor centerline
        self.w_ctrl = 0.10         # control effort
        self.w_energy = 0.0        # set >0 to penalise speed when SoC is low

        # Warm-started nominal control sequence: [v, w] per step.
        self.U_nom = np.zeros((self.H, 2), dtype=float)
        self.rng = np.random.default_rng(seed)

    def reset(self):
        self.U_nom[:] = 0.0

    def compute(self, ctx):
        carrot = carrot_on_path(ctx.path, ctx.state, lookahead=1.2) or ctx.goal_xy
        carrot = np.array([carrot[0], carrot[1]], dtype=float)
        goal = np.array([ctx.goal_xy[0], ctx.goal_xy[1]], dtype=float)

        x0 = float(ctx.state["x"])
        y0 = float(ctx.state["y"])
        th0 = float(ctx.state["theta"])

        # Obstacles as arrays (inflated by robot radius + margin).
        if ctx.obstacles:
            obs = np.array(ctx.obstacles, dtype=float)  # (M, 3)
            ox = obs[:, 0]
            oy = obs[:, 1]
            orad = obs[:, 2] + self.robot_radius + self.safety_margin
        else:
            ox = oy = orad = None

        # Sample noise: (K, H, 2).
        eps = self.rng.normal(size=(self.K, self.H, 2)) * self.sigma[None, None, :]
        U = self.U_nom[None, :, :] + eps  # (K, H, 2)
        U[:, :, 0] = np.clip(U[:, :, 0], -0.2 * self.v_max, self.v_max)
        U[:, :, 1] = np.clip(U[:, :, 1], -self.w_max, self.w_max)

        # Vectorised rollout of the unicycle model.
        xs = np.full(self.K, x0)
        ys = np.full(self.K, y0)
        ths = np.full(self.K, th0)
        cost = np.zeros(self.K)

        cx_line, cy_line = self._corridor_line(ctx)

        for t in range(self.H):
            v = U[:, t, 0]
            w = U[:, t, 1]
            ths = ths + w * self.dt
            xs = xs + v * np.cos(ths) * self.dt
            ys = ys + v * np.sin(ths) * self.dt

            # Obstacle cost: heavy penalty inside inflated radius, soft outside.
            if ox is not None:
                dx = xs[:, None] - ox[None, :]
                dy = ys[:, None] - oy[None, :]
                dist = np.sqrt(dx * dx + dy * dy)
                pen = orad[None, :] - dist           # >0 means penetration
                # collision: large fixed cost; near: quadratic falloff.
                collide = np.maximum(0.0, pen)
                cost += self.w_obs * np.sum(collide * collide, axis=1)
                cost += 1000.0 * np.any(pen > 0.0, axis=1)

            # Running tracking cost toward the carrot.
            cost += self.w_track * ((xs - carrot[0]) ** 2 + (ys - carrot[1]) ** 2)

            # Corridor-centerline deviation.
            if cx_line is not None:
                cost += self.w_corridor * ((xs - cx_line) ** 2 + (ys - cy_line) ** 2)

            # Control effort.
            cost += self.w_ctrl * (v * v + 0.1 * w * w)

            # Optional SoC-weighted energy term (faster -> more cost at low SoC).
            if self.w_energy > 0.0:
                soc_factor = 1.0 / (0.5 + 0.5 * float(ctx.soc) + 0.01)
                cost += self.w_energy * soc_factor * (v * v)

        # Terminal cost toward goal/carrot.
        cost += self.w_goal * ((xs - carrot[0]) ** 2 + (ys - carrot[1]) ** 2)
        cost += 0.5 * self.w_goal * ((xs - goal[0]) ** 2 + (ys - goal[1]) ** 2)

        # Softmax (information-theoretic) weighting.
        beta = np.min(cost)
        weights = np.exp(-(1.0 / self.lam) * (cost - beta))
        wsum = np.sum(weights)
        if not np.isfinite(wsum) or wsum < 1e-9:
            # All rollouts catastrophic -> rotate in place to re-orient.
            return 0.0, math.copysign(0.6 * self.w_max, self._goal_bearing_err(ctx)), {}
        weights /= wsum

        # Weighted update of the nominal sequence.
        self.U_nom = np.sum(weights[:, None, None] * U, axis=0)  # (H, 2)

        v_cmd = float(self.U_nom[0, 0])
        w_cmd = float(self.U_nom[0, 1])

        # Shift nominal forward for warm start.
        self.U_nom = np.vstack([self.U_nom[1:], self.U_nom[-1:]])

        # Corridor / sensor speed shaping (consistent with baseline).
        v_cmd *= float(getattr(ctx.corridor_state, "velocity_scale", 1.0))
        if ctx.sensor_front < 0.45:
            v_cmd *= 0.6
        v_cmd = float(np.clip(v_cmd, -0.2 * self.v_max, self.v_max))
        w_cmd = float(np.clip(w_cmd, -self.w_max, self.w_max))
        return v_cmd, w_cmd, {}

    def _corridor_line(self, ctx):
        cs = ctx.corridor_state
        if cs is not None and getattr(cs, "centerline", None):
            pts = cs.centerline
            mid = pts[min(len(pts) // 2, len(pts) - 1)]
            return float(mid[0]), float(mid[1])
        return None, None

    def _goal_bearing_err(self, ctx):
        dx = ctx.goal_xy[0] - float(ctx.state["x"])
        dy = ctx.goal_xy[1] - float(ctx.state["y"])
        desired = math.atan2(dy, dx)
        err = desired - float(ctx.state["theta"])
        return math.atan2(math.sin(err), math.cos(err))
