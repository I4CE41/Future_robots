"""CasADi-based NMPC with energy cost — thesis equations (Ch. 3)
  7-state: [x, y, theta, q1, q2, q3, SOC]
  5-control: [v, omega, dq1, dq2, dq3]
  Uses IPOPT with warm start, returns real solve time (~126ms as per Ch.4)
"""

import time
import numpy as np
import casadi as ca
from scipy.linalg import solve_discrete_are


class NMPCController:
    def __init__(self, config):
        self.cfg = config["nmpc"]
        self.robot_cfg = config["robot"]
        self.energy_cfg = config["energy"]
        self.N = self.cfg["horizon"]
        self.dt = self.cfg["dt"]
        self._prev_solution = None
        self._solve_times = []
        self._build_solver()

    def _compute_terminal_cost_matrix(self):
        nx = 7
        nu = 5
        dt = self.dt
        A = np.eye(nx)
        B = np.zeros((nx, nu))
        B[0, 0] = dt
        B[1, 0] = 0.0
        B[2, 1] = dt
        B[3, 2] = dt
        B[4, 3] = dt
        B[5, 4] = dt

        Q_dare = np.diag([
            self.cfg["Q_pos"][0],
            self.cfg["Q_pos"][1],
            self.cfg["Q_pos"][2],
            self.cfg["Q_arm"][0],
            self.cfg["Q_arm"][1],
            self.cfg["Q_arm"][2],
            0.1,
        ])
        R_dare = np.diag(self.cfg["R_control"])

        try:
            Pf = solve_discrete_are(A, B, Q_dare, R_dare)
            if np.all(np.linalg.eigvalsh(Pf) > 0) and np.max(Pf) < 1e4:
                return Pf
        except Exception:
            pass
        return Q_dare * 5.0

    def _build_solver(self):
        N, dt = self.N, self.dt
        nx = 7
        nu = 5

        X = ca.MX.sym("X", nx, N + 1)
        U = ca.MX.sym("U", nu, N)

        self.max_obstacles = 10
        n_param = nx + nx + self.max_obstacles * 3 + nu
        P = ca.MX.sym("P", n_param)
        x0_param = P[:nx]
        x_ref = P[nx:2 * nx]
        obs_param = P[2 * nx:2 * nx + self.max_obstacles * 3]
        u_prev_param = P[2 * nx + self.max_obstacles * 3 : 2 * nx + self.max_obstacles * 3 + nu]

        w_pos_x = self.cfg["Q_pos"][0]
        w_pos_y = self.cfg["Q_pos"][1]
        w_theta = self.cfg["Q_pos"][2]
        w_arm = self.cfg["Q_arm"]
        w_soc = 0.1
        R_ctrl = ca.diag(ca.DM(self.cfg["R_control"]))
        w_energy = self.cfg["R_energy"]
        w_smooth = self.cfg["R_smooth"]
        w_obs = self.cfg["obstacle_weight"]
        w_chatter = 0.0
        w_chatter = 0.0
        obs_margin = self.cfg["obstacle_margin"]

        bc = self.energy_cfg["base_power_coeffs"]
        ac = self.energy_cfg["arm_power_coeffs"]
        E_cap = self.energy_cfg["battery_capacity"]

        SOC_min = self.energy_cfg.get("emergency_soc_threshold", 0.10)
        wall_min, wall_max = -4.5, 4.5
        w_wall = 100.0

        Pf = self._compute_terminal_cost_matrix()
        Pf_casadi = ca.DM(Pf)

        cost = 0
        g = []
        lbg = []
        ubg = []

        g.append(X[:, 0] - x0_param)
        lbg += [0] * nx
        ubg += [0] * nx

        for k in range(N):
            xk = X[:, k]
            uk = U[:, k]
            xk1 = X[:, k + 1]

            P_base = bc["idle"] + bc["linear"] * uk[0]**2 + bc["angular"] * uk[1]**2
            P_arm = 0
            eps = 1e-4
            for j in range(3):
                P_arm += ac["velocity"] * uk[2 + j]**2 + ac["idle"]
            grav_comp = ca.sqrt(ca.sin(xk[3])**2 + eps) + \
                        0.5 * ca.sqrt(ca.sin(xk[3] + xk[4])**2 + eps)
            P_arm += ac["gravity_comp"] * grav_comp
            P_total = P_base + P_arm

            x_next = ca.vertcat(
                xk[0] + dt * uk[0] * ca.cos(xk[2]),
                xk[1] + dt * uk[0] * ca.sin(xk[2]),
                xk[2] + dt * uk[1],
                xk[3] + dt * uk[2],
                xk[4] + dt * uk[3],
                xk[5] + dt * uk[4],
                xk[6] - P_total * dt / (3600.0 * E_cap),
            )
            g.append(xk1 - x_next)
            lbg += [0] * nx
            ubg += [0] * nx

            g.append(xk[6])
            lbg += [SOC_min]
            ubg += [1.1]

            cost += w_pos_x * (xk[0] - x_ref[0])**2
            cost += w_pos_y * (xk[1] - x_ref[1])**2
            dx_goal = x_ref[0] - xk[0]
            dy_goal = x_ref[1] - xk[1]
            eps_a = 1e-8
            desired_theta = ca.atan2(dy_goal + eps_a, dx_goal + eps_a)
            dist_to_goal = ca.sqrt(dx_goal**2 + dy_goal**2 + 1e-6)
            heading_err = ca.atan2(ca.sin(xk[2] - desired_theta),
                                   ca.cos(xk[2] - desired_theta))
            cost += w_theta * ca.fmin(dist_to_goal, 1.0) * heading_err**2
            for j in range(3):
                cost += w_arm[j] * (xk[3 + j] - x_ref[3 + j])**2
            cost += w_soc * (xk[6] - x_ref[6])**2
            cost += uk.T @ R_ctrl @ uk
            if k == 0:
                du_exec = uk - u_prev_param
                cost += w_chatter * (du_exec.T @ du_exec)
            if k == 0:
                du_exec = uk - u_prev_param
                cost += w_chatter * (du_exec.T @ du_exec)
            # Thesis Eq. (3.16) + (3.25): J += w_E * phi(z) * P_loss * dt
            # with phi(z) = 1 / (0.5 + 0.5*z + eps), eps = 0.01.
            soc_factor = 1.0 / (0.5 + 0.5 * xk[6] + 0.01)
            cost += w_energy * soc_factor * P_total * dt
            if k > 0:
                du = uk - U[:, k - 1]
                cost += w_smooth * (du.T @ du)
            for i in range(self.max_obstacles):
                ox = obs_param[i * 3]
                oy = obs_param[i * 3 + 1]
                r = obs_param[i * 3 + 2]
                dist_sq = (xk[0] - ox)**2 + (xk[1] - oy)**2
                safe_dist_sq = (r + obs_margin)**2
                cost += w_obs * ca.fmax(0, safe_dist_sq - dist_sq)
            cost += w_wall * ca.fmax(0, wall_min - xk[0])**2
            cost += w_wall * ca.fmax(0, xk[0] - wall_max)**2
            cost += w_wall * ca.fmax(0, wall_min - xk[1])**2
            cost += w_wall * ca.fmax(0, xk[1] - wall_max)**2

        g.append(X[6, N])
        lbg += [SOC_min]
        ubg += [1.1]

        e_terminal = X[:, N] - x_ref
        cost += e_terminal.T @ Pf_casadi @ e_terminal

        opt_vars = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))

        v_lim = self.robot_cfg["velocity_limits"]
        j_lim = self.robot_cfg["joint_limits"]

        lbx = []
        ubx = []
        for k in range(N + 1):
            lbx += [wall_min, wall_min, -2 * np.pi,
                    j_lim["shoulder"][0], j_lim["elbow"][0], j_lim["wrist"][0],
                    SOC_min]
            ubx += [wall_max, wall_max, 2 * np.pi,
                    j_lim["shoulder"][1], j_lim["elbow"][1], j_lim["wrist"][1],
                    1.0]
        for k in range(N):
            lbx += [-v_lim["base_linear"], -v_lim["base_angular"],
                    -v_lim["arm_joints"], -v_lim["arm_joints"], -v_lim["arm_joints"]]
            ubx += [v_lim["base_linear"], v_lim["base_angular"],
                    v_lim["arm_joints"], v_lim["arm_joints"], v_lim["arm_joints"]]

        nlp = {"f": cost, "x": opt_vars, "g": ca.vertcat(*g), "p": P}
        opts = {
            "ipopt.print_level": 0,
            "ipopt.max_iter": self.cfg["max_iter"],
            "ipopt.warm_start_init_point": "yes",
            "ipopt.tol": 1e-4,
            # Early-exit once an acceptable KKT point is reached — keeps the
            # warm-started solve well inside the 100 ms control period
            # (thesis Sec 1.7.1: mean 18.3 ms, worst-case 32.1 ms target).
            "ipopt.acceptable_tol": 1e-3,
            "ipopt.acceptable_iter": 15,
            "ipopt.mu_strategy": "adaptive",
            "print_time": 0,
        }
        self.solver = ca.nlpsol("nmpc", "ipopt", nlp, opts)
        self.nx = nx
        self.nu = nu
        self.n_param = n_param
        self.lbx = lbx
        self.ubx = ubx
        self.lbg = lbg
        self.ubg = ubg

    def _make_initial_guess(self, x0, x_ref):
        X_init = np.zeros((self.N + 1, self.nx))
        for i in range(self.N + 1):
            alpha = i / self.N
            X_init[i] = x0 * (1 - alpha) + x_ref * alpha
        dx = x_ref[0] - x0[0]
        dy = x_ref[1] - x0[1]
        desired_theta = np.arctan2(dy, dx)
        for i in range(self.N + 1):
            X_init[i, 2] = desired_theta
        U_init = np.zeros((self.N, self.nu))
        dist = np.sqrt(dx**2 + dy**2)
        if dist > 0.1:
            v_cruise = min(0.5, dist / (self.N * self.dt))
            U_init[:, 0] = v_cruise
        return np.concatenate([X_init.flatten(), U_init.flatten()])

    def solve(self, current_state, reference, obstacles, soc=1.0, u_prev=None):
        t0 = time.time()

        if len(current_state) == 6:
            x0_full = np.append(current_state, soc)
        else:
            x0_full = np.array(current_state[:7])
            x0_full[6] = soc

        if len(reference) == 6:
            ref_full = np.append(reference, 1.0)
        else:
            ref_full = np.array(reference[:7])

        FAR = 100.0
        obs_flat = np.zeros(self.max_obstacles * 3)
        for i in range(self.max_obstacles):
            if i < len(obstacles):
                obs_flat[i * 3:i * 3 + 3] = obstacles[i][:3]
            else:
                obs_flat[i * 3:i * 3 + 3] = [FAR, FAR, 0.0]
        if u_prev is None: u_prev = np.zeros(5)
        p_val = np.concatenate([x0_full, ref_full, obs_flat, u_prev])

        if self._prev_solution is not None:
            x0_guess = self._prev_solution
        else:
            x0_guess = self._make_initial_guess(x0_full, ref_full)

        sol = self.solver(
            x0=x0_guess, p=p_val,
            lbx=self.lbx, ubx=self.ubx,
            lbg=self.lbg, ubg=self.ubg,
        )

        sol_x = np.array(sol["x"]).flatten()
        solve_time = time.time() - t0
        self._solve_times.append(solve_time * 1000)

        n_x_vars = self.nx * (self.N + 1)
        X_sol = sol_x[:n_x_vars].reshape(self.N + 1, self.nx)
        U_sol = sol_x[n_x_vars:].reshape(self.N, self.nu)

        shifted_x = np.vstack([X_sol[1:], X_sol[-1:]])
        shifted_u = np.vstack([U_sol[1:], U_sol[-1:]])
        self._prev_solution = np.concatenate([shifted_x.flatten(), shifted_u.flatten()])

        return U_sol[0], solve_time, X_sol

    def get_solve_time_stats(self):
        if not self._solve_times:
            return {"mean": 0, "std": 0, "p99": 0, "count": 0}
        times = np.array(self._solve_times)
        return {
            "mean": float(np.mean(times)),
            "std": float(np.std(times)),
            "p99": float(np.percentile(times, 99)) if len(times) >= 10 else float(np.max(times)),
            "count": len(times),
        }

    def reset(self):
        self._prev_solution = None
        self._solve_times = []

