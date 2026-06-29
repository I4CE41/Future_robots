"""NMPC-driven base navigation -- the "deep fix".

In the baseline architecture the NMPCController only runs while the base is
*locked* during arm phases; navigation is handled by a greedy analytic stack.
This planner instead uses the same NMPCController to optimise the **base**
motion during navigation, reusing the corridor constraints already built every
tick by ``CorridorGeometryExtractor.to_nmpc_corridor``.

Each tick it sets the NMPC reference to a lookahead carrot on the global path
(so the optimiser still respects the corridor-aware A* plan), holds the arm
reference at the current joint angles, passes the live obstacle list and
corridor band, solves, and returns the first base control ``[v, w]``. Solve
time and the predicted trajectory are surfaced for the dashboard/metrics.
"""

import numpy as np

from robot_rescue.control.main20_helpers.nav_common import LocalPlanner, carrot_on_path


class NMPCNavPlanner(LocalPlanner):
    name = "nmpc"

    def __init__(self, nmpc, corridor_extractor, nmpc_dt=0.1):
        self.nmpc = nmpc
        self.corridor_extractor = corridor_extractor
        self.nmpc_dt = float(nmpc_dt)
        self.u_prev = np.zeros(5)
        self.solves = 0

    def compute(self, ctx):
        state = ctx.state
        carrot = carrot_on_path(ctx.path, state, lookahead=1.4) or ctx.goal_xy

        # Reference: drive base to the carrot, hold the arm where it is, aim SoC high.
        arm_q = np.asarray(state.get("arm_q", np.zeros(3)), dtype=float)
        reference = np.array([
            float(carrot[0]), float(carrot[1]), 0.0,
            float(arm_q[0]) if len(arm_q) > 0 else 0.0,
            float(arm_q[1]) if len(arm_q) > 1 else 0.0,
            float(arm_q[2]) if len(arm_q) > 2 else 0.0,
        ], dtype=float)

        try:
            corridor = self.corridor_extractor.to_nmpc_corridor(
                ctx.corridor_state, state, self.nmpc.N + 1, self.nmpc_dt)
        except Exception:
            corridor = None

        try:
            u, solve_t, X_sol = self.nmpc.solve(
                state["full_q"], reference, ctx.obstacles, ctx.soc,
                u_prev=self.u_prev, corridor=corridor)
            self.u_prev = u.copy()
            self.solves += 1
            v_cmd = float(u[0])
            w_cmd = float(u[1])
            traj = np.asarray(X_sol)[:, :2]
            return v_cmd, w_cmd, {"nmpc_solve_t": solve_t, "nmpc_traj": traj}
        except Exception:
            # If the solve fails, hold still (the safety filter / recovery layer
            # in the main loop will take over).
            return 0.0, 0.0, {}
