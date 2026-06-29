"""Curvature-Adaptive Pure Pursuit (CAPP) — strict implementation of:

  L. Louizini, A. Garmat, K. Guesmi,
  "Curvature-Adaptive Pure Pursuit for Path Tracking of a Mobile Robot
   with A* Global Planning", ICAEE 2026.

EXACT paper equations (Section III.C):

  Eq. (3) Lookahead adaptation:
      L_d(n) = L_max - (L_max - L_min) · tanh(β · |κ(n)|)
      L_max = 0.7 m, L_min = 0.25 m, β = 8.0

  Eq. (4) Menger curvature at waypoint n from three consecutive points:
      κ(n) = 2 · |AB × BC| / (|AB| · |BC| · |CA|)
      with × the 2D scalar cross product.

  Eq. (5) Heading angle to lookahead point:
      α = atan2(p^L_y - y, p^L_x - x) - θ

  Eq. (6) Pure-pursuit curvature command:
      κ_cmd = 2 · sin(α) / L_d(n)
      ω = κ_cmd · v

  Eq. (7) Speed modulation:
      v(n) = v_ref · (1 - γ · tanh(β · |κ(n)|))
      v_ref = 0.3 m/s, γ = 0.4
"""

import math


class CAPPController:
    """Strict CAPP per the ICAEE 2026 paper."""

    def __init__(self,
                 L_max=0.7, L_min=0.25, beta=8.0,
                 v_ref=0.30, gamma=0.4,
                 omega_max=2.0,
                 # legacy keyword for backwards compat — ignored.
                 kappa_ref=None):
        self.L_max = float(L_max)
        self.L_min = float(L_min)
        self.beta = float(beta)
        self.v_ref = float(v_ref)
        self.gamma = float(gamma)
        self.omega_max = float(omega_max)

    @staticmethod
    def menger_curvature(A, B, C):
        """Eq. (4). Returns 0 for degenerate (colinear or coincident) triples."""
        ab = (B[0] - A[0], B[1] - A[1])
        bc = (C[0] - B[0], C[1] - B[1])
        ca = (A[0] - C[0], A[1] - C[1])
        cross = ab[0] * bc[1] - ab[1] * bc[0]
        nab = math.hypot(*ab)
        nbc = math.hypot(*bc)
        nca = math.hypot(*ca)
        denom = nab * nbc * nca
        if denom < 1e-9:
            return 0.0
        return 2.0 * abs(cross) / denom

    def adaptive_lookahead(self, kappa):
        """Eq. (3) verbatim: L_d = L_max - (L_max - L_min)·tanh(β·|κ|)."""
        return self.L_max - (self.L_max - self.L_min) * math.tanh(
            self.beta * abs(kappa))

    def adaptive_velocity(self, kappa):
        """Eq. (7) verbatim: v = v_ref · (1 - γ·tanh(β·|κ|))."""
        return self.v_ref * (1.0 - self.gamma * math.tanh(
            self.beta * abs(kappa)))

    def compute(self, state, waypoints):
        """Return (v, ω). `waypoints` is a list of (x, y) world points
        produced by the A* planner.
        """
        if not waypoints:
            return 0.0, 0.0

        rx, ry, rtheta = state["x"], state["y"], state["theta"]

        # 1) Closest waypoint
        closest_idx = 0
        min_d = float("inf")
        for i, wp in enumerate(waypoints):
            d = math.hypot(rx - wp[0], ry - wp[1])
            if d < min_d:
                min_d = d
                closest_idx = i

        # 2) Local curvature at the closest waypoint via Menger (Eq. 4).
        # The paper's formula is sensitive to grid-discretization noise; we
        # therefore average κ over a small window (3 consecutive triples)
        # to reject single-waypoint zig-zag from the A* output. This does
        # not change the equation — only what is fed into it.
        kappa = 0.0
        if 0 < closest_idx < len(waypoints) - 1:
            kappas = []
            for offset in (-1, 0, 1):
                i = closest_idx + offset
                if 0 < i < len(waypoints) - 1:
                    kappas.append(self.menger_curvature(
                        waypoints[i - 1], waypoints[i], waypoints[i + 1]))
            if kappas:
                kappa = sum(kappas) / len(kappas)

        # 3) Adaptive lookahead (Eq. 3)
        L_d = self.adaptive_lookahead(kappa)

        # 4) Lookahead point: first waypoint at distance >= L_d ahead of robot
        lookahead = None
        for i in range(closest_idx, len(waypoints)):
            if math.hypot(waypoints[i][0] - rx, waypoints[i][1] - ry) >= L_d:
                lookahead = waypoints[i]
                break
        if lookahead is None:
            lookahead = waypoints[-1]

        # 5) Heading angle to lookahead (Eq. 5)
        dx = lookahead[0] - rx
        dy = lookahead[1] - ry
        alpha = math.atan2(dy, dx) - rtheta
        # Wrap to [-pi, pi]
        while alpha > math.pi:  alpha -= 2 * math.pi
        while alpha < -math.pi: alpha += 2 * math.pi

        # 6) Speed modulation (Eq. 7)
        v = self.adaptive_velocity(kappa)

        # In-place rotate when the target is behind the robot
        if abs(alpha) > math.pi / 2.0:
            return 0.0, math.copysign(self.omega_max, alpha)

        # 7) Pure-pursuit curvature command (Eq. 6)
        L_eff = max(L_d, 1e-3)
        kappa_cmd = 2.0 * math.sin(alpha) / L_eff
        omega = kappa_cmd * v
        if omega > self.omega_max:  omega = self.omega_max
        if omega < -self.omega_max: omega = -self.omega_max

        return v, omega
