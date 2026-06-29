"""Smart-adaptive energy policy.

4-tier control of speed, arm-velocity, and planner weights as a function of SoC.
Also provides a safe-return-home reserve check.
"""

import math


class EnergyPolicy:
    def __init__(self, config, energy_mgr):
        self.cfg = config
        self.energy = energy_mgr
        ec = config.get("energy", {})
        self.battery_capacity_wh = float(ec.get("battery_capacity", 100.0))
        self.low_threshold = float(ec.get("low_soc_threshold", 0.3))
        self.emergency = float(config.get("safety", {}).get("emergency_soc_threshold", 0.10))

        # Power coefficients used for energy-to-distance estimate.
        bp = ec.get("base_power_coeffs", {})
        self.p_idle = float(bp.get("idle", 5.0))
        self.p_linear = float(bp.get("linear", 20.0))
        self.p_angular = float(bp.get("angular", 10.0))

        # Reference cruise speed (m/s)
        self.v_ref = float(config.get("capp", {}).get("v_ref", 0.5))

        # Safety-reserve multiplier for return-home decision
        self.reserve_factor = 1.5

    # ---------------------------------------------------------------- tiering
    def _tier(self, soc):
        if soc > 0.50:
            return "performance"
        if soc > self.low_threshold:
            return "balanced"
        if soc > 0.15:
            return "frugal"
        return "emergency"

    def speed_scale(self, soc, dist_remaining=0.0):
        tier = self._tier(soc)
        if tier == "performance":
            return 1.0
        if tier == "balanced":
            return 0.85
        if tier == "frugal":
            return 0.55
        return 0.0  # emergency

    def arm_velocity_scale(self, soc):
        tier = self._tier(soc)
        return {"performance": 1.0, "balanced": 0.9, "frugal": 0.6, "emergency": 0.0}[tier]

    def planner_weights(self, soc):
        """Cost weights for time, energy, risk.

        As SoC drops, energy weight grows so cost_planner prefers shorter paths.
        """
        tier = self._tier(soc)
        table = {
            "performance": {"w_t": 1.0, "w_e": 0.2, "w_r": 0.3},
            "balanced":    {"w_t": 0.6, "w_e": 0.5, "w_r": 0.4},
            "frugal":      {"w_t": 0.3, "w_e": 1.0, "w_r": 0.5},
            "emergency":   {"w_t": 0.0, "w_e": 1.5, "w_r": 0.7},
        }
        return table[tier]

    # ---------------------------------------------------------------- reserve
    def estimate_energy_for(self, distance_m):
        """Rough Wh estimate to travel `distance_m` at v_ref (linear-only)."""
        if distance_m <= 0:
            return 0.0
        travel_time_s = distance_m / max(0.05, self.v_ref)
        avg_power_w = self.p_idle + self.p_linear * self.v_ref
        return avg_power_w * travel_time_s / 3600.0

    def remaining_wh(self, soc):
        return soc * self.battery_capacity_wh

    def should_return_home(self, soc, dist_home):
        """True if remaining energy is dangerously close to the cost of going home."""
        need = self.estimate_energy_for(dist_home) * self.reserve_factor
        return self.remaining_wh(soc) < need

    def is_emergency(self, soc):
        return soc < self.emergency

    def tier_name(self, soc):
        return self._tier(soc)
