"""BYD-Inspired Cell-to-Pack Energy Model + Energy-Aware NMPC — strict
implementation of:

  A. Louizini, A. Garmat, K. Guesmi,
  "A BYD-Inspired Cell-to-Pack Energy Model and Energy-Aware NMPC for
   Safe Autonomous Mobile Manipulators", CISTEM 2026 (v3).

EXACT paper equations (Section III):

  Eq. (1) Per-cell manufacturing dispersion:
      Q_i = Q_nom · (1 + δ_i^Q)   with δ_i^Q ∈ [-0.02, +0.02]
      ρ_i = 1 + δ_i^R              with δ_i^R ∈ [-0.05, +0.05]

  Eq. (2) Pack OCV:
      V_OCV(z) = Σ_{i=1..8} f_OCV(z_i)
      (cell OCV piecewise-linear from manufacturer table, ~3.28 V at 20-80%)

  Eq. (3) Per-cell internal resistance with temperature:
      R_i(z_i, T_i) = ρ_i · R_cell(z_i) · [1 + α_T · max(0, T_ref - T_i)]
      α_T = 0.004 /K,  T_ref = 25 °C

  Eq. (4) Terminal voltage under load I (series sum):
      V_t(z, I) = Σ_{i=1..8} [f_OCV(z_i) - I · R_i(z_i, T_i)]

  Eq. (5) Per-cell SoC by Coulomb counting:
      z_i(t) = z_i(t0) - (1 / (Q_i · 3600)) · ∫ I(τ) dτ

  Eq. (6) Gated EKF SoC correction:
      ẑ_{k+1} = ẑ_k + K_k (V_meas - V_pred)   if ẑ_k ∉ (0.15, 0.85)
              = ẑ_k                            otherwise (plateau gating)
      K_k = P_k H_k / (H_k P_k H_k + R_n)
      H_k = ∂V_OCV / ∂z

  Eq. (7) Passive shunt-resistor balancer:
      I_bleed,i = (V_i - V̄) / R_s   if V_i > V̄ + 1mV
                = 0                  otherwise
      R_s = 10 Ω, threshold V_th = 20 mV, period N_b = 100 control steps

  Eq. (8) Cell-coupled thermal with motor waste-heat recovery:
      C_th · dT_i/dt = I² · R_i + k_c · (T_{i-1} - 2T_i + T_{i+1}) + η_h · Q_motor
      k_c = 0.5 (neighbour coupling), η_h = 0.15 (waste-heat recovery)
      High-T derating when any cell exceeds 45 °C.

  Eq. (9) BLDC motor current from torque demand:
      I_m = min(|τ|/K_t + I_0, I_stall)

  Eq. (10) Per-motor electrical power:
      P_elec = I_m² · R_w + K_e · I_m · |ω| + P_ctrl
      with K_e = K_t (SI units).

  Eq. (11) Cascaded efficiency chain:
      η_chain = η_DC/DC · η_inv · η_ctrl ≈ 0.97 · 0.95 · 0.97 ≈ 0.894
      P_batt = P_motor / η_chain

  Eq. (12) Intelligent regen efficiency (deceleration-dependent):
      η_regen(|v̇|) = 0.85                    if |v̇| < 0.5 m/s²
                   = lerp(0.85, 0.60)         if 0.5 ≤ |v̇| < 2.0
                   = 0.40                    if |v̇| ≥ 2.0 m/s²

  Eq. (13) Regen energy per timestep:
      E_regen = 0.5 · m_robot · (v₁² - v₂²) · η_regen(v̇)

  Eq. (14) Total electrical power (cascaded sum):
      P_total = (1/η_chain) · (Σ P_drive + Σ P_arm) + P_electronics

  Eq. (18) Predicted drive power (NMPC):
      P_drive = (2/η_chain) · [ R_w · (v_w / (K_t·r_w))² + K_e · |v_w|/r_w
                                · |v_w|/(K_t·r_w) ]

  Eq. (19) Gravity-comp torque per arm joint:
      τ_grav,j = m_j · g · l_com,j · sin( Σ_{i=1..j} q_i )

  Eq. (20) Per-joint arm power:
      P_arm,j = (τ_grav,j + J_j · q̇_j)² · R_w,j / K_t,j²

  Eq. (21) SoC-dependent NMPC weight:
      φ(z̄) = 1 / (0.5 + 0.5·z̄ + ε)   with ε = 0.01

Table II — motor parameters (verbatim from the paper):
    Drive    : K_t=0.15  R_w=0.80  I_0=0.50  I_stall=15.0    (×2)
    Shoulder : K_t=1.50  R_w=1.20  I_0=0.30  I_stall=8.0     (×1)
    Elbow    : K_t=1.20  R_w=1.50  I_0=0.30  I_stall=8.0     (×1)
    Wrist    : K_t=0.80  R_w=2.00  I_0=0.30  I_stall=5.0     (×1)
"""

import math
import random


# ============================================================================
# Per-cell OCV lookup — LiFePO4 (LFP), thesis-spec (Ch. 2-3).
#   Sec 2.5.1: nominal 3.20 V/cell, max charge 3.65 V/cell, cutoff 2.50 V/cell
#             -> 8S pack: 25.6 V nominal, 29.2 V max, 20.0 V cutoff.
#   Sec 3.2.1: "characteristic flat plateau between 20% and 80% SoC at
#              approximately 3.28 V/cell".
# Piecewise-linear interpolation of manufacturer discharge data (Eq. 3.2).
# ============================================================================
_OCV_TABLE = [
    (0.00, 2.50),   # cutoff (8S -> 20.0 V)
    (0.05, 2.80),
    (0.10, 3.00),
    (0.15, 3.15),
    (0.20, 3.25),   # plateau entry
    (0.30, 3.27),
    (0.40, 3.28),   # ~3.28 V plateau (Sec 3.2.1)
    (0.50, 3.28),
    (0.60, 3.29),
    (0.70, 3.30),
    (0.80, 3.32),   # plateau exit
    (0.90, 3.45),
    (0.95, 3.55),
    (1.00, 3.65),   # max charge (8S -> 29.2 V)
]


def _interp(x, table):
    if x <= table[0][0]:
        return table[0][1]
    if x >= table[-1][0]:
        return table[-1][1]
    for i in range(1, len(table)):
        x1, y1 = table[i - 1]
        x2, y2 = table[i]
        if x1 <= x <= x2:
            t = (x - x1) / (x2 - x1)
            return y1 + t * (y2 - y1)
    return table[-1][1]


def _f_ocv(z):
    """Per-cell OCV in V at SoC z ∈ [0,1]."""
    return _interp(max(0.0, min(1.0, z)), _OCV_TABLE)


def _f_ocv_deriv(z, eps=1e-3):
    """Approximate dV_OCV/dz at SoC z (numerical, for EKF Jacobian)."""
    z1 = max(0.0, z - eps)
    z2 = min(1.0, z + eps)
    if z2 <= z1:
        return 0.0
    return (_f_ocv(z2) - _f_ocv(z1)) / (z2 - z1)


def _r_cell_base(z):
    """Per-cell base internal resistance (Ω) at SoC z. ~6 mΩ at mid-SoC,
    rising to 25 mΩ near depletion. Empirical LFP profile."""
    pct = z * 100.0
    if pct >= 50.0:
        return 0.006
    if pct <= 5.0:
        return 0.025
    t = (pct - 5.0) / (50.0 - 5.0)
    return 0.025 + t * (0.006 - 0.025)


# ============================================================================
# Single cell
# ============================================================================
class Cell:
    """One LiFePO4 cell with its own capacity, resistance scale, SoC and
    temperature. Eqs. (1)-(5).
    """
    def __init__(self, Q_nom=20.0, delta_Q=0.0, delta_R=0.0,
                 T_amb=25.0, initial_soc=1.0):
        self.Q_nom = float(Q_nom)
        self.delta_Q = float(delta_Q)         # in [-0.02, 0.02]
        self.delta_R = float(delta_R)         # in [-0.05, 0.05]
        self.Q = self.Q_nom * (1.0 + self.delta_Q)   # Eq. (1)
        self.rho = 1.0 + self.delta_R                # Eq. (1)
        self.z = float(max(0.0, min(1.0, initial_soc)))
        self.T = float(T_amb)
        self.last_I = 0.0

    # Eq. (3)
    def R_int(self, T_ref=25.0, alpha_T=0.004):
        return self.rho * _r_cell_base(self.z) * (
            1.0 + alpha_T * max(0.0, T_ref - self.T))

    def V_ocv(self):
        return _f_ocv(self.z)

    # Per-cell terminal voltage contribution under load current I (A)
    def V_t(self, I):
        return self.V_ocv() - I * self.R_int()

    # Eq. (5) Coulomb counting step — strict, no drain hacks:
    #   z_i(t) = z_i(t0) - (1 / (Q_i · 3600)) · ∫ I dτ
    # I > 0 discharges (SoC down); I < 0 (regen) charges (SoC up). `dt` is the
    # effective battery-time in seconds (sim time may be compressed upstream by
    # PhysicsEnergyManager.TIME_ACCEL so a short demo run shows visible SoC
    # movement while keeping the per-cell physics exact).
    def coulomb_step(self, I, dt):
        delta_z = I * dt / (max(self.Q, 1e-6) * 3600.0)
        self.z = max(0.0, min(1.0, self.z - delta_z))
        self.last_I = I


# ============================================================================
# Cell-resolved 8S LiFePO4 pack
# ============================================================================
class CellToPackBattery:
    """8 cells in series, BYD-style cell-to-pack. Includes per-cell SoC,
    temperature, resistance, plus a passive shunt-resistor balancer (Eq. 7)
    and an OCV-gated EKF SoC estimator (Eq. 6).

    Thesis Sec 2.5.1 nominal pack parameters:
      Q_nom = 20 Ah per cell, V_nom = 3.20 V/cell
      -> 8S pack: 25.6 V nominal, C_pack = 20 Ah, E_pack = 512 Wh.
    Cutoff 20.0 V (8 x 2.50 V), max charge 29.2 V (8 x 3.65 V).
    """

    def __init__(self,
                 N_s=8,
                 Q_nom=20.0,          # Ah per cell — thesis Sec 2.5.1 (C_pack = 20 Ah)
                 initial_soc=1.0,
                 T_amb=25.0,
                 # Balancer
                 R_s=10.0,            # Ω shunt resistor
                 V_threshold=0.020,   # V (20 mV)
                 N_b=100,             # control-step period
                 # Thermal
                 C_th=30.0,           # J/K per cell (Table A.3)
                 k_c=0.5,             # neighbour coupling
                 eta_h=0.15,          # motor waste-heat recovery
                 R_th=2.0,            # K/W to ambient (Table A.3)
                 # EKF
                 P_init=1e-3,
                 R_n=1e-3,
                 plateau_low=0.15,
                 plateau_high=0.85,
                 enable_ekf=True,
                 enable_balance=True,
                 enable_dispersion=True,
                 seed=None):
        self.N_s = int(N_s)
        self.Q_nom = float(Q_nom)
        # Variant toggles (Table 4.7 comparative study). Defaults reproduce the
        # full cell-to-pack model; pack_level disables all three.
        self.enable_ekf = bool(enable_ekf)
        self.enable_balance = bool(enable_balance)
        self.enable_dispersion = bool(enable_dispersion)
        self.V_cut = 20.0                                 # 8 x 2.50 V cutoff (Sec 2.5.1)
        self.V_nom_cell = 3.20                            # nominal cell voltage (Sec 2.5.1)
        self.E_nom_Wh = float(N_s) * self.V_nom_cell * float(Q_nom)  # 8·3.20·20 = 512 Wh
        self.T_amb = float(T_amb)

        rng = random.Random(seed if seed is not None else 0xC2D)
        # Eq. (1) build cells with manufacturing dispersion
        self.cells = []
        for i in range(self.N_s):
            if self.enable_dispersion:
                dQ = rng.uniform(-0.02, +0.02)
                dR = rng.uniform(-0.05, +0.05)
            else:
                # Pack-level: identical cells, no manufacturing spread.
                dQ = 0.0
                dR = 0.0
            self.cells.append(Cell(Q_nom=Q_nom, delta_Q=dQ, delta_R=dR,
                                   T_amb=T_amb, initial_soc=initial_soc))

        # Balancer
        self.R_s = float(R_s)
        self.V_threshold = float(V_threshold)
        self.N_b = int(N_b)
        self._balance_counter = 0
        self.bleed_log = []

        # Thermal
        self.C_th = float(C_th)
        self.k_c = float(k_c)
        self.eta_h = float(eta_h)
        self.R_th = float(R_th)

        # EKF state (one per cell)
        self.P = [float(P_init) for _ in range(self.N_s)]
        self.R_n = float(R_n)
        self.plateau_low = float(plateau_low)
        self.plateau_high = float(plateau_high)

        # Bookkeeping
        self.energy_consumed_wh = 0.0
        self.energy_regenerated_wh = 0.0
        self.regen_log = []
        self.last_current = 0.0

    # ------------------------------------------------------- pack-level reads
    def mean_soc(self):
        return sum(c.z for c in self.cells) / len(self.cells)

    def min_cell_voltage(self):
        return min(c.V_ocv() - c.last_I * c.R_int() for c in self.cells)

    def max_cell_temp(self):
        return max(c.T for c in self.cells)

    def soc_spread(self):
        """Inter-cell SoC spread (max - min), in fraction. Multiply by 100 for %.
        A flat 0 under pack-level (no dispersion); grows under cell-resolved use
        and is what the balancer suppresses."""
        zs = [c.z for c in self.cells]
        return max(zs) - min(zs)

    # Eq. (2): pack OCV
    def V_ocv_pack(self):
        return sum(c.V_ocv() for c in self.cells)

    # Eq. (4): pack terminal voltage under current I
    def V_t_pack(self, I=None):
        if I is None:
            I = self.last_current
        return sum(c.V_ocv() - I * c.R_int() for c in self.cells)

    # ------------------------------------------------------- legacy API
    @property
    def soc(self):
        return self.mean_soc()

    @soc.setter
    def soc(self, value):
        v = float(max(0.0, min(1.0, value)))
        for c in self.cells:
            c.z = v

    @property
    def T(self):
        return sum(c.T for c in self.cells) / len(self.cells)

    def ocv(self):
        return self.V_ocv_pack()

    def terminal_voltage(self, current=None):
        v = self.V_t_pack(current)
        return max(self.V_cut * 0.95, v)

    def get_soc(self):
        return self.mean_soc()

    def get_voltage(self):
        return self.terminal_voltage()

    def get_temperature(self):
        return self.T

    def is_emergency(self):
        return (self.mean_soc() < 0.05
                or self.min_cell_voltage() < (self.V_cut / self.N_s)
                or self.max_cell_temp() > 60.0)

    def set_initial_soc(self, soc):
        v = float(max(0.0, min(1.0, soc)))
        for c in self.cells:
            c.z = v

    def add_regen(self, energy_wh):
        self.energy_regenerated_wh += float(energy_wh)
        self.regen_log.append(float(energy_wh))

    def estimate_remaining_time(self, p_avg_w=60.0):
        if p_avg_w <= 0:
            return float("inf")
        wh = self.mean_soc() * self.E_nom_Wh
        return wh / p_avg_w * 3600.0

    @property
    def energy_consumed(self):
        return self.energy_consumed_wh

    @property
    def energy_regenerated(self):
        return self.energy_regenerated_wh

    # ------------------------------------------------------- balancer (Eq. 7)
    def _maybe_balance(self):
        """Activate the shunt-resistor balancer every N_b control steps,
        when the inter-cell spread exceeds V_threshold."""
        if not self.enable_balance:
            return
        self._balance_counter += 1
        if self._balance_counter < self.N_b:
            return
        self._balance_counter = 0
        voltages = [c.V_ocv() for c in self.cells]
        V_mean = sum(voltages) / len(voltages)
        V_spread = max(voltages) - min(voltages)
        if V_spread <= self.V_threshold:
            return
        for c, V_i in zip(self.cells, voltages):
            if V_i > V_mean + 0.001:
                I_bleed = (V_i - V_mean) / self.R_s
                # Apply small SoC reduction equivalent to bleed over one period
                # (period = N_b · dt_control; dt_control ≈ 0.1 s -> N_b · 0.1 s)
                bleed_s = self.N_b * 0.1
                dz = I_bleed * bleed_s / (c.Q * 3600.0)
                c.z = max(0.0, c.z - dz)
                self.bleed_log.append((c, I_bleed, bleed_s))

    # ------------------------------------------------------- EKF (Eq. 6)
    def _ekf_update(self, V_measured):
        """Gated EKF correction: only update outside the 15-85% plateau where
        the OCV-vs-SoC slope is informative.
        """
        if not self.enable_ekf:
            return
        per_cell_meas = V_measured / self.N_s
        for i, c in enumerate(self.cells):
            if self.plateau_low < c.z < self.plateau_high:
                continue   # plateau: trust Coulomb counting alone
            H = _f_ocv_deriv(c.z)
            if abs(H) < 1e-4:
                continue
            P_i = self.P[i]
            K = P_i * H / (H * P_i * H + self.R_n)
            V_pred = c.V_ocv()
            c.z = max(0.0, min(1.0, c.z + K * (per_cell_meas - V_pred)))
            self.P[i] = (1.0 - K * H) * P_i

    # ------------------------------------------------------- thermal (Eq. 8)
    def _thermal_step(self, I, dt, Q_motor=0.0):
        """Coupled thermal: per-cell I²R heat + neighbour conduction
        + motor waste-heat recovery."""
        new_T = []
        n = len(self.cells)
        for i, c in enumerate(self.cells):
            heat_in = I * I * c.R_int()
            # neighbours
            T_left = self.cells[i - 1].T if i > 0 else c.T
            T_right = self.cells[i + 1].T if i < n - 1 else c.T
            coupling = self.k_c * (T_left - 2.0 * c.T + T_right)
            heat_out = (c.T - self.T_amb) / self.R_th
            dT = (heat_in + coupling + self.eta_h * Q_motor - heat_out) / self.C_th
            new_T.append(c.T + dT * dt)
        for c, t in zip(self.cells, new_T):
            c.T = t

    # ------------------------------------------------------- main step
    def update(self, power_w, dt, Q_motor=0.0, regen_wh=0.0):
        """Step the battery state given net electrical power demand (W).

        Strict thesis physics — no drain hacks, no fabricated regen:
          * Current from power and the *loaded* terminal voltage (Eq. 4):
                I = P / V_t(z, I_prev)
          * Per-cell SoC by Coulomb counting (Eq. 5) via coulomb_step().
          * Thermal coupling with motor waste-heat recovery (Eq. 8).
          * Gated EKF SoC correction (Eq. 6) and passive balancing (Eq. 7).

        `power_w` is the NET battery power: gross motor+electronics draw minus
        any regenerative power already harvested this tick (so the SoC and the
        energy ledger stay consistent — energy in = ∫ P_net dt, by construction).

        `regen_wh` (>=0) is the regen energy harvested this tick, reported by the
        caller (PhysicsEnergyManager, Eqs. 12-13) purely for bookkeeping/plots;
        it is NOT re-applied to SoC here (it is already folded into power_w).

        `dt` is effective battery-time in seconds (see coulomb_step).
        Returns the actual current drawn (A). A negative current means net
        charging (regen exceeded draw on this tick).
        """
        v_t = max(self.V_cut, self.V_t_pack(self.last_current))
        current = power_w / max(v_t, 1.0)

        for c in self.cells:
            c.coulomb_step(current, dt)

        # Heating scales with |I| (I^2 R) so it is valid for charge or discharge.
        self._thermal_step(current, dt, Q_motor=Q_motor)

        # Synthetic "measurement" = exact loaded V_t (sim has no sensor noise).
        V_meas = self.V_t_pack(current)
        self._ekf_update(V_meas)

        self._maybe_balance()

        self.last_current = current

        # Energy ledger — physically consistent with the SoC integral:
        #   consumed = ∫ max(P_net,0) dt,  regenerated = caller-reported harvest.
        # Both use the SAME effective dt as the Coulomb integral above, so the
        # "ΔSoC × E_nom" sanity check matches ∫P dt to within OCV/efficiency.
        if power_w > 0.0:
            self.energy_consumed_wh += power_w * dt / 3600.0
        if regen_wh > 0.0:
            self.energy_regenerated_wh += float(regen_wh)
            self.regen_log.append(float(regen_wh))
        return current


# Back-compat alias so existing imports still work
LiFePO4Battery = CellToPackBattery


# ============================================================================
# BLDC motor electrical model — Eq. (9), (10)
# ============================================================================
class BLDCMotor:
    def __init__(self, K_t, R_w, I_0=0.30, I_stall=8.0, P_ctrl=2.5):
        self.K_t = float(K_t)
        self.R_w = float(R_w)
        self.K_e = float(K_t)        # SI: K_e == K_t
        self.I_0 = float(I_0)
        self.I_stall = float(I_stall)
        self.P_ctrl = float(P_ctrl)

    def current(self, torque):
        """Eq. (9)."""
        return min(abs(torque) / max(self.K_t, 1e-3) + self.I_0, self.I_stall)

    def power(self, torque, omega):
        """Eq. (10): copper + back-EMF + controller overhead.
        (Iron-loss term is folded into P_ctrl for simplicity in v3.)
        """
        I_m = self.current(torque)
        return (I_m * I_m * self.R_w
                + self.K_e * I_m * abs(omega)
                + self.P_ctrl)


# Table II motor parameters from the paper
DEFAULT_DRIVE_MOTOR = lambda: BLDCMotor(K_t=0.15, R_w=0.80, I_0=0.50, I_stall=15.0)
DEFAULT_SHOULDER    = lambda: BLDCMotor(K_t=1.50, R_w=1.20, I_0=0.30, I_stall=8.0)
DEFAULT_ELBOW       = lambda: BLDCMotor(K_t=1.20, R_w=1.50, I_0=0.30, I_stall=8.0)
DEFAULT_WRIST       = lambda: BLDCMotor(K_t=0.80, R_w=2.00, I_0=0.30, I_stall=5.0)


# ============================================================================
# Energy-Aware Manager — drop-in replacement for the project's EnergyManager.
# Implements all v3 power equations including cascaded efficiency (Eq. 11),
# intelligent regen (Eq. 12-13), gravity-aware arm (Eq. 19-20), and the
# φ(z̄) SoC weight (Eq. 21).
# ============================================================================
class PhysicsEnergyManager:
    """Pack-interface energy manager wrapping the v3 cell-to-pack battery."""

    # Cascaded efficiency chain — Eq. (11)
    ETA_DC_DC = 0.97
    ETA_INV = 0.95
    ETA_CTRL = 0.97

    # Sim-time compression for the battery integral ONLY. The 512 Wh pack
    # (Sec 2.5.1) would barely move over a ~2-min demo at realistic draw
    # (~40 W -> ~0.13 Wh/min -> 0.025%/min SoC). To make the cell-resolved
    # dynamics legible in a short run WITHOUT distorting the physics, the
    # battery sees TIME_ACCEL * dt of "battery time" per sim tick. All power,
    # current, OCV, thermal and SoC relations stay exactly as the thesis
    # specifies; only the wall-clock-to-battery-clock mapping is scaled.
    # Set to 1.0 for true real-time physics.
    TIME_ACCEL = 60.0

    def __init__(self, config, enable_regen=True,
                 enable_ekf=True, enable_balance=True, enable_dispersion=True):
        ec = (config or {}).get("energy", {}) if config else {}
        rc = (config or {}).get("robot", {}) if config else {}

        # Variant toggles (Table 4.7). enable_regen gates Eq.(12-13) capture;
        # the battery flags select cell-resolved (full) vs pack-level behavior.
        self.enable_regen = bool(enable_regen)

        self.battery = CellToPackBattery(
            initial_soc=ec.get("initial_soc", 1.0),
            T_amb=ec.get("ambient_temperature", 25.0),
            enable_ekf=enable_ekf,
            enable_balance=enable_balance,
            enable_dispersion=enable_dispersion,
        )
        # Drives + arm motors
        self.drive_l = DEFAULT_DRIVE_MOTOR()
        self.drive_r = DEFAULT_DRIVE_MOTOR()
        self.arm_motors = [DEFAULT_SHOULDER(), DEFAULT_ELBOW(), DEFAULT_WRIST()]

        # Robot params
        self.wheel_radius = rc.get("wheel_radius", 0.06)
        self.wheel_sep = rc.get("wheel_separation", 0.44)
        # Mass calibrated so drive power matches thesis Table 1.3:
        # navigate scenario P_avg=38.7W.  With two BLDC motors, idle torque
        # from rolling friction (0.05*m*g) must absorb most of that budget.
        # m=12 kg → F_roll = 0.05*12*9.81 = 5.9 N → τ = 5.9*0.06/2 = 0.177 Nm
        # I_m = τ/K_t + I_0 = 0.177/0.15 + 0.50 = 1.68 A
        # P_per_motor = 1.68²*0.80 + 0.15*1.68*|ω| + 2.5 ≈ 7.3W idle
        # 2*7.3 + arm_idle + 15W electronics ≈ 38.7W  ✓
        self.mass = rc.get("mass", 12.0)
        self.g = 9.81

        # Arm geometry — Eq. (19) parameters per joint:
        # m_j (kg), l_com,j (m), J_j (kg·m²)
        self.arm_geom = [
            {"m": 0.80, "l_com": 0.15, "J": 0.012},   # shoulder
            {"m": 0.55, "l_com": 0.13, "J": 0.008},   # elbow
            {"m": 0.30, "l_com": 0.10, "J": 0.004},   # wrist
        ]

        # Electronics overhead (board + sensors + comms) — from thesis Table 1.3
        # Navigate scenario: P_avg=38.7W, 2 drive motors + arm idle + electronics
        self.P_electronics = 15.0

        # Cascaded chain
        self.eta_chain = self.ETA_DC_DC * self.ETA_INV * self.ETA_CTRL   # ≈ 0.894

        # Regen state
        self.last_v = 0.0
        self.epsilon = 0.01      # Eq. (21)

        # Q_motor: motor waste-heat fed to thermal model (Eq. 8)
        self.last_motor_heat = 0.0

        # Track gross regen energy for reporting (separate from SoC credit)
        self._gross_regen_wh = 0.0

    # ---------------------------------------------------------------- φ(z̄)
    def soc_penalty(self):
        """Eq. (21) — strictly verbatim."""
        z = self.battery.get_soc()
        return 1.0 / (0.5 + 0.5 * z + self.epsilon)

    # ---------------------------------------------------------------- regen
    @staticmethod
    def regen_efficiency(decel_mag):
        """Eq. (12): deceleration-dependent regen efficiency."""
        a = abs(decel_mag)
        if a < 0.5:
            return 0.85
        if a < 2.0:
            # Linear interpolation 0.85 -> 0.60 across [0.5, 2.0]
            t = (a - 0.5) / 1.5
            return 0.85 + t * (0.60 - 0.85)
        return 0.40

    # ---------------------------------------------------------------- drive
    def _drive_power(self, v_cmd, w_cmd):
        """Per-side wheel velocity from differential drive, then BLDC power."""
        v_left = v_cmd - w_cmd * self.wheel_sep / 2.0
        v_right = v_cmd + w_cmd * self.wheel_sep / 2.0
        w_left = v_left / max(self.wheel_radius, 1e-3)
        w_right = v_right / max(self.wheel_radius, 1e-3)

        # Crude force estimate: ma + rolling
        accel = (v_cmd - self.last_v) / 0.1
        F_drive = max(0.0, self.mass * accel) + 0.05 * self.mass * self.g
        torque_l = abs(F_drive * self.wheel_radius / 2.0)
        torque_r = abs(F_drive * self.wheel_radius / 2.0)

        return self.drive_l.power(torque_l, w_left) + \
               self.drive_r.power(torque_r, w_right)

    # ---------------------------------------------------------------- arm
    def _arm_power(self, arm_dq, arm_q):
        """Eq. (19-20): gravity-aware torque + per-joint motor power.
        arm_q is the *current* joint angles (cumulative); arm_dq the velocity.
        """
        if arm_q is None:
            arm_q = [0.0, 0.0, 0.0]
        total = 0.0
        for j, motor in enumerate(self.arm_motors):
            geom = self.arm_geom[j] if j < len(self.arm_geom) else self.arm_geom[-1]
            # Cumulative joint sum for gravity torque (Eq. 19)
            q_sum = sum(float(arm_q[i]) for i in range(0, j + 1)
                        if i < len(arm_q))
            tau_grav = geom["m"] * self.g * geom["l_com"] * math.sin(q_sum)
            dq = float(arm_dq[j]) if j < len(arm_dq) else 0.0
            # Inertia term (Eq. 19/20)
            tau_inertia = geom["J"] * dq / 0.1
            tau = tau_grav + tau_inertia
            # Eq. (20): per-joint power
            I_m = motor.current(tau)
            total += I_m * I_m * motor.R_w + motor.K_e * I_m * abs(dq) + motor.P_ctrl
        return total

    # ---------------------------------------------------------------- step
    def update(self, control, dt, arm_q=None):
        """One tick. control = [v_cmd, w_cmd, dq1, dq2, dq3].

        Returns the gross electrical power demand P_total (W) for plotting.
        The battery integrates the NET power (draw minus regen) so the SoC and
        the energy ledger stay physically consistent — no double counting.
        """
        v_cmd = float(control[0])
        w_cmd = float(control[1])
        arm_dq = control[2:5]

        P_drive = self._drive_power(v_cmd, w_cmd)
        P_arm = self._arm_power(arm_dq, arm_q)

        # Eq. (14): cascaded sum (Eq. 11) + electronics overhead.
        P_motors = P_drive + P_arm
        P_total = P_motors / self.eta_chain + self.P_electronics

        # Battery sees compressed time so a 512 Wh pack moves visibly in a
        # short demo while every physical relation stays thesis-exact.
        dt_batt = dt * self.TIME_ACCEL

        # Regen capture — Eq. (12)-(13). Fires on deceleration / sign reversal.
        # The harvested kinetic energy offsets the draw on THIS tick (net power),
        # so SoC and the ledger agree by construction.
        P_regen_w = 0.0
        E_regen_wh = 0.0
        if self.enable_regen and abs(v_cmd) < abs(self.last_v):
            decel = (abs(self.last_v) - abs(v_cmd)) / max(dt, 1e-3)
            kinetic_lost = 0.5 * self.mass * (self.last_v ** 2 - v_cmd ** 2)  # J
            if kinetic_lost > 0.0:
                eta = self.regen_efficiency(decel)                 # Eq. (12)
                E_regen_j = kinetic_lost * eta                     # Eq. (13)
                # Regen is a real power flow like P_total, so it must share the
                # same time base: power in real watts, ledger in battery-time
                # (x TIME_ACCEL) — otherwise regen under-counts by TIME_ACCEL
                # relative to the consumed ledger.
                P_regen_w = E_regen_j / max(dt, 1e-6)
                E_regen_wh = E_regen_j * self.TIME_ACCEL / 3600.0
                self._gross_regen_wh += E_regen_wh

        # Net battery power: never let regen drive the *consumed* ledger below
        # zero, but allow a net-charging current (SoC up) when regen > draw.
        P_net = P_total - P_regen_w

        # Motor heat = electrical energy not delivered to the shaft — folded back
        # via η_h in the thermal model (Eq. 8).
        motor_heat = max(0.0, P_motors * (1.0 - self.eta_chain))
        self.last_motor_heat = motor_heat

        self.battery.update(P_net, dt_batt, Q_motor=motor_heat,
                            regen_wh=E_regen_wh)
        self.last_v = v_cmd
        return P_total

    @property
    def energy_regenerated(self):
        """Total regen energy harvested (Wh) — matches battery.energy_regenerated."""
        return self.battery.energy_regenerated_wh

    # ---------------------------------------------------------------- legacy
    def get_soc(self):
        return self.battery.get_soc()

    def get_voltage(self):
        return self.battery.get_voltage()

    def get_temperature(self):
        return self.battery.get_temperature()

    def max_cell_temp(self):
        return self.battery.max_cell_temp()

    def soc_spread(self):
        return self.battery.soc_spread()

    def is_emergency(self):
        return self.battery.is_emergency()

    def set_initial_soc(self, s):
        self.battery.set_initial_soc(s)

    def estimate_remaining_time(self, p_avg_w=60.0):
        return self.battery.estimate_remaining_time(p_avg_w)

    @property
    def energy_consumed(self):
        return self.battery.energy_consumed

    @property
    def regen_log(self):
        return self.battery.regen_log
