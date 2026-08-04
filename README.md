Future Robots — Energy-Aware NMPC for Autonomous Mobile Manipulators
A simulation research framework for energy-aware Nonlinear Model Predictive Control (NMPC) applied to a 5-DOF mobile manipulator executing multi-box pick-and-place warehouse delivery missions through cluttered corridors with static and dynamic obstacles.

This codebase accompanies the peer-reviewed paper:

A. Louizini, A. Garmat, K. Guesmi — "A BYD-Inspired Cell-to-Pack Energy Model and Energy-Aware NMPC for Safe Autonomous Mobile Manipulators", CISTEM 2026 (v3).

Research Summary
The central research question: does cell-resolved battery modeling combined with energy-aware NMPC produce more reliable, longer-endurance autonomous missions than simpler energy abstractions?

Contributions
Cell-to-pack LiFePO₄ (LFP) battery model — a BYD-inspired 8S pack (25.6 V nominal, 20 Ah, 512 Wh) with per-cell manufacturing dispersion, gated EKF state-of-charge correction, passive shunt-resistor balancing, and cell-coupled thermal modeling with motor waste-heat recovery. All 21 equations from the paper are implemented in energy_paper.py.

Energy-aware NMPC — a 7-state nonlinear MPC solved via CasADi/IPOPT with SoC-dependent energy weighting φ(z) = 1/(0.5 + 0.5z + ε), warm-starting, horizon/iteration caps, and optional C JIT code generation. The NMPC runs during arm/manipulation phases; navigation uses a fast analytic curvature-adaptive pure-pursuit law with SoC weighting, keeping the solver off the navigation hot path.

Hybrid navigation pipeline — global A* planning (clearance-aware, with line-of-sight pruning and gradient smoothing) + local planning via pure-pursuit/CBF reactive avoidance, with pluggable alternatives (DWA, MPPI, NMPC-nav, Hybrid-A*) for A/B comparison in main20.py.

Corridor-aware navigation (main12+) — robot-centric local costmap with perpendicular cross-section sampling, corridor hysteresis memory, short-term blacklisting of collision-causing corridors, persistent JSON-backed cross-run memory, pinch-point detection, and escalating recovery with anti-loop wide reroute to break livelock in narrow passages.

Honest reliability measurement — real PyBullet contact-based collision detection (compliant bumper: penetration > 4 cm sustained 5 ticks = real collision), deterministic seeding, BLAS pinned to 1 thread per worker, worker count clamped to physical_cores − 2.

Controller Variants
Five variants are compared (Table 4.7 in the thesis):

Variant	Energy weight	SoC throttle	Regen	Cell EKF	Balancing	Dispersion	Speed scale
full	✓	✓	✓	✓	✓	✓	1.0
pack_level	✓	✓	✓	✗	✗	✗	1.0
no_energy	✗	✗	✓	✓	✓	✓	1.0
no_regen	✓	✓	✗	✓	✓	✓	1.0
speed_only	✗	✗	✓	✗	✗	✗	1.4
The comparison isolates how much the cell-resolved battery model, energy-aware NMPC cost, regenerative braking, and aggressive speed each contribute to reliability and endurance.

Architecture
Core Infrastructure
Package	Purpose
robot/	MobileManipulator — PyBullet URDF-driven differential-drive base + 3-DOF arm with gripper
simulation/	SimEnvironment — PyBullet world with obstacles, boxes, storage platforms, dynamic obstacles
perception/	NeuralObstacleDetector, SensorFusion, CameraVision (RGB color-blob box detection), OccupancyGrid
visualization/	Dashboard, AdvancedDashboard — matplotlib live telemetry
Control Trees
Each mainN.py has its own versioned helper tree under control/mainN_helpers/:

Module	Role
nmpc_controller.py	7-state CasADi/IPOPT NMPC with SoC-dependent energy weight
safety.py	CBF barrier-function safety filter
pure_pursuit.py	Curvature-adaptive pure pursuit with turn-in-place gate
astar_planner.py	8-connected grid A* with path pruning and smoothing
dwa_planner.py	Dynamic Window Approach local planner
path_learner.py	Path learning and refinement
Shared algorithms in control/main10_helpers/main3_helpers/:

Module	Role
energy_paper.py	BYD-inspired cell-to-pack LFP battery model + PhysicsEnergyManager (all 21 paper equations)
energy_policy.py	4-tier SoC policy (performance/balanced/frugal/emergency) + return-home reserve
warehouse_mission.py	Pick-place finite state machine (nav → grasp → place → next box)
world_aware_astar.py	Clearance-aware A* global planner with iterative inflation
dynamic_tracker.py	Moving-obstacle velocity estimation
perception_fusion.py	Fuses detector + IR + camera + tracker
path_clearance.py	Pinch-point detection, clearance scoring, CorridorPredictor, DynamicConflictPredictor
corridor_geometry.py	CorridorGeometryExtractor, is_wedged
shims.py	Compatibility layer for the energy manager
Experimental Milestones
Each mainN.py is a frozen milestone — the experiment runs as published at that point in development.

File	Role
main9.py	Foundation. Fixes overshoot/orbiting, dueling recovery loops, and 230 ms NMPC on the navigation hot path. Defines EnergyAwarePursuit, DockingController, ReactiveAvoider, ProgressWatchdog, RecoveryAction — imported by all later mains. NMPC runs only during arm phases; navigation uses the fast analytic pursuit law.
main10.py	Chapter-4 benchmark harness. Wraps main9's frozen control logic in the thesis experimental setup. Defines variant_settings() (5 variants), CorridorWarehouseMission (3-box delivery, diagonal slalom), CollisionMonitor (honest contact-based failure). Full metric set + batch harness with --batch N --workers W.
main11.py	Corridor-awareness prototype. Adds predictive obstacle inflation, stricter corridor scoring, cooldown-gated replan triggers, low-clearance speed reduction.
main12.py	Primary thesis benchmark. Adds LocalCostmap with cross-section sampling, CorridorMemory (hysteresis), CorridorBlacklist (TTL), PersistentCorridorMemory (JSON-backed), EscalatingRecovery (anti-livelock), EnhancedCorridorWarehouseMission (on-path slalom forcing weave), choose_best_path() (multi-candidate path scoring), real-time NMPC tuning (--nmpc-horizon, --nmpc-max-iter, --nmpc-jit).
main13.py	Incremental iteration fork of main12 with its own helper tree for parallel experimentation.
main14.py	Further iteration of the main12 corridor-aware architecture with its own isolated helper tree.
main20.py	Navigation architecture A/B comparison. Pluggable local planner (--nav-planner baseline/mppi/dwa/nmpc) and global planner (--global-planner astar/hybrid). Fully isolated from main12; reuses only main9 controllers + main10 mission machinery.
Patched Modules (Main12_fix/)
Root-cause fixes documented for the corridor wedging problem:

Module	Fix
control_patched.py	CorridorAwareWatchdog (velocity-scaled gain, skips kick if still moving) + WallAwareRecovery (pivots instead of reversing when rear is blocked)
planning_patched.py	plan_with_clearance_safe (returns None instead of fake straight line on failure) + reject_colliding
safety_patched.py	PatchedSafetyFilter (splits frontal vs lateral CBF violations so corridor walls don't zero forward velocity)
Key Algorithms
Battery / Energy Model (energy_paper.py)
Eq. 1: Per-cell manufacturing dispersion (δ_Q ∈ [−0.02, +0.02], δ_R ∈ [−0.05, +0.05])
Eq. 2–5: Pack OCV (piecewise-linear LFP table, ~3.28 V plateau 20–80% SoC), temperature-dependent internal resistance, terminal voltage, Coulomb-counting SoC
Eq. 6: Gated EKF SoC correction (only corrects outside the 0.15–0.85 plateau where dV/dz is informative)
Eq. 7: Passive shunt-resistor balancer (R_s = 10 Ω, threshold 20 mV, period 100 steps)
Eq. 8: Cell-coupled thermal model with motor waste-heat recovery (k_c = 0.5, η_h = 0.15)
Eq. 9–13: BLDC motor current, per-motor electrical power, cascaded efficiency chain (η ≈ 0.894), intelligent regen efficiency (0.85 at gentle decel → 0.40 at hard braking)
Eq. 14: Total electrical power
Eq. 18–21: NMPC predicted drive/arm power, gravity-comp torque, SoC-dependent weight φ(z̄)
Energy Policy (energy_policy.py)
4-tier SoC policy scaling speed, arm velocity, and planner weights:

Tier	SoC range	Speed scale	Behavior
Performance	> 0.50	1.00	Full speed, normal energy weight
Balanced	0.30 – 0.50	0.85	Reduced speed, higher energy weight
Frugal	0.15 – 0.30	0.55	Heavily reduced speed
Emergency	< 0.15	0.00	Stop + return home
Navigation
WorldAwareAStar — grid A* with line-of-sight pruning, gradient smoothing, iterative inflation for clearance
EnergyAwarePursuit — curvature-adaptive pure pursuit with turn-in-place gate (|α| > threshold → pivot with v=0; else drive). Adaptive lookahead, curvature speed law, SoC weighting.
ReactiveAvoider — CBF angular bias + admissible braking speed
DockingController — final-approach point-then-shoot within engage radius
CorridorPredictor — lookahead replan trigger for narrowing corridors
DynamicConflictPredictor — predicted moving-obstacle conflicts
EscalatingRecovery — escalating reverse-and-reface (level 1: 0.6 s, level 2: ~1.2 s, ..., capped at 4), alternating turn direction at level 3+, reset on genuine progress
Safety Filter
CBF barrier-function filter with frontal/lateral split (patched version) so corridor walls don't zero forward velocity — the root cause of "stuck near corridor" creep — adding a centering steer toward the freer side.

Mission FSM

init → nav_to_pick → pre_grasp → descend → grasp → lift →
nav_to_place → pre_place → place → retract → (next box | return_home) → done
With reachable place standoff, force-gated phase transitions, and grip constraints.

Installation
Requirements
Python 3.10+
PyBullet — physics simulation
CasADi — NMPC NLP solver (IPOPT backend)
NumPy, Matplotlib, PyYAML

pip install pybullet casadi numpy matplotlib pyyaml scipy
Configuration
A central config.yaml controls simulation timestep, NMPC parameters (dt, horizon, max_iter, R_energy), robot velocity/joint limits, energy thresholds, and safety parameters.

Usage
Single Mission

python main12.py --variant full --seed 1000 --gui
Options:

--variant — one of: full, pack_level, no_energy, no_regen, speed_only
--seed — random seed for reproducibility
--gui — show PyBullet GUI (omit for headless)
--nmpc-horizon N — NMPC horizon (default 8)
--nmpc-max-iter N — IPOPT max iterations (default 30)
--nmpc-jit — enable CasADi C code generation for faster solves
Batch Experiments

python main12.py --variant full --batch 50 --workers 4
Runs 50 seeded missions across 4 parallel workers. BLAS is pinned to 1 thread per worker; worker count is clamped to physical_cores − 2.

Navigation A/B Comparison (main20)

python main20.py --nav-planner mppi --global-planner hybrid --seed 1000
python main20.py --nav-planner dwa --global-planner astar --batch 30
Stress Tests

python _thesis_stress.py        # Initial-SoC sweep: {0.70, 0.55, 0.40, 0.30, 0.20, 0.15}
python _thesis_stress_dyn.py   # Dynamic-obstacle sweep: {0, 1, 2} moving obstacles
Thesis Figures and Statistics

python _thesis_stats.py     # Aggregate mean±std per variant, energy-savings percentages
python _thesis_newfigs.py   # Generate 5 comparison figures into Thesis_Final/figures/
python _thesis_audit.py     # Scan .tex manuscript for inconsistent claims
Results
Key Findings
Corridor-aware navigation (main12) improved reliability from ~85% to ~92% over the main10 baseline, while reducing NMPC solve time 5× (26 ms vs 160 ms mean) thanks to horizon 8 + max_iter 30 — the shorter horizon actually tracks the tight slalom better.

The dominant failure mode is recovery livelock — timeouts with 1300–1800 recovery kicks in narrow passages. EscalatingRecovery + anti-loop wide reroute were designed to break this.

Energy-aware variants are more energy-efficient than speed-only/no-energy variants, at the cost of longer mission times.

Low-SoC robustness: the system is reliable down to 40% initial SoC; below 30% the emergency-stop guard (10% SoC) trips and every run fails as "battery" — validating the SoC throttle φ(z).

Dynamic obstacles remain the hard case: even 1 sinusoidal AGV drops success significantly, motivating progressive-difficulty calibration.

Cell balance (SoC spread) stays low (~0.38%) under the full variant due to the passive balancer; pack_level has flat 0 spread (no dispersion modeled).

Output Data
Batch runs produce CSV files with per-mission metrics:

variant, seed, success, boxes, num_boxes, failure, mission_time_s, path_m, energy_wh, regen_wh, final_soc, terminal_v, max_cell_temp, soc_spread, nmpc_mean_ms, nmpc_p99_ms, nmpc_solves, switches, astar_replans, recoveries, collisions, wedge_escapes, corridor_narrow_s, corridor_class

Stress test summaries are in stress/stress_summary_soc.csv and stress/stress_summary_dyn.csv.

Project Structure

Future_robots/
├── main9.py                  # Foundation: frozen controllers (EnergyAwarePursuit, etc.)
├── main10.py                 # Chapter-4 benchmark: 5 variants, batch harness
├── main11.py                 # Corridor-awareness prototype
├── main12.py                 # Primary thesis benchmark (corridor-aware)
├── main13.py                 # Iteration fork of main12
├── main14.py                 # Further iteration of main12
├── main20.py                 # Navigation architecture A/B comparison
├── control/
│   ├── main10_helpers/
│   │   ├── main3_helpers/    # Shared algorithms (energy, planning, perception)
│   │   ├── astar_planner.py
│   │   ├── dwa_planner.py
│   │   └── ...
│   ├── main11_helpers/
│   ├── main12_helpers/
│   ├── main13_helpers/
│   ├── main14_helpers/
│   └── main20_helpers/
├── Main12_fix/               # Root-cause patches for corridor wedging
│   └── Main12_helpers/
├── _thesis_*.py              # Thesis analysis and figure generation
├── stress/                   # Stress test results
├── *.csv                     # Batch experiment results
└── Figure_robot_Amr.png      # Robot diagram
License and Academic Use
Copyright © 2026 Louizini Abderahmane, Garmat, Guesmi. All rights reserved.

This software and its accompanying documentation are provided for review and academic reference purposes only. The codebase supports a peer-reviewed publication submitted to CISTEM 2026.

Terms
No unauthorized use. You may not copy, modify, merge, publish, distribute, sublicense, or sell any part of this codebase without express written permission from the copyright holders.

No derivative works. Creating modified versions, forks, or adaptations of this software for any purpose — including research, coursework, or commercial use — is prohibited without prior written authorization.

Citation required. If this work informs your research, you must cite the original publication:

Louizini, A., Garmat, A., Guesmi, K. "A BYD-Inspired Cell-to-Pack Energy Model and Energy-Aware NMPC for Safe Autonomous Mobile Manipulators." CISTEM 2026.

No commercial use. Commercial use of this software, in whole or in part, is strictly prohibited without a separate commercial license.

Academic integrity. Submitting this work, in whole or in part, as your own — or using it to produce a derivative publication without proper attribution — constitutes academic misconduct and copyright infringement.

For licensing inquiries, contact the repository owner via GitHub.

Citation
If you reference this work in a publication, please cite:


@inproceedings{louizini2026energynmpc,
  title     = {A BYD-Inspired Cell-to-Pack Energy Model and Energy-Aware NMPC for Safe Autonomous Mobile Manipulators},
  author    = {Louizini, Abderahmane and Garmat, A. and Guesmi, K.},
  booktitle = {Proceedings of the International Conference on Electrical Sciences and Technologies (CISTEM)},
  year      = {2026},
  note      = {Manuscript v3}
}
Acknowledgments
This work was developed as part of a master's thesis on energy-aware autonomous mobile manipulators, combining cell-resolved battery modeling, nonlinear model predictive control, and hybrid navigation in warehouse environments.
