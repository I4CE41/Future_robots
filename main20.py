"""main20.py -- new navigation architectures for the robot rescue benchmark.

main20 is fully isolated from main12: it reuses only the shared foundation
(main9 controllers + main10 mission machinery) and its own self-contained
``control/main20_helpers`` tree (a copy of main12_helpers plus new planner
modules). It keeps the proven main12 mission flow -- perception, corridor
extraction, energy model, safety filter, docking / watchdog / recovery,
collision monitor, batch harness -- but replaces the *navigation* control law
with a pluggable local planner selected at run time:

  --nav-planner baseline   EnergyAwarePursuit + ReactiveAvoider (the main12 nav)
  --nav-planner mppi        Model Predictive Path Integral (sampling MPC)
  --nav-planner dwa         Dynamic Window Approach
  --nav-planner nmpc        NMPC-driven base navigation (one optimiser for
                            nav AND arm -- the "deep fix")

and a switchable global planner:

  --global-planner astar    WorldAwareAStar grid planner (default)
  --global-planner hybrid   kinodynamic Hybrid-A*

This lets the thesis A/B every method against the baseline on identical
missions, seeds and metrics.
"""

import argparse
import csv
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pybullet as p

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
PATH_MEMORY_FILE = os.path.join(_ROOT, "path_memory.json")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _HERE not in sys.path:
    sys.path.append(_HERE)

from robot.mobile_manipulator import MobileManipulator
from simulation.environment import SimEnvironment
from perception.neural_detector import NeuralObstacleDetector
from perception.sensor_fusion import SensorFusion
from perception.camera_vision import CameraVision
from perception.occupancy_grid import OccupancyGrid
from visualization.dashboard import Dashboard

from main9 import (
    EnergyAwarePursuit,
    DockingController,
    ReactiveAvoider,
    ProgressWatchdog,
    RecoveryAction,
    densify,
)

from robot_rescue.control.main20_helpers.nmpc_controller import NMPCController
from robot_rescue.control.main20_helpers.safety import SafetyFilter
from robot_rescue.control.main20_helpers.main3_helpers.shims import apply_shims
from robot_rescue.control.main20_helpers.main3_helpers.dynamic_tracker import DynamicTracker
from robot_rescue.control.main20_helpers.main3_helpers.world_aware_astar import WorldAwareAStar
from robot_rescue.control.main20_helpers.main3_helpers.energy_paper import PhysicsEnergyManager
from robot_rescue.control.main20_helpers.main3_helpers.perception_fusion import PerceptionFusion
from robot_rescue.control.main20_helpers.main3_helpers.path_clearance import (
    CorridorPredictor,
    DynamicConflictPredictor,
    lookahead_min_clearance,
    path_min_clearance,
    plan_with_clearance,
    los_safe,
    predicted_obstacle_sweep,
)
from robot_rescue.control.main20_helpers.main3_helpers.corridor_geometry import (
    CorridorGeometryExtractor,
    is_wedged,
)
from robot_rescue.control.main20_helpers.hybrid_astar import HybridAStarPlanner
from robot_rescue.control.main20_helpers.nav_common import (
    LocalCostmap,
    CorridorMemory,
    CorridorBlacklist,
    PersistentCorridorMemory,
    BaselinePursuitPlanner,
    DWAPlannerAdapter,
    NavContext,
    choose_best_path,
    corridor_escape_control,
)
from robot_rescue.control.main20_helpers.mppi_planner import MPPIPlanner
from robot_rescue.control.main20_helpers.nmpc_nav import NMPCNavPlanner

from robot_rescue.main10 import (
    variant_settings,
    CorridorWarehouseMission,
    CollisionMonitor,
    load_config,
    seed_everything,
)


class EnhancedCorridorWarehouseMission(CorridorWarehouseMission):
    """Keeps the main10 layout, but records a few corridor hints for clarity."""

    def _spawn_corridor_field(self):
        super()._spawn_corridor_field()
        try:
            p1 = np.asarray(self.pick_pos, float)
            p2 = np.asarray(self.place_pos, float)
            vec = p2 - p1
            dist = float(np.linalg.norm(vec))
            if dist > 1e-6:
                unit = vec / dist
                perp = np.array([-unit[1], unit[0]])
                self.corridor_centerline = (tuple(p1[:2]), tuple(p2[:2]))
                self.corridor_direction = tuple(unit[:2])
                self.corridor_perp = tuple(perp[:2])
        except Exception:
            self.corridor_centerline = None
            self.corridor_direction = None
            self.corridor_perp = None


def build_local_planner(name, config, vs, nmpc, corridor_extractor, nmpc_dt, args):
    """Factory for the selected navigation local planner."""
    name = (name or "baseline").lower()
    spd = vs["speed_scale"]
    if name == "baseline":
        return BaselinePursuitPlanner(v_ref=0.6 * spd, turn_gate=0.6, w_turn=1.8,
                                      d_safe=0.45)
    if name == "dwa":
        return DWAPlannerAdapter(config)
    if name == "mppi":
        return MPPIPlanner(config,
                           samples=args.mppi_samples,
                           horizon=args.mppi_horizon,
                           lam=args.mppi_lambda,
                           seed=args.seed)
    if name == "nmpc":
        return NMPCNavPlanner(nmpc, corridor_extractor, nmpc_dt=nmpc_dt)
    raise ValueError(f"unknown nav-planner: {name}")


def run_mission(args):
    config = load_config(args.config)
    vs = variant_settings(args.variant)

    if not vs["energy_weight"]:
        config.setdefault("nmpc", {})["R_energy"] = 0.0

    # NMPC tuning knobs (shared by arm-phase NMPC and the optional nmpc nav).
    nmpc_cfg = config.setdefault("nmpc", {})
    nmpc_cfg["horizon"] = int(args.nmpc_horizon) if args.nmpc_horizon > 0 else 8
    if args.nmpc_max_iter > 0:
        nmpc_cfg["max_iter"] = int(args.nmpc_max_iter)
    nmpc_cfg["jit"] = bool(getattr(args, "nmpc_jit", False))

    client = p.connect(p.GUI if args.gui else p.DIRECT)
    p.setRealTimeSimulation(0, physicsClientId=client)
    if args.gui:
        p.resetDebugVisualizerCamera(6, 0, -75, [0.25, 0.75, 0], physicsClientId=client)

    env = SimEnvironment(client, config)
    start = [0.0, 0.0, 0.0]
    robot = MobileManipulator(client, config, start_pos=[0.0, 0.0, 0.10],
                              start_orn=p.getQuaternionFromEuler([0, 0, 0]))

    docker = DockingController(engage_r=0.75, face_tol=0.35, v_max=0.35)
    nmpc = NMPCController(config)
    safety = SafetyFilter(config)
    astar = WorldAwareAStar()
    hybrid = HybridAStarPlanner()
    occ = OccupancyGrid()

    detector = NeuralObstacleDetector(client, config, robot_id=robot.robot_id,
                                      ground_id=env.plane_id)
    fusion = SensorFusion(config)
    camera = CameraVision(client, config, robot.robot_id)
    energy = PhysicsEnergyManager(config,
                                  enable_regen=vs["enable_regen"],
                                  enable_ekf=vs["enable_ekf"],
                                  enable_balance=vs["enable_balance"],
                                  enable_dispersion=vs["enable_dispersion"])
    apply_shims(energy)
    tracker = DynamicTracker(env)
    perception = PerceptionFusion(detector, fusion, camera, env, tracker)
    costmap = LocalCostmap(robot_radius=0.30, safety_margin=0.18, radius=2.8, resolution=0.10)
    corridor_extractor = CorridorGeometryExtractor(robot_radius=0.30, safety_margin=0.12,
                                                   ray_range=2.8, ray_resolution=0.05,
                                                   path_spacing=0.22, profile_span=2.4)
    corridor_memory = CorridorMemory(hysteresis=0.20, smooth=0.35)

    sim_dt = config["simulation"]["timestep"]
    nmpc_dt = config.get("nmpc", {}).get("dt", 0.1)
    ctrl_every = max(1, int(nmpc_dt / sim_dt))
    astar_period = ctrl_every * 10

    # Build the selected navigation planner + global planner.
    local_planner = build_local_planner(args.nav_planner, config, vs, nmpc,
                                         corridor_extractor, nmpc_dt, args)
    global_planner = hybrid if args.global_planner == "hybrid" else astar

    corridor_predictor = CorridorPredictor(lookahead_m=1.9, min_clearance=0.30,
                                            cooldown_s=1.2, dt=nmpc_dt)
    conflict_predictor = DynamicConflictPredictor(lookahead_path_m=2.2,
                                                  horizon_s=2.5, dt_sample=0.05,
                                                  conflict_dist=0.65, cooldown_s=1.2,
                                                  dt=nmpc_dt)
    watchdog = ProgressWatchdog(window_s=3.0, min_gain=0.15, dt=nmpc_dt)
    recovery = RecoveryAction(dt=nmpc_dt)
    corridor_blacklist = CorridorBlacklist(ttl_s=18.0, dt=nmpc_dt)
    persistent_corridor_memory = PersistentCorridorMemory(PATH_MEMORY_FILE)
    corridor_state = None
    corridor_escape_ticks = 0
    corridor_escape_limit = int(1.2 / nmpc_dt)
    corridor_escape_cool = 0
    wedge_escapes = 0
    corridor_narrow_s = 0.0
    corridor_class = "none"
    if getattr(args, "reset_path_memory", False):
        persistent_corridor_memory.reset()
        print(f"[main20] reset path memory: {PATH_MEMORY_FILE}")

    mission = EnhancedCorridorWarehouseMission(config, env, robot, energy,
                                               initial_soc=args.initial_soc,
                                               num_boxes=args.num_boxes,
                                               start=start, dyn_count=args.dyn_count,
                                               dynamic=bool(getattr(args, "dynamic", False)))
    collision = CollisionMonitor(client, robot.robot_id, env)

    dashboard = None
    if args.gui and not args.no_dashboard:
        try:
            dashboard = Dashboard(config)
            plt.show(block=False)
        except Exception:
            dashboard = None

    soc_throttle = vs["soc_throttle"]

    for _ in range(50):
        try:
            p.stepSimulation(physicsClientId=client)
        except Exception:
            break

    state = robot.get_state()
    current_path = []
    last_astar = -astar_period
    prev_phase = mission.phase
    final_control = np.zeros(5)
    nmpc_prev = np.zeros(5)
    sim_time = 0.0
    path_len = 0.0
    prev_xy = (state["x"], state["y"])
    nmpc_solves = 0
    last_nmpc_solve_time = 0.0
    last_predicted_traj = None
    power = 0.0
    dock_stall = 0
    dock_stall_limit = int(4.0 / nmpc_dt)
    astar_replans = 0
    switches = 0
    collision_hits = 0
    failure = None

    try:
        for step in range(args.max_steps):
            try:
                p.getConnectionInfo(physicsClientId=client)
            except Exception:
                break

            state = robot.get_state()
            path_len += math.hypot(state["x"] - prev_xy[0], state["y"] - prev_xy[1])
            prev_xy = (state["x"], state["y"])
            mission.update_goal(state)
            goal = mission.goal
            goal_xy = (goal[0], goal[1])

            if mission.phase != prev_phase:
                current_path = []
                watchdog.reset()
                dock_stall = 0
                corridor_escape_ticks = 0
                corridor_escape_cool = 0
                prev_phase = mission.phase
                collision_hits = 0

            soc = energy.get_soc()
            if corridor_escape_cool > 0:
                corridor_escape_cool -= 1
            if corridor_escape_ticks > 0:
                corridor_escape_ticks -= 1

            if step % ctrl_every == 0:
                corridor_predictor.tick()
                conflict_predictor.tick()
                corridor_blacklist.tick()

                want_vision_for = None
                try:
                    if args.use_camera and mission.phase == "nav_to_pick":
                        want_vision_for = ["yellow", "orange", "blue"][mission.pick_idx % 3]
                except Exception:
                    want_vision_for = None

                perc = perception.perceive(robot, state, sim_time, want_vision_for=want_vision_for)
                obstacles = list(perc.obstacles)
                ir = perc.ir

                try:
                    occ.update_from_obstacles(obstacles)
                    occ.update_from_ir(state["x"], state["y"], state["theta"], ir)
                    occ.decay()
                    for so in occ.get_obstacles_list():
                        if not any(math.hypot(so[0] - o[0], so[1] - o[1]) < 0.3 for o in obstacles):
                            obstacles.append(so)
                except Exception:
                    pass

                if energy.is_emergency():
                    failure = "battery"
                    break

                rx, ry = state["x"], state["y"]
                obstacles = [(ox, oy, r) for ox, oy, r in obstacles
                             if math.hypot(ox - rx, oy - ry) > 0.35
                             and math.hypot(ox - goal_xy[0], oy - goal_xy[1]) > 0.40]

                dynamic_sweep = predicted_obstacle_sweep(tracker, sim_time,
                                                         horizon_s=1.8,
                                                         dt_sample=0.25,
                                                         inflate=0.12)
                augmented_obstacles = obstacles + dynamic_sweep
                costmap.build((rx, ry), augmented_obstacles)
                corridor_state = corridor_extractor.extract(
                    state, goal_xy, current_path or densify([(rx, ry), goal_xy], 0.15),
                    augmented_obstacles)
                if (corridor_state.is_narrow or corridor_state.is_tight or
                        corridor_state.is_dead_end or
                        corridor_state.classification == "constrained"):
                    dynamic_sweep = predicted_obstacle_sweep(tracker, sim_time,
                                                             horizon_s=2.5,
                                                             dt_sample=0.05,
                                                             inflate=0.16)
                    augmented_obstacles = obstacles + dynamic_sweep
                    costmap.build((rx, ry), augmented_obstacles)
                    corridor_state = corridor_extractor.extract(
                        state, goal_xy, current_path or densify([(rx, ry), goal_xy], 0.15),
                        augmented_obstacles)

                corridor_class = getattr(corridor_state, "classification", "none")
                if (corridor_state.is_narrow or corridor_state.is_tight or
                        corridor_state.is_dead_end or
                        corridor_state.classification == "constrained"):
                    corridor_narrow_s += sim_dt

                if current_path and len(current_path) > 1:
                    nearest_i = min(range(len(current_path)),
                                    key=lambda i: math.hypot(current_path[i][0] - rx,
                                                             current_path[i][1] - ry))
                    ref_pt = current_path[min(nearest_i + 1, len(current_path) - 1)]
                else:
                    ref_pt = goal_xy
                ref_dir = (ref_pt[0] - rx, ref_pt[1] - ry)
                corridor_report = costmap.corridor_cross_section((rx, ry), ref_dir, span=2.2)
                corridor_offset = corridor_memory.update(corridor_report)
                if corridor_report:
                    corridor_report = dict(corridor_report)
                    corridor_report["bad_penalty"] = corridor_blacklist.penalty(corridor_report)

                current_lookahead_clearance = float("inf")
                if current_path:
                    current_lookahead_clearance = lookahead_min_clearance(
                        current_path, augmented_obstacles, (rx, ry), lookahead_m=1.8)

                sensor_front = float(getattr(perc, "front_clearance", 1.0))
                sensor_left = float(getattr(perc, "left_clearance", 1.0))
                sensor_right = float(getattr(perc, "right_clearance", 1.0))
                sensor_side_bias = sensor_left - sensor_right

                if mission.is_arm_phase:
                    final_control = np.zeros(5)
                    try:
                        corridor_for_nmpc = corridor_extractor.to_nmpc_corridor(
                            corridor_state, state, nmpc.N + 1, nmpc_dt)
                        u, solve_t, X_sol = nmpc.solve(state["full_q"], goal, obstacles, soc,
                                             u_prev=nmpc_prev,
                                             corridor=corridor_for_nmpc)
                        nmpc_prev = u.copy()
                        nmpc_solves += 1
                        last_nmpc_solve_time = solve_t
                        last_predicted_traj = np.asarray(X_sol)[:, :2]
                    except Exception:
                        pass
                else:
                    need = (not current_path or
                            (step - last_astar) >= astar_period or
                            (current_path and math.hypot(current_path[-1][0] - goal_xy[0],
                                                         current_path[-1][1] - goal_xy[1]) > 0.25))

                    if corridor_state.is_dead_end:
                        need = True
                    if current_lookahead_clearance < 0.30:
                        need = True
                    if sensor_front < 0.42:
                        need = True
                    if corridor_report and corridor_report["width"] < 1.00:
                        need = True
                    if corridor_memory.off_center(corridor_report):
                        need = True
                    if abs(corridor_offset) > 0.22:
                        need = True
                    if corridor_blacklist.is_bad(corridor_report) or persistent_corridor_memory.is_bad(corridor_report):
                        need = True

                    if current_path and not need:
                        ci = min(range(len(current_path)),
                                 key=lambda i: math.hypot(current_path[i][0] - rx,
                                                          current_path[i][1] - ry))
                        for wx, wy in current_path[ci:]:
                            if any(math.hypot(wx - ox, wy - oy) < r + 0.35 for ox, oy, r in augmented_obstacles):
                                need = True
                                break

                    if corridor_predictor.should_replan_now(state, current_path, augmented_obstacles):
                        need = True
                    if conflict_predictor.should_replan_now(state, current_path, tracker, sim_time):
                        need = True

                    if need:
                        candidate_paths = []
                        if los_safe(state, goal_xy, augmented_obstacles,
                                    robot_radius=0.3, safety=0.45):
                            direct_path = densify([(rx, ry), goal_xy], 0.15)
                            candidate_paths.append(("direct", direct_path))

                        plan_a, info_a = plan_with_clearance(
                            global_planner, (rx, ry), goal_xy, augmented_obstacles,
                            min_clearance=0.30, gap_min=0.95,
                            max_attempts=6, inflation_step=0.25,
                        )
                        candidate_paths.append(("clear", densify(plan_a if plan_a else [(rx, ry), goal_xy], 0.15)))

                        plan_b, info_b = plan_with_clearance(
                            global_planner, (rx, ry), goal_xy, augmented_obstacles,
                            min_clearance=0.38, gap_min=1.05,
                            max_attempts=4, inflation_step=0.30,
                        )
                        candidate_paths.append(("conservative", densify(plan_b if plan_b else [(rx, ry), goal_xy], 0.15)))

                        best_path, best_score, best_details, best_label = choose_best_path(
                            current_path, candidate_paths, augmented_obstacles, (rx, ry), goal_xy,
                            costmap=costmap,
                            persistent_memory=persistent_corridor_memory,
                            perception=perc,
                            hysteresis=0.20)
                        if best_path is not current_path:
                            current_path = best_path
                            if best_label != "hold":
                                astar_replans += 1
                        else:
                            current_path = best_path
                        if best_details.get("min_clearance", float("inf")) < 0.34:
                            astar_replans += 1
                        last_astar = step

                    bearing = perc.target_bearing
                    dist_goal = math.hypot(goal_xy[0] - rx, goal_xy[1] - ry)
                    if args.use_camera and mission.phase == "nav_to_pick" and dist_goal < 2.6:
                        bearing = perc.target_bearing if perc.target_bearing is not None else bearing

                    docking_now = False
                    geometric_wedge = (
                        corridor_state is not None
                        and (corridor_state.is_narrow or
                             corridor_state.is_tight or
                             corridor_state.is_dead_end)
                    )
                    wedge_now = (
                        geometric_wedge
                        and is_wedged(perc)
                        and corridor_escape_cool <= 0
                        and not docker.in_range(state, goal_xy)
                    )
                    if wedge_now:
                        final_control = corridor_escape_control(state, corridor_state)
                        corridor_escape_ticks = corridor_escape_limit
                        corridor_escape_cool = int(1.4 / nmpc_dt)
                        current_path = []
                        wedge_escapes += 1
                        corridor_blacklist.mark_bad(corridor_report)
                        persistent_corridor_memory.mark_bad(corridor_report, context="wedge")
                    elif recovery.active:
                        bv, bw, _ = recovery.step()
                        final_control = np.array([bv, bw, 0, 0, 0])
                    elif docker.in_range(state, goal_xy):
                        docking_now = True
                        v, w = docker.compute(state, goal_xy)
                        final_control = np.array([v, w, 0, 0, 0])
                        dock_stall += 1
                        if dock_stall > dock_stall_limit:
                            recovery.trigger(state, goal_xy)
                            current_path = []
                            dock_stall = 0
                            bv, bw, _ = recovery.step()
                            final_control = np.array([bv, bw, 0, 0, 0])
                    elif watchdog.update(state, goal_xy):
                        recovery.trigger(state, goal_xy)
                        corridor_blacklist.mark_bad(corridor_report)
                        persistent_corridor_memory.mark_bad(corridor_report, context="watchdog")
                        current_path = []
                        bv, bw, _ = recovery.step()
                        final_control = np.array([bv, bw, 0, 0, 0])
                    else:
                        dock_stall = 0
                        soc_in = soc if soc_throttle else 1.0
                        ctx = NavContext(
                            state=state, path=current_path, obstacles=augmented_obstacles,
                            corridor_state=corridor_state, corridor_report=corridor_report,
                            corridor_offset=corridor_offset, goal_xy=goal_xy, soc=soc_in,
                            perc=perc, bearing=bearing, sensor_front=sensor_front,
                            sensor_side_bias=sensor_side_bias, nmpc_dt=nmpc_dt)
                        v, w, info = local_planner.compute(ctx)
                        final_control = np.array([v, w, 0, 0, 0])
                        if info.get("nmpc_solve_t") is not None:
                            nmpc_solves += 1
                            last_nmpc_solve_time = info["nmpc_solve_t"]
                            last_predicted_traj = info.get("nmpc_traj")
                        switches = getattr(local_planner, "switches", switches)

                    try:
                        safe_obs = [] if docking_now else augmented_obstacles
                        final_control = safety.filter_control(final_control, state, safe_obs, soc)
                    except Exception:
                        final_control = np.zeros(5)
                        recovery.trigger(state, goal_xy)

            if collision.check():
                collision_hits += 1
                corridor_blacklist.mark_bad(corridor_report)
                persistent_corridor_memory.mark_bad(corridor_report, context="collision")
                recovery.trigger(state, goal_xy)
                current_path = []
                if collision_hits >= 3:
                    failure = "collision"
                    break
                try:
                    bv, bw, _ = recovery.step()
                    final_control = np.array([bv, bw, 0, 0, 0])
                except Exception:
                    final_control = np.zeros(5)

            if mission.is_arm_phase:
                robot.set_base_velocity(0.0, 0.0)
            else:
                try:
                    robot.apply_control(final_control)
                except Exception:
                    robot.set_base_velocity(0.0, 0.0)

            try:
                power = energy.update(final_control, sim_dt, state["arm_q"])
            except Exception:
                pass

            try:
                p.stepSimulation(physicsClientId=client)
            except Exception:
                break
            env.update_dynamic_obstacles(sim_time)
            sim_time += sim_dt

            if dashboard is not None and step % ctrl_every == 0:
                try:
                    dashboard.update(
                        state, power, soc, last_nmpc_solve_time,
                        augmented_obstacles, goal_xy,
                        predicted_traj=last_predicted_traj,
                        t=sim_time, energy_mgr=energy,
                    )
                except Exception:
                    pass

            if mission.is_done(state):
                break

    except KeyboardInterrupt:
        pass
    except Exception as e:
        failure = failure or f"error:{e}"

    try:
        boxes = mission.box_idx
    except Exception:
        boxes = 0
    delivered_all = (boxes >= mission.num_boxes)
    done = mission.is_done(state)
    success = bool(delivered_all and done and failure is None)
    kicks = watchdog.kicks

    st = nmpc.get_solve_time_stats()
    metrics = {
        "variant":        args.variant,
        "seed":           args.seed,
        "success":        int(success),
        "boxes":          boxes,
        "num_boxes":      mission.num_boxes,
        "failure":        failure or ("none" if success else "timeout"),
        "mission_time_s": round(sim_time, 2),
        "path_m":         round(path_len, 3),
        "energy_wh":      round(float(getattr(energy, "energy_consumed", 0.0)), 4),
        "regen_wh":       round(float(getattr(energy, "energy_regenerated", 0.0)), 4),
        "final_soc":      round(energy.get_soc() * 100.0, 2),
        "terminal_v":     round(float(energy.get_voltage()), 3),
        "max_cell_temp":  round(float(energy.max_cell_temp()), 2),
        "soc_spread":     round(float(energy.soc_spread()) * 100.0, 3),
        "nmpc_mean_ms":   round(float(st["mean"]), 2) if st["count"] else 0.0,
        "nmpc_p99_ms":    round(float(st.get("p99", 0.0)), 2) if st["count"] else 0.0,
        "nmpc_solves":    nmpc_solves,
        "switches":       switches,
        "astar_replans":  astar_replans,
        "recoveries":     kicks,
        "collisions":     collision.count,
        "wedge_escapes":  wedge_escapes,
        "corridor_narrow_s": round(corridor_narrow_s, 2),
        "corridor_class": corridor_class,
        "nav_planner":    args.nav_planner,
        "global_planner": args.global_planner,
    }

    try:
        p.disconnect(client)
    except Exception:
        pass
    return metrics


CSV_FIELDS = ["variant", "seed", "success", "boxes", "num_boxes", "failure",
              "mission_time_s", "path_m", "energy_wh", "regen_wh", "final_soc",
              "terminal_v", "max_cell_temp", "soc_spread", "nmpc_mean_ms",
              "nmpc_p99_ms", "nmpc_solves", "switches", "astar_replans",
              "recoveries", "collisions", "wedge_escapes", "corridor_narrow_s",
              "corridor_class", "nav_planner", "global_planner"]


def print_summary(m):
    print("\n" + "=" * 60)
    print(" Chapter-4 Warehouse Mission Summary (main20)")
    print("=" * 60)
    print(f"  Variant:               {m['variant']}")
    print(f"  Nav planner:           {m['nav_planner']}  (global: {m['global_planner']})")
    print(f"  Boxes delivered:       {m['boxes']} / {m['num_boxes']}")
    print(f"  Total sim time:        {m['mission_time_s']:6.1f} s")
    print(f"  Path length:           {m['path_m']:6.2f} m")
    print(f"  Energy consumed:       {m['energy_wh']:6.3f} Wh")
    print(f"  Energy regenerated:    {m['regen_wh']:6.4f} Wh")
    print(f"  Final SoC:             {m['final_soc']:5.1f} %")
    print(f"  Terminal voltage:      {m['terminal_v']:6.2f} V")
    print(f"  Max cell temp:         {m['max_cell_temp']:5.2f} C")
    print(f"  SoC spread:            {m['soc_spread']:5.2f} %")
    print(f"  NMPC solve time:       mean={m['nmpc_mean_ms']:.1f}ms p99={m['nmpc_p99_ms']:.1f}ms n={m['nmpc_solves']}")
    print(f"  Planner switches:      {m['switches']}")
    print(f"  A* replans:            {m['astar_replans']}")
    print(f"  Recovery kicks:        {m['recoveries']}")
    print(f"  Wedge corridor escapes:{m['wedge_escapes']}")
    print(f"  Narrow corridor time:  {m['corridor_narrow_s']:5.1f} s ({m['corridor_class']})")
    print(f"  Collisions:            {m['collisions']}")
    print(f"  Failure mode:          {m['failure']}")
    print(f"  SUCCESS:               {'YES' if m['success'] else 'NO'}")
    print("=" * 60)


def run_batch(args):
    import re
    import subprocess
    from concurrent.futures import ThreadPoolExecutor

    fields = CSV_FIELDS
    out_path = (args.batch_out if os.path.isabs(args.batch_out)
                else os.path.join(_HERE, args.batch_out))

    try:
        import psutil
        phys = psutil.cpu_count(logical=False) or ((os.cpu_count() or 4) // 2)
    except Exception:
        phys = max(1, (os.cpu_count() or 4) // 2)
    safe_workers = max(1, phys - 2)
    if args.workers > safe_workers:
        print(f"[batch] clamping workers {args.workers} -> {safe_workers} "
              f"({phys} physical cores; IPOPT solves are CPU-bound, "
              f"oversubscription causes timeouts). Override is intentional only.",
              flush=True)
        args.workers = safe_workers

    def one(seed):
        cmd = [sys.executable, os.path.abspath(__file__), "--no-gui",
               "--variant", args.variant, "--seed", str(seed),
               "--initial-soc", str(args.initial_soc),
               "--num-boxes", str(args.num_boxes),
               "--dyn-count", str(args.dyn_count),
               "--max-steps", str(args.max_steps),
               "--nmpc-horizon", str(args.nmpc_horizon),
               "--nmpc-max-iter", str(args.nmpc_max_iter),
               "--nav-planner", str(args.nav_planner),
               "--global-planner", str(args.global_planner),
               "--mppi-samples", str(args.mppi_samples),
               "--mppi-horizon", str(args.mppi_horizon),
               "--mppi-lambda", str(args.mppi_lambda),
               "--config", str(args.config), "--emit-csv-row"]
        if args.nmpc_jit:
            cmd.append("--nmpc-jit")
        if getattr(args, "dynamic", False):
            cmd.append("--dynamic")
        env = dict(os.environ, PYTHONHASHSEED=str(seed), OMP_NUM_THREADS="1",
                   OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   NUMEXPR_NUM_THREADS="1")
        try:
            r = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=args.timeout, cwd=_HERE, env=env)
            out = (r.stdout or "") + "\n" + (r.stderr or "")
        except subprocess.TimeoutExpired:
            out = ""
        m = re.search(r"CSVROW\|(.*)", out)
        if not m:
            return {**{k: 0 for k in fields}, "variant": args.variant,
                    "seed": seed, "failure": "timeout", "success": 0,
                    "num_boxes": args.num_boxes,
                    "nav_planner": args.nav_planner,
                    "global_planner": args.global_planner}
        vals = m.group(1).split("|")
        row = dict(zip(fields, vals))
        for k in ("success", "boxes", "num_boxes", "nmpc_solves", "switches",
                  "astar_replans", "recoveries", "collisions", "wedge_escapes"):
            row[k] = int(float(row[k])) if row[k] else 0
        for k in ("mission_time_s", "path_m", "energy_wh", "regen_wh",
                  "final_soc", "terminal_v", "max_cell_temp", "soc_spread",
                  "nmpc_mean_ms", "nmpc_p99_ms", "corridor_narrow_s"):
            row[k] = float(row[k]) if row[k] else 0.0
        print(f"[batch] {args.nav_planner:9s} seed={seed} success={row['success']} "
              f"boxes={row['boxes']}/{row['num_boxes']} E={row['energy_wh']:.1f}Wh "
              f"SoC={row['final_soc']:.0f}% t={row['mission_time_s']:.0f}s "
              f"fail={row['failure']}", flush=True)
        return row

    seeds = [args.seed_base + k for k in range(args.batch)]
    if args.workers <= 1:
        rows = [one(s) for s in seeds]
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            rows = list(pool.map(one, seeds))

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    ok = [r for r in rows if int(r["success"]) == 1]
    n = len(rows)

    def mean(key, src=ok):
        vals = [float(r[key]) for r in src]
        return sum(vals) / len(vals) if vals else 0.0

    print("\n" + "=" * 64)
    print(f" main20 batch: variant={args.variant} nav={args.nav_planner} "
          f"global={args.global_planner}  {len(ok)}/{n} success "
          f"({100 * len(ok) / max(n, 1):.1f}%)")
    if ok:
        print(f"  energy   {mean('energy_wh'):.2f} Wh   regen {mean('regen_wh'):.3f} Wh")
        print(f"  final SoC {mean('final_soc'):.1f}%   term V {mean('terminal_v'):.2f}")
        print(f"  max cell T {mean('max_cell_temp'):.2f} C   SoC spread {mean('soc_spread'):.2f}%")
        print(f"  mission   {mean('mission_time_s'):.1f} s   path {mean('path_m'):.2f} m")
        print(f"  NMPC mean {mean('nmpc_mean_ms'):.1f} ms   switches {mean('switches'):.1f}  "
              f"replans {mean('astar_replans'):.1f}  recoveries {mean('recoveries'):.2f}")
    from collections import Counter
    fc = Counter(r["failure"] for r in rows if int(r["success"]) == 0)
    if fc:
        print(f"  failures: {dict(fc)}")
    print(f"  CSV: {out_path}")
    print("=" * 64)
    return 0


def parse_args():
    ap = argparse.ArgumentParser(description="main20 multi-method navigation benchmark")
    ap.add_argument("--gui", action="store_true", default=True)
    ap.add_argument("--no-gui", dest="gui", action="store_false")
    ap.add_argument("--config", type=str, default=os.path.join(_ROOT, "config.yaml"))
    ap.add_argument("--max-steps", type=int, default=24000)
    ap.add_argument("--no-dashboard", action="store_true", default=False)
    ap.add_argument("--initial-soc", type=float, default=0.70)
    ap.add_argument("--num-boxes", type=int, default=3)
    ap.add_argument("--dyn-count", type=int, default=2)
    ap.add_argument("--dynamic", action="store_true", default=False,
                    help="Activate the dynamic-obstacle layer (sinusoidal sweep).")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--variant", type=str, default="full",
                    choices=["full", "pack_level", "no_energy", "no_regen", "speed_only"])
    ap.add_argument("--use-camera", action="store_true", default=True)
    ap.add_argument("--no-camera", dest="use_camera", action="store_false")
    # --- new navigation method selectors ---
    ap.add_argument("--nav-planner", type=str, default="baseline",
                    choices=["baseline", "mppi", "dwa", "nmpc"],
                    help="Local navigation planner. baseline = main12 nav "
                         "(pursuit+avoider); mppi = sampling MPC; dwa = dynamic "
                         "window; nmpc = NMPC-driven base nav.")
    ap.add_argument("--global-planner", type=str, default="astar",
                    choices=["astar", "hybrid"],
                    help="Global planner: astar (grid) or hybrid (kinodynamic Hybrid-A*).")
    ap.add_argument("--mppi-samples", type=int, default=600,
                    help="MPPI rollout count K.")
    ap.add_argument("--mppi-horizon", type=int, default=20,
                    help="MPPI rollout horizon H (steps of nmpc dt).")
    ap.add_argument("--mppi-lambda", type=float, default=0.30,
                    help="MPPI temperature lambda (lower = greedier).")
    # --- batch ---
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--batch-out", type=str, default="main20_batch.csv")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seed-base", type=int, default=1000)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--nmpc-horizon", type=int, default=0,
                    help="Override NMPC prediction horizon N (0 = default 8).")
    ap.add_argument("--nmpc-max-iter", type=int, default=30,
                    help="IPOPT max iterations (default 30, 0 = config value).")
    ap.add_argument("--nmpc-jit", action="store_true", default=False,
                    help="Compile the NMPC functions to C via CasADi JIT.")
    ap.add_argument("--reset-path-memory", action="store_true", default=False,
                    help="Reset the persistent path/corridor memory file before running.")
    ap.add_argument("--emit-csv-row", action="store_true", default=False,
                    help="Print a CSVROW|... line for the batch parent to parse.")
    return ap.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    if args.batch and args.batch > 0:
        return run_batch(args)

    m = run_mission(args)
    if args.emit_csv_row:
        print("CSVROW|" + "|".join(str(m[k]) for k in CSV_FIELDS), flush=True)
    print_summary(m)
    return 0


if __name__ == "__main__":
    sys.exit(main())
