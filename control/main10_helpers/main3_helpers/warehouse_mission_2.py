"""Warehouse mission state machine for main3.py.

End-to-end mission:
   init -> nav_to_pick(i) -> pre_grasp -> descend -> grasp -> lift
        -> nav_to_place(i) -> pre_place -> place -> retract
        -> (next box or return_home) -> done

Spawns its own world content via SimEnvironment (boxes, storage zone, static
& dynamic obstacles). No edits to scenarios.py or environment.py.
"""

import math
import numpy as np
import pybullet as p


NAV_PHASES = ("nav_to_pick", "nav_to_place", "return_home")
ARM_PHASES = ("pre_grasp", "descend", "grasp", "lift",
              "pre_place", "place", "retract")
SCENARIO_DRIVES_BASE = ()  # mission owns base; no sub-controller takes over


class WarehouseMission:
    def __init__(self, config, env, robot, energy_mgr,
                 initial_soc=0.8, num_boxes=3,
                 num_dynamic_obs=0, num_static_obs=4,
                 randomize=False, start=(0.0, 0.0, 0.0)):
        self.cfg = config
        self.env = env
        self.robot = robot
        self.energy = energy_mgr
        self.num_boxes = max(1, min(3, int(num_boxes)))
        self.start = np.array([start[0], start[1]])
        self.home_pos = np.array([start[0], start[1]])

        # ---- pick / place spots --------------------------------------
        if randomize:
            self.pick_pos = self._random_pos(exclusions=[(self.start[0], self.start[1], 1.5)])
            self.place_pos = self._random_pos(
                exclusions=[(self.start[0], self.start[1], 1.5),
                            (self.pick_pos[0], self.pick_pos[1], 1.8)])
        else:
            self.pick_pos = np.array([2.5, 0.5])
            self.place_pos = np.array([-2.0, 1.5])

        # ---- spawn boxes + storage -----------------------------------
        self.boxes = env.add_pick_objects(self.pick_pos[0], self.pick_pos[1], 0.25)
        self.boxes = self.boxes[:self.num_boxes]
        self.box_colors = ["yellow", "orange", "blue"][:self.num_boxes]
        env.add_storage_zone(self.place_pos[0], self.place_pos[1])
        env.add_storage_platform(self.place_pos[0], self.place_pos[1])
        env.add_target(self.pick_pos[0], self.pick_pos[1], z=0.01)

        self.place_offsets = [
            np.array([-0.06, -0.06]),
            np.array([0.06, -0.06]),
            np.array([0.0, 0.06]),
        ]

        # ---- static + dynamic obstacles ------------------------------
        # Real-world spawn-safety: keep obstacles >=1.5m from robot start
        # so the robot has room to accelerate and turn before encountering them.
        # Also keep them away from pick/place spots so the arm has clearance.
        exclusions = [
            (self.start[0], self.start[1], 1.5),     # was 1.0 -> 1.5m clearance
            (self.pick_pos[0], self.pick_pos[1], 1.2),
            (self.place_pos[0], self.place_pos[1], 1.2),
        ]
        # Mid-line obstacle for interesting geometry (only if pick/place are far apart)
        mid = ((self.pick_pos[0] + self.place_pos[0]) / 2.0,
               (self.pick_pos[1] + self.place_pos[1]) / 2.0)
        pick_place_dist = math.hypot(self.pick_pos[0] - self.place_pos[0],
                                     self.pick_pos[1] - self.place_pos[1])
        # Only place the mid obstacle if it would NOT block the robot's start->pick path.
        # Gated on num_static_obs so a subclass that owns its own obstacle field
        # (e.g. main10's CorridorWarehouseMission, called with num_static_obs=0)
        # gets a clean slate -- otherwise the base + subclass obstacles stack and
        # clutter the pick/place corridors.
        mid_clear_from_start = math.hypot(mid[0] - self.start[0],
                                          mid[1] - self.start[1]) > 1.5
        if num_static_obs > 0 and pick_place_dist > 2.0 and mid_clear_from_start:
            env.add_obstacle(mid[0], mid[1], radius=0.22, height=0.5)
            exclusions.append((mid[0], mid[1], 0.8))
        if num_static_obs > 0:
            # FIXED TEST OBSTACLES
            env.add_obstacle(1.5, -0.5, radius=0.22, height=0.5)   # near pick path
            env.add_obstacle(-1.0,  1.0, radius=0.22, height=0.5)  # near place path

        # Dynamic obstacles: gentler motion + spawn FAR from start.
        # Real-world parallel: pedestrians and AGVs move ~0.5-1.0 m/s, not 2 m/s.
        # New params: amp 0.3-0.6m (was 0.5-1.1), freq 0.08-0.15Hz (was 0.15-0.30)
        # -> peak sweep speed = 2*pi*freq*amp goes from 1.0-2.1 m/s down to 0.15-0.57 m/s
        #for i in range(max(0, num_dynamic_obs)):
            # Force dynamics to spawn >=2.5m from start so robot can plan around them
         #   dyn_exclusions = exclusions + [(self.start[0], self.start[1], 2.5)]
          #  ox, oy = self._random_pos(exclusions=dyn_exclusions)
           #env.add_dynamic_obstacle(
            #    ox, oy,
             #   amplitude=float(np.random.uniform(0.3, 0.6)),
              #  frequency=float(np.random.uniform(0.08, 0.15)),
               # direction=direction)
            #exclusions.append((ox, oy, 1.2))

        # ---- initial battery state -----------------------------------
        if hasattr(energy_mgr, "set_initial_soc"):
            energy_mgr.set_initial_soc(float(initial_soc))
        else:
            energy_mgr.soc = float(max(0.0, min(1.0, initial_soc)))

        # ---- mission state -------------------------------------------
        self.pick_idx = 0         # which box we are currently picking
        self.box_idx = 0          # boxes fully delivered (used by main/dashboard)
        self.phase = "nav_to_pick"
        self.phase_timer = 0
        self.grip_constraint = None
        self.goal = np.array([self.pick_pos[0], self.pick_pos[1], 0.0,
                              0.0, 0.0, 0.0])
        self.robot.open_gripper()
        self.robot.tuck_arm()
        self._force_return = False

        # ---- pick spot for current target ----------------------------
        self._refresh_pick_target()

        print(f"[Mission] Warehouse: {self.num_boxes} boxes  pick=({self.pick_pos[0]:.2f},{self.pick_pos[1]:.2f}) "
              f"place=({self.place_pos[0]:.2f},{self.place_pos[1]:.2f})  initial SoC={initial_soc:.0%}")

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _random_pos(exclusions=None, x_range=(-3.0, 3.0), y_range=(-3.0, 3.0),
                    attempts=200):
        ex = exclusions or []
        for _ in range(attempts):
            x = np.random.uniform(*x_range)
            y = np.random.uniform(*y_range)
            ok = True
            for cx, cy, cr in ex:
                if math.hypot(x - cx, y - cy) < cr:
                    ok = False
                    break
            if ok:
                return np.array([x, y])
        return np.array([x, y])

    def _current_box(self):
        """Box currently being PICKED at the pick zone."""
        if self.pick_idx >= len(self.boxes):
            return None
        return self.boxes[self.pick_idx]

    def _box_pos(self):
        b = self._current_box()
        if b is None:
            return None
        try:
            return self.env.get_object_position(b)
        except Exception:
            return np.array([self.pick_pos[0], self.pick_pos[1], 0.05])

    def _place_target(self):
        """Storage spot for the box currently being placed."""
        off = self.place_offsets[self.box_idx % len(self.place_offsets)]
        return np.array([self.place_pos[0] + off[0],
                         self.place_pos[1] + off[1],
                         0.27])

    def _refresh_pick_target(self):
        bp = self._box_pos()
        if bp is not None:
            self.goal[:2] = bp[:2]

    def _set_phase(self, new_phase):
        idx = max(0, min(self.box_idx, self.num_boxes - 1))
        color = self.box_colors[idx] if idx < len(self.box_colors) else "?"
        print(f"[Mission] box {idx + 1}/{self.num_boxes} ({color}) "
              f"phase: {self.phase} -> {new_phase}")
        self.phase = new_phase
        self.phase_timer = 0

    def force_return_home(self):
        """External signal (energy_policy) to abandon current pick and go home."""
        if self.phase == "return_home" or self.phase == "done":
            return
        self._force_return = True
        if self.phase in ("lift", "nav_to_place", "pre_place", "place"):
            self.robot.open_gripper()
            self._release_constraint()
        self._set_phase("return_home")
        self.goal = np.array([self.home_pos[0], self.home_pos[1], 0.0,
                               0.0, 0.0, 0.0])

    # ------------------------------------------------------------------ FSM
    def update_goal(self, state):
        self.phase_timer += 1
        ph = self.phase

        if ph == "nav_to_pick":
            bp = self._box_pos()
            if bp is None:
                self._set_phase("return_home")
                self.goal[:2] = self.home_pos
                return self.goal
            self.goal[:2] = bp[:2]
            self.robot.tuck_arm()
            d = math.hypot(state["x"] - bp[0], state["y"] - bp[1])
            if d < 0.55:
                self._set_phase("pre_grasp")
                self.robot.open_gripper()

        elif ph == "pre_grasp":
            bp = self._box_pos()
            if bp is None:
                self._set_phase("nav_to_pick")
                self._refresh_pick_target()
                return self.goal
            target = [bp[0], bp[1], bp[2] + 0.12]
            self.robot.move_arm_smooth(target, max_force=60, max_velocity=0.8)
            self.robot.open_gripper()
            self.goal[:2] = [state["x"], state["y"]]
            ee = self.robot.get_end_effector_pos()
            if np.linalg.norm(ee - np.array(target)) < 0.05 or self.phase_timer > 100:
                self._set_phase("descend")

        elif ph == "descend":
            bp = self._box_pos()
            if bp is None:
                self._set_phase("nav_to_pick")
                self._refresh_pick_target()
                return self.goal
            target = [bp[0], bp[1], bp[2] + 0.015]
            self.robot.move_arm_to(target, max_force=60, max_velocity=0.4)
            self.robot.open_gripper()
            self.goal[:2] = [state["x"], state["y"]]
            ee = self.robot.get_end_effector_pos()
            if np.linalg.norm(ee - np.array(target)) < 0.05 or self.phase_timer > 100:
                self._set_phase("grasp")

        elif ph == "grasp":
            bp = self._box_pos()
            if bp is None:
                self._set_phase("nav_to_pick")
                self._refresh_pick_target()
                return self.goal
            self.robot.move_arm_to([bp[0], bp[1], bp[2] + 0.015],
                                   max_force=60, max_velocity=0.2)
            self.robot.close_gripper()
            self.goal[:2] = [state["x"], state["y"]]
            has = False
            try:
                has = self.robot.is_gripping(self._current_box())
            except Exception:
                has = False
            if has or self.phase_timer > 80:
                if not has:
                    self._create_grip_constraint()
                self._set_phase("lift")

        elif ph == "lift":
            bp = self._box_pos()
            if bp is None:
                self._set_phase("nav_to_place")
                self.goal[:2] = self.place_pos
                return self.goal
            target = [bp[0], bp[1], bp[2] + 0.18]
            self.robot.move_arm_to(target, max_force=60, max_velocity=0.5)
            self.robot.close_gripper()
            self.goal[:2] = [state["x"], state["y"]]
            ee = self.robot.get_end_effector_pos()
            if ee[2] > 0.35 or self.phase_timer > 100:
                self._set_phase("nav_to_place")
                self.goal[:2] = self.place_pos

        elif ph == "nav_to_place":
            tgt = self._place_target()
            self.goal[:2] = tgt[:2]
            self.robot.tuck_arm()
            self.robot.close_gripper()
            d = math.hypot(state["x"] - tgt[0], state["y"] - tgt[1])
            if d < 0.55:
                self._set_phase("pre_place")

        elif ph == "pre_place":
            tgt = self._place_target()
            target = [tgt[0], tgt[1], tgt[2] + 0.15]
            self.robot.move_arm_to(target, max_force=50, max_velocity=0.8)
            self.robot.close_gripper()
            self.goal[:2] = [state["x"], state["y"]]
            ee = self.robot.get_end_effector_pos()
            if np.linalg.norm(ee - np.array(target)) < 0.06 or self.phase_timer > 100:
                self._set_phase("place")

        elif ph == "place":
            tgt = self._place_target()
            target = [tgt[0], tgt[1], tgt[2] + 0.05]
            self.robot.move_arm_to(target, max_force=50, max_velocity=0.5)
            self.goal[:2] = [state["x"], state["y"]]
            ee = self.robot.get_end_effector_pos()
            if np.linalg.norm(ee - np.array(target)) < 0.06 or self.phase_timer > 80:
                self.robot.open_gripper()
                self._release_constraint()
                self.box_idx += 1
                print(f"[Mission] box {self.box_idx} placed at storage")
                self._set_phase("retract")

        elif ph == "retract":
            self.robot.tuck_arm()
            self.robot.open_gripper()
            self.goal[:2] = [state["x"], state["y"]]
            err = float(np.linalg.norm(state["arm_q"] - np.array(self.robot.arm_tuck)))
            if err < 0.3 or self.phase_timer > 80:
                self.pick_idx += 1
                if self.pick_idx < self.num_boxes:
                    self._set_phase("nav_to_pick")
                    self._refresh_pick_target()
                else:
                    self._set_phase("return_home")
                    self.goal[:2] = self.home_pos

        elif ph == "return_home":
            self.robot.tuck_arm()
            self.goal[:2] = self.home_pos
            d = math.hypot(state["x"] - self.home_pos[0], state["y"] - self.home_pos[1])
            if d < 0.5:
                self._set_phase("done")

        elif ph == "done":
            pass

        return self.goal

    def _advance_box(self):
        """Backwards-compatible shim (used by main6 recovery skip): drop the
        current pick target and move on without picking it."""
        self.pick_idx += 1
        self.box_idx = max(self.box_idx, 0)
        if self.pick_idx >= self.num_boxes:
            self._set_phase("return_home")
            self.goal[:2] = self.home_pos
        else:
            self._set_phase("nav_to_pick")
            self._refresh_pick_target()
            self.robot.open_gripper()
            self.robot.tuck_arm()

    # ------------------------------------------------------------------ grip helpers
    def _create_grip_constraint(self):
        try:
            self.grip_constraint = p.createConstraint(
                self.robot.robot_id, self.robot.ee_link_idx,
                self._current_box(), -1,
                p.JOINT_FIXED, [0, 0, 0], [0, 0, 0.03], [0, 0, 0],
                physicsClientId=self.env.client)
        except Exception:
            self.grip_constraint = None

    def _release_constraint(self):
        if self.grip_constraint is not None:
            try:
                p.removeConstraint(self.grip_constraint,
                                   physicsClientId=self.env.client)
            except Exception:
                pass
            self.grip_constraint = None

    # ------------------------------------------------------------------ queries
    @property
    def is_arm_phase(self):
        return self.phase in ARM_PHASES

    @property
    def is_nav_phase(self):
        return self.phase in NAV_PHASES

    def is_done(self, state=None):
        return self.phase == "done"

    def dist_home(self, state):
        return math.hypot(state["x"] - self.home_pos[0],
                          state["y"] - self.home_pos[1])
