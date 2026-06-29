"""BatchWarehouseMission — pick all boxes, carry them on the robot's back,
deliver them together in a single trip.

Mission flow:
  init
    -> nav_to_pick[i]      drive to box i
    -> pre_grasp[i]        arm above box
    -> descend[i]
    -> grasp[i]
    -> lift[i]
    -> load_on_back[i]     constrain box to a tray slot on robot rear
    -> (next box or)
    -> nav_to_place        single drive to storage zone with all boxes loaded
    -> unload[i]           detach each box and lower to platform offset
    -> retract
    -> return_home
    -> done

Deterministic-obstacle placement: at construction time we place exactly
the obstacles needed on each path segment, so the demo always exercises
navigation. Difficulty levels:
  easy   = 1 obstacle on start->pick segment only
  normal = 1 obstacle on each of start->pick, pick->place, place->home
  hard   = 2 obstacles per segment + 1 narrow corridor on the pick->place leg
"""

import math
import numpy as np
import pybullet as p


NAV_PHASES = ("nav_to_pick", "nav_to_place", "return_home")
ARM_PHASES = ("pre_grasp", "descend", "grasp", "lift", "load_on_back",
              "unload", "retract")

# Tray slots on the robot rear deck, in robot-local frame (x, y, z).
# Robot body height ~0.10m + caster + body ~0.18m. Tray sits at z=0.22m.
# Three slots in a row across the back (negative x = behind robot).
TRAY_SLOTS_LOCAL = [
    np.array([-0.15,  0.10, 0.22]),   # rear-left
    np.array([-0.15,  0.00, 0.22]),   # rear-center
    np.array([-0.15, -0.10, 0.22]),   # rear-right
]


class BatchWarehouseMission:
    def __init__(self, config, env, robot, energy_mgr,
                 initial_soc=0.7, num_boxes=3,
                 num_dynamic_obs=0, difficulty="normal",
                 randomize=False, start=(0.0, 0.0, 0.0)):
        self.cfg = config
        self.env = env
        self.robot = robot
        self.energy = energy_mgr
        self.num_boxes = max(1, min(3, int(num_boxes)))
        self.start = np.array([start[0], start[1]])
        self.home_pos = np.array([start[0], start[1]])
        self.difficulty = difficulty

        # ---- pick / place spots (fixed positions for deterministic demo)
        if randomize:
            self.pick_pos = self._random_pos(exclusions=[(self.start[0], self.start[1], 1.5)])
            self.place_pos = self._random_pos(
                exclusions=[(self.start[0], self.start[1], 1.5),
                            (self.pick_pos[0], self.pick_pos[1], 2.0)])
        else:
            self.pick_pos = np.array([2.5, 0.5])
            self.place_pos = np.array([-2.0, 1.5])

        # ---- spawn boxes
        self.boxes = env.add_pick_objects(self.pick_pos[0], self.pick_pos[1], 0.25)
        self.boxes = self.boxes[:self.num_boxes]
        self.box_colors = ["yellow", "orange", "blue"][:self.num_boxes]
        env.add_storage_zone(self.place_pos[0], self.place_pos[1])
        env.add_storage_platform(self.place_pos[0], self.place_pos[1])
        env.add_target(self.pick_pos[0], self.pick_pos[1], z=0.01)

        # Disable collisions between robot and pick boxes from the start. The
        # boxes are meant to be picked, and the IR raycasts treat them as
        # obstacles which causes the robot to brake forever inside the pick
        # zone. The gripper still works via createConstraint regardless.
        try:
            num_links = p.getNumJoints(self.robot.robot_id,
                                       physicsClientId=self.env.client)
            for box_id in self.boxes:
                for link_i in range(-1, num_links):
                    p.setCollisionFilterPair(
                        bodyUniqueIdA=self.robot.robot_id,
                        bodyUniqueIdB=box_id,
                        linkIndexA=link_i, linkIndexB=-1,
                        enableCollision=0,
                        physicsClientId=self.env.client)
        except Exception:
            pass

        # ---- DETERMINISTIC obstacles on path segments
        self._spawn_path_obstacles(difficulty)

        # ---- dynamic obstacles, kept gentle and FAR from path
        for i in range(max(0, num_dynamic_obs)):
            ox, oy = self._random_pos(
                exclusions=[
                    (self.start[0], self.start[1], 2.5),
                    (self.pick_pos[0], self.pick_pos[1], 1.5),
                    (self.place_pos[0], self.place_pos[1], 1.5),
                ],
                x_range=(-3.5, 3.5), y_range=(-3.5, 3.5))
            direction = "x" if i % 2 == 0 else "y"
            env.add_dynamic_obstacle(
                ox, oy,
                amplitude=float(np.random.uniform(0.25, 0.45)),
                frequency=float(np.random.uniform(0.08, 0.13)),
                direction=direction)

        # ---- battery init
        if hasattr(energy_mgr, "set_initial_soc"):
            energy_mgr.set_initial_soc(float(initial_soc))
        else:
            energy_mgr.soc = float(max(0.0, min(1.0, initial_soc)))

        # ---- mission state
        self.box_idx = 0                 # currently being picked
        self.unload_idx = 0              # currently being unloaded
        self.phase = "nav_to_pick"
        self.phase_timer = 0
        # Kinematic-carry tracking. Each tick we *teleport* every loaded box
        # to (robot_pose * local_offset) and zero its velocity, so the loaded
        # mass does not couple into the chassis through a JOINT_FIXED constraint
        # (which we tried first and which paralyzed the wheels).
        self.loaded_boxes = []           # box_id list, in load order
        self.loaded_offsets = []         # parallel list of np.array([x,y,z]) in robot frame
        self.grip_constraint = None
        self.goal = np.array([self.pick_pos[0], self.pick_pos[1], 0.0,
                              0.0, 0.0, 0.0])
        self.robot.open_gripper()
        self.robot.tuck_arm()
        self._force_return = False
        self._refresh_pick_target()

        print(f"[BatchMission] {self.num_boxes} boxes  "
              f"pick=({self.pick_pos[0]:.2f},{self.pick_pos[1]:.2f}) "
              f"place=({self.place_pos[0]:.2f},{self.place_pos[1]:.2f}) "
              f"difficulty={difficulty}  SoC0={initial_soc:.0%}")

    # ==================================================================
    # Deterministic obstacles ON the path segments
    # ==================================================================
    def _segment_perp_at_t(self, A, B, t, perp_offset):
        """Return a point at parameter t along AB, offset perpendicularly."""
        cx = A[0] + t * (B[0] - A[0])
        cy = A[1] + t * (B[1] - A[1])
        dx, dy = B[0] - A[0], B[1] - A[1]
        d = math.hypot(dx, dy)
        if d < 1e-6:
            return cx, cy
        px, py = -dy / d, dx / d
        return cx + px * perp_offset, cy + py * perp_offset

    def _segment_midpoint(self, A, B, perp_offset=0.0):
        """Midpoint of segment AB, optionally offset perpendicular to it."""
        mx = (A[0] + B[0]) / 2.0
        my = (A[1] + B[1]) / 2.0
        if perp_offset != 0.0:
            dx, dy = B[0] - A[0], B[1] - A[1]
            d = math.hypot(dx, dy)
            if d > 1e-6:
                # perpendicular unit vector
                px, py = -dy / d, dx / d
                mx += px * perp_offset
                my += py * perp_offset
        return mx, my

    def _spawn_path_obstacles(self, difficulty):
        """Place obstacles ON the three path segments so the demo always
        has something to navigate around.

        Layout (X-Y plane, default fixed positions):
          start    (0.0, 0.0)
          pick     (2.5, 0.5)    <- east, slightly north
          place    (-2.0, 1.5)   <- west, north
          home     start

        Obstacles per segment (difficulty = normal):
          start -> pick   : 1 obstacle near (1.25, 0.25), offset 0.4m perpendicular
          pick -> place   : 1 obstacle near midpoint (0.25, 1.0)
          place -> home   : 1 obstacle near (-1.0, 0.75)
        """
        start = (self.start[0], self.start[1])
        pick = (self.pick_pos[0], self.pick_pos[1])
        place = (self.place_pos[0], self.place_pos[1])
        home = start

        seg_pick = (start, pick)
        seg_place = (pick, place)
        seg_home = (place, home)

        n_per_seg = {"easy": (1, 0, 0), "normal": (1, 1, 1), "hard": (2, 2, 2)}
        per = n_per_seg.get(difficulty, n_per_seg["normal"])

        # Clearance budget: robot_radius 0.30 + obs_radius 0.22 + safety 0.20 = 0.72m
        # We place obstacles ~0.75m perpendicular to the path line so the robot
        # has room to navigate around them while the obstacles are still
        # clearly ON the route (not far-field decoration).
        PERP = 0.75
        OBS_R = 0.22
        MIN_DIST_FROM_START = 1.2   # never place obstacle this close to start

        def safe_offset(A, B, t, prefer_sign):
            """Pick a perpendicular offset side that keeps the obstacle far
            from the robot start. If `prefer_sign` * PERP lands too close to
            the start point, flip the sign.
            """
            for sign in (prefer_sign, -prefer_sign):
                x, y = self._segment_perp_at_t(A, B, t, sign * PERP)
                if math.hypot(x - start[0], y - start[1]) >= MIN_DIST_FROM_START:
                    return x, y
            # Fallback: bigger offset on the prefer side
            return self._segment_perp_at_t(A, B, t, prefer_sign * (PERP + 0.4))

        # Segment 1: start -> pick
        for k in range(per[0]):
            prefer = +1 if k == 0 else -1
            t = 0.5 if per[0] == 1 else (0.35 + k * 0.30)
            mx, my = safe_offset(seg_pick[0], seg_pick[1], t, prefer)
            self.env.add_obstacle(mx, my, radius=OBS_R, height=0.5)
            print(f"[Spawn] start->pick obstacle #{k+1} at ({mx:+.2f},{my:+.2f})")

        # Segment 2: pick -> place
        for k in range(per[1]):
            prefer = +1 if k == 0 else -1
            t = 0.5 if per[1] == 1 else (0.30 + k * 0.40)
            mx, my = safe_offset(seg_place[0], seg_place[1], t, prefer)
            self.env.add_obstacle(mx, my, radius=OBS_R, height=0.5)
            print(f"[Spawn] pick->place obstacle #{k+1} at ({mx:+.2f},{my:+.2f})")

        # Hard mode: extra narrow corridor on pick->place (gated to keep
        # both gate posts away from the robot start).
        if difficulty == "hard" and per[1] >= 2:
            mx, my = safe_offset(seg_place[0], seg_place[1], 0.55, +1)
            mx2, my2 = safe_offset(seg_place[0], seg_place[1], 0.55, -1)
            # Force the second post to be on the opposite side of the path
            # from the first; if safe_offset flipped, push outward instead.
            if (math.copysign(1.0, mx - mx2) == math.copysign(1.0, my - my2)
                    or math.hypot(mx - mx2, my - my2) < 1.0):
                mx2, my2 = self._segment_perp_at_t(
                    seg_place[0], seg_place[1], 0.55,
                    -1.0 * (1.6 if math.hypot(mx, my) > 1.5 else 1.2))
            self.env.add_obstacle(mx, my, radius=0.18, height=0.5)
            self.env.add_obstacle(mx2, my2, radius=0.18, height=0.5)
            print(f"[Spawn] corridor (hard) at ({mx:+.2f},{my:+.2f}) | ({mx2:+.2f},{my2:+.2f})")

        # Segment 3: place -> home
        for k in range(per[2]):
            prefer = +1 if k == 0 else -1
            t = 0.5 if per[2] == 1 else (0.35 + k * 0.30)
            mx, my = safe_offset(seg_home[0], seg_home[1], t, prefer)
            self.env.add_obstacle(mx, my, radius=OBS_R, height=0.5)
            print(f"[Spawn] place->home obstacle #{k+1} at ({mx:+.2f},{my:+.2f})")

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

    # ==================================================================
    # Tray / back-carry mechanics (kinematic — no JOINT_FIXED)
    # ==================================================================
    def _attach_to_back(self, box_id, slot_idx):
        """Register `box_id` as carried at the given tray slot. We deliberately
        do NOT create a JOINT_FIXED constraint here — that approach caused the
        wheels to lock up under the added inertia. Instead we mark the box as
        kinematic-carried; `kinematic_carry_step()` (called from main4 every
        sim tick) teleports it onto the back.
        """
        local_offset = TRAY_SLOTS_LOCAL[slot_idx % len(TRAY_SLOTS_LOCAL)]
        try:
            # Make the box dynamics-inert while it rides on the back. We zero
            # its linear/angular damping so the teleport is the only thing
            # moving it, and disable collisions between this box and the robot
            # so it doesn't push the chassis around when we drop it on.
            try:
                p.changeDynamics(box_id, -1,
                                 mass=0.0,                # kinematic
                                 linearDamping=0.0,
                                 angularDamping=0.0,
                                 physicsClientId=self.env.client)
            except Exception:
                pass
            # Disable box<->robot collision so the teleport never pushes the chassis
            try:
                num_links = p.getNumJoints(self.robot.robot_id,
                                           physicsClientId=self.env.client)
                for link_i in range(-1, num_links):
                    p.setCollisionFilterPair(
                        bodyUniqueIdA=self.robot.robot_id,
                        bodyUniqueIdB=box_id,
                        linkIndexA=link_i,
                        linkIndexB=-1,
                        enableCollision=0,
                        physicsClientId=self.env.client)
            except Exception:
                pass
            self.loaded_boxes.append(box_id)
            self.loaded_offsets.append(np.asarray(local_offset, dtype=float))
            print(f"[Tray] box loaded on slot {slot_idx}: id={box_id}  "
                  f"local_offset=({local_offset[0]:.2f},{local_offset[1]:.2f},{local_offset[2]:.2f})")
            return True
        except Exception as e:
            print(f"[Tray] attach failed: {e}")
            return False

    def _detach_from_back(self, slot_idx):
        if slot_idx >= len(self.loaded_boxes):
            return None
        box_id = self.loaded_boxes[slot_idx]
        # Restore some mass so it sits on the platform under gravity.
        try:
            p.changeDynamics(box_id, -1, mass=0.10,
                             physicsClientId=self.env.client)
        except Exception:
            pass
        # Re-enable collisions with the robot so the gripper/arm interacts
        # normally again if needed.
        try:
            num_links = p.getNumJoints(self.robot.robot_id,
                                       physicsClientId=self.env.client)
            for link_i in range(-1, num_links):
                p.setCollisionFilterPair(
                    bodyUniqueIdA=self.robot.robot_id,
                    bodyUniqueIdB=box_id,
                    linkIndexA=link_i,
                    linkIndexB=-1,
                    enableCollision=1,
                    physicsClientId=self.env.client)
        except Exception:
            pass
        return box_id

    def kinematic_carry_step(self, state):
        """Teleport every box currently registered as carried to its slot on
        the robot rear deck. MUST be called every sim tick from main4 (after
        the physics step) so the boxes track the chassis.
        """
        if not self.loaded_boxes:
            return
        theta = state["theta"]
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        z_robot = 0.10  # approximate robot base height in world frame
        # Use actual base position from PyBullet to be precise even if state
        # is stale.
        try:
            base_pos, base_orn = p.getBasePositionAndOrientation(
                self.robot.robot_id, physicsClientId=self.env.client)
            bx, by, bz = base_pos
        except Exception:
            bx, by, bz = state["x"], state["y"], z_robot

        for box_id, local in zip(self.loaded_boxes, self.loaded_offsets):
            if local is None:
                continue   # already unloaded — leave it on the platform
            wx = bx + cos_t * local[0] - sin_t * local[1]
            wy = by + sin_t * local[0] + cos_t * local[1]
            wz = bz + float(local[2])
            try:
                p.resetBasePositionAndOrientation(
                    box_id, [wx, wy, wz], [0, 0, 0, 1],
                    physicsClientId=self.env.client)
                p.resetBaseVelocity(box_id, [0, 0, 0], [0, 0, 0],
                                    physicsClientId=self.env.client)
            except Exception:
                pass

    # ==================================================================
    # Helpers
    # ==================================================================
    def _current_box(self):
        if self.box_idx >= len(self.boxes):
            return None
        return self.boxes[self.box_idx]

    def _box_pos(self):
        b = self._current_box()
        if b is None:
            return None
        try:
            return self.env.get_object_position(b)
        except Exception:
            return np.array([self.pick_pos[0], self.pick_pos[1], 0.05])

    def _refresh_pick_target(self):
        bp = self._box_pos()
        if bp is not None:
            self.goal[:2] = bp[:2]

    def _set_phase(self, new_phase):
        idx = min(self.box_idx, self.num_boxes - 1)
        color = self.box_colors[idx] if 0 <= idx < len(self.box_colors) else "?"
        print(f"[BatchMission] box {idx + 1}/{self.num_boxes} ({color}) "
              f"phase: {self.phase} -> {new_phase}")
        self.phase = new_phase
        self.phase_timer = 0

    def force_return_home(self):
        if self.phase in ("return_home", "done"):
            return
        self._force_return = True
        self._set_phase("return_home")
        self.goal[:2] = self.home_pos

    def _place_target_for_loaded(self, idx):
        """Return target (x, y, z) for unloading box at index `idx`."""
        offsets = [
            np.array([-0.08, -0.08, 0.0]),
            np.array([+0.08, -0.08, 0.0]),
            np.array([+0.00, +0.08, 0.0]),
        ]
        off = offsets[idx % len(offsets)]
        return np.array([self.place_pos[0] + off[0],
                         self.place_pos[1] + off[1],
                         0.25])

    # ==================================================================
    # FSM
    # ==================================================================
    def update_goal(self, state):
        self.phase_timer += 1
        ph = self.phase

        # ---------- nav_to_pick ----------
        if ph == "nav_to_pick":
            bp = self._box_pos()
            if bp is None:
                self._set_phase("nav_to_place")
                self.goal[:2] = [self.place_pos[0], self.place_pos[1]]
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
                self._set_phase("retract")
                return self.goal
            target = [bp[0], bp[1], bp[2] + 0.12]
            self.robot.move_arm_smooth(target, max_force=60, max_velocity=0.8)
            self.robot.open_gripper()
            self.goal[:2] = [state["x"], state["y"]]
            ee = self.robot.get_end_effector_pos()
            if np.linalg.norm(ee - np.array(target)) < 0.05 or self.phase_timer > 150:
                self._set_phase("descend")

        elif ph == "descend":
            bp = self._box_pos()
            if bp is None:
                self._set_phase("retract")
                return self.goal
            target = [bp[0], bp[1], bp[2] + 0.015]
            self.robot.move_arm_to(target, max_force=60, max_velocity=0.4)
            self.robot.open_gripper()
            self.goal[:2] = [state["x"], state["y"]]
            ee = self.robot.get_end_effector_pos()
            if np.linalg.norm(ee - np.array(target)) < 0.05 or self.phase_timer > 150:
                self._set_phase("grasp")

        elif ph == "grasp":
            bp = self._box_pos()
            if bp is None:
                self._set_phase("retract")
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
                self._set_phase("load_on_back")
                return self.goal
            target = [bp[0], bp[1], bp[2] + 0.20]
            self.robot.move_arm_to(target, max_force=60, max_velocity=0.5)
            self.robot.close_gripper()
            self.goal[:2] = [state["x"], state["y"]]
            ee = self.robot.get_end_effector_pos()
            if ee[2] > 0.38 or self.phase_timer > 150:
                self._set_phase("load_on_back")

        elif ph == "load_on_back":
            # Bring the arm above the tray slot (in robot-local frame), then
            # release the gripper while creating a tray constraint at the slot.
            slot_idx = len(self.loaded_boxes)
            local = TRAY_SLOTS_LOCAL[slot_idx % len(TRAY_SLOTS_LOCAL)]
            # Compute world position of the slot
            theta = state["theta"]
            cos_t, sin_t = math.cos(theta), math.sin(theta)
            wx = state["x"] + cos_t * local[0] - sin_t * local[1]
            wy = state["y"] + sin_t * local[0] + cos_t * local[1]
            target = [wx, wy, local[2] + 0.05]
            self.robot.move_arm_to(target, max_force=60, max_velocity=0.5)
            self.robot.close_gripper()
            self.goal[:2] = [state["x"], state["y"]]
            ee = self.robot.get_end_effector_pos()
            if np.linalg.norm(ee - np.array(target)) < 0.08 or self.phase_timer > 120:
                # Release gripper-grip constraint if any
                self._release_grip_constraint()
                # Attach box to tray slot
                self._attach_to_back(self._current_box(), slot_idx)
                self.robot.open_gripper()
                # Move on
                self.box_idx += 1
                if self.box_idx < self.num_boxes:
                    self._set_phase("nav_to_pick")
                    self._refresh_pick_target()
                    self.robot.tuck_arm()
                else:
                    print(f"[BatchMission] all {self.num_boxes} boxes loaded — proceed to place")
                    self._set_phase("nav_to_place")
                    self.goal[:2] = [self.place_pos[0], self.place_pos[1]]
                    self.robot.tuck_arm()

        # ---------- nav_to_place ----------
        elif ph == "nav_to_place":
            self.goal[:2] = [self.place_pos[0], self.place_pos[1]]
            self.robot.tuck_arm()
            d = math.hypot(state["x"] - self.place_pos[0],
                           state["y"] - self.place_pos[1])
            if d < 0.55:
                self.unload_idx = 0
                self._set_phase("unload")

        elif ph == "unload":
            # Sequentially detach each box from the tray and lower it to the
            # platform via the arm.
            if self.unload_idx >= len(self.loaded_boxes):
                self._set_phase("retract")
                return self.goal
            tgt = self._place_target_for_loaded(self.unload_idx)
            self.robot.move_arm_to([tgt[0], tgt[1], tgt[2] + 0.10],
                                   max_force=60, max_velocity=0.6)
            self.goal[:2] = [state["x"], state["y"]]
            if self.phase_timer > 90:
                # We don't pop from loaded_boxes here, because kinematic_carry_step
                # iterates that list. Instead we mark the slot as unloaded by
                # nulling its offset so the carry step skips it (handled below
                # in kinematic_carry_step), and place the box on the platform.
                if self.unload_idx < len(self.loaded_boxes):
                    released = self.loaded_boxes[self.unload_idx]
                    self.loaded_offsets[self.unload_idx] = None   # sentinel: skip
                    try:
                        p.changeDynamics(released, -1, mass=0.10,
                                         physicsClientId=self.env.client)
                    except Exception:
                        pass
                    try:
                        p.resetBasePositionAndOrientation(
                            released, [tgt[0], tgt[1], tgt[2]],
                            [0, 0, 0, 1], physicsClientId=self.env.client)
                        p.resetBaseVelocity(released, [0, 0, 0], [0, 0, 0],
                                            physicsClientId=self.env.client)
                    except Exception:
                        pass
                    print(f"[Unload] box #{self.unload_idx+1} placed at "
                          f"({tgt[0]:+.2f},{tgt[1]:+.2f},{tgt[2]:.2f})")
                self.unload_idx += 1
                self.phase_timer = 0
                if self.unload_idx >= len(self.loaded_boxes):
                    self._set_phase("retract")

        elif ph == "retract":
            self.robot.tuck_arm()
            self.robot.open_gripper()
            self.goal[:2] = [state["x"], state["y"]]
            err = float(np.linalg.norm(state["arm_q"] - np.array(self.robot.arm_tuck)))
            if err < 0.3 or self.phase_timer > 100:
                self._set_phase("return_home")
                self.goal[:2] = self.home_pos

        elif ph == "return_home":
            self.robot.tuck_arm()
            self.goal[:2] = self.home_pos
            d = math.hypot(state["x"] - self.home_pos[0],
                           state["y"] - self.home_pos[1])
            if d < 0.5:
                self._set_phase("done")

        elif ph == "done":
            pass

        return self.goal

    # ------------------------------------------------------------------
    def _create_grip_constraint(self):
        try:
            self.grip_constraint = p.createConstraint(
                self.robot.robot_id, self.robot.ee_link_idx,
                self._current_box(), -1,
                p.JOINT_FIXED, [0, 0, 0], [0, 0, 0.03], [0, 0, 0],
                physicsClientId=self.env.client)
        except Exception:
            self.grip_constraint = None

    def _release_grip_constraint(self):
        if self.grip_constraint is not None:
            try:
                p.removeConstraint(self.grip_constraint,
                                   physicsClientId=self.env.client)
            except Exception:
                pass
            self.grip_constraint = None

    # ------------------------------------------------------------------
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
