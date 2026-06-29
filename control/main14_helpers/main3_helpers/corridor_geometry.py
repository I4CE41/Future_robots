"""Geometric corridor extraction and classification utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
import math


@dataclass
class CorridorState:
    angle: float = 0.0
    width: float = float("inf")
    min_width: float = float("inf")
    mean_width: float = float("inf")
    max_width: float = 0.0
    length: float = 0.0
    curvature: float = 0.0
    boundary_clearance: float = float("inf")
    center_offset: float = 0.0
    left_boundary: float = 0.0
    right_boundary: float = 0.0
    width_profile: list[float] = field(default_factory=list)
    centerline: list[tuple[float, float]] = field(default_factory=list)
    is_narrow: bool = False
    is_tight: bool = False
    is_dead_end: bool = False
    classification: str = "open"
    velocity_scale: float = 1.0
    score: float = 0.0


class CorridorGeometryExtractor:
    """Ray-casting corridor extractor used by main11.

    The extractor treats every obstacle as an inflated circle for the robot
    centerline. It builds a local free-space cross-section around the robot and,
    when a path is available, a width/curvature profile along that path.
    """

    def __init__(self, robot_radius=0.30, safety_margin=0.12,
                 ray_range=2.8, ray_resolution=0.05,
                 path_spacing=0.25, profile_span=2.4):
        self.robot_radius = float(robot_radius)
        self.safety_margin = float(safety_margin)
        self.inflate = self.robot_radius + self.safety_margin
        self.ray_range = float(ray_range)
        self.ray_resolution = float(ray_resolution)
        self.path_spacing = float(path_spacing)
        self.profile_span = float(profile_span)

    def _ray_distance(self, origin, angle, obstacles):
        ox, oy = origin
        dx = math.cos(angle)
        dy = math.sin(angle)
        best = self.ray_range
        for cx, cy, r in obstacles:
            vx = cx - ox
            vy = cy - oy
            proj = vx * dx + vy * dy
            if proj < 0:
                continue
            perp = math.hypot(vx * dy - vy * dx)
            safe_r = r + self.inflate
            if perp > safe_r:
                continue
            along = proj - math.sqrt(max(0.0, safe_r * safe_r - perp * perp))
            if 0.0 <= along <= best:
                best = along
        return max(0.0, best)

    def _cross_section(self, point, direction, obstacles):
        px, py = point
        dx, dy = direction
        norm = math.hypot(dx, dy)
        if norm < 1e-9:
            return 0.0, self.ray_range, self.ray_range, 0.0
        dx /= norm
        dy /= norm
        perp_x = -dy
        perp_y = dx
        left = self._ray_distance((px, py), math.atan2(perp_y, perp_x), obstacles)
        right = self._ray_distance((px, py), math.atan2(-perp_y, -perp_x), obstacles)
        width = max(0.0, left + right)
        center_offset = 0.5 * (left - right)
        return width, left, right, center_offset

    def _menger_curvature(self, a, b, c):
        ab = math.hypot(b[0] - a[0], b[1] - a[1])
        bc = math.hypot(c[0] - b[0], c[1] - b[1])
        ca = math.hypot(a[0] - c[0], a[1] - c[1])
        denom = ab * bc * ca
        if denom < 1e-9:
            return 0.0
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        return 2.0 * abs(cross) / denom

    def _sample_path(self, path):
        if not path or len(path) < 2:
            return []
        out = [tuple(path[0])]
        for (x0, y0), (x1, y1) in zip(path[:-1], path[1:]):
            seg = math.hypot(x1 - x0, y1 - y0)
            n = max(1, int(math.ceil(seg / max(self.path_spacing, 1e-3))))
            for k in range(1, n + 1):
                t = k / n
                out.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))
        return out

    def _classify(self, width, min_width, forward_distance):
        if forward_distance < 0.65 or min_width < 0.55:
            return "dead_end", True
        if min_width < 0.75:
            return "tight", True
        if width < 0.95:
            return "narrow", True
        if width < 1.35:
            return "constrained", False
        return "open", False

    def _velocity_scale(self, width, min_width):
        effective = min(width, min_width)
        if math.isinf(effective):
            return 1.0
        if effective <= 0.65:
            return 0.25
        if effective >= 1.35:
            return 1.0
        t = (effective - 0.65) / 0.70
        return 0.25 + 0.75 * t

    def _score(self, width, min_width, curvature, length, dead_end, goal_alignment):
        if math.isinf(width):
            return 0.0
        dead = 3.5 if dead_end else 0.0
        return (
            1.35 * min_width
            + 0.55 * width
            + 0.05 * length
            - 2.0 * curvature
            - dead
            + 0.5 * goal_alignment
        )

    def _open_state(self, robot_xy, goal_xy=None, obstacles=None):
        if goal_xy is None:
            return CorridorState()
        rx, ry = robot_xy
        angle = math.atan2(goal_xy[1] - ry, goal_xy[0] - rx)
        forward = self._ray_distance((rx, ry), angle, obstacles or [])
        left = self._ray_distance((rx, ry), angle + math.pi / 2.0, obstacles or [])
        right = self._ray_distance((rx, ry), angle - math.pi / 2.0, obstacles or [])
        width = left + right
        classification, narrow = self._classify(width, width, forward)
        return CorridorState(
            angle=angle,
            width=width,
            min_width=width,
            mean_width=width,
            max_width=width,
            boundary_clearance=min(left, right),
            left_boundary=-left,
            right_boundary=right,
            width_profile=[width],
            is_narrow=narrow and classification in {"narrow", "tight"},
            is_tight=classification == "tight",
            is_dead_end=classification == "dead_end",
            classification=classification,
            velocity_scale=self._velocity_scale(width, width),
            score=self._score(width, width, 0.0, 0.0, False, 1.0),
        )

    def extract(self, state, goal_xy, path=None, obstacles=None):
        """Extract the best corridor around the robot.

        If a path is available, the returned state is a path-aligned corridor
        with a width profile and curvature estimate. Without a path, it falls
        back to goal-aligned local ray casting.
        """
        obstacles = list(obstacles or [])
        rx = float(state["x"])
        ry = float(state["y"])
        theta = float(state.get("theta", 0.0))
        goal_angle = math.atan2(goal_xy[1] - ry, goal_xy[0] - rx)

        sampled = self._sample_path(path)
        if len(sampled) < 2:
            return self._open_state((rx, ry), (goal_xy[0], goal_xy[1]), obstacles)

        local_path = [(p[0] - rx, p[1] - ry) for p in sampled]
        forward = math.hypot(local_path[-1][0], local_path[-1][1])
        direction = (local_path[-1][0], local_path[-1][1])
        widths = []
        lefts = []
        rights = []
        offsets = []
        curvatures = []
        centerline = []
        for i, (lx, ly) in enumerate(local_path):
            wx, wy = rx + lx, ry + ly
            width, left, right, offset = self._cross_section(
                (wx, wy), direction, obstacles)
            widths.append(width)
            lefts.append(left)
            rights.append(right)
            offsets.append(offset)
            centerline.append((lx, ly))
            if 0 < i < len(local_path) - 1:
                curvatures.append(self._menger_curvature(
                    local_path[i - 1], local_path[i], local_path[i + 1]))

        width = widths[0] if widths else float("inf")
        min_width = min(widths) if widths else float("inf")
        mean_width = sum(widths) / len(widths) if widths else float("inf")
        max_width = max(widths) if widths else 0.0
        curvature = max(curvatures) if curvatures else 0.0
        boundary_clearance = min([min(l, r) for l, r in zip(lefts, rights)] or [float("inf")])
        center_offset = offsets[0] if offsets else 0.0
        left_boundary = -lefts[0] if lefts else 0.0
        right_boundary = rights[0] if rights else 0.0
        classification, narrow = self._classify(width, min_width, forward)
        is_tight = classification == "tight"
        is_dead_end = classification == "dead_end"
        alignment = max(0.0, math.cos(goal_angle - math.atan2(direction[1], direction[0])))
        score = self._score(mean_width, min_width, curvature, forward, is_dead_end, alignment)

        return CorridorState(
            angle=math.atan2(direction[1], direction[0]),
            width=width,
            min_width=min_width,
            mean_width=mean_width,
            max_width=max_width,
            length=forward,
            curvature=curvature,
            boundary_clearance=boundary_clearance,
            center_offset=center_offset,
            left_boundary=left_boundary,
            right_boundary=right_boundary,
            width_profile=widths,
            centerline=centerline,
            is_narrow=narrow,
            is_tight=is_tight,
            is_dead_end=is_dead_end,
            classification=classification,
            velocity_scale=self._velocity_scale(width, min_width),
            score=score,
        )

    def to_nmpc_corridor(self, corridor, state, points, dt=0.1):
        """Build fixed-length NMPC corridor parameters from a CorridorState."""
        if corridor is None or not corridor.centerline:
            half_width = 100.0
            return {
                "center_x": [float(state["x"])] * points,
                "center_y": [float(state["y"])] * points,
                "half_width": [half_width] * points,
            }
        rx = float(state["x"])
        ry = float(state["y"])
        theta = float(state.get("theta", 0.0))
        heading = (math.cos(theta), math.sin(theta))
        centerline = list(corridor.centerline)
        if len(centerline) < points:
            tail = centerline[-1] if centerline else (0.0, 0.0)
            centerline.extend([tail] * (points - len(centerline)))
        centerline = centerline[:points]
        center_x = []
        center_y = []
        half_width = []
        for i, (lx, ly) in enumerate(centerline):
            projected = lx * heading[0] + ly * heading[1]
            target_x = rx + heading[0] * min(projected, max(0.05, dt * i))
            target_y = ry + heading[1] * min(projected, max(0.05, dt * i))
            center_x.append(target_x)
            center_y.append(target_y)
            half_width.append(max(0.45, min(1.20, 0.5 * float(corridor.width))))
        return {
            "center_x": center_x,
            "center_y": center_y,
            "half_width": half_width,
        }

def is_wedged(perc, front_thresh=0.10, lateral_clip=0.95):
    return (perc.front_clearance < front_thresh
            and perc.left_clearance >= lateral_clip
            and perc.right_clearance >= lateral_clip)
