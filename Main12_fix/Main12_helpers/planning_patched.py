"""Collision-safe planning wrappers.

ROOT CAUSE (path_clearance.py:164 + main12.py:844,851)
------------------------------------------------------
plan_with_clearance returns `[tuple(goal_xy)]` (a single point) when A* finds
no path. main12 then turns that into densify([(rx,ry), goal_xy]) -- a straight
line that is NEVER collision-checked and is scored as a real candidate. In a
tight corridor where the required clearance exceeds the corridor half-width,
A* legitimately returns nothing, this straight line wins, and the robot drives
through the wall.

FIX
---
* plan_with_clearance_safe: same iterative-inflation logic, but returns None on
  failure instead of a fake straight-line path. Callers must treat None as "no
  path" rather than "go straight to the goal".
* reject_colliding: drop any candidate path whose minimum clearance to the
  ORIGINAL obstacles is below a small tolerance, so even the los_safe "direct"
  shortcut can't slip a wall-cutting path through.
"""

import math

from robot_rescue.control.main12_helpers.main3_helpers.path_clearance import (
    detect_pinch_points,
    path_min_clearance,
)


def plan_with_clearance_safe(astar, start_xy, goal_xy, obstacles,
                             min_clearance=0.25, gap_min=0.80,
                             max_attempts=3, inflation_step=0.15):
    """Like plan_with_clearance but returns (path|None, info).

    On planner failure (A* returns a degenerate <2-point path on every
    attempt) this returns (None, info) instead of synthesizing an unchecked
    straight line to the goal.
    """
    attempts = 0
    inflation_bonus = 0.0
    final_path = None
    final_clearance = float("-inf")
    final_pinches = []

    while attempts < max_attempts:
        inflated = [(ox, oy, orad + inflation_bonus) for (ox, oy, orad) in obstacles]
        try:
            path = astar.plan(start_xy, goal_xy, inflated)
        except Exception:
            path = None

        # WorldAwareAStar returns [tuple(goal_xy)] (len 1) when no path exists;
        # treat anything shorter than 2 points as failure.
        if not path or len(path) < 2:
            break

        clearance, _ = path_min_clearance(path, obstacles)
        pinches = detect_pinch_points(path, obstacles, gap_min=gap_min)

        final_path = path
        final_clearance = clearance
        final_pinches = pinches

        if clearance >= min_clearance and not pinches:
            return path, {
                "attempts": attempts + 1,
                "inflation_used": inflation_bonus,
                "min_clearance": clearance,
                "pinches": pinches,
                "failed": False,
            }

        attempts += 1
        inflation_bonus += inflation_step

    # No clean path. Return the best real path we found (still collision-checked
    # by the caller via reject_colliding); only None if A* never produced one.
    return final_path, {
        "attempts": attempts,
        "inflation_used": inflation_bonus,
        "min_clearance": final_clearance,
        "pinches": final_pinches,
        "failed": final_path is None,
    }


def reject_colliding(path, obstacles, min_ok=0.05):
    """Return True if the path is safe to keep (min clearance >= min_ok)."""
    if not path or len(path) < 2:
        return False
    clearance, _ = path_min_clearance(path, obstacles)
    return clearance >= min_ok
