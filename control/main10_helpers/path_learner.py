"""Path learning: memorize successful paths and learn optimal movements.

Stores waypoint sequences from successful runs. On future runs, uses
the closest stored path as a guide, biasing DWA toward proven waypoints.
Learns by keeping only improving paths (shorter time, less energy).
"""

import os
import json
import numpy as np


class PathLearner:
    MEMORY_FILE = os.path.join(os.path.dirname(__file__), "..", "path_memory.json")

    def __init__(self):
        self.current_path = []       # [(x, y, theta, t), ...]
        self.current_waypoints = []  # decimated version for storage
        self.recording = False
        self.memory = self._load_memory()

    def _load_memory(self):
        if os.path.exists(self.MEMORY_FILE):
            try:
                with open(self.MEMORY_FILE, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"paths": {}}

    def _save_memory(self):
        with open(self.MEMORY_FILE, "w") as f:
            json.dump(self.memory, f, indent=2)

    def start_recording(self, task_key):
        """Begin recording a new path for a task (e.g. 'navigate_0_0_to_3_2')."""
        self.task_key = task_key
        self.current_path = []
        self.recording = True

    def record_step(self, x, y, theta, t):
        """Record a position along the path."""
        if not self.recording:
            return
        self.current_path.append((x, y, theta, t))

    def finish_recording(self, success, total_energy, total_time):
        """Finish recording. Save if successful and better than previous."""
        self.recording = False
        if not success or len(self.current_path) < 5:
            return

        # Decimate path to ~20 waypoints for storage
        step = max(1, len(self.current_path) // 20)
        waypoints = self.current_path[::step]
        if self.current_path[-1] not in waypoints:
            waypoints.append(self.current_path[-1])

        entry = {
            "waypoints": waypoints,
            "energy": total_energy,
            "time": total_time,
            "steps": len(self.current_path),
        }

        # Keep only if better than existing
        existing = self.memory["paths"].get(self.task_key)
        if existing is None or total_time < existing["time"]:
            self.memory["paths"][self.task_key] = entry
            self._save_memory()
            print(f"[PathLearner] Saved path for '{self.task_key}' "
                  f"({len(waypoints)} waypoints, {total_time:.1f}s, {total_energy:.3f}Wh)")
            if existing:
                improvement = (existing["time"] - total_time) / existing["time"] * 100
                print(f"[PathLearner] Improved by {improvement:.1f}%!")
        else:
            print(f"[PathLearner] Path not better than stored "
                  f"({total_time:.1f}s vs {existing['time']:.1f}s)")

    def get_learned_path(self, task_key):
        """Retrieve stored waypoints for a task. Returns list of (x,y,theta,t) or None."""
        entry = self.memory["paths"].get(task_key)
        if entry:
            return [tuple(w) for w in entry["waypoints"]]
        return None

    def get_next_waypoint(self, x, y, task_key, lookahead=1.0):
        """Get the next waypoint from learned path that's ahead of current position.

        Returns (wx, wy) or None if no learned path.
        Finds the closest point on the learned path, then returns
        the point 'lookahead' meters ahead along the path.
        """
        path = self.get_learned_path(task_key)
        if not path:
            return None

        # Find closest point on path
        min_dist = float("inf")
        closest_idx = 0
        for i, (px, py, _, _) in enumerate(path):
            d = np.sqrt((x - px) ** 2 + (y - py) ** 2)
            if d < min_dist:
                min_dist = d
                closest_idx = i

        # Walk ahead along path by lookahead distance
        accumulated = 0.0
        for i in range(closest_idx, len(path) - 1):
            dx = path[i + 1][0] - path[i][0]
            dy = path[i + 1][1] - path[i][1]
            seg_len = np.sqrt(dx ** 2 + dy ** 2)
            accumulated += seg_len
            if accumulated >= lookahead:
                return (path[i + 1][0], path[i + 1][1])

        # Return final waypoint if near end
        return (path[-1][0], path[-1][1])

    def make_task_key(self, start, goal):
        """Create a unique key for a start->goal task."""
        sx, sy = round(start[0], 1), round(start[1], 1)
        gx, gy = round(goal[0], 1), round(goal[1], 1)
        return f"{sx}_{sy}_to_{gx}_{gy}"

    def get_stats(self):
        """Return summary of learned paths."""
        stats = {}
        for key, entry in self.memory["paths"].items():
            stats[key] = {
                "waypoints": len(entry["waypoints"]),
                "time": entry["time"],
                "energy": entry["energy"],
            }
        return stats
