"""Runtime monkey-patches for known main2 bugs.

NO existing files are edited; everything here is applied at runtime from main3.py.
"""

import types
import numpy as np


def apply_shims(energy_mgr):
    """Patch missing methods on EnergyManager so main3 can call them safely."""
    if not hasattr(energy_mgr, "set_initial_soc"):
        def _set_initial_soc(self, soc):
            self.soc = float(max(0.0, min(1.0, soc)))
        energy_mgr.set_initial_soc = types.MethodType(_set_initial_soc, energy_mgr)

    if not hasattr(energy_mgr, "energy_regenerated"):
        energy_mgr.energy_regenerated = 0.0

    if not hasattr(energy_mgr, "regen_log"):
        energy_mgr.regen_log = []

    if not hasattr(energy_mgr, "estimate_remaining_time"):
        def _est(self):
            cap_wh = getattr(self, "Q_nom", 20.0) * getattr(self, "v_nom", 26.2)
            soc = self.get_soc()
            return soc * cap_wh / max(40.0, 1e-3) * 3600
        energy_mgr.estimate_remaining_time = types.MethodType(_est, energy_mgr)


def guard_array(arr, dim, default=0.0):
    """Return arr as a numpy array of length `dim`; replace NaN/inf with `default`."""
    a = np.asarray(arr, dtype=float).flatten()
    if a.size < dim:
        a = np.concatenate([a, np.full(dim - a.size, default)])
    elif a.size > dim:
        a = a[:dim]
    if not np.all(np.isfinite(a)):
        a = np.where(np.isfinite(a), a, default)
    return a


def safe_call(fn, *args, default=None, label=""):
    """Run fn(*args). On any exception, log once and return `default`."""
    try:
        return fn(*args)
    except Exception as e:
        if label and not getattr(safe_call, "_warned", set()).__contains__(label):
            print(f"[shim] {label} failed: {e}")
            seen = getattr(safe_call, "_warned", set())
            seen.add(label)
            safe_call._warned = seen
        return default
