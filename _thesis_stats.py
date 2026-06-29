"""Compute aggregate statistics for Chapter 4 from the new main12 CSVs."""
import csv
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# variant label -> csv file (mapping chosen by user: v2 family + pack from full.csv)
FILES = {
    "Full":       "main12_batch_full_v2.csv",
    "Pack-Level": "main12_batch_full.csv",
    "No-Energy":  "main12_batch_no_energy_v2.csv",
    "No-Regen":   "main12_batch_no_regenerative_v2.csv",
    "Speed-Only": "main12_batch_speed_only_v2.csv",
}

NUM_FIELDS = ["mission_time_s", "path_m", "energy_wh", "regen_wh", "final_soc",
              "terminal_v", "max_cell_temp", "soc_spread", "nmpc_mean_ms",
              "nmpc_p99_ms", "switches", "astar_replans", "recoveries",
              "collisions", "wedge_escapes", "corridor_narrow_s"]


def load(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def fnum(r, k):
    try:
        return float(r[k])
    except (ValueError, KeyError, TypeError):
        return 0.0


def mean_std(vals):
    n = len(vals)
    if n == 0:
        return 0.0, 0.0
    m = sum(vals) / n
    if n == 1:
        return m, 0.0
    var = sum((v - m) ** 2 for v in vals) / (n - 1)
    return m, math.sqrt(var)


print("=" * 100)
for label, fname in FILES.items():
    path = os.path.join(HERE, fname)
    rows = load(path)
    n = len(rows)
    ok = [r for r in rows if int(float(r["success"])) == 1]
    nok = len(ok)
    succ = 100.0 * nok / n if n else 0.0
    print(f"\n### {label}  ({fname})")
    print(f"  runs={n}  success={nok}/{n} = {succ:.1f}%")
    # failure breakdown
    fails = {}
    for r in rows:
        if int(float(r["success"])) == 0:
            fails[r["failure"]] = fails.get(r["failure"], 0) + 1
    if fails:
        print(f"  failures: {fails}")
    # stats over successful runs only (exclude timeouts that zero-out)
    for k in NUM_FIELDS:
        vals = [fnum(r, k) for r in ok]
        m, s = mean_std(vals)
        print(f"    {k:18s} {m:10.3f} +/- {s:7.3f}")

print("\n" + "=" * 100)
print("ENERGY SAVINGS (mean energy_wh over successful runs):")
emean = {}
for label, fname in FILES.items():
    rows = load(os.path.join(HERE, fname))
    ok = [r for r in rows if int(float(r["success"])) == 1]
    emean[label] = sum(fnum(r, "energy_wh") for r in ok) / len(ok)
for label in FILES:
    print(f"  {label:12s} {emean[label]:8.3f} Wh")
full = emean["Full"]
for label in ["Pack-Level", "No-Regen", "Speed-Only"]:
    base = emean[label]
    red = 100.0 * (base - full) / base
    print(f"  Full vs {label:12s}: {red:6.2f}% reduction")
