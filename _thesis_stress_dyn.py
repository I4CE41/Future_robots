"""
Calibrated dynamic-obstacle sweep for the 5-DOF mobile manipulator.

WHY the previous config gave 0% success
────────────────────────────────────────
  1. NUM_BOXES=3 + TIMEOUT=300 s  →  robot needs ~120 s per box → 360 s minimum,
     no margin for navigation or recovery.  Every "full-timeout" seed hit 240 s
     (the internal mission cap) with boxes=0-2/3 still pending.
  2. initial_soc=0.70  →  timeout seeds drained to SoC≈40% (≈102 Wh) just
     wandering, leaving nothing for the remaining sub-tasks.
  3. WORKERS=75 on 16 GB RAM  →  ~75 PyBullet processes × 200-400 MB ≈ crash /
     swap, inflating solve times and causing spurious collisions.

Calibration strategy (progressive difficulty)
──────────────────────────────────────────────
  Stage 1 (this script)  : NUM_BOXES=1, TIMEOUT=400, SoC=0.90
     → expect 80-95% success at dyn=0, degrading gracefully with dyn=1,2.
  Stage 2 (edit below)   : NUM_BOXES=2, TIMEOUT=450, SoC=0.85
  Stage 3 (thesis final) : NUM_BOXES=3, TIMEOUT=500, SoC=0.70, SEEDS=75
"""

import csv
import os
import subprocess
import sys

HERE        = os.path.dirname(os.path.abspath(__file__))
MAIN12      = os.path.join(HERE, "main12.py")
PY          = sys.executable
STRESS_DIR  = os.path.join(HERE, "stress")
os.makedirs(STRESS_DIR, exist_ok=True)

# ── Batch configuration ────────────────────────────────────────────────────────
#  Quick-iteration defaults (Stage 1).
#  For the final thesis run set: SEEDS=75, NUM_BOXES=3, TIMEOUT=500, SoC=0.70
SEEDS       = 75           # 20 seeds → fast iteration; 75 for thesis
SEED_BASE   = 1000
NUM_BOXES   = 3            # ← was 3; start here, increase once robot passes
TIMEOUT     = 500           # s per run (was 300 — too tight for 3 boxes)
RUN_TIMEOUT = 4 * 60 * 60  # wall-clock cap for the whole subprocess call (4 h)
WORKERS     = 6             # safe for 16 GB RAM (each PyBullet ≈ 200-400 MB)
INITIAL_SOC = 0.7          # ← was 0.70; gave too little energy margin
DYN_VALUES  = [0, 1, 2]    # axis: number of sinusoidal AGV/pedestrian obstacles

NUM_FIELDS = [
    "mission_time_s", "energy_wh", "regen_wh", "final_soc",
    "soc_spread", "nmpc_mean_ms", "recoveries", "collisions",
]


# ── Core helpers ───────────────────────────────────────────────────────────────

def run_batch(out_csv: str, extra: list) -> None:
    cmd = [
        PY, MAIN12,
        "--no-gui", "--no-dashboard",
        "--variant",   "full",
        "--batch",     str(SEEDS),
        "--workers",   str(WORKERS),
        "--seed-base", str(SEED_BASE),
        "--num-boxes", str(NUM_BOXES),
        "--timeout",   str(TIMEOUT),
        "--batch-out", out_csv,
    ] + extra
    print("[dyn] " + " ".join(cmd[2:]), flush=True)
    subprocess.run(cmd, cwd=HERE, timeout=RUN_TIMEOUT)


def load(p: str) -> list:
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def fnum(row: dict, key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, ValueError, TypeError):
        return 0.0


def mean(values: list) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize(rows: list) -> dict:
    n  = len(rows)
    ok = [r for r in rows if int(float(r["success"])) == 1]

    out = {
        "runs":        n,
        "n_success":   len(ok),
        "success_pct": round(100.0 * len(ok) / n, 1) if n else 0.0,
        "n_timeout":   sum(
            1 for r in rows
            if int(float(r["success"])) == 0 and r.get("failure") == "timeout"
        ),
        "n_battery":   sum(
            1 for r in rows
            if int(float(r["success"])) == 0 and r.get("failure") == "battery"
        ),
        "n_collision": sum(
            1 for r in rows
            if int(float(r["success"])) == 0 and r.get("failure") == "collision"
        ),
    }

    # Use successful runs for metric averages; fall back to all if none succeeded
    src = ok if ok else rows
    for k in NUM_FIELDS:
        out[f"mean_{k}"] = round(mean([fnum(r, k) for r in src]), 3)

    return out


def calibration_hint(s: dict, dyn: int) -> str:
    """Return a short diagnostic hint based on the success rate."""
    pct    = s["success_pct"]
    n_col  = s["n_collision"]
    n_tout = s["n_timeout"]
    n      = s["runs"]

    if pct >= 80:
        return "✓  good — robot handles this difficulty level"
    if pct >= 50:
        dominant = "collisions" if n_col > n_tout else "timeouts"
        return f"△  marginal ({pct:.0f}%) — dominated by {dominant}"
    if n_col / max(n, 1) > 0.5:
        return "✗  CBF/DWA struggling with dynamic obstacles — too many collisions"
    if n_tout / max(n, 1) > 0.5:
        return "✗  robot stuck in recovery loops — increase TIMEOUT or fix escalation"
    return f"✗  {pct:.0f}% — config is too hard for this stage"


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> int:
    print(
        f"\n[config]  seeds={SEEDS}  num_boxes={NUM_BOXES}  "
        f"timeout={TIMEOUT} s  workers={WORKERS}  initial_soc={INITIAL_SOC:.0%}",
        flush=True,
    )

    summ = []

    for dyn in DYN_VALUES:
        out_csv = os.path.join(STRESS_DIR, f"stress_dynbox{NUM_BOXES}_dyn{dyn}.csv")
        extra   = ["--initial-soc", str(INITIAL_SOC), "--dyn-count", str(dyn)]
        if dyn > 0:
            extra.append("--dynamic")

        run_batch(out_csv, extra)

        s          = summarize(load(out_csv))
        s["axis"]  = "dyn_count"
        s["value"] = dyn
        summ.append(s)

    # ── Write summary CSV ──────────────────────────────────────────────────────
    cols = (
        ["axis", "value", "runs", "n_success", "success_pct",
         "n_timeout", "n_battery", "n_collision"]
        + [f"mean_{k}" for k in NUM_FIELDS]
    )
    summary_path = os.path.join(STRESS_DIR, f"stress_summary_dynbox{NUM_BOXES}.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(summ)

    # ── Console report ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  DYNAMIC AXIS  |  num_boxes={NUM_BOXES}  soc={INITIAL_SOC:.0%}  timeout={TIMEOUT}s")
    print(f"{'='*60}")
    print(f"  {'dyn':>3}  {'succ':>6}  {'E(Wh)':>7}  {'t(s)':>5}  "
          f"{'col':>4}  {'rec':>4}  {'tout':>4}  note")
    print(f"  {'-'*55}")
    for s in summ:
        dyn_val = int(s["value"])
        hint    = calibration_hint(s, dyn_val)
        print(
            f"  {dyn_val:>3}  "
            f"{s['success_pct']:>5.1f}%  "
            f"{s['mean_energy_wh']:>7.1f}  "
            f"{s['mean_mission_time_s']:>5.0f}  "
            f"{s['mean_collisions']:>4.2f}  "
            f"{s['mean_recoveries']:>4.1f}  "
            f"{s['n_timeout']:>4}  "
            f"{hint}"
        )
    print(f"{'='*60}")
    print(f"\n[done] summary → {summary_path}")

    # ── Next-step guidance ─────────────────────────────────────────────────────
    print("""
── Next steps ────────────────────────────────────────────────────────────
  If dyn=0 success < 80% :  the robot itself has a problem (not the test).
                             Debug main12.py before continuing.
  If dyn=0 OK, dyn=1/2 bad: CBF angular correction is failing under
                             dynamic loads. Check corridor classification
                             and CBF frame alignment in main12.py.
  Once Stage 1 passes      : set NUM_BOXES=2, TIMEOUT=450, SoC=0.85
  Final thesis run         : set NUM_BOXES=3, TIMEOUT=500, SoC=0.70, SEEDS=75
─────────────────────────────────────────────────────────────────────────
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())