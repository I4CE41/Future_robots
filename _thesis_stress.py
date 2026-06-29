"""Section 4.9 stress sweep for main12.py.

Runs the FULL variant across two honestly-sweepable operating axes and writes
one summary CSV per axis plus the raw per-config batch CSVs:

  * Initial-SoC axis  : --initial-soc in {0.70 .. 0.15}.  Exercises phi(z_bar),
                        the low-SoC speed throttle (low_soc_threshold=0.30) and
                        the emergency-stop guard (emergency_soc_threshold=0.10).
  * Dynamic-obstacle  : --dyn-count in {0,1,2} with --dynamic.  Adds the
    axis              sinusoidal AGV/pedestrian sweep on top of the static
                        3-corridor slalom.

Each axis point is a real `main12.py --batch` run (subprocess, BLAS pinned to
1 thread, workers clamped to phys-2). Nothing is fabricated: every number in
the summary CSVs is aggregated from recorded mission logs.
"""

import csv
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN12 = os.path.join(HERE, "main12.py")
PY = sys.executable
STRESS_DIR = os.path.join(HERE, "stress")
os.makedirs(STRESS_DIR, exist_ok=True)

SEEDS = 8
SEED_BASE = 1000
NUM_BOXES = 3
TIMEOUT = 300
RUN_TIMEOUT = 8 * 60 * 60  # whole-batch subprocess guard (s)

SOC_VALUES = [0.70, 0.55, 0.40, 0.30, 0.20, 0.15]
DYN_VALUES = [0, 1, 2]

FIELDS = ["variant", "seed", "success", "boxes", "num_boxes", "failure",
          "mission_time_s", "path_m", "energy_wh", "regen_wh", "final_soc",
          "terminal_v", "max_cell_temp", "soc_spread", "nmpc_mean_ms",
          "nmpc_p99_ms", "nmpc_solves", "switches", "astar_replans",
          "recoveries", "collisions", "wedge_escapes", "corridor_narrow_s",
          "corridor_class"]
NUM_FIELDS = ["mission_time_s", "energy_wh", "regen_wh", "final_soc",
              "soc_spread", "nmpc_mean_ms", "recoveries", "collisions"]


def run_batch(out_csv, extra):
    cmd = [PY, MAIN12, "--no-gui", "--no-dashboard", "--variant", "full",
           "--batch", str(SEEDS), "--workers", "8", "--seed-base", str(SEED_BASE),
           "--num-boxes", str(NUM_BOXES), "--timeout", str(TIMEOUT),
           "--batch-out", out_csv] + extra
    print("[stress] " + " ".join(cmd[2:]), flush=True)
    subprocess.run(cmd, cwd=HERE, timeout=RUN_TIMEOUT)


def load(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def fnum(r, k):
    try:
        return float(r[k])
    except (ValueError, KeyError, TypeError):
        return 0.0


def mean(vals):
    return sum(vals) / len(vals) if vals else 0.0


def summarize(rows):
    n = len(rows)
    ok = [r for r in rows if int(float(r["success"])) == 1]
    out = {"runs": n, "n_success": len(ok),
           "success_pct": round(100.0 * len(ok) / n, 1) if n else 0.0,
           "n_timeout": sum(1 for r in rows
                            if int(float(r["success"])) == 0
                            and r.get("failure") == "timeout"),
           "n_battery": sum(1 for r in rows
                            if int(float(r["success"])) == 0
                            and r.get("failure") == "battery"),
           "n_collision": sum(1 for r in rows
                              if int(float(r["success"])) == 0
                              and r.get("failure") == "collision")}
    src = ok if ok else rows
    for k in NUM_FIELDS:
        out["mean_" + k] = round(mean([fnum(r, k) for r in src]), 3)
    return out


def main():
    soc_summ, dyn_summ = [], []

    for soc in SOC_VALUES:
        tag = f"soc_{int(round(soc * 100)):03d}"
        out = os.path.join(STRESS_DIR, f"stress_{tag}.csv")
        run_batch(out, ["--initial-soc", str(soc)])
        s = summarize(load(out))
        s["axis"] = "initial_soc"
        s["value"] = soc
        soc_summ.append(s)

    for dyn in DYN_VALUES:
        tag = f"dyn_{dyn}"
        out = os.path.join(STRESS_DIR, f"stress_{tag}.csv")
        extra = ["--initial-soc", "0.70", "--dyn-count", str(dyn)]
        if dyn > 0:
            extra.append("--dynamic")
        run_batch(out, extra)
        s = summarize(load(out))
        s["axis"] = "dyn_count"
        s["value"] = dyn
        dyn_summ.append(s)

    cols = (["axis", "value", "runs", "n_success", "success_pct",
             "n_timeout", "n_battery", "n_collision"]
            + ["mean_" + k for k in NUM_FIELDS])
    for name, summ in (("stress_summary_soc.csv", soc_summ),
                       ("stress_summary_dyn.csv", dyn_summ)):
        path = os.path.join(STRESS_DIR, name)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(summ)
        print(f"\n[stress] wrote {path}", flush=True)

    print("\n=== INITIAL-SoC AXIS ===")
    for s in soc_summ:
        print(f"  SoC0={s['value']:.2f}  succ={s['success_pct']:5.1f}%  "
              f"E={s['mean_energy_wh']:.1f}Wh  t={s['mean_mission_time_s']:.0f}s  "
              f"fin={s['mean_final_soc']:.1f}%  to={s['n_timeout']} bat={s['n_battery']}")
    print("=== DYNAMIC-OBSTACLE AXIS ===")
    for s in dyn_summ:
        print(f"  dyn={int(s['value'])}  succ={s['success_pct']:5.1f}%  "
              f"E={s['mean_energy_wh']:.1f}Wh  t={s['mean_mission_time_s']:.0f}s  "
              f"col={s['mean_collisions']:.2f}  rec={s['mean_recoveries']:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
