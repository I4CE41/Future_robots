"""Chapter-4 comparative figures from the main12 batch CSVs (5 variants).

Honest reliability/health narrative. Clean line-style plots; bar charts have
no error-bar whisker lines (replaced with a faint +/-1 std shaded band).
"""
import csv
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(os.path.expanduser("~"), "Downloads",
                       "Louzini_Manuscript", "Thesis_Final", "figures")

FILES = {
    "Full":       "main12_batch_full_v2.csv",
    "Pack-Level": "main12_batch_full.csv",
    "No-Energy":  "main12_batch_no_energy_v2.csv",
    "No-Regen":   "main12_batch_no_regenerative_v2.csv",
    "Speed-Only": "main12_batch_speed_only_v2.csv",
}
ORDER = ["Full", "Pack-Level", "No-Energy", "No-Regen", "Speed-Only"]
COLORS = {"Full": "#1f77b4", "Pack-Level": "#ff7f0e", "No-Energy": "#9467bd",
          "No-Regen": "#2ca02c", "Speed-Only": "#d62728"}
MARKERS = {"Full": "o", "Pack-Level": "s", "No-Energy": "D",
           "No-Regen": "^", "Speed-Only": "v"}

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True,
})


def load(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


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
    return m, math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1))


DATA = {}
for label, fname in FILES.items():
    rows = load(os.path.join(HERE, fname))
    ok = [r for r in rows if int(float(r["success"])) == 1]
    DATA[label] = {"n": len(rows), "nok": len(ok),
                   "succ": 100.0 * len(ok) / len(rows), "ok": ok}


def col(label, key):
    return [fnum(r, key) for r in DATA[label]["ok"]]


def clean_bar(ax, key, ylabel, title):
    """Bar chart with a soft +/-1 std band instead of ugly whisker caps."""
    means, stds = [], []
    for lb in ORDER:
        m, s = mean_std(col(lb, key))
        means.append(m); stds.append(s)
    x = np.arange(len(ORDER))
    for xi, m, s, lb in zip(x, means, stds, ORDER):
        ax.bar(xi, m, width=0.62, color=COLORS[lb], edgecolor="none",
               alpha=0.88, zorder=2)
        if s > 0:
            ax.add_patch(plt.Rectangle((xi - 0.31, m - s), 0.62, 2 * s,
                         color="black", alpha=0.10, zorder=3))
        ax.text(xi, m + (max(means) * 0.012), f"{m:.2f}", ha="center",
                va="bottom", fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels(ORDER, fontsize=8.5, rotation=12)
    ax.set_ylabel(ylabel); ax.set_title(title, fontweight="bold")
    ax.grid(axis="x", visible=False)


# ---- Fig: success rate (clean bar) -------------------------------------
def fig_success():
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    succ = [DATA[lb]["succ"] for lb in ORDER]
    x = np.arange(len(ORDER))
    for xi, s, lb in zip(x, succ, ORDER):
        ax.bar(xi, s, width=0.62, color=COLORS[lb], alpha=0.88, zorder=2)
        ax.text(xi, s + 1, f"{s:.1f}%\n({DATA[lb]['nok']}/{DATA[lb]['n']})",
                ha="center", va="bottom", fontsize=8.5)
    ax.axhline(90, color="gray", ls="--", lw=1.0, alpha=0.6)
    ax.set_xticks(x); ax.set_xticklabels(ORDER, fontsize=8.5, rotation=12)
    ax.set_ylabel("Mission success rate (%)"); ax.set_ylim(0, 108)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig_compare_success.png")
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    print("saved", out)


# ---- Fig: 2x2 metric panel (clean bars) --------------------------------
def fig_metrics_panel():
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    clean_bar(axes[0, 0], "energy_wh", "Energy (Wh)", "(a) Energy consumed")
    clean_bar(axes[0, 1], "mission_time_s", "Mission time (s)", "(b) Mission time")
    clean_bar(axes[1, 0], "final_soc", "Final SoC (%)", "(c) Final state of charge")
    clean_bar(axes[1, 1], "nmpc_mean_ms", "NMPC solve (ms)", "(d) Mean NMPC solve time")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig_compare_metrics.png")
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    print("saved", out)


# ---- Fig: health (clean bars) ------------------------------------------
def fig_health():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    clean_bar(axes[0], "regen_wh", "Energy regenerated (Wh)", "(a) Regenerative recovery")
    clean_bar(axes[1], "soc_spread", "Inter-cell SoC spread (%)", "(b) Cell balance (lower is better)")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig_compare_health.png")
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    print("saved", out)


# ---- Fig: reliability vs energy as a clean LINE (sorted by energy) ------
def fig_tradeoff():
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    pts = []
    for lb in ORDER:
        e_m, _ = mean_std(col(lb, "energy_wh"))
        pts.append((e_m, DATA[lb]["succ"], lb))
    pts.sort(key=lambda p: p[0])
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    ax.plot(xs, ys, "-", color="#888888", lw=1.6, zorder=1)
    for e_m, s, lb in pts:
        ax.scatter(e_m, s, s=170, color=COLORS[lb], marker=MARKERS[lb],
                   edgecolor="white", linewidth=1.4, zorder=3, label=lb)
        ax.annotate(lb, (e_m, s), textcoords="offset points",
                    xytext=(6, 7), fontsize=9)
    ax.set_xlabel("Mean energy per mission (Wh)")
    ax.set_ylabel("Mission success rate (%)")
    ax.set_title("Reliability vs. energy across variants", fontweight="bold")
    ax.set_ylim(70, 102)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig_compare_tradeoff.png")
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    print("saved", out)


# ---- Fig: energy as a clean sorted LINE with +/-1 std band -------------
def fig_energy_line():
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    stats = []
    for lb in ORDER:
        m, s = mean_std(col(lb, "energy_wh"))
        stats.append((lb, m, s))
    x = np.arange(len(ORDER))
    means = [m for _, m, _ in stats]
    stds = [s for _, _, s in stats]
    ax.fill_between(x, [m - s for m, s in zip(means, stds)],
                    [m + s for m, s in zip(means, stds)],
                    color="#1f77b4", alpha=0.15, zorder=1,
                    label=r"$\pm 1$ std")
    ax.plot(x, means, "-o", color="#1f77b4", lw=2.2, markersize=8,
            zorder=3, label="Mean energy")
    for xi, (lb, m, s) in zip(x, stats):
        ax.scatter(xi, m, s=90, color=COLORS[lb], zorder=4,
                   edgecolor="white", linewidth=1.2)
        ax.text(xi, m + s + 0.6, f"{m:.1f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(ORDER, fontsize=9, rotation=12)
    ax.set_ylabel("Energy per mission (Wh)")
    ax.set_title("Mean mission energy with $\\pm1$ std band (successful runs)",
                 fontweight="bold")
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig_compare_energy_box.png")
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    fig_success()
    fig_metrics_panel()
    fig_health()
    fig_tradeoff()
    fig_energy_line()
    print("\n--- LaTeX numbers (mean +/- std over successful runs) ---")
    keys = ["mission_time_s", "energy_wh", "regen_wh", "final_soc",
            "soc_spread", "nmpc_mean_ms", "nmpc_p99_ms", "switches",
            "recoveries"]
    print("metric," + ",".join(ORDER))
    print("success_pct," + ",".join(f"{DATA[lb]['succ']:.1f}" for lb in ORDER))
    for k in keys:
        cells = []
        for lb in ORDER:
            m, s = mean_std(col(lb, k))
            cells.append(f"{m:.2f}+/-{s:.2f}")
        print(k + "," + ",".join(cells))
