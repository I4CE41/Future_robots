import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
STRESS_DIR = os.path.join(HERE, "stress")
FIG_DIR = os.path.join(os.path.expanduser("~"), "Downloads",
                       "Louzini_Manuscript", "Thesis_Final", "figures")

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

def plot_soc_stress():
    path = os.path.join(STRESS_DIR, "stress_summary_soc.csv")
    if not os.path.exists(path):
        print(f"Error: {path} not found.")
        return
    rows = load(path)
    # Sort by initial SoC descending (0.70 down to 0.15)
    rows.sort(key=lambda r: fnum(r, "value"), reverse=True)
    
    socs = [fnum(r, "value") for r in rows]
    success_pct = [fnum(r, "success_pct") for r in rows]
    times = [fnum(r, "mean_mission_time_s") for r in rows]
    energies = [fnum(r, "mean_energy_wh") for r in rows]
    
    fig, ax1 = plt.subplots(figsize=(7.5, 4.4))
    
    # Plot Success Rate (Bar chart)
    color = "#1f77b4"
    x = np.arange(len(socs))
    bars = ax1.bar(x, success_pct, width=0.4, color=color, alpha=0.7, label="Success Rate", zorder=2)
    ax1.set_xlabel("Initial State of Charge ($SoC_0$)", fontweight="bold")
    ax1.set_ylabel("Mission Success Rate (%)", color=color, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.set_ylim(0, 110)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{s:.2f}" for s in socs])
    
    # Text labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 2, f"{height:.1f}%",
                 ha="center", va="bottom", fontsize=9, color=color)
        
    # Plot Mission Time (Line plot on secondary axis)
    ax2 = ax1.twinx()
    color2 = "#d62728"
    # Show mission time only for configurations with success > 0, or show it for all
    ax2.plot(x, times, "-o", color=color2, lw=2.2, label="Mean Mission Time", zorder=3)
    ax2.set_ylabel("Mean Mission Time (s)", color=color2, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(0, 200)
    
    # Value annotations for time
    for i, txt in enumerate(times):
        ax2.annotate(f"{txt:.1f}s", (x[i], times[i]), textcoords="offset points",
                     xytext=(0,10), ha="center", fontsize=8.5, color=color2,
                     bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3))
        
    plt.title("Initial-SoC Stress Axis: Success Rate and Mission Time", fontweight="bold")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig_stress_soc.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)

def plot_dyn_stress():
    path = os.path.join(STRESS_DIR, "stress_summary_dyn.csv")
    if not os.path.exists(path):
        print(f"Error: {path} not found.")
        return
    rows = load(path)
    rows.sort(key=lambda r: fnum(r, "value"))
    
    dyns = [int(fnum(r, "value")) for r in rows]
    success_pct = [fnum(r, "success_pct") for r in rows]
    collisions = [fnum(r, "mean_collisions") for r in rows]
    recoveries = [fnum(r, "mean_recoveries") for r in rows]
    
    fig, ax1 = plt.subplots(figsize=(7.5, 4.4))
    
    # Success Rate (Bar chart)
    color = "#2ca02c"
    x = np.arange(len(dyns))
    bars = ax1.bar(x, success_pct, width=0.35, color=color, alpha=0.7, label="Success Rate", zorder=2)
    ax1.set_xlabel("Dynamic Obstacles Count", fontweight="bold")
    ax1.set_ylabel("Mission Success Rate (%)", color=color, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.set_ylim(0, 110)
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(d) for d in dyns])
    
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 2, f"{height:.1f}%",
                 ha="center", va="bottom", fontsize=9, color=color)
        
    # Collisions and Recoveries on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(x, recoveries, "-^", color="#ff7f0e", lw=2.0, label="Mean Recoveries", zorder=3)
    ax2.plot(x, [c * 8 for c in collisions], "-s", color="#d62728", lw=2.0, label="Total Collisions (out of 8)", zorder=4)
    ax2.set_ylabel("Stuck Recoveries / Total Collisions", color="black", fontweight="bold")
    ax2.tick_params(axis="y", labelcolor="black")
    ax2.set_ylim(0, 22)
    
    for i in range(len(dyns)):
        ax2.annotate(f"Rec={recoveries[i]:.1f}", (x[i], recoveries[i]), textcoords="offset points",
                     xytext=(-15,-15), ha="center", fontsize=8, color="#ff7f0e")
        ax2.annotate(f"Coll={int(collisions[i]*8)}", (x[i], collisions[i]*8), textcoords="offset points",
                     xytext=(15,10), ha="center", fontsize=8, color="#d62728")
        
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)
    
    plt.title("Dynamic-Obstacle Stress Axis: Success and Interaction Metrics", fontweight="bold")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig_stress_dynamic.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)

if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    plot_soc_stress()
    plot_dyn_stress()
