"""
Ergodicity_Pipeline.py
======================
Project: Quant Trader Lab - Ergodicity Economics Visualization
Author: quant.traderr (Instagram)
License: MIT

Description:
    Visualizes the core insight of Ergodicity Economics: for multiplicative
    dynamics, the ensemble average (across N realizations) diverges from
    the time average (single trajectory). As N grows, the ensemble average
    converges to the theoretical expected value — but NO single agent
    actually experiences that growth.

    Simulation: x(t+1) = x(t) * r(t), coin-flip multiplicative gamble
    (x1.5 or x0.6 each step). Positive expected value, negative median.

    Pipeline Steps:
    1. DATA — Simulate multiplicative random walks for N in {1, 100, 10000, 1000000}.
    2. VISUALIZATION — Static dual-panel snapshot (linear + log scale).

    NOTE: Video rendering has been removed for pipeline efficiency.

Dependencies:
    pip install numpy matplotlib

Usage:
    python Ergodicity_Pipeline.py
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# --- CONFIGURATION ---

CONFIG = {
    "T": 50,                        # Total time steps
    "N_VALUES": [1, 100, 10_000, 1_000_000],
    "UP_MULT": 1.5,                 # Win multiplier
    "DOWN_MULT": 0.6,               # Loss multiplier
    "X0": 1.0,                      # Starting wealth
    "OUTPUT_IMAGE": "ergodicity_static.png",
}

THEME = {
    "BG": "#0b0b0b",
    "PANEL_BG": "#0e0e0e",
    "GRID": "#1f1f1f",
    "TEXT": "#ffffff",
    "TEXT_MUTED": "#aaaaaa",
    "COLORS": {
        1:          "#00bfff",      # Electric Blue  — N=1
        100:        "#00ff41",      # Neon Green     — N=100
        10_000:     "#ff3333",      # Red            — N=10,000
        1_000_000:  "#ffffff",      # White          — N=1,000,000
    },
    "LINEWIDTHS": {
        1:          1.8,
        100:        1.6,
        10_000:     1.4,
        1_000_000:  2.0,
    },
}

# --- UTILS ---

def log(msg):
    """Simple logger."""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}")

# --- MODULE 1: DATA ---

def simulate_ensemble_averages(seed=42):
    """
    For each N in CONFIG['N_VALUES'], simulate N multiplicative random walks
    of length T and compute the ensemble average <x(t)>_N at each step.
    """
    log("[Data] Simulating multiplicative random walks...")
    rng = np.random.default_rng(seed)
    T = CONFIG["T"]
    up = CONFIG["UP_MULT"]
    down = CONFIG["DOWN_MULT"]

    results = {}

    for N in CONFIG["N_VALUES"]:
        log(f"[Data]   N={N:>10,d} — simulating...")
        flips = rng.choice([up, down], size=(N, T))
        cum_prod = np.cumprod(flips, axis=1)
        paths = np.hstack([np.ones((N, 1)), cum_prod])
        ensemble_avg = paths.mean(axis=0)
        results[N] = ensemble_avg
        log(f"[Data]   N={N:>10,d} — final <x(T)> = {ensemble_avg[-1]:.4f}")

    log("[Data] Simulation complete.")
    return results

# --- MODULE 2: STATIC VISUALIZATION ---

def visualize(data):
    """Generates a static dual-panel snapshot (linear + log scale)."""
    log("[Visual] Generating static snapshot...")

    T = CONFIG["T"]
    t_range = np.arange(T + 1)

    fig, (ax_lin, ax_log) = plt.subplots(
        1, 2, figsize=(19.2, 10.8), dpi=100, facecolor=THEME["BG"],
    )

    for ax, is_log, label in [(ax_lin, False, "(A)"), (ax_log, True, "(B)")]:
        ax.set_facecolor(THEME["PANEL_BG"])

        # Plot each N series
        for N in CONFIG["N_VALUES"]:
            y = data[N]
            color = THEME["COLORS"][N]
            lw = THEME["LINEWIDTHS"][N]
            n_label = f"N={N:,}"
            ax.step(t_range, y, where="post", color=color, linewidth=lw,
                    label=n_label, alpha=0.95, zorder=3 if N == 1_000_000 else 2)

        # Axis config
        if is_log:
            ax.set_yscale("log")
            ax.set_ylim(0.08, 15)
            ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.yaxis.get_major_formatter().set_scientific(False)
            ax.set_yticks([0.1, 1, 10])
            ax.set_yticklabels(["10\u207B\u00B9", "10\u2070", "10\u00B9"])
        else:
            ax.set_ylim(-0.3, 10.5)
            ax.set_yticks(range(0, 11))

        ax.set_xlim(0, T + 1)
        ax.set_xticks(range(0, T + 5, 5))
        ax.set_xlabel("t", color=THEME["TEXT"], fontsize=13)
        ax.set_ylabel(r"$\langle x(t) \rangle_N$", color=THEME["TEXT"], fontsize=13)

        # Grid
        ax.grid(True, color=THEME["GRID"], linewidth=0.5, alpha=0.7)
        ax.tick_params(colors=THEME["TEXT_MUTED"], labelsize=10)
        for spine in ax.spines.values():
            spine.set_color(THEME["GRID"])

        # Legend
        ax.legend(loc="upper left", fontsize=9, facecolor="#111111", edgecolor=THEME["GRID"],
                  labelcolor=THEME["TEXT"], framealpha=0.9)

        # Panel label
        ax.text(0.5, -0.10, label, transform=ax.transAxes, fontsize=14,
                ha="center", va="top", color=THEME["TEXT_MUTED"])

    # Title
    fig.suptitle(
        "ERGODICITY ECONOMICS  \u2014  Ensemble Average vs Time Average",
        fontsize=16, color=THEME["TEXT"], fontweight="bold", y=0.96
    )

    # Subtitle
    fig.text(
        0.5, 0.915,
        f"Multiplicative gamble: \u00D7{CONFIG['UP_MULT']} or \u00D7{CONFIG['DOWN_MULT']}  |  "
        f"E[r] = {(CONFIG['UP_MULT'] + CONFIG['DOWN_MULT']) / 2:.2f}  |  "
        f"T = {T} steps",
        fontsize=11, color=THEME["TEXT_MUTED"], ha="center",
    )

    # Watermark
    fig.text(0.5, 0.02, "@quant.traderr", fontsize=10, color="#333333",
             ha="center", style="italic")

    plt.subplots_adjust(left=0.06, right=0.97, top=0.87, bottom=0.13, wspace=0.22)

    # Save
    out_path = os.path.join(os.path.dirname(__file__), CONFIG["OUTPUT_IMAGE"])
    plt.savefig(out_path, facecolor=THEME["BG"], dpi=100)
    plt.close(fig)
    log(f"[Visual] Saved to {out_path}")

# --- MAIN ---

def main():
    log("=== ERGODICITY ECONOMICS PIPELINE ===")

    # 1. Simulate
    data = simulate_ensemble_averages()

    # 2. Visualize
    visualize(data)

    log("=== PIPELINE FINISHED ===")

if __name__ == "__main__":
    main()
