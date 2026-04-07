"""
Hidden_Markov_Pipeline.py
=========================
Static cinematic image of Hidden Markov Model regime detection.

Left panel  : 3D transition matrix bars  P(s_{t+1} | s_t)
Right panel : 4-stack 2D — Price+regimes · Returns barcode · Smoothed P(s) · Equity

Pipeline: SIMULATE -> FORWARD-BACKWARD -> RENDER STATIC PNG
Resolution: 1920x1080, Bloomberg Dark aesthetic

Dependencies: pip install numpy matplotlib
"""

import os, time, warnings
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════
CONFIG = {
    "RESOLUTION": (1920, 1080),
    "OUTPUT_FILE": "Hidden_Markov_Regime.png",
    "LOG_FILE": "hmm_pipeline.log",
    "VIEW_ELEV": 32,
    "VIEW_AZIM": 30,
}

# 3-state Gaussian HMM for market returns
HMM = {
    "n_states": 3,
    "names": ["Bull", "Bear", "Crisis"],
    "means": np.array([0.0009, -0.0006, -0.0020]),
    "stds":  np.array([0.0080,  0.0145,  0.0310]),
    "trans": np.array([
        [0.970, 0.025, 0.005],   # from Bull
        [0.040, 0.945, 0.015],   # from Bear
        [0.020, 0.080, 0.900],   # from Crisis
    ]),
    "init": np.array([0.7, 0.2, 0.1]),
    "T": 1000,        # days
    "S0": 100.0,
    "seed": 11,
}

# ─── THEME ────────────────────────────────────────────────────────
THEME = {
    "BG":        "#0b0b0b",
    "PANEL_BG":  "#0e0e0e",
    "GRID":      "#1a1a1a",
    "GRID_ALT":  "#1f1f1f",
    "SPINE":     "#333333",
    "TEXT":      "#ffffff",
    "TEXT_DIM":  "#888888",
    "TEXT_SEC":  "#c0c0c0",
    "CYAN":      "#00f2ff",
    "GREEN":     "#00ff41",
    "RED":       "#ff0055",
    "MAGENTA":   "#ff1493",
    "ORANGE":    "#ff9800",
    "BLUE":      "#00bfff",
    "YELLOW":    "#ffcc00",
    "FONT":      "DejaVu Sans",
}

# Regime colors: Bull / Bear / Crisis
STATE_COLORS = ["#00ff41", "#ff9800", "#ff0055"]


# ═══════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    try:
        with open(CONFIG["LOG_FILE"], "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _hex_to_rgba(hexc, alpha=1.0):
    h = hexc.lstrip("#")
    return (int(h[0:2], 16) / 255,
            int(h[2:4], 16) / 255,
            int(h[4:6], 16) / 255,
            alpha)


# ═══════════════════════════════════════════════════════════════════
# MODULE 1 — SIMULATION + INFERENCE
# ═══════════════════════════════════════════════════════════════════

def simulate_hmm():
    """Sample returns from a 3-state Gaussian HMM."""
    p = HMM
    rng = np.random.default_rng(p["seed"])
    n = p["T"]
    K = p["n_states"]

    states  = np.zeros(n, dtype=int)
    returns = np.zeros(n)
    prices  = np.zeros(n)

    states[0]  = rng.choice(K, p=p["init"])
    returns[0] = rng.normal(p["means"][states[0]], p["stds"][states[0]])
    prices[0]  = p["S0"]

    for t in range(1, n):
        states[t]  = rng.choice(K, p=p["trans"][states[t - 1]])
        returns[t] = rng.normal(p["means"][states[t]], p["stds"][states[t]])
        prices[t]  = prices[t - 1] * np.exp(returns[t])

    return states, returns, prices


def forward_backward(returns, means, stds, trans, init):
    """Smoothed posteriors  P(s_t | y_{1:T})  via Forward-Backward."""
    n = len(returns)
    K = len(means)

    # log emission probabilities (Gaussian)
    log_emit = np.zeros((n, K))
    for k in range(K):
        log_emit[:, k] = (-0.5 * np.log(2 * np.pi * stds[k] ** 2)
                          - (returns - means[k]) ** 2 / (2 * stds[k] ** 2))

    log_trans = np.log(trans + 1e-15)
    log_init  = np.log(init + 1e-15)

    # Forward
    log_alpha = np.full((n, K), -np.inf)
    log_alpha[0] = log_init + log_emit[0]
    for t in range(1, n):
        for j in range(K):
            log_alpha[t, j] = (np.logaddexp.reduce(log_alpha[t - 1] + log_trans[:, j])
                               + log_emit[t, j])

    # Backward
    log_beta = np.zeros((n, K))
    for t in range(n - 2, -1, -1):
        for i in range(K):
            log_beta[t, i] = np.logaddexp.reduce(
                log_trans[i] + log_emit[t + 1] + log_beta[t + 1])

    # Posterior
    log_gamma = log_alpha + log_beta
    log_gamma -= np.logaddexp.reduce(log_gamma, axis=1, keepdims=True)
    return np.exp(log_gamma)


def build_data():
    states, returns, prices = simulate_hmm()
    gamma = forward_backward(returns, HMM["means"], HMM["stds"],
                             HMM["trans"], HMM["init"])
    return dict(
        t=np.arange(HMM["T"]),
        states=states,
        returns=returns,
        prices=prices,
        gamma=gamma,
    )


# ═══════════════════════════════════════════════════════════════════
# MODULE 2 — RENDERING
# ═══════════════════════════════════════════════════════════════════

def _style_2d(ax, xlabel=False):
    ax.set_facecolor(THEME["PANEL_BG"])
    for sp in ax.spines.values():
        sp.set_color(THEME["SPINE"]); sp.set_linewidth(0.6)
    ax.tick_params(axis="both", colors=THEME["TEXT_DIM"], labelsize=9,
                   direction="in", length=4)
    ax.yaxis.grid(True, lw=0.3, alpha=0.4, color=THEME["GRID_ALT"])
    ax.xaxis.grid(False)
    if not xlabel:
        ax.tick_params(axis="x", labelbottom=False)


def render_static(sim, out_path):
    """Render the full-history static image."""
    K       = HMM["n_states"]
    trans   = HMM["trans"]
    names   = HMM["names"]
    states  = sim["states"]
    prices  = sim["prices"]
    rets    = sim["returns"]
    gamma   = sim["gamma"]
    t_arr   = sim["t"]

    cur_state = int(states[-1])
    alpha_3d  = 0.88

    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor=THEME["BG"])
    gs = gridspec.GridSpec(
        4, 2, width_ratios=[1, 1.15],
        height_ratios=[1.2, 1, 1, 1.2],
        hspace=0.30, wspace=0.28,
        left=0.05, right=0.97, top=0.86, bottom=0.06,
    )

    # ── LEFT: 3D transition matrix bars ───────────────────────────
    ax = fig.add_subplot(gs[:, 0], projection="3d", computed_zorder=False)
    ax.set_facecolor(THEME["BG"])
    pane = (0.043, 0.043, 0.043, 1)
    ax.xaxis.set_pane_color(pane)
    ax.yaxis.set_pane_color(pane)
    ax.zaxis.set_pane_color(pane)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a._axinfo["grid"]["color"] = (0.1, 0.1, 0.1, 0.5)
        a._axinfo["grid"]["linewidth"] = 0.4

    xpos, ypos = np.meshgrid(np.arange(K), np.arange(K), indexing="ij")
    xpos = xpos.flatten().astype(float)
    ypos = ypos.flatten().astype(float)
    zpos = np.zeros(K * K)
    dx = dy = 0.65
    xpos -= dx / 2
    ypos -= dy / 2
    dz = trans.flatten()

    bar_colors = []
    bar_edges  = []
    for i in range(K):
        for j in range(K):
            base = STATE_COLORS[i]
            a = alpha_3d if i == cur_state else alpha_3d * 0.40
            bar_colors.append(_hex_to_rgba(base, a))
            bar_edges.append(_hex_to_rgba(THEME["CYAN"], 0.55))

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz,
             color=bar_colors, edgecolor=bar_edges, linewidth=0.7,
             shade=True, zorder=1)

    for k in range(K * K):
        i, j = k // K, k % K
        label_alpha = 1.0 if i == cur_state else 0.55
        label_color = THEME["TEXT"] if i == cur_state else THEME["TEXT_DIM"]
        ax.text(xpos[k] + dx / 2, ypos[k] + dy / 2, dz[k] + 0.04,
                f"{dz[k]:.2f}",
                color=label_color, fontsize=9, ha="center", va="bottom",
                fontfamily=THEME["FONT"],
                fontweight="bold" if i == cur_state else "normal",
                alpha=label_alpha)

    ax.set_xticks(np.arange(K))
    ax.set_yticks(np.arange(K))
    ax.set_xticklabels(names, color=THEME["TEXT_DIM"], fontsize=10)
    ax.set_yticklabels(names, color=THEME["TEXT_DIM"], fontsize=10)
    ax.set_xlabel("From  $s_t$", fontsize=11, color=THEME["TEXT_DIM"], labelpad=14)
    ax.set_ylabel("To  $s_{t+1}$", fontsize=11, color=THEME["TEXT_DIM"], labelpad=14)
    ax.set_zlabel(r"$P(s_{t+1} \mid s_t)$", fontsize=11,
                  color=THEME["TEXT_DIM"], labelpad=8)
    ax.set_zlim(0, 1.15)
    ax.tick_params(axis="z", colors=THEME["TEXT_DIM"], labelsize=8)
    ax.view_init(elev=CONFIG["VIEW_ELEV"], azim=CONFIG["VIEW_AZIM"])
    ax.set_title("Transition Matrix", fontsize=14, fontweight="bold",
                 color=THEME["CYAN"], fontfamily=THEME["FONT"], pad=8)

    # ── RIGHT: 4 panels ───────────────────────────────────────────
    p_lo, p_hi = prices.min(), prices.max()
    p_pad = (p_hi - p_lo) * 0.06
    yr_price = (p_lo - p_pad, p_hi + p_pad)

    r_lo, r_hi = rets.min(), rets.max()
    r_pad = (r_hi - r_lo) * 0.10
    yr_returns = (r_lo - r_pad, r_hi + r_pad)

    eq = prices / HMM["S0"] * 100
    e_lo, e_hi = eq.min(), eq.max()
    e_pad = (e_hi - e_lo) * 0.06
    yr_equity = (e_lo - e_pad, e_hi + e_pad)

    # Panel 1 — Price + regime fill
    a1 = fig.add_subplot(gs[0, 1])
    _style_2d(a1)
    for s in range(K):
        mask = (states == s)
        a1.fill_between(t_arr, yr_price[0], yr_price[1],
                        where=mask, color=STATE_COLORS[s], alpha=0.13,
                        step="mid", linewidth=0)
    a1.plot(t_arr, prices, color=THEME["TEXT"], lw=1.0)
    a1.set_xlim(0, HMM["T"]); a1.set_ylim(*yr_price)
    a1.set_title("Price  +  HMM Regime", fontsize=11, fontweight="bold",
                 color=THEME["TEXT_SEC"], loc="left", pad=4)

    for i in range(K):
        a1.scatter([], [], s=40, color=STATE_COLORS[i], label=names[i])
    a1.legend(loc="upper left", fontsize=8, facecolor=THEME["BG"],
              edgecolor=THEME["SPINE"], labelcolor=THEME["TEXT_SEC"],
              framealpha=0.85, ncol=3, columnspacing=1.0,
              handletextpad=0.4)

    # Panel 2 — Daily returns barcode
    a2 = fig.add_subplot(gs[1, 1])
    _style_2d(a2)
    cols = [STATE_COLORS[s] for s in states]
    a2.vlines(t_arr, 0, rets, colors=cols, linewidth=0.7, alpha=0.85)
    a2.axhline(0, color=THEME["SPINE"], lw=0.5, alpha=0.5)
    a2.set_xlim(0, HMM["T"]); a2.set_ylim(*yr_returns)
    a2.set_title("Daily Returns", fontsize=11, fontweight="bold",
                 color=THEME["TEXT_SEC"], loc="left", pad=4)

    # Panel 3 — Smoothed state probabilities (stacked)
    a3 = fig.add_subplot(gs[2, 1])
    _style_2d(a3)
    a3.stackplot(t_arr, gamma[:, 0], gamma[:, 1], gamma[:, 2],
                 colors=[STATE_COLORS[i] for i in range(K)],
                 alpha=0.78, edgecolor="none")
    a3.set_xlim(0, HMM["T"]); a3.set_ylim(0, 1)
    a3.set_title(r"Smoothed State Probabilities  $P(s_t \mid y)$",
                 fontsize=11, fontweight="bold",
                 color=THEME["TEXT_SEC"], loc="left", pad=4)

    # Panel 4 — Equity (base 100)
    a4 = fig.add_subplot(gs[3, 1])
    _style_2d(a4, xlabel=True)
    a4.plot(t_arr, eq, color=THEME["CYAN"], lw=1.1)
    a4.fill_between(t_arr, 100, eq, where=eq >= 100,
                    color=THEME["CYAN"], alpha=0.12)
    a4.fill_between(t_arr, 100, eq, where=eq < 100,
                    color=THEME["RED"], alpha=0.12)
    a4.axhline(100, color=THEME["SPINE"], lw=0.5, alpha=0.5)
    a4.set_xlim(0, HMM["T"]); a4.set_ylim(*yr_equity)
    a4.set_xlabel("Time [days]", fontsize=10, color=THEME["TEXT_DIM"],
                   fontfamily=THEME["FONT"])
    a4.set_title("Equity  (base 100)", fontsize=11, fontweight="bold",
                 color=THEME["TEXT_SEC"], loc="left", pad=4)

    # ── Title bar ─────────────────────────────────────────────────
    fig.text(0.50, 0.955, "Hidden Markov Regime Detection",
             ha="center", va="center", fontsize=24, fontweight="bold",
             color=THEME["TEXT"], fontfamily=THEME["FONT"])
    fig.text(0.50, 0.916,
             r"$y_t \mid s_t \sim \mathcal{N}(\mu_{s_t},\, \sigma_{s_t}^2)$"
             "          |          "
             r"$P(s_t \mid y) \propto \alpha_t(s_t)\, \beta_t(s_t)$",
             ha="center", va="center", fontsize=13,
             color=THEME["TEXT_DIM"], fontfamily=THEME["FONT"])
    fig.text(0.50, 0.891,
             r"3 regimes:    Bull  $\mu = +0.09\%$    Bear  $\mu = -0.06\%$"
             r"    Crisis  $\mu = -0.20\%$",
             ha="center", va="center", fontsize=10,
             color="#555555", fontfamily=THEME["FONT"])

    # ── HUD ───────────────────────────────────────────────────────
    eq_now = eq[-1]
    fig.text(0.97, 0.875,
             f"day {len(t_arr) - 1:4d}    regime: {names[cur_state].upper():7s}"
             f"    eq = {eq_now:6.1f}",
             ha="right", va="center", fontsize=11, fontweight="bold",
             color=STATE_COLORS[cur_state], fontfamily=THEME["FONT"], alpha=0.9)

    # ── Footer ────────────────────────────────────────────────────
    fig.text(0.98, 0.012, "@quant.traderr",
             ha="right", va="bottom", fontsize=10,
             color=THEME["TEXT_DIM"], fontfamily=THEME["FONT"], alpha=0.55)

    fig.savefig(out_path, dpi=100, facecolor=THEME["BG"])
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    log("=" * 60)
    log("Hidden Markov Regime Pipeline START (static)")
    log("=" * 60)

    log("Simulating 3-state HMM ...")
    sim = build_data()
    counts = np.bincount(sim["states"], minlength=HMM["n_states"])
    log(f"  T={HMM['T']} days  |  state counts: "
        f"Bull={counts[0]}  Bear={counts[1]}  Crisis={counts[2]}")
    log(f"  Final equity: {sim['prices'][-1] / HMM['S0'] * 100:.2f}")

    log("Rendering static image ...")
    render_static(sim, CONFIG["OUTPUT_FILE"])
    log(f"Image saved: {CONFIG['OUTPUT_FILE']}")
    log(f"Pipeline complete in {time.time() - t0:.1f}s")
    log("=" * 60)


if __name__ == "__main__":
    main()
