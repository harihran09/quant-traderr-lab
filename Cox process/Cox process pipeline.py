"""
Cox_Process_Pipeline.py
=======================
Static 3D visualization of a Doubly Stochastic Poisson Process (Cox process).

The intensity is itself stochastic — we drive it with a CIR (Cox-Ingersoll-
Ross) SDE, then conditional on the path, events arrive as an inhomogeneous
Poisson process:

    dλ_t = κ (θ - λ_t) dt + σ_v √λ_t  dW_t        (CIR intensity)
    N_t  ~ Poisson( ∫_0^t λ_s ds )                  (conditional point process)

This generalises Hawkes (where intensity is endogenous and self-exciting):
in a Cox process the randomness in the intensity is exogenous.

Left panel  : 3D ribbon of N intensity paths λ_t  + event dots scattered on top
Right panel : 4-stack 2D of one highlighted realisation
              (intensity · event marks · counting process N_t · inter-arrival hist)

NOTE: Video rendering has been removed for pipeline efficiency.

Dependencies: pip install numpy matplotlib
"""

import os, time, warnings
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════
CONFIG = {
    "RESOLUTION": (1920, 1080),
    "OUTPUT_IMAGE": "Cox_Process_Output.png",
    "LOG_FILE": "cox_pipeline.log",
}

# CIR-driven Cox process parameters
COX = {
    "n_paths":   60,
    "T":         5.0,
    "dt":        0.005,        # 1000 time steps
    # CIR intensity (Feller condition: 2*kappa*theta > sigma^2 → 6 > 1.96 ✓)
    "kappa":     1.5,
    "theta":     2.0,
    "sigma_v":   1.4,
    "lambda0":   0.6,
    # Highlighted realisation (the one shown in 2D detail)
    "highlight": 17,
    "seed":      11,
}

# ─── THEME ────────────────────────────────────────────────────────
THEME = {
    "BG":         "#000000",
    "PANEL_BG":   "#0a0a0a",
    "GRID":       "#222222",
    "GRID_ALT":   "#1f1f1f",
    "SPINE":      "#333333",
    "TEXT":       "#ffffff",
    "TEXT_DIM":   "#aaaaaa",
    "TEXT_SEC":   "#c0c0c0",
    "ORANGE":     "#ff9500",
    "ORANGE_HOT": "#ff6b00",
    "YELLOW":     "#ffd400",
    "CYAN":       "#00f2ff",
    "GREEN":      "#00ff7f",
    "RED":        "#ff3050",
    "PINK":       "#ff2a9e",
    "BLUE":       "#00bfff",
    "FONT":       "Arial",
}

CMAP = cm.get_cmap("viridis")


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


# ═══════════════════════════════════════════════════════════════════
# MODULE 1 — SIMULATE INTENSITY + EVENTS
# ═══════════════════════════════════════════════════════════════════

def simulate_cir_intensities():
    """N independent CIR paths via full-truncation Euler."""
    p = COX
    rng = np.random.default_rng(p["seed"])
    n_steps = int(p["T"] / p["dt"])
    n_total = n_steps + 1
    N = p["n_paths"]

    L = np.zeros((N, n_total))
    L[:, 0] = p["lambda0"]
    for i in range(n_steps):
        L_plus = np.maximum(L[:, i], 0.0)
        dW = rng.standard_normal(N) * np.sqrt(p["dt"])
        L[:, i + 1] = L[:, i] + p["kappa"] * (p["theta"] - L_plus) * p["dt"] \
                      + p["sigma_v"] * np.sqrt(L_plus) * dW
    L = np.maximum(L, 1e-6)
    t_grid = np.linspace(0.0, p["T"], n_total)
    return t_grid, L


def simulate_events(t_grid, L):
    """Inhomogeneous Poisson via thinning under each intensity path."""
    p = COX
    rng = np.random.default_rng(p["seed"] + 19)
    N, n_total = L.shape
    dt = p["dt"]
    L_max = float(L.max() * 1.1)

    events_per_path = []          # list of arrays of event times
    events_xyz = []               # for 3D scatter (t, path_idx, lambda_at_event)
    counting = np.zeros((N, n_total), dtype=int)

    for k in range(N):
        # candidate times via Poisson(L_max)
        T_total = p["T"]
        n_candidates = rng.poisson(L_max * T_total)
        if n_candidates == 0:
            events_per_path.append(np.array([]))
            continue
        cand_times = np.sort(rng.uniform(0, T_total, size=n_candidates))
        # accept with prob L(t)/L_max
        idx_grid = np.minimum((cand_times / dt).astype(int), n_total - 1)
        L_at = L[k, idx_grid]
        accept = rng.random(n_candidates) < (L_at / L_max)
        ev = cand_times[accept]
        events_per_path.append(ev)
        # build counting process
        for tev in ev:
            j = min(int(tev / dt), n_total - 1)
            counting[k, j:] += 1
        # store for 3D scatter
        for tev in ev:
            j = min(int(tev / dt), n_total - 1)
            events_xyz.append((tev, k, L[k, j]))

    if events_xyz:
        events_xyz = np.array(events_xyz)
    else:
        events_xyz = np.zeros((0, 3))

    return events_per_path, events_xyz, counting


# ═══════════════════════════════════════════════════════════════════
# MODULE 2 — RENDERING
# ═══════════════════════════════════════════════════════════════════

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


def render_static(data):
    try:
        fp = CONFIG["OUTPUT_IMAGE"]

        elev = 34
        azim = 260
        dist = 1.15
        salpha = 0.95

        t_grid    = data["t_grid"]
        L         = data["L"]
        ev_xyz    = data["ev_xyz"]
        events    = data["events_per_path"]
        counting  = data["counting"]
        yr        = data["yr"]

        N, n_total = L.shape
        n_visible = n_total
        cur_t = t_grid[n_visible - 1]
        sl = slice(0, n_visible)

        h = COX["highlight"]

        fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor=THEME["BG"])
        gs = gridspec.GridSpec(
            4, 2, width_ratios=[1.15, 1],
            height_ratios=[1.2, 1, 1, 1.2],
            hspace=0.30, wspace=0.24,
            left=0.04, right=0.97, top=0.86, bottom=0.06,
        )

        # ═══ LEFT: 3D ribbon of intensity paths ═══════════════════
        ax = fig.add_subplot(gs[:, 0], projection="3d", computed_zorder=False)
        ax.set_facecolor(THEME["BG"])
        pane = (0.02, 0.02, 0.02, 1)
        ax.xaxis.set_pane_color(pane)
        ax.yaxis.set_pane_color(pane)
        ax.zaxis.set_pane_color(pane)
        for a in (ax.xaxis, ax.yaxis, ax.zaxis):
            a._axinfo["grid"]["color"]     = (0.13, 0.13, 0.13, 0.55)
            a._axinfo["grid"]["linewidth"] = 0.4

        # color paths by their average intensity (over visible portion)
        path_avg = L[:, :n_visible].mean(axis=1)
        v_min = float(L.min())
        v_max = float(L.max())
        norm_avg = (path_avg - v_min) / max(v_max - v_min, 1e-6)

        # plot all non-highlighted paths
        for k in range(N):
            if k == h:
                continue
            color = CMAP(norm_avg[k])
            color = (color[0], color[1], color[2], 0.55 * salpha)
            ax.plot(t_grid[sl], np.full(n_visible, k, dtype=float), L[k, sl],
                    color=color, lw=0.7, zorder=2)

        # highlighted path — bright cyan, bold
        ax.plot(t_grid[sl], np.full(n_visible, h, dtype=float), L[h, sl],
                color=THEME["CYAN"], lw=2.4, alpha=1.0, zorder=14)

        # mean-reversion floor θ as a faint horizontal plane line
        for k in (0, N // 2, N - 1):
            ax.plot([0, t_grid[-1]], [k, k], [COX["theta"], COX["theta"]],
                    color=THEME["YELLOW"], lw=0.6, alpha=0.25, ls="--", zorder=1)

        # event dots in 3D — only the ones at t <= cur_t
        if len(ev_xyz):
            mask_ev = ev_xyz[:, 0] <= cur_t
            if mask_ev.any():
                ev_vis = ev_xyz[mask_ev]
                # highlighted path's events drawn slightly bigger / orange
                hi_mask = ev_vis[:, 1] == h
                lo_mask = ~hi_mask
                if lo_mask.any():
                    ax.scatter(ev_vis[lo_mask, 0], ev_vis[lo_mask, 1],
                               ev_vis[lo_mask, 2],
                               s=10, color=THEME["RED"], alpha=0.65 * salpha,
                               edgecolors="none", zorder=10, depthshade=False)
                if hi_mask.any():
                    ax.scatter(ev_vis[hi_mask, 0], ev_vis[hi_mask, 1],
                               ev_vis[hi_mask, 2],
                               s=42, color=THEME["ORANGE_HOT"],
                               edgecolors="white", linewidths=0.6,
                               zorder=15, depthshade=False)

        ax.set_xlabel("TIME  t", fontsize=12, fontweight="bold",
                      color=THEME["TEXT_DIM"], labelpad=14, fontfamily=THEME["FONT"])
        ax.set_ylabel("PATH #", fontsize=12, fontweight="bold",
                      color=THEME["TEXT_DIM"], labelpad=14, fontfamily=THEME["FONT"])
        ax.set_zlabel(r"INTENSITY  $\lambda_t$", fontsize=12, fontweight="bold",
                      color=THEME["TEXT_DIM"], labelpad=10, fontfamily=THEME["FONT"])
        ax.tick_params(axis="both", colors=THEME["TEXT_DIM"], labelsize=8)

        ax.set_xlim(0, COX["T"])
        ax.set_ylim(0, N - 1)
        ax.set_zlim(0, max(L.max() * 1.05, COX["theta"] * 1.5))
        ax.set_box_aspect([1.55 * dist, 1.0 * dist, 0.85 * dist])
        ax.view_init(elev=elev, azim=azim)
        ax.set_title("INTENSITY RIBBON  +  EVENT DOTS",
                     fontsize=13, fontweight="bold",
                     color=THEME["YELLOW"], fontfamily=THEME["FONT"], pad=8)

        # ═══ RIGHT: 4 panels (highlighted realisation) ════════════
        L_hi  = L[h]
        ev_hi = events[h]
        N_hi  = counting[h]

        # Panel 1 — λ_t with band of all paths
        a1 = fig.add_subplot(gs[0, 1])
        _style_2d(a1)
        L_lo = np.percentile(L, 5, axis=0)
        L_md = np.percentile(L, 50, axis=0)
        L_hh = np.percentile(L, 95, axis=0)
        a1.fill_between(t_grid[sl], L_lo[sl], L_hh[sl],
                        color=THEME["GREEN"], alpha=0.10, linewidth=0)
        a1.plot(t_grid[sl], L_md[sl], color=THEME["TEXT_DIM"],
                lw=0.8, ls=":", label="median")
        a1.plot(t_grid[sl], L_hi[sl], color=THEME["CYAN"], lw=1.3,
                label=r"highlighted  $\lambda_t$")
        a1.axhline(COX["theta"], color=THEME["YELLOW"], lw=0.7,
                   ls="--", alpha=0.85, label=r"$\theta$")
        a1.set_xlim(0, COX["T"]); a1.set_ylim(*yr["intensity"])
        a1.legend(loc="upper right", fontsize=8, facecolor=THEME["BG"],
                  edgecolor=THEME["SPINE"], labelcolor=THEME["TEXT_SEC"],
                  framealpha=0.85, ncol=3)
        a1.set_title(r"Stochastic Intensity  $\lambda_t$  (CIR)",
                     fontsize=11, fontweight="bold",
                     color=THEME["TEXT_SEC"], loc="left", pad=4)

        # Panel 2 — Event marks (rug-style vlines)
        a2 = fig.add_subplot(gs[1, 1])
        _style_2d(a2)
        ev_visible = ev_hi[ev_hi <= cur_t] if len(ev_hi) else ev_hi
        if len(ev_visible):
            a2.vlines(ev_visible, 0, 1,
                      colors=THEME["ORANGE_HOT"], linewidths=1.4)
        a2.set_xlim(0, COX["T"]); a2.set_ylim(0, 1.1)
        a2.set_yticks([])
        a2.set_title(r"Conditional Poisson events  (highlighted path)",
                     fontsize=11, fontweight="bold",
                     color=THEME["TEXT_SEC"], loc="left", pad=4)

        # Panel 3 — Counting process N(t)
        a3 = fig.add_subplot(gs[2, 1])
        _style_2d(a3)
        a3.step(t_grid[sl], N_hi[sl], where="post",
                color=THEME["BLUE"], lw=1.2)
        a3.fill_between(t_grid[sl], 0, N_hi[sl], step="post",
                        color=THEME["BLUE"], alpha=0.12)
        a3.set_xlim(0, COX["T"]); a3.set_ylim(0, yr["count"])
        a3.set_title(r"Counting process  $N_t$",
                     fontsize=11, fontweight="bold",
                     color=THEME["TEXT_SEC"], loc="left", pad=4)

        # Panel 4 — Inter-arrival distribution (across ALL paths up to cur_t)
        a4 = fig.add_subplot(gs[3, 1])
        _style_2d(a4, xlabel=True)
        all_iat = []
        for ev in events:
            if len(ev) >= 2:
                ev_v = ev[ev <= cur_t]
                if len(ev_v) >= 2:
                    all_iat.extend(np.diff(ev_v).tolist())
        all_iat = np.array(all_iat)
        if len(all_iat) > 5:
            a4.hist(all_iat, bins=30, range=(0, yr["iat"]),
                    color=THEME["PINK"], alpha=0.65,
                    edgecolor=THEME["SPINE"], linewidth=0.4, density=True)
            # marginal exp(theta) reference (for comparison)
            x_ref = np.linspace(0, yr["iat"], 200)
            a4.plot(x_ref, COX["theta"] * np.exp(-COX["theta"] * x_ref),
                    color=THEME["YELLOW"], lw=1.4,
                    label=r"Exp$(\theta)$ ref")
            a4.legend(loc="upper right", fontsize=8, facecolor=THEME["BG"],
                      edgecolor=THEME["SPINE"], labelcolor=THEME["TEXT_SEC"],
                      framealpha=0.85)
        a4.set_xlim(0, yr["iat"])
        a4.set_xlabel(r"$\Delta t$  between consecutive events",
                       fontsize=10, color=THEME["TEXT_DIM"],
                       fontfamily=THEME["FONT"])
        a4.set_title("Inter-arrival distribution  (all paths)",
                     fontsize=11, fontweight="bold",
                     color=THEME["TEXT_SEC"], loc="left", pad=4)

        # ═══ Title bar ════════════════════════════════════════════
        fig.text(0.50, 0.955, "DOUBLY STOCHASTIC POISSON  /  COX PROCESS",
                 ha="center", va="center", fontsize=24, fontweight="bold",
                 color=THEME["ORANGE"], fontfamily=THEME["FONT"])
        fig.text(0.50, 0.918,
                 r"$d\lambda_t = \kappa(\theta - \lambda_t)\,dt"
                 r" + \sigma_v\,\sqrt{\lambda_t}\,dW_t$"
                 "          "
                 r"$N_t \;\sim\; \mathrm{Poisson}\!\left(\, \int_0^t \lambda_s\,ds \,\right)$",
                 ha="center", va="center", fontsize=13,
                 color=THEME["TEXT"], fontfamily=THEME["FONT"])
        fig.text(0.50, 0.886,
                 r"CIR INTENSITY    "
                 r"$\kappa = 1.5$    $\theta = 2.0$    "
                 r"$\sigma_v = 1.4$    "
                 r"$\lambda_0 = 0.6$    "
                 r"60 paths   $T = 5$",
                 ha="center", va="center", fontsize=10, fontweight="bold",
                 color=THEME["TEXT_DIM"], fontfamily=THEME["FONT"])

        # ═══ HUD ══════════════════════════════════════════════════
        n_ev_hi = int(np.sum(ev_hi <= cur_t))
        total_events = int(sum(np.sum(ev <= cur_t) for ev in events))
        fig.text(0.97, 0.875,
                 f"t = {cur_t:4.2f}    "
                 f"highlighted N_t = {n_ev_hi:3d}    "
                 f"total events (all paths) = {total_events:4d}",
                 ha="right", va="center", fontsize=11, fontweight="bold",
                 color=THEME["YELLOW"], fontfamily=THEME["FONT"])

        # ═══ Footer ═══════════════════════════════════════════════
        fig.text(0.98, 0.012, "@quant.traderr",
                 ha="right", va="bottom", fontsize=10,
                 color=THEME["TEXT_DIM"], fontfamily=THEME["FONT"], alpha=0.6)

        fig.savefig(fp, dpi=100, facecolor=THEME["BG"])
        plt.close(fig)
        log(f"Static visualization saved: {fp}")

        return True

    except Exception:
        import traceback
        traceback.print_exc()
        return False


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    log("=" * 60)
    log("Cox Process Pipeline START")
    log("=" * 60)

    log("Simulating CIR intensity paths ...")
    t_grid, L = simulate_cir_intensities()
    log(f"  N = {L.shape[0]}  steps = {L.shape[1] - 1}  "
        f"lambda range = [{L.min():.3f}, {L.max():.3f}]")

    log("Sampling conditional Poisson events ...")
    events_per_path, ev_xyz, counting = simulate_events(t_grid, L)
    n_total_events = sum(len(e) for e in events_per_path)
    log(f"  total events = {n_total_events}  "
        f"(avg {n_total_events / L.shape[0]:.1f} per path)")
    log(f"  highlighted path #{COX['highlight']} has "
        f"{len(events_per_path[COX['highlight']])} events")

    yr = {
        "intensity": (0, max(L.max() * 1.05, COX["theta"] * 1.6)),
        "count":     int(counting.max() * 1.05) + 2,
        "iat":       1.0,
    }

    data_payload = dict(t_grid=t_grid, L=L, ev_xyz=ev_xyz,
                        events_per_path=events_per_path, counting=counting,
                        yr=yr)
    
    log("Rendering static visualization ...")
    render_static(data_payload)

    log(f"Pipeline complete in {time.time() - t0:.1f}s")
    log("=" * 60)


if __name__ == "__main__":
    main()
