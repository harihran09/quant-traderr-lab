"""
GAN_Pipeline.py
===============
Project : Quant Trader Lab - Synthetic Market Data via GANs
Author  : quant.traderr (Instagram)
License : MIT

Description
-----------
A production-ready pipeline for training a Generative Adversarial Network
to synthesize realistic financial return sequences, visualized as a
cinematic timelapse.

The GAN learns the distribution of real BTC-USD log returns using a pure
NumPy + autograd implementation (no PyTorch/TensorFlow).

Pipeline Steps
--------------
    1. **Data Acquisition** : Fetches BTC-USD via yfinance, computes log returns.
    2. **GAN Engine**       : Trains G(z)->returns and D(x)->real/fake using
                              autograd for backprop and a manual Adam optimizer.
    3. **Rendering**        : Single static matplotlib frame (final state).

Dependencies
------------
    pip install numpy pandas matplotlib yfinance autograd
"""

import os
import sys
import time
import warnings
import numpy as np
import autograd.numpy as anp
from autograd import grad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    # -- Data --
    "TICKER":         "BTC-USD",
    "PERIOD":         "1y",
    "INTERVAL":       "1d",

    # -- GAN Architecture --
    "Z_DIM":          16,
    "SEQ_LEN":        50,
    "G_HIDDEN":       [64, 128],
    "D_HIDDEN":       [128, 64],
    "BATCH_SIZE":     64,

    # -- Training --
    "EPOCHS":         3000,
    "SNAPSHOT_EVERY": 15,
    "LR_G":           2e-4,
    "LR_D":           2e-4,
    "BETA1":          0.5,
    "BETA2":          0.999,
    "ADAM_EPS":       1e-8,
    "SEED":           42,

    # -- Visualization --
    "N_GEN_PATHS":    30,
    "HIST_BINS":      50,

    # -- Output --
    "RESOLUTION":     (1920, 1080),
    "OUTPUT_FILE":    os.path.join(os.path.dirname(__file__), "GAN_Output.png"),
    "LOG_FILE":       os.path.join(os.path.dirname(__file__), "gan_pipeline.log"),
}

THEME = {
    "BG":        "#0e0e0e",
    "GRID":      "#1f1f1f",
    "SPINE":     "#333333",
    "TEXT":      "#c0c0c0",
    "WHITE":     "#ffffff",
    "FONT":      "DejaVu Sans",

    # Data colors (Hawkes palette)
    "REAL_PATH": "#ffffff",
    "GEN_PATH":  "#00d4ff",
    "GEN_FILL":  "#00d4ff",
    "REAL_HIST": "#ffffff",
    "G_LOSS":    "#00d4ff",
    "D_LOSS":    "#ff9800",
    "EPOCH_MKR": "#ff1493",
}

# =============================================================================
# UTILS
# =============================================================================

def log(msg):
    """Timestamped console + file logger."""
    timestamp = time.strftime("%H:%M:%S")
    formatted = f"[{timestamp}] {msg}"
    try:
        print(formatted)
    except UnicodeEncodeError:
        print(formatted.encode("ascii", errors="replace").decode())
    try:
        with open(CONFIG["LOG_FILE"], "a", encoding="utf-8") as f:
            f.write(formatted + "\n")
    except Exception:
        pass


# =============================================================================
# ADAM OPTIMIZER (manual implementation for autograd)
# =============================================================================

class AdamOptimizer:
    """Manual Adam optimizer for lists of numpy parameter arrays."""

    def __init__(self, params, lr=2e-4, beta1=0.5, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]

    def step(self, params, grads):
        self.t += 1
        updated = []
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (g ** 2)
            m_hat = self.m[i] / (1.0 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1.0 - self.beta2 ** self.t)
            updated.append(p - self.lr * m_hat / (np.sqrt(v_hat) + self.eps))
        return updated


# =============================================================================
# GAN NETWORK FUNCTIONS (functional style for autograd)
# =============================================================================

def leaky_relu(x, alpha=0.01):
    return anp.maximum(alpha * x, x)

def relu(x):
    return anp.maximum(0.0, x)

def generator_forward(g_params, z):
    """G: z(Z_DIM) -> Dense(64,ReLU) -> Dense(128,ReLU) -> Dense(SEQ_LEN,tanh)"""
    W1, b1, W2, b2, W3, b3 = g_params
    h = relu(anp.dot(z, W1) + b1)
    h = relu(anp.dot(h, W2) + b2)
    return anp.tanh(anp.dot(h, W3) + b3)

def discriminator_forward(d_params, x):
    """D: x(SEQ_LEN) -> Dense(128,LReLU) -> Dense(64,LReLU) -> Dense(1,sigmoid)"""
    W1, b1, W2, b2, W3, b3 = d_params
    h = leaky_relu(anp.dot(x, W1) + b1)
    h = leaky_relu(anp.dot(h, W2) + b2)
    logit = anp.dot(h, W3) + b3
    return 1.0 / (1.0 + anp.exp(-anp.clip(logit, -20, 20)))

def d_loss_fn(d_params, g_params, real_batch, noise):
    """D loss: -E[log D(x)] - E[log(1 - D(G(z)))]"""
    d_real = discriminator_forward(d_params, real_batch)
    fake = generator_forward(g_params, noise)
    d_fake = discriminator_forward(d_params, fake)
    return -anp.mean(anp.log(d_real + 1e-8) + anp.log(1.0 - d_fake + 1e-8))

def g_loss_fn(g_params, d_params, noise):
    """G loss: -E[log D(G(z))]  (non-saturating)"""
    fake = generator_forward(g_params, noise)
    d_fake = discriminator_forward(d_params, fake)
    return -anp.mean(anp.log(d_fake + 1e-8))


# =============================================================================
# MODULE 1 : DATA + GAN TRAINING
# =============================================================================

def fetch_real_returns():
    """Fetch BTC-USD and compute log returns."""
    import yfinance as yf
    import pandas as pd

    log(f"[Data] Fetching {CONFIG['TICKER']} ({CONFIG['PERIOD']})...")

    try:
        df = yf.download(CONFIG["TICKER"], period=CONFIG["PERIOD"],
                         interval=CONFIG["INTERVAL"], progress=False)
    except Exception as e:
        log(f"[Error] YF Download failed: {e}. Using synthetic fallback.")
        return _synthetic_fallback()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
        log("[Data] Flattened MultiIndex columns.")

    if df.empty:
        log("[Warning] Empty dataframe. Using synthetic fallback.")
        return _synthetic_fallback()

    price = df["Close"].values.flatten().astype(float)
    log_returns = np.diff(np.log(price + 1e-9))
    log_returns = log_returns[~np.isnan(log_returns)]

    log(f"[Data] {len(log_returns)} log returns | "
        f"mean={np.mean(log_returns):.6f}, std={np.std(log_returns):.6f}")
    return log_returns, price


def _synthetic_fallback():
    """Generate synthetic returns if yfinance fails."""
    log("[Data] Generating synthetic returns (fat-tail mixture)...")
    np.random.seed(CONFIG["SEED"])
    n = 300
    normal = np.random.normal(0.0005, 0.02, int(n * 0.9))
    heavy = np.random.standard_t(3, int(n * 0.1)) * 0.03
    returns = np.concatenate([normal, heavy])
    np.random.shuffle(returns)
    price = 100.0 * np.exp(np.concatenate([[0.0], np.cumsum(returns)]))
    return returns, price


def init_gan_params():
    """Initialize Generator and Discriminator weights (He/Xavier init)."""
    np.random.seed(CONFIG["SEED"])
    z_dim = CONFIG["Z_DIM"]
    seq_len = CONFIG["SEQ_LEN"]
    g_h = CONFIG["G_HIDDEN"]
    d_h = CONFIG["D_HIDDEN"]

    # Generator: z_dim -> g_h[0] -> g_h[1] -> seq_len
    gW1 = np.random.randn(z_dim, g_h[0])   * np.sqrt(2.0 / z_dim)
    gb1 = np.zeros(g_h[0])
    gW2 = np.random.randn(g_h[0], g_h[1])  * np.sqrt(2.0 / g_h[0])
    gb2 = np.zeros(g_h[1])
    gW3 = np.random.randn(g_h[1], seq_len)  * np.sqrt(1.0 / g_h[1])
    gb3 = np.zeros(seq_len)
    g_params = [gW1, gb1, gW2, gb2, gW3, gb3]

    # Discriminator: seq_len -> d_h[0] -> d_h[1] -> 1
    dW1 = np.random.randn(seq_len, d_h[0])  * np.sqrt(2.0 / seq_len)
    db1 = np.zeros(d_h[0])
    dW2 = np.random.randn(d_h[0], d_h[1])   * np.sqrt(2.0 / d_h[0])
    db2 = np.zeros(d_h[1])
    dW3 = np.random.randn(d_h[1], 1)         * np.sqrt(1.0 / d_h[1])
    db3 = np.zeros(1)
    d_params = [dW1, db1, dW2, db2, dW3, db3]

    return g_params, d_params


def train_gan(real_returns):
    """
    Train the GAN and collect snapshots for timelapse visualization.

    Returns: (snapshots, g_losses, d_losses, ret_mean, ret_std)
    """
    log("[GAN] Initializing architecture...")
    g_params, d_params = init_gan_params()

    # Normalize returns to [-1, 1] for tanh output
    ret_std = np.std(real_returns) + 1e-9
    ret_mean = np.mean(real_returns)
    real_norm = (real_returns - ret_mean) / (3.0 * ret_std)
    real_norm = np.clip(real_norm, -1.0, 1.0)

    # Build training windows of length SEQ_LEN
    seq_len = CONFIG["SEQ_LEN"]
    n_samples = len(real_norm) - seq_len + 1
    if n_samples < CONFIG["BATCH_SIZE"]:
        real_norm = np.tile(real_norm, 5)
        n_samples = len(real_norm) - seq_len + 1

    real_windows = np.array([real_norm[i:i+seq_len] for i in range(n_samples)])
    log(f"[GAN] Training samples: {real_windows.shape[0]} windows of length {seq_len}")

    # Optimizers
    g_opt = AdamOptimizer(g_params, lr=CONFIG["LR_G"],
                          beta1=CONFIG["BETA1"], beta2=CONFIG["BETA2"],
                          eps=CONFIG["ADAM_EPS"])
    d_opt = AdamOptimizer(d_params, lr=CONFIG["LR_D"],
                          beta1=CONFIG["BETA1"], beta2=CONFIG["BETA2"],
                          eps=CONFIG["ADAM_EPS"])

    # Gradient functions
    d_grad_fn = grad(d_loss_fn, argnum=0)
    g_grad_fn = grad(g_loss_fn, argnum=0)

    g_losses = []
    d_losses = []
    snapshots = []

    epochs = CONFIG["EPOCHS"]
    batch_size = CONFIG["BATCH_SIZE"]
    z_dim = CONFIG["Z_DIM"]
    snapshot_every = CONFIG["SNAPSHOT_EVERY"]

    log(f"[GAN] Training for {epochs} epochs (snapshot every {snapshot_every})...")
    t_start = time.time()

    for epoch in range(epochs):
        # Sample mini-batch of real data
        idx = np.random.choice(len(real_windows), batch_size, replace=True)
        real_batch = real_windows[idx]

        # Update Discriminator
        noise = np.random.randn(batch_size, z_dim)
        d_grads = d_grad_fn(d_params, g_params, real_batch, noise)
        d_params = d_opt.step(d_params, d_grads)

        # Update Generator
        noise = np.random.randn(batch_size, z_dim)
        g_grads = g_grad_fn(g_params, d_params, noise)
        g_params = g_opt.step(g_params, g_grads)

        # Record losses
        noise_eval = np.random.randn(batch_size, z_dim)
        dl = float(d_loss_fn(d_params, g_params, real_batch, noise_eval))
        gl = float(g_loss_fn(g_params, d_params, noise_eval))
        d_losses.append(dl)
        g_losses.append(gl)

        # Snapshot
        if epoch % snapshot_every == 0 or epoch == epochs - 1:
            z_vis = np.random.randn(CONFIG["N_GEN_PATHS"], z_dim)
            gen_samples = np.array(generator_forward(g_params, z_vis))
            snapshots.append({
                "epoch": epoch,
                "g_loss": gl,
                "d_loss": dl,
                "gen_samples": gen_samples,
            })

        if (epoch + 1) % 500 == 0:
            log(f"[GAN] Epoch {epoch+1}/{epochs} | D_loss: {dl:.4f} | G_loss: {gl:.4f}")

    elapsed = time.time() - t_start
    log(f"[GAN] Training complete in {elapsed:.1f}s | {len(snapshots)} snapshots")
    log(f"[GAN] Final D_loss: {d_losses[-1]:.4f} | G_loss: {g_losses[-1]:.4f}")

    return snapshots, g_losses, d_losses, ret_mean, ret_std


# =============================================================================
# MODULE 2 : RENDERING
# =============================================================================

def draw_flowchart(ax, g_loss, d_loss):
    """Draw GAN architecture flowchart with neural network nodes."""
    from matplotlib.patches import FancyBboxPatch

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor(THEME["BG"])

    mid = 0.50
    rng = np.random.RandomState(42)

    # ── Helpers ──────────────────────────────────────────────────────────

    def box(x, y, w, h, ec, label, label_pos="below"):
        p = FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.06",
            facecolor=THEME["BG"], edgecolor=ec, linewidth=1.8, zorder=2,
        )
        ax.add_patch(p)
        if label_pos == "below":
            ly, va = y - h / 2 - 0.06, "top"
        else:
            ly, va = y + h / 2 + 0.06, "bottom"
        ax.text(x, ly, label, ha="center", va=va,
                color=ec, fontsize=10, fontweight="bold", zorder=3)

    def nn_nodes(cx, cy, w, h, layers, color):
        """Draw layered neural network nodes with connecting edges."""
        n_l = len(layers)
        all_pos = []
        for li, n_nodes in enumerate(layers):
            lx = cx - w * 0.4 + li / max(n_l - 1, 1) * w * 0.8
            lpos = []
            for ni in range(n_nodes):
                ny = cy if n_nodes == 1 else (
                    cy - h * 0.3 + ni / max(n_nodes - 1, 1) * h * 0.6)
                lpos.append((lx, ny))
            all_pos.append(lpos)
        # Edges
        for li in range(len(all_pos) - 1):
            for p1 in all_pos[li]:
                for p2 in all_pos[li + 1]:
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                            color=color, alpha=0.12, linewidth=0.6, zorder=2)
        # Nodes (alternating brightness for variety)
        node_colors = [color, "#ffffff", color]
        for li, layer in enumerate(all_pos):
            for ni, (px, py) in enumerate(layer):
                c = node_colors[(ni + li) % 3]
                a = 0.65 + 0.25 * ((ni + li) % 2)
                ax.scatter([px], [py], s=35, c=[c], edgecolors="none",
                           alpha=a, zorder=3)

    def arrow(x1, y1, x2, y2, color=THEME["WHITE"], ls="-", lw=1.5):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                    lw=lw, linestyle=ls, mutation_scale=15),
                    zorder=4)

    # ── 1. Noise (z) ────────────────────────────────────────────────────
    box(0.7, mid, 1.0, 0.55, "#888888", "Noise (z)")
    # Waveform inside
    nx = np.linspace(0.30, 1.10, 40)
    ny = mid + 0.10 * np.sin(nx * 30) * rng.uniform(0.3, 1.0, 40)
    ax.plot(nx, ny, color=THEME["TEXT"], linewidth=0.8, alpha=0.5, zorder=3)

    # ── 2. Arrow → Generator ────────────────────────────────────────────
    arrow(1.20, mid, 1.88, mid)

    # ── 3. Generator box + NN nodes ─────────────────────────────────────
    box(2.8, mid, 1.7, 0.65, THEME["GEN_PATH"], "Generator")
    nn_nodes(2.8, mid, 1.3, 0.50, [3, 5, 5, 3], THEME["GEN_PATH"])

    # ── 4. Arrow → Fake Data ────────────────────────────────────────────
    arrow(3.65, mid, 4.35, mid + 0.10)

    # ── 5. Fake Data box ────────────────────────────────────────────────
    box(5.0, mid + 0.16, 1.0, 0.35, THEME["GEN_PATH"], "Fake Data")
    ax.scatter(5.0 + rng.uniform(-0.30, 0.30, 12),
               mid + 0.16 + rng.uniform(-0.08, 0.08, 12),
               s=7, c=THEME["GEN_PATH"], alpha=0.6, zorder=3)

    # ── 6. Real Data box ────────────────────────────────────────────────
    box(5.0, mid - 0.25, 1.0, 0.35, THEME["WHITE"], "Real Data")
    ax.scatter(5.0 + rng.uniform(-0.30, 0.30, 12),
               mid - 0.25 + rng.uniform(-0.08, 0.08, 12),
               s=7, c=THEME["WHITE"], alpha=0.6, zorder=3)

    # ── 7. Arrows → Discriminator ───────────────────────────────────────
    arrow(5.50, mid + 0.16, 6.18, mid + 0.05)
    arrow(5.50, mid - 0.25, 6.18, mid - 0.05)

    # ── 8. Discriminator box + NN nodes ─────────────────────────────────
    box(7.1, mid, 1.7, 0.65, THEME["D_LOSS"], "Discriminator")
    nn_nodes(7.1, mid, 1.3, 0.50, [3, 5, 5, 1], THEME["D_LOSS"])

    # ── 9. Arrow → Loss ─────────────────────────────────────────────────
    arrow(7.95, mid, 8.72, mid)

    # ── 10. Loss box ────────────────────────────────────────────────────
    box(9.2, mid, 0.8, 0.45, THEME["EPOCH_MKR"], "Loss")

    # ── Feedback arrow (dashed): Loss → Generator ───────────────────────
    ax.annotate("", xy=(2.8, mid - 0.43), xytext=(9.2, mid - 0.32),
                arrowprops=dict(arrowstyle="-|>", color=THEME["TEXT"],
                                lw=1.0, linestyle="--",
                                connectionstyle="arc3,rad=0.25",
                                mutation_scale=12),
                zorder=1)
    ax.text(6.0, mid - 0.47, "Fine-tuning", ha="center",
            color=THEME["TEXT"], fontsize=9, fontstyle="italic", alpha=0.6)

    # ── Live loss values above network boxes ────────────────────────────
    ax.text(2.8, mid + 0.40, f"G: {g_loss:.3f}", ha="center",
            color=THEME["GEN_PATH"], fontsize=9, alpha=0.9,
            fontfamily="monospace", zorder=5)
    ax.text(7.1, mid + 0.40, f"D: {d_loss:.3f}", ha="center",
            color=THEME["D_LOSS"], fontsize=9, alpha=0.9,
            fontfamily="monospace", zorder=5)


def render_static(snapshots, real_returns, real_price,
                   g_losses, d_losses, ret_mean, ret_std):
    """Renders a single static image showing the final GAN state."""
    try:
        snap = snapshots[-1]
        epoch = snap["epoch"]
        gen_norm = snap["gen_samples"]  # (N_GEN_PATHS, SEQ_LEN) in [-1,1]

        # De-normalize generated returns
        gen_returns = gen_norm * (3.0 * ret_std) + ret_mean
        seq_len = CONFIG["SEQ_LEN"]

        # Convert returns to price paths (rebased to 100)
        start_px = 100.0
        real_seg = real_price[:seq_len + 1]
        real_path = real_seg / real_seg[0] * start_px

        gen_paths = []
        for s in range(gen_returns.shape[0]):
            cumret = np.cumsum(gen_returns[s])
            path = start_px * np.exp(np.concatenate([[0.0], cumret]))
            gen_paths.append(path)
        gen_paths = np.array(gen_paths)

        # Histogram data
        real_hist = real_returns
        gen_hist = gen_norm.flatten() * (3.0 * ret_std) + ret_mean

        # Loss curves up to current epoch
        gl_curve = g_losses[:epoch + 1]
        dl_curve = d_losses[:epoch + 1]

        # ── Figure Setup ────────────────────────────────────────────────
        fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor=THEME["BG"])
        gs = gridspec.GridSpec(
            4, 1,
            height_ratios=[1.2, 3, 1.5, 1],
            hspace=0.10,
            left=0.07, right=0.95, top=0.90, bottom=0.08,
        )
        ax_flow  = fig.add_subplot(gs[0])
        ax_price = fig.add_subplot(gs[1])
        ax_dist  = fig.add_subplot(gs[2])
        ax_loss  = fig.add_subplot(gs[3])

        # ── GAN Architecture Flowchart ──────────────────────────────────
        draw_flowchart(ax_flow, snap["g_loss"], snap["d_loss"])

        axes = [ax_price, ax_dist, ax_loss]

        # ── Common Styling (Hawkes pattern) ─────────────────────────────
        for ax in axes:
            ax.set_facecolor(THEME["BG"])
            ax.tick_params(axis="both", which="major", labelsize=12,
                           colors=THEME["TEXT"], direction="in", length=5)
            ax.tick_params(axis="both", which="minor",
                           colors=THEME["TEXT"], direction="in", length=3)
            for spine in ax.spines.values():
                spine.set_color(THEME["SPINE"])
                spine.set_linewidth(0.6)
            ax.yaxis.grid(True, linewidth=0.3, alpha=0.45, color=THEME["GRID"])
            ax.xaxis.grid(False)

        # Hide x-tick labels on upper panels
        plt.setp(ax_price.get_xticklabels(), visible=False)
        plt.setp(ax_dist.get_xticklabels(), visible=False)

        # ── Panel 1: Price Paths ────────────────────────────────────────
        x_axis = np.arange(seq_len + 1)

        # Generated paths (cyan, semi-transparent)
        for p in gen_paths:
            ax_price.plot(x_axis, p, color=THEME["GEN_PATH"],
                         alpha=0.15, linewidth=1.0, zorder=2)

        # One highlighted generated path
        ax_price.plot(x_axis, gen_paths[0], color=THEME["GEN_PATH"],
                     linewidth=1.8, alpha=0.7, zorder=2, label="Generated")

        # Real path (white, bold)
        ax_price.plot(x_axis, real_path, color=THEME["REAL_PATH"],
                     linewidth=2.5, alpha=0.95, zorder=3,
                     label=f"Real {CONFIG['TICKER']}")

        # Fixed y-limits across all frames
        all_vals = np.concatenate([real_path, gen_paths.flatten()])
        all_vals = all_vals[np.isfinite(all_vals)]
        if len(all_vals) > 0:
            margin = (np.max(all_vals) - np.min(all_vals)) * 0.1 + 1.0
            ax_price.set_ylim(np.min(all_vals) - margin,
                              np.max(all_vals) + margin)

        ax_price.set_xlim(0, seq_len)
        ax_price.set_ylabel("Price (rebased to 100)", color=THEME["TEXT"],
                           fontsize=14, fontweight="bold")

        # ── Panel 2: Return Distribution ────────────────────────────────
        all_ret = np.concatenate([real_hist, gen_hist])
        all_ret = all_ret[np.isfinite(all_ret)]
        if len(all_ret) > 0:
            lo = np.percentile(all_ret, 1)
            hi = np.percentile(all_ret, 99)
            pad = (hi - lo) * 0.15
            bins = np.linspace(lo - pad, hi + pad, CONFIG["HIST_BINS"])
        else:
            bins = np.linspace(-0.1, 0.1, CONFIG["HIST_BINS"])

        # Real returns (white outline)
        ax_dist.hist(real_hist, bins=bins, density=True,
                    histtype="step", color=THEME["REAL_HIST"],
                    linewidth=2.0, alpha=0.9, label="Real Returns", zorder=3)

        # Generated returns (cyan fill)
        ax_dist.hist(gen_hist, bins=bins, density=True,
                    color=THEME["GEN_FILL"], alpha=0.55,
                    edgecolor=THEME["GEN_FILL"], linewidth=0.5,
                    label="Generated Returns", zorder=2)

        ax_dist.set_ylabel("Density", color=THEME["TEXT"],
                          fontsize=14, fontweight="bold")

        # ── Panel 3: Training Losses ────────────────────────────────────
        epochs_range = list(range(epoch + 1))
        if len(gl_curve) > 1:
            ax_loss.plot(epochs_range, gl_curve, color=THEME["G_LOSS"],
                        linewidth=1.6, alpha=0.95, label="G Loss")
            ax_loss.plot(epochs_range, dl_curve, color=THEME["D_LOSS"],
                        linewidth=1.6, alpha=0.95, label="D Loss")
            ax_loss.axvline(epoch, color=THEME["EPOCH_MKR"],
                          linewidth=1.5, linestyle="--", alpha=0.7)

        ax_loss.set_xlim(0, CONFIG["EPOCHS"])
        ax_loss.set_ylabel("Loss", color=THEME["TEXT"],
                          fontsize=14, fontweight="bold")
        ax_loss.set_xlabel("Epoch", color=THEME["TEXT"], fontsize=14)

        # ── Legend (bottom of figure) ───────────────────────────────────
        h1, l1 = ax_price.get_legend_handles_labels()
        h2, l2 = ax_dist.get_legend_handles_labels()
        h3, l3 = ax_loss.get_legend_handles_labels()
        all_h = h1 + h2 + h3
        all_l = l1 + l2 + l3

        if all_h:
            leg = fig.legend(
                all_h, all_l,
                loc="lower center", ncol=len(all_l),
                fontsize=12, frameon=True, fancybox=False,
                borderpad=0.5, handlelength=2.5, columnspacing=2.0,
                bbox_to_anchor=(0.52, 0.002),
                edgecolor=THEME["SPINE"],
                facecolor=THEME["BG"],
            )
            for txt in leg.get_texts():
                txt.set_color(THEME["WHITE"])

        # ── Title / HUD ────────────────────────────────────────────────
        fig.suptitle(
            "GAN  //  Synthetic Market Data Generation",
            fontsize=20, fontweight="bold", color=THEME["WHITE"],
            y=0.965,
        )
        fig.text(
            0.95, 0.965,
            f"Epoch {epoch}/{CONFIG['EPOCHS']}",
            fontsize=13, color=THEME["TEXT"], ha="right",
            fontfamily="monospace",
        )
        fig.text(
            0.07, 0.940,
            r"$\min_G \max_D \; \mathbb{E}[\log D(x)]"
            r" + \mathbb{E}[\log(1 - D(G(z)))]$",
            fontsize=13, color="#999999", fontfamily="serif",
        )

        # ── Save ────────────────────────────────────────────────────────
        out_path = CONFIG["OUTPUT_FILE"]
        fig.savefig(out_path, dpi=100, facecolor=THEME["BG"])
        plt.close(fig)
        log(f"[Success] Image saved to: {out_path}")
        return True

    except Exception as e:
        print(f"[Error] Render: {e}")
        plt.close("all")
        return False


# =============================================================================
# MAIN
# =============================================================================

def main():
    if os.path.exists(CONFIG["LOG_FILE"]):
        os.remove(CONFIG["LOG_FILE"])

    log("=" * 60)
    log("  GAN SYNTHETIC MARKET DATA PIPELINE")
    log("  Generative Adversarial Network // Return Distribution")
    log("=" * 60)
    log(f"    Ticker: {CONFIG['TICKER']}")
    log(f"    Epochs: {CONFIG['EPOCHS']}, Snapshot every: {CONFIG['SNAPSHOT_EVERY']}")
    g_arch = "->".join(str(h) for h in CONFIG["G_HIDDEN"])
    d_arch = "->".join(str(h) for h in CONFIG["D_HIDDEN"])
    log(f"    G: [{CONFIG['Z_DIM']}->{g_arch}->{CONFIG['SEQ_LEN']}]")
    log(f"    D: [{CONFIG['SEQ_LEN']}->{d_arch}->1]")

    # 1. Data
    real_returns, real_price = fetch_real_returns()

    # 2. GAN Training
    snapshots, g_losses, d_losses, ret_mean, ret_std = train_gan(real_returns)

    # 3. Render static image
    render_static(snapshots, real_returns, real_price,
                  g_losses, d_losses, ret_mean, ret_std)

    log("=" * 60)
    log("  PIPELINE FINISHED")
    log("=" * 60)


if __name__ == "__main__":
    main()
