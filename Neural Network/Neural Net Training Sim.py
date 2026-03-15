"""
ML_Training_Pipeline.py
=======================
Project: Quant Trader Lab - ML Training Visualization
Author: quant.traderr (Instagram)
License: MIT

Description:
    A production-ready pipeline for visualizing neural network training as a
    static side-by-side infographic. The network learns to classify Market Regimes
    (Bull / Bear / Sideways) from real BTC-USD price features.

    Left Panel:  3D Neural Network graph with trained weights & activations.
    Right Panel: 3D Loss Landscape with gradient descent traversal.

    Pipeline Steps:
    1.  **Data Acquisition**: Fetches BTC-USD via yfinance, engineers quant features.
    2.  **Regime Labeling**: Classifies each day as Bull / Bear / Sideways.
    3.  **Training Simulation**: Pure NumPy 8-layer feedforward network (171+ nodes).
    4.  **3D Rendering**: Plotly dual-scene static image via Kaleido.
    5.  **Output**: Static PNG image.

Dependencies:
    pip install numpy scipy plotly kaleido yfinance
"""

import os
import sys
import time
import warnings
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    # Data
    "TICKER": "BTC-USD",
    "PERIOD": "1y",
    "INTERVAL": "1d",

    # Neural Network
    "LAYER_SIZES": [5, 16, 32, 64, 32, 16, 8, 3],   # 8 layers, 176 nodes
    "LEARNING_RATE": 0.05,
    "MOMENTUM": 0.9,
    "TRAINING_EPOCHS": 200,
    "SEED": 42,

    # Loss Landscape
    "LANDSCAPE_RES": 80,
    "LANDSCAPE_RANGE": 4.0,

    # Output
    "RESOLUTION": (1920, 1080),
    "OUTPUT_FILE": "ML_Training_Visualization.png",
    "LOG_FILE": "ml_training_pipeline.log",

    # Rendering
    "MAX_EDGES_PER_LAYER": 80,
}

THEME = {
    "BG": "#0b0b0b",
    "GRID": "#1a1a1a",
    "TEXT": "#ffffff",
    "TEXT_DIM": "#888888",
    "FONT": "Roboto Mono",

    # Node colors per layer (8 layers)
    "NODE_COLORS": [
        "#ff1493",   # Deep Pink        (Input: 5 features)
        "#9932cc",   # Dark Orchid
        "#4169e1",   # Royal Blue
        "#00bfff",   # Deep Sky Blue
        "#00fa9a",   # Medium Spring Green
        "#7fff00",   # Chartreuse
        "#ffffff",   # White
        "#da70d6",   # Orchid            (Output: 3 regimes)
    ],

    # Loss landscape
    "SURFACE_COLORSCALE": "Jet",
    "PATH_COLOR": "#ff1493",
    "MARKER_COLOR": "#ffffff",

    # Edges
    "EDGE_GLOW": "rgba(0, 191, 255, 0.08)",
    "EDGE_CORE": "rgba(255, 255, 255, 0.25)",
}

# Regime labels
REGIME_NAMES = ["Bull", "Bear", "Sideways"]

# =============================================================================
# UTILS
# =============================================================================

def log(msg):
    """Centralized logger."""
    timestamp = time.strftime("%H:%M:%S")
    formatted = f"[{timestamp}] {msg}"
    print(formatted)
    try:
        with open(CONFIG["LOG_FILE"], "a") as f:
            f.write(formatted + "\n")
    except:
        pass

# =============================================================================
# MODULE 1: DATA
# =============================================================================

def fetch_and_engineer_features():
    """
    Fetches BTC-USD data and engineers quant features for regime classification.
    Returns: X (n, 5), y (n,) with labels {0: Bull, 1: Bear, 2: Sideways}
    """
    import yfinance as yf
    import pandas as pd

    log(f"[Data] Fetching {CONFIG['TICKER']} ({CONFIG['PERIOD']})...")

    try:
        df = yf.download(CONFIG["TICKER"], period=CONFIG["PERIOD"],
                         interval=CONFIG["INTERVAL"], progress=False)
    except Exception as e:
        log(f"[Error] YF Download failed: {e}. Using synthetic fallback.")
        return _generate_synthetic_fallback()

    # Handle MultiIndex columns (yfinance v0.2+)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
        log("[Data] Flattened MultiIndex columns.")

    if df.empty:
        log("[Warning] Empty dataframe. Using synthetic fallback.")
        return _generate_synthetic_fallback()

    price = df["Close"].values.flatten().astype(float)

    # Feature Engineering
    log("[Data] Engineering quant features...")

    # 1. Log Returns
    log_ret = np.diff(np.log(price + 1e-9))

    # 2. Rolling Volatility (20-day)
    window = 20
    vol = np.array([np.std(log_ret[max(0, i - window):i + 1])
                     for i in range(len(log_ret))])

    # 3. Momentum (10-day ROC)
    mom_window = 10
    momentum = np.zeros(len(log_ret))
    for i in range(mom_window, len(log_ret)):
        momentum[i] = (price[i + 1] - price[i + 1 - mom_window]) / \
                       (price[i + 1 - mom_window] + 1e-9)

    # 4. RSI-like oscillator (14-day)
    rsi = np.zeros(len(log_ret))
    rsi_window = 14
    for i in range(rsi_window, len(log_ret)):
        gains = np.maximum(log_ret[i - rsi_window:i], 0)
        losses = np.maximum(-log_ret[i - rsi_window:i], 0)
        avg_gain = np.mean(gains) + 1e-9
        avg_loss = np.mean(losses) + 1e-9
        rsi[i] = avg_gain / (avg_gain + avg_loss)  # Normalized 0-1

    # 5. Mean-Reversion Score (z-score of price vs 30-day MA)
    mr_window = 30
    mean_rev = np.zeros(len(log_ret))
    for i in range(mr_window, len(log_ret)):
        window_prices = price[i + 1 - mr_window:i + 2]
        ma = np.mean(window_prices)
        std = np.std(window_prices) + 1e-9
        mean_rev[i] = (price[i + 1] - ma) / std

    # Stack features
    X = np.column_stack([log_ret, vol, momentum, rsi, mean_rev])

    # Trim warmup period
    trim = max(window, mom_window, rsi_window, mr_window)
    X = X[trim:]

    # Regime Labels
    ret_trimmed = log_ret[trim:]
    sigma = np.std(ret_trimmed)
    y = np.ones(len(ret_trimmed), dtype=int) * 2   # Default: Sideways
    y[ret_trimmed > sigma] = 0                       # Bull
    y[ret_trimmed < -sigma] = 1                      # Bear

    # Normalize features
    for col in range(X.shape[1]):
        mu = np.mean(X[:, col])
        std = np.std(X[:, col]) + 1e-9
        X[:, col] = (X[:, col] - mu) / std

    log(f"[Data] {len(X)} samples | Features: 5 | "
        f"Regimes: Bull={np.sum(y==0)}, Bear={np.sum(y==1)}, "
        f"Sideways={np.sum(y==2)}")

    return X, y


def _generate_synthetic_fallback():
    """Synthetic spiral dataset if yfinance fails."""
    log("[Data] Generating synthetic spiral fallback...")
    np.random.seed(CONFIG["SEED"])
    n_per_class = 100
    n_classes = 3
    X = np.zeros((n_per_class * n_classes, 5))
    y = np.zeros(n_per_class * n_classes, dtype=int)

    for cls in range(n_classes):
        ix = range(n_per_class * cls, n_per_class * (cls + 1))
        r = np.linspace(0.0, 1.0, n_per_class)
        t = np.linspace(cls * 4.0, (cls + 1) * 4.0, n_per_class) + \
            np.random.randn(n_per_class) * 0.15
        X[ix, 0] = r * np.sin(t)
        X[ix, 1] = r * np.cos(t)
        X[ix, 2] = np.random.randn(n_per_class) * 0.3
        X[ix, 3] = np.random.randn(n_per_class) * 0.3
        X[ix, 4] = np.random.randn(n_per_class) * 0.3
        y[ix] = cls

    return X, y

# =============================================================================
# MODULE 2: NEURAL NETWORK (Pure NumPy)
# =============================================================================

class NumpyNeuralNetwork:
    """
    Pure NumPy feedforward neural network for market regime classification.
    Architecture: [5, 16, 32, 64, 32, 16, 8, 3]
    """

    def __init__(self, layer_sizes, lr=0.05, momentum=0.9, seed=42):
        np.random.seed(seed)
        self.layers = layer_sizes
        self.lr = lr
        self.momentum = momentum
        self.weights = []
        self.biases = []
        self.vel_w = []
        self.vel_b = []

        # He initialization
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * \
                np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(w)
            self.biases.append(b)
            self.vel_w.append(np.zeros_like(w))
            self.vel_b.append(np.zeros_like(b))

        # Training history
        self.history = {
            "loss": [],
            "weights_snapshots": [],
            "activations_snapshots": [],
            "param_trajectory": [],
        }

    def _relu(self, z):
        return np.maximum(0, z)

    def _relu_deriv(self, z):
        return (z > 0).astype(float)

    def _softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / (np.sum(exp_z, axis=1, keepdims=True) + 1e-9)

    def forward(self, X):
        self.activations = [X]
        self.pre_acts = []
        a = X
        for i in range(len(self.weights) - 1):
            z = a @ self.weights[i] + self.biases[i]
            self.pre_acts.append(z)
            a = self._relu(z)
            self.activations.append(a)
        # Output
        z = a @ self.weights[-1] + self.biases[-1]
        self.pre_acts.append(z)
        a = self._softmax(z)
        self.activations.append(a)
        return a

    def _cross_entropy(self, y_pred, y_true):
        n = y_true.shape[0]
        return -np.mean(np.log(y_pred[range(n), y_true] + 1e-9))

    def backward(self, y_true):
        n = y_true.shape[0]
        delta = self.activations[-1].copy()
        delta[range(n), y_true] -= 1
        delta /= n

        for i in reversed(range(len(self.weights))):
            dw = self.activations[i].T @ delta
            db = np.sum(delta, axis=0, keepdims=True)
            if i > 0:
                delta = (delta @ self.weights[i].T) * \
                        self._relu_deriv(self.pre_acts[i - 1])
            self.vel_w[i] = self.momentum * self.vel_w[i] - self.lr * dw
            self.vel_b[i] = self.momentum * self.vel_b[i] - self.lr * db
            self.weights[i] += self.vel_w[i]
            self.biases[i] += self.vel_b[i]

    def train_step(self, X, y):
        y_pred = self.forward(X)
        loss = self._cross_entropy(y_pred, y)
        self.backward(y)
        return loss

    def get_weight_magnitudes(self, layer_idx):
        m = np.abs(self.weights[layer_idx])
        return m / (m.max() + 1e-9)

    def get_activation_magnitudes(self):
        return [np.mean(np.abs(a), axis=0) for a in self.activations]

    def get_param_2d(self):
        return np.array([np.mean(self.weights[0]),
                         np.mean(self.weights[-1])])


def run_training(X, y):
    """Runs full training, recording snapshots every epoch."""
    log("[Training] Initializing neural network...")
    net = NumpyNeuralNetwork(
        CONFIG["LAYER_SIZES"],
        lr=CONFIG["LEARNING_RATE"],
        momentum=CONFIG["MOMENTUM"],
        seed=CONFIG["SEED"],
    )

    epochs = CONFIG["TRAINING_EPOCHS"]
    log(f"[Training] Running {epochs} epochs...")

    for epoch in range(epochs):
        loss = net.train_step(X, y)
        net.history["loss"].append(loss)
        net.history["weights_snapshots"].append(
            [w.copy() for w in net.weights]
        )
        net.history["activations_snapshots"].append(
            net.get_activation_magnitudes()
        )
        net.history["param_trajectory"].append(net.get_param_2d())

        if (epoch + 1) % 50 == 0:
            log(f"[Training] Epoch {epoch+1}/{epochs} | Loss: {loss:.4f}")

    log(f"[Training] Final loss: {net.history['loss'][-1]:.4f}")
    return net

# =============================================================================
# MODULE 3: RENDERING
# =============================================================================

def compute_network_layout(layer_sizes):
    """
    Arena3D-style: nodes scattered on flat horizontal planes stacked vertically.
    Each layer is a transparent plane with nodes spread across it.
    """
    np.random.seed(42)
    positions = []
    n_layers = len(layer_sizes)

    # Vertical spacing between planes — balanced with XY range
    z_spacing = 0.9
    plane_half = 2.0  # Half-width of each plane

    for li, n_nodes in enumerate(layer_sizes):
        z_val = li * z_spacing

        if n_nodes <= 3:
            # Small layers: evenly spaced in a line
            x = np.linspace(-1.0, 1.0, n_nodes)
            y = np.zeros(n_nodes)
        elif n_nodes <= 16:
            cols = int(np.ceil(np.sqrt(n_nodes)))
            rows = int(np.ceil(n_nodes / cols))
            xg = np.linspace(-plane_half * 0.7, plane_half * 0.7, cols)
            yg = np.linspace(-plane_half * 0.7, plane_half * 0.7, rows)
            xx, yy = np.meshgrid(xg, yg)
            x = xx.flatten()[:n_nodes]
            y = yy.flatten()[:n_nodes]
        else:
            cols = int(np.ceil(np.sqrt(n_nodes)))
            rows = int(np.ceil(n_nodes / cols))
            xg = np.linspace(-plane_half * 0.85, plane_half * 0.85, cols)
            yg = np.linspace(-plane_half * 0.85, plane_half * 0.85, rows)
            xx, yy = np.meshgrid(xg, yg)
            x = xx.flatten()[:n_nodes] + np.random.randn(n_nodes) * 0.1
            y = yy.flatten()[:n_nodes] + np.random.randn(n_nodes) * 0.1

        z = np.full(n_nodes, z_val)
        positions.append((x, y, z))

    return positions


# Quant-specific layer labels for the network visualization
LAYER_LABELS = [
    "Market Features",        # Input: returns, vol, momentum, RSI, mean-rev
    "Signal Extraction",      # Hidden 1
    "Pattern Recognition",    # Hidden 2
    "Regime Detection",       # Hidden 3 (largest — 64 nodes)
    "Risk Assessment",        # Hidden 4
    "Alpha Generation",       # Hidden 5
    "Position Sizing",        # Hidden 6
    "Regime Output",          # Output: Bull / Bear / Sideways
]


def generate_loss_landscape(res, rng):
    """
    Generates a dramatic multi-modal loss surface with clear peaks and valleys.
    Matches the reference: vivid rainbow surface with visible topography.
    """
    x = np.linspace(-rng, rng, res)
    y = np.linspace(-rng, rng, res)
    X, Y = np.meshgrid(x, y)

    # Dramatic multi-modal surface: Rastrigin-like + Gaussian wells
    Z = (X**2 + Y**2) / 8.0  # Base bowl
    Z += 1.5 * (np.cos(2.0 * X) + np.cos(2.0 * Y))  # Ripples (peaks/valleys)
    Z += -3.0 * np.exp(-((X - 2.5)**2 + (Y - 2.5)**2) / 1.5)  # Deep global min
    Z += -1.5 * np.exp(-((X + 1.5)**2 + (Y - 1.0)**2) / 1.0)  # Local min
    Z += -1.0 * np.exp(-((X - 0.5)**2 + (Y + 2.0)**2) / 0.8)  # Another local min
    Z += 2.0 * np.exp(-((X + 2.5)**2 + (Y + 2.5)**2) / 3.0)   # High plateau

    # Normalize to nice visual range
    Z = (Z - Z.min()) / (Z.max() - Z.min() + 1e-9) * 5.0

    return X, Y, Z


def generate_gd_path(n_points, rng):
    """Synthetic gradient descent spiral path from high loss to minimum."""
    t = np.linspace(0, 4 * np.pi, n_points)
    start = np.array([-2.0, -2.0])
    end = np.array([2.5, 2.5])

    progress = t / t.max()
    ease = progress * progress * (3 - 2 * progress)

    spiral_amp = 1.5 * (1 - ease)
    px = start[0] + (end[0] - start[0]) * ease + spiral_amp * np.sin(t * 1.5)
    py = start[1] + (end[1] - start[1]) * ease + spiral_amp * np.cos(t * 1.5)

    # Clamp to landscape bounds
    px = np.clip(px, -rng + 0.1, rng - 0.1)
    py = np.clip(py, -rng + 0.1, rng - 0.1)

    return px, py


def get_path_z(path_x, path_y, lx, ly, lz):
    """Interpolate Z values on the loss surface for the GD path."""
    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator(
        (lx[0, :], ly[:, 0]), lz.T,
        method="linear", bounds_error=False, fill_value=float(lz.min())
    )
    pts = np.column_stack([path_x, path_y])
    return interp(pts)


def _build_edge_traces(positions, layer_idx, weight_snap, max_edges):
    """
    Builds edge line coordinates between adjacent layers.
    Filters to top-N edges by weight magnitude.
    """
    pos_from = positions[layer_idx]
    pos_to = positions[layer_idx + 1]
    n_from, n_to = len(pos_from[0]), len(pos_to[0])

    if weight_snap is not None:
        w = np.abs(weight_snap)
        w_norm = w / (w.max() + 1e-9)
        flat = w_norm.flatten()
        if len(flat) > max_edges:
            thresh = np.sort(flat)[-max_edges]
        else:
            thresh = 0.0
    else:
        w_norm = None
        thresh = 0.0

    xe, ye, ze = [], [], []
    for i in range(n_from):
        for j in range(n_to):
            if w_norm is not None and w_norm[i, j] < thresh:
                continue
            xe.extend([pos_from[0][i], pos_to[0][j], None])
            ye.extend([pos_from[1][i], pos_to[1][j], None])
            ze.extend([pos_from[2][i], pos_to[2][j], None])

    return xe, ye, ze


# --------------- STATIC IMAGE RENDERER ---------------

def render_static_image(data):
    """Renders a single static PNG showing the final trained state."""
    output_path = os.path.abspath(CONFIG["OUTPUT_FILE"])
    log(f"[Render] Generating static image...")

    try:
        epoch_idx = data["n_epochs"] - 1  # Final epoch

        # --- Create dual subplot figure ---
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "scene"}, {"type": "scene"}]],
            horizontal_spacing=0.02,
            subplot_titles=[
                "<b>Neural Network Architecture</b>",
                "<b>Loss Landscape</b>"
            ],
        )

        # ============================================================
        # LEFT PANEL: Neural Network — Arena3D stacked planes
        # ============================================================

        n_layers = len(CONFIG["LAYER_SIZES"])
        weights_snap = data["weights"][epoch_idx]
        act_mags = data["activations"][epoch_idx]
        plane_half = 2.0
        z_spacing = 0.9

        # Draw transparent floor planes + nodes + labels per layer
        for li in range(n_layers):
            pos = data["positions"][li]
            color = THEME["NODE_COLORS"][li]
            z_val = li * z_spacing

            # --- Transparent floor plane (Mesh3d rectangle) ---
            fig.add_trace(
                go.Mesh3d(
                    x=[-plane_half, plane_half, plane_half, -plane_half],
                    y=[-plane_half, -plane_half, plane_half, plane_half],
                    z=[z_val, z_val, z_val, z_val],
                    i=[0, 0], j=[1, 2], k=[2, 3],
                    color="#333333", opacity=0.15,
                    showlegend=False, hoverinfo="skip",
                ),
                row=1, col=1,
            )

            # --- Layer label ---
            fig.add_trace(
                go.Scatter3d(
                    x=[-plane_half + 0.2], y=[plane_half - 0.2], z=[z_val + 0.05],
                    mode="text",
                    text=[LAYER_LABELS[li]],
                    textfont=dict(size=11, color="#999999", family="Roboto Mono"),
                    showlegend=False, hoverinfo="skip",
                ),
                row=1, col=1,
            )

            # --- Nodes (static sizes based on final activations) ---
            if li < len(act_mags):
                a = np.clip(act_mags[li], 0, 1)
                sizes = 6 + 10 * a
                if np.isscalar(sizes):
                    sizes = np.full(len(pos[0]), float(sizes))
            else:
                sizes = np.full(len(pos[0]), 8.0)

            fig.add_trace(
                go.Scatter3d(
                    x=pos[0], y=pos[1], z=pos[2],
                    mode="markers",
                    marker=dict(
                        size=sizes, color=color, opacity=0.92,
                        line=dict(width=0.5, color="white"),
                    ),
                    showlegend=False, hoverinfo="skip",
                ),
                row=1, col=1,
            )

        # --- Edges between layers — colored by source layer ---
        for li in range(n_layers - 1):
            w_snap = weights_snap[li]
            xe, ye, ze = _build_edge_traces(
                data["positions"], li, w_snap,
                CONFIG["MAX_EDGES_PER_LAYER"]
            )
            if not xe:
                continue

            layer_hex = THEME["NODE_COLORS"][li]
            r_c = int(layer_hex[1:3], 16)
            g_c = int(layer_hex[3:5], 16)
            b_c = int(layer_hex[5:7], 16)

            fig.add_trace(
                go.Scatter3d(
                    x=xe, y=ye, z=ze, mode="lines",
                    line=dict(color=f"rgba({r_c},{g_c},{b_c},0.15)", width=2),
                    showlegend=False, hoverinfo="skip",
                ),
                row=1, col=1,
            )
            fig.add_trace(
                go.Scatter3d(
                    x=xe, y=ye, z=ze, mode="lines",
                    line=dict(color=f"rgba({r_c},{g_c},{b_c},0.4)", width=1),
                    showlegend=False, hoverinfo="skip",
                ),
                row=1, col=1,
            )

        # ============================================================
        # RIGHT PANEL: Loss Landscape
        # ============================================================

        fig.add_trace(
            go.Surface(
                x=data["landscape_X"],
                y=data["landscape_Y"],
                z=data["landscape_Z"],
                colorscale=THEME["SURFACE_COLORSCALE"],
                opacity=0.75,
                showscale=True,
                colorbar=dict(
                    title=dict(text="Loss", font=dict(size=14, color="#c0c0c0",
                                                       family="Roboto Mono")),
                    tickfont=dict(size=12, color="#888888", family="Roboto Mono"),
                    len=0.6, thickness=15, x=1.0,
                    bgcolor="rgba(11,11,11,0.5)",
                    bordercolor="#333333", borderwidth=1,
                ),
                contours=dict(
                    x=dict(show=True, color="rgba(255,255,255,0.25)",
                           highlightcolor="rgba(255,255,255,0.4)", width=1),
                    y=dict(show=True, color="rgba(255,255,255,0.25)",
                           highlightcolor="rgba(255,255,255,0.4)", width=1),
                    z=dict(show=True, usecolormap=True,
                           highlightcolor="#ffffff", project_z=True),
                ),
                lighting=dict(
                    ambient=0.6, diffuse=0.5, specular=0.4,
                    roughness=0.3, fresnel=0.3,
                ),
            ),
            row=1, col=2,
        )

        # Start / End annotations on landscape
        fig.add_trace(
            go.Scatter3d(
                x=[data["path_x"][0]], y=[data["path_y"][0]],
                z=[data["path_z"][0] + 0.3],
                mode="markers+text",
                marker=dict(size=8, color="#ff69b4"),
                text=["Starting here"], textposition="top center",
                textfont=dict(size=11, color="#ff69b4", family="Roboto Mono"),
                showlegend=False, hoverinfo="skip",
            ),
            row=1, col=2,
        )
        fig.add_trace(
            go.Scatter3d(
                x=[data["path_x"][-1]], y=[data["path_y"][-1]],
                z=[data["path_z"][-1] + 0.3],
                mode="markers+text",
                marker=dict(size=8, color="#69ffb4"),
                text=["Global minimum"], textposition="bottom center",
                textfont=dict(size=11, color="#69ffb4", family="Roboto Mono"),
                showlegend=False, hoverinfo="skip",
            ),
            row=1, col=2,
        )

        # Full gradient descent path
        fig.add_trace(
            go.Scatter3d(
                x=data["path_x"], y=data["path_y"],
                z=data["path_z"] + 0.08,
                mode="lines",
                line=dict(color=THEME["PATH_COLOR"], width=5),
                showlegend=False, hoverinfo="skip",
            ),
            row=1, col=2,
        )

        # Final position marker
        fig.add_trace(
            go.Scatter3d(
                x=[data["path_x"][-1]], y=[data["path_y"][-1]],
                z=[data["path_z"][-1] + 0.15],
                mode="markers",
                marker=dict(
                    size=8, color=THEME["MARKER_COLOR"],
                    symbol="diamond",
                    line=dict(width=1, color=THEME["PATH_COLOR"]),
                ),
                showlegend=False, hoverinfo="skip",
            ),
            row=1, col=2,
        )

        # ============================================================
        # CAMERAS (fixed angles)
        # ============================================================

        angle_l = 270
        cam_left = dict(
            eye=dict(
                x=2.2 * np.cos(np.radians(angle_l)),
                y=2.2 * np.sin(np.radians(angle_l)),
                z=1.3,
            ),
            center=dict(x=0, y=0, z=0.1),
            up=dict(x=0, y=0, z=1),
        )

        angle_r = 260
        cam_right = dict(
            eye=dict(
                x=2.0 * np.cos(np.radians(angle_r)),
                y=2.0 * np.sin(np.radians(angle_r)),
                z=1.8,
            ),
            center=dict(x=0, y=0, z=0.4),
            up=dict(x=0, y=0, z=1),
        )

        # ============================================================
        # ANNOTATIONS
        # ============================================================
        final_loss = data["losses"][-1]
        annotations = [
            dict(
                text=f"Epoch: {data['n_epochs']}/{data['n_epochs']}  |  "
                     f"Loss: {final_loss:.4f}",
                xref="paper", yref="paper", x=0.5, y=0.02,
                showarrow=False,
                font=dict(size=16, color="#aaaaaa", family=THEME["FONT"]),
                bgcolor="rgba(11,11,11,0.85)",
                bordercolor="#333333", borderwidth=1, borderpad=8,
            ),
            dict(
                text=f"Training Complete  |  Final Loss: {final_loss:.4f}",
                xref="paper", yref="paper", x=0.5, y=0.06,
                showarrow=False,
                font=dict(size=20, color="#d0d0d0", family=THEME["FONT"]),
                bgcolor="rgba(11,11,11,0.85)",
                bordercolor="#555555", borderwidth=1, borderpad=10,
            ),
        ]

        # ============================================================
        # LAYOUT
        # ============================================================

        net_axes = dict(
            showgrid=False, showbackground=True,
            backgroundcolor=THEME["BG"],
            showticklabels=False, title="",
            zeroline=False, showline=False, showspikes=False,
        )
        scene_net = dict(
            xaxis=net_axes, yaxis=net_axes, zaxis=net_axes,
            bgcolor=THEME["BG"],
            camera=cam_left,
            aspectmode="cube",
        )

        grid_color = "#2a2a2a"
        label_color = "#c0c0c0"
        tick_color = "#888888"
        landscape_axis = dict(
            showgrid=True, gridcolor=grid_color, gridwidth=1,
            showbackground=True, backgroundcolor="#080808",
            showticklabels=True, tickfont=dict(size=12, color=tick_color,
                                                family="Roboto Mono"),
            zeroline=False, showline=True, linecolor="#333333",
            showspikes=False,
        )
        scene_landscape = dict(
            xaxis={**landscape_axis, "title": dict(
                text="Weights (θ₁)", font=dict(size=16, color=label_color,
                                                 family="Roboto Mono"))},
            yaxis={**landscape_axis, "title": dict(
                text="Weights (θ₂)", font=dict(size=16, color=label_color,
                                                 family="Roboto Mono"))},
            zaxis={**landscape_axis, "title": dict(
                text="Loss", font=dict(size=16, color=label_color,
                                        family="Roboto Mono"))},
            bgcolor="#080808",
            camera=cam_right,
            aspectmode="cube",
        )

        fig.update_layout(
            width=CONFIG["RESOLUTION"][0],
            height=CONFIG["RESOLUTION"][1],
            paper_bgcolor=THEME["BG"],
            plot_bgcolor=THEME["BG"],
            showlegend=False,
            margin=dict(l=10, r=10, t=60, b=50),
            scene=scene_net,
            scene2=scene_landscape,
            annotations=annotations,
            font=dict(family=THEME["FONT"], color=THEME["TEXT"]),
        )

        fig.update_annotations(
            selector=dict(text="<b>Neural Network Architecture</b>"),
            font=dict(size=20, color="#d0d0d0", family=THEME["FONT"]),
        )
        fig.update_annotations(
            selector=dict(text="<b>Loss Landscape</b>"),
            font=dict(size=20, color="#d0d0d0", family=THEME["FONT"]),
        )

        fig.write_image(output_path)
        log(f"[Success] Image saved: {output_path}")

    except Exception as e:
        log(f"[Error] Render failed: {e}")

# =============================================================================
# MAIN
# =============================================================================

def main():
    if os.path.exists(CONFIG["LOG_FILE"]):
        os.remove(CONFIG["LOG_FILE"])

    log("=" * 60)
    log("  ML TRAINING SIMULATION PIPELINE")
    log("  Market Regime Classification (Bull / Bear / Sideways)")
    log("=" * 60)

    # 1. Data
    X, y = fetch_and_engineer_features()

    # 2. Training
    net = run_training(X, y)

    # 3. Loss Landscape
    log("[Landscape] Generating 3D loss surface...")
    lx, ly, lz = generate_loss_landscape(
        CONFIG["LANDSCAPE_RES"], CONFIG["LANDSCAPE_RANGE"]
    )

    # 4. Gradient Path
    log("[Landscape] Computing gradient descent path...")
    path_x, path_y = generate_gd_path(
        CONFIG["TRAINING_EPOCHS"], CONFIG["LANDSCAPE_RANGE"]
    )
    path_z = get_path_z(path_x, path_y, lx, ly, lz)

    # 5. Network Layout
    positions = compute_network_layout(CONFIG["LAYER_SIZES"])

    # 6. Pack data for renderer
    render_data = {
        "positions": positions,
        "weights": net.history["weights_snapshots"],
        "activations": net.history["activations_snapshots"],
        "losses": net.history["loss"],
        "n_epochs": CONFIG["TRAINING_EPOCHS"],
        "landscape_X": lx,
        "landscape_Y": ly,
        "landscape_Z": lz,
        "path_x": path_x,
        "path_y": path_y,
        "path_z": path_z,
    }

    # 7. Render static image
    render_static_image(render_data)

    log("=" * 60)
    log("  PIPELINE FINISHED")
    log("=" * 60)


if __name__ == "__main__":
    main()
