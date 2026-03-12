"""
OU Process Pipeline.py
======================
Project: Quant Trader Lab - Mean Reversion Simulation
Author: quant.traderr (Instagram)
License: MIT

Description:
    A production-ready pipeline for Ornstein-Uhlenbeck (OU) Process Simulation.
    
    It models mean-reverting behavior commonly found in interest rates, 
    volatility, or pairs trading spreads.
    
    Pipeline Steps:
    1.  **Data Acquisition & Calibration**: Fetches data or uses custom data to calibrate OU parameters (Theta, Mu, Sigma).
    2.  **Simulation Engine**: Runs vectorized OU process simulations.
    3.  **Analysis**: Computes expected values and confidence bounds.
    4.  **Static Visualization**: Generates a high-quality snapshot of the pathways demonstrating mean reversion.

    NOTE: Video rendering has been removed for pipeline efficiency.

Dependencies:
    pip install numpy pandas matplotlib yfinance scipy
"""

import time
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

# Ignore warnings
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---

CONFIG = {
    # Simulation Parameters
    "SIMULATIONS": 2500,
    "DAYS_TO_PROJECT": 252,
    "DT": 1/252, # Assuming daily steps over a trading year 
    
    # Base OU Parameters (used if AUTO_CALIBRATE = False)
    "THETA": 3.5,   # Mean reversion speed
    "MU": 0.5,      # Long term mean
    "SIGMA": 0.55,  # Volatility
    
    "AUTO_CALIBRATE": False, # If True, estimates Theta, Mu, Sigma from historical data
    
    # Data (if AUTO_CALIBRATE = True)
    # Good candidates: VIX, interest rates, or ratio/spread between two cointegrated assets
    "TICKER": "^VIX",  
    "LOOKBACK_YEARS": 5,
    "INTERVAL": "1d",
    
    # Output
    "OUTPUT_IMAGE": "ou_static_pipeline.png",
    
    # Aesthetics (Matching the convergence video & MC pipeline)
    "COLOR_BG": '#050508',
    "COLOR_LINE": (0.0, 0.83, 1.0, 0.05), # Cyan with low alpha
    "COLOR_MEAN": '#f5a623', # Amber
    "COLOR_BOUNDS": '#ff3cac', # Magenta
    "COLOR_GRID": '#141424'
}

# --- UTILS ---

def log(msg):
    """Simple logger."""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}")

# --- MODULE 1: DATA & CALIBRATION ---

def get_data_and_calibrate():
    """
    Fetches market data or user-provided data, and calibrates OU parameters 
    using linear regression (if AUTO_CALIBRATE is True).
    """
    if not CONFIG["AUTO_CALIBRATE"]:
        log("[Data] Auto-Calibration OFF. Using predefined Config parameters.")
        # If no starting data, spread the starts out to demo the "All Paths Converge" look
        start_vals = np.random.uniform(CONFIG["MU"] - 2.0, CONFIG["MU"] + 2.0, CONFIG["SIMULATIONS"])
        return CONFIG["THETA"], CONFIG["MU"], CONFIG["SIGMA"], start_vals
        
    log(f"[Data] Fetching historical data ({CONFIG['TICKER']}) for calibration...")
    
    try:
        # yfinance logic
        data = yf.download(CONFIG["TICKER"], period=f"{CONFIG['LOOKBACK_YEARS']}y", interval=CONFIG["INTERVAL"], progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data = data.xs(CONFIG["TICKER"], axis=1, level=1)
            
        prices = data['Close'].dropna().values

        # ======================================================================
        # 💡 USER DATA INPUT BLOCK
        # ======================================================================
        # To use your own static data instead of yfinance, uncomment and modify 
        # the lines below to load from your own CSV file:
        #
        # log("[Data] Loading custom user data...")
        # df = pd.read_csv("your_data_path.csv")
        # prices = df['Your_Spread_or_Price_Column'].dropna().values
        # ======================================================================

        if len(prices) < 2:
            raise ValueError("Not enough data to calibrate.")

        # Current value to start the simulation
        start_vals = np.full(CONFIG["SIMULATIONS"], prices[-1])

        # Calibrate using Linear Regression: X_{t} - X_{t-1} = a + b * X_{t-1} + error
        # Because: dX_t = theta * (mu - X_t) dt + sigma * dW_t
        #         X_t - X_t_1 = (theta * mu) * dt - (theta * dt) * X_t_1 + sigma * dW_t
        X_t = prices[1:]
        X_t_1 = prices[:-1]
        
        # OLS
        b, a = np.polyfit(X_t_1, X_t - X_t_1, 1)
        
        dt = CONFIG["DT"]
        
        theta = -b / dt
        mu = a / (theta * dt)
        
        residuals = (X_t - X_t_1) - (a + b * X_t_1)
        sigma = np.std(residuals) / np.sqrt(dt)
        
        log(f"[Calibration] Calibrated from data -> Theta: {theta:.4f}, Mu: {mu:.4f}, Sigma: {sigma:.4f}")
        return theta, mu, sigma, start_vals

    except Exception as e:
        log(f"[Error] Failed to calibrate: {e}. Fallback to Config.")
        fallback_starts = np.random.uniform(CONFIG["MU"] - 2.0, CONFIG["MU"] + 2.0, CONFIG["SIMULATIONS"])
        return CONFIG["THETA"], CONFIG["MU"], CONFIG["SIGMA"], fallback_starts

# --- MODULE 2: SIMULATION ENGINE ---

class OUEngine:
    def __init__(self, theta, mu, sigma, start_vals):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.start_vals = start_vals
        self.sims = CONFIG['SIMULATIONS']
        self.steps = CONFIG['DAYS_TO_PROJECT']
        self.dt = CONFIG['DT']

    def run(self):
        """
        Executes the OU Simulation using the Euler-Maruyama discretization.
        """
        log(f"[Simulation] Starting {self.sims} OU paths for {self.steps} steps...")
        start_time = time.time()
        
        # Initialize paths array
        paths = np.zeros((self.steps + 1, self.sims))
        
        # Starting point for all paths
        paths[0] = self.start_vals

        # Vectorized Euler-Maruyama Method
        for t in range(1, self.steps + 1):
            dW = np.random.standard_normal(self.sims) * np.sqrt(self.dt)
            # dX = theta * (mu - X) * dt + sigma * dW
            paths[t] = paths[t-1] + self.theta * (self.mu - paths[t-1]) * self.dt + self.sigma * dW

        duration = time.time() - start_time
        log(f"[Simulation] Completed in {duration:.2f}s.")
        return paths

# --- MODULE 3: ANALYSIS & VISUALIZATION ---

def analyze_and_visualize(paths, theta, mu, sigma):
    """
    Calculates stats and generates a static summary image showcasing mean reversion.
    """
    steps = paths.shape[0]
    final_values = paths[-1, :]
    
    # Statistics
    mean_path = np.mean(paths, axis=1)
    upper_path = np.percentile(paths, 95, axis=1)
    lower_path = np.percentile(paths, 5, axis=1)
    
    log("=== RESULTS SUMMARY ===")
    log(f"Theoretical Long-term Mean (Mu): {mu:.4f}")
    log(f"Simulated Cross-sectional Mean (End): {np.mean(final_values):.4f}")
    log(f"95% Confidence Bound (End): {upper_path[-1]:.4f}")
    log(f" 5% Confidence Bound (End): {lower_path[-1]:.4f}")
    
    # --- VISUALIZATION ---
    log("[Visual] Generating static snapshot...")
    
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor(CONFIG['COLOR_BG'])
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.3)
    ax_main = fig.add_subplot(gs[0])
    ax_hist = fig.add_subplot(gs[1])
    
    # 1. Trajectories Plot
    ax_main.set_facecolor(CONFIG['COLOR_BG'])
    ax_main.grid(True, color=CONFIG['COLOR_GRID'], linestyle='--')
    ax_main.set_title(f"Ornstein-Uhlenbeck (Mean Reversion) Pipeline: ({CONFIG['SIMULATIONS']} runs)\n"
                      f"$\\theta$={theta:.2f}, $\\mu$={mu:.2f}, $\\sigma$={sigma:.2f}", 
                      color='white', fontsize=16, weight='bold', pad=20)
    ax_main.set_ylabel("Process Value", color='gray')
    ax_main.tick_params(colors='gray')
    for spine in ax_main.spines.values(): spine.set_edgecolor('#333')
    
    ax_main.set_xlim(0, steps - 1)
    
    # Plot paths (subset to avoid memory overload)
    step_plot = max(1, CONFIG['SIMULATIONS'] // 1000)
    ax_main.plot(paths[:, ::step_plot], color=CONFIG['COLOR_LINE'], linewidth=1.0)
    
    # Stats Lines
    x_axis = np.arange(steps)
    ax_main.plot(x_axis, mean_path, color=CONFIG['COLOR_MEAN'], linewidth=2.5, linestyle='-', label='Simulated Average')
    ax_main.axhline(mu, color='#ffffff', linewidth=1.5, linestyle=':', label='Theoretical Mean (μ)')
    ax_main.plot(x_axis, upper_path, color=CONFIG['COLOR_BOUNDS'], linewidth=1.5, linestyle='--', label='95% Bound')
    ax_main.plot(x_axis, lower_path, color=CONFIG['COLOR_BOUNDS'], linewidth=1.5, linestyle='--', label='5% Bound')
    
    leg = ax_main.legend(loc='lower left', facecolor=CONFIG['COLOR_BG'], edgecolor='#333')
    for text in leg.get_texts(): text.set_color('white')
    
    # 2. Histogram Plot
    ax_hist.set_facecolor(CONFIG['COLOR_BG'])
    ax_hist.grid(True, color=CONFIG['COLOR_GRID'], linestyle=':')
    ax_hist.set_xlabel(f"Value Distribution @ Step {steps-1}", color='gray')
    ax_hist.set_ylabel("Frequency", color='gray')
    ax_hist.tick_params(colors='gray')
    for spine in ax_hist.spines.values(): spine.set_edgecolor('#333')
    
    # Plot histogram
    ax_hist.hist(final_values, bins=100, color='#00d4ff', edgecolor=CONFIG['COLOR_BG'], alpha=0.9)
    # Highlight final empirical mean
    ax_hist.axvline(np.mean(final_values), color=CONFIG['COLOR_MEAN'], linestyle='dashed', linewidth=2)
    
    # Save image
    out_path = os.path.join(os.path.dirname(__file__), CONFIG['OUTPUT_IMAGE'])
    plt.savefig(out_path, facecolor=CONFIG['COLOR_BG'], dpi=100)
    plt.close(fig)
    log(f"[Visual] Saved to {out_path}")

# --- MAIN ---

def main():
    log("=== OU PROCESS PIPELINE ===")
    
    # 1. Data & Calibration
    theta, mu, sigma, start_vals = get_data_and_calibrate()
    
    # 2. Simulation
    engine = OUEngine(theta, mu, sigma, start_vals)
    paths = engine.run()
    
    # 3. Analysis & Output
    analyze_and_visualize(paths, theta, mu, sigma)
    
    log("=== PIPELINE FINISHED ===")

if __name__ == "__main__":
    main()
