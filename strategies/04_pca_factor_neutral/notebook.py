# %% [markdown]
# # Strategy 4 — PCA Factor-Neutral Long/Short (interactive walkthrough)
#
# This is the **section-by-section** companion to `run.py`. Every `# %%` marks a
# cell you can run individually in VS Code (Python Interactive), Jupyter, or
# Spyder. It imports the exact same shared framework and strategy logic used by
# the automated runner — nothing is re-implemented — so what you step through
# here is what the pipeline actually does. Run cells top to bottom; each one
# prints or plots its output so you can inspect intermediate results and lift
# any figure/table straight into a report.
#
# > Synthetic data validates *implementation correctness and edge mechanics*,
# > not live profitability. Swap in real data at the marked cell.

# %%
# --- Cell 1: setup & imports -------------------------------------------------
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Make the shared framework importable regardless of where you launch from.
ROOT = os.path.abspath(os.path.join(os.getcwd().split("strategies")[0]))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from quant_framework import metrics, robustness, data  # noqa: E402

# Load the strategy module (numbered dir isn't a normal import path).
_sp = importlib.util.spec_from_file_location(
    "pca_strategy", os.path.join(ROOT, "strategies", "04_pca_factor_neutral", "strategy.py"))
strat = importlib.util.module_from_spec(_sp)
sys.modules["pca_strategy"] = strat
_sp.loader.exec_module(strat)
print("framework + strategy loaded")

# %%
# --- Cell 2: data (SWAP HERE for live data) ----------------------------------
# Synthetic factor universe with a tradeable multi-day idiosyncratic reversion.
# To use REAL data instead, replace this cell with e.g.:
#     import yfinance as yf
#     px = yf.download(tickers, start="2015-01-01")["Adj Close"]
#     returns = px.pct_change().dropna()
# The rest of the notebook is unchanged — that is the point of the data adapter.
returns = data.factor_universe(n_assets=40, n_days=1750, n_factors=3,
                               mean_reversion=0.6, seed=11)
print(returns.shape)
returns.iloc[:3, :6]

# %%
# --- Cell 3: fit the PCA factor model on TRAIN only --------------------------
# Components + scaler are frozen here and only *applied* out-of-sample later.
train_end = 500
model = strat.PCAFactorModel(n_components=3).fit(returns.iloc[:train_end])
print(f"K=3 PCs explain {model.explained_variance_ratio_.sum():.1%} of variance")
pd.Series(model.explained_variance_ratio_,
          index=[f"PC{i+1}" for i in range(3)], name="var_ratio")

# %%
# --- Cell 4: residuals & rolling z-score -------------------------------------
resid = model.residuals(returns)                 # idiosyncratic part
z = strat.residual_zscore(resid, lookback=20)    # causal rolling z
z.iloc[train_end:train_end + 3, :6]

# %%
# --- Cell 5: SIGN CHECK via information coefficient --------------------------
# Never trust the weight-sign label — verify it. A negative residual z (cheap)
# should predict a POSITIVE next-day residual return (reversion up).
ic = (-z).corrwith(resid.shift(-1), axis=0).mean()
print(f"IC(-z, next-day residual) = {ic:+.4f}")
print("Positive => long-undervalued (positive weight) is the correct sign.")

# %%
# --- Cell 6: build the dollar-neutral book & inspect a single day ------------
w = strat.construct_portfolio(z, entry_z=1.0, n_long=5, n_short=5)
day = w.iloc[train_end + 50]
print("net weight (should be ~0):", round(day.sum(), 6),
      "| gross:", round(day.abs().sum(), 3))
day[day != 0].sort_values()

# %%
# --- Cell 7: out-of-sample backtest on the held-out span ---------------------
# NOTE: this is ONE train/test split with FIXED params, for teaching the
# mechanics. The official headline (Sharpe ~1.71) comes from the full expanding
# walk-forward in run.py, which re-selects params per fold and is more
# conservative. A single fixed split like this one flatters the result — treat
# the number below as illustrative, not the reported figure.
net, w_held = strat.backtest(returns.iloc[train_end:], w.iloc[train_end:],
                             rebal_freq=3, cost_bps=5.0, return_weights=True)
summary = metrics.performance_summary(net, weights=w_held)
print(metrics.format_summary(summary))

# %%
# --- Cell 8: is the book actually factor-neutral? ----------------------------
fexp = strat.realized_factor_exposure(w_held, model, list(returns.columns))
print("avg |net loading| per PC (near zero = neutral):")
print(fexp.round(4))

# %%
# --- Cell 9: robustness — bootstrap Sharpe CI --------------------------------
boot = robustness.stationary_bootstrap_sharpe(net, n_boot=2000, seed=4)
print(f"Sharpe {boot['point']:.2f}  95% CI [{boot['lower']:.2f}, "
      f"{boot['upper']:.2f}]  P[>0]={boot['prob_positive']:.0%}")

# %%
# --- Cell 10: inline figure — equity & drawdown ------------------------------
# Any figure you like can be produced inline and dropped into your report.
eq = (1 + net).cumprod()
dd = eq / eq.cummax() - 1
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                               gridspec_kw={"height_ratios": [2, 1]})
ax1.plot(eq.index, eq.values, color="#1f3b57"); ax1.set_title("OOS equity curve")
ax2.fill_between(dd.index, dd.values, 0, color="#b03a2e", alpha=.5)
ax2.set_title("Drawdown")
plt.tight_layout(); plt.show()

# %%
# --- Cell 11: parameter-sensitivity heatmap (inline) -------------------------
half = len(returns) // 2
sens = robustness.parameter_sensitivity(
    param_x=("lookback", [20, 40, 60, 90]),
    param_y=("entry_z", [1.0, 1.5, 2.0, 2.5]),
    backtest_fn=lambda lookback, entry_z: strat.backtest(
        returns.iloc[half // 2:half],
        strat.construct_portfolio(
            strat.residual_zscore(model.residuals(returns), lookback=lookback),
            entry_z=entry_z, n_long=5, n_short=5).iloc[half // 2:half],
        rebal_freq=3, cost_bps=5.0),
    metric_fn=metrics.sharpe_ratio)
fig, ax = plt.subplots(figsize=(6, 4))
im = ax.imshow(sens.values, cmap="RdYlGn", aspect="auto")
ax.set_xticks(range(len(sens.columns))); ax.set_xticklabels(sens.columns)
ax.set_yticks(range(len(sens.index))); ax.set_yticklabels(sens.index)
ax.set_xlabel("lookback"); ax.set_ylabel("entry_z"); ax.set_title("OOS Sharpe")
plt.colorbar(im); plt.show()

# %%
# --- Cell 12: (optional) render the polished HTML report ---------------------
# Reuses the same report generator the pipeline uses.
from quant_framework.report import generate_report
path = generate_report(
    os.path.join(ROOT, "strategies", "04_pca_factor_neutral"),
    os.path.join(ROOT, "outputs", "04_pca_factor_neutral"))
print("report written to:", path)
