"""
quant_framework.plotting
========================
Reusable, publication-quality matplotlib charts for strategy research. Every
function takes an optional `ax`/`path` so plots can be composed into dashboards
or saved individually. A single house style is applied on import.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")            # headless / deterministic rendering
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from . import metrics

# ---- house style -----------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 140,
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "figure.facecolor": "white",
})
NAVY, RED, GREY = "#1f3b57", "#b03a2e", "#7f8c8d"


def _save(fig, path):
    if path:
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
    return path


def plot_equity_and_drawdown(returns: pd.Series, title: str, path: str,
                             benchmark: pd.Series | None = None):
    """Equity curve (log-ish) above, drawdown fill below."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    eq = metrics.equity_curve(returns)
    ax1.plot(eq.index, eq.values, color=NAVY, lw=1.4, label="Strategy (OOS)")
    if benchmark is not None:
        beq = metrics.equity_curve(benchmark.reindex(returns.index).fillna(0))
        ax1.plot(beq.index, beq.values, color=GREY, lw=1.0, ls="--", label="Benchmark")
    ax1.set_ylabel("Growth of $1")
    ax1.set_title(title)
    ax1.legend(loc="upper left", frameon=False)

    dd = metrics.drawdown_series(returns)
    ax2.fill_between(dd.index, dd.values, 0, color=RED, alpha=0.5)
    ax2.set_ylabel("Drawdown")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    return _save(fig, path)


def plot_rolling_metrics(returns: pd.Series, path: str,
                         sharpe_window: int = 126, vol_window: int = 63):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    rs = metrics.rolling_sharpe(returns, sharpe_window)
    ax1.plot(rs.index, rs.values, color=NAVY, lw=1.1)
    ax1.axhline(0, color=GREY, lw=0.8)
    ax1.set_ylabel(f"Rolling Sharpe ({sharpe_window}d)")
    ax1.set_title("Rolling risk-adjusted performance")

    rv = metrics.rolling_vol(returns, vol_window)
    ax2.plot(rv.index, rv.values, color=RED, lw=1.1)
    ax2.set_ylabel(f"Rolling Vol ({vol_window}d, ann.)")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    return _save(fig, path)


def plot_monthly_heatmap(returns: pd.Series, path: str, title: str = "Monthly returns"):
    tbl = metrics.monthly_returns_table(returns)
    if tbl.empty:
        return None
    fig, ax = plt.subplots(figsize=(9, max(2.5, 0.5 * len(tbl) + 1)))
    data = tbl.values.astype(float)
    vmax = np.nanmax(np.abs(data)) or 0.01
    im = ax.imshow(data, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(tbl.columns)))
    ax.set_xticklabels(["JFMAMJJASOND"[m - 1] for m in tbl.columns])
    ax.set_yticks(range(len(tbl.index)))
    ax.set_yticklabels(tbl.index)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.1%}", ha="center", va="center", fontsize=7)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.025, format=plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    return _save(fig, path)


def plot_annual_bars(returns: pd.Series, path: str):
    ann = metrics.annual_returns(returns)
    if ann.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 3))
    colors = [NAVY if v >= 0 else RED for v in ann.values]
    ax.bar([d.year for d in ann.index], ann.values, color=colors)
    ax.axhline(0, color="black", lw=0.8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title("Annual returns")
    return _save(fig, path)


def plot_param_heatmap(grid: pd.DataFrame, path: str, title: str,
                       cbar_label: str = "Sharpe"):
    fig, ax = plt.subplots(figsize=(7, 5))
    data = grid.values.astype(float)
    im = ax.imshow(data, cmap="viridis", aspect="auto", origin="lower")
    ax.set_xticks(range(len(grid.columns)))
    ax.set_xticklabels([f"{c}" for c in grid.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(grid.index)))
    ax.set_yticklabels([f"{i}" for i in grid.index])
    ax.set_xlabel(grid.columns.name)
    ax.set_ylabel(grid.index.name)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if not np.isnan(data[i, j]):
                ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center",
                        color="white", fontsize=7)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.03, label=cbar_label)
    return _save(fig, path)


def plot_monte_carlo(mc: dict, path: str, title: str = "Monte Carlo equity paths"):
    if not mc:
        return None
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4),
                                   gridspec_kw={"width_ratios": [2, 1]})
    paths = mc["paths"]
    x = np.arange(paths.shape[1])
    sample = paths[np.random.default_rng(0).integers(0, len(paths), size=min(200, len(paths)))]
    for p in sample:
        ax1.plot(x, p, color=NAVY, alpha=0.04, lw=0.6)
    pc = mc["percentiles"]
    ax1.plot(x, pc["p50"], color="black", lw=1.5, label="Median")
    ax1.plot(x, pc["p5"], color=RED, lw=1.0, ls="--", label="5th pct")
    ax1.plot(x, pc["p95"], color="#27ae60", lw=1.0, ls="--", label="95th pct")
    ax1.set_title(title)
    ax1.set_xlabel("Period")
    ax1.set_ylabel("Growth of $1")
    ax1.legend(loc="upper left", frameon=False)

    ax2.hist(mc["max_dd"], bins=40, color=RED, alpha=0.7)
    ax2.set_title("Max drawdown distribution")
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    return _save(fig, path)


def plot_bootstrap_sharpe(boot: dict, path: str, title: str = "Bootstrap Sharpe"):
    if not boot or len(boot.get("distribution", [])) == 0:
        return None
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(boot["distribution"], bins=50, color=NAVY, alpha=0.75)
    ax.axvline(boot["point"], color="black", lw=1.5, label=f"Point {boot['point']:.2f}")
    ax.axvline(boot["lower"], color=RED, lw=1.0, ls="--",
               label=f"95% CI [{boot['lower']:.2f}, {boot['upper']:.2f}]")
    ax.axvline(boot["upper"], color=RED, lw=1.0, ls="--")
    ax.axvline(0, color=GREY, lw=0.8)
    ax.set_title(f"{title}  (P[Sharpe>0]={boot['prob_positive']:.0%})")
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, path)
