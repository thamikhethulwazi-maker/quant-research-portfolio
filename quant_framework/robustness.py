"""
quant_framework.robustness
===========================
Tools that turn a single backtest point-estimate into a *distribution*, so we
can talk honestly about how much of a result is signal and how much is luck.

Contents
--------
- stationary_bootstrap_sharpe : bootstrap CI for the Sharpe ratio that respects
  autocorrelation (Politis & Romano 1994 stationary bootstrap).
- monte_carlo_paths           : block-bootstrap resampling of the return stream
  to build a fan of equity-curve outcomes and a distribution of terminal
  wealth / max drawdown.
- parameter_sensitivity       : evaluate a metric across a 2-D parameter grid so
  robustness (a broad plateau) vs. overfitting (a lone spike) is visible.

All randomness flows through an explicit `np.random.default_rng(seed)` so every
result is reproducible.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd

from .metrics import sharpe_ratio, max_drawdown, cagr, TRADING_DAYS


# ---------------------------------------------------------------------------
# Stationary bootstrap for Sharpe confidence intervals
# ---------------------------------------------------------------------------
def _stationary_bootstrap_indices(n: int, avg_block: float, rng) -> np.ndarray:
    """Generate one stationary-bootstrap resample of indices 0..n-1."""
    p = 1.0 / max(avg_block, 1.0)
    idx = np.empty(n, dtype=int)
    idx[0] = rng.integers(0, n)
    for t in range(1, n):
        if rng.random() < p:
            idx[t] = rng.integers(0, n)          # start a new block
        else:
            idx[t] = (idx[t - 1] + 1) % n        # continue the block
    return idx


def stationary_bootstrap_sharpe(returns: pd.Series,
                                n_boot: int = 2000,
                                avg_block: float = 10.0,
                                periods_per_year: int = TRADING_DAYS,
                                ci: float = 0.95,
                                seed: int = 0) -> dict:
    """
    Bootstrap CI for the annualised Sharpe ratio.

    Returns dict: point, lower, upper, prob_positive, distribution (ndarray).
    """
    r = returns.dropna().values
    n = len(r)
    rng = np.random.default_rng(seed)
    if n < 20:
        return {"point": 0.0, "lower": 0.0, "upper": 0.0,
                "prob_positive": 0.0, "distribution": np.array([])}

    boot = np.empty(n_boot)
    for b in range(n_boot):
        sample = r[_stationary_bootstrap_indices(n, avg_block, rng)]
        sd = sample.std(ddof=1)
        boot[b] = (sample.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else 0.0

    alpha = (1 - ci) / 2
    return {
        "point": sharpe_ratio(returns, periods_per_year=periods_per_year),
        "lower": float(np.quantile(boot, alpha)),
        "upper": float(np.quantile(boot, 1 - alpha)),
        "prob_positive": float((boot > 0).mean()),
        "distribution": boot,
    }


# ---------------------------------------------------------------------------
# Monte Carlo equity-path simulation via block bootstrap
# ---------------------------------------------------------------------------
def monte_carlo_paths(returns: pd.Series,
                      n_paths: int = 1000,
                      block: int = 10,
                      seed: int = 0) -> dict:
    """
    Resample the realised return stream in contiguous blocks (preserving local
    autocorrelation) to build a distribution of possible equity paths of the
    same length.

    Returns dict with:
      paths        : ndarray (n_paths, n+1) equity curves starting at 1.0
      terminal     : ndarray of terminal wealth per path
      max_dd       : ndarray of max drawdown per path
      percentiles  : dict of p5/p50/p95 equity curves
    """
    r = returns.dropna().values
    n = len(r)
    rng = np.random.default_rng(seed)
    if n < block or n == 0:
        return {}

    n_blocks = int(np.ceil(n / block))
    paths = np.empty((n_paths, n + 1))
    max_dds = np.empty(n_paths)

    for i in range(n_paths):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        seq = np.concatenate([r[s:s + block] for s in starts])[:n]
        eq = np.empty(n + 1)
        eq[0] = 1.0
        eq[1:] = np.cumprod(1.0 + seq)
        paths[i] = eq
        peak = np.maximum.accumulate(eq)
        max_dds[i] = (eq / peak - 1.0).min()

    return {
        "paths": paths,
        "terminal": paths[:, -1],
        "max_dd": max_dds,
        "percentiles": {
            "p5": np.percentile(paths, 5, axis=0),
            "p50": np.percentile(paths, 50, axis=0),
            "p95": np.percentile(paths, 95, axis=0),
        },
    }


# ---------------------------------------------------------------------------
# 2-D parameter sensitivity grid
# ---------------------------------------------------------------------------
def parameter_sensitivity(param_x: tuple[str, Sequence],
                          param_y: tuple[str, Sequence],
                          backtest_fn: Callable[..., pd.Series],
                          metric_fn: Callable[[pd.Series], float] = sharpe_ratio,
                          fixed_params: dict | None = None) -> pd.DataFrame:
    """
    Evaluate `metric_fn(backtest_fn(**params))` over the outer product of two
    parameter ranges. Returns a DataFrame (index=param_y, cols=param_x) ready
    to feed a heatmap.
    """
    x_name, x_vals = param_x
    y_name, y_vals = param_y
    fixed = fixed_params or {}
    out = pd.DataFrame(index=list(y_vals), columns=list(x_vals), dtype=float)
    for yv in y_vals:
        for xv in x_vals:
            params = {**fixed, x_name: xv, y_name: yv}
            try:
                ret = backtest_fn(**params)
                out.loc[yv, xv] = metric_fn(ret)
            except Exception:
                out.loc[yv, xv] = np.nan
    out.index.name = y_name
    out.columns.name = x_name
    return out
