"""
Strategy 4 runner — PCA Factor-Neutral Long/Short
=================================================
Expanding walk-forward where PCA is REFIT on each training fold and frozen
before being applied to the OOS fold. Parameters (entry_z, lookback) are
selected on training Sharpe only.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from quant_framework import metrics, robustness, plotting, data       # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "pca_strategy", os.path.join(os.path.dirname(__file__), "strategy.py"))
strat = importlib.util.module_from_spec(_spec)
sys.modules["pca_strategy"] = strat
_spec.loader.exec_module(strat)

OUT = os.path.join(ROOT, "outputs", "04_pca_factor_neutral")
os.makedirs(OUT, exist_ok=True)
COST_BPS = 5.0
N_PC = 3


def oos_returns_for_params(returns, train_end, test_end, entry_z, lookback,
                           n_long=5, n_short=5, rebal=3):
    """Fit PCA on returns[:train_end], apply to returns[:test_end], backtest the
    test slice. Causal: components frozen from train."""
    model = strat.PCAFactorModel(n_components=N_PC).fit(returns.iloc[:train_end])
    resid = model.residuals(returns.iloc[:test_end])
    z = strat.residual_zscore(resid, lookback=lookback)
    w = strat.construct_portfolio(z, entry_z=entry_z, n_long=n_long, n_short=n_short)
    w_test = w.iloc[train_end:test_end]
    ret_test = returns.iloc[train_end:test_end]
    net, w_held = strat.backtest(ret_test, w_test, rebal_freq=rebal,
                                 cost_bps=COST_BPS, return_weights=True)
    return net, w_held, model


def main():
    print("=" * 70)
    print("STRATEGY 4 — PCA FACTOR-NEUTRAL LONG/SHORT")
    print("=" * 70)

    returns = data.factor_universe(n_assets=40, n_days=1750, n_factors=3,
                                   mean_reversion=0.6, seed=11)
    print(f"Universe: {returns.shape[1]} assets x {returns.shape[0]} days")

    n = len(returns)
    min_train = 500
    folds = 5
    fold_size = (n - min_train) // folds
    grid = [(ez, lb) for ez in (1.0, 1.5, 2.0) for lb in (15, 20, 25)]

    oos_pieces, w_pieces, records = [], [], []
    last_model = None
    for i in range(folds):
        te_s = min_train + i * fold_size
        te_e = te_s + fold_size if i < folds - 1 else n

        best_obj, best = -np.inf, grid[0]
        for (ez, lb) in grid:
            # In-sample objective: fit on [:te_s - fold], score on train tail.
            tr_split = max(min_train // 2, te_s - fold_size)
            net_is, _, _ = oos_returns_for_params(returns, tr_split, te_s, ez, lb)
            obj = metrics.sharpe_ratio(net_is)
            if np.isfinite(obj) and obj > best_obj:
                best_obj, best = obj, (ez, lb)

        net_oos, w_oos, model = oos_returns_for_params(
            returns, te_s, te_e, best[0], best[1])
        oos_pieces.append(net_oos)
        w_pieces.append(w_oos)
        last_model = model
        records.append({"fold": i + 1, "entry_z": best[0], "lookback": best[1]})

    oos = pd.concat(oos_pieces).sort_index()
    weights = pd.concat(w_pieces).sort_index()

    summ = metrics.performance_summary(oos, weights=weights)
    print("\n--- OOS performance (expanding WF, PCA refit per fold) ---")
    print(metrics.format_summary(summ))
    print(f"chosen params per fold: {records}")
    print(f"PCA explains {last_model.explained_variance_ratio_.sum():.1%} "
          f"of variance (last fold)")

    # Factor-neutrality diagnostic
    fexp = strat.realized_factor_exposure(weights, last_model,
                                          list(returns.columns))
    print(f"Avg |net factor loading| per PC: "
          f"{ {k: round(v, 4) for k, v in fexp.items()} }")

    # ---- Robustness ----
    boot = robustness.stationary_bootstrap_sharpe(oos, n_boot=2000, seed=4)
    mc = robustness.monte_carlo_paths(oos, n_paths=1000, block=10, seed=4)
    print(f"\nBootstrap Sharpe: {boot['point']:.2f} "
          f"(95% CI [{boot['lower']:.2f}, {boot['upper']:.2f}], "
          f"P[>0]={boot['prob_positive']:.0%})")

    # ---- Parameter sensitivity (first-half in-sample) ----
    half = n // 2
    sens = robustness.parameter_sensitivity(
        param_x=("lookback", [20, 40, 60, 90]),
        param_y=("entry_z", [1.0, 1.5, 2.0, 2.5]),
        backtest_fn=lambda lookback, entry_z: oos_returns_for_params(
            returns, half // 2, half, entry_z, lookback)[0],
        metric_fn=metrics.sharpe_ratio)

    # ---- Figures ----
    plotting.plot_equity_and_drawdown(
        oos, "PCA Factor-Neutral L/S — OOS equity & drawdown",
        os.path.join(OUT, "equity_drawdown.png"))
    plotting.plot_rolling_metrics(oos, os.path.join(OUT, "rolling_metrics.png"))
    plotting.plot_monthly_heatmap(oos, os.path.join(OUT, "monthly_returns.png"))
    plotting.plot_annual_bars(oos, os.path.join(OUT, "annual_returns.png"))
    plotting.plot_param_heatmap(sens, os.path.join(OUT, "param_sensitivity.png"),
                                "Sharpe vs entry_z / lookback (in-sample)")
    plotting.plot_monte_carlo(mc, os.path.join(OUT, "monte_carlo.png"))
    plotting.plot_bootstrap_sharpe(boot, os.path.join(OUT, "bootstrap_sharpe.png"))

    payload = {
        "strategy": "04_pca_factor_neutral",
        "data": "synthetic factor universe w/ idio reversion (seed=11)",
        "cost_bps": COST_BPS, "n_components": N_PC,
        "oos": summ,
        "avg_abs_factor_loading": {k: float(v) for k, v in fexp.items()},
        "bootstrap_sharpe": {k: v for k, v in boot.items() if k != "distribution"},
        "monte_carlo": {"median_terminal": float(np.median(mc["terminal"])) if mc else None},
    }
    with open(os.path.join(OUT, "metrics.json"), "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    print(f"\nSaved outputs to {OUT}")
    return payload


if __name__ == "__main__":
    main()
