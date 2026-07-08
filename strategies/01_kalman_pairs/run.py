"""
Strategy 1 runner — Kalman Pairs Trading
========================================
End-to-end research pipeline:
  1. Generate a reproducible cointegrated pair.
  2. Walk-forward optimise (entry_z, exit_z, delta) with strict OOS separation,
     under BOTH expanding and rolling schemes.
  3. Measure performance on the concatenated OOS track record only.
  4. Robustness: bootstrap Sharpe CI, Monte Carlo paths, parameter heatmap.
  5. Save all figures + a metrics JSON to outputs/.

Run:  python -m strategies.01_kalman_pairs.run     (from repo root)
   or: python strategies/01_kalman_pairs/run.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

# Allow running as a plain script from the repo root.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from quant_framework import metrics, robustness, plotting, data       # noqa: E402
from quant_framework.validation import WalkForwardValidator, tuple_of_series_slicer  # noqa: E402

import importlib.util                                                 # noqa: E402
_spec = importlib.util.spec_from_file_location(
    "kalman_strategy", os.path.join(os.path.dirname(__file__), "strategy.py"))
strat = importlib.util.module_from_spec(_spec)
sys.modules["kalman_strategy"] = strat        # required for @dataclass resolution
_spec.loader.exec_module(strat)

OUT = os.path.join(ROOT, "outputs", "01_kalman_pairs")
os.makedirs(OUT, exist_ok=True)
COST_BPS = 1.0


def bt_from_slice(pair_slice, entry_z, exit_z, delta):
    """(x, y) tuple slice -> net returns. Used by the walk-forward validator."""
    px, py = pair_slice
    sig = strat.generate_signals(px, py, entry_z=entry_z, exit_z=exit_z,
                                 delta=delta, lookback_vol=20)
    return strat.backtest(px, py, sig, cost_bps=COST_BPS)


def main():
    print("=" * 70)
    print("STRATEGY 1 — KALMAN PAIRS TRADING")
    print("=" * 70)

    pair = data.cointegrated_pair(n=1500, seed=42)
    px, py = pair["x"], pair["y"]
    print(f"Generated cointegrated pair: {len(pair)} obs "
          f"({pair.index[0].date()} -> {pair.index[-1].date()})")

    param_grid = {
        "entry_z": [1.5, 2.0, 2.5],
        "exit_z": [0.25, 0.5, 0.75],
        "delta": [1e-5, 1e-4, 1e-3],
    }

    results = {}
    oos_curves = {}
    for scheme in ("expanding", "rolling"):
        wf = WalkForwardValidator(n_folds=5, scheme=scheme, train_span=504,
                                  min_train=400,
                                  objective_fn=lambda r: metrics.sharpe_ratio(r))
        wf_res = wf.run(data=(px, py), param_grid=param_grid,
                        backtest_fn=bt_from_slice, slicer=tuple_of_series_slicer)
        oos = wf_res.oos_returns
        oos_curves[scheme] = oos
        summ = metrics.performance_summary(oos)
        results[scheme] = summ
        print(f"\n--- {scheme.upper()} walk-forward (OOS) ---")
        print(metrics.format_summary(summ))
        print("  chosen params per fold:")
        print(wf_res.chosen_params.to_string(index=False))

    # Use the expanding-window OOS series as the headline track record.
    oos = oos_curves["expanding"]

    # Trade-level stats (recompute signals on full sample with a central param
    # set purely for descriptive trade stats; performance numbers stay OOS).
    sig_full = strat.generate_signals(px, py, entry_z=2.0, exit_z=0.5, delta=1e-4)
    trade_pnls, holding = strat.extract_trade_pnls(px, py, sig_full, COST_BPS)
    if len(trade_pnls):
        results["expanding"]["n_trades"] = int(len(trade_pnls))
        results["expanding"]["win_rate"] = metrics.win_rate(trade_pnls)
        results["expanding"]["profit_factor"] = metrics.profit_factor(trade_pnls)
        results["expanding"]["avg_holding_days"] = float(holding.mean())

    # ---- Robustness on the headline OOS track record ----
    boot = robustness.stationary_bootstrap_sharpe(oos, n_boot=2000, seed=1)
    mc = robustness.monte_carlo_paths(oos, n_paths=1000, block=15, seed=1)
    print(f"\nBootstrap Sharpe: {boot['point']:.2f} "
          f"(95% CI [{boot['lower']:.2f}, {boot['upper']:.2f}], "
          f"P[>0]={boot['prob_positive']:.0%})")
    if mc:
        print(f"Monte Carlo: median terminal {np.median(mc['terminal']):.3f}, "
              f"5th pct max DD {np.percentile(mc['max_dd'], 5):.1%}")

    # ---- Parameter sensitivity heatmap (in-sample first half only) ----
    half = len(pair) // 2
    px_is, py_is = px.iloc[:half], py.iloc[:half]
    grid = robustness.parameter_sensitivity(
        param_x=("entry_z", [1.5, 2.0, 2.5, 3.0]),
        param_y=("exit_z", [0.25, 0.5, 0.75, 1.0]),
        backtest_fn=lambda entry_z, exit_z: strat.backtest(
            px_is, py_is,
            strat.generate_signals(px_is, py_is, entry_z=entry_z, exit_z=exit_z),
            cost_bps=COST_BPS),
        metric_fn=metrics.sharpe_ratio)

    # ---- Figures ----
    plotting.plot_equity_and_drawdown(
        oos, "Kalman Pairs — OOS equity & drawdown (expanding WF)",
        os.path.join(OUT, "equity_drawdown.png"))
    plotting.plot_rolling_metrics(oos, os.path.join(OUT, "rolling_metrics.png"))
    plotting.plot_monthly_heatmap(oos, os.path.join(OUT, "monthly_returns.png"))
    plotting.plot_annual_bars(oos, os.path.join(OUT, "annual_returns.png"))
    plotting.plot_param_heatmap(grid, os.path.join(OUT, "param_sensitivity.png"),
                                "Sharpe vs entry/exit z (in-sample)")
    plotting.plot_monte_carlo(mc, os.path.join(OUT, "monte_carlo.png"))
    plotting.plot_bootstrap_sharpe(boot, os.path.join(OUT, "bootstrap_sharpe.png"))

    # ---- Persist metrics ----
    payload = {
        "strategy": "01_kalman_pairs",
        "data": "synthetic cointegrated pair (seed=42)",
        "cost_bps": COST_BPS,
        "oos_expanding": results["expanding"],
        "oos_rolling": results["rolling"],
        "bootstrap_sharpe": {k: (v if k != "distribution" else None)
                             for k, v in boot.items()},
        "monte_carlo": {
            "median_terminal": float(np.median(mc["terminal"])) if mc else None,
            "p5_max_dd": float(np.percentile(mc["max_dd"], 5)) if mc else None,
            "p95_max_dd": float(np.percentile(mc["max_dd"], 95)) if mc else None,
        },
    }
    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nSaved outputs to {OUT}")
    return payload


if __name__ == "__main__":
    main()
