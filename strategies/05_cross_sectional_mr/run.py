"""
Strategy 5 runner — Cross-Sectional Mean Reversion
==================================================
Short-horizon cross-sectional reversal. The signal is real gross, but this is a
strategy where TRANSACTION COSTS are the binding constraint, so the headline
analysis is the gross-vs-net gap across cost levels and rebalance frequencies.
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
    "csmr_strategy", os.path.join(os.path.dirname(__file__), "strategy.py"))
strat = importlib.util.module_from_spec(_spec)
sys.modules["csmr_strategy"] = strat
_spec.loader.exec_module(strat)

OUT = os.path.join(ROOT, "outputs", "05_cross_sectional_mr")
os.makedirs(OUT, exist_ok=True)
COST_BPS = 5.0


def build_oos(returns, lookback, entry_threshold, rebal, cost_bps=COST_BPS):
    sc = strat.cross_sectional_score(returns, lookback=lookback, method="z_score")
    w = strat.build_portfolio(sc, n_long=10, n_short=10, entry_threshold=entry_threshold)
    return sc, w


def main():
    print("=" * 70)
    print("STRATEGY 5 — CROSS-SECTIONAL MEAN REVERSION")
    print("=" * 70)

    returns = data.cross_sectional_reversion_panel(n_assets=40, n_days=1500,
                                                   reversion=0.22, seed=31)
    print(f"Universe: {returns.shape[1]} assets x {returns.shape[0]} days")

    # ---- Sign verification via information coefficient (loser -> up) ----
    sc_full = strat.cross_sectional_score(returns, lookback=5, method="z_score")
    ic = (-sc_full).corrwith(returns.shift(-1), axis=0).mean()
    print(f"IC(-score, next-day return) = {ic:+.4f}  "
          f"(positive => losers bounce, long-loser sign is correct)")

    # ---- Expanding walk-forward ----
    n = len(returns)
    min_train, folds = 400, 5
    fold_size = (n - min_train) // folds
    grid = [(lb, et, rb) for lb in (3, 5, 7) for et in (0.3, 0.5, 0.75)
            for rb in (1, 2)]

    oos_pieces, w_pieces, records = [], [], []
    for i in range(folds):
        te_s = min_train + i * fold_size
        te_e = te_s + fold_size if i < folds - 1 else n

        best_obj, best = -np.inf, grid[0]
        for (lb, et, rb) in grid:
            _, w = build_oos(returns.iloc[:te_s], lb, et, rb)
            tr_split = max(min_train // 2, te_s - fold_size)
            net_is = strat.backtest(returns.iloc[tr_split:te_s],
                                    w.iloc[tr_split:te_s], rebal_freq=rb,
                                    cost_bps=COST_BPS)
            obj = metrics.sharpe_ratio(net_is)
            if np.isfinite(obj) and obj > best_obj:
                best_obj, best = obj, (lb, et, rb)

        _, w = build_oos(returns.iloc[:te_e], best[0], best[1], best[2])
        net, w_held = strat.backtest(returns.iloc[te_s:te_e], w.iloc[te_s:te_e],
                                     rebal_freq=best[2], cost_bps=COST_BPS,
                                     return_weights=True)
        oos_pieces.append(net)
        w_pieces.append(w_held)
        records.append({"fold": i + 1, "lookback": best[0],
                        "entry_threshold": best[1], "rebal": best[2]})

    oos = pd.concat(oos_pieces).sort_index()
    weights = pd.concat(w_pieces).sort_index()
    summ = metrics.performance_summary(oos, weights=weights)
    print("\n--- OOS performance (expanding WF, net of 5bps) ---")
    print(metrics.format_summary(summ))
    print(f"chosen params per fold: {records}")

    # ---- Headline: cost & rebalance sensitivity (in-sample, first 60%) ----
    half = int(0.6 * n)
    ins = returns.iloc[:half]
    _, w_is = build_oos(ins, lookback=5, entry_threshold=0.5, rebal=1)
    print("\n--- Cost sensitivity: annualised Sharpe (lookback=5) ---")
    cost_table = {}
    for rb in (1, 2, 3):
        row = {}
        for cb in (0, 2, 5, 10, 20):
            nt = strat.backtest(ins, w_is, rebal_freq=rb, cost_bps=cb)
            row[cb] = round(metrics.sharpe_ratio(nt), 2)
        cost_table[f"rebal_{rb}"] = row
        print(f"  rebal={rb}: " + "  ".join(f"{cb}bps={row[cb]:+.2f}"
                                            for cb in (0, 2, 5, 10, 20)))

    # ---- Robustness ----
    boot = robustness.stationary_bootstrap_sharpe(oos, n_boot=2000, seed=5)
    print(f"\nBootstrap Sharpe: {boot['point']:.2f} "
          f"(95% CI [{boot['lower']:.2f}, {boot['upper']:.2f}], "
          f"P[>0]={boot['prob_positive']:.0%})")

    sens = robustness.parameter_sensitivity(
        param_x=("lookback", [2, 3, 5, 7, 10]),
        param_y=("entry_threshold", [0.25, 0.5, 0.75, 1.0]),
        backtest_fn=lambda lookback, entry_threshold: strat.backtest(
            ins, build_oos(ins, lookback, entry_threshold, 1)[1],
            rebal_freq=1, cost_bps=COST_BPS),
        metric_fn=metrics.sharpe_ratio)

    # ---- Figures ----
    plotting.plot_equity_and_drawdown(
        oos, "Cross-Sectional MR — OOS equity & drawdown (net 5bps)",
        os.path.join(OUT, "equity_drawdown.png"))
    plotting.plot_rolling_metrics(oos, os.path.join(OUT, "rolling_metrics.png"))
    plotting.plot_monthly_heatmap(oos, os.path.join(OUT, "monthly_returns.png"))
    plotting.plot_param_heatmap(sens, os.path.join(OUT, "param_sensitivity.png"),
                                "Net Sharpe vs lookback / entry threshold")
    plotting.plot_bootstrap_sharpe(boot, os.path.join(OUT, "bootstrap_sharpe.png"))

    # Cost-sensitivity chart
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    costs = [0, 2, 5, 10, 20]
    for rb in (1, 2, 3):
        ax.plot(costs, [cost_table[f"rebal_{rb}"][c] for c in costs],
                marker="o", label=f"rebal every {rb}d")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Transaction cost (bps per side)")
    ax.set_ylabel("Annualised Sharpe")
    ax.set_title("Cross-sectional reversion: signal is real, costs bind")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(os.path.join(OUT, "cost_sensitivity.png"),
                bbox_inches="tight", dpi=140)
    plt.close(fig)

    payload = {
        "strategy": "05_cross_sectional_mr",
        "data": "synthetic cross-sectional reversion panel (seed=31)",
        "cost_bps": COST_BPS,
        "information_coefficient": float(ic),
        "oos": summ,
        "cost_sensitivity_sharpe": cost_table,
        "bootstrap_sharpe": {k: v for k, v in boot.items() if k != "distribution"},
        "wf_params": records,
    }
    with open(os.path.join(OUT, "metrics.json"), "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    print(f"\nSaved outputs to {OUT}")
    return payload


if __name__ == "__main__":
    main()
