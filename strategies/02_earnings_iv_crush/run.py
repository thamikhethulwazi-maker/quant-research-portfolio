"""
Strategy 2 runner — Earnings IV Crush (Variance Risk Premium)
=============================================================
Expanding walk-forward over earnings QUARTERS with strict leakage discipline:

  * For each test quarter q, estimate every name's premium using ONLY events
    from quarters < q (the training set), then trade quarter q out-of-sample.
  * Concatenate the OOS quarters into one honest track record.

This is a bespoke (rather than generic) walk-forward because the strategy fits a
per-name *model* (the premium map), not just scalar hyper-parameters — and that
model must be estimated on past events only.

Run:  python strategies/02_earnings_iv_crush/run.py
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
    "iv_crush_strategy", os.path.join(os.path.dirname(__file__), "strategy.py"))
strat = importlib.util.module_from_spec(_spec)
sys.modules["iv_crush_strategy"] = strat
_spec.loader.exec_module(strat)

OUT = os.path.join(ROOT, "outputs", "02_earnings_iv_crush")
os.makedirs(OUT, exist_ok=True)
COST_BPS = 15.0


def expanding_walk_forward(panel: pd.DataFrame, min_train_quarters: int = 4,
                           min_premium: float = 0.002, min_iv_pre: float = 0.30):
    """Return concatenated OOS trades and per-quarter records (leakage-free)."""
    quarters = sorted(panel["earnings_date"].unique())
    all_trades, records = [], []
    for i, q in enumerate(quarters):
        if i < min_train_quarters:
            continue
        train = panel[panel["earnings_date"] < q]
        test = panel[panel["earnings_date"] == q]
        name_prem = strat.estimate_name_premium(train)          # fit on past only
        trades = strat.trade_events(test, name_prem,
                                    min_premium=min_premium,
                                    min_iv_pre=min_iv_pre, cost_bps=COST_BPS)
        if not trades.empty:
            all_trades.append(trades)
            records.append({"quarter": pd.Timestamp(q).date(),
                            "n_trades": len(trades),
                            "avg_net": float(trades["net_ret"].mean())})
    trades_df = pd.concat(all_trades) if all_trades else pd.DataFrame()
    return trades_df, pd.DataFrame(records)


def main():
    print("=" * 70)
    print("STRATEGY 2 — EARNINGS IV CRUSH (Variance Risk Premium)")
    print("=" * 70)

    panel = data.earnings_iv_panel(n_names=40, quarters=16, seed=7)
    print(f"Earnings panel: {len(panel)} events, {panel['name'].nunique()} names, "
          f"{panel['earnings_date'].nunique()} quarters")
    print(f"Sanity — mean implied move {panel['implied_move_pct'].mean():.3f} "
          f"vs realised {panel['realized_move_pct'].mean():.3f} "
          f"(premium exists if implied > realised)")

    trades, fold_rec = expanding_walk_forward(panel)
    oos = strat.events_to_periodic_returns(trades)
    if oos.empty:
        print("No OOS trades produced — check selection thresholds.")
        return

    # HEADLINE: per-event returns. This is the honest risk unit — it does NOT
    # let ~30 simultaneous earnings events diversify away the idiosyncratic move
    # noise, which real earnings clustering would not allow.
    per_event = pd.Series(trades["net_ret"].values,
                          index=pd.to_datetime(trades["earnings_date"].values)).sort_index()
    summ = metrics.performance_summary(per_event,
                                       trade_pnls=trades["net_ret"].values,
                                       periods_per_year=4)
    summ["avg_holding_days"] = 2.0  # enter T-1, exit T+1
    print("\n--- OOS performance (per-event, expanding WF) [HEADLINE] ---")
    print(metrics.format_summary(summ))

    # DIAGNOSTIC ONLY: quarterly-aggregated Sharpe. Reported to make the point
    # that naive cross-sectional aggregation massively inflates Sharpe by hiding
    # per-event tail risk — NOT used as the headline.
    q_sharpe = metrics.sharpe_ratio(oos, periods_per_year=4)
    print(f"\n[diagnostic] quarterly-aggregated Sharpe = {q_sharpe:.1f} "
          f"(inflated by fake diversification — see README)")
    print(f"\nPer-quarter OOS record:\n{fold_rec.to_string(index=False)}")
    print(f"Worst single event: {trades['net_ret'].min():.2%} of notional "
          f"(the short-vol tail)")
    boot = robustness.stationary_bootstrap_sharpe(per_event, n_boot=2000,
                                                  avg_block=5, periods_per_year=4,
                                                  seed=2)
    mc = robustness.monte_carlo_paths(per_event, n_paths=1000, block=8, seed=2)
    print(f"\nBootstrap Sharpe (per-event): {boot['point']:.2f} "
          f"(95% CI [{boot['lower']:.2f}, {boot['upper']:.2f}], "
          f"P[>0]={boot['prob_positive']:.0%})")

    # ---- Parameter sensitivity: selection thresholds ----
    def bt_thresholds(min_premium, min_iv_pre):
        t, _ = expanding_walk_forward(panel, min_premium=min_premium,
                                      min_iv_pre=min_iv_pre)
        if t.empty:
            return pd.Series([0.0])
        return pd.Series(t["net_ret"].values,
                         index=pd.to_datetime(t["earnings_date"].values)).sort_index()

    grid = robustness.parameter_sensitivity(
        param_x=("min_iv_pre", [0.25, 0.35, 0.45, 0.55]),
        param_y=("min_premium", [0.000, 0.002, 0.004, 0.006]),
        backtest_fn=bt_thresholds,
        metric_fn=lambda r: metrics.sharpe_ratio(r, periods_per_year=4))

    # ---- Figures ----
    plotting.plot_equity_and_drawdown(
        oos, "Earnings IV Crush — OOS equity & drawdown (expanding WF)",
        os.path.join(OUT, "equity_drawdown.png"))
    plotting.plot_annual_bars(oos, os.path.join(OUT, "annual_returns.png"))
    plotting.plot_param_heatmap(
        grid, os.path.join(OUT, "param_sensitivity.png"),
        "OOS Sharpe vs selection thresholds", cbar_label="Sharpe")
    plotting.plot_monte_carlo(mc, os.path.join(OUT, "monte_carlo.png"))
    plotting.plot_bootstrap_sharpe(boot, os.path.join(OUT, "bootstrap_sharpe.png"))

    # Distribution of per-event PnL (shows the fat-tailed short-vol loser risk)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.hist(trades["net_ret"].values, bins=40, color="#1f3b57", alpha=0.8)
    ax.axvline(0, color="black", lw=1)
    ax.axvline(trades["net_ret"].mean(), color="#b03a2e", lw=1.5,
               label=f"mean {trades['net_ret'].mean():.3%}")
    ax.set_title("Per-event net P&L distribution (note the left tail)")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1%}"))
    ax.legend(frameon=False)
    fig.savefig(os.path.join(OUT, "event_pnl_hist.png"), bbox_inches="tight", dpi=140)
    plt.close(fig)

    payload = {
        "strategy": "02_earnings_iv_crush",
        "data": "synthetic earnings panel (seed=7)",
        "cost_bps": COST_BPS,
        "oos": summ,
        "bootstrap_sharpe": {k: v for k, v in boot.items() if k != "distribution"},
        "monte_carlo": {
            "median_terminal": float(np.median(mc["terminal"])) if mc else None,
            "p5_max_dd": float(np.percentile(mc["max_dd"], 5)) if mc else None,
        },
        "left_tail_worst_event": float(trades["net_ret"].min()),
    }
    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nSaved outputs to {OUT}")
    return payload


if __name__ == "__main__":
    main()
