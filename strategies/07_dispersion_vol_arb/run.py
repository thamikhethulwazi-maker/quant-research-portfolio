"""
Strategy 7 runner — Dispersion Volatility Arbitrage
===================================================
Event-driven, like the earnings strategy. The implied-correlation entry
threshold is chosen on PAST events only (walk-forward) and applied to future
events. Headline is the PER-EVENT distribution — including the crisis left tail —
not a diversified aggregate.
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
    "disp_strategy", os.path.join(os.path.dirname(__file__), "strategy.py"))
strat = importlib.util.module_from_spec(_spec)
sys.modules["disp_strategy"] = strat
_spec.loader.exec_module(strat)

OUT = os.path.join(ROOT, "outputs", "07_dispersion_vol_arb")
os.makedirs(OUT, exist_ok=True)


def per_event_sharpe(pnl: pd.Series) -> float:
    return float(pnl.mean() / pnl.std()) if len(pnl) > 1 and pnl.std() > 0 else 0.0


def main():
    print("=" * 70)
    print("STRATEGY 7 — DISPERSION VOLATILITY ARBITRAGE")
    print("=" * 70)

    events = data.dispersion_vol_panel(n_events=300, n_stocks=12, seed=41)
    print(f"Events: {len(events)}  "
          f"(crisis fraction realised>implied corr: "
          f"{(events['realised_corr'] > events['implied_corr']).mean():.1%})")

    # ---- Walk-forward over the implied-correlation entry threshold ----
    n = len(events)
    min_train, folds = 100, 5
    fold_size = (n - min_train) // folds
    grid = [0.45, 0.50, 0.55, 0.60, 0.65]

    oos_pieces, records = [], []
    for i in range(folds):
        te_s = min_train + i * fold_size
        te_e = te_s + fold_size if i < folds - 1 else n
        train_ev = events.iloc[:te_s]

        best_obj, best_thr = -np.inf, grid[0]
        for thr in grid:
            r = strat.run_events(train_ev, thr)
            obj = per_event_sharpe(r["net_pnl"]) if not r.empty else -np.inf
            if obj > best_obj:
                best_obj, best_thr = obj, thr

        oos_r = strat.run_events(events.iloc[te_s:te_e], best_thr)
        if not oos_r.empty:
            oos_pieces.append(oos_r)
        records.append({"fold": i + 1, "corr_threshold": best_thr,
                        "n_trades": 0 if oos_r.empty else len(oos_r)})

    oos = pd.concat(oos_pieces).sort_index().reset_index(drop=True)
    pnl = oos["net_pnl"]

    # ---- Per-event statistics (the honest headline) ----
    sharpe = per_event_sharpe(pnl)
    win_rate = float((pnl > 0).mean())
    worst = float(pnl.min())
    avg_spread = float(oos["corr_spread"].mean())
    print(f"\n--- OOS per-event results ({len(pnl)} trades) ---")
    print(f"Per-event Sharpe : {sharpe:.2f}")
    print(f"Win rate         : {win_rate:.1%}")
    print(f"Avg net P&L      : {pnl.mean():.5f} (variance pts)")
    print(f"Worst event      : {worst:.5f}  (the correlation-spike tail)")
    print(f"Avg corr spread  : {avg_spread:+.3f} (implied − realised)")
    print(f"chosen thresholds: {records}")

    # Cumulative P&L series (equal size per event)
    cum = pnl.cumsum()

    boot = robustness.stationary_bootstrap_sharpe(pnl, n_boot=3000, avg_block=5,
                                                  periods_per_year=1, seed=7)
    print(f"\nBootstrap per-event Sharpe: {boot['point']:.2f} "
          f"(95% CI [{boot['lower']:.2f}, {boot['upper']:.2f}], "
          f"P[>0]={boot['prob_positive']:.0%})")

    # ---- Figures ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Cumulative P&L
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(len(cum)), cum.values, color="#1f3b57")
    ax.set_title("Dispersion — cumulative per-event P&L (OOS, variance pts)")
    ax.set_xlabel("trade #"); ax.set_ylabel("cumulative net P&L")
    ax.axhline(0, color="black", lw=.7)
    fig.savefig(os.path.join(OUT, "cumulative_pnl.png"), bbox_inches="tight", dpi=140)
    plt.close(fig)

    # Event P&L histogram (shows crisis left tail)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(pnl.values, bins=40, color="#1f3b57", alpha=.85)
    ax.axvline(0, color="black", lw=.8)
    ax.axvline(pnl.mean(), color="#1f9d55", ls="--", label=f"mean {pnl.mean():.4f}")
    ax.axvline(worst, color="#b03a2e", ls="--", label=f"worst {worst:.4f}")
    ax.set_title("Per-event P&L distribution (left tail = correlation spikes)")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(os.path.join(OUT, "event_pnl_hist.png"), bbox_inches="tight", dpi=140)
    plt.close(fig)

    # Implied vs realised correlation scatter, coloured by P&L sign
    fig, ax = plt.subplots(figsize=(6, 6))
    win = pnl > 0
    ax.scatter(oos.loc[win, "implied_corr"], oos.loc[win, "realised_corr"],
               s=18, color="#1f9d55", alpha=.6, label="win")
    ax.scatter(oos.loc[~win, "implied_corr"], oos.loc[~win, "realised_corr"],
               s=18, color="#b03a2e", alpha=.7, label="loss")
    lims = [0.2, 1.0]
    ax.plot(lims, lims, color="black", lw=.8, ls="--", label="realised = implied")
    ax.set_xlabel("implied correlation"); ax.set_ylabel("realised correlation")
    ax.set_title("Wins below the diagonal; losses above (corr spike)")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(os.path.join(OUT, "corr_scatter.png"), bbox_inches="tight", dpi=140)
    plt.close(fig)

    plotting.plot_bootstrap_sharpe(boot, os.path.join(OUT, "bootstrap_sharpe.png"))

    payload = {
        "strategy": "07_dispersion_vol_arb",
        "data": "synthetic dispersion vol panel with crisis tail (seed=41)",
        "n_oos_trades": int(len(pnl)),
        "per_event_sharpe": sharpe,
        "win_rate": win_rate,
        "avg_net_pnl": float(pnl.mean()),
        "worst_event": worst,
        "avg_corr_spread": avg_spread,
        "bootstrap_sharpe": {k: v for k, v in boot.items() if k != "distribution"},
        "wf_thresholds": records,
    }
    with open(os.path.join(OUT, "metrics.json"), "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    print(f"\nSaved outputs to {OUT}")
    return payload


if __name__ == "__main__":
    main()
