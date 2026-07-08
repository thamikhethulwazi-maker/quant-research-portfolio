"""
Strategy 3 runner — VPIN Order-Flow Toxicity Overlay
====================================================
Question under test: does scaling a base strategy down when VPIN flags toxic
flow improve out-of-sample RISK-ADJUSTED performance and cut tail risk?

Method:
  * bucket size estimated on the first training fold's volume only.
  * expanding walk-forward selects the overlay thresholds (cut_at, floor_at) by
    maximising TRAIN Sortino of the overlaid strategy; applied OOS.
  * base vs overlaid compared on the SAME concatenated OOS periods.

Intraday bars use a periods_per_year suited to 5-minute sampling.
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
    "vpin_strategy", os.path.join(os.path.dirname(__file__), "strategy.py"))
strat = importlib.util.module_from_spec(_spec)
sys.modules["vpin_strategy"] = strat
_spec.loader.exec_module(strat)

OUT = os.path.join(ROOT, "outputs", "03_vpin_overlay")
os.makedirs(OUT, exist_ok=True)
COST_BPS = 0.5
# ~5-min bars, ~78 bars/day * 252 -> annualisation factor for intraday returns.
PPY = 78 * 252


def main():
    print("=" * 70)
    print("STRATEGY 3 — VPIN ORDER-FLOW TOXICITY OVERLAY")
    print("=" * 70)

    bars = data.toxic_flow_bars(n=6000, seed=23)
    print(f"Bars: {len(bars)} 5-min bars "
          f"({bars.index[0].date()} -> {bars.index[-1].date()})")

    # Bucket size from the first 40% (training) volume only — no leakage.
    n = len(bars)
    train_end = int(0.4 * n)
    bucket = strat.estimate_bucket_size(bars.iloc[:train_end])
    vpin = strat.compute_vpin(bars, bucket_size=bucket, n_buckets=50)
    print(f"VPIN computed (bucket≈{bucket:,.0f}). "
          f"Corr(VPIN, latent toxicity) = "
          f"{vpin.corr(bars['_toxicity']):.2f}")

    # ---- Expanding walk-forward over the overlay thresholds ----
    folds = 5
    fold_size = (n - train_end) // folds
    grid = [(c, f) for c in (0.30, 0.35, 0.40, 0.45)
            for f in (0.60, 0.70, 0.80) if f > c]

    base_pieces, over_pieces, records = [], [], []
    for i in range(folds):
        te_s = train_end + i * fold_size
        te_e = te_s + fold_size if i < folds - 1 else n
        tr = bars.iloc[:te_s]
        vp_tr = vpin.iloc[:te_s]

        best_obj, best = -np.inf, grid[0]
        for (c, f) in grid:
            bt = strat.backtest_overlay(tr, vp_tr, cut_at=c, floor_at=f, cost_bps=COST_BPS)
            obj = metrics.sortino_ratio(bt["overlay_ret"], periods_per_year=PPY)
            if np.isfinite(obj) and obj > best_obj:
                best_obj, best = obj, (c, f)

        te = bars.iloc[te_s:te_e]
        vp_te = vpin.iloc[te_s:te_e]
        bt_oos = strat.backtest_overlay(te, vp_te, cut_at=best[0], floor_at=best[1],
                                        cost_bps=COST_BPS)
        base_pieces.append(bt_oos["base_ret"])
        over_pieces.append(bt_oos["overlay_ret"])
        records.append({"fold": i + 1, "cut_at": best[0], "floor_at": best[1]})

    base_oos = pd.concat(base_pieces).sort_index()
    over_oos = pd.concat(over_pieces).sort_index()

    base_summ = metrics.performance_summary(base_oos, periods_per_year=PPY)
    over_summ = metrics.performance_summary(over_oos, periods_per_year=PPY)

    print("\n--- OOS: BASE (no overlay) ---")
    print(metrics.format_summary(base_summ))
    print("\n--- OOS: VPIN-OVERLAID ---")
    print(metrics.format_summary(over_summ))
    print(f"\nchosen thresholds per fold: {records}")

    improvement = {
        "sharpe_delta": over_summ["sharpe"] - base_summ["sharpe"],
        "sortino_delta": over_summ["sortino"] - base_summ["sortino"],
        "maxdd_delta": over_summ["max_drawdown"] - base_summ["max_drawdown"],
        "vol_reduction": base_summ["ann_vol"] - over_summ["ann_vol"],
    }
    print(f"\nOverlay effect: Sharpe {improvement['sharpe_delta']:+.2f}, "
          f"Sortino {improvement['sortino_delta']:+.2f}, "
          f"maxDD {improvement['maxdd_delta']:+.2%}, "
          f"vol {improvement['vol_reduction']:+.2%}")

    # ---- Conditional diagnostic: base performance by VPIN quintile (OOS) ----
    # Direct evidence that VPIN separates good from bad periods for the base.
    diag = pd.DataFrame({"base_ret": base_oos,
                         "vpin": vpin.reindex(base_oos.index)}).dropna()
    diag["q"] = pd.qcut(diag["vpin"], 5, labels=[f"Q{i}" for i in range(1, 6)])
    by_q = diag.groupby("q", observed=True)["base_ret"].agg(
        mean_ret="mean", vol="std", n="count")
    by_q["ann_sharpe"] = (by_q["mean_ret"] / by_q["vol"]) * np.sqrt(PPY)
    print("\n--- Base return by VPIN quintile (Q5 = most toxic) [OOS] ---")
    print(by_q.to_string())

    # ---- Robustness on the overlaid OOS stream ----
    boot = robustness.stationary_bootstrap_sharpe(over_oos, n_boot=1500,
                                                  avg_block=20, periods_per_year=PPY,
                                                  seed=3)

    # ---- Parameter sensitivity: cut_at x floor_at on training only ----
    tr_all = bars.iloc[:train_end]
    vp_all = vpin.iloc[:train_end]
    sens = robustness.parameter_sensitivity(
        param_x=("floor_at", [0.55, 0.65, 0.75, 0.85]),
        param_y=("cut_at", [0.25, 0.30, 0.35, 0.40]),
        backtest_fn=lambda cut_at, floor_at: strat.backtest_overlay(
            tr_all, vp_all, cut_at=cut_at, floor_at=max(floor_at, cut_at + 0.05),
            cost_bps=COST_BPS)["overlay_ret"],
        metric_fn=lambda r: metrics.sortino_ratio(r, periods_per_year=PPY))

    # ---- Figures ----
    plotting.plot_equity_and_drawdown(
        over_oos, "VPIN Overlay — OOS equity & drawdown (overlaid)",
        os.path.join(OUT, "equity_drawdown.png"), benchmark=base_oos)
    plotting.plot_rolling_metrics(over_oos, os.path.join(OUT, "rolling_metrics.png"),
                                  sharpe_window=1000, vol_window=500)
    plotting.plot_param_heatmap(sens, os.path.join(OUT, "param_sensitivity.png"),
                                "Overlaid Sortino vs thresholds (train)",
                                cbar_label="Sortino")
    plotting.plot_bootstrap_sharpe(boot, os.path.join(OUT, "bootstrap_sharpe.png"))

    # VPIN vs toxicity diagnostic
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 3))
    sample = slice(0, 2000)
    ax.plot(bars.index[sample], vpin.iloc[sample].values, color="#b03a2e", lw=0.8, label="VPIN")
    ax.plot(bars.index[sample], bars["_toxicity"].iloc[sample].values, color="#1f3b57",
            lw=0.8, alpha=0.6, label="latent toxicity (unobserved)")
    ax.set_title("VPIN tracks latent order-flow toxicity")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(os.path.join(OUT, "vpin_vs_toxicity.png"), bbox_inches="tight", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(by_q.index.astype(str), by_q["ann_sharpe"].values,
           color=["#1f3b57", "#2e5f7f", "#7f8c8d", "#c26a5a", "#b03a2e"])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Base strategy annualised Sharpe by VPIN quintile (OOS)")
    ax.set_ylabel("Ann. Sharpe")
    fig.savefig(os.path.join(OUT, "sharpe_by_vpin_quintile.png"),
                bbox_inches="tight", dpi=140)
    plt.close(fig)

    payload = {
        "strategy": "03_vpin_overlay",
        "data": "synthetic toxic-flow intraday bars (seed=23)",
        "cost_bps": COST_BPS,
        "oos_base": base_summ,
        "oos_overlay": over_summ,
        "overlay_effect": improvement,
        "vpin_toxicity_corr": float(vpin.corr(bars["_toxicity"])),
        "base_sharpe_by_vpin_quintile": {str(k): float(v) for k, v
                                         in by_q["ann_sharpe"].items()},
        "bootstrap_sharpe": {k: v for k, v in boot.items() if k != "distribution"},
    }
    with open(os.path.join(OUT, "metrics.json"), "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    print(f"\nSaved outputs to {OUT}")
    return payload


if __name__ == "__main__":
    main()
