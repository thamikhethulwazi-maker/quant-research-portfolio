"""
Strategy 6 runner — HMM Regime Detection
========================================
Expanding walk-forward: the HMM is refit on each training fold and used to infer
regimes on the OOS fold. We measure (a) regime-recovery accuracy vs ground truth
and (b) whether gating a trend base strategy on the inferred regime improves
risk-adjusted OOS performance and cuts drawdown.
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
    "hmm_strategy", os.path.join(os.path.dirname(__file__), "strategy.py"))
strat = importlib.util.module_from_spec(_spec)
sys.modules["hmm_strategy"] = strat
_spec.loader.exec_module(strat)

OUT = os.path.join(ROOT, "outputs", "06_hmm_regime")
os.makedirs(OUT, exist_ok=True)
COST_BPS = 2.0


def _align_regime_labels(true_r, pred_r):
    """Best-match accuracy: our labels are vol-sorted, truth is also emitted in
    ascending-risk order (0 bull,1 bear,2 crash), so compare directly."""
    common = true_r.index.intersection(pred_r.index)
    t, p = true_r.loc[common], pred_r.loc[common]
    return float((t.values == p.values).mean())


def main():
    print("=" * 70)
    print("STRATEGY 6 — HMM REGIME DETECTION")
    print("=" * 70)

    N_REGIMES = 2   # calm vs turbulent — far more robustly estimable than 3
    panel = data.regime_switching_series(n=2500, seed=13, return_regimes=True)
    prices = panel["close"]
    # Collapse ground truth to calm(0) vs turbulent(1) = {bear, crash}.
    true_regime = (panel["true_regime"] > 0).astype(int)
    print(f"Series: {len(prices)} days; true calm/turbulent mix "
          f"{true_regime.value_counts(normalize=True).round(2).to_dict()}")

    feats = strat.build_features(prices)
    n = len(feats)
    min_train, folds = 800, 5
    fold_size = (n - min_train) // folds

    pred_pieces, records = [], []
    for i in range(folds):
        te_s = min_train + i * fold_size
        te_e = te_s + fold_size if i < folds - 1 else n
        det = strat.HMMRegimeDetector(n_regimes=N_REGIMES, seed=42).fit(feats.iloc[:te_s])
        pred = det.predict(feats.iloc[te_s:te_e])
        pred_pieces.append(pred)
        acc = _align_regime_labels(true_regime.reindex(pred.index), pred)
        records.append({"fold": i + 1, "oos_regime_accuracy": round(acc, 3)})

    pred_oos = pd.concat(pred_pieces).sort_index()
    oos_acc = _align_regime_labels(true_regime.reindex(pred_oos.index), pred_oos)
    print(f"\nOOS regime-recovery accuracy: {oos_acc:.1%}")
    print(f"per-fold: {records}")

    # Gated vs base on the OOS span
    oos_prices = prices.reindex(pred_oos.index)
    bt = strat.backtest_gated(oos_prices, pred_oos, n_regimes=N_REGIMES, cost_bps=COST_BPS)
    base_summ = metrics.performance_summary(bt["base_ret"])
    gated_summ = metrics.performance_summary(bt["gated_ret"])

    print("\n--- OOS: BASE static long (no gating) ---")
    print(metrics.format_summary(base_summ))
    print("\n--- OOS: REGIME-GATED ---")
    print(metrics.format_summary(gated_summ))

    effect = {
        "sharpe_delta": gated_summ["sharpe"] - base_summ["sharpe"],
        "sortino_delta": gated_summ["sortino"] - base_summ["sortino"],
        "maxdd_delta": gated_summ["max_drawdown"] - base_summ["max_drawdown"],
    }
    print(f"\nGating effect: Sharpe {effect['sharpe_delta']:+.2f}, "
          f"maxDD {effect['maxdd_delta']:+.2%} "
          f"(base {base_summ['max_drawdown']:.1%} -> gated {gated_summ['max_drawdown']:.1%})")

    # Return by regime (evidence the base is worst in the detected crash regime)
    reg_ret = pd.DataFrame({"base_ret": bt["base_ret"], "regime": bt["regime"]}).dropna()
    by_reg = reg_ret.groupby("regime")["base_ret"].agg(
        mean="mean", vol="std", n="count")
    by_reg["ann_sharpe"] = by_reg["mean"] / by_reg["vol"] * np.sqrt(252)
    print("\n--- Base (static long) return by DETECTED regime (0 calm / 1 turbulent) [OOS] ---")
    print(by_reg.round(4).to_string())

    boot = robustness.stationary_bootstrap_sharpe(bt["gated_ret"], n_boot=2000, seed=6)

    # ---- Figures ----
    plotting.plot_equity_and_drawdown(
        bt["gated_ret"], "HMM Regime-Gated — OOS equity & drawdown",
        os.path.join(OUT, "equity_drawdown.png"), benchmark=bt["base_ret"])
    plotting.plot_bootstrap_sharpe(boot, os.path.join(OUT, "bootstrap_sharpe.png"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Regime timeline: price shaded where turbulent detected + truth ribbon
    colors = {0: "#1f9d55", 1: "#b03a2e"}   # calm / turbulent
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 5), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(oos_prices.index, oos_prices.values, color="#22303c", lw=0.8)
    turb = (pred_oos.reindex(oos_prices.index) == 1)
    ax1.fill_between(oos_prices.index, oos_prices.min(), oos_prices.max(),
                     where=turb.values, color=colors[1], alpha=0.13,
                     label="detected turbulent")
    ax1.set_title(f"Detected turbulent regime over price (OOS accuracy {oos_acc:.0%})")
    ax1.legend(frameon=False, fontsize=8)
    tr = true_regime.reindex(oos_prices.index)
    ax2.fill_between(oos_prices.index, 0, 1, where=(tr == 0).values, color=colors[0], alpha=.5)
    ax2.fill_between(oos_prices.index, 0, 1, where=(tr == 1).values, color=colors[1], alpha=.5)
    ax2.set_ylabel("true"); ax2.set_yticks([])
    ax2.set_title("Ground truth (green=calm, red=turbulent)", fontsize=9)
    fig.savefig(os.path.join(OUT, "regime_timeline.png"), bbox_inches="tight", dpi=140)
    plt.close(fig)

    # Sharpe by detected regime
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.bar(["calm (0)", "turbulent (1)"],
           [by_reg["ann_sharpe"].get(r, 0) for r in (0, 1)],
           color=[colors[0], colors[1]])
    ax.axhline(0, color="black", lw=.8)
    ax.set_title("Static-long Sharpe by detected regime (OOS)")
    ax.set_ylabel("Ann. Sharpe")
    fig.savefig(os.path.join(OUT, "sharpe_by_regime.png"), bbox_inches="tight", dpi=140)
    plt.close(fig)

    payload = {
        "strategy": "06_hmm_regime",
        "data": "synthetic regime-switching series (seed=13)",
        "cost_bps": COST_BPS,
        "oos_regime_accuracy": oos_acc,
        "oos_base": base_summ,
        "oos_gated": gated_summ,
        "gating_effect": effect,
        "base_sharpe_by_regime": {str(k): float(by_reg["ann_sharpe"].get(k, 0))
                                  for k in (0, 1)},
        "bootstrap_sharpe": {k: v for k, v in boot.items() if k != "distribution"},
    }
    with open(os.path.join(OUT, "metrics.json"), "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    print(f"\nSaved outputs to {OUT}")
    return payload


if __name__ == "__main__":
    main()
