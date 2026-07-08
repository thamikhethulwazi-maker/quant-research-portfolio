"""
Combined Multi-Strategy Portfolio
=================================
Combines the three daily-return ALPHA generators — Kalman Pairs (S1), PCA
Factor-Neutral (S4), and Cross-Sectional Mean Reversion (S5) — into a single
book to illustrate diversification: near-uncorrelated edges combine into a
portfolio with a higher Sharpe than any single sleeve.

Honesty notes
-------------
* The event-driven (earnings, dispersion) and overlay (VPIN, HMM) strategies do
  not produce comparable daily return streams, so they are excluded here and
  discussed in the README instead.
* Each sleeve uses its locked configuration on a held-out span; the per-sleeve
  Sharpe here can differ slightly from the walk-forward headline because this
  uses fixed (not per-fold) parameters — this section illustrates the
  diversification math, it is not a re-run of each strategy's validation.
* The sleeves run on independent synthetic universes, so their low correlation is
  partly mechanical. On real data these strategies are empirically low- but not
  zero-correlation; the diversification *principle* is what this demonstrates.
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

from quant_framework import metrics, plotting, data                   # noqa: E402

OUT = os.path.join(ROOT, "outputs", "combined_portfolio")
os.makedirs(OUT, exist_ok=True)


def _load(mod_name, path):
    spec = importlib.util.spec_from_file_location(mod_name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = m
    spec.loader.exec_module(m)
    return m


def kalman_stream():
    s = _load("s1", os.path.join(ROOT, "strategies/01_kalman_pairs/strategy.py"))
    pair = data.cointegrated_pair(n=1500, seed=1)
    px, py = pair["x"], pair["y"]
    sig = s.generate_signals(px, py, entry_z=2.0, exit_z=0.5, delta=1e-4)
    bt = s.backtest(px, py, sig)
    return bt.rename("Kalman_Pairs")


def pca_stream():
    s = _load("s4", os.path.join(ROOT, "strategies/04_pca_factor_neutral/strategy.py"))
    ret = data.factor_universe(n_assets=40, n_days=1500, n_factors=3,
                               mean_reversion=0.6, seed=11)
    split = 500
    model = s.PCAFactorModel(n_components=3).fit(ret.iloc[:split])
    z = s.residual_zscore(model.residuals(ret), lookback=20)
    w = s.construct_portfolio(z, entry_z=1.0, n_long=5, n_short=5)
    net = s.backtest(ret.iloc[split:], w.iloc[split:], rebal_freq=3, cost_bps=5.0)
    return net.rename("PCA_Factor_Neutral")


def csmr_stream():
    s = _load("s5", os.path.join(ROOT, "strategies/05_cross_sectional_mr/strategy.py"))
    ret = data.cross_sectional_reversion_panel(n_assets=40, n_days=1500,
                                               reversion=0.22, seed=31)
    sc = s.cross_sectional_score(ret, lookback=5, method="z_score")
    w = s.build_portfolio(sc, n_long=10, n_short=10, entry_threshold=0.5)
    net = s.backtest(ret.iloc[300:], w.iloc[300:], rebal_freq=1, cost_bps=5.0)
    return net.rename("CrossSectional_MR")


def align(streams):
    """Put independent streams on a shared calendar of the common length."""
    L = min(len(s.dropna()) for s in streams)
    cols = {}
    for s in streams:
        cols[s.name] = s.dropna().iloc[-L:].values
    idx = pd.bdate_range("2018-01-01", periods=L)
    return pd.DataFrame(cols, index=idx)


def main():
    print("=" * 70)
    print("COMBINED MULTI-STRATEGY PORTFOLIO")
    print("=" * 70)

    R = align([kalman_stream(), pca_stream(), csmr_stream()])
    print(f"Aligned {R.shape[1]} sleeves over {R.shape[0]} common days\n")

    # Per-sleeve Sharpe
    print("--- Per-sleeve Sharpe (fixed-param, held-out) ---")
    for c in R.columns:
        print(f"  {c:22s}: {metrics.sharpe_ratio(R[c]):+.2f}")

    corr = R.corr()
    print("\n--- Sleeve correlation matrix ---")
    print(corr.round(2).to_string())

    # Equal-weight and inverse-vol (risk-parity-lite) combinations
    ew = R.mean(axis=1)
    inv_vol = 1.0 / R.std()
    iv_w = inv_vol / inv_vol.sum()
    rp = (R * iv_w).sum(axis=1)

    ew_summ = metrics.performance_summary(ew)
    rp_summ = metrics.performance_summary(rp)
    print("\n--- Equal-weight portfolio ---")
    print(metrics.format_summary(ew_summ))
    print("\n--- Inverse-vol (risk-parity-lite) portfolio ---")
    print(metrics.format_summary(rp_summ))

    best_sleeve = max(metrics.sharpe_ratio(R[c]) for c in R.columns)
    print(f"\nDiversification: best single sleeve Sharpe {best_sleeve:.2f} -> "
          f"equal-weight {ew_summ['sharpe']:.2f}, risk-parity {rp_summ['sharpe']:.2f}")

    # ---- Figures ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Combined equity vs individual sleeves
    fig, ax = plt.subplots(figsize=(10, 5))
    for c in R.columns:
        ax.plot((1 + R[c]).cumprod().values, lw=1, alpha=.6, label=c)
    ax.plot((1 + rp).cumprod().values, lw=2.2, color="black", label="Risk-parity portfolio")
    ax.set_title("Combined portfolio vs individual sleeves (held-out)")
    ax.set_ylabel("Growth of $1"); ax.legend(frameon=False, fontsize=8)
    fig.savefig(os.path.join(OUT, "combined_equity.png"), bbox_inches="tight", dpi=140)
    plt.close(fig)

    # Correlation heatmap
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns, fontsize=8)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                    color="white" if abs(corr.values[i, j]) > 0.5 else "black", fontsize=9)
    ax.set_title("Sleeve return correlation")
    plt.colorbar(im, fraction=0.046)
    fig.savefig(os.path.join(OUT, "correlation_heatmap.png"), bbox_inches="tight", dpi=140)
    plt.close(fig)

    plotting.plot_equity_and_drawdown(
        rp, "Risk-parity portfolio — equity & drawdown",
        os.path.join(OUT, "equity_drawdown.png"))

    payload = {
        "component": "combined_portfolio",
        "sleeves": list(R.columns),
        "per_sleeve_sharpe": {c: float(metrics.sharpe_ratio(R[c])) for c in R.columns},
        "correlation_matrix": corr.round(4).to_dict(),
        "equal_weight": ew_summ,
        "risk_parity": rp_summ,
        "best_single_sleeve_sharpe": float(best_sleeve),
    }
    with open(os.path.join(OUT, "metrics.json"), "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    print(f"\nSaved outputs to {OUT}")
    return payload


if __name__ == "__main__":
    main()
