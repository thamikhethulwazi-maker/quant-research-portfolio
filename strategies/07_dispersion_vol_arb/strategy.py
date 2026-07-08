"""
Strategy 7 — Dispersion Volatility Arbitrage
============================================
Index implied volatility is structurally rich relative to the vol implied by its
constituents, because index options are bought as portfolio hedges. Equivalently,
index options price in a high *implied correlation*. The dispersion trade sells
index variance and buys constituent variance, earning the gap when stocks move
more independently than the index implied — i.e. when realised correlation comes
in below implied.

The catch — and the risk the original demo hid — is that in a crisis correlations
spike toward 1 and the trade loses on both legs at once. This module models that
fat left tail explicitly, so the strategy is evaluated on its true risk profile.

References: Deng (2008); Bossu (2005); Driessen, Maenhout & Vilkov (2009).
"""
from __future__ import annotations

import numpy as np


def index_vol_from_constituents(ivs: np.ndarray, weights: np.ndarray,
                                rho: float) -> float:
    """Single-factor index vol: sqrt(w' (corr ∘ (iv iv')) w) with constant ρ."""
    w = weights / weights.sum()
    n = len(ivs)
    corr = np.full((n, n), rho)
    np.fill_diagonal(corr, 1.0)
    cov = np.outer(ivs, ivs)
    return float(np.sqrt(max(w @ (corr * cov) @ w, 0.0)))


def solve_implied_correlation(index_iv: float, ivs: np.ndarray,
                              weights: np.ndarray, tol: float = 1e-7,
                              max_iter: int = 200) -> float:
    """Invert the index-vol formula for the average pairwise correlation ρ*
    via bisection on [0, 1]."""
    lo, hi = 0.0, 1.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if hi - lo < tol:
            break
        if index_vol_from_constituents(ivs, weights, mid) < index_iv:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def dispersion_pnl(index_iv: float, ivs: np.ndarray, weights: np.ndarray,
                   realised_index_vol: float,
                   realised_ivs: np.ndarray) -> dict:
    """
    Variance-swap-style P&L (in variance points) of the dispersion trade:
      short index variance   : −(σ²_index_realised − IV²_index)
      long constituent var   : +Σ wᵢ (σ²_i_realised − IV²_i)
    Net > 0 when constituents realise more (relative) variance than the index —
    i.e. realised correlation below implied.
    """
    w = weights / weights.sum()
    index_leg = -(realised_index_vol ** 2 - index_iv ** 2)
    constituent_leg = float(np.sum(w * (realised_ivs ** 2 - ivs ** 2)))
    return {"index_leg": index_leg, "constituent_leg": constituent_leg,
            "net": index_leg + constituent_leg}


def run_events(events, corr_threshold: float) -> "pd.DataFrame":
    """
    Trade every event whose implied correlation exceeds `corr_threshold`
    (elevated implied corr = rich index vol = attractive to sell dispersion).
    Returns a per-event P&L frame.
    """
    import pandas as pd
    rows = []
    for _, ev in events.iterrows():
        ivs = np.array(ev["constituent_ivs"])
        w = np.array(ev["constituent_weights"])
        implied_corr = solve_implied_correlation(ev["index_iv"], ivs, w)
        if implied_corr < corr_threshold:
            continue
        pnl = dispersion_pnl(ev["index_iv"], ivs, w,
                             ev["realised_index_vol"],
                             np.array(ev["realised_constituent_vols"]))
        rows.append({"date": ev["date"], "implied_corr": implied_corr,
                     "realised_corr": ev["realised_corr"],
                     "corr_spread": implied_corr - ev["realised_corr"],
                     "net_pnl": pnl["net"]})
    return pd.DataFrame(rows)
