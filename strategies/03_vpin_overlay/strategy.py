"""
Strategy 3 — VPIN Order-Flow Toxicity Overlay
=============================================
VPIN (Volume-Synchronised Probability of Informed Trading) measures order-flow
imbalance in *volume time* rather than clock time. Elevated VPIN => high adverse-
selection risk => liquidity is about to deteriorate. VPIN is NOT a standalone
signal; it is a **risk overlay** that scales a base strategy down when flow turns
toxic.

This module preserves the original VPIN estimator (bulk-volume classification +
volume buckets), which was already causal, and adds the missing pieces needed to
test it honestly:

  * bucket size is estimated on TRAINING volume only (was full-sample -> leak),
  * the size multiplier is LAGGED before it scales any position,
  * a base strategy is supplied so the overlay has returns to modulate, and the
    overlaid strategy is compared to the base out-of-sample.

Reference: Easley, Lopez de Prado & O'Hara (2012), RFS 25(5).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Bulk Volume Classification (causal): split each bar's volume into buy/sell
# ---------------------------------------------------------------------------
def bulk_volume_classify(close: np.ndarray, volume: np.ndarray,
                         sigma_window: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """
    Fraction of each bar's volume classified as buy via BVC:
        Z = Phi(dP / sigma_dP);  V_B = Z*V;  V_S = (1-Z)*V.
    sigma_dP is a trailing rolling std (past-only), so the classification is
    causal.
    """
    delta_p = np.diff(close, prepend=close[0])
    sigma_dp = (pd.Series(delta_p).rolling(sigma_window, min_periods=1)
                .std().fillna(1e-8).values)
    sigma_dp = np.where(sigma_dp == 0, 1e-8, sigma_dp)
    z = norm.cdf(delta_p / sigma_dp)
    return z * volume, (1 - z) * volume


# ---------------------------------------------------------------------------
# VPIN estimator over volume buckets (preserved from original, causal)
# ---------------------------------------------------------------------------
class VPINEstimator:
    def __init__(self, bucket_size: float, n_buckets: int = 50):
        self.V_star = bucket_size
        self.n_buckets = n_buckets
        self._vb = 0.0
        self._vs = 0.0
        self._imb: list[float] = []

    def feed(self, vb_bar: float, vs_bar: float) -> Optional[float]:
        rvb, rvs, vpin = vb_bar, vs_bar, None
        while (rvb + rvs) > 0:
            needed = self.V_star - (self._vb + self._vs)
            available = rvb + rvs
            if available < needed:
                self._vb += rvb
                self._vs += rvs
                rvb = rvs = 0
            else:
                frac = needed / available
                self._vb += frac * rvb
                self._vs += frac * rvs
                rvb *= (1 - frac)
                rvs *= (1 - frac)
                self._imb.append(abs(self._vb - self._vs) / self.V_star)
                if len(self._imb) > self.n_buckets:
                    self._imb.pop(0)
                vpin = float(np.mean(self._imb))
                self._vb = self._vs = 0.0
        return vpin


def compute_vpin(bars: pd.DataFrame, bucket_size: float,
                 n_buckets: int = 50) -> pd.Series:
    """Return a forward-filled VPIN series aligned to `bars` (causal)."""
    vb, vs = bulk_volume_classify(bars["close"].values, bars["volume"].values)
    est = VPINEstimator(bucket_size=bucket_size, n_buckets=n_buckets)
    vals = [est.feed(b, s) for b, s in zip(vb, vs)]
    return pd.Series(vals, index=bars.index).ffill()


def estimate_bucket_size(train_bars: pd.DataFrame, buckets_per_window: int = 50) -> float:
    """Bucket size from TRAINING volume only: mean bar volume scaled so that a
    window holds ~buckets_per_window buckets. Never touches test data."""
    return float(train_bars["volume"].mean())


# ---------------------------------------------------------------------------
# Overlay: map VPIN -> size multiplier in [0,1]
# ---------------------------------------------------------------------------
def size_multiplier(vpin: pd.Series, cut_at: float = 0.35, floor_at: float = 0.70) -> pd.Series:
    """
    Linearly de-risk between `cut_at` (start reducing) and `floor_at` (flat).
    Returns a multiplier in [0,1]. This is applied LAGGED by the backtest.
    """
    m = 1.0 - (vpin - cut_at) / (floor_at - cut_at)
    return m.clip(0.0, 1.0).where(vpin.notna(), 1.0)


# ---------------------------------------------------------------------------
# Base strategy: short-term mean reversion (a liquidity-provision proxy)
# ---------------------------------------------------------------------------
def base_signal(bars: pd.DataFrame, lookback: int = 6) -> pd.Series:
    """
    Contrarian / liquidity-provision base: FADE the recent move (buy dips, sell
    rips) over `lookback` bars. This is the class of strategy VPIN exists to
    protect — a mean-reverter gets adversely selected exactly when informed
    (toxic) flow trends the price against the fade. +1 long / -1 short.
    """
    mom = bars["close"].pct_change(lookback)
    return (-np.sign(mom)).fillna(0.0)


# ---------------------------------------------------------------------------
# Backtest: base vs VPIN-overlaid, both fully lagged
# ---------------------------------------------------------------------------
def backtest_overlay(bars: pd.DataFrame, vpin: pd.Series,
                     lookback: int = 6, cut_at: float = 0.35,
                     floor_at: float = 0.70, cost_bps: float = 1.0) -> pd.DataFrame:
    """
    Returns a DataFrame with base and overlaid net returns.

    position and size multiplier are both lagged one bar (set on prior close,
    earn this bar's return). Cost charged on turnover of the *effective*
    (size-scaled) position.
    """
    ret = bars["close"].pct_change().fillna(0.0)
    pos = base_signal(bars, lookback).shift(1).fillna(0.0)
    mult = size_multiplier(vpin, cut_at, floor_at).shift(1).fillna(1.0)

    eff_base = pos
    eff_over = pos * mult

    tc = cost_bps / 1e4
    base_ret = eff_base * ret - eff_base.diff().abs().fillna(0) * tc
    over_ret = eff_over * ret - eff_over.diff().abs().fillna(0) * tc

    return pd.DataFrame({"base_ret": base_ret, "overlay_ret": over_ret,
                         "vpin": vpin, "mult": mult}, index=bars.index)
