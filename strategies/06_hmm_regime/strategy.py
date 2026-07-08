"""
Strategy 6 — Hidden Markov Regime Detection
===========================================
Markets cycle through latent regimes (calm/bull, volatile/bear, crash). A
Gaussian HMM infers the current regime from observable features (returns,
volatility), and a trading overlay conditions leverage on it: full risk in calm
regimes, reduced in volatile ones, flat in crash.

Two honesty disciplines
------------------------
1. The HMM AND the feature scaler are fit on TRAINING data only, then frozen.
2. HMM state indices are permutation-invariant, so we PIN the state->regime map
   deterministically by sorting states on their mean realised volatility (lowest
   vol = calm). Without this, labels would shuffle across refits and nothing
   would reproduce.

Like VPIN, the HMM is a regime FILTER, not standalone alpha: it is judged by
whether gating a base strategy on the inferred regime improves risk-adjusted
performance, and separately by whether it recovers the true regimes.

Reference: Hamilton (1989); Nystrup et al. (2020).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler


def build_features(prices: pd.Series, vol_window: int = 10,
                   ret_window: int = 5) -> pd.DataFrame:
    """Causal observable features for the HMM (all trailing)."""
    logret = np.log(prices / prices.shift(1))
    feats = pd.DataFrame({
        "ret": logret.rolling(ret_window).sum(),
        "vol": logret.rolling(vol_window).std() * np.sqrt(252),
        "vol_chg": (logret.rolling(vol_window).std() * np.sqrt(252)).diff(),
    }, index=prices.index).dropna()
    return feats


class HMMRegimeDetector:
    """Gaussian HMM with a pinned, vol-sorted state->regime label map."""

    def __init__(self, n_regimes: int = 3, n_iter: int = 200, seed: int = 42):
        self.n_regimes = n_regimes
        self.model = hmm.GaussianHMM(n_components=n_regimes, covariance_type="full",
                                     n_iter=n_iter, random_state=seed)
        self.scaler = StandardScaler()
        self._fitted = False

    def fit(self, features: pd.DataFrame) -> "HMMRegimeDetector":
        X = self.scaler.fit_transform(features.values)     # scaler fit on TRAIN
        self.model.fit(X)
        # Pin labels: sort states by mean 'vol' feature (index 1) ascending.
        vol_idx = list(features.columns).index("vol")
        means_vol = self.model.means_[:, vol_idx]
        order = np.argsort(means_vol)                       # calm -> crash
        self.state_to_regime = {int(s): rank for rank, s in enumerate(order)}
        self._fitted = True
        return self

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if not self._fitted:
            raise RuntimeError("fit() first")
        X = self.scaler.transform(features.values)          # transform only
        states = self.model.predict(X)
        regimes = np.array([self.state_to_regime[int(s)] for s in states])
        return pd.Series(regimes, index=features.index, name="regime")


# Regime -> leverage overlay. Regime 0 is always the calmest (full risk); risk
# is trimmed as the regime index rises toward turbulent/crash.
REGIME_LEVERAGE = {2: {0: 1.0, 1: 0.30},                 # 2-state: calm / turbulent
                   3: {0: 1.0, 1: 0.50, 2: 0.0}}         # 3-state: calm/bear/crash


def base_signal(prices: pd.Series) -> pd.Series:
    """
    Static long exposure (buy-and-hold). This is the canonical vehicle a regime
    overlay manages: a long book that earns in calm regimes and bleeds in
    turbulent ones. The overlay's job is to trim that exposure when the HMM flags
    turbulence. (A trend-following base self-protects by going flat in
    downtrends, which partly duplicates the overlay and muddies the test — hence
    a static long base isolates the overlay's contribution.)
    """
    return pd.Series(1.0, index=prices.index)


def backtest_gated(prices: pd.Series, regimes: pd.Series, n_regimes: int = 2,
                   cost_bps: float = 2.0) -> pd.DataFrame:
    """
    Compare static long exposure to a regime-gated version. The regime leverage
    is lagged one day. Returns base and gated net returns.
    """
    lev_map = REGIME_LEVERAGE[n_regimes]
    ret = prices.pct_change().fillna(0.0)
    pos = base_signal(prices).shift(1).fillna(0.0)
    lev = regimes.map(lev_map).reindex(prices.index).ffill().shift(1).fillna(1.0)

    tc = cost_bps / 1e4
    base = pos * ret - pos.diff().abs().fillna(0) * tc
    gated_pos = pos * lev
    gated = gated_pos * ret - gated_pos.diff().abs().fillna(0) * tc
    return pd.DataFrame({"base_ret": base, "gated_ret": gated,
                         "regime": regimes.reindex(prices.index).ffill(),
                         "leverage": lev}, index=prices.index)
