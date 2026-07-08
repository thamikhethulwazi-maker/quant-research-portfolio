"""
Strategy 4 — PCA Factor-Neutral Long/Short
==========================================
Decompose the cross-section of returns with PCA. The first K principal
components are systematic factor risk (market, sectors, styles); the residuals
are idiosyncratic and tend to mean-revert. Go long the most negative residual
z-scores and short the most positive, staying (approximately) neutral to the K
factors.

Leakage discipline
------------------
The PCA components and the standardiser are **fit on training data only** and
then FROZEN. When applied out-of-sample they merely *transform* new returns —
they never see future data. Refitting PCA on the full sample would leak the
future factor structure into past decisions; this module never does that.

Reference: Avellaneda & Lee (2010), *Statistical Arbitrage in the U.S. Equities
Market*, Quantitative Finance 10(7).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class PCAFactorModel:
    """PCA factor model. Fit once on training returns; then only transform."""

    def __init__(self, n_components: int = 5, scale: bool = True):
        self.n_components = n_components
        self.scale = scale
        self.pca = PCA(n_components=n_components)
        self.scaler = StandardScaler()
        self._fitted = False

    def fit(self, returns: pd.DataFrame) -> "PCAFactorModel":
        X = returns.values
        if self.scale:
            X = self.scaler.fit_transform(X)
        self.pca.fit(X)
        self._fitted = True
        self.explained_variance_ratio_ = self.pca.explained_variance_ratio_
        return self

    def residuals(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Idiosyncratic residuals using the FROZEN train-fitted model."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        X = returns.values
        if self.scale:
            X = self.scaler.transform(X)          # transform only, no re-fit
        scores = self.pca.transform(X)
        systematic = scores @ self.pca.components_
        resid = X - systematic
        if self.scale:
            resid = resid * self.scaler.scale_    # back to return space
        return pd.DataFrame(resid, index=returns.index, columns=returns.columns)

    def loadings(self, assets: list[str]) -> pd.DataFrame:
        cols = [f"PC{i+1}" for i in range(self.n_components)]
        return pd.DataFrame(self.pca.components_.T, index=assets, columns=cols)


def residual_zscore(residuals: pd.DataFrame, lookback: int = 60) -> pd.DataFrame:
    """Rolling (causal) z-score of each asset's residual series."""
    mu = residuals.rolling(lookback).mean()
    sd = residuals.rolling(lookback).std().replace(0, np.nan)
    return (residuals - mu) / sd


def construct_portfolio(z: pd.DataFrame, entry_z: float = 1.5,
                        n_long: int = 5, n_short: int = 5) -> pd.DataFrame:
    """
    Dollar-neutral L/S weights: long the most negative z (cheap residual),
    short the most positive (rich residual). Equal-weighted within each leg.
    """
    w = pd.DataFrame(0.0, index=z.index, columns=z.columns)
    for date in z.index:
        row = z.loc[date].dropna()
        if len(row) < n_long + n_short:
            continue
        shorts = row[row > entry_z].nlargest(n_short)
        longs = row[row < -entry_z].nsmallest(n_long)
        if longs.empty or shorts.empty:
            continue
        # Long the undervalued (negative z, expected to revert UP) => +weight.
        # Short the overvalued (positive z, expected to revert DOWN) => -weight.
        w.loc[date, longs.index] = 1.0 / len(longs)
        w.loc[date, shorts.index] = -1.0 / len(shorts)
    return w


def backtest(returns: pd.DataFrame, weights: pd.DataFrame,
             rebal_freq: int = 5, cost_bps: float = 5.0,
             return_weights: bool = False):
    """
    Hold weights for `rebal_freq` days; positions lagged one day; turnover-based
    costs. Returns a net periodic return series (or (net, held_weights) if
    return_weights=True, for accurate exposure/turnover reporting).
    """
    mask = np.zeros(len(weights), dtype=bool)
    mask[::rebal_freq] = True
    w_held = weights[mask].reindex(weights.index, method="ffill").shift(1).fillna(0.0)
    gross = (w_held * returns).sum(axis=1)
    turnover = w_held.diff().abs().sum(axis=1).fillna(0.0)
    net = gross - turnover * (cost_bps / 1e4)
    net.name = "net_ret"
    return (net, w_held) if return_weights else net


def realized_factor_exposure(weights: pd.DataFrame, model: PCAFactorModel,
                             assets: list[str]) -> pd.Series:
    """
    Diagnostic: average absolute net portfolio loading on each PC. Near-zero
    means the L/S book is genuinely factor-neutral (not just claimed to be).
    """
    L = model.loadings(assets)                      # N x K
    exposures = weights.fillna(0.0).values @ L.values   # T x K
    return pd.Series(np.abs(exposures).mean(axis=0), index=L.columns)
