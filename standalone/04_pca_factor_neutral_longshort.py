"""
=============================================================================
Strategy 4: PCA Factor Neutral Long/Short (Residual Exposure)
=============================================================================
Core Idea:
    Decompose cross-sectional returns using Principal Component Analysis.
    The first K principal components capture systematic factor risk (market,
    sector, size, momentum). The residuals — unexplained idiosyncratic returns
    — are modelled as mean-reverting Ornstein-Uhlenbeck processes.
    We go long undervalued (negative residual z-score) and short overvalued
    (positive residual z-score) stocks while being net-neutral to all K factors.

Key References:
    - Avellaneda, M. & Lee, J.H. (2010). "Statistical Arbitrage in the U.S.
      Equities Market." Quantitative Finance, 10(7), 761–782.
    - Connor, G. & Korajczyk, R.A. (1988). "Risk and Return in an Equilibrium
      APT." Journal of Financial Economics, 21(2).
    - Xiang, J. & He, L. (2022). "PCA vs ETF-based Factor Construction."
      Working Paper, University of Warsaw.

Works On: Any liquid universe — US equities, crypto top-50, futures basket.
=============================================================================
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Optional


# ---------------------------------------------------------------------------
# PCA Factor Model
# ---------------------------------------------------------------------------

class PCAFactorModel:
    """
    Fits a PCA factor model to a cross-section of returns and extracts
    idiosyncratic (residual) exposures.

    Parameters
    ----------
    n_components : int   — number of principal components (factors) to retain
    scale        : bool  — standardise returns before PCA (recommended)
    """

    def __init__(self, n_components: int = 5, scale: bool = True):
        self.n_components = n_components
        self.scale        = scale
        self.pca          = PCA(n_components=n_components)
        self.scaler       = StandardScaler()
        self._fitted      = False
        self.explained_variance_ratio_: Optional[np.ndarray] = None

    def fit(self, returns: pd.DataFrame) -> "PCAFactorModel":
        """
        Fit PCA on a (T × N) DataFrame of asset returns.

        Parameters
        ----------
        returns : pd.DataFrame — rows=dates, cols=assets
        """
        X = returns.values
        if self.scale:
            X = self.scaler.fit_transform(X)
        self.pca.fit(X)
        self._fitted = True
        self.explained_variance_ratio_ = self.pca.explained_variance_ratio_
        print(f"[PCA] {self.n_components} components explain "
              f"{self.explained_variance_ratio_.sum():.1%} of variance")
        return self

    def residuals(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Project returns onto factor space and return idiosyncratic residuals.

        residual_it = r_it - Σ_k (loading_ik × factor_kt)

        Returns
        -------
        pd.DataFrame — same shape as returns, idiosyncratic residuals
        """
        if not self._fitted:
            raise RuntimeError("Call fit() first.")

        X = returns.values
        if self.scale:
            X = self.scaler.transform(X)

        # Factor scores (T × K)
        scores = self.pca.transform(X)
        # Reconstructed systematic part (T × N)
        systematic = scores @ self.pca.components_
        # Residuals in original scale
        residuals  = X - systematic

        if self.scale:
            # Back-transform residuals to return space
            residuals = residuals * self.scaler.scale_

        return pd.DataFrame(residuals, index=returns.index, columns=returns.columns)

    def factor_loadings(self, assets: list[str]) -> pd.DataFrame:
        """Return loadings (N × K) as a DataFrame."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        cols = [f"PC{i+1}" for i in range(self.n_components)]
        return pd.DataFrame(
            self.pca.components_.T,
            index=assets,
            columns=cols,
        )


# ---------------------------------------------------------------------------
# OU Signal: residual z-score with OU speed-of-reversion filter
# ---------------------------------------------------------------------------

def ou_parameters(residual_series: pd.Series) -> dict:
    """
    Estimate Ornstein-Uhlenbeck parameters via OLS on discretised form:
        Δr_t = κ(μ - r_{t-1})Δt + σ ε_t
    Returns: {'kappa': float, 'mu': float, 'sigma': float, 'half_life': float}
    """
    r   = residual_series.dropna()
    lag = r.shift(1).dropna()
    delta = r.diff().dropna()
    # Align
    common = lag.index.intersection(delta.index)
    lag, delta = lag.loc[common], delta.loc[common]

    # OLS: delta ~ a + b * lag  =>  kappa = -b, mu = a / kappa
    X = np.column_stack([np.ones(len(lag)), lag.values])
    beta = np.linalg.lstsq(X, delta.values, rcond=None)[0]
    a, b = beta
    kappa    = max(-b, 1e-6)         # mean-reversion speed (per period)
    mu       = a / kappa if kappa > 0 else 0.0
    sigma    = delta.std()
    half_life = np.log(2) / kappa    # in periods

    return {"kappa": kappa, "mu": mu, "sigma": sigma, "half_life": half_life}


def generate_residual_zscore(
    residuals: pd.DataFrame,
    lookback: int = 60,
) -> pd.DataFrame:
    """
    Convert residual matrix into z-scores over a rolling lookback window.

    A high positive z-score → stock has outperformed factors → mean-revert → SHORT
    A high negative z-score → stock has underperformed factors → mean-revert → LONG
    """
    rolling_mean = residuals.rolling(lookback).mean()
    rolling_std  = residuals.rolling(lookback).std().replace(0, np.nan)
    z_scores     = (residuals - rolling_mean) / rolling_std
    return z_scores


# ---------------------------------------------------------------------------
# Portfolio construction: factor-neutral long/short
# ---------------------------------------------------------------------------

def construct_portfolio(
    z_scores: pd.DataFrame,
    entry_z: float = 1.5,
    n_long: int = 10,
    n_short: int = 10,
    weight_type: str = "equal",   # 'equal' or 'z_score'
) -> pd.DataFrame:
    """
    On each rebalancing date, select top/bottom n assets by z-score.

    Returns
    -------
    pd.DataFrame of target weights (rows=dates, cols=assets), summing to 0
    (dollar-neutral).
    """
    weights = pd.DataFrame(0.0, index=z_scores.index, columns=z_scores.columns)

    for date in z_scores.index:
        z = z_scores.loc[date].dropna()
        if len(z) < n_long + n_short:
            continue

        short_candidates = z[z >  entry_z].nlargest(n_short)
        long_candidates  = z[z < -entry_z].nsmallest(n_long)

        if len(short_candidates) == 0 or len(long_candidates) == 0:
            continue

        # FIX (2026 audit): the original gave LONG candidates NEGATIVE weight
        # and shorts positive — the book was inverted (buying what it meant to
        # sell). Longs (underperformers, negative z) get POSITIVE weight;
        # shorts (overperformers, positive z) get NEGATIVE weight.
        if weight_type == "z_score":
            long_w  =  long_candidates.abs() / long_candidates.abs().sum()
            short_w = -short_candidates.abs() / short_candidates.abs().sum()
        else:
            long_w  = pd.Series( 1.0 / len(long_candidates),  index=long_candidates.index)
            short_w = pd.Series(-1.0 / len(short_candidates), index=short_candidates.index)

        weights.loc[date, long_w.index]  = long_w.values
        weights.loc[date, short_w.index] = short_w.values

    return weights


# ---------------------------------------------------------------------------
# Backtester
# ---------------------------------------------------------------------------

def backtest_pca_ls(
    returns: pd.DataFrame,
    weights: pd.DataFrame,
    rebal_freq: int = 5,   # rebalance every N days
) -> pd.DataFrame:
    """
    Compute portfolio returns from weight signals and asset returns.

    Returns
    -------
    pd.DataFrame: portfolio_ret, cum_pnl, drawdown
    """
    # Hold weights for rebal_freq days
    w_rebal = weights.copy()
    mask    = np.zeros(len(weights), dtype=bool)
    mask[::rebal_freq] = True
    w_rebal = w_rebal[mask].reindex(weights.index, method="ffill").shift(1)

    port_ret = (w_rebal * returns).sum(axis=1)
    cum_pnl  = (1 + port_ret).cumprod()
    drawdown = cum_pnl / cum_pnl.cummax() - 1

    sharpe = port_ret.mean() / port_ret.std() * np.sqrt(252) if port_ret.std() > 0 else 0

    result = pd.DataFrame({
        "portfolio_ret": port_ret,
        "cum_pnl":       cum_pnl,
        "drawdown":      drawdown,
    })
    result.attrs["sharpe"] = sharpe
    return result


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(21)
    n_assets = 30
    n_days   = 500
    assets   = [f"A{i:02d}" for i in range(n_assets)]

    # Simulate returns with 3 common factors
    factors  = np.random.randn(n_days, 3) * 0.01
    loadings = np.random.randn(n_assets, 3) * 0.5
    idio     = np.random.randn(n_days, n_assets) * 0.005
    ret_matrix = factors @ loadings.T + idio

    returns = pd.DataFrame(ret_matrix,
                           index=pd.date_range("2022-01-01", periods=n_days),
                           columns=assets)

    # Fit on first 250 days, test on next 250
    train_ret = returns.iloc[:250]
    test_ret  = returns.iloc[250:]

    model = PCAFactorModel(n_components=3)
    model.fit(train_ret)

    residuals = model.residuals(test_ret)
    z_scores  = generate_residual_zscore(residuals, lookback=20)
    weights   = construct_portfolio(z_scores, entry_z=1.2, n_long=5, n_short=5)
    results   = backtest_pca_ls(test_ret, weights, rebal_freq=5)

    print("=== PCA Factor Neutral L/S ===")
    print(f"Annualised Sharpe : {results.attrs['sharpe']:.2f}")
    print(f"Max Drawdown      : {results['drawdown'].min():.2%}")
    print(f"Final Cum PnL     : {results['cum_pnl'].iloc[-1]:.4f}")
    print("\nFactor Loadings (top 5 assets):")
    print(model.factor_loadings(assets).head())
