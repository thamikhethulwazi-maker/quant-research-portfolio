"""
=============================================================================
Strategy 5: Cross-Sectional Mean Reversion Engine (Relative Extremes)
=============================================================================
Core Idea:
    Stocks (or any assets) that are relative extremes — farthest above or
    below their peers on a short lookback window — tend to revert toward the
    cross-sectional mean within 1–5 days. The engine ranks the universe by
    return z-score each day, goes long the lowest decile and short the highest,
    rebalancing daily or weekly.

Key References:
    - Lehmann, B.N. (1990). "Fads, Martingales, and Market Efficiency."
      Quarterly Journal of Economics, 105(1), 1–28.
    - Jegadeesh, N. (1990). "Evidence of Predictable Behavior of Security
      Returns." Journal of Finance, 45(3), 881–898.
    - Balvers, R. & Wu, Y. (2006). "Momentum and mean reversion across national
      equity markets." Journal of Empirical Finance, 13(1), 24–48.
    - Avellaneda, M. & Lee, J.H. (2010). "Statistical Arbitrage in the U.S.
      Equities Market." Quantitative Finance, 10(7), 761–782.

Works On: US equities, crypto top-100, futures basket, FX crosses.
=============================================================================
"""

import numpy as np
import pandas as pd
from typing import Literal


# ---------------------------------------------------------------------------
# Cross-Sectional Scorer
# ---------------------------------------------------------------------------

class CrossSectionalScorer:
    """
    Scores each asset relative to its peers using one of several methods.

    Methods
    -------
    'z_score'   : (ret - mean) / std  over lookback window
    'rank'      : cross-sectional rank normalised to [-1, +1]
    'rsi'       : RSI-based momentum oscillator, inverted for mean-reversion
    'bb_pct'    : Bollinger Band %B  (0=lower band, 1=upper band)
    """

    def __init__(
        self,
        lookback: int = 5,
        method: Literal["z_score", "rank", "rsi", "bb_pct"] = "z_score",
    ):
        self.lookback = lookback
        self.method   = method

    def score(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Compute scores on each date across all assets.

        Parameters
        ----------
        returns : pd.DataFrame (T × N) — daily returns

        Returns
        -------
        pd.DataFrame (T × N) — score matrix
            Positive score → overperformed → candidate SHORT
            Negative score → underperformed → candidate LONG
        """
        if self.method == "z_score":
            return self._z_score(returns)
        elif self.method == "rank":
            return self._rank(returns)
        elif self.method == "rsi":
            return self._rsi_score(returns)
        elif self.method == "bb_pct":
            return self._bb_pct(returns)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _z_score(self, returns: pd.DataFrame) -> pd.DataFrame:
        roll_ret  = returns.rolling(self.lookback).mean()
        roll_std  = returns.rolling(self.lookback).std().replace(0, np.nan)
        return (roll_ret - roll_ret.mean(axis=1).values.reshape(-1, 1)) / roll_std

    def _rank(self, returns: pd.DataFrame) -> pd.DataFrame:
        cumret = (1 + returns).rolling(self.lookback).apply(np.prod, raw=True) - 1
        ranked = cumret.rank(axis=1, pct=True) * 2 - 1   # normalise to [-1, +1]
        return ranked

    def _rsi_score(self, returns: pd.DataFrame) -> pd.DataFrame:
        gain = returns.clip(lower=0)
        loss = (-returns).clip(lower=0)
        avg_gain = gain.rolling(self.lookback).mean()
        avg_loss = loss.rolling(self.lookback).mean()
        rs   = avg_gain / avg_loss.replace(0, np.nan)
        rsi  = 100 - 100 / (1 + rs)
        # Convert to [-1, +1]: RSI>70 → +1 (overbought=short), RSI<30 → -1
        return (rsi - 50) / 50

    def _bb_pct(self, returns: pd.DataFrame) -> pd.DataFrame:
        cumret = returns.rolling(self.lookback).sum()
        mu     = cumret.rolling(self.lookback).mean()
        sigma  = cumret.rolling(self.lookback).std()
        upper  = mu + 2 * sigma
        lower  = mu - 2 * sigma
        pct_b  = (cumret - lower) / (upper - lower).replace(0, np.nan)
        return pct_b * 2 - 1   # normalise to [-1, +1]


# ---------------------------------------------------------------------------
# Portfolio Builder
# ---------------------------------------------------------------------------

def build_portfolio(
    scores: pd.DataFrame,
    n_long: int = 10,
    n_short: int = 10,
    entry_threshold: float = 0.5,
    weight_scheme: Literal["equal", "score", "inverse_vol"] = "equal",
    vol_lookback: int = 20,
    returns: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Build daily long/short portfolio weights from score matrix.

    Long  = bottom n_long  assets (negative scores, most underperformed)
    Short = top   n_short  assets (positive scores, most overperformed)

    Parameters
    ----------
    scores           : pd.DataFrame — score matrix (T × N)
    n_long/n_short   : int          — number of positions per leg
    entry_threshold  : float        — minimum |score| to enter trade
    weight_scheme    : str          — 'equal' | 'score' | 'inverse_vol'
    vol_lookback     : int          — days for vol estimate (inverse_vol only)
    returns          : pd.DataFrame — required if weight_scheme='inverse_vol'

    Returns
    -------
    pd.DataFrame of weights (T × N), dollar-neutral (long + short ≈ 0)
    """
    weights = pd.DataFrame(0.0, index=scores.index, columns=scores.columns)

    if weight_scheme == "inverse_vol" and returns is not None:
        vol_est = returns.rolling(vol_lookback).std()
    else:
        vol_est = None

    for date in scores.index:
        s = scores.loc[date].dropna()
        if s.empty:
            continue

        longs  = s[s < -entry_threshold].nsmallest(n_long)
        shorts = s[s >  entry_threshold].nlargest(n_short)

        if longs.empty or shorts.empty:
            continue

        # FIX (2026 audit): identical sign inversion to Strategy 4 — longs
        # were given negative weight. Longs (losers) +, shorts (winners) −.
        if weight_scheme == "equal":
            lw = pd.Series( 1.0 / len(longs),  index=longs.index)
            sw = pd.Series(-1.0 / len(shorts), index=shorts.index)

        elif weight_scheme == "score":
            lw = longs.abs()  / longs.abs().sum()
            sw = shorts.abs() / shorts.abs().sum() * -1

        elif weight_scheme == "inverse_vol" and vol_est is not None:
            if date in vol_est.index:
                v = vol_est.loc[date]
                inv_vol_l = 1.0 / v[longs.index].replace(0, np.nan)
                inv_vol_s = 1.0 / v[shorts.index].replace(0, np.nan)
                lw = ( inv_vol_l / inv_vol_l.sum()).fillna( 1.0 / len(longs))
                sw = (-inv_vol_s / inv_vol_s.sum()).fillna(-1.0 / len(shorts))
            else:
                lw = pd.Series( 1.0 / len(longs),  index=longs.index)
                sw = pd.Series(-1.0 / len(shorts), index=shorts.index)
        else:
            lw = pd.Series( 1.0 / len(longs),  index=longs.index)
            sw = pd.Series(-1.0 / len(shorts), index=shorts.index)

        weights.loc[date, lw.index] = lw.values
        weights.loc[date, sw.index] = sw.values

    return weights


# ---------------------------------------------------------------------------
# Backtester + analytics
# ---------------------------------------------------------------------------

def backtest_cs_mr(
    returns: pd.DataFrame,
    weights: pd.DataFrame,
    transaction_cost_bps: float = 5.0,
    rebal_freq: int = 1,
) -> pd.DataFrame:
    """
    Vectorised backtest with transaction cost deduction.

    Parameters
    ----------
    returns              : pd.DataFrame
    weights              : pd.DataFrame
    transaction_cost_bps : float — one-way cost in basis points
    rebal_freq           : int   — rebalance every N bars

    Returns
    -------
    pd.DataFrame: gross_ret, net_ret, cum_pnl, drawdown
    """
    tc = transaction_cost_bps / 10_000

    # Resample rebalancing
    w_rebal = weights.iloc[::rebal_freq].reindex(weights.index, method="ffill")
    w_held  = w_rebal.shift(1).fillna(0)

    gross_ret = (w_held * returns).sum(axis=1)

    # Turnover-based transaction cost
    turnover = w_held.diff().abs().sum(axis=1)
    cost     = turnover * tc
    net_ret  = gross_ret - cost

    cum_pnl  = (1 + net_ret).cumprod()
    drawdown = cum_pnl / cum_pnl.cummax() - 1

    sharpe_gross = gross_ret.mean() / gross_ret.std() * np.sqrt(252) if gross_ret.std() > 0 else 0
    sharpe_net   = net_ret.mean()   / net_ret.std()   * np.sqrt(252) if net_ret.std()   > 0 else 0

    result = pd.DataFrame({
        "gross_ret": gross_ret,
        "net_ret":   net_ret,
        "cum_pnl":   cum_pnl,
        "drawdown":  drawdown,
    })
    result.attrs.update({
        "sharpe_gross": sharpe_gross,
        "sharpe_net":   sharpe_net,
        "avg_turnover": float(turnover.mean()),
    })
    return result


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(55)
    n_assets = 50
    n_days   = 750
    assets   = [f"ASSET_{i:03d}" for i in range(n_assets)]

    # Simulate with short-term mean reversion embedded
    returns = pd.DataFrame(
        np.random.randn(n_days, n_assets) * 0.01,
        index=pd.date_range("2021-01-01", periods=n_days, freq="B"),
        columns=assets,
    )
    # Add cross-sectional mean reversion: tomorrow's return negatively
    # correlated with today's cross-sectional rank
    for i in range(1, n_days):
        prev_ret = returns.iloc[i - 1].values
        rank_pct = pd.Series(prev_ret).rank(pct=True).values - 0.5
        returns.iloc[i] -= rank_pct * 0.003  # mean reversion signal

    scorer  = CrossSectionalScorer(lookback=5, method="z_score")
    scores  = scorer.score(returns)
    weights = build_portfolio(scores, n_long=10, n_short=10,
                              entry_threshold=0.3, weight_scheme="equal")
    results = backtest_cs_mr(returns, weights, transaction_cost_bps=5, rebal_freq=1)

    print("=== Cross-Sectional Mean Reversion ===")
    print(f"Gross Sharpe : {results.attrs['sharpe_gross']:.2f}")
    print(f"Net Sharpe   : {results.attrs['sharpe_net']:.2f}")
    print(f"Max Drawdown : {results['drawdown'].min():.2%}")
    print(f"Avg Turnover : {results.attrs['avg_turnover']:.2%}")
    print(f"Final PnL    : {results['cum_pnl'].iloc[-1]:.4f}")
