"""
=============================================================================
Strategy 1: Kalman Filter Pairs Trading (Adaptive Signals)
=============================================================================
Core Idea:
    Dynamically estimate the hedge ratio between two correlated assets using
    a Kalman filter. The spread is modelled as an Ornstein–Uhlenbeck process;
    when it deviates beyond a z-score threshold we enter mean-reversion trades.

Key References:
    - Kalman, R.E. (1960). "A New Approach to Linear Filtering and Prediction
      Problems." Transactions of the ASME—Journal of Basic Engineering.
    - Chan, E.P. (2013). "Algorithmic Trading." Wiley. (Chapter on Kalman pairs)
    - Clegg, M. & Krauss, C. (2018). "Pairs trading with partial cointegration."
      Quantitative Finance, 18(1), 121–138.

Works On: Equities, ETFs, Crypto — any two correlated assets.
=============================================================================
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Kalman Filter: State-space model for dynamic hedge ratio estimation
# ---------------------------------------------------------------------------
# State:   β_t  = hedge ratio  (scalar or vector)
# Obs:     y_t  = α + β_t * x_t + ε_t,   ε ~ N(0, R)
# Process: β_t  = β_{t-1} + w_t,          w ~ N(0, Q)
# ---------------------------------------------------------------------------

@dataclass
class KalmanPairsFilter:
    """
    Univariate Kalman filter for estimating the time-varying hedge ratio
    β between asset y (dependent) and asset x (independent).

    Parameters
    ----------
    delta      : Process noise variance.  Higher → ratio adapts faster.
    obs_noise  : Initial observation noise variance (R). Updated online.
    """
    delta: float = 1e-4          # process noise (Q)
    obs_noise: float = 1e-3      # observation noise (R)

    # Internal state — initialised on first observation
    beta: float = field(default=0.0, init=False)
    P: float = field(default=1.0, init=False)    # state covariance

    def update(self, x: float, y: float) -> tuple[float, float]:
        """
        Feed one (x, y) observation; return (spread, innovation variance).

        Returns
        -------
        spread : float  — y - beta * x  (the observable)
        S      : float  — innovation variance (for z-score normalisation)
        """
        Q = self.delta / (1 - self.delta)   # process noise (random-walk)

        # Predict
        P_pred = self.P + Q

        # Innovation
        yhat  = self.beta * x
        innov = y - yhat
        S     = x * P_pred * x + self.obs_noise   # innovation covariance

        # Kalman gain
        K = P_pred * x / S

        # Update
        self.beta += K * innov
        self.P     = (1 - K * x) * P_pred

        spread = y - self.beta * x
        return spread, S


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------

def generate_signals(
    prices_x: pd.Series,
    prices_y: pd.Series,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    delta: float = 1e-4,
    lookback_vol: int = 20,
) -> pd.DataFrame:
    """
    Run Kalman filter over price series and generate L/S signals.

    Parameters
    ----------
    prices_x : pd.Series  — prices of the 'independent' asset (e.g. SPY)
    prices_y : pd.Series  — prices of the 'dependent' asset   (e.g. IVV)
    entry_z  : float      — enter trade when |z| > entry_z
    exit_z   : float      — exit trade  when |z| < exit_z
    delta    : float      — Kalman process noise
    lookback_vol : int    — rolling window for spread std (z-score denominator)

    Returns
    -------
    DataFrame with columns: beta, spread, z_score, signal
        signal: +1 = long y / short x  |  -1 = short y / long x  |  0 = flat
    """
    kf = KalmanPairsFilter(delta=delta)
    records = []

    for x, y in zip(prices_x.values, prices_y.values):
        spread, S = kf.update(x, y)
        records.append({"beta": kf.beta, "spread": spread, "innov_var": S})

    df = pd.DataFrame(records, index=prices_x.index)

    # Rolling z-score of the spread
    roll_mean = df["spread"].rolling(lookback_vol).mean()
    roll_std  = df["spread"].rolling(lookback_vol).std().replace(0, np.nan)
    df["z_score"] = (df["spread"] - roll_mean) / roll_std

    # Entry / exit logic
    df["signal"] = 0
    position = 0
    signals  = []

    for z in df["z_score"].values:
        if np.isnan(z):
            signals.append(0)
            continue
        if position == 0:
            if z > entry_z:
                position = -1   # spread too high → short y, long x
            elif z < -entry_z:
                position = +1   # spread too low  → long y, short x
        elif position == 1 and z > -exit_z:
            position = 0        # close long
        elif position == -1 and z < exit_z:
            position = 0        # close short
        signals.append(position)

    df["signal"] = signals
    return df


# ---------------------------------------------------------------------------
# Backtester
# ---------------------------------------------------------------------------

def backtest(
    prices_x: pd.Series,
    prices_y: pd.Series,
    signals: pd.DataFrame,
    cost_bps: float = 1.0,
) -> pd.DataFrame:
    """
    Vectorised P&L calculation.  Position is held at close; PnL realised
    on next bar's return.

    FIX (2026 audit): the original ignored the Kalman-estimated hedge ratio
    and traded the two legs 1-for-1 — i.e. it was not trading the strategy
    it claimed to. The position is now hedge-ratio weighted (dollar-neutral
    per unit gross) and beta is lagged one bar so no look-ahead. A simple
    one-way transaction cost is charged on position changes.

    Parameters
    ----------
    signals : DataFrame from generate_signals (needs 'signal' and 'beta').

    Returns
    -------
    DataFrame: cum_pnl, drawdown, sharpe (annualised)
    """
    ret_x = prices_x.pct_change()
    ret_y = prices_y.pct_change()

    sig  = signals["signal"].shift(1).fillna(0.0)
    # Dollar hedge ratio: short beta*px dollars of x per py dollars of y.
    w = (signals["beta"] * prices_x / prices_y).shift(1)
    w = w.clip(lower=0.0).fillna(0.0)

    # Return per unit of gross exposure (long 1 of y, short w of x).
    gross = 1.0 + w
    strategy_ret = sig * (ret_y - w * ret_x) / gross

    # Transaction costs on position changes (both legs).
    tc = cost_bps / 1e4
    strategy_ret = strategy_ret - sig.diff().abs().fillna(0.0) * tc * gross
    strategy_ret = strategy_ret.fillna(0)

    cum_pnl  = (1 + strategy_ret).cumprod()
    drawdown = cum_pnl / cum_pnl.cummax() - 1

    sharpe = (
        strategy_ret.mean() / strategy_ret.std() * np.sqrt(252)
        if strategy_ret.std() > 0 else 0.0
    )

    result = pd.DataFrame({
        "strategy_ret": strategy_ret,
        "cum_pnl":      cum_pnl,
        "drawdown":     drawdown,
    })
    result.attrs["sharpe"] = sharpe
    return result


# ---------------------------------------------------------------------------
# Demo / sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Synthetic correlated prices (replace with real data via yfinance / CCXT)
    np.random.seed(42)
    n = 500
    common = np.cumsum(np.random.randn(n) * 0.01)
    px = pd.Series(100 * np.exp(common + np.cumsum(np.random.randn(n) * 0.005)),
                   name="asset_x")
    py = pd.Series(50  * np.exp(common + np.cumsum(np.random.randn(n) * 0.005)),
                   name="asset_y")

    df_signals = generate_signals(px, py, entry_z=1.5, exit_z=0.3)
    df_bt      = backtest(px, py, df_signals)

    print("=== Kalman Pairs Trading — Backtest Summary ===")
    print(f"Total trades        : {(df_signals['signal'].diff().abs() > 0).sum()}")
    print(f"Final cumulative PnL: {df_bt['cum_pnl'].iloc[-1]:.4f}")
    print(f"Max Drawdown        : {df_bt['drawdown'].min():.2%}")
    print(f"Annualised Sharpe   : {df_bt.attrs['sharpe']:.2f}")
    print(df_signals.tail(10).to_string())
