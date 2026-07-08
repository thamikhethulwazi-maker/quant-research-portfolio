"""
Strategy 1 — Kalman Filter Pairs Trading
========================================
Dynamically estimate the hedge ratio between two cointegrated assets with a
Kalman filter, model the spread as mean-reverting, and trade z-score extremes.

This module preserves the original research idea and the (causal) Kalman filter
but corrects the economics of the backtest:

  * The position is properly hedge-ratio weighted and dollar-neutral: long 1
    unit of y, short beta units of x, gross exposure normalised to 1.
  * Transaction costs are charged on turnover.
  * Signals and the hedge ratio are lagged one bar before being applied to
    forward returns — no look-ahead.

The parameter selection is done OUT of this file, by the walk-forward validator
in run.py, so nothing here ever peeks at test data.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Causal Kalman filter for a time-varying hedge ratio  y_t = beta_t * x_t + e_t
# ---------------------------------------------------------------------------
@dataclass
class KalmanPairsFilter:
    """
    Random-walk state model for beta. `delta` controls how fast beta adapts
    (larger -> faster). Each update() consumes only information up to time t.
    """
    delta: float = 1e-4
    obs_noise: float = 1e-3
    beta: float = field(default=0.0, init=False)
    P: float = field(default=1.0, init=False)

    def update(self, x: float, y: float) -> tuple[float, float]:
        Q = self.delta / (1.0 - self.delta)
        P_pred = self.P + Q
        innov = y - self.beta * x
        S = x * P_pred * x + self.obs_noise
        K = P_pred * x / S
        self.beta += K * innov
        self.P = (1.0 - K * x) * P_pred
        return y - self.beta * x, S     # (spread, innovation variance)


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------
def generate_signals(prices_x: pd.Series,
                     prices_y: pd.Series,
                     entry_z: float = 2.0,
                     exit_z: float = 0.5,
                     delta: float = 1e-4,
                     lookback_vol: int = 20) -> pd.DataFrame:
    """
    Run the filter and produce a stateful position series.

    signal: +1 = long y / short (beta)x ;  -1 = short y / long (beta)x ;  0 flat.
    Returns columns: beta, spread, z_score, signal.
    """
    kf = KalmanPairsFilter(delta=delta)
    beta, spread = [], []
    for x, y in zip(prices_x.values, prices_y.values):
        s, _ = kf.update(float(x), float(y))
        beta.append(kf.beta)
        spread.append(s)

    df = pd.DataFrame({"beta": beta, "spread": spread}, index=prices_x.index)
    roll_mean = df["spread"].rolling(lookback_vol).mean()
    roll_std = df["spread"].rolling(lookback_vol).std().replace(0, np.nan)
    df["z_score"] = (df["spread"] - roll_mean) / roll_std

    position, signals = 0, []
    for z in df["z_score"].values:
        if np.isnan(z):
            signals.append(0)
            continue
        if position == 0:
            if z > entry_z:
                position = -1
            elif z < -entry_z:
                position = +1
        elif position == 1 and z > -exit_z:
            position = 0
        elif position == -1 and z < exit_z:
            position = 0
        signals.append(position)
    df["signal"] = signals
    return df


# ---------------------------------------------------------------------------
# Backtest — hedge-ratio weighted, dollar-neutral, cost-aware
# ---------------------------------------------------------------------------
def backtest(prices_x: pd.Series,
             prices_y: pd.Series,
             signals: pd.DataFrame,
             cost_bps: float = 1.0) -> pd.Series:
    """
    Compute net periodic strategy returns.

    A signal of +1 means: long 1 unit of y and short beta units of x, with the
    two legs normalised so gross exposure = 1. We lag both the signal and the
    hedge ratio by one bar (positions are set at yesterday's close and earn
    today's return) to avoid look-ahead.

    Cost: `cost_bps` one-way basis points charged on gross leg turnover.
    """
    ret_x = prices_x.pct_change().fillna(0.0)
    ret_y = prices_y.pct_change().fillna(0.0)

    sig = signals["signal"].shift(1).fillna(0.0)
    beta = signals["beta"].shift(1).fillna(0.0)

    # Leg weights, gross-normalised to 1 unit of capital.
    denom = (1.0 + beta.abs()).replace(0, np.nan)
    w_y = sig * (1.0 / denom)
    w_x = -sig * (beta / denom)
    w_y, w_x = w_y.fillna(0.0), w_x.fillna(0.0)

    gross_ret = w_y * ret_y + w_x * ret_x

    # Turnover = sum of |Δleg weight| across both legs.
    turnover = w_y.diff().abs().fillna(0) + w_x.diff().abs().fillna(0)
    cost = turnover * (cost_bps / 1e4)

    net_ret = (gross_ret - cost).fillna(0.0)
    net_ret.name = "net_ret"
    return net_ret


def extract_trade_pnls(prices_x: pd.Series, prices_y: pd.Series,
                       signals: pd.DataFrame, cost_bps: float = 1.0) -> pd.Series:
    """
    Roll periodic returns up into discrete round-trip trade PnLs (for win rate,
    profit factor, holding period). A trade spans from position open to close.
    """
    net = backtest(prices_x, prices_y, signals, cost_bps)
    sig = signals["signal"].fillna(0.0)
    pnls, holding, cur, hold_len = [], [], 0.0, 0
    prev = 0
    for i, s in enumerate(sig.values):
        if prev == 0 and s != 0:                 # open
            cur, hold_len = 0.0, 0
        if s != 0:
            cur += net.iloc[i]
            hold_len += 1
        if prev != 0 and s == 0:                 # close
            pnls.append(cur)
            holding.append(hold_len)
        prev = s
    return pd.Series(pnls), pd.Series(holding)
