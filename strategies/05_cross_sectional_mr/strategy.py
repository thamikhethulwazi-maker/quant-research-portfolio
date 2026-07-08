"""
Strategy 5 — Cross-Sectional Mean Reversion
===========================================
Rank the universe each day by short-horizon relative performance. The relative
winners (top of the cross-section) tend to give back, the relative losers tend
to bounce, within a few days. Go long the losers, short the winners, dollar-
neutral, and rebalance frequently.

Sign discipline
---------------
For mean reversion you must be LONG the loser (it is expected to revert UP) with
a POSITIVE weight. The original code assigned the loser leg a negative weight
(which is momentum, not reversion). Here the convention is set correctly and
*verified* against the information coefficient in run.py — never trusted from the
label alone.

References: Lehmann (1990), Jegadeesh (1990), Avellaneda & Lee (2010).
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


def cross_sectional_score(returns: pd.DataFrame, lookback: int = 5,
                          method: Literal["z_score", "rank"] = "z_score") -> pd.DataFrame:
    """
    Score each asset relative to its peers each day (causal — uses only trailing
    returns). Positive score = relative WINNER (over-performed) = short candidate.
    Negative score = relative LOSER (under-performed) = long candidate.
    """
    cum = (1 + returns).rolling(lookback).apply(np.prod, raw=True) - 1
    if method == "rank":
        return cum.rank(axis=1, pct=True) * 2 - 1          # [-1, +1]
    # z_score: cross-sectional standardisation each day
    mu = cum.mean(axis=1)
    sd = cum.std(axis=1).replace(0, np.nan)
    return cum.sub(mu, axis=0).div(sd, axis=0)


def build_portfolio(scores: pd.DataFrame, n_long: int = 10, n_short: int = 10,
                    entry_threshold: float = 0.5) -> pd.DataFrame:
    """
    Dollar-neutral weights. LONG the biggest losers (most negative score) with
    POSITIVE weight; SHORT the biggest winners (most positive score) with
    NEGATIVE weight. Equal-weighted within each leg.
    """
    w = pd.DataFrame(0.0, index=scores.index, columns=scores.columns)
    for date in scores.index:
        s = scores.loc[date].dropna()
        if s.empty:
            continue
        losers = s[s < -entry_threshold].nsmallest(n_long)   # revert UP -> long
        winners = s[s > entry_threshold].nlargest(n_short)   # revert DOWN -> short
        if losers.empty or winners.empty:
            continue
        w.loc[date, losers.index] = 1.0 / len(losers)
        w.loc[date, winners.index] = -1.0 / len(winners)
    return w


def backtest(returns: pd.DataFrame, weights: pd.DataFrame,
             rebal_freq: int = 1, cost_bps: float = 5.0,
             return_weights: bool = False):
    """Positions lagged one day; held `rebal_freq` days; turnover-based costs."""
    mask = np.zeros(len(weights), dtype=bool)
    mask[::rebal_freq] = True
    w_held = weights[mask].reindex(weights.index, method="ffill").shift(1).fillna(0.0)
    gross = (w_held * returns).sum(axis=1)
    turnover = w_held.diff().abs().sum(axis=1).fillna(0.0)
    net = gross - turnover * (cost_bps / 1e4)
    net.name = "net_ret"
    return (net, w_held) if return_weights else net
