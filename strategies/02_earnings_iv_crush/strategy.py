"""
Strategy 2 — Earnings Implied-Volatility Crush (Variance Risk Premium)
=====================================================================
Sell an at-the-money straddle into an earnings announcement and buy it back the
morning after. The edge is NOT "IV falls" per se — it is that the options market
prices in a larger move than tends to be realised (a variance risk premium),
so the premium collected exceeds the cost of the stock actually moving.

Leakage discipline
------------------
The original implementation decided whether to trade using the *realised*
post-earnings IV — information that does not exist at entry. That is look-ahead
bias, and it was the entire source of the apparent edge.

Here, entry uses ONLY pre-earnings information:
  * `iv_pre` / `implied_move_pct`  (known at T−1), and
  * an estimate of each name's premium built on PAST earnings events only,
    supplied by the walk-forward loop in run.py.

The realised move and post-earnings IV are used only to compute the P&L of a
trade we already committed to — never to decide whether to place it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Black-Scholes straddle (for premium collected at entry)
# ---------------------------------------------------------------------------
def bs_price(S, K, T, r, sigma, flag="call"):
    if T <= 0 or sigma <= 0:
        return max(0.0, S - K) if flag == "call" else max(0.0, K - S)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if flag == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def atm_straddle_price(S, T, r, sigma):
    return bs_price(S, S, T, r, sigma, "call") + bs_price(S, S, T, r, sigma, "put")


# ---------------------------------------------------------------------------
# Per-event economics of a SHORT ATM straddle held across the announcement
# ---------------------------------------------------------------------------
def short_straddle_event_pnl(spot: float,
                             iv_pre: float,
                             realized_move_pct: float,
                             iv_post: float,
                             r: float = 0.04,
                             entry_dte: float = 5.0,
                             exit_dte: float = 4.0,
                             cost_bps: float = 15.0) -> dict:
    """
    Faithful P&L of the classic short-straddle-into-earnings trade, expressed
    as a fraction of the underlying notional (`spot`) so names are comparable.

    Timeline:
      T−1 (entry): sell an ATM straddle with `entry_dte` trading days to expiry,
                   at the elevated pre-earnings IV.
      T+1 (exit) : one day later the announcement has crushed IV to `iv_post`
                   and the stock has moved by `realized_move_pct`. Buy the
                   straddle back — now with `exit_dte` days left, struck at the
                   original strike, priced on the MOVED spot at the crushed IV.

    This correctly captures the three P&L drivers: the IV crush (gain), one day
    of theta (gain), and the intrinsic cost of the realised move (loss). The
    edge exists only if premium collected > move cost + residual extrinsic +
    transaction cost.

    Returns dict: premium, buyback, gross, cost, net (all fractions of spot).
    """
    T_entry = entry_dte / TRADING_DAYS
    T_exit = max(exit_dte, 0.1) / TRADING_DAYS
    K = spot

    premium = atm_straddle_price(spot, T_entry, r, iv_pre)

    # Stock has moved by the realised amount; ATM straddle payoff is symmetric
    # in the sign of the move, so an up-move is representative.
    spot_exit = spot * (1.0 + realized_move_pct)
    buyback = (bs_price(spot_exit, K, T_exit, r, max(iv_post, 1e-3), "call") +
               bs_price(spot_exit, K, T_exit, r, max(iv_post, 1e-3), "put"))

    gross = premium - buyback
    cost = cost_bps / 1e4 * spot          # round-trip option spread, in notional
    net = gross - cost
    return {
        "premium": premium / spot,
        "buyback": buyback / spot,
        "gross": gross / spot,
        "cost": cost / spot,
        "net": net / spot,
    }


# ---------------------------------------------------------------------------
# Name-level premium estimator (fit on TRAIN events only)
# ---------------------------------------------------------------------------
def estimate_name_premium(train_events: pd.DataFrame) -> pd.Series:
    """
    For each name, estimate the expected variance risk premium as the mean gap
    between the options-implied move and the realised move over the training
    events. Positive => the market historically over-priced the move => a
    candidate to SHORT vol.

    Uses only columns available after past events have fully played out. It is
    fit on the training slice and applied to future (OOS) events by the caller.
    """
    g = train_events.copy()
    g["prem"] = g["implied_move_pct"] - g["realized_move_pct"]
    return g.groupby("name")["prem"].mean()


# ---------------------------------------------------------------------------
# Trade one OOS event set given a fitted premium map + selection rule
# ---------------------------------------------------------------------------
def trade_events(oos_events: pd.DataFrame,
                 name_premium: pd.Series,
                 min_premium: float = 0.002,
                 min_iv_pre: float = 0.30,
                 cost_bps: float = 15.0) -> pd.DataFrame:
    """
    For each OOS event, decide (using only pre-earnings info + the pre-fit
    premium map) whether to short the straddle, then compute realised P&L.

    Selection: trade only names whose TRAIN-estimated premium exceeds
    `min_premium` and whose pre-earnings IV clears `min_iv_pre` (enough juice to
    be worth the tail risk / costs).

    Returns a per-trade DataFrame with net event returns (fraction of notional).
    """
    rows = []
    for _, ev in oos_events.iterrows():
        prem_est = name_premium.get(ev["name"], np.nan)
        # --- decision uses ONLY pre-earnings info ---
        if not np.isfinite(prem_est) or prem_est < min_premium:
            continue
        if ev["iv_pre"] < min_iv_pre:
            continue

        # --- realised P&L of the committed trade ---
        pnl = short_straddle_event_pnl(
            spot=ev["spot"], iv_pre=ev["iv_pre"],
            realized_move_pct=ev["realized_move_pct"], iv_post=ev["iv_post"],
            cost_bps=cost_bps)
        rows.append({
            "earnings_date": ev["earnings_date"],
            "name": ev["name"],
            "premium_est": prem_est,
            "iv_pre": ev["iv_pre"],
            "net_ret": pnl["net"],
            "gross_ret": pnl["gross"],
        })
    return pd.DataFrame(rows)


def events_to_periodic_returns(trades: pd.DataFrame) -> pd.Series:
    """
    Collapse per-event trades into a portfolio return series: on each earnings
    date, capital is spread equally across that date's traded names. This gives
    a time-indexed return stream for standard performance metrics.
    """
    if trades.empty:
        return pd.Series(dtype=float)
    daily = trades.groupby("earnings_date")["net_ret"].mean()
    daily.index = pd.to_datetime(daily.index)
    return daily.sort_index()
