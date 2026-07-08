"""
=============================================================================
Strategy 2: Earnings Implied Volatility Crush (Premium Collapse)
=============================================================================
Core Idea:
    Options implied volatility inflates before earnings as market participants
    pay a premium for uncertainty. Once the announcement lands, uncertainty
    resolves and IV collapses 30–50%+ regardless of price direction.
    The strategy systematically sells options straddles/strangles in the days
    leading up to an earnings event and closes them the morning after.

Key References:
    - CBOE (2023). "S&P 500 Earnings Events Implied vs Realized Move Study."
    - Goyal, A. & Saretto, A. (2009). "Cross-section of option returns and
      volatility." Journal of Financial Economics, 94(2), 310–326.
    - Muravyev, D. & Pearson, N.D. (2020). "Option Trading Costs Are Lower
      Than You Think." Review of Financial Studies, 33(11).

Works On: US Equities with listed options (high-volume liquid names).
          Adapt for crypto perpetual options (Deribit / Bybit) or FX.

Risk Warning:
    Gap risk on earnings — always size to max acceptable loss on a binary
    event. Consider defined-risk spreads (iron condors) over naked straddles.
=============================================================================
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Black-Scholes helpers
# ---------------------------------------------------------------------------

def bs_price(S: float, K: float, T: float, r: float, sigma: float,
             flag: str = "call") -> float:
    """
    Vanilla Black-Scholes price.

    Parameters
    ----------
    S     : spot price
    K     : strike
    T     : time to expiry in years
    r     : risk-free rate (annualised)
    sigma : implied volatility (annualised)
    flag  : 'call' or 'put'
    """
    if T <= 0 or sigma <= 0:
        return max(0.0, S - K) if flag == "call" else max(0.0, K - S)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if flag == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def straddle_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """ATM straddle = call + put at same strike."""
    return bs_price(S, K, T, r, sigma, "call") + bs_price(S, K, T, r, sigma, "put")


def strangle_price(S: float, K_call: float, K_put: float, T: float,
                   r: float, sigma: float) -> float:
    """OTM strangle = OTM call + OTM put at different strikes."""
    return (bs_price(S, K_call, T, r, sigma, "call") +
            bs_price(S, K_put,  T, r, sigma, "put"))


def implied_vol_newton(price: float, S: float, K: float, T: float,
                       r: float, flag: str = "call",
                       tol: float = 1e-6, max_iter: int = 100) -> float:
    """Newton-Raphson IV solver for single option."""
    sigma = 0.3   # initial guess
    for _ in range(max_iter):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        vega = S * norm.pdf(d1) * np.sqrt(T)
        price_err = bs_price(S, K, T, r, sigma, flag) - price
        if abs(price_err) < tol:
            break
        sigma -= price_err / (vega + 1e-12)
        sigma  = max(sigma, 1e-6)
    return sigma


# ---------------------------------------------------------------------------
# IV Crush signal model
# ---------------------------------------------------------------------------

@dataclass
class EarningsIVCrushSignal:
    """
    Models the expected IV crush magnitude and trade economics for a
    single earnings event.

    Parameters
    ----------
    entry_days_before : int   — days before earnings to enter the short vol trade
    exit_session      : str   — 'open' or 'close' of next session after earnings
    min_crush_factor  : float — minimum expected IV collapse % to qualify
    max_position_pct  : float — max % of account risked per trade
    """
    entry_days_before: int   = 2
    exit_session:      str   = "open"
    min_crush_factor:  float = 0.20    # minimum 20% IV drop required
    max_position_pct:  float = 0.02    # 2% account risk per event

    def expected_crush(self, iv_pre: float, iv_post_estimate: float) -> float:
        """Estimated post-earnings IV as fraction of pre-earnings IV."""
        return 1.0 - iv_post_estimate / iv_pre

    def trade_economics(
        self,
        S: float, K: float, T_entry: float, T_exit: float,
        iv_entry: float, iv_exit: float, r: float = 0.05
    ) -> dict:
        """
        Calculate P&L for a short ATM straddle from entry to exit,
        holding stock price constant (pure vol move).

        Returns dict with: premium_sold, buyback_cost, gross_pnl, pnl_pct
        """
        premium_sold   = straddle_price(S, K, T_entry, r, iv_entry)
        buyback_cost   = straddle_price(S, K, T_exit,  r, iv_exit)
        gross_pnl      = premium_sold - buyback_cost
        pnl_pct        = gross_pnl / premium_sold if premium_sold > 0 else 0.0

        return {
            "S":             S,
            "K":             K,
            "iv_entry":      iv_entry,
            "iv_exit":       iv_exit,
            "premium_sold":  premium_sold,
            "buyback_cost":  buyback_cost,
            "gross_pnl":     gross_pnl,
            "pnl_pct":       pnl_pct,
        }

    def position_size(self, account_value: float, max_loss_per_contract: float) -> int:
        """
        Kelly-inspired position sizing: risk at most max_position_pct of account.
        Returns number of contracts (each = 100 shares).
        """
        max_dollar_risk = account_value * self.max_position_pct
        if max_loss_per_contract <= 0:
            return 0
        return max(1, int(max_dollar_risk / (max_loss_per_contract * 100)))


# ---------------------------------------------------------------------------
# Backtester over a universe of earnings events
# ---------------------------------------------------------------------------

def run_iv_crush_backtest(
    events: pd.DataFrame,
    account_value: float = 100_000,
    entry_days_before: int = 2,
    min_entry_iv: float = 0.50,
) -> pd.DataFrame:
    """
    Simulate the IV crush strategy over a DataFrame of earnings events.

    Parameters
    ----------
    events : pd.DataFrame with columns:
        ticker, earnings_date, S (spot), K (atm strike),
        T_entry (tte at entry, years), T_exit (tte at exit),
        iv_entry, iv_exit, stock_move_pct (actual post-earnings move)

    Returns
    -------
    DataFrame with per-trade results and cumulative statistics.
    """
    signal = EarningsIVCrushSignal(entry_days_before=entry_days_before)

    results = []
    running_pnl = 0.0

    for _, ev in events.iterrows():
        # FIX (2026 audit): the original gated entry on the *post-earnings*
        # IV (ev["iv_exit"]) — information that does not exist at entry time.
        # That is fatal look-ahead bias. The entry decision may only use
        # pre-event information; here we require an elevated entry IV, which
        # is observable when the trade is placed.
        if ev["iv_entry"] < min_entry_iv:
            continue
        crush = signal.expected_crush(ev["iv_entry"], ev["iv_exit"])  # ex-post stat, reporting only

        trade = signal.trade_economics(
            S        = ev["S"],
            K        = ev["K"],
            T_entry  = ev["T_entry"],
            T_exit   = ev["T_exit"],
            iv_entry = ev["iv_entry"],
            iv_exit  = ev["iv_exit"],
        )

        # FIX (2026 audit): the original charged a dimensionally wrong
        # "delta_pnl = -|move| * S" term. The faithful treatment reprices the
        # straddle at the moved spot: buy back at S*(1+move) with post IV.
        S_exit   = ev["S"] * (1.0 + ev["stock_move_pct"])
        buyback  = straddle_price(S_exit, ev["K"], ev["T_exit"], 0.05, ev["iv_exit"])
        net_pnl  = trade["premium_sold"] - buyback

        running_pnl += net_pnl
        results.append({
            "ticker":        ev["ticker"],
            "earnings_date": ev["earnings_date"],
            "iv_crush_pct":  crush,
            "gross_pnl":     trade["gross_pnl"],
            "delta_adj_pnl": net_pnl,
            "running_pnl":   running_pnl,
        })

    result_df = pd.DataFrame(results)
    if not result_df.empty:
        wins = result_df["delta_adj_pnl"] > 0
        print(f"Win Rate     : {wins.mean():.1%}")
        print(f"Avg Trade    : ${result_df['delta_adj_pnl'].mean():.2f}")
        print(f"Total PnL    : ${running_pnl:.2f}")
    return result_df


# ---------------------------------------------------------------------------
# Synthetic demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(7)
    n_events = 100

    events = pd.DataFrame({
        "ticker":        [f"TICK{i}" for i in range(n_events)],
        "earnings_date": pd.date_range("2023-01-01", periods=n_events, freq="W"),
        "S":             np.random.uniform(50, 500, n_events),
        "K":             np.random.uniform(50, 500, n_events),   # simplified ATM
        "T_entry":       np.full(n_events, 7 / 252),             # 7 trading days
        "T_exit":        np.full(n_events, 5 / 252),             # 5 trading days
        "iv_entry":      np.random.uniform(0.45, 0.90, n_events),
        "iv_exit":       np.random.uniform(0.20, 0.50, n_events),
        "stock_move_pct":np.random.uniform(-0.12, 0.12, n_events),
    })
    # Align K with S for ATM
    events["K"] = events["S"]

    print("=== Earnings IV Crush Backtest ===")
    results = run_iv_crush_backtest(events, account_value=100_000)
    print(results.tail(10).to_string())
