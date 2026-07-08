"""
=============================================================================
Strategy 7: Dispersion Volatility Arbitrage (Correlation Spread)
=============================================================================
Core Idea:
    Index implied volatility is structurally overpriced relative to the
    weighted average of constituent implied volatilities. This spread exists
    because index options are used as portfolio hedges, creating excess demand
    for index puts and inflating index IV. The dispersion trade:
      • Short index volatility (sell index straddle / variance swap)
      • Long constituent volatility (buy individual straddles / variance swaps)
    Profits from the implied correlation premium collapsing toward realised
    correlation when stocks move more independently than the index price implies.

Key References:
    - Deng, Q. (2008). "Volatility Dispersion Trading." University of Illinois.
    - Jacquier, A. & Slaoui, S. (2010). "Variance Dispersion and Correlation
      Swaps." Imperial College London Working Paper.
    - Bossu, S. (2005). "Arbitrage Pricing of Equity Correlation Swaps."
      JP Morgan Equity Derivatives Research.
    - Brière, M. & Drut, B. (2009). "Dispersion Trading: Empirical Evidence
      from US Options Markets." Finance Research Letters.

Works On: Index + constituent options — S&P500, Nasdaq100, BTC/ETH ecosystem.
=============================================================================
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Optional


# ---------------------------------------------------------------------------
# Implied Correlation Estimation
# ---------------------------------------------------------------------------

def implied_index_variance(
    constituent_ivs:   np.ndarray,
    constituent_weights: np.ndarray,
    pairwise_implied_corr: float,
) -> float:
    """
    Reconstruct index variance from constituents under a single-factor
    correlation model.

    σ²_index ≈ Σᵢ Σⱼ wᵢ wⱼ ρᵢⱼ σᵢ σⱼ

    Parameters
    ----------
    constituent_ivs     : np.ndarray — implied vols of each constituent
    constituent_weights : np.ndarray — index weights (sum to 1)
    pairwise_implied_corr: float     — single correlation assumption ρ

    Returns
    -------
    float — implied index volatility (annualised)
    """
    n   = len(constituent_ivs)
    w   = constituent_weights / constituent_weights.sum()
    cov = np.outer(constituent_ivs, constituent_ivs)

    # Build correlation matrix: ρ everywhere except diagonal (=1)
    corr = np.full((n, n), pairwise_implied_corr)
    np.fill_diagonal(corr, 1.0)

    var_index = float(w @ (corr * cov) @ w)
    return np.sqrt(max(var_index, 0))


def solve_implied_correlation(
    index_iv:            float,
    constituent_ivs:     np.ndarray,
    constituent_weights: np.ndarray,
    tol: float = 1e-6,
    max_iter: int = 200,
) -> float:
    """
    Invert the index variance formula to find the implied correlation ρ*
    such that reconstructed index IV matches observed index IV.

    Uses bisection over ρ ∈ [0, 1].

    Returns
    -------
    float — implied (average pairwise) correlation
    """
    target_var = index_iv ** 2
    w = constituent_weights / constituent_weights.sum()
    n = len(constituent_ivs)

    def _residual(rho: float) -> float:
        corr = np.full((n, n), rho)
        np.fill_diagonal(corr, 1.0)
        cov  = np.outer(constituent_ivs, constituent_ivs)
        return float(w @ (corr * cov) @ w) - target_var

    lo, hi = 0.0, 1.0
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        if abs(hi - lo) < tol:
            break
        if _residual(mid) < 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ---------------------------------------------------------------------------
# Variance Swap Payoff (simplified)
# ---------------------------------------------------------------------------

def variance_swap_pnl(
    realised_vol: float,
    strike_vol:   float,
    vega_notional: float = 100_000,
) -> float:
    """
    Variance swap P&L at expiry.
    Payer of realised variance receives: (σ²_realised - K²_vol) × vega_notional / (2 × K_vol)

    Note: simplified — ignores convexity adjustment.

    Returns
    -------
    float — P&L in currency units
    """
    return (realised_vol ** 2 - strike_vol ** 2) * vega_notional / (2 * strike_vol)


# ---------------------------------------------------------------------------
# Dispersion Trade Simulator
# ---------------------------------------------------------------------------

class DispersionTradeSimulator:
    """
    Simulates a dispersion trade using straddle proxies.

    The trade is:
        - Short 1 index straddle  (short index vol)
        - Long  wᵢ constituent straddles for each i (long stock vol)

    Profit comes from the spread between implied correlation and
    subsequently realised correlation.

    Parameters
    ----------
    index_weight    : float — dollar notional allocated to index short
    vega_per_unit   : float — vega exposure per unit notional
    """

    def __init__(self, index_weight: float = 1.0, vega_per_unit: float = 1.0):
        self.index_weight = index_weight
        self.vega_per_unit = vega_per_unit

    def trade_entry(
        self,
        index_iv:             float,
        constituent_ivs:      np.ndarray,
        constituent_weights:  np.ndarray,
    ) -> dict:
        """
        Set up the dispersion trade and compute entry metrics.

        Returns dict with: implied_corr, dispersion_premium, position_summary
        """
        impl_corr = solve_implied_correlation(
            index_iv, constituent_ivs, constituent_weights
        )
        reconstructed_iv = implied_index_variance(
            constituent_ivs, constituent_weights, impl_corr
        )
        dispersion_premium = index_iv - reconstructed_iv

        return {
            "index_iv":          index_iv,
            "reconstructed_iv":  reconstructed_iv,
            "implied_corr":      impl_corr,
            "dispersion_premium": dispersion_premium,
            "constituent_ivs":   constituent_ivs,
            "constituent_weights": constituent_weights,
        }

    def trade_exit(
        self,
        entry: dict,
        realised_index_vol:      float,
        realised_constituent_vols: np.ndarray,
        realised_corr: float,
    ) -> dict:
        """
        Calculate P&L at trade exit.

        Short index var-swap pays: (σ²_index_realised - IV²_index)
        Long  stock var-swaps get: Σᵢ wᵢ(σ²_i_realised - IV²_i)

        Returns dict with: index_pnl, constituent_pnl, net_pnl, corr_spread
        """
        w = entry["constituent_weights"] / entry["constituent_weights"].sum()

        # Short index leg (we sold index variance at IV, bought realised)
        index_pnl = -(realised_index_vol ** 2 - entry["index_iv"] ** 2)

        # Long constituent legs (we bought constituent variance at IV)
        constituent_pnl = float(np.sum(
            w * (realised_constituent_vols ** 2 - entry["constituent_ivs"] ** 2)
        ))

        net_pnl    = index_pnl + constituent_pnl
        corr_spread = entry["implied_corr"] - realised_corr

        return {
            "index_pnl":        index_pnl * self.index_weight,
            "constituent_pnl":  constituent_pnl * self.index_weight,
            "net_pnl":          net_pnl * self.index_weight,
            "corr_spread":      corr_spread,
            "realised_corr":    realised_corr,
            "implied_corr":     entry["implied_corr"],
        }


# ---------------------------------------------------------------------------
# Historical backtest over multiple events
# ---------------------------------------------------------------------------

def run_dispersion_backtest(
    events: pd.DataFrame,
    min_dispersion_premium: float = 0.02,
    min_implied_corr: float = 0.50,
) -> pd.DataFrame:
    """
    Backtest dispersion strategy over multiple periods/events.

    Parameters
    ----------
    events : pd.DataFrame with columns:
        date, index_iv, constituent_ivs (list), constituent_weights (list),
        realised_index_vol, realised_constituent_vols (list), realised_corr
    min_dispersion_premium : float — min IV spread to enter trade
    min_implied_corr       : float — only trade when implied corr is elevated

    Returns
    -------
    DataFrame with per-trade P&L
    """
    sim = DispersionTradeSimulator()
    results = []
    cumulative = 0.0

    for _, ev in events.iterrows():
        entry = sim.trade_entry(
            index_iv             = ev["index_iv"],
            constituent_ivs      = np.array(ev["constituent_ivs"]),
            constituent_weights  = np.array(ev["constituent_weights"]),
        )

        # FIX (2026 audit): `dispersion_premium` is ~0 BY CONSTRUCTION —
        # the bisection solver finds the corr that makes reconstructed IV
        # equal observed IV, so their difference is meaningless as a filter
        # (the original only traded when the solver saturated at rho=1).
        # The economically meaningful entry filter is an ELEVATED implied
        # correlation, which is the premium actually being sold.
        if entry["implied_corr"] < min_implied_corr:
            continue

        exit_r = sim.trade_exit(
            entry                    = entry,
            realised_index_vol       = ev["realised_index_vol"],
            realised_constituent_vols= np.array(ev["realised_constituent_vols"]),
            realised_corr            = ev["realised_corr"],
        )

        cumulative += exit_r["net_pnl"]
        results.append({
            "date":               ev["date"],
            "implied_corr":       entry["implied_corr"],
            "realised_corr":      ev["realised_corr"],
            "corr_spread":        exit_r["corr_spread"],
            "dispersion_premium": entry["dispersion_premium"],
            "net_pnl":            exit_r["net_pnl"],
            "cumulative_pnl":     cumulative,
        })

    result_df = pd.DataFrame(results)
    if not result_df.empty:
        wins = result_df["net_pnl"] > 0
        print(f"Win Rate    : {wins.mean():.1%}")
        print(f"Avg PnL     : {result_df['net_pnl'].mean():.4f}")
        print(f"Avg ρ Spread: {result_df['corr_spread'].mean():.4f}")
        print(f"Total PnL   : {cumulative:.4f}")
    return result_df


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(77)
    n_events = 60
    n_stocks = 10

    def make_weights(n):
        w = np.abs(np.random.randn(n))
        return w / w.sum()

    events = []
    for i in range(n_events):
        w   = make_weights(n_stocks)
        ivs = np.random.uniform(0.20, 0.60, n_stocks)   # constituent IVs

        # Implied correlation: elevated (this is the premium we're selling)
        rho_impl = np.random.uniform(0.50, 0.80)
        idx_iv   = implied_index_variance(ivs, w, rho_impl) * np.random.uniform(1.02, 1.10)

        # FIX (2026 audit): the original defined realised correlation as
        # implied MINUS a positive draw — the trade could never lose, which
        # hides the defining risk of dispersion. Realised correlation now
        # spikes ABOVE implied ~12% of the time (a correlation crisis), giving
        # the honest fat left tail.
        if np.random.random() < 0.12:
            rho_real = min(0.98, rho_impl + np.random.uniform(0.05, 0.25))
        else:
            rho_real = rho_impl - np.random.uniform(0.02, 0.22)
        r_ivs    = ivs * np.random.uniform(0.80, 1.10, n_stocks)
        r_idx    = implied_index_variance(r_ivs, w, max(rho_real, 0)) * 0.95

        events.append({
            "date":                    pd.Timestamp("2023-01-01") + pd.Timedelta(days=i * 5),
            "index_iv":                idx_iv,
            "constituent_ivs":         ivs.tolist(),
            "constituent_weights":     w.tolist(),
            "realised_index_vol":      r_idx,
            "realised_constituent_vols": r_ivs.tolist(),
            "realised_corr":           max(rho_real, 0),
        })

    events_df = pd.DataFrame(events)

    print("=== Dispersion Volatility Arbitrage ===")
    results = run_dispersion_backtest(events_df,
                                      min_dispersion_premium=0.01,
                                      min_implied_corr=0.45)
    if not results.empty:
        print(results[["date", "implied_corr", "realised_corr", "net_pnl",
                        "cumulative_pnl"]].tail(10).to_string())
