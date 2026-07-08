"""
=============================================================================
Strategy 3: Order Flow Toxicity Model — VPIN (Liquidity Stress)
=============================================================================
Core Idea:
    Volume-Synchronized Probability of Informed Trading (VPIN) measures
    order-flow imbalance using volume buckets rather than calendar time.
    When VPIN is elevated (>0.7+), adverse selection risk is high — market
    makers widen spreads and liquidity evaporates. The strategy uses VPIN
    as a real-time liquidity stress indicator:
      • High VPIN → reduce position sizes, widen execution algos, go flat
      • Low VPIN  → normal/aggressive liquidity provision or trend following

Key References:
    - Easley, D., Lopez de Prado, M., O'Hara, M. (2012). "Flow Toxicity and
      Liquidity in a High-frequency World." Review of Financial Studies 25(5).
    - Easley, D., Lopez de Prado, M., O'Hara, M. (2011). "The Exchange of
      Flow Toxicity." Journal of Trading, 6(2), 8–13.
    - Easley, D., Lopez de Prado, M., O'Hara, M. (2011). "VPIN and the Flash
      Crash." Journal of Portfolio Management.

Works On: Any instrument with tick/bar data — equities, futures, crypto.
          Crypto is ideal (24/7 + high HFT noise).
=============================================================================
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Optional


# ---------------------------------------------------------------------------
# Bulk Volume Classification (BVC)
# ---------------------------------------------------------------------------
# Rather than classifying each trade individually (Lee-Ready, tick rule),
# BVC classifies a fraction of each bar's total volume as buy vs. sell
# based on price change vs. normally distributed returns.

def bulk_volume_classify(
    close: np.ndarray,
    volume: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Bulk Volume Classification per Easley, Lopez de Prado & O'Hara (2012).

    For each bar, the fraction of volume classified as buy (V_B) is:
        Z = Φ( ΔP / σ_ΔP )
        V_B = Z * V
        V_S = (1 - Z) * V

    Parameters
    ----------
    close   : np.ndarray — closing prices per bar
    volume  : np.ndarray — total volume per bar

    Returns
    -------
    vb : np.ndarray — buy volume per bar
    vs : np.ndarray — sell volume per bar
    """
    delta_p  = np.diff(close, prepend=close[0])
    sigma_dp = pd.Series(delta_p).rolling(50, min_periods=1).std().fillna(1e-8).values
    sigma_dp = np.where(sigma_dp == 0, 1e-8, sigma_dp)

    z   = norm.cdf(delta_p / sigma_dp)   # buy fraction ∈ (0, 1)
    vb  = z * volume
    vs  = (1 - z) * volume
    return vb, vs


# ---------------------------------------------------------------------------
# VPIN Estimator
# ---------------------------------------------------------------------------

class VPINEstimator:
    """
    Computes VPIN using a sliding window of N_buckets volume buckets.

    Each bucket contains exactly V* units of volume. Once a bucket is
    filled, |V_B - V_S| / V* is recorded as the imbalance for that bucket.
    VPIN is the rolling average of the last n_buckets imbalances.

    Parameters
    ----------
    bucket_size  : float — target volume per bucket V*
                           Rule of thumb: total_daily_volume / 50
    n_buckets    : int   — number of buckets in the VPIN rolling window
    """

    def __init__(self, bucket_size: float, n_buckets: int = 50):
        self.V_star   = bucket_size
        self.n_buckets = n_buckets
        self._reset()

    def _reset(self):
        self._vb_acc = 0.0    # accumulated buy  volume in current bucket
        self._vs_acc = 0.0    # accumulated sell volume in current bucket
        self._imbalances: list[float] = []

    def feed(self, vb_bar: float, vs_bar: float) -> Optional[float]:
        """
        Feed one bar of (buy volume, sell volume). Returns VPIN if a new
        bucket completes, else returns None.
        """
        remaining_vb = vb_bar
        remaining_vs = vs_bar
        vpin = None

        while (remaining_vb + remaining_vs) > 0:
            needed = self.V_star - (self._vb_acc + self._vs_acc)
            available = remaining_vb + remaining_vs

            if available < needed:
                # Not enough to fill the bucket
                self._vb_acc += remaining_vb
                self._vs_acc += remaining_vs
                remaining_vb  = 0
                remaining_vs  = 0
            else:
                # Fill the bucket
                frac = needed / available
                self._vb_acc += frac * remaining_vb
                self._vs_acc += frac * remaining_vs
                remaining_vb  = (1 - frac) * remaining_vb
                remaining_vs  = (1 - frac) * remaining_vs

                imbalance = abs(self._vb_acc - self._vs_acc) / self.V_star
                self._imbalances.append(imbalance)
                if len(self._imbalances) > self.n_buckets:
                    self._imbalances.pop(0)

                vpin = np.mean(self._imbalances) if self._imbalances else None
                self._vb_acc = 0.0
                self._vs_acc = 0.0

        return vpin


# ---------------------------------------------------------------------------
# VPIN signals + position sizing overlay
# ---------------------------------------------------------------------------

def compute_vpin_series(
    bars: pd.DataFrame,
    bucket_size: Optional[float] = None,
    n_buckets: int = 50,
) -> pd.DataFrame:
    """
    Compute VPIN over a bar DataFrame and produce position-size multipliers.

    Parameters
    ----------
    bars : pd.DataFrame with columns: close, volume (datetime index)
    bucket_size : float or None. If None, estimated as mean_daily_vol / 50.
    n_buckets   : int

    Returns
    -------
    DataFrame with: vb, vs, vpin, liquidity_regime, size_multiplier
    """
    close  = bars["close"].values
    volume = bars["volume"].values

    if bucket_size is None:
        # FIX (2026 audit): estimate the bucket size from the FIRST 30% of the
        # sample only. Using the full-sample mean volume leaks future
        # information into a parameter used from bar one.
        n_train = max(1, int(len(volume) * 0.30))
        bucket_size = float(np.mean(volume[:n_train])) * 0.5

    vb, vs = bulk_volume_classify(close, volume)

    estimator = VPINEstimator(bucket_size=bucket_size, n_buckets=n_buckets)
    vpin_vals = []
    for b, s in zip(vb, vs):
        v = estimator.feed(b, s)
        vpin_vals.append(v)

    bars = bars.copy()
    bars["vb"]   = vb
    bars["vs"]   = vs
    bars["vpin"] = vpin_vals

    # Forward-fill VPIN between bucket completions
    bars["vpin"] = bars["vpin"].ffill()

    # Liquidity regime labels
    def _regime(v):
        if pd.isna(v):   return "unknown"
        if v < 0.35:     return "benign"      # normal flow
        if v < 0.55:     return "elevated"    # caution
        if v < 0.70:     return "stressed"    # reduce sizes
        return "toxic"                         # stand aside / close

    bars["liquidity_regime"] = bars["vpin"].map(_regime)

    # Position size multiplier: scale DOWN as VPIN rises
    bars["size_multiplier"] = bars["vpin"].apply(
        lambda v: max(0.0, 1.0 - (v - 0.35) / 0.35) if pd.notna(v) else 1.0
    ).clip(0, 1)
    # FIX (2026 audit): anything acting on this multiplier must LAG it one
    # bar — you can only size on the toxicity you have already observed.
    bars["size_multiplier"] = bars["size_multiplier"].shift(1).fillna(1.0)

    return bars


# ---------------------------------------------------------------------------
# Execution quality monitor
# ---------------------------------------------------------------------------

def vpin_execution_report(vpin_series: pd.Series, threshold: float = 0.60) -> dict:
    """
    Summarise VPIN-based execution quality for a trading period.

    Returns dict with: pct_toxic_time, avg_vpin, max_vpin, n_stress_episodes
    """
    valid = vpin_series.dropna()
    if valid.empty:
        return {}

    stressed = valid > threshold
    # Count transitions into stressed state
    transitions = (stressed & ~stressed.shift(1, fill_value=False)).sum()

    return {
        "avg_vpin":          round(float(valid.mean()), 4),
        "max_vpin":          round(float(valid.max()), 4),
        "pct_toxic_time":    round(float(stressed.mean()), 4),
        "n_stress_episodes": int(transitions),
    }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(99)
    n = 2000
    # Simulate OHLCV bars (crypto-like: 1-min bars)
    price = 30_000 + np.cumsum(np.random.randn(n) * 100)
    vol   = np.random.exponential(scale=5_000_000, size=n)   # volume in USD notional

    bars = pd.DataFrame({
        "close":  price,
        "volume": vol,
    }, index=pd.date_range("2024-01-01", periods=n, freq="1min"))

    result = compute_vpin_series(bars, n_buckets=50)

    print("=== VPIN Order Flow Toxicity ===")
    print(result[["close", "vpin", "liquidity_regime", "size_multiplier"]].dropna().tail(20))
    report = vpin_execution_report(result["vpin"])
    print("\nExecution Quality Report:")
    for k, v in report.items():
        print(f"  {k:25s}: {v}")
