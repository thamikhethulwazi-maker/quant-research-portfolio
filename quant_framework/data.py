"""
quant_framework.data
====================
The single source of market data for the whole portfolio.

WHY SYNTHETIC DATA?
-------------------
This research environment has no outbound access to market-data providers
(Yahoo, exchanges, vendor APIs are unreachable). Rather than fabricate a track
record on data we cannot obtain, we take the intellectually honest route:

  * We generate *reproducible synthetic data* whose statistical properties
    (cointegration, fat tails, volatility clustering, factor structure, an
    implied-vs-realised vol wedge) are chosen to match the stylised facts each
    strategy is designed to exploit.
  * Because the data-generating process (DGP) is known, synthetic data doubles
    as a *unit test of the strategy logic*: a correctly implemented mean-
    reversion strategy MUST profit on a series with real mean reversion, and a
    strategy that only profits because of a look-ahead bug will be exposed when
    that bug is removed.

WHAT THIS DOES AND DOES NOT PROVE
---------------------------------
Synthetic results validate *implementation correctness and the mechanics of the
edge*. They do NOT constitute evidence that the edge survives in live markets
against real transaction costs, capacity limits and competition. Every README
states this explicitly.

SWAPPING IN REAL DATA
---------------------
Each generator returns plain pandas objects with a DatetimeIndex. To go live,
implement the `MarketDataProvider` protocol below against yfinance / CCXT /
your vendor and hand the resulting frames to the exact same strategy code — no
strategy logic changes required.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd

TRADING_DAYS = 252


class MarketDataProvider(Protocol):
    """Interface a real-data adapter must satisfy to drop into the strategies."""
    def prices(self, tickers: list[str], start: str, end: str) -> pd.DataFrame: ...


# ---------------------------------------------------------------------------
# 1. Cointegrated pair (for Kalman pairs trading)
# ---------------------------------------------------------------------------
def cointegrated_pair(n: int = 1500,
                      beta: float = 1.8,
                      alpha: float = 5.0,
                      spread_kappa: float = 0.06,
                      spread_sigma: float = 0.6,
                      drift_sigma: float = 0.01,
                      start: str = "2015-01-01",
                      seed: int = 42) -> pd.DataFrame:
    """
    Two price series y, x that share a common stochastic trend AND whose spread
    (y - alpha - beta*x) is a stationary Ornstein-Uhlenbeck process. beta drifts
    slowly through time so a *dynamic* hedge ratio (Kalman) is genuinely needed.

    Returns a DataFrame with columns ['x', 'y'] indexed by business days.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)

    # Common trend drives both legs (source of cointegration).
    common = np.cumsum(rng.normal(0, 0.012, n))
    x_log = 4.0 + common + np.cumsum(rng.normal(0, drift_sigma, n))

    # Slowly time-varying hedge ratio.
    beta_t = beta + 0.4 * np.sin(np.linspace(0, 3 * np.pi, n))

    # Stationary OU spread.
    spread = np.zeros(n)
    for t in range(1, n):
        spread[t] = spread[t - 1] * (1 - spread_kappa) + rng.normal(0, spread_sigma)

    x = np.exp(x_log)
    y = np.exp(alpha / 50 + beta_t * x_log + spread * 0.03)  # keep on sane scale
    return pd.DataFrame({"x": x, "y": y}, index=idx)


# ---------------------------------------------------------------------------
# 2. Earnings implied-vol crush panel (for strategy 2)
# ---------------------------------------------------------------------------
def earnings_iv_panel(n_names: int = 40,
                      quarters: int = 12,
                      start: str = "2018-01-01",
                      base_crush_mean: float = 0.38,
                      base_crush_sd: float = 0.10,
                      seed: int = 7) -> pd.DataFrame:
    """
    Panel of earnings events with a realistic implied-vs-realised move wedge.

    Stylised facts encoded:
      * Pre-earnings ATM IV embeds an over-estimate of the actual move: the
        options market's implied 1-day move is on average LARGER than the
        realised move (the source of the short-vol premium).
      * IV collapses post-announcement by a crush factor that is *persistent*
        per name (some names systematically crush more) but noisy per event.
      * Occasional large gap surprises create fat-tailed losers for the short-
        vol seller (the tail risk that eats naive sellers alive).

    Crucially, columns that are only knowable AFTER the event
    (`realized_move_pct`, `iv_post`) are clearly named so the strategy code
    cannot accidentally use them at entry time.
    """
    rng = np.random.default_rng(seed)
    names = [f"N{i:02d}" for i in range(n_names)]

    # Persistent per-name characteristics.
    name_crush = np.clip(rng.normal(base_crush_mean, 0.06, n_names), 0.15, 0.6)
    name_iv_level = rng.uniform(0.35, 0.85, n_names)
    name_premium = rng.uniform(0.85, 1.35, n_names)  # implied/realised move ratio

    rows = []
    for q in range(quarters):
        edate = pd.Timestamp(start) + pd.Timedelta(days=q * 91)
        for i, nm in enumerate(names):
            spot = float(rng.uniform(40, 400))
            iv_pre = float(np.clip(name_iv_level[i] * rng.uniform(0.85, 1.20), 0.2, 1.5))

            # Options-implied 1-day move (approx ATM straddle breakeven).
            T_days = 1.0
            implied_move = iv_pre * np.sqrt(T_days / TRADING_DAYS)

            # Realised move: on average smaller than implied (the premium the
            # short-vol seller collects), BUT with a fat right tail — roughly
            # 1 in 12 events is a "surprise gap" of 3-8x the typical move. This
            # is the tail risk that compensates the premium and that eats naive
            # sellers alive in live markets.
            base = implied_move / name_premium[i]
            if rng.random() < 0.08:                     # surprise-gap event
                realized_move = base * rng.uniform(3.0, 8.0)
            else:
                shock = rng.standard_t(df=4) * 0.35
                realized_move = abs(rng.normal(base, base * 0.45) + base * shock)

            # Post-earnings IV crush.
            crush = float(np.clip(rng.normal(name_crush[i], base_crush_sd), 0.05, 0.7))
            iv_post = iv_pre * (1 - crush)

            rows.append({
                "name": nm,
                "earnings_date": edate,
                "spot": spot,
                # --- known at entry (T-1) ---
                "iv_pre": iv_pre,
                "implied_move_pct": implied_move,
                # --- known only at/after exit (T+1) — DO NOT use for entry ---
                "iv_post": iv_post,
                "realized_crush": crush,
                "realized_move_pct": realized_move,
            })
    return pd.DataFrame(rows).sort_values("earnings_date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Cross-sectional universe with factor structure (strategies 4 & 5)
# ---------------------------------------------------------------------------
def factor_universe(n_assets: int = 40, n_days: int = 1500, n_factors: int = 3,
                    mean_reversion: float = 0.0, start: str = "2015-01-01",
                    seed: int = 11) -> pd.DataFrame:
    """
    Daily returns for a universe driven by `n_factors` common factors plus
    idiosyncratic noise. If `mean_reversion` > 0, an AR(-1) term is injected
    into the idiosyncratic component so residual mean-reversion strategies have
    something real to capture. Returned as a (T x N) returns DataFrame.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n_days)
    factors = rng.normal(0, 0.01, (n_days, n_factors))
    loadings = rng.normal(0, 0.8, (n_assets, n_factors))
    idio = rng.normal(0, 0.008, (n_days, n_assets))
    if mean_reversion > 0:
        # Multi-day idiosyncratic reversion: each asset's idio component reverts
        # its trailing k-day move (half-life of several days), which is the
        # horizon a residual z-score strategy can actually trade net of costs.
        k = 5
        for t in range(k, n_days):
            recent = idio[t - k:t].sum(axis=0) / k
            idio[t] -= mean_reversion * recent
    ret = factors @ loadings.T + idio
    cols = [f"A{i:02d}" for i in range(n_assets)]
    return pd.DataFrame(ret, index=idx, columns=cols)


# ---------------------------------------------------------------------------
# 4. Regime-switching single asset (strategy 6)
# ---------------------------------------------------------------------------
def regime_switching_series(n: int = 2000, start: str = "2013-01-01",
                            seed: int = 13, return_regimes: bool = False):
    """Price series that cycles bull/bear/crash regimes (for HMM detection).
    If return_regimes=True, returns a DataFrame with close + true_regime so the
    detector's accuracy can be measured against ground truth."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    price = [100.0]
    regime, dwell = 0, 0
    regimes = [0]
    params = {0: (0.0006, 0.008), 1: (-0.0002, 0.019), 2: (-0.0015, 0.035)}
    trans = np.array([[0.985, 0.013, 0.002],
                      [0.020, 0.972, 0.008],
                      [0.030, 0.050, 0.920]])
    for _ in range(1, n):
        regime = rng.choice(3, p=trans[regime])
        mu, sig = params[regime]
        price.append(price[-1] * np.exp(rng.normal(mu, sig)))
        regimes.append(regime)
    close = pd.Series(price, index=idx, name="close")
    if return_regimes:
        return pd.DataFrame({"close": close.values, "true_regime": regimes}, index=idx)
    return close


# ---------------------------------------------------------------------------
# 5. Intraday bars with predictive order-flow toxicity (strategy 3, VPIN)
# ---------------------------------------------------------------------------
def toxic_flow_bars(n: int = 6000,
                    start: str = "2022-01-03",
                    freq: str = "5min",
                    base_vol: float = 0.0006,
                    seed: int = 23) -> pd.DataFrame:
    """
    Simulate intraday OHLCV-style bars in which order-flow *toxicity* is a
    latent state that (a) skews buy/sell volume imbalance and (b) RAISES
    forward return volatility and adverse drift. This is exactly the structure
    VPIN is designed to detect: imbalance today foreshadows stress tomorrow.

    Without this predictive link there would be nothing for the overlay to add,
    so the generator makes the link explicit and tunable — and therefore makes
    the overlay's value (or lack of it) an honest, testable question.

    Returns a DataFrame with columns: close, volume, buy_volume, sell_volume.
    Note: buy/sell split is provided so the strategy can *also* be tested with
    true signed volume; the VPIN estimator itself only consumes close+volume
    via bulk-volume classification, so it never uses this ground truth.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq=freq)

    # Latent toxicity in [0,1] as a slow-moving AR(1) with occasional bursts.
    tox = np.zeros(n)
    for t in range(1, n):
        shock = rng.normal(0, 0.05)
        if rng.random() < 0.010:                # toxicity burst
            shock += rng.uniform(0.3, 0.6)
        tox[t] = np.clip(0.94 * tox[t - 1] + shock, 0, 1)

    # Volatility rises with toxicity. The k-bar autocorrelation flips with the
    # regime: benign flow REVERTS its recent k-bar move (a k-bar fader earns),
    # toxic flow CONTINUES it (the fader gets run over). Matching the reversion
    # horizon to the base strategy's lookback keeps the demonstration clean.
    k = 6
    vol_t = base_vol * (1.0 + 4.0 * tox)
    ret = np.zeros(n)
    for t in range(n):
        noise = rng.normal(0, vol_t[t])
        if t >= k:
            recent = ret[t - k:t].sum() / k
            coef = -0.55 * (1.0 - tox[t]) + 0.65 * tox[t]  # revert benign/trend toxic
            ret[t] = coef * recent + noise
        else:
            ret[t] = noise
    close = 100.0 * np.exp(np.cumsum(ret))

    # Volume rises with toxicity; imbalance skews with the current move sign
    # under informed pressure (stronger when toxic).
    base_volume = rng.lognormal(mean=11.5, sigma=0.4, size=n)
    volume = base_volume * (1.0 + 3.0 * tox)
    move_sign = np.sign(ret)
    informed_skew = np.clip(0.5 + 0.35 * tox * move_sign + rng.normal(0, 0.05, n),
                            0.05, 0.95)
    buy_volume = volume * informed_skew
    sell_volume = volume * (1.0 - informed_skew)

    return pd.DataFrame({
        "close": close,
        "volume": volume,
        "buy_volume": buy_volume,
        "sell_volume": sell_volume,
        "_toxicity": tox,          # latent ground truth (diagnostics only)
    }, index=idx)


# ---------------------------------------------------------------------------
# 6. Cross-sectional reversion panel (strategy 5)
# ---------------------------------------------------------------------------
def cross_sectional_reversion_panel(n_assets: int = 40, n_days: int = 1500,
                                    n_factors: int = 2, reversion: float = 0.05,
                                    start: str = "2015-01-01",
                                    seed: int = 31) -> pd.DataFrame:
    """
    Daily returns with a common-factor component PLUS an explicit *cross-
    sectional* reversion effect: an asset that is a relative winner today tends
    to underperform its peers tomorrow. Built so a cross-sectional mean-
    reversion strategy has a genuine (but noisy) signal to capture.

    reversion : strength of next-day pull against today's cross-sectional rank.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n_days)
    cols = [f"A{i:02d}" for i in range(n_assets)]

    factors = rng.normal(0, 0.009, (n_days, n_factors))
    loadings = rng.normal(0, 0.7, (n_assets, n_factors))
    idio = rng.normal(0, 0.010, (n_days, n_assets))
    ret = factors @ loadings.T + idio

    # Inject next-day cross-sectional reversion against demeaned rank.
    for t in range(1, n_days):
        prev = ret[t - 1]
        rank = prev.argsort().argsort() / (n_assets - 1) - 0.5   # in [-0.5, 0.5]
        ret[t] -= reversion * rank * 0.02
    return pd.DataFrame(ret, index=idx, columns=cols)


# ---------------------------------------------------------------------------
# 7. Dispersion volatility events: index + constituents (strategy 7)
# ---------------------------------------------------------------------------
def dispersion_vol_panel(n_events: int = 240, n_stocks: int = 12,
                         start: str = "2016-01-01", seed: int = 41) -> pd.DataFrame:
    """
    Sequence of dispersion-trade events. Implied correlation is elevated (the
    premium being sold); realised correlation is USUALLY lower (the trade wins)
    but OCCASIONALLY SPIKES ABOVE implied (a correlation crisis — the trade
    loses on both legs). This asymmetric, fat-left-tail structure is the whole
    risk profile of dispersion trading, and the original demo omitted it.

    Columns: date, index_iv, constituent_ivs (list), constituent_weights (list),
    realised_index_vol, realised_constituent_vols (list), implied_corr,
    realised_corr.
    """
    rng = np.random.default_rng(seed)

    def _index_vol(ivs, w, rho):
        w = w / w.sum()
        n = len(ivs)
        corr = np.full((n, n), rho)
        np.fill_diagonal(corr, 1.0)
        cov = np.outer(ivs, ivs)
        return float(np.sqrt(max(w @ (corr * cov) @ w, 0.0)))

    events = []
    for i in range(n_events):
        w = np.abs(rng.normal(size=n_stocks)); w /= w.sum()
        ivs = rng.uniform(0.18, 0.55, n_stocks)
        implied_corr = rng.uniform(0.45, 0.80)                 # elevated premium
        index_iv = _index_vol(ivs, w, implied_corr)

        # Realised correlation: usually below implied (premium earned) with a
        # ~12% chance of a spike ABOVE implied (crisis -> loss).
        if rng.random() < 0.12:
            realised_corr = min(0.98, implied_corr + rng.uniform(0.05, 0.25))
        else:
            realised_corr = max(0.02, implied_corr - rng.uniform(0.02, 0.22))

        r_ivs = ivs * rng.uniform(0.85, 1.15, n_stocks)        # ~neutral vol move
        r_index = _index_vol(r_ivs, w, realised_corr)

        events.append({
            "date": pd.Timestamp(start) + pd.Timedelta(days=i * 7),
            "index_iv": index_iv,
            "constituent_ivs": ivs.tolist(),
            "constituent_weights": w.tolist(),
            "realised_index_vol": r_index,
            "realised_constituent_vols": r_ivs.tolist(),
            "implied_corr": implied_corr,
            "realised_corr": realised_corr,
        })
    return pd.DataFrame(events)
