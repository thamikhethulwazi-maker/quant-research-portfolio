"""
quant_framework.metrics
=======================
Institutional-grade performance and risk metrics computed from a series of
*periodic strategy returns* (not prices). All functions are deterministic and
side-effect free.

Conventions
-----------
- `returns` is a pandas Series of simple (arithmetic) periodic returns indexed
  by a DatetimeIndex. A daily series is assumed unless `periods_per_year` says
  otherwise.
- Sharpe / Sortino are annualised with sqrt-time scaling of the *excess* return
  over a per-period risk-free rate.
- We never annualise a mean and a vol computed on different horizons.

A note on rigour
----------------
Sharpe ratios from a single backtest are point estimates with wide sampling
error. Use `quant_framework.robustness` to attach bootstrap confidence
intervals before drawing conclusions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Return-curve helpers
# ---------------------------------------------------------------------------
def equity_curve(returns: pd.Series, start_value: float = 1.0) -> pd.Series:
    """Compound periodic returns into an equity curve."""
    return start_value * (1.0 + returns.fillna(0.0)).cumprod()


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Drawdown at each point relative to the running peak (<= 0)."""
    eq = equity_curve(returns)
    return eq / eq.cummax() - 1.0


# ---------------------------------------------------------------------------
# Headline return / risk metrics
# ---------------------------------------------------------------------------
def cagr(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """Compound annual growth rate implied by the return stream."""
    r = returns.dropna()
    if len(r) == 0:
        return 0.0
    total_growth = (1.0 + r).prod()
    years = len(r) / periods_per_year
    if years <= 0 or total_growth <= 0:
        return 0.0
    return total_growth ** (1.0 / years) - 1.0


def annualized_return(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """Arithmetic annualised mean return."""
    return float(returns.mean() * periods_per_year)


def annualized_vol(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """Annualised standard deviation of returns."""
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(returns: pd.Series, rf_annual: float = 0.0,
                 periods_per_year: int = TRADING_DAYS) -> float:
    """Annualised Sharpe ratio using per-period excess returns."""
    r = returns.dropna()
    if len(r) < 2 or r.std(ddof=1) == 0:
        return 0.0
    rf_per = rf_annual / periods_per_year
    excess = r - rf_per
    return float(excess.mean() / excess.std(ddof=1) * np.sqrt(periods_per_year))


def sortino_ratio(returns: pd.Series, rf_annual: float = 0.0,
                  periods_per_year: int = TRADING_DAYS) -> float:
    """Annualised Sortino ratio (downside deviation in the denominator)."""
    r = returns.dropna()
    if len(r) < 2:
        return 0.0
    rf_per = rf_annual / periods_per_year
    excess = r - rf_per
    downside = excess[excess < 0]
    dd = np.sqrt((downside ** 2).mean()) if len(downside) else 0.0
    if dd == 0:
        return 0.0
    return float(excess.mean() / dd * np.sqrt(periods_per_year))


def max_drawdown(returns: pd.Series) -> float:
    """Maximum peak-to-trough drawdown (negative number)."""
    dd = drawdown_series(returns)
    return float(dd.min()) if len(dd) else 0.0


def calmar_ratio(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """CAGR divided by the absolute maximum drawdown."""
    mdd = abs(max_drawdown(returns))
    if mdd == 0:
        return 0.0
    return cagr(returns, periods_per_year) / mdd


def var_cvar(returns: pd.Series, level: float = 0.05) -> tuple[float, float]:
    """Historical Value-at-Risk and Conditional VaR at the given tail level."""
    r = returns.dropna()
    if len(r) == 0:
        return 0.0, 0.0
    var = float(np.quantile(r, level))
    cvar = float(r[r <= var].mean()) if (r <= var).any() else var
    return var, cvar


# ---------------------------------------------------------------------------
# Trade-level statistics (operate on realised trade PnLs, not periodic returns)
# ---------------------------------------------------------------------------
def win_rate(trade_pnls: pd.Series | np.ndarray) -> float:
    t = np.asarray(trade_pnls, dtype=float)
    return float((t > 0).mean()) if t.size else 0.0


def profit_factor(trade_pnls: pd.Series | np.ndarray) -> float:
    """Gross profit / gross loss. inf if there are no losing trades."""
    t = np.asarray(trade_pnls, dtype=float)
    gains = t[t > 0].sum()
    losses = -t[t < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


# ---------------------------------------------------------------------------
# Exposure / turnover (operate on a weight matrix: rows=dates, cols=assets)
# ---------------------------------------------------------------------------
def exposure(weights: pd.DataFrame) -> float:
    """Average gross exposure (sum of |weights|) across time."""
    if weights is None or weights.empty:
        return 0.0
    return float(weights.abs().sum(axis=1).mean())


def turnover(weights: pd.DataFrame, periods_per_year: int = TRADING_DAYS) -> float:
    """
    Average annualised one-way turnover: mean per-period |Δw| summed across
    assets, scaled to a yearly figure.
    """
    if weights is None or weights.empty:
        return 0.0
    per_period = weights.diff().abs().sum(axis=1).mean()
    return float(per_period * periods_per_year)


# ---------------------------------------------------------------------------
# Rolling metrics (for time-series diagnostics)
# ---------------------------------------------------------------------------
def rolling_sharpe(returns: pd.Series, window: int = 126,
                   periods_per_year: int = TRADING_DAYS) -> pd.Series:
    mean = returns.rolling(window).mean()
    std = returns.rolling(window).std(ddof=1)
    return (mean / std) * np.sqrt(periods_per_year)


def rolling_vol(returns: pd.Series, window: int = 63,
                periods_per_year: int = TRADING_DAYS) -> pd.Series:
    return returns.rolling(window).std(ddof=1) * np.sqrt(periods_per_year)


# ---------------------------------------------------------------------------
# Calendar aggregations
# ---------------------------------------------------------------------------
def monthly_returns_table(returns: pd.Series) -> pd.DataFrame:
    """Pivot of monthly compounded returns (rows=year, cols=month)."""
    r = returns.dropna()
    if r.empty:
        return pd.DataFrame()
    monthly = (1 + r).resample("ME").prod() - 1
    tbl = monthly.to_frame("ret")
    tbl["year"] = tbl.index.year
    tbl["month"] = tbl.index.month
    return tbl.pivot_table(index="year", columns="month", values="ret")


def annual_returns(returns: pd.Series) -> pd.Series:
    r = returns.dropna()
    if r.empty:
        return pd.Series(dtype=float)
    return (1 + r).resample("YE").prod() - 1


# ---------------------------------------------------------------------------
# One-shot summary
# ---------------------------------------------------------------------------
def performance_summary(returns: pd.Series,
                        weights: pd.DataFrame | None = None,
                        trade_pnls: pd.Series | np.ndarray | None = None,
                        rf_annual: float = 0.0,
                        periods_per_year: int = TRADING_DAYS) -> dict:
    """Return a flat dict of the standard institutional metric set."""
    r = returns.dropna()
    summary = {
        "cagr": cagr(r, periods_per_year),
        "ann_return": annualized_return(r, periods_per_year),
        "ann_vol": annualized_vol(r, periods_per_year),
        "sharpe": sharpe_ratio(r, rf_annual, periods_per_year),
        "sortino": sortino_ratio(r, rf_annual, periods_per_year),
        "calmar": calmar_ratio(r, periods_per_year),
        "max_drawdown": max_drawdown(r),
        "var_95": var_cvar(r, 0.05)[0],
        "cvar_95": var_cvar(r, 0.05)[1],
        "n_periods": int(len(r)),
    }
    if weights is not None:
        summary["avg_gross_exposure"] = exposure(weights)
        summary["ann_turnover"] = turnover(weights, periods_per_year)
    if trade_pnls is not None and len(trade_pnls) > 0:
        summary["n_trades"] = int(len(trade_pnls))
        summary["win_rate"] = win_rate(trade_pnls)
        summary["profit_factor"] = profit_factor(trade_pnls)
        summary["avg_trade_pnl"] = float(np.mean(trade_pnls))
    return summary


def format_summary(summary: dict) -> str:
    """Human-readable multi-line rendering of a performance_summary dict."""
    pct = {"cagr", "ann_return", "ann_vol", "max_drawdown", "var_95",
           "cvar_95", "win_rate", "avg_gross_exposure", "ann_turnover"}
    lines = []
    for k, v in summary.items():
        label = k.replace("_", " ").title()
        if k in pct:
            lines.append(f"  {label:<22}: {v:>10.2%}")
        elif isinstance(v, float):
            lines.append(f"  {label:<22}: {v:>10.3f}")
        else:
            lines.append(f"  {label:<22}: {v:>10}")
    return "\n".join(lines)
