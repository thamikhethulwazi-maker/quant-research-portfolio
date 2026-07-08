"""
quant_framework
===============
Shared, strategy-agnostic research infrastructure for the quant portfolio.

Modules
-------
metrics     : performance & risk metrics (Sharpe, Sortino, Calmar, DD, ...)
validation  : expanding / rolling walk-forward with strict OOS separation
robustness  : Monte Carlo, stationary-bootstrap CIs, parameter sensitivity
plotting    : publication-quality research charts
data        : reproducible synthetic data generators + real-data interface

Design goals: deterministic, reproducible, no look-ahead, easy to swap in real
data. See the top-level README for the research standards this enforces.
"""
from . import metrics, validation, robustness, plotting, data  # noqa: F401

__all__ = ["metrics", "validation", "robustness", "plotting", "data"]
__version__ = "0.1.0"
