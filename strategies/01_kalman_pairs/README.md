# Strategy 1 — Kalman Filter Pairs Trading

**Status:** ✅ Complete · **Headline OOS Sharpe:** ~0.37 (expanding) / ~0.41 (rolling), 95% bootstrap CI **[−0.60, 1.19]**

---

## Research motivation
Pairs trading is the canonical statistical-arbitrage strategy: find two assets
whose prices are tied together by a common economic driver, trade the spread
when it dislocates, and collect the reversion. The hard part in practice is that
the relationship between two assets is **not constant** — the hedge ratio drifts
as fundamentals, index weights and liquidity change. A static OLS hedge ratio
estimated once is stale almost immediately.

## Economic intuition
If two assets are cointegrated, a linear combination of their prices
(`spread = y − β·x`) is stationary and mean-reverting even though each leg is a
non-stationary random walk. Trading the standardised spread (z-score) is a bet
on reversion, not on market direction — the position is dollar-neutral, so it is
(in principle) insulated from broad market moves.

## Academic background
- Kalman, R.E. (1960) — the recursive linear filter itself.
- Chan, E.P. (2013), *Algorithmic Trading* — Kalman hedge-ratio pairs.
- Clegg & Krauss (2018), *Pairs trading with partial cointegration*, *Quant.
  Finance* 18(1).

## Implementation methodology
1. **Dynamic hedge ratio via Kalman filter.** β is modelled as a random walk;
   the filter updates it each bar using only past-and-present data (fully
   causal). `delta` controls adaptation speed.
2. **Spread & signal.** The spread is standardised with a rolling 20-day mean/std
   into a z-score. Enter when `|z| > entry_z`, exit when `|z| < exit_z`, with a
   stateful position so we don't re-enter mid-trade.
3. **Position construction.** A +1 signal = long 1 unit of *y*, short *β* units of
   *x*, with both legs normalised so gross exposure = 1 (dollar-neutral).
4. **Costs.** 1 bp one-way charged on leg turnover.

## What I changed from the original code (and why)
| Issue in original | Consequence | Fix |
|---|---|---|
| Backtest used `signal·(ret_y − ret_x)` | The estimated hedge ratio β was **thrown away**; the "hedge" left naked directional exposure whenever β≠1 | Position now weighted **long 1·y / short β·x**, gross-normalised |
| No transaction costs | Frequent flips looked free | 1 bp/side on turnover |
| Parameters hand-picked on full sample | Implicit optimisation on test data | Moved all selection into **walk-forward** (see below) |
| Demo data not truly cointegrated | Spread not guaranteed stationary | Synthetic pair has an explicit stationary **OU spread** + drifting β |

The Kalman `update()` was already causal and was preserved unchanged.

## Validation process
- **Walk-forward, strict OOS.** Parameters (`entry_z`, `exit_z`, `delta`) are
  chosen by maximising **in-sample** Sharpe on each training fold, then applied
  untouched to the next test fold. Test folds are non-overlapping and
  concatenated into one honest OOS track record.
- Both **expanding** and **rolling** (504-day) windows are run; results agree
  closely, which is a good robustness sign.
- **Bootstrap** (stationary, avg block 10) puts a confidence interval on the
  Sharpe. **Monte Carlo** block-bootstrap resamples the OOS returns into 1,000
  equity paths. A **parameter-sensitivity heatmap** (in-sample only) checks
  whether the edge is a broad plateau or a lone overfit spike.

## Results (out-of-sample)
| Metric | Expanding | Rolling |
|---|---|---|
| CAGR | 1.4% | 1.5% |
| Ann. vol | 4.0% | 3.8% |
| Sharpe | 0.37 | 0.41 |
| Sortino | 0.14 | 0.16 |
| Calmar | 0.24 | 0.28 |
| Max drawdown | −5.9% | −5.4% |

Bootstrap Sharpe 95% CI **[−0.60, 1.19]**, P[Sharpe > 0] ≈ **80%**.
Monte Carlo median terminal wealth ≈ 1.06 over the OOS window; 5th-percentile
max drawdown ≈ −12%.

Figures: `outputs/01_kalman_pairs/` — equity/drawdown, rolling metrics, monthly
heatmap, annual bars, parameter sensitivity, Monte Carlo fan, bootstrap
histogram.

## Honest assessment
The edge is **real but small**, and the bootstrap CI **includes zero** — with
this much data we cannot reject "no skill" at 95% confidence. That is the
correct, non-overfit conclusion. The consistency between expanding and rolling
schemes and the smooth parameter surface argue against a pure fluke, but the
Sharpe is not, on its own, deployable capital-at-scale territory.

## Limitations
- **Synthetic data.** Results validate *implementation correctness and the
  mechanics of the edge*, not live-market viability. Real pairs face regime
  breaks, cointegration decay, borrow costs and crowding not modelled here.
- Single pair — no portfolio of pairs, so no diversification of the (thin) edge.
- Costs are a flat 1 bp; real bid/ask, market impact and short borrow would bite.
- No cointegration test gate (Engle–Granger / Johansen) before trading — a live
  system should trade only pairs that pass a rolling cointegration test.

## Practical considerations & capacity
Dollar-neutral single-name pairs scale to low-to-mid eight figures before market
impact on the thinner leg dominates; capacity is set by the least liquid leg.
The real-world version should (a) screen a universe for cointegration, (b) run
dozens of pairs for diversification, (c) add a cointegration-breakdown stop, and
(d) model borrow. As a single-pair demo the honest capacity statement is: this
is a *mechanism* demonstration, not a capacity study.

## Conclusion
A correctly-implemented, look-ahead-free Kalman pairs strategy extracts a small,
statistically-marginal reversion premium from a cointegrated spread. The value
of this module for a portfolio is less the Sharpe than the demonstration of
**disciplined validation**: the hedge-ratio bug fix, cost accounting, and
walk-forward OOS testing are exactly what separates a defensible backtest from a
misleading one.

## Reproduce
```bash
python strategies/01_kalman_pairs/run.py
```
Deterministic (seed=42). Swap in real data by replacing
`data.cointegrated_pair(...)` with any provider returning a two-column price
DataFrame.
