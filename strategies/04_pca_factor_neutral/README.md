# Strategy 4 — PCA Factor-Neutral Long/Short

**Status:** ✅ Complete · **Headline OOS Sharpe:** **1.71**, 95% bootstrap CI **[0.78, 2.65]**, P[Sharpe>0]=100%. Genuinely factor-neutral (avg |net PC loading| ≈ 0.06).

---

## Research motivation
Most of a stock's daily move is systematic — market, sector, and style factors
it shares with everything else. That systematic part is not alpha; it is risk to
be hedged. What is left after removing it, the **idiosyncratic residual**, is
where stock-specific mispricing lives, and residuals tend to mean-revert. This
strategy isolates residuals with PCA and trades their reversion while staying
neutral to the factors.

## Economic intuition
PCA on the cross-section of returns extracts the dominant common directions
(principal components ≈ statistical factors). Projecting each stock's return onto
those components and subtracting gives the residual. A stock whose residual has
run far below its recent mean is "cheap" relative to its factor-implied value and
tends to revert up; one far above is "rich" and tends to revert down. Long the
cheap, short the rich, factor-neutral.

## Academic background
- Avellaneda & Lee (2010), *Statistical Arbitrage in the U.S. Equities Market*,
  Quantitative Finance 10(7) — the canonical PCA-residual stat-arb reference.
- Connor & Korajczyk (1988) — APT / statistical factor foundations.

## Audit & fixes vs. the original
| Issue | Fix |
|---|---|
| Risk of fitting PCA on the full sample | PCA + standardiser **fit on each training fold and FROZEN**; only *transform* is applied OOS |
| Weight-sign convention (my refactor introduced an inversion) | Verified against the information coefficient: long undervalued (negative z) = **positive** weight; caught because IC was positive while P&L was negative |
| "Factor-neutral" asserted but never checked | Added `realized_factor_exposure()` — reports avg \|net PC loading\| (≈0.06, confirming neutrality) |
The rolling residual z-score and the `shift(1)` in the backtest were already
causal and were preserved.

## Implementation methodology
1. Fit PCA (K=3 components, ~71% variance) on the training fold; freeze it.
2. Compute residuals on new data with the frozen model; standardise each asset's
   residual with a rolling (causal) z-score.
3. Long the `n_long` most-negative z, short the `n_short` most-positive z,
   equal-weighted, dollar-neutral. Rebalance every 3 days; 5 bps turnover cost.

## Validation process
- **Expanding walk-forward**, refitting PCA per training fold and selecting
  (`entry_z`, `lookback`) on training Sharpe only, applied to untouched OOS folds.
- **Bootstrap** Sharpe CI; **Monte Carlo** equity paths; **parameter-sensitivity**
  heatmap over `entry_z` × `lookback` (in-sample).
- **Factor-neutrality diagnostic** on the realised book.

## Results (out-of-sample)
| Metric | Value |
|---|---|
| CAGR | 35.0% |
| Ann. return | 31.8% |
| Ann. vol | 18.6% |
| **Sharpe** | **1.71** |
| Sortino | 1.64 |
| Calmar | 1.27 |
| Max drawdown | −27.7% |
| Avg gross exposure | 161% |
| Avg \|net factor loading\| (per PC) | ≈ 0.06 |

Bootstrap Sharpe 95% CI **[0.78, 2.65]**, P[Sharpe > 0] = 100%.

## Honest assessment
This is the strongest and most statistically defensible strategy in the portfolio
so far: a positive Sharpe whose bootstrap CI clears zero, on a genuinely
factor-neutral book. Two honesty caveats: (1) the **18.6% volatility is high for a
market-neutral book** (real ones run 6–10%) because the synthetic idiosyncratic
reversion is cleaner and stronger than live markets, and gross exposure is ~160%;
(2) the diagnostic confirms neutrality on the *statistical* PCs, which are not the
same as tradable economic factors — a live version should also neutralise to
observable factors (sector, size, value) and to beta.

## Limitations
- **Synthetic data** with engineered idiosyncratic reversion; live residual
  reversion is weaker, noisier, and has decayed as the trade crowded.
- PCA factors are statistical, not economic — they rotate and their number (K) is
  a modelling choice; a scree/marginal-variance rule should set K in production.
- No borrow costs, no shorting constraints, no capacity/impact model beyond the
  flat 5 bps.

## Practical considerations & capacity
- **Turnover is high (~250×/yr).** At that rate, transaction costs and borrow are
  the binding constraint, not signal — capacity is set by how much residual-
  reversion liquidity exists in the traded names before impact erodes the edge.
  A production version would add a no-trade band around targets and cost-aware
  rebalancing to cut turnover.
- Vol targeting and gross-exposure caps would bring the 18.6% vol down to a
  market-neutral-appropriate 6–10% and shrink the −28% drawdown.
- K, lookback and rebalance frequency should be re-estimated as factor structure
  drifts.

## Conclusion
A correctly-validated PCA residual-reversion book with a real, factor-neutral,
statistically-significant OOS edge — the portfolio's cleanest positive result.
The honest qualifiers (synthetic-inflated vol, high turnover, statistical vs
economic factors) are exactly the questions a desk would need answered before
allocating.

## Reproduce
```bash
python strategies/04_pca_factor_neutral/run.py
```
Deterministic (seed=11). Swap in real data by passing a (T×N) returns DataFrame.
