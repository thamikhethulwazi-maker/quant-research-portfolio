# Strategy 5 — Cross-Sectional Mean Reversion

**Status:** ✅ Complete · **Headline:** the signal is **real gross** (IC +0.042, gross Sharpe ~1.8) but **transaction costs are the binding constraint** — net OOS Sharpe **0.52**, 95% CI **[−0.38, 1.45]** (includes zero). This is an honest study of the gap between a signal and a strategy.

---

## Research motivation
Over short horizons (days), the cross-section of returns mean-reverts: the
relative winners of the last few days tend to give some back and the relative
losers tend to bounce. This is the classic **short-term reversal** effect. It is
one of the most statistically reliable anomalies in the data — and one of the
hardest to actually harvest, because capturing it requires trading fast enough
that costs dominate.

## Economic intuition
Short-term reversal is largely **liquidity provision**: buyers demanding
immediacy push a name up, sellers push it down, and the reversal is the
compensation earned by whoever supplies liquidity against that pressure. Ranking
the universe and going long losers / short winners is a systematic way to provide
that liquidity — but you are paying the spread every time you rebalance, which is
exactly why the net edge is thin.

## Academic background
- Lehmann (1990), *Fads, Martingales, and Market Efficiency*, QJE.
- Jegadeesh (1990), *Evidence of Predictable Behavior of Security Returns*, JF.
- Avellaneda & Lee (2010) — the reversal interpreted as liquidity provision.

## Audit & fixes vs. the original
| Issue | Fix |
|---|---|
| Loser (long) leg assigned a **negative** weight — that is momentum, not reversion | Corrected: **long losers = positive weight**, short winners = negative; **verified via the information coefficient**, not the label |
| In-sample demo reported as result | Expanding walk-forward, params chosen on **training** Sharpe only |
| Costs understated relative to turnover | Kept 5 bps but made **cost/rebalance sensitivity the headline**, since that is where this strategy lives or dies |
The cross-sectional scoring was already causal and is preserved.

## Validation process
- **Sign verification:** IC(−score, next-day return) = **+0.042 > 0**, so losers
  do bounce and the long-loser convention is correct.
- **Expanding walk-forward** over lookback × entry-threshold × rebalance,
  selected on training Sharpe, net of 5 bps.
- **Cost & rebalance sensitivity grid** (the key deliverable).
- **Bootstrap** Sharpe CI and a **parameter-sensitivity** heatmap.

## Results (out-of-sample)
| Metric | Value |
|---|---|
| Information coefficient | +0.042 |
| Net Sharpe (5 bps) | **0.52** |
| Bootstrap 95% CI | **[−0.38, 1.45]** (P>0 = 87%) |
| CAGR | 9.1% |
| Ann. vol | 20.9% |
| Max drawdown | −23.0% |
| Ann. turnover | ~410× |

**The headline — annualised Sharpe vs cost, by rebalance frequency (in-sample):**

| Rebalance | 0 bps | 2 bps | 5 bps | 10 bps | 20 bps |
|---|---|---|---|---|---|
| Daily | **1.78** | 1.44 | 0.91 | 0.05 | −1.68 |
| Every 2 days | 0.95 | 0.71 | 0.36 | −0.23 | −1.38 |
| Every 3 days | 0.75 | 0.56 | 0.28 | −0.18 | −1.08 |

The signal is unambiguous **gross** (Sharpe 1.78) and collapses through zero by
~10 bps. Slowing the rebalance cuts turnover but also throws away the very
short-horizon reversion that is the entire edge. There is no free lunch here.

## Honest assessment
This strategy is a case study in why "positive backtest" and "viable strategy"
are different claims. The reversal is real and the sign is verified, but the net
OOS Sharpe (0.52) has a **bootstrap CI that includes zero**, and profitability is
entirely contingent on trading at institutional cost levels (≤5 bps) with ~410×
annual turnover. At retail costs it is a loser. Reporting it any other way would
be dishonest.

## Limitations
- **Synthetic** cross-sectional reversion; the live short-term-reversal premium
  has compressed as it has been widely harvested.
- Turnover (~410×) makes the result extremely sensitive to the cost and slippage
  model — the single most important and least certain assumption here.
- 20.9% vol / −23% drawdown reflect an ungoverned equal-weight book; vol-targeting
  and a no-trade band are essential in practice.
- Equal-weight legs ignore liquidity — the biggest reversion often sits in the
  least liquid names, where realised costs are worst (adverse to this strategy).

## Practical considerations & capacity
- Viable **only** with low-cost execution: internalised crossing, patient/passive
  fills, and a no-trade band to suppress churn. Capacity is capped by the
  liquidity of the names carrying the reversion, which is precisely where impact
  is highest — so realistic capacity is modest.
- Should be run vol-targeted and combined with slower, higher-capacity signals
  rather than standalone.
- The correct KPI is **net-of-cost** Sharpe under a *conservative* slippage model,
  not the seductive gross number.

## Conclusion
A real, correctly-signed cross-sectional reversal signal whose economic value is
dominated by transaction costs. The professional takeaway is the **cost-
sensitivity curve**, not the point estimate: this is where a desk decides whether
its execution is good enough to trade the effect at all. Presented honestly, with
its zero-inclusive confidence interval and its turnover flagged.

## Reproduce
```bash
python strategies/05_cross_sectional_mr/run.py
```
Deterministic (seed=31). Swap in real data by passing a (T×N) returns DataFrame.
