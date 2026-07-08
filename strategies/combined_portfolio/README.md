# Combined Multi-Strategy Portfolio

**Status:** ✅ Complete · **Headline:** three near-uncorrelated alpha sleeves combine into an equal-weight book at **Sharpe 1.97, 8.3% vol, −8% max drawdown** — beating the *average* sleeve (0.80) by a wide margin, even though one sleeve is negative on this split.

---

## What this shows
The point of a multi-strategy book is not to find one perfect strategy — it's to
combine several *imperfect, uncorrelated* ones so their idiosyncratic risks
cancel. This section takes the three daily-return alpha generators and blends
them, to make the diversification math concrete.

Sleeves combined: **Kalman Pairs (S1)**, **PCA Factor-Neutral (S4)**,
**Cross-Sectional Mean Reversion (S5)**. The event-driven strategies (Earnings
IV Crush, Dispersion) and the overlays (VPIN, HMM) don't produce comparable daily
return streams, so they're excluded here — the overlays in particular are meant
to be *applied on top of* sleeves like these, not blended alongside them.

## Results
Per-sleeve Sharpe (fixed-param, held-out span):

| Sleeve | Sharpe |
|---|---|
| Kalman Pairs | −1.0 |
| PCA Factor-Neutral | +2.4 |
| Cross-Sectional MR | +1.0 |
| **Average sleeve** | **0.8** |

Sleeve correlation matrix: essentially zero off-diagonal (|ρ| ≤ 0.02).

| Portfolio | Sharpe | Ann. vol | Max DD |
|---|---|---|---|
| Equal-weight | **1.97** | 8.3% | −8.1% |
| Inverse-vol (risk-parity-lite) | 1.38 | 5.0% | −6.8% |

## The honest reading
Three things worth stating plainly:

1. **Diversification beats the average, not necessarily the best.** The equal-
   weight portfolio (1.97) trounces the average sleeve (0.80) and is far
   steadier — but it does **not** beat the single best sleeve (PCA at 2.4). When
   one strategy dominates, an equal blend dilutes it. The payoff of
   diversification is *robustness*, not a higher ceiling: you don't have to know
   in advance which sleeve will win, and here the book is strong even though the
   Kalman sleeve was **negative** on this split.

2. **Volatility, not return, is where the benefit shows.** Because the sleeves
   are near-uncorrelated, portfolio vol (8.3%) is roughly half a typical single
   sleeve's, and the drawdown is a third of the standalone strategies'. That is
   the diversification "free lunch" — in the second moment.

3. **The near-zero correlation is partly mechanical.** Each sleeve trades an
   independent synthetic universe, so their independence is cleaner than reality.
   On live data these strategies are empirically low-correlation (different
   horizons, different anomalies) but not *zero*, and correlations rise in
   crises — the moment you most need them not to. A real allocation would stress
   the correlation matrix, not take it at face value.

Also note the per-sleeve Sharpes here use **fixed parameters on a single split**,
so they differ from each strategy's walk-forward headline (e.g. PCA's validated
OOS Sharpe is 1.71, not 2.4). This section illustrates portfolio construction; the
per-strategy READMEs carry the rigorously-validated numbers.

## Why equal-weight beat inverse-vol here
Inverse-vol weighting down-weights the high-volatility sleeve — which happened to
be the strongest one (PCA). Sizing by risk alone threw away return that a naive
equal weight kept. It's a clean reminder that risk-parity optimises for balanced
*risk contribution*, not Sharpe, and isn't automatically superior.

## Practical considerations
- A production book would add a **stressed correlation** assumption, a portfolio-
  level vol target, and the overlays as risk governors (HMM to cut gross exposure
  in turbulent regimes, VPIN at the execution layer).
- Rebalancing the sleeve weights itself incurs cost and should be infrequent and
  banded.

## Reproduce
```bash
python strategies/combined_portfolio/build_portfolio.py
```
Deterministic. Each sleeve is recomputed from its own strategy module.
