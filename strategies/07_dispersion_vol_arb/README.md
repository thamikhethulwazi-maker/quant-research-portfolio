# Strategy 7 — Dispersion Volatility Arbitrage

**Status:** ✅ Complete · **Headline:** per-event Sharpe **0.87**, 95% CI **[0.69, 1.10]**, win rate **86.5%**, worst event **−0.034 variance pts**. A structural correlation-premium harvest with a fat crisis left tail.

---

## Research motivation
Index options are persistently expensive relative to the options on the stocks
that make up the index. The reason is demand: institutions buy index puts to
hedge portfolios, bidding up index implied volatility — which is mathematically
equivalent to the index pricing in a high *implied correlation* between its
members. Dispersion trading sells that rich index vol and buys the cheaper
constituent vol, collecting the difference when stocks actually move more
independently than the index implied.

## Economic intuition
If you know each stock's implied vol and the index's implied vol, you can back
out the average pairwise correlation the market is pricing (solve for the ρ that
reconciles them). That implied correlation is usually higher than what
subsequently realises — the premium. The trade is short index variance / long
constituent variance, and it profits when **realised correlation < implied**.
The defining risk: in a crisis, correlations gap toward 1, and the trade loses on
both legs simultaneously.

## Academic background
- Bossu (2005) — arbitrage pricing of correlation swaps.
- Deng (2008) — dispersion trading empirics.
- Driessen, Maenhout & Vilkov (2009) — the priced correlation risk premium.

## Audit & fixes vs. the original
| Issue | Fix |
|---|---|
| **The demo guaranteed a profit** — realised correlation was *defined* as implied minus a positive draw, so the trade could never lose | Rebuilt the data so realised correlation **spikes above implied ~12% of the time** (crisis), giving the trade its true asymmetric, fat-left-tail payoff |
| In-sample only | Walk-forward: the implied-correlation entry threshold is chosen on **past events only** and applied out-of-sample |
| No tail reporting | Per-event distribution, win rate, and **worst event** are the headline, not a smooth aggregate |
The bisection implied-correlation solver and the variance-P&L accounting were
sound and are preserved.

## Validation process
- **Walk-forward** over the implied-correlation entry threshold, chosen on
  training events by per-event Sharpe, applied to future events.
- **Per-event** Sharpe, win rate, and worst-case (not a diversified aggregate).
- **Bootstrap** per-event Sharpe CI (reported in raw per-event units — annualising
  event returns by 252 would be meaningless and is explicitly avoided).

## Results (out-of-sample, 200 events)
| Metric | Value |
|---|---|
| Per-event Sharpe | **0.87** |
| Bootstrap 95% CI | **[0.69, 1.10]** (P>0 = 100%) |
| Win rate | 86.5% |
| Avg net P&L | +0.0112 variance pts |
| Worst event | −0.0341 variance pts |
| Avg implied − realised corr | +0.092 |

The implied-vs-realised correlation scatter tells the whole story: winning trades
sit **below** the diagonal (realised correlation came in under implied), losses
sit **above** it (a correlation spike). The average spread is positive — the
premium is real — but the left tail is roughly 3× the average gain.

## Honest assessment
This is the same shape as the earnings strategy: a genuine structural premium
(here, the correlation risk premium) with a high win rate and a dangerous left
tail. Two honesty points. First, the entry threshold added little — the walk-
forward kept selecting the loosest cut, meaning the premium exists across the
implied-correlation range rather than only at extremes, so "selectivity" isn't
where the edge is. Second, and more important: **the losing events are not
independent.** Correlation spikes are systemic — a single crisis hits many
constituents and many trades at once — so the effective number of independent
bets is far below the 200-event count, and the true tail is worse than a naive
per-event view suggests. The bootstrap CI does not fully capture that clustering.

## Limitations
- **Synthetic** vol panel; real dispersion also carries a single-factor-model
  approximation error, vega mismatch between legs, and heavy multi-leg option
  transaction costs not modelled here.
- P&L is in idealised variance-swap units; a straddle implementation adds gamma/
  theta management and path dependence.
- Correlation-spike clustering (above) means position sizing must assume the tail
  events arrive together.

## Practical considerations & capacity
- Best expressed via **variance swaps** to avoid gamma/theta management; where
  only listed options exist, delta-hedged straddles with disciplined vega
  matching.
- Size for the **systemic tail**, not the average event — this trade has bankrupted
  desks in past correlation crises (2008, 2011, 2020). A hard tail hedge or a
  regime overlay (see Strategy 6) that cuts exposure in turbulent regimes is a
  natural complement.
- Capacity is reasonable in large, liquid index/constituent option markets, but
  the crowdedness of the trade compresses the premium in calm periods.

## Conclusion
A correctly-validated dispersion strategy with a real, statistically significant
per-event edge from the correlation risk premium — and an honestly-modelled crisis
tail that the original guaranteed-profit demo hid entirely. Judge it on the tail
and the clustering, not the 86% win rate.

## Reproduce
```bash
python strategies/07_dispersion_vol_arb/run.py
```
Deterministic (seed=41). Swap in real data by supplying an events table with
index IV, constituent IVs/weights, and the realised vols/correlation.
