# Strategy 2 — Earnings Implied-Volatility Crush (Variance Risk Premium)

**Status:** ✅ Complete · **Headline per-event OOS Sharpe:** ~0.90, 95% bootstrap CI **[0.40, 2.17]**, but with a **−15% worst-event tail**

---

## Research motivation
Implied volatility inflates before an earnings announcement because the market
pays up for uncertainty. Once earnings land, uncertainty resolves and IV
collapses. The folk version of the trade — "sell vol before earnings, IV always
crushes" — is real but dangerously incomplete: the crush is only profitable if
the premium you collect exceeds the cost of the move the stock actually makes.

## Economic intuition (the real edge)
The edge is a **variance risk premium (VRP)**: options-implied moves are, on
average and for certain names, *larger* than realised moves. Selling that
over-priced insurance earns a premium — but you are short a fat left tail,
because occasionally a stock gaps far beyond the implied move and the short
straddle loses several multiples of the premium collected.

## Academic background
- Goyal & Saretto (2009), *Cross-section of option returns and volatility*, JFE.
- Muravyev & Pearson (2020), *Option Trading Costs Are Lower Than You Think*, RFS.
- CBOE (2023), *S&P 500 Earnings Events Implied vs Realised Move Study*.

## ⚠️ The bug that mattered: look-ahead bias in the original
The original code selected which events to trade using `expected_crush(iv_pre,
iv_exit)` — where `iv_exit` is the **realised post-earnings IV**. That number
does not exist at entry. **The strategy only appeared to work because it was
allowed to see the answer before betting.** This is the single most common and
most fatal error in derivatives backtests.

**The fix (this implementation):**
- Entry decisions use **only** pre-earnings information (`iv_pre`,
  `implied_move_pct`) plus a premium estimate built on **past events only**.
- Which names to trade is decided by `estimate_name_premium()` fit on the
  training quarters and applied out-of-sample; a name is shorted only if its
  historically-estimated (implied − realised) gap is positive and its
  pre-earnings IV clears a floor.
- The realised move and post-earnings IV are used **solely to price the P&L of a
  trade already committed to** — never to decide whether to place it.

## Implementation methodology
1. **Faithful trade P&L** (`short_straddle_event_pnl`): sell an ATM straddle at
   `iv_pre` with ~5 DTE; one day later buy it back at the crushed `iv_post`,
   struck at the original strike, priced on the **moved** spot with correct
   remaining tenor. This captures all three drivers — IV crush (gain), one day
   of theta (gain), intrinsic cost of the move (loss) — with no hand-waving.
2. **Costs:** 15 bps of notional round-trip (option spreads are wide).
3. **Selection model** fit on training events, applied OOS.

## Validation process
- **Expanding walk-forward over earnings quarters.** For each test quarter *q*,
  the per-name premium is estimated using only quarters `< q`, then quarter *q*
  is traded out-of-sample. OOS quarters are concatenated. This is a bespoke
  (not generic) walk-forward because a per-name *model* — not just scalar
  hyper-parameters — must be estimated on past data only.
- **Bootstrap** Sharpe CI and **Monte Carlo** on the per-event stream.
- **Parameter-sensitivity heatmap** over the two selection thresholds
  (`min_premium`, `min_iv_pre`).

## Results (out-of-sample, per-event = honest risk unit)
| Metric | Value |
|---|---|
| Ann. return | ~4.8% |
| Ann. vol | ~5.3% |
| **Sharpe (per-event)** | **~0.90** |
| Sortino | ~0.43 |
| Max drawdown | **−16%** |
| Win rate | 86% |
| Profit factor | 3.7 |
| VaR / CVaR (95%) | −1.8% / **−7.4%** |
| Worst single event | **−15% of notional** |

Bootstrap Sharpe 95% CI **[0.40, 2.17]**, P[Sharpe > 0] = 100%.

## The methodology trap I deliberately expose
Aggregating the ~10–30 simultaneous earnings events each quarter into a single
quarterly return produces a **quarterly Sharpe of ~2.8** — versus **0.90**
per-event. That 3× inflation is an artifact: it assumes you can perfectly
diversify away idiosyncratic move risk across names that all report in the same
window. In live markets, earnings cluster and short-vol books all bleed together
when market-wide vol spikes. **The per-event number is the honest one; the
quarterly number is a cautionary example, not a result.** (See the CVaR ≫ VaR
gap and the −15% worst event: the risk is emphatically not diversified away.)

## Honest assessment
Removing the look-ahead bias transformed the strategy from "trivially and
falsely profitable" into "profitable **only** where a genuine ex-ante premium
exists in the selected names, and only if you can survive the tail." The
positive expectancy is real on this data, but 86% win rate + a −15% tail is the
textbook short-vol return profile: you get paid steadily to warehouse a risk
that occasionally detonates. Sizing, not signal, is what kills these strategies.

## Limitations
- **Synthetic data.** Validates *implementation correctness and the mechanics of
  the VRP*, not live viability. The premium's size here is a modelling choice;
  in live markets the VRP has compressed as the trade crowded.
- **No true early-exercise / American features, no skew, no term structure.** A
  single ATM straddle at one tenor is a simplification of real vol surfaces.
- Realised-move tail is calibrated (1-in-12 surprise gap of 3–8×); real gap
  distributions are heavier and more state-dependent (e.g. macro regimes).
- Costs are a flat 15 bps of notional; real option spreads widen exactly when
  you most want to close (post-print), worsening the tail.

## Practical considerations & capacity
- **Use defined-risk structures** (iron condors / short strangles with wings),
  not naked straddles, so the −15% tail becomes a known, capped loss.
- **Size to the tail, not the mean.** With a plausible 1-in-12 blow-up, position
  size should assume a multi-multiple loss on the worst leg.
- Capacity is set by single-name option open interest and the ability to exit
  the morning after into a wider market; realistically low-to-mid seven figures
  per name before you move the vol you are trying to sell.

## Conclusion
This strategy's value to the portfolio is as much a **methodology exhibit** as a
signal: it shows (1) how to detect and remove a load-bearing look-ahead bias,
(2) how to reframe a folk trade around its actual economic edge (the VRP), and
(3) how naive aggregation can inflate a Sharpe threefold. The leakage-free edge
is genuine but tail-dominated — deployable only with defined risk and tail-aware
sizing.

## Reproduce
```bash
python strategies/02_earnings_iv_crush/run.py
```
Deterministic (seed=7). Swap in real data by replacing
`data.earnings_iv_panel(...)` with a provider returning the same event schema
(pre-earnings IV and spot known at entry; realised move / post IV known at exit).
