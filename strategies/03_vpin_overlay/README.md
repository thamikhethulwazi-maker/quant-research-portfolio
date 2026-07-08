# Strategy 3 — VPIN Order-Flow Toxicity Overlay

**Status:** ✅ Complete · **Headline:** overlay improves OOS **Sharpe +0.48 / Sortino +0.53** and cuts max drawdown (−3.44% → −2.88%) by de-risking toxic flow. VPIN is a **risk overlay, not a standalone alpha.**

---

## Research motivation
Market makers and mean-reversion strategies provide liquidity — they profit by
fading transient moves. Their nightmare is **adverse selection**: trading against
better-informed counterparties whose order flow keeps pushing price the "wrong"
way. VPIN (Volume-Synchronised Probability of Informed Trading) tries to measure,
in real time, how *toxic* current flow is, so a liquidity provider can step back
before it gets run over.

## Economic intuition
VPIN works in **volume time**, not clock time: it fills equal-volume buckets and
measures the buy/sell imbalance in each. Persistent one-sided imbalance signals
informed trading; the rolling average of bucket imbalances is VPIN. High VPIN ⇒
elevated adverse-selection risk ⇒ widen, reduce size, or stand aside.

## Academic background
- Easley, López de Prado & O'Hara (2012), *Flow Toxicity and Liquidity in a
  High-Frequency World*, RFS 25(5).
- Easley, López de Prado & O'Hara (2011), *The Microstructure of the Flash Crash*.

## Why this is tested as an overlay
VPIN has **no P&L of its own** — it is a sizing signal. Testing it in isolation
is meaningless. So it is applied to a **base strategy** (here a short-term
mean-reversion / liquidity-provision fader), and the question is strictly:
*does conditioning position size on VPIN improve out-of-sample risk-adjusted
performance and cut tail risk?*

## Audit & fixes vs. the original
| Issue | Fix |
|---|---|
| `bucket_size = mean(volume)·0.5` used the **full sample** | Estimated on **training volume only** |
| `size_multiplier` applied without lag | **Lagged one bar** before scaling any position (the estimator and multiplier are both causal) |
| No base strategy → nothing to evaluate | Added a base mean-reversion strategy + base-vs-overlay OOS comparison |
The VPIN estimator itself (bulk-volume classification + volume buckets) was
already causal and is preserved.

## Validation process
- **Expanding walk-forward** over 5 folds selects the overlay thresholds
  (`cut_at`, `floor_at`) by maximising **training** Sortino of the overlaid
  strategy, applied to untouched OOS folds.
- **Conditional diagnostic:** base returns bucketed into VPIN quintiles (the
  direct evidence).
- **Bootstrap** Sharpe CI on the overlaid stream; **parameter-sensitivity**
  heatmap over the two thresholds (training only).

## Results (out-of-sample)
**The key evidence — base Sharpe by VPIN quintile (Q5 = most toxic):**

| Quintile | Q1 | Q2 | Q3 | Q4 | Q5 |
|---|---|---|---|---|---|
| Base ann. Sharpe | 14.0 | 13.5 | 13.4 | 6.9 | **0.6** |

Base performance **declines monotonically** as VPIN rises and collapses to ~0 in
the most toxic quintile. VPIN is genuinely separating benign from toxic periods.

**Overlay effect (same OOS periods):**

| | Base | Overlaid | Δ |
|---|---|---|---|
| Ann. return | 119.8% | 115.4% | −4.4% |
| Ann. vol | 14.1% | 12.9% | **−1.2%** |
| Sharpe | 8.48 | 8.95 | **+0.48** |
| Sortino | 8.31 | 8.84 | **+0.53** |
| Max drawdown | −3.44% | −2.88% | **+0.56%** |

The overlay gives up a little return to buy a better risk profile — exactly what
a toxicity filter should do.

## ⚠️ Honest caveat on the absolute numbers
The **absolute** Sharpes (~8–9) are **inflated by intraday annualisation**: with
~19,656 five-minute periods per year, the √N scaling produces large annualised
Sharpes from modest per-bar edges. This is a property of the annualisation
convention, not evidence of a world-beating strategy. **The credible, transferable
results are the *deltas* (overlay vs base) and the *quintile pattern*, not the
absolute level.** Real intraday reversion also faces queue position, latency, and
impact costs a bar-level backtest cannot capture.

## Limitations
- **Synthetic data** with a deliberately engineered (and detectable) toxicity–
  forward-risk link. In live markets the predictive power of VPIN is contested:
  it flagged stress before the 2010 Flash Crash, but later studies find its
  forecasting value is noisier and sample-dependent.
- Bulk-volume classification is a proxy; true signed volume (available on some
  venues) would change the estimate.
- The base strategy is a stylised liquidity-provision proxy, not a production
  market-making model (no inventory, no quoting, no queue dynamics).

## Practical considerations & capacity
- VPIN is best used as a **governor**, not a signal: reduce quoting size / widen
  spreads / pause when it spikes. Capacity is that of the underlying liquidity-
  provision book, not VPIN itself.
- Bucket size and `n_buckets` should be recalibrated per instrument and revisited
  as volume profiles drift; a stale bucket size silently degrades the estimate.
- The overlay's value is realised as **tail reduction and drawdown control**, so
  it should be judged on Sortino / Calmar / max-DD, not headline return.

## Conclusion
Conditioning position size on VPIN cleanly separates benign from toxic periods on
this data and improves out-of-sample risk-adjusted performance. Presented
correctly, the result is not "VPIN makes money" — it is "VPIN is an informative
**risk governor** whose value shows up in the second moment and the tail." The
absolute return figures are annualisation artifacts and are flagged as such.

## Reproduce
```bash
python strategies/03_vpin_overlay/run.py
```
Deterministic (seed=23). Swap in real data by supplying intraday bars with
`close` and `volume` (and optionally true signed volume).
