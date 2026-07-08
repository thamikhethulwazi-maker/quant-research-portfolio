# Strategy 6 — Hidden Markov Regime Detection

**Status:** ✅ Complete · **Headline:** a 2-state HMM recovers the latent calm/turbulent regime at **91.6% OOS accuracy**; gating a long book on it improves **Sharpe 0.97 → 1.05, Calmar 0.52 → 0.93**, and cuts **max drawdown −42% → −13%**. A regime *filter*, not standalone alpha.

---

## Research motivation
Markets are not one process — they alternate between calm, trending periods and
turbulent, high-volatility ones, and a strategy (or a plain long book) that is
fine in the first can be wrecked in the second. If you could tell, in real time
and without hindsight, which regime you're in, you'd carry less risk into the
turbulent stretches. The catch is that the regime is *latent* — you never observe
it directly, only its noisy symptoms.

## Economic intuition
A Hidden Markov Model treats the regime as an unobserved state that emits
observable features (returns, realised volatility) with regime-specific
distributions. Fitting the HMM recovers those distributions and the transition
probabilities between states; decoding then gives the most likely regime at each
point. The overlay conditions exposure on the inferred regime: full risk when
calm, trimmed when turbulent.

## Academic background
- Hamilton (1989) — the foundational regime-switching model.
- Nystrup et al. (2020) — regime-based allocation with HMMs.

## Why 2 states, not 3
The original design used 3 states (calm/bear/crash). On realistic data the crash
regime is rare (~3% here), too infrequent for a 3-state Gaussian HMM to learn
stably — recovery accuracy was ~46% and unstable across folds, and the state
labels didn't map cleanly to the true regimes. Collapsing to **2 states
(calm vs turbulent)** — the standard robust choice in the literature — lifts OOS
accuracy to **91.6%** and makes it stable. This is a deliberate, documented
trade of resolution for reliability, not a tuning trick.

## Audit & fixes vs. the original
| Issue | Fix |
|---|---|
| Model / scaler fit discipline under walk-forward | HMM **and** `StandardScaler` fit on each training fold only, then frozen for OOS |
| HMM states are permutation-invariant across refits | **Pinned** the state→regime map by sorting states on mean realised volatility (lowest = calm) — labels reproduce across folds |
| Regime detection asserted, never measured | Added ground-truth labels to the generator and report **OOS recovery accuracy** (91.6%) and a confusion vs truth |
| "Regime overlay helps" untested | Base-vs-gated OOS comparison on a static long book |

## Validation process
- **Expanding walk-forward**: HMM refit per training fold, regimes inferred OOS.
- **Regime-recovery accuracy** vs ground truth (the direct evidence).
- **Base-vs-gated** comparison on the same OOS span; **bootstrap** Sharpe CI.

## Results (out-of-sample)
| | Base (static long) | Regime-gated | Δ |
|---|---|---|---|
| Ann. return | 22.6% | 12.5% | −10.1% |
| Ann. vol | 23.3% | 11.9% | **−11.4%** |
| Sharpe | 0.97 | 1.05 | **+0.08** |
| Sortino | 0.97 | 1.05 | +0.09 |
| Calmar | 0.52 | 0.93 | **+0.41** |
| Max drawdown | −42.2% | −13.5% | **+28.7%** |

OOS regime-recovery accuracy **91.6%** (per fold: 90%, 100%, 91%, 91%, 86%).

## Honest assessment
The detector genuinely works — 91.6% is a strong, stable recovery rate, and the
regime timeline visibly tracks the true turbulent stretches. The overlay's value
is overwhelmingly in **risk reduction**: it nearly halves volatility and cuts the
drawdown by two-thirds, while only nudging the Sharpe up (+0.08). That small
Sharpe gain is the honest number — the turbulent regime here still has positive
average return (it includes sharp volatile rallies, not just declines), so
de-risking it costs return as well as saving it. The right way to read this
strategy is Calmar and max-drawdown, where the improvement is large and clear,
not headline return.

## Limitations
- **Synthetic** regime-switching data with cleaner state separation than live
  markets; real regimes are fuzzier and transitions noisier.
- Gaussian emissions assume within-regime normality; real returns are fat-tailed
  (a Student-t or regime-switching-GARCH emission would be more faithful).
- The HMM confirms the current regime with a short lag; it does not *predict*
  transitions, so it protects against persistent turbulence, not sudden jumps.
- 2 states buy robustness at the cost of resolution — a true crash state is
  folded into "turbulent."

## Practical considerations & capacity
- Best used as a **portfolio-level risk governor** (scale gross exposure / target
  vol by regime), not as a signal generator. Capacity is that of the underlying
  book.
- Retrain periodically; monitor the transition matrix and state means for drift.
- Pair with, don't replace, an explicit vol target — the two are complementary.

## Conclusion
A correctly-validated latent-regime detector that reliably separates calm from
turbulent markets out-of-sample and, used as an exposure overlay, delivers a
large drawdown and volatility reduction for a small Sharpe improvement. The
2-state choice and the modest Sharpe gain are reported plainly rather than dressed
up.

## Reproduce
```bash
python strategies/06_hmm_regime/run.py
```
Deterministic (seed=13, HMM random_state=42). Swap in real data by passing a
price series to `build_features`.
