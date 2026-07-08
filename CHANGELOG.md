# CHANGELOG — Documented Changes from Original Repository

This file is the audit trail. Every change from your original seven files is
recorded here with the **reason** and the **before → after**, so any reviewer
(or future you) can see exactly what was altered and why. Nothing about the
underlying *strategy ideas* was changed — only implementation correctness,
validation rigor, and presentation.

Convention: 🐛 = bug / bias fix · ➕ = addition · ♻️ = refactor · 📝 = docs

---

## 0. Repository-level additions (new — did not exist before)
| | Item | Why |
|---|---|---|
| ➕ | `quant_framework/` package | Your 7 files each re-implemented Sharpe, drawdown, and backtest loops. Extracted into one shared, tested library so logic is written once and validation is identical everywhere. |
| ➕ | `quant_framework/metrics.py` | Single `performance_summary()` covering the full institutional metric set (CAGR, Sharpe, Sortino, Calmar, DD, VaR/CVaR, turnover, exposure, win rate, profit factor, rolling + calendar tables). |
| ➕ | `quant_framework/validation.py` | Expanding **and** rolling walk-forward with enforced train/test separation. The optimiser never sees test data. |
| ➕ | `quant_framework/robustness.py` | Stationary-bootstrap Sharpe CIs, block-bootstrap Monte Carlo, 2-D parameter-sensitivity grids. |
| ➕ | `quant_framework/plotting.py` | Consistent publication-quality figures. |
| ➕ | `quant_framework/data.py` | Reproducible synthetic generators + a `MarketDataProvider` interface for dropping in real data. (Environment has no market-data access — documented in README.) |
| ➕ | `README.md`, `PROGRESS.md`, `requirements.txt`, `run_all.py` | Repo overview/roadmap, resumable progress tracker, pinned deps, one-command reproducer. |
| ➕ | Per-strategy `README.md` | Motivation → intuition → academic background → methodology → validation → results → limitations → conclusion, for each completed strategy. |

---

## 1. Strategy 1 — Kalman Pairs Trading (`01_kalman_pairs/`)

### 🐛 Hedge-ratio ignored in P&L (most important fix)
- **Before:** `strategy_ret = signals.shift(1) * (ret_y - ret_x)` — a flat 1:1
  spread. The time-varying β that the Kalman filter exists to estimate was
  computed and then **discarded**, leaving naked directional exposure whenever
  β ≠ 1.
- **After:** position is properly hedge-ratio weighted and dollar-neutral —
  long `1/(1+|β|)` of *y*, short `β/(1+|β|)` of *x* — with β **lagged one bar**
  so it uses only information available at position-open.
- **Impact:** the "hedge" now actually hedges; reported returns reflect the real
  spread trade rather than a mislabeled directional bet.

### 🐛 No transaction costs
- **Before:** costless fills; a strategy that flips frequently looked free.
- **After:** 1 bp one-way charged on leg turnover in `backtest()`.

### 🐛 Parameters hand-picked on the full sample
- **Before:** `entry_z`, `exit_z`, `delta` chosen by hand and evaluated on all
  data — implicit optimisation on the test set.
- **After:** all three selected inside expanding **and** rolling walk-forward;
  chosen on training folds only, applied to untouched OOS folds.

### ➕ Trade-level statistics
- Added `extract_trade_pnls()` to roll periodic returns into discrete round-trip
  trades for win rate, profit factor, and average holding period.

### ♻️ / 📝 Structure
- Split the single file into `strategy.py` (logic), `run.py` (validation +
  robustness + figures), and `README.md`.
- **Preserved unchanged:** the Kalman `update()` — it was already causal and
  correct. The signal state machine logic is preserved, only lagging/weighting
  around it changed.

### Result
OOS Sharpe **0.37 (expanding) / 0.41 (rolling)**, bootstrap 95% CI
**[−0.60, 1.19]**. Reported honestly as a real-but-marginal edge whose CI
includes zero.

---

## 2. Strategy 2 — Earnings IV Crush (`02_earnings_iv_crush/`)

### 🐛 Look-ahead bias in the entry decision (fatal — was the entire "edge")
- **Before:** `expected_crush(iv_pre, iv_exit)` gated entry on **`iv_exit`, the
  realised post-earnings IV** — information that does not exist at entry time.
  The strategy only appeared profitable because it was allowed to see the
  outcome before betting.
- **After:** entry uses **only** pre-earnings information (`iv_pre`,
  `implied_move_pct`) plus a per-name premium estimate fit on **past events
  only** (`estimate_name_premium()`), applied out-of-sample. Realised move /
  post IV are used *solely* to price the P&L of an already-committed trade.

### 🐛 Dimensionally-wrong delta adjustment
- **Before:** `delta_pnl = -abs(stock_move_pct) * S` — mixed a percentage move
  with a price level and bolted it onto the vol P&L.
- **After:** replaced with a **faithful straddle repricing** in
  `short_straddle_event_pnl()`: sell an ATM straddle at `iv_pre` (~5 DTE), buy
  it back one day later at the crushed `iv_post`, struck at the original strike,
  priced on the **moved** spot with correct remaining tenor. Captures IV crush,
  one day of theta, and the intrinsic cost of the move — no hand-waving.

### 🐛 Reframed around the actual economic edge
- The real edge is the **variance risk premium** (implied move > realised move),
  not "IV falls." Selection now targets names with a genuine, historically
  estimated ex-ante premium.

### ➕ Realistic tail risk in the data
- **Before (original demo):** random IVs with a thin tail — no catastrophic
  losers, so the defining risk of short-vol was invisible.
- **After:** the earnings panel injects a **1-in-12 surprise-gap** of 3–8× the
  typical move. The −15% worst-event tail and CVaR ≫ VaR now show the real
  short-vol risk profile.

### 🐛 / 📝 Honest risk unit
- Report **per-event** metrics (Sharpe ~0.90) as the headline, and flag that
  naive **quarterly aggregation** inflates Sharpe to ~2.8 by pretending
  simultaneous earnings can be fully diversified. Documented as a cautionary
  example, not a result.

### ♻️ Structure
- Split into `strategy.py`, `run.py`, `README.md`. Bespoke walk-forward over
  earnings quarters (a per-name *model* is fit on past data, not just scalars).

### Result
Per-event OOS Sharpe **0.90**, bootstrap 95% CI **[0.40, 2.17]**, worst event
**−15% of notional**. Genuine VRP edge, tail-dominated.

---

## 3. Strategy 3 — VPIN Order-Flow Toxicity Overlay (`03_vpin_overlay/`)

### 🐛 Bucket size used the full sample
- **Before:** `bucket_size = mean(volume)·0.5` over **all** bars — the estimator
  was calibrated with data from the future.
- **After:** estimated on the **first training window's volume only**.

### 🐛 Size multiplier applied without a lag
- **Before:** the VPIN-derived `size_multiplier` scaled the position on the same
  bar VPIN was computed from.
- **After:** multiplier **lagged one bar** before it scales any position. (The
  VPIN estimator itself — bulk-volume classification + volume buckets — was
  already causal and is preserved.)

### ➕ Gave the overlay something to overlay
- **Before:** the original printed VPIN levels but never connected them to P&L,
  so "does it help?" was untestable.
- **After:** added a base mean-reversion / liquidity-provision strategy (the kind
  of book VPIN protects) and a **base-vs-overlaid OOS comparison**, plus a
  conditional diagnostic (base Sharpe by VPIN quintile).

### 📝 Honest framing of an annualisation artifact
- Intraday (5-min) annualisation inflates absolute Sharpes via √N (~19,656
  periods/yr). The README **leads with the overlay delta and the quintile
  pattern**, and flags the absolute level as an artifact.

**Result:** overlay improves OOS **Sharpe +0.48 / Sortino +0.53**, cuts max DD
(−3.44% → −2.88%); base Sharpe by VPIN quintile **14.0 → 13.5 → 13.4 → 6.9 → 0.6**.

---

## 4. Strategy 4 — PCA Factor-Neutral Long/Short (`04_pca_factor_neutral/`)

### 🐛 Look-ahead risk in the factor model
- **Before:** nothing forced the PCA + standardiser to be fit on training data
  only; fitting on the full sample would leak future factor structure.
- **After:** PCA and the scaler are **fit on each training fold and FROZEN**;
  OOS data is only ever *transformed*, never re-fit.

### 🐛 Weight-sign inversion (introduced during refactor, caught by IC)
- **Before:** the equal-weight refactor assigned the **long** (undervalued,
  negative-z) leg a **negative** weight — i.e. it was shorting the cheap names
  (momentum, not reversion).
- **Detection:** the information coefficient was **positive** while P&L was
  **negative** — an impossible combination that revealed the flipped sign.
- **After:** long undervalued = **positive** weight, short overvalued = negative.
  P&L and IC now agree.

### ➕ Verified factor-neutrality instead of asserting it
- Added `realized_factor_exposure()` → avg |net PC loading| ≈ **0.06**,
  confirming the book is genuinely factor-neutral.

### ♻️ Tradeable reversion in the data layer
- `factor_universe` idiosyncratic reversion changed from a 1-day AR term (untradeable
  after costs) to a **multi-day** reversion with a several-day half-life.

**Result:** OOS **Sharpe 1.71**, bootstrap 95% CI **[0.78, 2.65]**, P[>0]=100%.
The portfolio's strongest, cleanest positive result.

---

## 5. Strategy 5 — Cross-Sectional Mean Reversion (`05_cross_sectional_mr/`)

### 🐛 Weight-sign inversion (same pattern as the S4 original)
- **Before:** the loser (long) leg was assigned a **negative** weight — shorting
  the losers, which is momentum, the opposite of the intended reversion.
- **After:** **long losers = positive weight**, short winners = negative, and the
  sign is **verified against the information coefficient** (+0.042), not trusted
  from the label.

### 🐛 In-sample demo reported as a result
- **Before:** a single in-sample synthetic pass printed a Sharpe.
- **After:** expanding walk-forward; parameters chosen on **training** Sharpe only.

### ➕ Cost sensitivity made the headline
- Because short-term reversal lives or dies on execution cost, added a
  **cost × rebalance-frequency Sharpe grid**: gross **1.78** → **0.91** at 5 bps
  → **0.05** at 10 bps → **−1.68** at 20 bps.

**Result:** net OOS **Sharpe 0.52**, bootstrap 95% CI **[−0.38, 1.45]**
(includes zero) — reported honestly as a real-but-cost-bound signal.

---

## 6. Strategy 6 — HMM Regime Detection (`06_hmm_regime/`)
| | Change | Before → After | Reason |
|---|---|---|---|
| 🐛 | Fit discipline under walk-forward | HMM/scaler fit on available data → **HMM *and* `StandardScaler` fit per training fold, frozen for OOS** | No leakage of test-period statistics into the model |
| 🐛 | Label permutation across refits | raw HMM state indices → **pinned map: states sorted by mean realised vol (calm→turbulent)** | HMM states are permutation-invariant; without pinning nothing reproduces |
| ♻️ | 3 states → **2 states (calm/turbulent)** | 3-state recovery ~46% and unstable → **2-state 91.6% stable** | Crash regime (~3%) too rare for a stable 3-state fit; documented resolution-for-robustness trade |
| ➕ | Measure, don't assert | detection accuracy now reported vs **ground-truth labels** (added to generator) | Turns a claim into a validated number |
| ➕ | Base = **static long** exposure | (new) | Isolates the overlay's contribution; a trend base self-protects and muddies the test |

Result: gating cuts maxDD −42%→−13%, vol ~halved, Sharpe 0.97→1.05.

---

## 7. Strategy 7 — Dispersion Vol Arbitrage (`07_dispersion_vol_arb/`)
| | Change | Before → After | Reason |
|---|---|---|---|
| 🐛 | **The demo guaranteed a profit** | realised corr = implied − positive draw → **realised corr spikes *above* implied ~12% of events** | Restores the crisis left tail that *is* dispersion's risk |
| ➕ | Walk-forward entry threshold | fixed → **implied-corr threshold chosen on past events only** | OOS discipline |
| 🐛 | Per-event bootstrap units | annualised by √252 → **`periods_per_year=1`** | Events aren't daily; annualising is meaningless |
| 📝 | Loss-clustering disclosed | (none) → **README notes correlation spikes are systemic → effective independent bets ≪ event count** | Honest tail accounting |

The bisection implied-correlation solver and variance-P&L accounting were sound and preserved. Result: per-event Sharpe 0.87, CI [0.69, 1.10].

---

## 8. Combined Multi-Strategy Portfolio (`combined_portfolio/`)
- ➕ New capstone blending the three daily-return sleeves (S1, S4, S5) with a
  correlation matrix, equal-weight and inverse-vol books.
- 📝 Honest framing: equal-weight (Sharpe 1.97, 8.3% vol) beats the *average*
  sleeve (0.80) and is robust even with a negative sleeve, but does **not** beat
  the best single sleeve; near-zero sleeve correlation is partly mechanical
  (independent synthetic universes) and would be stressed on real data.

---

## 9. Shared data layer additions (`quant_framework/data.py`)
- ➕ `toxic_flow_bars` — intraday bars where latent order-flow toxicity is
  *predictive* of forward volatility/adverse drift (so VPIN has real signal to
  detect); regime-dependent autocorrelation (reverts benign / trends toxic).
- ➕ `cross_sectional_reversion_panel` — factor returns plus explicit next-day
  cross-sectional rank reversion.
- ♻️ `factor_universe` — idiosyncratic reversion changed from 1-day to multi-day.
- ♻️ `regime_switching_series` — added `return_regimes` to emit ground-truth
  labels for measuring HMM recovery accuracy.
- ➕ `dispersion_vol_panel` — index + constituent implied/realised vols and
  correlation, with an explicit crisis tail (realised corr > implied).

---

## Reproducibility note
All changes preserve determinism (fixed seeds; HMM `random_state=42`).
`python run_all.py` regenerates every `metrics.json`, figure, and HTML report
from a clean state.
