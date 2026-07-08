# Quantitative Research Portfolio

An institutional-quality, reproducible quant research repository. Each strategy
is implemented, **audited for the biases that make backtests lie**
(look-ahead, data leakage, survivorship, costless fills), validated with
walk-forward out-of-sample testing, and written up honestly — including when the
edge is thin or the risk is ugly.

> **Data disclosure (read this first).** This research environment has no access
> to market-data vendors. Rather than fabricate a live track record, every
> strategy runs on **reproducible synthetic data** whose statistical properties
> are engineered to match the stylised facts the strategy targets (cointegration,
> variance risk premium, factor structure, regime switching). Synthetic results
> validate **implementation correctness and the mechanics of the edge** — they
> are *not* evidence of live-market profitability. The data layer
> (`quant_framework/data.py`) exposes a `MarketDataProvider` interface so real
> data (yfinance / CCXT / vendor) drops in with **no changes to strategy code**.

---

## Research standards (enforced everywhere)
- **No look-ahead bias.** Signals and models use only information available at
  decision time. (Strategy 2 exists partly to demonstrate catching a fatal one.)
- **No data leakage.** Parameters and models are selected on training folds only.
- **Strict OOS validation.** Walk-forward (expanding + rolling); test folds are
  never used for selection and are concatenated into one honest track record.
- **Reproducible & deterministic.** All randomness flows through seeded
  `np.random.default_rng`. Re-running reproduces every number and figure.
- **Robustness over returns.** Point-estimate Sharpes get bootstrap CIs and
  Monte Carlo distributions before any conclusion is drawn.
- **Honest write-ups.** A poor or tail-heavy result is explained, not buried.

## Repository layout
```
quant_research_portfolio/
├── README.md                 ← this file (overview + roadmap)
├── PROGRESS.md               ← live progress tracker (resume here each session)
├── requirements.txt
├── quant_framework/          ← shared, strategy-agnostic infrastructure
│   ├── metrics.py            ← CAGR, Sharpe, Sortino, Calmar, DD, turnover, ...
│   ├── validation.py         ← expanding/rolling walk-forward, strict OOS
│   ├── robustness.py         ← bootstrap CIs, Monte Carlo, param sensitivity
│   ├── plotting.py           ← equity/DD, rolling, heatmaps, MC fan, bootstrap
│   └── data.py               ← reproducible synthetic generators + real-data API
├── strategies/
│   ├── 01_kalman_pairs/         {strategy.py, run.py, README.md}   ✅
│   ├── 02_earnings_iv_crush/    {strategy.py, run.py, README.md}   ✅
│   ├── 03_vpin_overlay/         {strategy.py, run.py, README.md}   ✅
│   ├── 04_pca_factor_neutral/   {strategy.py, run.py, README.md, notebook.py}  ✅
│   ├── 05_cross_sectional_mr/   {strategy.py, run.py, README.md}   ✅
│   ├── 06_hmm_regime/           {strategy.py, run.py, README.md}   ✅
│   ├── 07_dispersion_vol_arb/   {strategy.py, run.py, README.md}   ✅
│   └── combined_portfolio/      {build_portfolio.py, README.md}    ✅
├── standalone/                ← the original single-file strategies, audited & fixed
├── CHANGELOG.md              ← audit trail: every change vs. the originals
├── GITHUB_UPLOAD.md          ← step-by-step publishing guide
├── LICENSE  ·  .gitignore  ·  requirements.txt  ·  run_all.py
└── outputs/                   ← per strategy: figures + metrics.json + report.html
```

**Presentation helpers** (tooling, not research): `quant_framework/report.py`
generates a self-contained `report.html` per strategy (embedded figures, metric
cards, bootstrap-CI banner, rendered README); `04_.../notebook.py` is a
section-by-section walkthrough of the strongest strategy.

## The shared framework (build once, reuse everywhere)
Every strategy imports the same infrastructure, so validation is consistent and
duplicated logic lives in one place:

| Module | What it gives every strategy |
|---|---|
| `metrics` | One `performance_summary()` → CAGR, ann. return/vol, Sharpe, Sortino, Calmar, max DD, VaR/CVaR, exposure, turnover, win rate, profit factor, monthly/annual tables, rolling metrics |
| `validation` | `WalkForwardValidator` with expanding **and** rolling schemes; the optimiser only ever sees training data |
| `robustness` | Stationary-bootstrap Sharpe CIs, block-bootstrap Monte Carlo equity fans, 2-D parameter-sensitivity grids |
| `plotting` | Consistent, publication-quality charts saved to `outputs/` |
| `data` | Cointegrated pairs, earnings VRP panel, factor universe, regime series — all seeded; plus the real-data adapter interface |

## Strategy roadmap
| # | Strategy | Edge | Status |
|---|---|---|---|
| 1 | Kalman Pairs Trading | Dynamic-hedge-ratio spread reversion | ✅ Complete |
| 2 | Earnings IV Crush | Variance risk premium (short vol) | ✅ Complete |
| 3 | VPIN Order-Flow Toxicity | Liquidity-stress timing overlay | ✅ Complete |
| 4 | PCA Factor-Neutral L/S | Idiosyncratic residual reversion | ✅ Complete |
| 5 | Cross-Sectional Mean Reversion | Relative-extreme reversion | ✅ Complete |
| 6 | HMM Regime Detection | Regime-adaptive allocation | ✅ Complete |
| 7 | Dispersion Vol Arbitrage | Index-vs-constituent correlation premium | ✅ Complete |
| — | Combined Multi-Strategy Portfolio | Diversification across uncorrelated sleeves | ✅ Complete |

See `PROGRESS.md` for the detailed per-strategy audit findings and the
session-by-session plan, and `CHANGELOG.md` for every change made vs. the
original files.

## Headline OOS results (completed strategies)
All figures are **out-of-sample**, net of costs, from concatenated walk-forward
folds. Bootstrap 95% CIs in brackets. Read each strategy's README for the full
honest assessment — several are deliberately marginal, which is the point.

| # | Strategy | OOS Sharpe | Bootstrap CI | Honest one-line verdict |
|---|---|---|---|---|
| 1 | Kalman Pairs | 0.37–0.41 | [−0.60, 1.19] | Real but marginal; CI includes zero |
| 2 | Earnings IV Crush | 0.90 (per-event) | [0.40, 2.17] | Genuine VRP edge, tail-dominated (−15% worst event) |
| 3 | VPIN Overlay | **Δ +0.48** vs base | — | Risk overlay: improves risk-adj return, cuts DD; absolute level an intraday-annualisation artifact |
| 4 | PCA Factor-Neutral L/S | **1.71** | **[0.78, 2.65]** | Strongest & cleanest; genuinely factor-neutral |
| 5 | Cross-Sectional MR | 0.52 (net) | [−0.38, 1.45] | Real gross signal, but cost-bound; CI includes zero |
| 6 | HMM Regime Detection | detection **91.6%** acc | — | Overlay: cuts maxDD −42%→−13%, Sharpe 0.97→1.05; value is risk reduction |
| 7 | Dispersion Vol Arb | 0.87 (per-event) | [0.69, 1.10] | Real correlation-premium edge; crisis tail, losses cluster (systemic) |
| — | Combined Portfolio | **1.97** (EW) | — | 3 uncorrelated sleeves; beats avg sleeve (0.80), vol 8.3%, maxDD −8% |

**A note on what "good" means here.** Several strategies have confidence
intervals that touch or cross zero, and the strongest single number (the
portfolio's 1.97) rests partly on synthetic independence. That is not a failure
of the research — it is the research working. Marginal-but-honest beats
impressive-but-overfit, and every write-up says plainly where the edge is thin,
where the tail is ugly, where costs decide the outcome, and where the synthetic
data flatters the result.

## Reproduce everything
```bash
pip install -r requirements.txt
python run_all.py          # every strategy + portfolio + HTML reports
# or run one at a time:
python strategies/04_pca_factor_neutral/run.py
```
Each run prints its OOS metrics and writes figures + `metrics.json` (+ a
self-contained `report.html`) to `outputs/<strategy>/`. Everything is
deterministic (fixed seeds).

## The single most important caveat
All results run on **reproducible synthetic data**, because this environment has
no live-market access. The synthetic generators are transparent, seeded
parametric processes designed to contain the specific effect each strategy
targets — so the backtests validate **implementation correctness and edge
mechanics**, not live-market viability. The `data.py` layer exposes a real-data
adapter so a yfinance/CCXT feed drops in with no change to strategy code; porting
one strategy to real data is the clear next step and is called out in each README.
