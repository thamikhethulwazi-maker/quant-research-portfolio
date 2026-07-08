# Standalone Strategies (original single-file versions, audited & fixed)

These are the **original seven single-file strategies**, kept deliberately
simple — one self-contained, runnable file per strategy — but with every
critical bug found in the audit **fixed and clearly marked** with
`FIX (2026 audit):` comments so a reader can see exactly what changed and why.

Run any of them directly, e.g.:

```bash
python 01_kalman_pairs_trading.py
```

| File | Critical fix applied |
|---|---|
| `01_kalman_pairs_trading.py` | Backtest now actually uses the Kalman hedge ratio (was trading 1:1) + transaction costs, lagged beta |
| `02_earnings_iv_crush.py` | Removed fatal look-ahead (entry gated on post-earnings IV); faithful straddle repricing for the stock move |
| `03_vpin_order_flow_toxicity.py` | VPIN bucket size estimated on early sample only (was full-sample); size multiplier lagged |
| `04_pca_factor_neutral_longshort.py` | Long/short weight signs were inverted (buying what it meant to sell) — corrected |
| `05_cross_sectional_mean_reversion.py` | Same weight-sign inversion — corrected in all three weighting schemes |
| `06_hidden_markov_regime_detection.py` | Structurally sound; note added that 2 states are far more robust than 3 when crashes are rare |
| `07_dispersion_volatility_arb.py` | Demo guaranteed a profit (realised corr defined below implied) — now has a ~12% correlation-crisis tail; degenerate entry filter (premium ≈ 0 by construction) replaced with the meaningful implied-correlation filter |

**Where's the full validation?** These files are the *simple showcase*. The
walk-forward out-of-sample validation, bootstrap confidence intervals,
figures and honest per-strategy write-ups live in the main repository
(`strategies/` + `quant_framework/` + `outputs/`). If you read one thing per
strategy, read its README in `strategies/<name>/`.
