"""
run_all.py — regenerate every completed strategy's outputs in one command.

Usage:  python run_all.py
Deterministic: each strategy is seeded and self-contained. Re-running
reproduces all metrics and figures under outputs/.
"""
import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
COMPLETED = [
    ("Strategy 1 — Kalman Pairs", "strategies/01_kalman_pairs/run.py"),
    ("Strategy 2 — Earnings IV Crush", "strategies/02_earnings_iv_crush/run.py"),
    ("Strategy 3 — VPIN Overlay", "strategies/03_vpin_overlay/run.py"),
    ("Strategy 4 — PCA Factor-Neutral L/S", "strategies/04_pca_factor_neutral/run.py"),
    ("Strategy 5 — Cross-Sectional Mean Reversion", "strategies/05_cross_sectional_mr/run.py"),
    ("Strategy 6 — HMM Regime Detection", "strategies/06_hmm_regime/run.py"),
    ("Strategy 7 — Dispersion Vol Arbitrage", "strategies/07_dispersion_vol_arb/run.py"),
    ("Combined Multi-Strategy Portfolio", "strategies/combined_portfolio/build_portfolio.py"),
]


def _run(path):
    spec = importlib.util.spec_from_file_location("run_mod", os.path.join(ROOT, path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_mod"] = mod
    spec.loader.exec_module(mod)
    mod.main()


if __name__ == "__main__":
    for name, path in COMPLETED:
        print("\n" + "#" * 70 + f"\n# {name}\n" + "#" * 70)
        _run(path)
    # Rebuild the self-contained HTML reports from the fresh outputs.
    print("\n" + "#" * 70 + "\n# Generating per-strategy HTML reports\n" + "#" * 70)
    from quant_framework.report import generate_report
    for sdir in sorted(os.listdir(os.path.join(ROOT, "strategies"))):
        sp = os.path.join(ROOT, "strategies", sdir)
        op = os.path.join(ROOT, "outputs", sdir)
        if os.path.isdir(sp) and os.path.exists(op):
            print("report:", generate_report(sp, op))
    print("\nAll strategies + portfolio regenerated. See outputs/.")
