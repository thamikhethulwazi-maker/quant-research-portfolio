"""
quant_framework/report.py — assemble a standalone, portable HTML report per
strategy from its metrics.json, figures, and README.

Design goals
------------
* Self-contained: images are base64-embedded, so the single .html file can be
  emailed, committed, or opened anywhere with no external assets.
* Printable: open in a browser and "Print to PDF" for a clean report deliverable.
* Reusable: one function drives every strategy — no per-strategy code.

Usage
-----
    from quant_framework.report import generate_report
    generate_report("strategies/04_pca_factor_neutral",
                    "outputs/04_pca_factor_neutral")
    # -> outputs/04_pca_factor_neutral/report.html
"""
from __future__ import annotations

import base64
import json
import os
from typing import Optional

import markdown as _md

# House style — matches the plotting module (navy / red / grey).
_CSS = """
:root { --navy:#1f3b57; --red:#b03a2e; --grey:#7f8c8d; --ink:#22303c; }
* { box-sizing: border-box; }
body { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
       color: var(--ink); max-width: 980px; margin: 0 auto; padding: 40px 28px;
       line-height: 1.55; }
h1 { color: var(--navy); border-bottom: 3px solid var(--navy); padding-bottom: 8px; }
h2 { color: var(--navy); margin-top: 34px; border-bottom: 1px solid #d9e0e6;
     padding-bottom: 4px; }
h3 { color: var(--red); margin-top: 22px; }
code, pre { background: #f4f6f8; border-radius: 4px; }
pre { padding: 12px 14px; overflow-x: auto; font-size: 13px; }
table { border-collapse: collapse; width: 100%; margin: 14px 0; font-size: 14px; }
th, td { border: 1px solid #d9e0e6; padding: 7px 10px; text-align: left; }
th { background: var(--navy); color: white; }
tr:nth-child(even) td { background: #f7f9fa; }
.metric-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px;
               margin: 18px 0; }
.metric-card { background: #f7f9fa; border: 1px solid #d9e0e6; border-radius: 8px;
               padding: 12px 14px; }
.metric-card .label { font-size: 11px; text-transform: uppercase; letter-spacing:.4px;
                      color: var(--grey); }
.metric-card .value { font-size: 22px; font-weight: 600; color: var(--navy);
                      margin-top: 2px; }
figure { margin: 18px 0; text-align: center; }
figure img { max-width: 100%; border: 1px solid #e2e8ee; border-radius: 6px; }
figcaption { font-size: 12px; color: var(--grey); margin-top: 6px; }
.banner { background: #fff8e1; border-left: 4px solid #f0ad4e; padding: 10px 14px;
          border-radius: 4px; font-size: 13px; margin: 16px 0; }
.footer { margin-top: 40px; padding-top: 14px; border-top: 1px solid #d9e0e6;
          color: var(--grey); font-size: 12px; }
"""

# Headline metrics to surface as cards (label -> key in metrics summary).
_HEADLINE = [
    ("Sharpe", "sharpe"), ("Sortino", "sortino"), ("CAGR", "cagr"),
    ("Ann. Vol", "ann_vol"), ("Max DD", "max_drawdown"), ("Calmar", "calmar"),
    ("Win Rate", "win_rate"), ("Ann. Turnover", "ann_turnover"),
]
_PCT_KEYS = {"cagr", "ann_return", "ann_vol", "max_drawdown", "win_rate",
             "ann_turnover", "avg_gross_exposure", "var_95", "cvar_95"}

# Human-friendly captions per known figure filename.
_CAPTIONS = {
    "equity_drawdown.png": "Equity curve and drawdown (out-of-sample).",
    "rolling_metrics.png": "Rolling Sharpe and rolling volatility.",
    "monthly_returns.png": "Monthly return heatmap.",
    "annual_returns.png": "Annual returns.",
    "param_sensitivity.png": "Parameter-sensitivity heatmap.",
    "monte_carlo.png": "Monte Carlo equity-path fan (block bootstrap).",
    "bootstrap_sharpe.png": "Bootstrap distribution of the Sharpe ratio.",
    "cost_sensitivity.png": "Sharpe vs transaction cost and rebalance frequency.",
    "sharpe_by_vpin_quintile.png": "Base-strategy Sharpe by VPIN quintile.",
    "vpin_vs_toxicity.png": "VPIN vs latent order-flow toxicity.",
}
# Preferred display order.
_ORDER = ["equity_drawdown.png", "rolling_metrics.png", "cost_sensitivity.png",
          "sharpe_by_vpin_quintile.png", "vpin_vs_toxicity.png",
          "monthly_returns.png", "annual_returns.png", "param_sensitivity.png",
          "monte_carlo.png", "bootstrap_sharpe.png"]


def _fmt(key: str, val) -> str:
    if not isinstance(val, (int, float)):
        return str(val)
    if key in _PCT_KEYS:
        return f"{val:.2%}"
    return f"{val:.2f}"


def _img_tag(path: str) -> str:
    with open(path, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode()
    return f'data:image/png;base64,{b64}'


def _find_summary(metrics: dict) -> dict:
    """Locate the primary performance-summary dict inside metrics.json.
    Handles the different shapes across strategies (oos / oos_overlay / etc.)."""
    for key in ("oos", "oos_overlay", "summary"):
        if isinstance(metrics.get(key), dict) and "sharpe" in metrics[key]:
            return metrics[key]
    # else: first nested dict that looks like a summary
    for v in metrics.values():
        if isinstance(v, dict) and "sharpe" in v:
            return v
    return {}


def generate_report(strategy_dir: str, output_dir: str,
                    title: Optional[str] = None) -> str:
    """Build outputs/<strategy>/report.html. Returns the path written."""
    metrics_path = os.path.join(output_dir, "metrics.json")
    metrics = json.load(open(metrics_path)) if os.path.exists(metrics_path) else {}
    summary = _find_summary(metrics)

    readme_path = os.path.join(strategy_dir, "README.md")
    readme_html = ""
    if os.path.exists(readme_path):
        readme_html = _md.markdown(open(readme_path).read(),
                                   extensions=["tables", "fenced_code"])

    name = title or os.path.basename(strategy_dir.rstrip("/"))

    # Headline metric cards
    cards = ""
    for label, key in _HEADLINE:
        if key in summary:
            cards += (f'<div class="metric-card"><div class="label">{label}</div>'
                      f'<div class="value">{_fmt(key, summary[key])}</div></div>')
    cards_block = f'<div class="metric-grid">{cards}</div>' if cards else ""

    # Bootstrap CI banner, if present
    boot = metrics.get("bootstrap_sharpe", {})
    ci_block = ""
    if boot:
        ci_block = (f'<div class="banner"><b>Bootstrap Sharpe:</b> '
                    f'{boot.get("point", float("nan")):.2f} '
                    f'(95% CI [{boot.get("lower", float("nan")):.2f}, '
                    f'{boot.get("upper", float("nan")):.2f}], '
                    f'P[Sharpe&gt;0] = {boot.get("prob_positive", float("nan")):.0%})</div>')

    # Full metrics table
    rows = "".join(f"<tr><td>{k.replace('_',' ').title()}</td><td>{_fmt(k, v)}</td></tr>"
                   for k, v in summary.items())
    metrics_table = f"<h2>Performance summary (out-of-sample)</h2><table>{rows}</table>" if rows else ""

    # Figures
    figs = ""
    present = [f for f in os.listdir(output_dir) if f.endswith(".png")]
    ordered = [f for f in _ORDER if f in present] + [f for f in present if f not in _ORDER]
    for f in ordered:
        cap = _CAPTIONS.get(f, f.replace("_", " ").replace(".png", "").title())
        figs += (f'<figure><img src="{_img_tag(os.path.join(output_dir, f))}"/>'
                 f'<figcaption>{cap}</figcaption></figure>')
    figs_block = f"<h2>Figures</h2>{figs}" if figs else ""

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>{name} — research report</title><style>{_CSS}</style></head><body>
<h1>{name.replace('_', ' ').title()}</h1>
<p><i>Institutional quant research portfolio — out-of-sample results on
reproducible synthetic data. Synthetic results validate implementation
correctness and edge mechanics, not live-market profitability.</i></p>
{cards_block}
{ci_block}
{metrics_table}
{figs_block}
<h2>Full write-up</h2>
{readme_html}
<div class="footer">Generated by quant_framework.report · deterministic ·
data source: {metrics.get('data', 'synthetic (seeded)')}</div>
</body></html>"""

    out_path = os.path.join(output_dir, "report.html")
    with open(out_path, "w") as fh:
        fh.write(html)
    return out_path


if __name__ == "__main__":
    import sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pairs = [
        ("01_kalman_pairs", "01_kalman_pairs"),
        ("02_earnings_iv_crush", "02_earnings_iv_crush"),
        ("03_vpin_overlay", "03_vpin_overlay"),
        ("04_pca_factor_neutral", "04_pca_factor_neutral"),
        ("05_cross_sectional_mr", "05_cross_sectional_mr"),
    ]
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for sdir, odir in pairs:
        if only and only not in sdir:
            continue
        sp = os.path.join(root, "strategies", sdir)
        op = os.path.join(root, "outputs", odir)
        if os.path.exists(op):
            print("wrote", generate_report(sp, op))
