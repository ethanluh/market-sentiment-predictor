"""
The go/no-go rule for moving from historical evaluation to paper trading.

The thresholds are fixed here, in code under review, BEFORE any real result is seen, so the rule
cannot be tuned to the numbers. Every criterion must hold; a failing run reports each reason.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from src.eval.multiple import benjamini_hochberg, one_sided_p
from src.eval.panel import PanelTest

REFERENCE = "empirical_quantile"


@dataclass(frozen=True)
class GateRule:
    max_panel_p: float = 0.05  # one-sided, panel average across tickers
    max_fdr: float = 0.10  # Benjamini-Hochberg q-value for a ticker to count as a real edge
    min_tickers_passing: int = 3  # tickers with positive skill and q <= max_fdr
    coverage_band: tuple[float, float] = (0.70, 0.90)  # mean P10-P90 coverage, nominal 0.80
    ticker_coverage_band: tuple[float, float] = (0.60, 0.95)  # no single ticker may leave this
    min_predictions: int = 500  # a ticker with fewer scored rows is excluded, and reported

    def __post_init__(self) -> None:
        for name in ("max_panel_p", "max_fdr"):
            if not 0.0 < getattr(self, name) < 1.0:
                raise ValueError(f"{name} must be in (0, 1)")
        if self.min_tickers_passing < 1 or self.min_predictions < 1:
            raise ValueError("min_tickers_passing and min_predictions must be >= 1")
        for band in (self.coverage_band, self.ticker_coverage_band):
            if not 0.0 < band[0] < band[1] < 1.0:
                raise ValueError("coverage bands must satisfy 0 < low < high < 1")


@dataclass
class GateResult:
    passed: bool
    reasons: list[str] = field(default_factory=list)  # one line per failed criterion
    details: dict[str, object] = field(default_factory=dict)


def apply_gate(
    summaries: dict[str, dict], panel: PanelTest, rule: GateRule = GateRule()
) -> GateResult:
    """
    ``summaries`` maps ticker -> the ``summary`` dict that ``src.eval.runner.summarize`` produces.
    """
    reasons: list[str] = []
    usable = {t: s for t, s in summaries.items() if s["n_predictions"] >= rule.min_predictions}
    excluded = sorted(set(summaries) - set(usable))
    if len(usable) < max(rule.min_tickers_passing, math.ceil(len(summaries) / 2)):
        reasons.append(
            f"only {len(usable)} of {len(summaries)} tickers have >= {rule.min_predictions} scored "
            f"rows (excluded: {excluded})"
        )

    # 1. The panel average must beat the baseline, with one-sided significance.
    panel_p = one_sided_p(panel.dm_stat, panel.dm_p_two_sided)
    if not (panel.mean_relative_diff < 0 and panel_p <= rule.max_panel_p):
        reasons.append(
            f"panel: mean relative loss difference {panel.mean_relative_diff:+.4%} with one-sided "
            f"p = {panel_p:.3f} (need < 0 and p <= {rule.max_panel_p})"
        )

    # 2. Enough individual tickers must show a real edge after FDR control.
    tickers = sorted(usable)
    passing: list[str] = []
    q_values: dict[str, float] = {}
    if tickers:
        raw = [
            one_sided_p(
                usable[t]["model_vs"][REFERENCE]["dm_stat"],
                usable[t]["model_vs"][REFERENCE]["dm_p"],
            )
            for t in tickers
        ]
        qs = benjamini_hochberg(raw)
        q_values = dict(zip(tickers, (float(q) for q in qs)))
        passing = [
            t
            for t in tickers
            if usable[t]["skill_vs_empirical_quantile"] > 0 and q_values[t] <= rule.max_fdr
        ]
    if len(passing) < rule.min_tickers_passing:
        reasons.append(
            f"only {len(passing)} ticker(s) show a positive edge at FDR {rule.max_fdr} "
            f"(need {rule.min_tickers_passing}): {passing}"
        )

    # 3. The forecasts must be calibrated, not merely sharp.
    coverages = [usable[t]["forecasters"]["model"]["coverage_80"] for t in tickers]
    mean_cov = sum(coverages) / len(coverages) if coverages else float("nan")
    lo, hi = rule.coverage_band
    if not lo <= mean_cov <= hi:
        reasons.append(f"mean P10-P90 coverage {mean_cov:.3f} is outside [{lo}, {hi}]")

    tlo, thi = rule.ticker_coverage_band
    off = {t: c for t, c in zip(tickers, coverages) if not tlo <= c <= thi}
    if off:
        reasons.append(f"ticker P10-P90 coverage outside [{tlo}, {thi}]: {off}")

    return GateResult(
        passed=not reasons,
        reasons=reasons,
        details={
            "panel_one_sided_p": panel_p,
            "q_values": q_values,
            "passing_tickers": passing,
            "excluded_tickers": excluded,
            "mean_coverage": mean_cov,
        },
    )
