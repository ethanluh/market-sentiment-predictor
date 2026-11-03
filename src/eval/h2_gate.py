"""
The pre-registered pass criteria for H2 (``docs/protocol_h2.md``, "The holdout test").

The thresholds are fixed here, in reviewed code, before any development run. H2a: the overall
winner beats both volatility baselines. H2b: the best news pipeline beats the best price pipeline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.eval.gate import GateResult
from src.eval.live import with_baseline
from src.eval.multiple import benjamini_hochberg, one_sided_p
from src.eval.panel import PanelTest, panel_test
from src.eval.runner import EvalResult


@dataclass(frozen=True)
class H2GateRule:
    references: tuple[str, ...] = ("garch", "fhs")
    max_panel_p: float = 0.05
    max_fdr: float = 0.10
    min_tickers_passing: int = 3
    coverage_band: tuple[float, float] = (0.70, 0.90)
    ticker_coverage_band: tuple[float, float] = (0.60, 0.95)
    min_predictions: int = 500
    min_usable_fraction: float = 0.75

    def __post_init__(self) -> None:
        if not self.references:
            raise ValueError("at least one reference baseline is required")
        for name in ("max_panel_p", "max_fdr", "min_usable_fraction"):
            if not 0.0 < getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")
        if self.min_tickers_passing < 1 or self.min_predictions < 1:
            raise ValueError("min_tickers_passing and min_predictions must be >= 1")


def _panel_ok(panel: PanelTest, rule: H2GateRule) -> tuple[bool, float]:
    p = one_sided_p(panel.dm_stat, panel.dm_p_two_sided)
    return panel.mean_relative_diff < 0 and p <= rule.max_panel_p, p


def apply_h2_gate(results: dict[str, EvalResult], rule: H2GateRule = H2GateRule()) -> GateResult:
    """
    H2a on per-ticker results of one candidate. Tickers with fewer than ``min_predictions`` scored
    rows leave every criterion, the panel included; more than ``1 - min_usable_fraction`` of them
    excluded fails the gate outright.
    """
    if not results:
        raise ValueError("no results to gate")
    usable = {t: r for t, r in results.items() if len(r.y) >= rule.min_predictions}
    excluded = sorted(set(results) - set(usable))
    reasons: list[str] = []
    details: dict[str, object] = {"excluded_tickers": excluded}
    if len(usable) < math.ceil(rule.min_usable_fraction * len(results)):
        reasons.append(
            f"only {len(usable)} of {len(results)} tickers have >= {rule.min_predictions} scored "
            f"rows (need {rule.min_usable_fraction:.0%}; excluded: {excluded})"
        )
        return GateResult(False, reasons, details)

    tickers = sorted(usable)
    edges: dict[str, set[str]] = {}
    for ref in rule.references:
        panel = panel_test(usable, ref)
        ok, p = _panel_ok(panel, rule)
        details[f"panel_vs_{ref}"] = {
            "mean_relative_diff": panel.mean_relative_diff,
            "one_sided_p": p,
        }
        if not ok:
            reasons.append(
                f"panel vs {ref}: mean relative loss difference {panel.mean_relative_diff:+.4%}, "
                f"one-sided p = {p:.3f} (need < 0 and p <= {rule.max_panel_p})"
            )
        raw = [
            one_sided_p(
                usable[t].summary["model_vs"][ref]["dm_stat"],  # type: ignore[index]
                usable[t].summary["model_vs"][ref]["dm_p"],  # type: ignore[index]
            )
            for t in tickers
        ]
        q = dict(zip(tickers, (float(x) for x in benjamini_hochberg(raw))))
        edges[ref] = {
            t
            for t in tickers
            if usable[t].summary["model_vs"][ref]["loss_diff"] < 0  # type: ignore[index]
            and q[t] <= rule.max_fdr
        }
        details[f"q_values_vs_{ref}"] = q
    passing = sorted(set.intersection(*edges.values()))
    details["passing_tickers"] = passing
    if len(passing) < rule.min_tickers_passing:
        reasons.append(
            f"only {len(passing)} ticker(s) have a positive edge at FDR {rule.max_fdr} against every "
            f"reference {list(rule.references)} (need {rule.min_tickers_passing}): {passing}"
        )

    coverages = {t: float(usable[t].summary["forecasters"]["model"]["coverage_80"]) for t in tickers}  # type: ignore[index]
    mean_cov = sum(coverages.values()) / len(coverages)
    details["mean_coverage"] = mean_cov
    lo, hi = rule.coverage_band
    if not lo <= mean_cov <= hi:
        reasons.append(f"mean P10-P90 coverage {mean_cov:.3f} is outside [{lo}, {hi}]")
    tlo, thi = rule.ticker_coverage_band
    off = {t: c for t, c in coverages.items() if not tlo <= c <= thi}
    if off:
        reasons.append(f"ticker P10-P90 coverage outside [{tlo}, {thi}]: {off}")
    return GateResult(not reasons, reasons, details)


def compare_pipelines(
    news: dict[str, EvalResult], price: dict[str, EvalResult], rule: H2GateRule = H2GateRule()
) -> GateResult:
    """H2b: panel test of the frozen best-news pipeline against the frozen best-price pipeline."""
    if set(news) != set(price):
        raise ValueError("the two pipelines must cover the same tickers")
    usable = [t for t in sorted(news) if len(news[t].y) >= rule.min_predictions]
    if len(usable) < math.ceil(rule.min_usable_fraction * len(news)):
        return GateResult(False, [f"only {len(usable)} of {len(news)} tickers are usable"], {})
    paired = {t: with_baseline(news[t], "price", price[t]) for t in usable}
    panel = panel_test(paired, "price")
    ok, p = _panel_ok(panel, rule)
    excluded = sorted(set(news) - set(usable))
    reasons = (
        []
        if ok
        else [
            f"news vs price: mean relative loss difference {panel.mean_relative_diff:+.4%}, "
            f"one-sided p = {p:.3f} (need < 0 and p <= {rule.max_panel_p})"
        ]
    )
    return GateResult(
        ok,
        reasons,
        {
            "mean_relative_diff": panel.mean_relative_diff,
            "one_sided_p": p,
            "n_dates": panel.n_dates,
            "excluded_tickers": excluded,
        },
    )
