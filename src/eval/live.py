"""
Real-data evaluation: per-ticker walk-forward runs, ablations, panel tests and the go/no-go gate.

Every ticker's features come from ``prediction.dataset.build_dataset``, the same path training and
live forecasting use, so the numbers here describe what a deployed model would have done.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.eval.ablation import ablation_configs, neutralize
from src.eval.gate import REFERENCE, GateResult, GateRule, apply_gate
from src.eval.metrics import mean_pinball_per_row
from src.eval.multiple import benjamini_hochberg, one_sided_p
from src.eval.panel import PanelTest, panel_test
from src.eval.runner import EvalResult, evaluate
from src.eval.significance import diebold_mariano
from src.prediction.dataset import DatasetConfig, Sources, build_dataset


@dataclass(frozen=True)
class LiveEvalConfig:
    universe: tuple[str, ...]
    start: str = "2019-01-01"  # prices start a year before the news archive so features warm up
    end: str | None = None
    horizon: str = "5d"
    configs: tuple[str, ...] = ("price_only", "full")
    train_window: int = 252
    test_window: int = 21
    backend: str = "gbr"
    use_news_archive: bool = True
    use_trends: bool = False  # off by default: a rate-limited source whose stamping is conservative
    use_insider_flow: bool = False
    lookback_days: int = 7
    cost_bps: float = 1.0
    n_boot: int = 500
    seed: int = 0

    def __post_init__(self) -> None:
        if len(self.universe) < 2:
            raise ValueError(
                "a live evaluation needs at least two tickers (peers come from the universe)"
            )
        known = ablation_configs()
        unknown = [c for c in self.configs if c not in known]
        if unknown:
            raise ValueError(f"unknown ablation configs {unknown}; choose from {sorted(known)}")
        if "full" not in self.configs:
            raise ValueError("configs must include 'full'")


@dataclass
class LiveEvalReport:
    config: LiveEvalConfig
    data: dict[str, dict[str, object]]
    results: dict[str, dict[str, EvalResult]]  # config name -> ticker -> result
    panels: dict[str, PanelTest]
    thesis: dict[str, object]  # does the full model beat the price-only model?
    gate: GateResult

    def to_dict(self) -> dict[str, object]:
        return {
            "config": {
                k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(self.config).items()
            },
            "data": self.data,
            "summaries": {
                c: {t: r.summary for t, r in rs.items()} for c, rs in self.results.items()
            },
            "panels": {name: vars(p) for name, p in self.panels.items()},
            "thesis_full_vs_price_only": self.thesis,
            "gate": {
                "passed": self.gate.passed,
                "reasons": self.gate.reasons,
                "details": self.gate.details,
            },
        }


def with_baseline(result: EvalResult, name: str, other: EvalResult) -> EvalResult:
    """``result`` with ``other``'s model forecasts added as the baseline ``name`` (same rows only)."""
    if not result.index.equals(other.index) or not np.array_equal(result.y, other.y):
        raise ValueError("cannot compare models evaluated on different rows")
    return EvalResult(
        result.horizon,
        result.steps,
        result.n_folds,
        result.index,
        result.y,
        {**result.preds, name: other.preds["model"]},
        result.summary,
    )


def _thesis(
    full: dict[str, EvalResult], price_only: dict[str, EvalResult], max_fdr: float
) -> dict[str, object]:
    """Per ticker and pooled: does the full model (news, graph, regime) beat price features alone?"""
    tickers = sorted(full)
    stats: dict[str, dict[str, float]] = {}
    pvals = []
    for t in tickers:
        a, b = full[t], price_only[t]
        loss_a, loss_b = mean_pinball_per_row(a.y, a.preds["model"]), mean_pinball_per_row(
            b.y, b.preds["model"]
        )
        stat, p = diebold_mariano(loss_a, loss_b, h=a.steps)
        stats[t] = {
            "loss_diff": float((loss_a - loss_b).mean()),
            "dm_stat": stat,
            "p_one_sided": one_sided_p(stat, p),
        }
        pvals.append(stats[t]["p_one_sided"])
    for t, q in zip(tickers, benjamini_hochberg(pvals)):
        stats[t]["q_value"] = float(q)
    paired = {t: with_baseline(full[t], "price_only", price_only[t]) for t in tickers}
    pooled = panel_test(paired, "price_only")
    return {
        "per_ticker": stats,
        "panel": vars(pooled),
        "tickers_with_a_real_edge": [
            t for t in tickers if stats[t]["loss_diff"] < 0 and stats[t]["q_value"] <= max_fdr
        ],
    }


def run_live_eval(
    cfg: LiveEvalConfig,
    sources: Sources | None = None,
    rule: GateRule = GateRule(),
    progress: Callable[[str], None] = print,
) -> LiveEvalReport:
    names = ablation_configs()
    results: dict[str, dict[str, EvalResult]] = {c: {} for c in cfg.configs}
    data: dict[str, dict[str, object]] = {}
    for i, ticker in enumerate(cfg.universe, start=1):
        peers = tuple(t for t in cfg.universe if t != ticker)
        ds = build_dataset(
            DatasetConfig(
                ticker=ticker,
                start=cfg.start,
                end=cfg.end,
                peers=peers,
                use_vix=True,
                use_trends=cfg.use_trends,
                use_insider_flow=cfg.use_insider_flow,
                use_news_archive=cfg.use_news_archive,
                lookback_days=cfg.lookback_days,
            ),
            sources,
            regime="none",
        )
        data[ticker] = {
            "rows": len(ds.features),
            "first": f"{ds.features.index[0]:%Y-%m-%d}",
            "last": f"{ds.features.index[-1]:%Y-%m-%d}",
            "news_coverage": ds.news_coverage,
        }
        for name in cfg.configs:
            dropped = names[name]
            results[name][ticker] = evaluate(
                neutralize(ds.features, dropped),
                ds.log_returns,
                cfg.horizon,
                vix=None if "regime" in dropped else ds.vix,
                train_window=cfg.train_window,
                test_window=cfg.test_window,
                backend=cfg.backend,
                cost_bps=cfg.cost_bps,
                n_boot=cfg.n_boot,
                seed=cfg.seed,
            )
        progress(
            f"[{i}/{len(cfg.universe)}] {ticker}: {len(ds.features)} rows, news coverage {ds.news_coverage}"
        )

    # The panel uses the same tickers the gate keeps: a short-history name must not count in one
    # place and be excluded in another.
    usable = [
        t
        for t, r in results["full"].items()
        if _dig(r.summary, "n_predictions") >= rule.min_predictions
    ]
    if not usable:
        rows = {t: _dig(r.summary, "n_predictions") for t, r in results["full"].items()}
        raise RuntimeError(
            f"no ticker has >= {rule.min_predictions} scored rows; nothing can be evaluated (rows: {rows})"
        )
    full = {t: results["full"][t] for t in usable}
    panels = {f"full_vs_{REFERENCE}": panel_test(full, REFERENCE)}
    thesis = (
        _thesis(full, {t: results["price_only"][t] for t in usable}, rule.max_fdr)
        if "price_only" in results
        else {}
    )
    gate = apply_gate(
        {t: r.summary for t, r in results["full"].items()}, panels[f"full_vs_{REFERENCE}"], rule
    )
    return LiveEvalReport(cfg, data, results, panels, thesis, gate)


def _clean(obj: object) -> object:
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return _clean(obj.item())
    return obj


def save_report(report: LiveEvalReport, path: Path, snapshot: str | None) -> None:
    doc = report.to_dict()
    doc["price_snapshot"] = snapshot
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(doc), indent=2) + "\n")


def _dig(obj: Any, *keys: str) -> Any:
    for k in keys:
        obj = obj[k]
    return obj


def render_report_markdown(report: LiveEvalReport) -> str:
    """The results table for ``docs/results.md``."""
    full = report.results["full"]
    q: dict[str, float] = _dig(report.gate.details, "q_values")
    lines = [
        f"Gate: **{'PASS' if report.gate.passed else 'FAIL'}**",
        "",
        *(f"- {r}" for r in report.gate.reasons),
        "",
        "| Ticker | Rows | News coverage | Skill vs empirical quantiles | DM p (one-sided) | q (BH) | P10-P90 coverage |",
        "|---|---|---|---|---|---|---|",
    ]
    for t in sorted(full):
        s = full[t].summary
        mv = _dig(s, "model_vs", REFERENCE)
        cov = _dig(s, "forecasters", "model", "coverage_80")
        nc = report.data[t]["news_coverage"]
        coverage_text = "" if nc is None else f"{nc:.0%}"
        q_text = f"{q[t]:.3f}" if t in q else "excluded"  # too few scored rows for the gate
        lines.append(
            f"| {t} | {report.data[t]['rows']} | {coverage_text} | {s['skill_vs_empirical_quantile']:+.2%} | "
            f"{one_sided_p(mv['dm_stat'], mv['dm_p']):.3f} | {q_text} | {cov:.3f} |"
        )
    p = report.panels[f"full_vs_{REFERENCE}"]
    lines += [
        "",
        f"Panel (average across tickers per date): mean relative loss difference {p.mean_relative_diff:+.3%}, "
        f"DM statistic {p.dm_stat:+.2f}, two-sided p {p.dm_p_two_sided:.3f}, {p.n_dates} dates, {p.n_tickers} tickers.",
    ]
    if report.thesis:
        tp = _dig(report.thesis, "panel")
        lines += [
            "",
            "Does sentiment/graph/regime add anything over price features alone (full vs price-only)? "
            f"Panel mean relative loss difference {tp['mean_relative_diff']:+.3%}, "
            f"two-sided p {tp['dm_p_two_sided']:.3f}; tickers with a real edge: "
            f"{report.thesis['tickers_with_a_real_edge'] or 'none'}.",
        ]
    return "\n".join(lines)
