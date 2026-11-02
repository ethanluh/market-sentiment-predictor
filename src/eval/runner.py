"""
Leakage-controlled evaluation of the quantile model against per-row baselines.

``evaluate`` runs a purged walk-forward over a feature matrix and keeps every
per-row prediction, so metrics, significance tests and bootstrap intervals are all
computed from the same arrays. The regime classifier is refit on each fold's
training rows only, and baselines are updated every test row.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.baselines import make_baselines
from src.eval.metrics import (
    QUANTILES,
    christoffersen_independence,
    interval_coverage,
    kupiec_pof,
    mean_pinball_per_row,
    pinball_per_row,
    pinball_skill_score,
    quantile_hit_rates,
    weighted_interval_score_per_row,
)
from src.eval.significance import block_bootstrap_ci, diebold_mariano
from src.eval.splits import purged_walk_forward
from src.eval.targets import forward_log_return
from src.eval.trading import ANNUALISATION, annualized_sharpe, buy_and_hold_pnl, trade_pnl
from src.prediction.baselines import horizon_to_steps
from src.prediction.model import QuantileReturnModel
from src.prediction.regime import RegimeClassifier

REFERENCE = "empirical_quantile"


@dataclass
class EvalResult:
    horizon: str
    steps: int
    n_folds: int
    index: pd.DatetimeIndex
    y: np.ndarray
    preds: dict[str, np.ndarray]  # forecaster -> (n, 3) columns P10, P50, P90
    summary: dict[str, object] = field(default_factory=dict)


def evaluate(
    features: pd.DataFrame,
    log_returns: pd.Series,
    horizon: str,
    *,
    vix: pd.Series | None = None,
    train_window: int = 252,
    test_window: int = 21,
    backend: str = "gbr",
    include_arima: bool = False,
    cost_bps: float = 1.0,
    n_boot: int = 500,
    seed: int = 0,
) -> EvalResult:
    """
    Purged walk-forward evaluation for one horizon.

    Folds advance by ``test_window``, so test rows are contiguous and never repeat;
    the DM test, bootstrap, Kupiec and trading all assume that.

    ``features`` must be NaN-free with consecutive bars from ``log_returns.index``.
    When ``vix`` is given, ``regime_label`` is overwritten per fold from a classifier
    fit on that fold's training rows only.
    """
    steps = horizon_to_steps(horizon)
    if features.isna().any().any():
        raise ValueError("features contain NaN; drop incomplete rows before evaluating")
    positions = log_returns.index.get_indexer(features.index)
    if (positions < 0).any():
        raise ValueError("features index must be a subset of log_returns index")
    if not (np.diff(positions) == 1).all():
        raise ValueError("feature rows must be consecutive bars")

    target = forward_log_return(log_returns, steps)
    y = target.reindex(features.index).to_numpy()
    folds = purged_walk_forward(
        len(features), train_window, test_window, step=test_window, embargo=steps
    )
    baselines = make_baselines(log_returns, target, horizon, steps, train_window, include_arima)

    vix_rows: np.ndarray | None = None
    if vix is not None:
        vix_rows = vix.sort_index().reindex(features.index, method="ffill").to_numpy()
        if np.isnan(vix_rows).any():
            raise ValueError("vix has no value at or before some feature rows")

    names = ["model", *baselines]
    chunks: dict[str, list[np.ndarray]] = {name: [] for name in names}
    y_out: list[np.ndarray] = []
    idx_out: list[pd.DatetimeIndex] = []

    for fold in folds:
        x_tr, x_te = features.iloc[fold.train].copy(), features.iloc[fold.test].copy()
        if vix_rows is not None:
            clf = RegimeClassifier().fit(vix_rows[fold.train])
            x_tr["regime_label"] = clf.predict(vix_rows[fold.train]).astype(float)
            x_te["regime_label"] = clf.predict(vix_rows[fold.test]).astype(float)

        model = QuantileReturnModel(horizons=(horizon,), backend=backend)
        model.fit(x_tr, {horizon: pd.Series(y[fold.train], index=x_tr.index)})
        fold_preds = model.predict(x_te)
        if horizon not in fold_preds:
            raise RuntimeError(
                f"model produced no {horizon!r} forecasts for fold at {x_te.index[0]}"
            )

        keep = ~np.isnan(y[fold.test])  # the last ``steps`` rows have no realised target
        chunks["model"].append(np.array([[p.p10, p.p50, p.p90] for p in fold_preds[horizon]])[keep])
        for name, forecast in baselines.items():
            rows = [forecast(int(positions[i])) for i in fold.test[keep]]
            chunks[name].append(np.array(rows))
        y_out.append(y[fold.test][keep])
        idx_out.append(x_te.index[keep])

    result = EvalResult(
        horizon=horizon,
        steps=steps,
        n_folds=len(folds),
        index=idx_out[0].append(idx_out[1:]) if len(idx_out) > 1 else idx_out[0],
        y=np.concatenate(y_out),
        preds={name: np.concatenate(parts) for name, parts in chunks.items()},
    )
    result.summary = summarize(result, cost_bps=cost_bps, n_boot=n_boot, seed=seed)
    return result


def _forecaster_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, object]:
    low_viol, high_viol = y < p[:, 0], y > p[:, 2]
    return {
        "mean_pinball": float(mean_pinball_per_row(y, p).mean()),
        "pinball": {
            f"p{int(q * 100)}": float(pinball_per_row(y, p[:, k], q).mean())
            for k, q in enumerate(QUANTILES)
        },
        "wis": float(weighted_interval_score_per_row(y, p).mean()),
        "coverage_80": interval_coverage(y, p[:, 0], p[:, 2]),
        "quantile_hit_rates": quantile_hit_rates(y, p),
        "kupiec_p": {
            "p10": kupiec_pof(low_viol, QUANTILES[0]),
            "p90": kupiec_pof(high_viol, 1.0 - QUANTILES[2]),
        },
        "christoffersen_p": {
            "p10": christoffersen_independence(low_viol),
            "p90": christoffersen_independence(high_viol),
        },
    }


def summarize(result: EvalResult, *, cost_bps: float, n_boot: int, seed: int) -> dict[str, object]:
    """All reported statistics, computed from the stored per-row arrays."""
    y, steps = result.y, result.steps
    forecasters = {name: _forecaster_metrics(y, p) for name, p in result.preds.items()}
    model_row_loss = mean_pinball_per_row(y, result.preds["model"])

    versus: dict[str, object] = {}
    for name, p in result.preds.items():
        if name == "model":
            continue
        base_row_loss = mean_pinball_per_row(y, p)
        stat, p_value = diebold_mariano(model_row_loss, base_row_loss, h=steps)
        lo, hi = block_bootstrap_ci(
            lambda a, b: float(np.mean(a - b)),
            model_row_loss,
            base_row_loss,
            block=max(2 * steps, 10),
            n_boot=n_boot,
            seed=seed,
        )
        versus[name] = {
            "loss_diff": float(np.mean(model_row_loss - base_row_loss)),
            "diff_ci95": [lo, hi],
            "dm_stat": stat,
            "dm_p": p_value,
        }

    p50 = result.preds["model"][:, 1]
    periods = ANNUALISATION[result.horizon]
    net, gross = trade_pnl(p50, y, steps, cost_bps), trade_pnl(p50, y, steps, 0.0)
    trading = {
        "cost_bps": cost_bps,
        "sharpe_net": annualized_sharpe(net, periods),
        "sharpe_gross": annualized_sharpe(gross, periods),
        "sharpe_buy_and_hold": annualized_sharpe(buy_and_hold_pnl(y, steps), periods),
        "n_periods": int(len(net)),
        "directional_hit_rate": float(np.mean(np.sign(p50) == np.sign(y))),
    }

    ref_loss = forecasters[REFERENCE]["mean_pinball"]
    assert isinstance(ref_loss, float)
    return {
        "horizon": result.horizon,
        "n_predictions": int(len(y)),
        "n_folds": result.n_folds,
        "skill_vs_empirical_quantile": pinball_skill_score(
            float(forecasters["model"]["mean_pinball"]), ref_loss  # type: ignore[arg-type]
        ),
        "forecasters": forecasters,
        "model_vs": versus,
        "trading": trading,
    }


def _json_safe(obj: object) -> object:
    """Replace NaN with null: a NaN statistic means undefined, and JSON has no NaN."""
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def save_summary(summary: dict[str, object], path: Path) -> None:
    path.write_text(json.dumps(_json_safe(summary), indent=2) + "\n")


def render_markdown(summary: dict[str, object]) -> str:
    """Compact results table, in the form ``docs/results.md`` expects."""
    forecasters = summary["forecasters"]
    versus = summary["model_vs"]
    assert isinstance(forecasters, dict) and isinstance(versus, dict)
    lines = [
        f"Horizon {summary['horizon']}: {summary['n_predictions']} predictions over "
        f"{summary['n_folds']} folds. Skill vs empirical quantiles: "
        f"{summary['skill_vs_empirical_quantile']:+.3%}.",
        "",
        "| Forecaster | Mean pinball | WIS | P10-P90 coverage | Model - this (DM p) |",
        "|---|---|---|---|---|",
    ]
    for name, m in forecasters.items():
        diff = (
            ""
            if name == "model"
            else f"{versus[name]['loss_diff']:+.2e} ({versus[name]['dm_p']:.3f})"
        )
        lines.append(
            f"| {name} | {m['mean_pinball']:.6f} | {m['wis']:.6f} | {m['coverage_80']:.3f} | {diff} |"
        )
    return "\n".join(lines)
