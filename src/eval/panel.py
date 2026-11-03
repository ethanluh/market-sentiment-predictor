"""
Panel (cross-ticker) comparison of the model against a baseline.

Tickers move together (market-wide shocks hit every name the same day), so stacking all tickers'
rows and testing them as independent would overstate the evidence. Instead, each date's loss
differential is averaged across tickers, and the Diebold-Mariano test (HAC variance for overlapping
horizons) runs on that one time series.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.eval.metrics import mean_pinball_per_row
from src.eval.runner import EvalResult
from src.eval.significance import block_bootstrap_mean_p, diebold_mariano, newey_west_lags


@dataclass(frozen=True)
class PanelTest:
    mean_relative_diff: float  # negative: the model has lower loss than the baseline, on average
    dm_stat: float
    dm_p_two_sided: float  # the more conservative of the HAC test and the block bootstrap
    n_dates: int
    n_tickers: int
    dm_p_hac: float = float("nan")
    p_bootstrap: float = float("nan")


def relative_loss_differential(result: EvalResult, baseline: str) -> pd.Series:
    """
    Per-row ``(model loss - baseline loss) / mean baseline loss``, indexed by timestamp.

    Dividing by the ticker's own mean baseline loss puts quiet and volatile names on one scale, so
    a high-volatility ticker does not dominate the panel average.
    """
    if baseline not in result.preds:
        raise ValueError(f"no baseline {baseline!r}; have {sorted(result.preds)}")
    model = mean_pinball_per_row(result.y, result.preds["model"])
    base = mean_pinball_per_row(result.y, result.preds[baseline])
    scale = float(base.mean())
    if scale <= 0.0:
        raise ValueError("the baseline's mean loss is not positive")
    return pd.Series((model - base) / scale, index=result.index)


def mean_differential_test(d: np.ndarray, h: int, seed: int = 0) -> tuple[float, float, float]:
    """
    Test that the mean of a loss-differential series is zero; returns ``(stat, p_hac, p_bootstrap)``.

    ``p_hac`` is a Diebold-Mariano test with Newey-West lags (never fewer than the h-1 that
    overlapping horizons force). ``p_bootstrap`` is a moving-block bootstrap with blocks of about
    ``4 * n^(1/3)``. Both are needed because the differential is persistent (volatility regimes), not
    just overlapping: in simulation on AR(0.8) noise the old h-1-lag test rejected a true null 22%
    of the time, this pair about 9%. It stays somewhat liberal for extremely persistent series
    (AR 0.9 and above), which the gate's other criteria are there to backstop.
    """
    n = len(d)
    stat, p_hac = diebold_mariano(d, np.zeros(n), h=h, lags=min(newey_west_lags(n, h), n - 1))
    p_boot = block_bootstrap_mean_p(d, block=max(2 * h, int(4 * n ** (1 / 3))), seed=seed)
    return stat, p_hac, p_boot


def _longest_run(mask: pd.Series) -> pd.Series:
    """The longest stretch of consecutive True values, as a boolean mask over the same index."""
    best_start = best_len = start = length = 0
    for i, ok in enumerate(mask.to_numpy()):
        if ok:
            if length == 0:
                start = i
            length += 1
            if length > best_len:
                best_start, best_len = start, length
        else:
            length = 0
    out = pd.Series(False, index=mask.index)
    out.iloc[best_start : best_start + best_len] = True
    return out


def panel_test(
    results: dict[str, EvalResult],
    baseline: str,
    min_tickers_per_date: int | None = None,
    seed: int = 0,
) -> PanelTest:
    """
    Average the relative loss differential across tickers per date, then test that series.

    Only dates covered by at least ``min_tickers_per_date`` tickers (default: half of them) are
    used, and only the longest consecutive stretch of such dates, so the series has no gaps for the
    autocorrelation correction to straddle. The p-value is the larger of a Newey-West HAC
    Diebold-Mariano test and a block bootstrap, because volatility regimes make the loss
    differential persistent well beyond the overlap of the forecast horizon.
    """
    if not results:
        raise ValueError("no results to pool")
    steps = {r.steps for r in results.values()}
    if len(steps) != 1:
        raise ValueError(f"results mix horizons with different step counts: {sorted(steps)}")
    h = steps.pop()
    frame = pd.DataFrame(
        {t: relative_loss_differential(r, baseline) for t, r in results.items()}
    ).sort_index()
    need = (
        min_tickers_per_date if min_tickers_per_date is not None else max(1, -(-len(results) // 2))
    )
    covered = frame.notna().sum(axis=1) >= need
    if not covered.any():
        raise ValueError("no date has enough tickers")
    mean_by_date = frame[_longest_run(covered)].mean(axis=1).to_numpy()
    n = len(mean_by_date)
    stat, p_hac, p_boot = mean_differential_test(mean_by_date, h, seed)
    return PanelTest(
        mean_relative_diff=float(mean_by_date.mean()),
        dm_stat=stat,
        dm_p_two_sided=max(p_hac, p_boot),
        n_dates=n,
        n_tickers=len(results),
        dm_p_hac=p_hac,
        p_bootstrap=p_boot,
    )
