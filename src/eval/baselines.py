"""
Per-row baselines for the evaluation harness.

Each baseline is a callable ``position -> (p10, p50, p90)`` that sees only data
realised by that row, so baselines are updated every test row like the model's
features are. ``position`` indexes the full log-return series.

Failures raise: the legacy ARIMA/GARCH classes silently fall back to momentum on
a fit error, which would make a benchmark compare against the wrong model, so the
fallback is detected and rejected here.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from src.prediction.baselines import (
    ARIMABaseline,
    BaselinePredictor,
    GARCHBaseline,
    MomentumBaseline,
)
from src.prediction.model import QUANTILES
from src.prediction.volatility import ewma_vol, fhs_quantiles, scale_back, standardize

RowForecast = tuple[float, float, float]
RowBaseline = Callable[[int], RowForecast]

# Smallest history for which a baseline forecast is meaningful.
MIN_HISTORY = 30


def _strict_return_baseline(
    make: Callable[[], BaselinePredictor], log_returns: pd.Series, horizon: str
) -> RowBaseline:
    def forecast(pos: int) -> RowForecast:
        history = log_returns.iloc[: pos + 1].dropna()
        if len(history) < MIN_HISTORY:
            raise ValueError(f"baseline needs >= {MIN_HISTORY} returns, got {len(history)}")
        baseline = make().fit(history)
        min_obs = getattr(baseline, "min_obs", 0)
        if getattr(baseline, "_fallback", None) is not None and len(history) >= min_obs:
            raise RuntimeError(
                f"{type(baseline).__name__} fell back to momentum with {len(history)} "
                "observations; refusing to benchmark against the wrong model"
            )
        qp = baseline.predict(horizon)
        return qp.p10, qp.p50, qp.p90

    return forecast


def _empirical_quantile_baseline(forward_target: pd.Series, steps: int, window: int) -> RowBaseline:
    """Quantiles of the last ``window`` horizon returns fully realised by the row."""
    values = forward_target.to_numpy()

    def forecast(pos: int) -> RowForecast:
        # Row i's target is known only once bar i+steps has closed.
        end = pos - steps + 1  # exclusive; a non-positive end would wrap around the array
        if end <= 0:
            raise ValueError(f"no realised targets at position {pos} for steps={steps}")
        hist = values[max(0, end - window) : end]
        hist = hist[~np.isnan(hist)]
        if len(hist) < MIN_HISTORY:
            raise ValueError(f"empirical baseline needs >= {MIN_HISTORY} targets, got {len(hist)}")
        q10, q50, q90 = np.quantile(hist, QUANTILES)
        return float(q10), float(q50), float(q90)

    return forecast


def _fhs_baseline(
    forward_target: pd.Series, log_returns: pd.Series, steps: int, window: int
) -> RowBaseline:
    """
    Filtered historical simulation: the EWMA volatility forecast times the empirical quantiles of
    past volatility-standardised returns. Same scale forecast as ``garch``; only the distribution
    shape differs (empirical instead of normal). Uses only targets fully realised by the row.
    """
    vol = ewma_vol(log_returns).to_numpy()
    standardized = standardize(forward_target.to_numpy(), vol, steps)

    def forecast(pos: int) -> RowForecast:
        end = pos - steps + 1  # row i's target is known only once bar i+steps has closed
        if end <= 0:
            raise ValueError(f"no realised targets at position {pos} for steps={steps}")
        hist = standardized[max(0, end - window) : end]
        hist = hist[~np.isnan(hist)]
        if len(hist) < MIN_HISTORY:
            raise ValueError(
                f"FHS baseline needs >= {MIN_HISTORY} standardised targets, got {len(hist)}"
            )
        if np.isnan(vol[pos]):
            raise ValueError(f"EWMA volatility is not available at position {pos}")
        q10, q50, q90 = scale_back(fhs_quantiles(hist), float(vol[pos]), steps)
        return float(q10), float(q50), float(q90)

    return forecast


def make_baselines(
    log_returns: pd.Series,
    forward_target: pd.Series,
    horizon: str,
    steps: int,
    window: int,
    include_arima: bool = False,
) -> dict[str, RowBaseline]:
    """
    The benchmark set: zero forecast, rolling empirical quantiles, momentum, EWMA volatility with
    normal quantiles (``garch``) and with empirical standardised quantiles (``fhs``). ARIMA refits
    every row and is slow, so it is opt-in.
    """
    out: dict[str, RowBaseline] = {
        "zero": lambda pos: (0.0, 0.0, 0.0),
        "empirical_quantile": _empirical_quantile_baseline(forward_target, steps, window),
        "momentum": _strict_return_baseline(MomentumBaseline, log_returns, horizon),
        "garch": _strict_return_baseline(GARCHBaseline, log_returns, horizon),
        "fhs": _fhs_baseline(forward_target, log_returns, steps, window),
    }
    if include_arima:
        out["arima"] = _strict_return_baseline(ARIMABaseline, log_returns, horizon)
    return out
