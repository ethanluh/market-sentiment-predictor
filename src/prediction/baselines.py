"""
Baseline return predictors for benchmarking the quantile model.

All baselines share the interface ``fit(returns) -> self`` /
``predict(horizon) -> QuantilePrediction`` so the backtester can swap them in
for :class:`~src.prediction.model.QuantileReturnModel`.

  - ``MomentumBaseline`` : drift = recent mean log-return; intervals from
    empirical residual quantiles.
  - ``ARIMABaseline``    : statsmodels ARIMA mean forecast + normal interval.
  - ``GARCHBaseline``    : EWMA/rolling volatility (a lightweight GARCH-like
    stand-in; statsmodels has no native GARCH and ``arch`` is out of scope)
    with a zero-drift mean and z-score intervals.

Any model that needs statsmodels degrades to momentum on insufficient history
or a fit/convergence error rather than raising.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from src.prediction.model import QuantilePrediction

# z-scores for the 10th / 50th / 90th percentiles of a standard normal.
_Z = {0.10: -1.2815515594600314, 0.50: 0.0, 0.90: 1.2815515594600314}
_EPS = 1e-9


def horizon_to_steps(horizon: str) -> int:
    """
    Map a horizon label to a number of forward bars (best-effort).

    Steps are counted in *bars*, so the mapping depends on the price frame's
    interval. ``"1h"`` is only meaningful on an intraday frame; on the default
    daily bars it resolves to one step — identical to ``"1d"`` — so callers
    backtesting daily data should either supply intraday bars or drop ``"1h"``.
    """
    mapping = {"1h": 1, "1d": 1, "5d": 5}
    if horizon not in mapping:
        raise ValueError(f"Unknown horizon {horizon!r}")
    return mapping[horizon]


@runtime_checkable
class BaselinePredictor(Protocol):
    def fit(self, returns: pd.Series) -> "BaselinePredictor": ...
    def predict(self, horizon: str) -> QuantilePrediction: ...


def _interval_from_vol(mean: float, vol: float) -> QuantilePrediction:
    """Build a quantile prediction from a mean and per-horizon volatility."""
    vol = max(vol, _EPS)
    return QuantilePrediction(
        p10=mean + _Z[0.10] * vol,
        p50=mean,
        p90=mean + _Z[0.90] * vol,
    )


@dataclass
class MomentumBaseline:
    lookback: int = 20
    _returns: pd.Series | None = field(default=None, repr=False)

    def fit(self, returns: pd.Series) -> "MomentumBaseline":
        self._returns = pd.Series(returns).dropna()
        return self

    def predict(self, horizon: str) -> QuantilePrediction:
        if self._returns is None or self._returns.empty:
            return QuantilePrediction(0.0, 0.0, 0.0)
        steps = horizon_to_steps(horizon)
        recent = self._returns.tail(self.lookback)
        drift = float(recent.mean()) * steps
        vol = float(recent.std(ddof=0)) * np.sqrt(steps)
        return _interval_from_vol(drift, vol)


@dataclass
class ARIMABaseline:
    order: tuple[int, int, int] = (1, 0, 0)
    min_obs: int = 30
    _result: object | None = field(default=None, repr=False)
    _resid_std: float = 0.0
    _fallback: MomentumBaseline | None = field(default=None, repr=False)

    def fit(self, returns: pd.Series) -> "ARIMABaseline":
        series = pd.Series(returns).dropna().reset_index(drop=True)
        if len(series) < self.min_obs:
            self._fallback = MomentumBaseline().fit(series)
            return self
        try:
            from statsmodels.tsa.arima.model import ARIMA

            self._result = ARIMA(series, order=self.order).fit()
            self._resid_std = float(np.std(self._result.resid))  # type: ignore[attr-defined]
        except Exception:  # convergence / linalg errors → degrade gracefully
            self._fallback = MomentumBaseline().fit(series)
        return self

    def predict(self, horizon: str) -> QuantilePrediction:
        if self._fallback is not None or self._result is None:
            return (self._fallback or MomentumBaseline()).predict(horizon)
        steps = horizon_to_steps(horizon)
        forecast = float(np.asarray(self._result.forecast(steps))[-1])  # type: ignore[attr-defined]
        vol = max(self._resid_std, _EPS) * np.sqrt(steps)
        return _interval_from_vol(forecast, vol)


@dataclass
class GARCHBaseline:
    """EWMA volatility model (lightweight GARCH stand-in), zero-drift mean."""

    lam: float = 0.94
    min_obs: int = 20
    _vol: float = 0.0
    _fallback: MomentumBaseline | None = field(default=None, repr=False)

    def fit(self, returns: pd.Series) -> "GARCHBaseline":
        series = pd.Series(returns).dropna()
        if len(series) < self.min_obs:
            self._fallback = MomentumBaseline().fit(series)
            return self
        # Exponentially weighted variance, RiskMetrics-style.
        ewm_var = series.pow(2).ewm(alpha=1 - self.lam).mean().iloc[-1]
        self._vol = float(np.sqrt(max(ewm_var, _EPS)))
        return self

    def predict(self, horizon: str) -> QuantilePrediction:
        if self._fallback is not None:
            return self._fallback.predict(horizon)
        steps = horizon_to_steps(horizon)
        vol = self._vol * np.sqrt(steps)
        return _interval_from_vol(0.0, vol)
