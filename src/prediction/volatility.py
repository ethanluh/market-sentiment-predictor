"""
Causal EWMA volatility and the pieces of filtered historical simulation (FHS).

Volatility is the one thing in daily returns that is reliably forecastable, so the baseline to beat
is not "no model" but "forecast the scale well and treat the standardised return as noise". These
helpers define that baseline once; the baselines, the pooled model and its tests all use them.

``ewma_vol`` has exactly the definition the legacy ``GARCHBaseline`` uses (RiskMetrics, lambda 0.94,
pandas ``ewm`` with ``alpha = 1 - lambda``), so the normal-quantile baseline and the FHS baseline
share one volatility forecast and differ only in the quantile shape.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.prediction.model import QUANTILES

LAMBDA = 0.94
MIN_OBS = 20  # returns needed before a volatility estimate is trusted
_VAR_FLOOR = 1e-9  # the same floor the legacy GARCHBaseline applies to the variance


def ewma_vol(log_returns: pd.Series, lam: float = LAMBDA, min_obs: int = MIN_OBS) -> pd.Series:
    """
    Per-bar volatility known at the close of each bar: ``sqrt(EWMA of r^2 up to and including t)``.

    Causal by construction (the value at *t* uses returns at or before *t*). NaN until ``min_obs``
    returns exist, and at bars where the return itself is missing.
    """
    if not 0.0 < lam < 1.0:
        raise ValueError(f"lam must be in (0, 1), got {lam}")
    if min_obs < 1:
        raise ValueError(f"min_obs must be >= 1, got {min_obs}")
    r = log_returns.dropna()
    var = r.pow(2).ewm(alpha=1.0 - lam).mean()
    vol = pd.Series(np.sqrt(np.maximum(var.to_numpy(), _VAR_FLOOR)), index=r.index)
    vol.iloc[: min_obs - 1] = np.nan
    return vol.reindex(log_returns.index)


def standardize(y: np.ndarray, sigma: np.ndarray, steps: int) -> np.ndarray:
    """A ``steps``-bar cumulative return divided by its expected scale, ``sigma * sqrt(steps)``."""
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    return np.asarray(y, dtype=float) / (np.asarray(sigma, dtype=float) * np.sqrt(steps))


def scale_back(std_quantiles: np.ndarray, sigma: float, steps: int) -> np.ndarray:
    """Standardised quantiles back to return units: the inverse of :func:`standardize`."""
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError(f"sigma must be finite and positive, got {sigma}")
    return np.asarray(std_quantiles, dtype=float) * sigma * np.sqrt(steps)


def fhs_quantiles(std_samples: np.ndarray, quantiles: tuple[float, ...] = QUANTILES) -> np.ndarray:
    """Empirical quantiles of standardised returns: the shape of the distribution, scale removed."""
    s = np.asarray(std_samples, dtype=float)
    if s.size == 0 or not np.isfinite(s).all():
        raise ValueError("FHS needs a non-empty sample of finite standardised returns")
    return np.quantile(s, quantiles)
