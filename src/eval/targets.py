"""Forecast targets for the evaluation harness."""

from __future__ import annotations

import pandas as pd


def forward_log_return(log_returns: pd.Series, steps: int) -> pd.Series:
    """
    Cumulative log-return over the next ``steps`` bars, indexed at the forecast origin.

    The value at bar *t* is ``sum(log_returns[t+1 .. t+steps])``. The last ``steps``
    rows are NaN because the horizon is not yet realised. This differs from a plain
    ``shift(-steps)``, which for ``steps > 1`` returns the single bar *t+steps* rather
    than the return over the whole horizon.
    """
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    return log_returns.rolling(steps).sum().shift(-steps)
