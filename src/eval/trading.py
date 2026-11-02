"""Position-based PnL with transaction costs and horizon-correct annualisation."""

from __future__ import annotations

import numpy as np

# Periods per year of the non-overlapping holding period for each horizon.
ANNUALISATION: dict[str, float] = {"1h": 252 * 6.5, "1d": 252.0, "5d": 252.0 / 5.0}


def trade_pnl(
    p50: np.ndarray, realized: np.ndarray, steps: int, cost_bps: float, long_only: bool = False
) -> np.ndarray:
    """
    Net log-return PnL per holding period from ``sign(P50)`` positions.

    Forecasts overlap when ``steps > 1``, so only every ``steps``-th row is traded;
    the held period then matches the forecast horizon and periods do not overlap.
    Each unit of position change costs ``cost_bps`` basis points. ``long_only``
    floors short positions at flat.
    """
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    pos = np.sign(np.asarray(p50, dtype=float))[::steps]
    if long_only:
        pos = np.maximum(pos, 0.0)
    r = np.asarray(realized, dtype=float)[::steps]
    prev = np.concatenate([[0.0], pos[:-1]])
    return pos * r - np.abs(pos - prev) * cost_bps / 1e4


def buy_and_hold_pnl(realized: np.ndarray, steps: int) -> np.ndarray:
    """Always-long PnL over the same non-overlapping periods, entered once at no cost."""
    return np.asarray(realized, dtype=float)[::steps]


def annualized_sharpe(pnl: np.ndarray, periods_per_year: float) -> float:
    """
    Annualised Sharpe ratio of per-period PnL (zero risk-free rate).

    Returns NaN when PnL has zero variance (for example a flat position throughout),
    where the ratio is undefined; raises if there are fewer than two periods.
    """
    if len(pnl) < 2:
        raise ValueError("Sharpe needs at least two periods")
    std = float(np.std(pnl, ddof=1))
    if std == 0.0:
        return float("nan")
    return float(np.mean(pnl) / std * np.sqrt(periods_per_year))
