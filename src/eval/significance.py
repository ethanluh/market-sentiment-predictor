"""Significance tests for forecast comparisons."""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import stats


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, h: int = 1) -> tuple[float, float]:
    """
    Diebold-Mariano test of equal expected loss, with Bartlett HAC variance and the
    Harvey-Leybourne-Newbold small-sample correction.

    ``h`` is the forecast horizon in bars; overlapping h-step forecasts have
    MA(h-1) loss differentials, so the variance uses lags 1..h-1.
    Returns ``(statistic, two-sided p-value)``. A negative statistic means A has the
    lower loss.
    """
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    n = len(d)
    if h < 1:
        raise ValueError(f"h must be >= 1, got {h}")
    if n <= 2 * h:
        raise ValueError(f"need more than 2*h = {2 * h} observations, got {n}")
    centered = d - d.mean()
    var = float(centered @ centered) / n
    for k in range(1, h):
        var += 2.0 * (1.0 - k / h) * float(centered[k:] @ centered[:-k]) / n
    if var <= 0.0:
        raise ValueError("loss differential has zero or negative long-run variance")
    stat = float(d.mean() / np.sqrt(var / n))
    stat *= float(np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n))
    return stat, float(2.0 * stats.t.sf(abs(stat), df=n - 1))


def block_bootstrap_ci(
    stat_fn: Callable[..., float],
    *arrays: np.ndarray,
    block: int,
    n_boot: int = 1000,
    seed: int = 0,
    level: float = 0.95,
) -> tuple[float, float]:
    """
    Moving-block bootstrap confidence interval for ``stat_fn(*arrays)``.

    All arrays are resampled with the same block indices, so paired series stay
    paired. ``block`` should be at least the forecast horizon in bars.
    """
    n = len(arrays[0])
    if any(len(a) != n for a in arrays):
        raise ValueError("all arrays must have the same length")
    if not 1 <= block <= n:
        raise ValueError(f"block must be in [1, {n}], got {block}")
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")
    rng = np.random.default_rng(seed)
    n_blocks = -(-n // block)
    offsets = np.arange(block)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        idx = (starts[:, None] + offsets[None, :]).ravel()[:n]
        draws[b] = stat_fn(*(a[idx] for a in arrays))
    lo, hi = np.quantile(draws, [(1.0 - level) / 2.0, (1.0 + level) / 2.0])
    return float(lo), float(hi)
