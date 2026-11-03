"""Multiple-testing control for the many per-ticker comparisons."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def one_sided_p(stat: float, two_sided_p: float) -> float:
    """
    One-sided p-value for "the model has LOWER loss" from a Diebold-Mariano statistic.

    A negative statistic means the model's loss is lower, so the one-sided p is half the two-sided
    p; a positive statistic is evidence against the model and gets ``1 - p/2``.
    """
    if not 0.0 <= two_sided_p <= 1.0:
        raise ValueError(f"p-value must be in [0, 1], got {two_sided_p}")
    return two_sided_p / 2.0 if stat < 0 else 1.0 - two_sided_p / 2.0


def benjamini_hochberg(pvalues: Sequence[float]) -> np.ndarray:
    """
    Benjamini-Hochberg adjusted p-values (q-values), in the input order.

    Rejecting every hypothesis with ``q <= alpha`` controls the false discovery rate at ``alpha``
    for independent tests and for tests that are positively dependent (PRDS). Here the per-ticker
    statistics share market-wide shocks, which makes them positively correlated, so BH is the
    standard, defensible choice; it is not proven for these particular statistics. With 23 tickers,
    testing each at p < 0.05 would produce about one false "edge" by chance alone.
    """
    p = np.asarray(pvalues, dtype=float)
    if p.size == 0:
        raise ValueError("need at least one p-value")
    if np.isnan(p).any() or (p < 0).any() or (p > 1).any():
        raise ValueError("p-values must be in [0, 1] and not NaN")
    m = p.size
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    adjusted_sorted = np.minimum.accumulate(ranked[::-1])[::-1]  # enforce monotonicity from the top
    out = np.empty(m)
    out[order] = np.minimum(adjusted_sorted, 1.0)
    return out
