"""
Quantile-forecast metrics. Loss functions return per-row arrays so they can feed
significance tests; callers take ``.mean()`` for a summary.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.special import xlogy

QUANTILES: tuple[float, float, float] = (0.10, 0.50, 0.90)


def pinball_per_row(y: np.ndarray, pred: np.ndarray, quantile: float) -> np.ndarray:
    """Pinball loss of each prediction at one quantile level."""
    diff = np.asarray(y, dtype=float) - np.asarray(pred, dtype=float)
    return np.maximum(quantile * diff, (quantile - 1.0) * diff)


def mean_pinball_per_row(y: np.ndarray, preds: np.ndarray) -> np.ndarray:
    """Per-row pinball averaged over P10/P50/P90; ``preds`` has shape (n, 3)."""
    if preds.ndim != 2 or preds.shape[1] != len(QUANTILES):
        raise ValueError(f"preds must have shape (n, {len(QUANTILES)}), got {preds.shape}")
    return np.mean([pinball_per_row(y, preds[:, k], q) for k, q in enumerate(QUANTILES)], axis=0)


def interval_score_per_row(y: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Interval (Winkler) score of the central 80% interval, alpha = 0.2."""
    alpha = 1.0 - (QUANTILES[2] - QUANTILES[0])
    y = np.asarray(y, dtype=float)
    return (
        (upper - lower)
        + (2.0 / alpha) * np.maximum(lower - y, 0.0)
        + (2.0 / alpha) * np.maximum(y - upper, 0.0)
    )


def weighted_interval_score_per_row(y: np.ndarray, preds: np.ndarray) -> np.ndarray:
    """
    Weighted interval score from the median and the 80% interval (Bracher et al. 2021).

    Equals ``(0.5 * |y - median| + (alpha / 2) * interval_score) / 1.5`` for one interval.
    """
    alpha = 1.0 - (QUANTILES[2] - QUANTILES[0])
    abs_err = np.abs(np.asarray(y, dtype=float) - preds[:, 1])
    iscore = interval_score_per_row(y, preds[:, 0], preds[:, 2])
    return (0.5 * abs_err + (alpha / 2.0) * iscore) / 1.5


def interval_coverage(y: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Fraction of actuals inside ``[lower, upper]``."""
    if len(y) == 0:
        raise ValueError("coverage of an empty sample is undefined")
    return float(np.mean((y >= lower) & (y <= upper)))


def quantile_hit_rates(y: np.ndarray, preds: np.ndarray) -> dict[str, float]:
    """Fraction of actuals at or below each predicted quantile; ideal is the level itself."""
    return {f"p{int(q * 100)}": float(np.mean(y <= preds[:, k])) for k, q in enumerate(QUANTILES)}


def pinball_skill_score(loss: float, reference_loss: float) -> float:
    """``1 - loss / reference_loss``: positive means better than the reference."""
    if reference_loss <= 0:
        raise ValueError(f"reference loss must be positive, got {reference_loss}")
    return 1.0 - loss / reference_loss


def kupiec_pof(violations: np.ndarray, expected_rate: float) -> float:
    """
    Kupiec proportion-of-failures test p-value.

    Null: the violation probability equals ``expected_rate``. ``violations`` is a
    boolean array, True where the actual breached the quantile.
    """
    v = np.asarray(violations, dtype=bool)
    n, x = len(v), int(v.sum())
    if n == 0:
        raise ValueError("Kupiec test needs at least one observation")
    pi = x / n
    ll_null = xlogy(n - x, 1.0 - expected_rate) + xlogy(x, expected_rate)
    ll_alt = xlogy(n - x, 1.0 - pi) + xlogy(x, pi)
    return float(stats.chi2.sf(2.0 * (ll_alt - ll_null), df=1))


def christoffersen_independence(violations: np.ndarray) -> float:
    """
    Christoffersen independence test p-value: are violations clustered in time?

    Overlapping multi-step horizons autocorrelate violations by construction, so for
    ``steps > 1`` a small p-value is expected and not evidence of a bad model.
    """
    h = np.asarray(violations, dtype=int)
    if len(h) < 2:
        raise ValueError("independence test needs at least two observations")
    prev, curr = h[:-1], h[1:]
    n00 = int(np.sum((prev == 0) & (curr == 0)))
    n01 = int(np.sum((prev == 0) & (curr == 1)))
    n10 = int(np.sum((prev == 1) & (curr == 0)))
    n11 = int(np.sum((prev == 1) & (curr == 1)))
    # A zero denominator means the matching counts are zero, and xlogy(0, .) = 0 below.
    pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    ll_ind = xlogy(n00 + n10, 1.0 - pi) + xlogy(n01 + n11, pi)
    ll_dep = xlogy(n00, 1.0 - pi01) + xlogy(n01, pi01) + xlogy(n10, 1.0 - pi11) + xlogy(n11, pi11)
    return float(stats.chi2.sf(2.0 * (ll_dep - ll_ind), df=1))
