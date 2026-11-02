import numpy as np
import pytest

from src.eval.metrics import (
    christoffersen_independence,
    interval_coverage,
    interval_score_per_row,
    kupiec_pof,
    mean_pinball_per_row,
    pinball_per_row,
    pinball_skill_score,
    quantile_hit_rates,
    weighted_interval_score_per_row,
)


def test_pinball_known_values() -> None:
    y, pred = np.array([1.0, 1.0]), np.array([0.0, 2.0])
    # Under-prediction costs q * error; over-prediction costs (1 - q) * error.
    assert pinball_per_row(y, pred, 0.9).tolist() == pytest.approx([0.9, 0.1])
    assert pinball_per_row(y, pred, 0.1).tolist() == pytest.approx([0.1, 0.9])


def test_pinball_is_minimised_at_the_true_quantile() -> None:
    y = np.random.default_rng(0).normal(size=20000)
    grid = np.linspace(-2, 2, 81)
    best = grid[np.argmin([pinball_per_row(y, np.full_like(y, c), 0.9).mean() for c in grid])]
    assert best == pytest.approx(1.2816, abs=0.1)


def test_mean_pinball_averages_three_quantiles() -> None:
    y = np.array([0.5])
    preds = np.array([[0.0, 0.0, 0.0]])
    expected = np.mean([0.1 * 0.5, 0.5 * 0.5, 0.9 * 0.5])
    assert mean_pinball_per_row(y, preds)[0] == pytest.approx(expected)


def test_mean_pinball_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError):
        mean_pinball_per_row(np.zeros(3), np.zeros((3, 2)))


def test_interval_score() -> None:
    lo, hi = np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0])
    y = np.array([0.0, 2.0, -3.0])
    # width 2; each unit outside costs 2 / 0.2 = 10.
    assert interval_score_per_row(y, lo, hi).tolist() == pytest.approx([2.0, 12.0, 22.0])


def test_wis_matches_formula() -> None:
    y = np.array([2.0])
    preds = np.array([[-1.0, 0.0, 1.0]])
    iscore = 2.0 + 10.0 * 1.0
    assert weighted_interval_score_per_row(y, preds)[0] == pytest.approx(
        (0.5 * 2.0 + 0.1 * iscore) / 1.5
    )


def test_coverage_and_hit_rates() -> None:
    y = np.array([0.0, 1.0, 2.0, 3.0])
    preds = np.column_stack([np.full(4, 0.5), np.full(4, 1.5), np.full(4, 2.5)])
    assert interval_coverage(y, preds[:, 0], preds[:, 2]) == 0.5
    assert quantile_hit_rates(y, preds) == {"p10": 0.25, "p50": 0.5, "p90": 0.75}


def test_empty_coverage_raises() -> None:
    with pytest.raises(ValueError):
        interval_coverage(np.array([]), np.array([]), np.array([]))


def test_skill_score() -> None:
    assert pinball_skill_score(0.8, 1.0) == pytest.approx(0.2)
    assert pinball_skill_score(1.2, 1.0) == pytest.approx(-0.2)
    with pytest.raises(ValueError):
        pinball_skill_score(1.0, 0.0)


def test_kupiec_matches_direct_likelihood_ratio() -> None:
    from scipy.stats import chi2

    n, x, p = 100, 20, 0.10
    pi = x / n
    lr = -2 * ((n - x) * np.log(1 - p) + x * np.log(p) - (n - x) * np.log(1 - pi) - x * np.log(pi))
    violations = np.array([True] * x + [False] * (n - x))
    assert kupiec_pof(violations, p) == pytest.approx(chi2.sf(lr, 1))


def test_kupiec_accepts_exact_rate_and_rejects_bad_rate() -> None:
    ok = np.array([True] * 10 + [False] * 90)
    assert kupiec_pof(ok, 0.10) == pytest.approx(1.0)
    assert kupiec_pof(np.zeros(100, dtype=bool), 0.10) < 0.01


def test_kupiec_empty_raises() -> None:
    with pytest.raises(ValueError):
        kupiec_pof(np.array([], dtype=bool), 0.1)


def test_christoffersen_flags_clustering() -> None:
    clustered = np.array(([1] * 10 + [0] * 90) * 5, dtype=bool)
    spread = np.array(([1] + [0] * 9) * 50, dtype=bool)
    assert christoffersen_independence(clustered) < 0.01
    assert christoffersen_independence(spread) > christoffersen_independence(clustered)


def test_christoffersen_without_violations_is_one() -> None:
    assert christoffersen_independence(np.zeros(50, dtype=bool)) == pytest.approx(1.0)
