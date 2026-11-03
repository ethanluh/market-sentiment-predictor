import numpy as np
import pytest
from scipy import stats

from src.eval.significance import block_bootstrap_ci, diebold_mariano


def test_dm_equals_one_sample_t_statistic_at_h1() -> None:
    rng = np.random.default_rng(0)
    a, b = rng.normal(1.0, 1.0, 200), rng.normal(1.1, 1.0, 200)
    stat, p = diebold_mariano(a, b, h=1)
    t = stats.ttest_1samp(a - b, 0.0)
    assert stat == pytest.approx(t.statistic)
    assert p == pytest.approx(t.pvalue)


def test_dm_negative_statistic_when_a_is_better() -> None:
    rng = np.random.default_rng(1)
    b = rng.exponential(1.0, 400)
    stat, p = diebold_mariano(b - 0.3 + rng.normal(0, 0.1, 400), b, h=1)
    assert stat < 0 and p < 0.001


def test_dm_null_rejects_about_five_percent() -> None:
    rng = np.random.default_rng(2)
    rejections = sum(
        diebold_mariano(rng.normal(size=300), rng.normal(size=300))[1] < 0.05 for _ in range(400)
    )
    assert 0.02 < rejections / 400 < 0.09


def test_dm_hac_widens_the_interval_for_overlapping_losses() -> None:
    rng = np.random.default_rng(3)
    shocks = rng.normal(size=605)
    d = np.convolve(shocks, np.ones(5) / 5, mode="valid")[:600]  # MA(4): overlapping 5-step losses
    naive = diebold_mariano(d, np.zeros(600), h=1)[1]
    hac = diebold_mariano(d, np.zeros(600), h=5)[1]
    assert hac > naive


def test_dm_rejects_degenerate_input() -> None:
    same = np.ones(50)
    with pytest.raises(ValueError, match="variance"):
        diebold_mariano(same, same)
    with pytest.raises(ValueError):
        diebold_mariano(np.ones(5), np.zeros(5), h=3)


def test_bootstrap_ci_covers_the_mean() -> None:
    x = np.random.default_rng(0).normal(size=500)
    lo, hi = block_bootstrap_ci(lambda a: float(a.mean()), x, block=1, n_boot=500, seed=1)
    assert lo < 0.0 < hi
    assert (hi - lo) == pytest.approx(2 * 1.96 / np.sqrt(500), rel=0.25)


def test_bootstrap_keeps_arrays_paired() -> None:
    x = np.arange(100, dtype=float)
    lo, hi = block_bootstrap_ci(
        lambda a, b: float(np.mean(a - b)), x, x + 7.0, block=10, n_boot=200, seed=0
    )
    assert lo == pytest.approx(-7.0) and hi == pytest.approx(-7.0)


def test_bootstrap_is_seeded() -> None:
    x = np.random.default_rng(0).normal(size=100)
    run = lambda: block_bootstrap_ci(
        lambda a: float(a.mean()), x, block=5, n_boot=100, seed=7
    )  # noqa: E731
    assert run() == run()


def test_bootstrap_validates_arguments() -> None:
    x = np.zeros(10)
    with pytest.raises(ValueError):
        block_bootstrap_ci(lambda a: 0.0, x, block=11)
    with pytest.raises(ValueError):
        block_bootstrap_ci(lambda a, b: 0.0, x, np.zeros(9), block=2)
    with pytest.raises(ValueError):
        block_bootstrap_ci(lambda a: 0.0, x, block=2, level=1.0)


# ---- Newey-West lags and the block-bootstrap mean test --------------------------------------------


def test_newey_west_lags_never_fall_below_the_overlap() -> None:
    from src.eval.significance import newey_west_lags

    assert newey_west_lags(100, 1) == 4 and newey_west_lags(1000, 1) == 6
    assert newey_west_lags(50, 20) == 19  # overlap dominates a short series


def test_default_lags_are_the_overlap_and_explicit_lags_match_when_equal() -> None:
    rng = np.random.default_rng(0)
    a, b = rng.normal(1, 1, 300), rng.normal(1, 1, 300)
    assert diebold_mariano(a, b, h=5) == diebold_mariano(a, b, h=5, lags=4)
    assert diebold_mariano(a, b, h=5, lags=10) != diebold_mariano(a, b, h=5, lags=4)


def test_invalid_lags_are_rejected() -> None:
    d = np.random.default_rng(1).normal(size=50)
    for bad in (-1, 50, 80):
        with pytest.raises(ValueError, match="lags must be in"):
            diebold_mariano(d, np.zeros(50), h=1, lags=bad)


def test_bootstrap_mean_p_detects_a_real_mean_and_ignores_noise() -> None:
    from src.eval.significance import block_bootstrap_mean_p

    rng = np.random.default_rng(2)
    assert block_bootstrap_mean_p(rng.normal(0.4, 1, 500), block=10, seed=0) < 0.01
    assert block_bootstrap_mean_p(rng.normal(0.0, 1, 500), block=10, seed=0) > 0.05


def test_bootstrap_mean_p_is_seeded_and_validates() -> None:
    from src.eval.significance import block_bootstrap_mean_p

    d = np.random.default_rng(3).normal(size=100)
    assert block_bootstrap_mean_p(d, 5, seed=7) == block_bootstrap_mean_p(d, 5, seed=7)
    with pytest.raises(ValueError):
        block_bootstrap_mean_p(d, 0)
    with pytest.raises(ValueError):
        block_bootstrap_mean_p(d, 101)


def test_the_combined_test_is_better_calibrated_than_overlap_only_on_persistent_noise() -> None:
    """On AR(0.8) noise (volatility-regime-like persistence) the old h-1-lag test over-rejects."""
    from src.eval.panel import mean_differential_test

    rng = np.random.default_rng(4)
    n, h, reps = 1000, 5, 120
    old = new = 0
    for _ in range(reps):
        e = rng.normal(size=n)
        d = np.zeros(n)
        for t in range(1, n):
            d[t] = 0.8 * d[t - 1] + e[t]
        old += diebold_mariano(d, np.zeros(n), h=h)[1] < 0.05
        _, p_hac, p_boot = mean_differential_test(d, h)
        new += max(p_hac, p_boot) < 0.05
    assert old / reps > 0.15  # the problem is real: the overlap-only test is clearly too liberal
    assert new / reps < 0.13  # and the combined test is close to the nominal 5%
    assert new < old


def test_the_combined_test_keeps_its_power_against_a_real_effect() -> None:
    from src.eval.panel import mean_differential_test

    rng = np.random.default_rng(5)
    hits = 0
    for _ in range(60):
        d = rng.normal(0.25, 1, 800)
        _, p_hac, p_boot = mean_differential_test(d, 5)
        hits += max(p_hac, p_boot) < 0.05
    assert hits / 60 > 0.9
