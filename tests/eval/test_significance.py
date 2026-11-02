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
