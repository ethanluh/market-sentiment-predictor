import numpy as np
import pandas as pd
import pytest

from src.prediction.baselines import GARCHBaseline
from src.prediction.volatility import ewma_vol, fhs_quantiles, scale_back, standardize


def returns(n: int = 300, seed: int = 0, df: float | None = None) -> pd.Series:
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 0.01, n) if df is None else rng.standard_t(df, n) * 0.01
    s = pd.Series(x, index=pd.bdate_range("2022-01-03", periods=n, tz="UTC"))
    s.iloc[0] = np.nan  # a return needs a previous close
    return s


def test_ewma_vol_matches_the_legacy_garch_baseline_at_every_point() -> None:
    r = returns()
    vol = ewma_vol(r)
    for pos in (30, 100, 299):
        legacy = GARCHBaseline().fit(r.iloc[: pos + 1].dropna())
        assert vol.iloc[pos] == pytest.approx(legacy._vol, rel=1e-12)


def test_ewma_vol_is_causal() -> None:
    r = returns()
    altered = r.copy()
    altered.iloc[150:] = 0.5  # a wild future
    a, b = ewma_vol(r), ewma_vol(altered)
    pd.testing.assert_series_equal(a.iloc[:150], b.iloc[:150])
    assert not a.iloc[150:].equals(b.iloc[150:])


def test_ewma_vol_needs_a_minimum_history_and_skips_missing_returns() -> None:
    vol = ewma_vol(returns(), min_obs=20)
    assert (
        vol.iloc[:20].isna().all() and vol.iloc[20:].notna().all()
    )  # 1 NaN return + 19 observations
    assert vol.index.equals(returns().index)


def test_ewma_vol_tracks_a_volatility_regime_change() -> None:
    rng = np.random.default_rng(1)
    calm, wild = rng.normal(0, 0.005, 200), rng.normal(0, 0.03, 200)
    vol = ewma_vol(pd.Series(np.r_[calm, wild]))
    assert vol.iloc[190] < 0.008 and vol.iloc[390] > 0.02


@pytest.mark.parametrize("kwargs", [{"lam": 0.0}, {"lam": 1.0}, {"min_obs": 0}])
def test_ewma_vol_validates_arguments(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        ewma_vol(returns(), **kwargs)


def test_standardize_and_scale_back_are_inverses() -> None:
    y, sigma = np.array([0.02, -0.01]), np.array([0.01, 0.02])
    z = standardize(y, sigma, steps=5)
    assert np.allclose(z, y / (sigma * np.sqrt(5)))
    assert np.allclose(scale_back(z[0], float(sigma[0]), 5), y[0])
    with pytest.raises(ValueError):
        standardize(y, sigma, steps=0)
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            scale_back(np.array([1.0]), bad, 5)


def test_fhs_quantiles_are_the_empirical_quantiles_and_validate() -> None:
    sample = np.arange(1.0, 101.0)
    assert np.allclose(fhs_quantiles(sample), np.quantile(sample, [0.1, 0.5, 0.9]))
    for bad in (np.array([]), np.array([1.0, np.nan])):
        with pytest.raises(ValueError):
            fhs_quantiles(bad)
