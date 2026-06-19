"""Tests for prediction.baselines."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.prediction.baselines import (
    HORIZON_SPECS,
    ARIMABaseline,
    BaselinePredictor,
    GARCHBaseline,
    MomentumBaseline,
    horizon_to_interval,
    horizon_to_steps,
)


def _ar1(n: int = 200, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.3 * x[t - 1] + rng.normal(0, 0.01)
    return pd.Series(x)


class TestMomentum:
    def test_positive_drift(self):
        rets = pd.Series(np.full(30, 0.01))
        qp = MomentumBaseline(lookback=20).fit(rets).predict("1d")
        assert qp.p50 > 0
        assert qp.p10 <= qp.p50 <= qp.p90

    def test_protocol_conformance(self):
        assert isinstance(MomentumBaseline(), BaselinePredictor)


class TestARIMA:
    def test_happy_path_ordered(self):
        qp = ARIMABaseline().fit(_ar1()).predict("1d")
        assert qp.p10 <= qp.p50 <= qp.p90

    def test_insufficient_history_fallback(self):
        qp = ARIMABaseline(min_obs=30).fit(pd.Series([0.01, 0.0, -0.01])).predict("1d")
        assert qp.p10 <= qp.p50 <= qp.p90  # degrades to momentum, no crash


class TestGARCH:
    def test_intervals_widen_with_horizon(self):
        garch = GARCHBaseline().fit(_ar1())
        d1 = garch.predict("1d")
        d5 = garch.predict("5d")
        assert (d5.p90 - d5.p10) >= (d1.p90 - d1.p10)

    def test_insufficient_history_fallback(self):
        qp = GARCHBaseline(min_obs=20).fit(pd.Series([0.01, 0.0])).predict("1d")
        assert qp.p10 <= qp.p50 <= qp.p90


def test_horizon_to_steps():
    assert horizon_to_steps("5d") == 5
    assert horizon_to_steps("1d") == 1


def test_horizon_specs_intraday_vs_daily():
    # "1h" is measured on hourly bars, distinct from the daily "1d"/"5d".
    assert horizon_to_interval("1h") == "1h"
    assert horizon_to_interval("1d") == "1d"
    assert horizon_to_interval("5d") == "1d"
    assert set(HORIZON_SPECS) == {"1h", "1d", "5d"}


def test_horizon_to_interval_unknown_raises():
    import pytest

    with pytest.raises(ValueError):
        horizon_to_interval("3d")
