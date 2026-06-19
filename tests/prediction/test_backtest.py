"""Tests for prediction.backtest."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.prediction.backtest import (
    BacktestResult,
    interval_coverage,
    pinball_loss,
    walk_forward,
)
from src.prediction.features import FEATURE_NAMES
from src.prediction.model import QuantileReturnModel


def test_pinball_loss_known_value():
    # Under-prediction at q=0.5: loss = 0.5 * |error|.
    y = np.array([1.0, 1.0])
    pred = np.array([0.0, 0.0])
    assert pinball_loss(y, pred, 0.5) == pytest.approx(0.5)


def test_interval_coverage():
    y = np.array([0.0, 1.0, 2.0, 3.0])
    low = np.array([-1.0, -1.0, -1.0, -1.0])
    high = np.array([1.0, 1.0, 1.0, 1.0])
    assert interval_coverage(y, low, high) == pytest.approx(0.5)


def test_interval_coverage_empty_is_nan():
    assert np.isnan(interval_coverage(np.array([]), np.array([]), np.array([])))


def _features_targets(n: int = 320, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="D", tz="UTC")
    X = pd.DataFrame(rng.normal(0, 1, (n, len(FEATURE_NAMES))), columns=FEATURE_NAMES, index=idx)
    y = pd.Series(0.01 * X["sentiment_agg"].to_numpy() + rng.normal(0, 0.01, n), index=idx)
    return X, {"1d": y}


class TestWalkForward:
    def test_result_fields_populated(self):
        X, targets = _features_targets()
        model = QuantileReturnModel(horizons=("1d",), backend="linear")
        result = walk_forward(model, X, targets, train_window=200, test_window=20, step=20)
        assert isinstance(result, BacktestResult)
        assert result.n_predictions > 0
        assert 0.0 <= result.coverage["1d"] <= 1.0
        assert 0.0 <= result.directional_hit_rate["1d"] <= 1.0
        assert len(result.pnl_curve) == result.n_predictions

    def test_too_short_raises(self):
        X, targets = _features_targets(n=50)
        model = QuantileReturnModel(horizons=("1d",), backend="linear")
        with pytest.raises(ValueError):
            walk_forward(model, X, targets, train_window=200, test_window=20)
