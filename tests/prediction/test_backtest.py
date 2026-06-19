"""Tests for prediction.backtest."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.prediction.backtest import (
    BacktestResult,
    interval_coverage,
    pinball_loss,
    run_backtest,
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

    def test_baselines_evaluated_when_returns_given(self):
        X, targets = _features_targets()
        returns = targets["1d"]  # a log-return-like series aligned to X.index
        model = QuantileReturnModel(horizons=("1d",), backend="linear")
        result = walk_forward(
            model, X, targets, train_window=200, test_window=20, step=20, returns=returns
        )
        assert set(result.baseline_pinball) == {"momentum", "arima", "garch"}
        for losses in result.baseline_pinball.values():
            assert set(losses) == {"p10", "p50", "p90"}
            assert all(np.isfinite(v) for v in losses.values())

    def test_no_baselines_without_returns(self):
        X, targets = _features_targets()
        model = QuantileReturnModel(horizons=("1d",), backend="linear")
        result = walk_forward(model, X, targets, train_window=200, test_window=20, step=20)
        assert result.baseline_pinball == {}


def _synthetic_prices(n: int = 300, freq: str = "D", seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq=freq, tz="UTC")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    df = pd.DataFrame({"close": close, "volume": rng.integers(1e6, 5e6, n)}, index=idx)
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    return df


class TestRunBacktestInterval:
    def test_intraday_horizon_uses_hourly_interval(self, monkeypatch):
        captured = {}

        def _fake_fetch(ticker, start=None, end=None, interval="1d"):
            captured["interval"] = interval
            freq = "h" if interval == "1h" else "D"
            return _synthetic_prices(freq=freq)

        import src.ingestion.price as price_mod

        monkeypatch.setattr(price_mod, "fetch_prices", _fake_fetch)

        result = run_backtest(
            "AAPL",
            start="2024-01-01",
            horizon="1h",
            backend="linear",
            train_window=200,
            test_window=20,
        )
        assert captured["interval"] == "1h"
        assert result.n_predictions > 0

    def test_news_archive_reader_is_wired(self, monkeypatch):
        import src.ingestion.news_archive as na
        import src.ingestion.price as price_mod

        monkeypatch.setattr(price_mod, "fetch_prices", lambda *a, **k: _synthetic_prices())

        calls = {"made": 0, "read": 0}

        def _fake_make_reader(ticker, lookback_days=7):
            calls["made"] += 1

            def _reader(as_of):
                calls["read"] += 1
                return []

            return _reader

        monkeypatch.setattr(na, "make_archive_reader", _fake_make_reader)

        run_backtest(
            "AAPL",
            start="2024-01-01",
            horizon="1d",
            backend="linear",
            train_window=200,
            test_window=20,
            use_news_archive=True,
        )
        assert calls["made"] == 1
        assert calls["read"] > 0
