"""Tests for prediction.backtest."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.prediction.backtest import (
    BacktestResult,
    _sharpe,
    interval_coverage,
    main,
    pinball_loss,
    run_backtest,
    walk_forward,
)
from src.prediction.features import FEATURE_NAMES
from src.prediction.model import QuantilePrediction, QuantileReturnModel


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


class _FakeModel:
    """Minimal QuantileReturnModel stand-in for exercising walk_forward folds.

    ``available_horizon=None`` mimics a fit that produced no usable horizon (so
    every fold is skipped); otherwise ``predict`` returns ``preds`` for it.
    """

    def __init__(self, *, available_horizon: str | None, preds=None):
        self.available_horizon = available_horizon
        self.preds = preds or []
        self._available: dict[str, object] = {}

    def fit(self, X, targets):
        if self.available_horizon is not None:
            self._available[self.available_horizon] = object()
        return self

    def predict(self, X):
        if self.available_horizon is None:
            return {}
        return {self.available_horizon: list(self.preds)}


class TestSharpe:
    def test_too_few_points_is_zero(self):
        assert _sharpe(pd.Series([0.01])) == 0.0

    def test_zero_variance_is_zero(self):
        assert _sharpe(pd.Series([0.01, 0.01, 0.01])) == 0.0

    def test_positive_drift_is_positive(self):
        assert _sharpe(pd.Series([0.01, 0.02, 0.015, 0.012])) > 0.0


class TestWalkForwardEdgeCases:
    def test_horizon_never_available_returns_empty(self):
        # A model that fits but exposes no horizon -> every fold skipped.
        X, targets = _features_targets()
        model = _FakeModel(available_horizon=None)
        result = walk_forward(model, X, targets, train_window=200, test_window=20, step=20)
        assert result.n_predictions == 0
        assert result.pinball_loss == {}
        assert result.pnl_curve.empty

    def test_nan_actuals_are_skipped(self):
        X, targets = _features_targets()
        y = targets["1d"].copy()
        y.iloc[200] = np.nan  # first evaluated position of the first fold
        preds = [QuantilePrediction(-0.01, 0.0, 0.01)] * 20
        model = _FakeModel(available_horizon="1d", preds=preds)
        result = walk_forward(model, X, {"1d": y}, train_window=200, test_window=20, step=20)
        # 6 folds (starts 0,20,...,100) x 20 preds = 120, minus the 1 NaN actual.
        assert result.n_predictions == 119

    def test_baseline_failure_falls_back_to_zero(self):
        class _ExplodingBaseline:
            def fit(self, returns):
                raise RuntimeError("baseline blew up")

            def predict(self, horizon):  # pragma: no cover - never reached
                raise AssertionError("predict should not be called after fit fails")

        X, targets = _features_targets()
        returns = targets["1d"]
        model = QuantileReturnModel(horizons=("1d",), backend="linear")
        result = walk_forward(
            model,
            X,
            targets,
            train_window=200,
            test_window=20,
            step=20,
            returns=returns,
            baselines={"boom": _ExplodingBaseline()},
        )
        # The failing baseline still appears, scored against its zero forecasts.
        assert "boom" in result.baseline_pinball
        assert all(np.isfinite(v) for v in result.baseline_pinball["boom"].values())


class TestMainCLI:
    def _result(self, with_baselines: bool) -> BacktestResult:
        r = BacktestResult()
        r.n_predictions = 5
        r.pinball_loss = {"1d_p10": 0.1, "1d_p50": 0.2, "1d_p90": 0.1}
        r.coverage = {"1d": 0.8}
        r.directional_hit_rate = {"1d": 0.6}
        r.sharpe = 1.23
        if with_baselines:
            r.baseline_pinball = {"momentum": {"p10": 0.2, "p50": 0.3, "p90": 0.2}}
        return r

    def test_main_prints_summary(self, monkeypatch, capsys):
        import src.prediction.backtest as bt

        monkeypatch.setattr(bt, "run_backtest", lambda *a, **k: self._result(False))
        rc = main(["--ticker", "AAPL", "--start", "2024-01-01"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Backtest AAPL" in out
        assert "sharpe" in out

    def test_main_prints_baseline_comparison(self, monkeypatch, capsys):
        import src.prediction.backtest as bt

        monkeypatch.setattr(bt, "run_backtest", lambda *a, **k: self._result(True))
        rc = main(["--ticker", "AAPL", "--start", "2024-01-01", "--horizon", "1d"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "model vs baselines" in out
        assert "momentum" in out

    def test_main_threads_source_flags(self, monkeypatch):
        import src.prediction.backtest as bt

        captured = {}

        def _capture(*a, **k):
            captured.update(k)
            return self._result(False)

        monkeypatch.setattr(bt, "run_backtest", _capture)
        rc = main(
            [
                "--ticker",
                "AAPL",
                "--start",
                "2024-01-01",
                "--peers",
                "MSFT",
                "GOOG",
                "--use-trends",
                "--use-insider-flow",
                "--use-vix",
                "--use-news-archive",
            ]
        )
        assert rc == 0
        assert captured["peers"] == ["MSFT", "GOOG"]
        assert captured["use_trends"] is True
        assert captured["use_insider_flow"] is True
        assert captured["use_vix"] is True
        assert captured["use_news_archive"] is True


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

    def test_peers_wire_sector_returns_matrix(self, monkeypatch):
        import src.ingestion.price as price_mod

        prices = _synthetic_prices()
        monkeypatch.setattr(price_mod, "fetch_prices", lambda *a, **k: prices)

        captured = {}

        def _fake_returns_matrix(tickers, start=None, end=None, interval="1d"):
            captured["tickers"] = list(tickers)
            return pd.DataFrame({t: prices["log_return"] for t in tickers}, index=prices.index)

        monkeypatch.setattr(price_mod, "fetch_returns_matrix", _fake_returns_matrix)

        result = run_backtest(
            "AAPL",
            start="2024-01-01",
            horizon="1d",
            backend="linear",
            train_window=200,
            test_window=20,
            peers=["MSFT", "GOOG"],
        )
        assert captured["tickers"] == ["AAPL", "MSFT", "GOOG"]
        assert result.n_predictions > 0

    def test_vix_wires_regime_classifier(self, monkeypatch):
        import src.ingestion.price as price_mod

        prices = _synthetic_prices()
        monkeypatch.setattr(price_mod, "fetch_prices", lambda *a, **k: prices)

        rng = np.random.default_rng(1)
        vix = pd.Series(15 + rng.normal(0, 3, len(prices)).cumsum() % 20, index=prices.index)

        result = run_backtest(
            "AAPL",
            start="2024-01-01",
            horizon="1d",
            backend="linear",
            train_window=200,
            test_window=20,
            vix=vix,
        )
        assert result.n_predictions > 0

    def test_trends_wires_search_interest(self, monkeypatch):
        import src.ingestion.price as price_mod

        prices = _synthetic_prices()
        monkeypatch.setattr(price_mod, "fetch_prices", lambda *a, **k: prices)

        rng = np.random.default_rng(2)
        trends = pd.Series(rng.integers(0, 100, len(prices)).astype(float), index=prices.index)

        result = run_backtest(
            "AAPL",
            start="2024-01-01",
            horizon="1d",
            backend="linear",
            train_window=200,
            test_window=20,
            trends=trends,
        )
        assert result.n_predictions > 0

    def test_form4_wires_insider_flow(self, monkeypatch):
        import src.ingestion.price as price_mod

        prices = _synthetic_prices()
        monkeypatch.setattr(price_mod, "fetch_prices", lambda *a, **k: prices)

        rng = np.random.default_rng(3)
        signed = rng.integers(-5000, 5000, len(prices)).astype(float)
        insider_flow = pd.Series(signed, index=prices.index)

        result = run_backtest(
            "AAPL",
            start="2024-01-01",
            horizon="1d",
            backend="linear",
            train_window=200,
            test_window=20,
            insider_flow=insider_flow,
        )
        assert result.n_predictions > 0

    def test_use_trends_flag_fetches_series(self, monkeypatch):
        import src.ingestion.price as price_mod
        import src.ingestion.trends as trends_mod

        prices = _synthetic_prices()
        monkeypatch.setattr(price_mod, "fetch_prices", lambda *a, **k: prices)
        calls = {"n": 0}

        def _fake_trends(ticker, start=None, end=None, **k):
            calls["n"] += 1
            return pd.Series(
                np.linspace(10, 90, len(prices)), index=prices.index, name="search_interest"
            )

        monkeypatch.setattr(trends_mod, "fetch_trends", _fake_trends)
        result = run_backtest(
            "AAPL",
            start="2024-01-01",
            backend="linear",
            train_window=200,
            test_window=20,
            use_trends=True,
        )
        assert calls["n"] == 1
        assert result.n_predictions > 0

    def test_use_insider_flow_flag_fetches_series(self, monkeypatch):
        import src.ingestion.form4 as form4_mod
        import src.ingestion.price as price_mod

        prices = _synthetic_prices()
        monkeypatch.setattr(price_mod, "fetch_prices", lambda *a, **k: prices)
        calls = {"n": 0}

        def _fake_form4(ticker, after=None, limit=50):
            calls["n"] += 1
            rng = np.random.default_rng(4)
            return pd.Series(
                rng.integers(-5000, 5000, len(prices)).astype(float),
                index=prices.index,
                name="insider_net_shares",
            )

        monkeypatch.setattr(form4_mod, "fetch_form4", _fake_form4)
        result = run_backtest(
            "AAPL",
            start="2024-01-01",
            backend="linear",
            train_window=200,
            test_window=20,
            use_insider_flow=True,
        )
        assert calls["n"] == 1
        assert result.n_predictions > 0

    def test_use_vix_flag_fetches_vix_via_prices(self, monkeypatch):
        # use_vix re-uses fetch_prices("^VIX"); the patched fetch_prices serves it.
        import src.ingestion.price as price_mod

        prices = _synthetic_prices()
        monkeypatch.setattr(price_mod, "fetch_prices", lambda *a, **k: prices)
        result = run_backtest(
            "AAPL",
            start="2024-01-01",
            backend="linear",
            train_window=200,
            test_window=20,
            use_vix=True,
        )
        assert result.n_predictions > 0

    def test_failed_source_fetch_degrades(self, monkeypatch):
        # A raising trends fetch must neutralize only that feature, not abort.
        import src.ingestion.price as price_mod
        import src.ingestion.trends as trends_mod

        prices = _synthetic_prices()
        monkeypatch.setattr(price_mod, "fetch_prices", lambda *a, **k: prices)

        def _boom(*a, **k):
            raise RuntimeError("rate limited")

        monkeypatch.setattr(trends_mod, "fetch_trends", _boom)
        result = run_backtest(
            "AAPL",
            start="2024-01-01",
            backend="linear",
            train_window=200,
            test_window=20,
            use_trends=True,
        )
        assert result.n_predictions > 0
