"""Tests for prediction.features."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src.graph.categories import NODE_SEC_CORP, NODE_UNINFORMED_RETAIL
from src.prediction.features import (
    FEATURE_NAMES,
    build_feature_matrix,
    build_feature_vector,
    compute_technical_features,
    select_origin_node,
)
from src.prediction.regime import RegimeClassifier
from src.sentiment.aggregation import ScoredArticle


def _make_prices(n: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="D", tz="UTC")
    rets = rng.normal(0.0005, 0.01, n)
    close = 100 * np.exp(np.cumsum(rets))
    df = pd.DataFrame({"close": close, "volume": rng.integers(1e6, 5e6, n)}, index=idx)
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    return df


def _make_sector_returns(n: int = 120, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="D", tz="UTC")
    base = rng.normal(0, 0.01, n)
    return pd.DataFrame(
        {
            "AAPL": base + rng.normal(0, 0.002, n),
            "MSFT": base + rng.normal(0, 0.002, n),
            "GOOG": base + rng.normal(0, 0.002, n),
        },
        index=idx,
    )


def _make_trends(
    n: int = 60, seed: int = 2, freq: str = "D", start: str = "2023-01-01"
) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    return pd.Series(rng.integers(0, 100, n).astype(float), index=idx, name="search_interest")


def _articles(as_of: datetime) -> list[ScoredArticle]:
    return [
        ScoredArticle(0.6, NODE_SEC_CORP, as_of - timedelta(hours=2)),
        ScoredArticle(-0.2, NODE_UNINFORMED_RETAIL, as_of - timedelta(hours=1)),
    ]


class TestTechnicalFeatures:
    def test_no_lookahead(self):
        prices = _make_prices()
        as_of = prices.index[60].to_pydatetime()
        before = compute_technical_features(prices, as_of)
        # Appending future rows must not change features computed at as_of.
        future = prices.copy()
        future.loc[future.index[-1] + timedelta(days=1)] = [200.0, 9e6, 0.5]
        after = compute_technical_features(future, as_of)
        assert before == after

    def test_short_history_returns_nan_not_error(self):
        prices = _make_prices(n=3)
        as_of = prices.index[-1].to_pydatetime()
        feats = compute_technical_features(prices, as_of)
        assert np.isnan(feats["realized_vol_20d"])
        assert np.isnan(feats["momentum_20d"])


class TestOriginSelection:
    def test_empty_falls_back(self):
        assert select_origin_node([]) == NODE_UNINFORMED_RETAIL

    def test_picks_highest_credibility(self):
        as_of = datetime(2023, 3, 1, tzinfo=timezone.utc)
        assert select_origin_node(_articles(as_of)) == NODE_SEC_CORP


class TestFeatureVector:
    def test_to_array_matches_feature_names(self):
        prices = _make_prices()
        sector = _make_sector_returns()
        as_of = prices.index[80].to_pydatetime()
        fv = build_feature_vector("AAPL", as_of, _articles(as_of), prices, sector)
        arr = fv.to_array()
        assert arr.shape == (len(FEATURE_NAMES),)
        assert list(fv.to_series().index) == FEATURE_NAMES

    def test_sentiment_in_range(self):
        prices = _make_prices()
        sector = _make_sector_returns()
        as_of = prices.index[80].to_pydatetime()
        fv = build_feature_vector("AAPL", as_of, _articles(as_of), prices, sector)
        assert -1.0 <= fv.sentiment_agg <= 1.0
        assert fv.estimated_lag_hours >= 0.0
        assert fv.sector_proximity >= 0.0

    def test_empty_articles_zero_sentiment(self):
        prices = _make_prices()
        sector = _make_sector_returns()
        as_of = prices.index[80].to_pydatetime()
        fv = build_feature_vector("AAPL", as_of, [], prices, sector)
        assert fv.sentiment_agg == 0.0

    def test_naive_datetime_coerced(self):
        prices = _make_prices()
        sector = _make_sector_returns()
        naive = datetime(2023, 3, 1)  # no tzinfo
        fv = build_feature_vector("AAPL", naive, [], prices, sector)
        assert fv.as_of.tzinfo is not None

    def test_no_regime_is_unknown(self):
        prices = _make_prices()
        sector = _make_sector_returns()
        as_of = prices.index[80].to_pydatetime()
        fv = build_feature_vector("AAPL", as_of, [], prices, sector)
        assert fv.regime_label == -1

    def test_sector_proximity_no_lookahead(self):
        # Build a sector frame and compute proximity at an early as_of; appending
        # future rows must not change the proximity computed at that as_of.
        prices = _make_prices()
        sector = _make_sector_returns()
        as_of = sector.index[70].to_pydatetime()
        before = build_feature_vector("AAPL", as_of, [], prices, sector).sector_proximity

        future = sector.copy()
        extra_idx = pd.date_range(
            future.index[-1] + pd.Timedelta(days=1), periods=30, freq="D", tz="UTC"
        )
        # Strongly correlated future block that would shift correlations if leaked.
        common = np.linspace(0.0, 0.05, len(extra_idx))
        future = pd.concat(
            [future, pd.DataFrame({c: common for c in future.columns}, index=extra_idx)]
        )
        after = build_feature_vector("AAPL", as_of, [], prices, future).sector_proximity
        assert before == after


class TestSearchInterest:
    def test_inert_zero_when_no_trends(self):
        # The regression guard for the FEATURE_NAMES .dropna() edge case: with no
        # trends series the feature must be 0.0 (neutral), NOT NaN, so the row survives.
        prices = _make_prices()
        sector = _make_sector_returns()
        as_of = prices.index[80].to_pydatetime()
        fv = build_feature_vector("AAPL", as_of, [], prices, sector)  # trends=None default
        assert fv.search_interest_zscore == 0.0

    def test_computed_when_trends_supplied(self):
        prices = _make_prices()
        sector = _make_sector_returns()
        as_of = prices.index[80].to_pydatetime()
        fv = build_feature_vector("AAPL", as_of, [], prices, sector, trends=_make_trends(n=120))
        assert np.isfinite(fv.search_interest_zscore)

    def test_nan_on_insufficient_history(self):
        # Fewer than the 20-point window before as_of -> NaN (row drops downstream,
        # exactly like volume_zscore on short history).
        prices = _make_prices()
        sector = _make_sector_returns()
        as_of = prices.index[80].to_pydatetime()
        fv = build_feature_vector("AAPL", as_of, [], prices, sector, trends=_make_trends(n=5))
        assert np.isnan(fv.search_interest_zscore)

    def test_no_lookahead(self):
        prices = _make_prices()
        sector = _make_sector_returns()
        as_of = prices.index[80].to_pydatetime()
        trends = _make_trends(n=90)
        before = build_feature_vector(
            "AAPL", as_of, [], prices, sector, trends=trends
        ).search_interest_zscore
        # A spiking future block (all after as_of) must not change the value at as_of.
        future_idx = pd.date_range(
            trends.index[-1] + pd.Timedelta(days=1), periods=10, freq="D", tz="UTC"
        )
        future = pd.concat([trends, pd.Series([999.0] * 10, index=future_idx)])
        after = build_feature_vector(
            "AAPL", as_of, [], prices, sector, trends=future
        ).search_interest_zscore
        assert before == after

    def test_finite_on_weekly_grid(self):
        # Daily as_of slicing a weekly-granularity trends index must not crash and
        # yields a finite z-score (no exact-index-membership dependency).
        prices = _make_prices()
        sector = _make_sector_returns()
        as_of = prices.index[80].to_pydatetime()
        trends = _make_trends(n=60, freq="W", start="2022-06-01")
        fv = build_feature_vector("AAPL", as_of, [], prices, sector, trends=trends)
        assert np.isfinite(fv.search_interest_zscore)

    def test_tz_mismatch_degrades_to_zero(self):
        from src.prediction.features import _search_interest_zscore

        naive = pd.Series(
            [1.0] * 30, index=pd.date_range("2023-01-01", periods=30, freq="D")  # tz-naive
        )
        as_of = pd.Timestamp("2023-02-01", tz="UTC")
        assert _search_interest_zscore(naive, as_of) == 0.0


class TestVixAsof:
    def test_regime_resolved_via_asof_on_offset_grid(self):
        # VIX stamped at midnight; as_of grid at 21:00 (close). Exact-membership
        # would miss every row, but asof picks the prior value -> live regime.
        prices = _make_prices(n=60)
        sector = _make_sector_returns(n=60)
        clf = RegimeClassifier(k=3).fit(
            np.concatenate([np.full(20, 13.0), np.full(20, 20.0), np.full(20, 35.0)])
        )
        vix = pd.Series(
            np.linspace(12, 36, 60),
            index=pd.date_range("2023-01-01", periods=60, freq="D", tz="UTC"),
        )
        as_of_index = pd.date_range("2023-01-05 21:00", periods=20, freq="D", tz="UTC")

        matrix = build_feature_matrix(
            "AAPL",
            as_of_index,
            articles_by_time=lambda _a: [],
            prices=prices,
            sector_returns=sector,
            vix=vix,
            regime_classifier=clf,
        )
        # Regime feature is live (not all the -1 "unknown" sentinel).
        assert (matrix["regime_label"] >= 0).any()

    def test_tz_mismatch_warns_and_degrades(self, caplog):
        import logging

        prices = _make_prices(n=40)
        sector = _make_sector_returns(n=40)
        clf = RegimeClassifier(k=3).fit(
            np.concatenate([np.full(14, 13.0), np.full(13, 20.0), np.full(13, 35.0)])
        )
        # tz-naive VIX index cannot be compared to tz-aware as_of -> no hits.
        vix = pd.Series(
            np.linspace(12, 36, 40),
            index=pd.date_range("2023-01-01", periods=40, freq="D"),
        )
        as_of_index = pd.date_range("2023-01-01", periods=40, freq="D", tz="UTC")
        with caplog.at_level(logging.WARNING, logger="prediction.features"):
            matrix = build_feature_matrix(
                "AAPL",
                as_of_index,
                articles_by_time=lambda _a: [],
                prices=prices,
                sector_returns=sector,
                vix=vix,
                regime_classifier=clf,
            )
        assert (matrix["regime_label"] == -1).all()
        assert any(
            "VIX was provided but matched no timestamps" in r.message for r in caplog.records
        )
