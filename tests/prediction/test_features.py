"""Tests for prediction.features."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src.graph.categories import NODE_SEC_CORP, NODE_UNINFORMED_RETAIL
from src.prediction.features import (
    FEATURE_NAMES,
    build_feature_vector,
    compute_technical_features,
    select_origin_node,
)
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
