"""Tests for the ingestion layer — external clients are mocked, no network."""

from __future__ import annotations

import sys
import types
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from src.graph.categories import (
    NODE_FIN_PRESS,
    NODE_INFORMED_RETAIL,
    NODE_UNINFORMED_RETAIL,
)
from src.ingestion import cache
from src.ingestion.news import (
    NewsArticle,
    map_source_to_category,
    to_scored_article,
)
from src.ingestion.price import to_log_returns
from src.ingestion.social import subreddit_category


class TestCache:
    def test_roundtrip(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cache, "CACHE_ROOT", tmp_path)
        cache.write_cache("news", "AAPL", "2023-01-01", {"a": 1})
        assert cache.read_cache("news", "AAPL", "2023-01-01") == {"a": 1}

    def test_miss_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cache, "CACHE_ROOT", tmp_path)
        assert cache.read_cache("news", "ZZZZ", "2099-01-01") is None


class TestPrice:
    def test_log_returns_math(self):
        close = pd.Series([100.0, 110.0, 99.0])
        lr = to_log_returns(close)
        assert np.isnan(lr.iloc[0])
        assert lr.iloc[1] == pytest.approx(np.log(110 / 100))

    def test_fetch_prices_parses_yfinance(self, monkeypatch):
        idx = pd.date_range("2023-01-01", periods=3, freq="D")  # naive
        fake = pd.DataFrame({"Close": [100.0, 101.0, 102.0], "Volume": [1e6, 2e6, 3e6]}, index=idx)
        fake_yf = types.SimpleNamespace(download=lambda *a, **k: fake)
        monkeypatch.setitem(sys.modules, "yfinance", fake_yf)

        from src.ingestion.price import fetch_prices

        df = fetch_prices("AAPL")
        assert df.index.tz is not None  # coerced to UTC
        assert list(df.columns) == ["close", "volume", "log_return"]
        assert df["log_return"].iloc[1] == pytest.approx(np.log(101 / 100))


class TestNews:
    def test_source_mapping(self):
        assert map_source_to_category("Reuters") == "reuters"
        assert map_source_to_category("The Wall Street Journal") == "wsj"
        assert map_source_to_category("Random Finance Blog") == NODE_INFORMED_RETAIL
        assert map_source_to_category("Some Outlet") == NODE_FIN_PRESS
        assert map_source_to_category("") == NODE_UNINFORMED_RETAIL

    def test_to_scored_article(self):
        art = NewsArticle(
            headline="h",
            body="b",
            source="Reuters",
            url="http://x",
            published_at=datetime(2023, 1, 1, tzinfo=timezone.utc),
        )
        scored = to_scored_article(art, 0.5)
        assert scored.score == 0.5
        assert scored.source_category == "reuters"

    def test_fetch_news_parses_payload(self, monkeypatch):
        payload = {
            "articles": [
                {
                    "title": "Earnings beat",
                    "description": "strong quarter",
                    "source": {"name": "Reuters"},
                    "url": "http://x",
                    "publishedAt": "2023-01-02T10:00:00Z",
                }
            ]
        }

        class _Client:
            def __init__(self, api_key):
                pass

            def get_everything(self, **kwargs):
                return payload

        fake_mod = types.SimpleNamespace(NewsApiClient=_Client)
        monkeypatch.setitem(sys.modules, "newsapi", fake_mod)
        monkeypatch.setenv("NEWS_API_KEY", "dummy")

        from src.ingestion.news import fetch_news

        articles = fetch_news("AAPL")
        assert len(articles) == 1
        assert articles[0].source == "Reuters"
        assert articles[0].published_at.tzinfo is not None

    def test_fetch_news_requires_key(self, monkeypatch):
        monkeypatch.delenv("NEWS_API_KEY", raising=False)
        from src.ingestion.news import fetch_news

        with pytest.raises(RuntimeError):
            fetch_news("AAPL")


class TestFilings:
    def test_parse_acceptance_datetime(self, tmp_path):
        from src.ingestion.filings import _parse_filing_date

        entry = tmp_path / "0000320193-23-000006"
        entry.mkdir()
        (entry / "full-submission.txt").write_text(
            "<SEC-DOCUMENT>...\n<ACCEPTANCE-DATETIME>20230115083000\n"
            "FILED AS OF DATE:\t\t20230115\n"
        )
        dt = _parse_filing_date(entry)
        assert dt is not None
        assert (dt.year, dt.month, dt.day, dt.hour) == (2023, 1, 15, 8)
        assert dt.tzinfo is not None

    def test_parse_filed_as_of_date_fallback(self, tmp_path):
        from src.ingestion.filings import _parse_filing_date

        entry = tmp_path / "acc"
        entry.mkdir()
        (entry / "full-submission.txt").write_text("FILED AS OF DATE:    20221231\n")
        dt = _parse_filing_date(entry)
        assert dt is not None
        assert (dt.year, dt.month, dt.day) == (2022, 12, 31)

    def test_parse_returns_none_when_absent(self, tmp_path):
        from src.ingestion.filings import _parse_filing_date

        entry = tmp_path / "empty"
        entry.mkdir()
        assert _parse_filing_date(entry) is None


class TestSocial:
    def test_subreddit_category(self):
        assert subreddit_category("investing") == NODE_INFORMED_RETAIL
        assert subreddit_category("wallstreetbets") == NODE_UNINFORMED_RETAIL
