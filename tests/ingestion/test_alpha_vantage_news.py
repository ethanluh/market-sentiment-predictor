"""Mocked tests for the Alpha Vantage news client (no real network)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest


def _feed_item(title="AAPL beats", time_published="20230315T143000", source="Benzinga"):
    return {
        "title": title,
        "summary": "Apple beat estimates.",
        "source": source,
        "url": "https://example.com/a",
        "time_published": time_published,
        "overall_sentiment_score": 0.42,  # present but intentionally ignored
        "ticker_sentiment": [{"ticker": "AAPL", "ticker_sentiment_score": "0.5"}],
    }


class TestAlphaVantageNews:
    def test_requires_api_key(self, monkeypatch):
        monkeypatch.delenv("ALPHAVANTAGE_API_KEY", raising=False)
        from src.ingestion.alpha_vantage_news import fetch_alpha_vantage_news

        with pytest.raises(RuntimeError, match="ALPHAVANTAGE_API_KEY"):
            fetch_alpha_vantage_news("AAPL")

    def test_parses_feed_into_articles(self, monkeypatch):
        import src.ingestion.alpha_vantage_news as av

        monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "demo")
        monkeypatch.setattr(
            av, "_get_json", lambda url, params: {"items": "1", "feed": [_feed_item()]}
        )

        articles = av.fetch_alpha_vantage_news("AAPL")
        assert len(articles) == 1
        a = articles[0]
        assert a.headline == "AAPL beats"
        assert a.body == "Apple beat estimates."
        assert a.source == "Benzinga"
        # time_published parsed as tz-aware UTC.
        assert a.published_at == datetime(2023, 3, 15, 14, 30, tzinfo=timezone.utc)

    def test_sends_uppercased_ticker_and_window(self, monkeypatch):
        import src.ingestion.alpha_vantage_news as av

        captured: dict = {}

        def _fake_get(url, params):
            captured["url"] = url
            captured["params"] = params
            return {"feed": []}

        monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "demo")
        monkeypatch.setattr(av, "_get_json", _fake_get)

        av.fetch_alpha_vantage_news(
            "aapl",
            time_from=datetime(2022, 1, 1, tzinfo=timezone.utc),
            time_to=datetime(2022, 6, 1, 12, 30, tzinfo=timezone.utc),
        )
        p = captured["params"]
        assert p["function"] == "NEWS_SENTIMENT"
        assert p["tickers"] == "AAPL"
        assert p["time_from"] == "20220101T0000"
        assert p["time_to"] == "20220601T1230"
        assert "apikey" in p and p["apikey"] == "demo"

    def test_rate_limit_note_raises(self, monkeypatch):
        import src.ingestion.alpha_vantage_news as av

        monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "demo")
        monkeypatch.setattr(
            av, "_get_json", lambda url, params: {"Note": "call frequency exceeded"}
        )

        with pytest.raises(RuntimeError, match="Alpha Vantage API error"):
            av.fetch_alpha_vantage_news("AAPL")

    def test_missing_time_published_falls_back_to_now(self, monkeypatch):
        import src.ingestion.alpha_vantage_news as av

        monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "demo")
        item = _feed_item()
        del item["time_published"]
        monkeypatch.setattr(av, "_get_json", lambda url, params: {"feed": [item]})

        before = datetime.now(timezone.utc)
        articles = av.fetch_alpha_vantage_news("AAPL")
        assert articles[0].published_at >= before

    def test_empty_feed_returns_empty(self, monkeypatch):
        import src.ingestion.alpha_vantage_news as av

        monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "demo")
        monkeypatch.setattr(av, "_get_json", lambda url, params: {})
        assert av.fetch_alpha_vantage_news("AAPL") == []
