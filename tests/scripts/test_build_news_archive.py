"""Tests for scripts/build_news_archive.py (ingestion mocked, no network)."""

from __future__ import annotations

import importlib.util
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.ingestion.news import NewsArticle

_REPO = Path(__file__).resolve().parents[2]


def _load_builder():
    spec = importlib.util.spec_from_file_location(
        "build_news_archive", _REPO / "scripts" / "build_news_archive.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _article(day: str) -> NewsArticle:
    return NewsArticle(
        headline="h",
        body="b",
        source="Reuters",
        url=f"https://example.com/{day}",
        published_at=datetime.fromisoformat(day).replace(tzinfo=timezone.utc),
    )


def test_dispatch_alphavantage(monkeypatch):
    import src.ingestion.alpha_vantage_news as av

    captured: dict = {}

    def _fake(ticker, time_from=None, time_to=None, limit=1000):
        captured["ticker"] = ticker
        captured["time_from"] = time_from
        return [_article("2023-01-02")]

    monkeypatch.setattr(av, "fetch_alpha_vantage_news", _fake)
    mod = _load_builder()
    start = datetime(2023, 1, 1, tzinfo=timezone.utc)
    out = mod._fetch_articles("alphavantage", "AAPL", start, None)
    assert len(out) == 1
    assert captured["ticker"] == "AAPL"
    assert captured["time_from"] == start


def test_dispatch_newsapi(monkeypatch):
    import src.ingestion.news as news_mod

    monkeypatch.setattr(news_mod, "fetch_news", lambda ticker, since=None: [_article("2023-01-03")])
    mod = _load_builder()
    out = mod._fetch_articles("newsapi", "AAPL", None, None)
    assert len(out) == 1


def test_dispatch_unknown_source_raises():
    mod = _load_builder()
    with pytest.raises(ValueError, match="Unknown news source"):
        mod._fetch_articles("bogus", "AAPL", None, None)


def test_build_groups_by_day_and_writes(monkeypatch, tmp_path):
    import src.ingestion.news_archive as archive_mod

    monkeypatch.setattr(archive_mod, "ARCHIVE_ROOT", tmp_path)

    mod = _load_builder()
    articles = [_article("2023-01-02"), _article("2023-01-02"), _article("2023-01-03")]
    monkeypatch.setattr(mod, "_fetch_articles", lambda *a, **k: articles)

    written = mod.build("AAPL", source="newsapi")
    # Two distinct days written.
    assert written == 2
    assert (tmp_path / "AAPL" / "2023-01-02.json").exists()
    assert (tmp_path / "AAPL" / "2023-01-03.json").exists()
