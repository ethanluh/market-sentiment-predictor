"""Tests for the point-in-time news archive (no network, FinBERT mocked)."""

from __future__ import annotations

import sys
import types
from datetime import datetime, timedelta, timezone

import pytest

import src.ingestion.news_archive as na
from src.ingestion.news import NewsArticle


def _stub_finbert(monkeypatch, score_fn):
    """Inject a fake src.sentiment.inference so FinBERT/torch are never imported."""
    fake = types.ModuleType("src.sentiment.inference")
    fake.score_batch = score_fn  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "src.sentiment.inference", fake)


def _article(headline: str, days_ago: float, base: datetime, url: str) -> NewsArticle:
    return NewsArticle(
        headline=headline,
        body="body",
        source="Reuters",
        url=url,
        published_at=base - timedelta(days=days_ago),
    )


@pytest.fixture
def archive(tmp_path, monkeypatch):
    monkeypatch.setattr(na, "ARCHIVE_ROOT", tmp_path / "raw")
    monkeypatch.setattr(na, "SCORE_CACHE_ROOT", tmp_path / "scores")
    return tmp_path


class TestArchiveIO:
    def test_write_then_read(self, archive):
        day = datetime(2024, 1, 2, tzinfo=timezone.utc)
        arts = [_article("h", 0, day, "u1")]
        path = na.write_day("AAPL", day, arts)
        assert path is not None
        loaded = na.read_all("AAPL")
        assert len(loaded) == 1
        assert loaded[0].published_at.tzinfo is not None

    def test_write_is_append_only(self, archive):
        day = datetime(2024, 1, 2, tzinfo=timezone.utc)
        assert na.write_day("AAPL", day, [_article("first", 0, day, "u1")]) is not None
        # Second write for the same day must not overwrite.
        assert na.write_day("AAPL", day, [_article("second", 0, day, "u2")]) is None
        loaded = na.read_all("AAPL")
        assert [a.headline for a in loaded] == ["first"]


class TestReader:
    def test_point_in_time_no_future_leak(self, archive, monkeypatch):
        base = datetime(2024, 1, 10, tzinfo=timezone.utc)
        day = datetime(2024, 1, 10, tzinfo=timezone.utc)
        arts = [
            _article("past_in_window", 2, base, "u_past"),
            _article("today", 0, base, "u_today"),
            _article("too_old", 30, base, "u_old"),
            _article("future", -2, base, "u_future"),
        ]
        na.write_day("AAPL", day, arts)

        _stub_finbert(monkeypatch, lambda texts: [0.5] * len(texts))

        reader = na.make_archive_reader("AAPL", lookback_days=7)
        out = reader(base)
        cats = {a.score for a in out}
        # Only the two in-window, non-future articles are returned.
        assert len(out) == 2
        assert cats == {0.5}

    def test_scores_are_cached(self, archive, monkeypatch):
        base = datetime(2024, 1, 10, tzinfo=timezone.utc)
        na.write_day("AAPL", base, [_article("a", 1, base, "u1")])

        calls = {"n": 0}

        def _fake(texts):
            calls["n"] += 1
            return [0.3] * len(texts)

        _stub_finbert(monkeypatch, _fake)
        na.make_archive_reader("AAPL")(base)
        # A fresh reader reuses the on-disk cache; the scorer is not called again.
        na.make_archive_reader("AAPL")(base)
        assert calls["n"] == 1
