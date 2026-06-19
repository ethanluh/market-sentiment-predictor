"""
Point-in-time news archive.

Persists fetched news per day under ``data/raw/news/<TICKER>/<YYYY-MM-DD>.json``
(append-only, per the ``data/raw`` convention) and provides a leakage-safe reader
for backtests: :func:`make_archive_reader` returns a callable suitable for
``features.build_feature_matrix(articles_by_time=...)`` that yields only articles
published at or before the requested ``as_of`` within a lookback window.

FinBERT scores are computed lazily and cached under
``data/processed/news_scores/<TICKER>.json`` keyed by article URL so a walk-forward
backtest does not re-run the model on every fold.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from src.ingestion.news import NewsArticle, to_scored_article
from src.sentiment.aggregation import ScoredArticle

_REPO_ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_ROOT = _REPO_ROOT / "data" / "raw" / "news"
SCORE_CACHE_ROOT = _REPO_ROOT / "data" / "processed" / "news_scores"


def archive_path(ticker: str, day: datetime) -> Path:
    """Return the per-day archive file path for ``ticker``."""
    return ARCHIVE_ROOT / ticker.upper() / f"{day.date().isoformat()}.json"


def _article_to_dict(a: NewsArticle) -> dict:
    return {
        "headline": a.headline,
        "body": a.body,
        "source": a.source,
        "url": a.url,
        "published_at": a.published_at.astimezone(timezone.utc).isoformat(),
    }


def _article_from_dict(d: dict) -> NewsArticle:
    pub = datetime.fromisoformat(d["published_at"])
    if pub.tzinfo is None:
        pub = pub.replace(tzinfo=timezone.utc)
    return NewsArticle(
        headline=d.get("headline", ""),
        body=d.get("body", ""),
        source=d.get("source", ""),
        url=d.get("url", ""),
        published_at=pub.astimezone(timezone.utc),
    )


def write_day(ticker: str, day: datetime, articles: list[NewsArticle]) -> Path | None:
    """
    Write a day's articles to the archive. Append-only: if the day file already
    exists it is left untouched (``data/raw`` is never overwritten) and ``None``
    is returned.
    """
    path = archive_path(ticker, day)
    if path.exists():
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump([_article_to_dict(a) for a in articles], fh, indent=2)
    return path


def read_all(ticker: str) -> list[NewsArticle]:
    """Load every archived article for ``ticker``, sorted by publish time."""
    base = ARCHIVE_ROOT / ticker.upper()
    articles: list[NewsArticle] = []
    if base.exists():
        for path in base.glob("*.json"):
            try:
                with path.open("r", encoding="utf-8") as fh:
                    payload = json.load(fh)
            except (OSError, json.JSONDecodeError):
                continue
            articles.extend(_article_from_dict(d) for d in payload)
    articles.sort(key=lambda a: a.published_at)
    return articles


def _score_cache_path(ticker: str) -> Path:
    return SCORE_CACHE_ROOT / f"{ticker.upper()}.json"


def _load_score_cache(ticker: str) -> dict[str, float]:
    path = _score_cache_path(ticker)
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            return {str(k): float(v) for k, v in json.load(fh).items()}
    except (OSError, json.JSONDecodeError):
        return {}


def _save_score_cache(ticker: str, cache: dict[str, float]) -> None:
    path = _score_cache_path(ticker)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(cache, fh, indent=2)


def make_archive_reader(
    ticker: str, lookback_days: int = 7
) -> Callable[[datetime], list[ScoredArticle]]:
    """
    Return a point-in-time ``articles_by_time(as_of)`` provider.

    The returned callable yields :class:`ScoredArticle` objects for all archived
    articles with ``published_at <= as_of`` within the trailing ``lookback_days``
    window — never future articles — with FinBERT scores resolved from (and
    written back to) the on-disk score cache.
    """
    all_articles = read_all(ticker)
    score_cache = _load_score_cache(ticker)

    def reader(as_of: datetime) -> list[ScoredArticle]:
        if as_of.tzinfo is None:
            as_of = as_of.replace(tzinfo=timezone.utc)
        since = as_of - timedelta(days=lookback_days)
        window = [a for a in all_articles if since <= a.published_at <= as_of]
        if not window:
            return []

        missing = [a for a in window if a.url not in score_cache]
        if missing:
            from src.sentiment.inference import score_batch  # lazy: FinBERT

            for a, s in zip(missing, score_batch([a.text for a in missing])):
                score_cache[a.url] = float(s)
            _save_score_cache(ticker, score_cache)

        return [to_scored_article(a, score_cache[a.url]) for a in window]

    return reader
