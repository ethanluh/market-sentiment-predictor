"""
News ingestion via the Alpaca Market Data news API (Benzinga-sourced, ticker-tagged).

Returns the same :class:`~src.ingestion.news.NewsArticle` type as the other sources, so it is a
drop-in source for the point-in-time archive (``scripts/build_news_archive.py --source alpaca``).
Credentials come from ``ALPACA_PAPER_KEY_ID`` / ``ALPACA_PAPER_SECRET_KEY`` (a paper account's
keys authorise market data). Nothing is hardcoded.

Point-in-time safety: ``published_at`` is the article's ``created_at``. ``updated_at`` can move
after publication (edits, re-tagging), so using it would leak later information into history.
"""

from __future__ import annotations

import html
import os
import re
from datetime import datetime, timezone
from typing import Any

from src.ingestion.news import NewsArticle

_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s+")


def _clean(text: str | None) -> str:
    """Strip HTML tags and entities and collapse whitespace."""
    if not text:
        return ""
    return _SPACE_RE.sub(" ", html.unescape(_TAG_RE.sub(" ", text))).strip()


def _article(item: Any) -> NewsArticle:
    if item.created_at is None:
        raise ValueError(f"Alpaca news item {item.id} has no created_at")
    created = item.created_at
    if created.tzinfo is None:
        raise ValueError(f"Alpaca news item {item.id} has a naive created_at: {created!r}")
    return NewsArticle(
        headline=_clean(item.headline),
        body=_clean(item.content) or _clean(item.summary),
        source=item.source or "",
        url=item.url or "",
        published_at=created.astimezone(timezone.utc),
    )


def fetch_alpaca_news(
    ticker: str,
    start: datetime,
    end: datetime | None = None,
    limit: int | None = None,
    client: Any | None = None,
) -> list[NewsArticle]:
    """
    Fetch news tagged with ``ticker`` whose ``created_at`` is in ``[start, end]``, oldest first.

    The API sorts and may bound by ``updated_at``, so the window is re-checked here against
    ``created_at``: an article created before ``start`` but edited inside the window is dropped.
    The server's ``limit`` cap is applied before this check, so leave ``limit`` unset for a backfill.

    ``start`` is required: the API defaults to the beginning of the current day, which would
    silently return almost nothing. The SDK handles pagination. ``client`` is a test seam.
    """
    if start is None:
        raise ValueError("start is required; the API's default window is only today")
    api: Any = client
    if api is None:
        key, secret = os.environ.get("ALPACA_PAPER_KEY_ID"), os.environ.get(
            "ALPACA_PAPER_SECRET_KEY"
        )
        if not key or not secret:
            raise RuntimeError("ALPACA_PAPER_KEY_ID / ALPACA_PAPER_SECRET_KEY are not set")
        from alpaca.data.historical.news import NewsClient  # lazy: network SDK

        api = NewsClient(api_key=key, secret_key=secret)

    from alpaca.data.requests import NewsRequest

    request = NewsRequest(
        symbols=ticker.upper(),
        start=start,
        end=end,
        sort="asc",
        limit=limit,
        include_content=True,
    )
    news = api.get_news(request).data["news"]
    stray = [n for n in news if ticker.upper() not in (n.symbols or [])]
    if stray:
        raise ValueError(
            f"{len(stray)} items lack the {ticker.upper()} tag (e.g. symbols={stray[0].symbols}); "
            "the ticker may be spelled differently by the API"
        )
    articles = [_article(n) for n in news]
    in_window = [
        a
        for a in articles
        if a.published_at >= start.astimezone(timezone.utc)
        and (end is None or a.published_at <= end.astimezone(timezone.utc))
    ]
    return sorted(in_window, key=lambda a: a.published_at)
