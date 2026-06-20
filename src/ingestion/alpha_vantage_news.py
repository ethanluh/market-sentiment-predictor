"""
News ingestion via Alpha Vantage ``NEWS_SENTIMENT``.

An alternative to NewsAPI (:mod:`src.ingestion.news`) with deeper free historical
coverage (articles back to roughly 2022) and native ticker tagging. It returns
the same :class:`~src.ingestion.news.NewsArticle` type, so it is a drop-in source
for the point-in-time archive (``scripts/build_news_archive.py``). The API key is
read from the ``ALPHAVANTAGE_API_KEY`` environment variable only (never
hardcoded).

Alpha Vantage ships its own sentiment scores, but they are intentionally ignored
here: the pipeline scores every article with FinBERT
(:mod:`src.sentiment.inference`) for consistency, so this client only extracts
article text/metadata.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from urllib.parse import urlencode

from src.ingestion.news import NewsArticle

AV_QUERY_URL = "https://www.alphavantage.co/query"

# Alpha Vantage uses ``YYYYMMDDTHHMM`` for the request time_from/time_to window
# and ``YYYYMMDDTHHMMSS`` for the ``time_published`` field in the response.
_AV_REQUEST_TIME_FMT = "%Y%m%dT%H%M"
_AV_PUBLISHED_FMT = "%Y%m%dT%H%M%S"

# Rate-limit / error responses carry one of these keys instead of ``feed``.
_AV_ERROR_KEYS = ("Error Message", "Note", "Information")


def _parse_av_published(value: str | None) -> datetime:
    """
    Parse an Alpha Vantage ``time_published`` (``YYYYMMDDTHHMMSS``) as UTC.

    Falls back to "now" (UTC) for a missing/unparseable value rather than
    dropping the whole batch.
    """
    if not value:
        return datetime.now(timezone.utc)
    try:
        return datetime.strptime(value, _AV_PUBLISHED_FMT).replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return datetime.now(timezone.utc)


def _article_from_av_item(item: dict) -> NewsArticle:
    """Build a :class:`NewsArticle` from a single Alpha Vantage ``feed[]`` entry."""
    return NewsArticle(
        headline=item.get("title") or "",
        body=item.get("summary") or "",
        source=item.get("source") or "",
        url=item.get("url") or "",
        published_at=_parse_av_published(item.get("time_published")),
    )


def _get_json(url: str, params: dict[str, str]) -> dict:
    """
    GET ``url?params`` and parse the JSON body (stdlib only, no extra deps).

    Isolated as a tiny seam so tests can monkeypatch it without real network.
    """
    from urllib.request import urlopen  # lazy: network

    with urlopen(f"{url}?{urlencode(params)}", timeout=30) as resp:  # nosec B310 - fixed https host
        return json.loads(resp.read().decode("utf-8"))


def fetch_alpha_vantage_news(
    ticker: str,
    time_from: datetime | None = None,
    time_to: datetime | None = None,
    limit: int = 1000,
) -> list[NewsArticle]:
    """
    Fetch ticker-tagged news for ``ticker`` from Alpha Vantage ``NEWS_SENTIMENT``.

    Requires the ``ALPHAVANTAGE_API_KEY`` environment variable. ``time_from`` /
    ``time_to`` bound the publish window (both coerced to UTC); ``limit`` caps the
    number of articles (Alpha Vantage allows up to 1000). Returns a list of
    :class:`NewsArticle`, oldest first.
    """
    api_key = os.environ.get("ALPHAVANTAGE_API_KEY")
    if not api_key:
        raise RuntimeError("ALPHAVANTAGE_API_KEY environment variable is not set")

    params: dict[str, str] = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker.upper(),
        "apikey": api_key,
        "limit": str(limit),
        "sort": "EARLIEST",
    }
    if time_from is not None:
        params["time_from"] = time_from.astimezone(timezone.utc).strftime(_AV_REQUEST_TIME_FMT)
    if time_to is not None:
        params["time_to"] = time_to.astimezone(timezone.utc).strftime(_AV_REQUEST_TIME_FMT)

    payload = _get_json(AV_QUERY_URL, params)
    if not isinstance(payload, dict):
        return []
    for key in _AV_ERROR_KEYS:
        if key in payload:
            raise RuntimeError(f"Alpha Vantage API error ({key}): {payload[key]}")

    feed = payload.get("feed", [])
    return [_article_from_av_item(item) for item in feed]
