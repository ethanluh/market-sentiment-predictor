"""
News ingestion via NewsAPI.

Maps outlets to the canonical information-flow node categories defined in
``src/graph/categories.py`` so that downstream sentiment aggregation can apply
the right credibility weight and recency decay. API key is read from the
``NEWS_API_KEY`` environment variable only (never hardcoded).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone

from src.graph.categories import (
    NODE_FIN_PRESS,
    NODE_INFORMED_RETAIL,
    NODE_UNINFORMED_RETAIL,
)
from src.sentiment.aggregation import ScoredArticle

# Map a NewsAPI source id/name to the credibility key used by aggregation.
# Keys are matched case-insensitively as substrings of the source name.
SOURCE_CATEGORY: dict[str, str] = {
    "reuters": "reuters",
    "bloomberg": "bloomberg",
    "the-wall-street-journal": "wsj",
    "wall street journal": "wsj",
    "wsj": "wsj",
    "financial-times": "ft",
    "financial times": "ft",
    "ft": "ft",
    "benzinga": "benzinga",
    "seeking alpha": "seeking_alpha",
    "seekingalpha": "seeking_alpha",
}


@dataclass
class NewsArticle:
    headline: str
    body: str
    source: str
    url: str
    published_at: datetime  # tz-aware UTC

    @property
    def text(self) -> str:
        """Concatenated headline + body for sentiment scoring."""
        return f"{self.headline}. {self.body}".strip()


def map_source_to_category(source: str) -> str:
    """
    Map a NewsAPI source name to a credibility category key.

    Recognised outlets map to their specific key (``reuters``, ``wsj``, ...).
    Anything else falls back to the generic financial-press category.
    """
    if not source:
        return NODE_UNINFORMED_RETAIL
    low = source.strip().lower()
    for needle, category in SOURCE_CATEGORY.items():
        if needle in low:
            return category
    # Blogs / unknown outlets: treat as low-credibility financial commentary.
    if "blog" in low:
        return NODE_INFORMED_RETAIL
    return NODE_FIN_PRESS


def _parse_published(value: str | None) -> datetime:
    """
    Parse a NewsAPI ISO timestamp into a tz-aware UTC datetime.

    NewsAPI may return ``publishedAt: null`` or omit the field; rather than
    crashing the whole batch, fall back to "now" (UTC) for such articles.
    """
    if not value:
        return datetime.now(timezone.utc)
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _article_from_payload(item: dict) -> NewsArticle:
    """Build a ``NewsArticle`` from a single NewsAPI ``articles[]`` entry."""
    source = (item.get("source") or {}).get("name", "") or ""
    return NewsArticle(
        headline=item.get("title") or "",
        body=item.get("description") or item.get("content") or "",
        source=source,
        url=item.get("url") or "",
        published_at=_parse_published(item.get("publishedAt", "")),
    )


def fetch_news(
    ticker: str,
    since: datetime | None = None,
    page_size: int = 50,
) -> list[NewsArticle]:
    """
    Fetch recent news for ``ticker`` from NewsAPI.

    Requires the ``NEWS_API_KEY`` environment variable. Returns a list of
    :class:`NewsArticle`, newest first.
    """
    api_key = os.environ.get("NEWS_API_KEY")
    if not api_key:
        raise RuntimeError("NEWS_API_KEY environment variable is not set")

    from newsapi import NewsApiClient  # lazy: optional dep + network

    client = NewsApiClient(api_key=api_key)
    kwargs: dict[str, object] = {
        "q": ticker,
        "language": "en",
        "sort_by": "publishedAt",
        "page_size": page_size,
    }
    if since is not None:
        kwargs["from_param"] = since.astimezone(timezone.utc).date().isoformat()

    response = client.get_everything(**kwargs)
    items = response.get("articles", []) if isinstance(response, dict) else []
    return [_article_from_payload(item) for item in items]


def to_scored_article(article: NewsArticle, score: float) -> ScoredArticle:
    """Combine a :class:`NewsArticle` with a sentiment score in [-1, 1]."""
    return ScoredArticle(
        score=score,
        source_category=map_source_to_category(article.source),
        published_at=article.published_at,
    )
