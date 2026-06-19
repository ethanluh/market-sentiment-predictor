"""
Social ingestion via Reddit (``praw``).

Posts from finance subreddits are categorised as retail nodes. Active
trading-focused subreddits (e.g. ``options``, ``investing``) map to
``informed_retail``; broad meme/mainstream subreddits map to
``uninformed_retail``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone

from src.graph.categories import NODE_INFORMED_RETAIL, NODE_UNINFORMED_RETAIL

DEFAULT_SUBREDDITS = ["investing", "stocks", "options", "wallstreetbets"]

# Subreddits whose audience trades actively / processes filings quickly.
INFORMED_SUBREDDITS = {"investing", "stocks", "options", "securityanalysis"}


def subreddit_category(subreddit: str) -> str:
    """Map a subreddit name to a retail node category."""
    return (
        NODE_INFORMED_RETAIL if subreddit.lower() in INFORMED_SUBREDDITS else NODE_UNINFORMED_RETAIL
    )


@dataclass
class SocialPost:
    title: str
    body: str
    subreddit: str
    score: int  # reddit upvotes, not sentiment
    created_at: datetime  # tz-aware UTC
    source_category: str

    @property
    def text(self) -> str:
        return f"{self.title}. {self.body}".strip()


def _reddit_client():  # type: ignore[no-untyped-def]
    """Build a read-only praw Reddit client from environment credentials."""
    import praw  # lazy: optional dep + network

    client_id = os.environ.get("REDDIT_CLIENT_ID")
    client_secret = os.environ.get("REDDIT_CLIENT_SECRET")
    user_agent = os.environ.get("REDDIT_USER_AGENT", "market-sentiment-predictor")
    if not client_id or not client_secret:
        raise RuntimeError(
            "REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET environment variables are required"
        )
    return praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
        check_for_async=False,
    )


def fetch_reddit(
    ticker: str,
    subreddits: list[str] | None = None,
    limit: int = 25,
) -> list[SocialPost]:
    """
    Search the given subreddits for ``ticker`` mentions and return posts.

    Requires ``REDDIT_CLIENT_ID`` / ``REDDIT_CLIENT_SECRET`` environment
    variables. Returns up to ``limit`` posts per subreddit.
    """
    subreddits = subreddits or DEFAULT_SUBREDDITS
    reddit = _reddit_client()

    posts: list[SocialPost] = []
    for name in subreddits:
        category = subreddit_category(name)
        for submission in reddit.subreddit(name).search(ticker, limit=limit):
            posts.append(
                SocialPost(
                    title=getattr(submission, "title", "") or "",
                    body=getattr(submission, "selftext", "") or "",
                    subreddit=name,
                    score=int(getattr(submission, "score", 0) or 0),
                    created_at=datetime.fromtimestamp(
                        float(getattr(submission, "created_utc", 0.0)), tz=timezone.utc
                    ),
                    source_category=category,
                )
            )
    return posts
