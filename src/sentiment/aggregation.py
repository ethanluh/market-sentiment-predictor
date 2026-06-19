"""
Aggregate sentiment scores across multiple articles.

Weights encode source credibility and recency decay:
  w_i = credibility_i * exp(-lambda * delta_t_hours_i)

Lambda values are tuned per source category.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone

from src.graph.categories import (
    NODE_FIN_PRESS,
    NODE_INFORMED_RETAIL,
    NODE_SEC_CORP,
    NODE_UNINFORMED_RETAIL,
)

# Credibility weights by source category
CREDIBILITY: dict[str, float] = {
    NODE_SEC_CORP: 1.0,
    "reuters": 1.0,
    "bloomberg": 1.0,
    "wsj": 0.9,
    "ft": 0.9,
    NODE_FIN_PRESS: 0.8,
    "benzinga": 0.7,
    "seeking_alpha": 0.5,
    NODE_INFORMED_RETAIL: 0.4,
    NODE_UNINFORMED_RETAIL: 0.2,
    "unknown": 0.3,
}

# Decay rate (lambda) per source category — faster decay = more time-sensitive
DECAY_LAMBDA: dict[str, float] = {
    NODE_SEC_CORP: 0.02,  # slow decay; filings stay relevant
    NODE_FIN_PRESS: 0.10,
    NODE_INFORMED_RETAIL: 0.20,
    NODE_UNINFORMED_RETAIL: 0.40,  # fast decay; social noise
    "unknown": 0.15,
}


@dataclass
class ScoredArticle:
    score: float  # sentiment score in [-1, 1]
    source_category: str  # must be a key in CREDIBILITY
    published_at: datetime


def aggregate(articles: list[ScoredArticle], as_of: datetime | None = None) -> float:
    """
    Return credibility- and recency-weighted mean sentiment score.
    Returns 0.0 if the article list is empty.

    Parameters
    ----------
    articles : list[ScoredArticle]
    as_of : datetime
        Reference time for recency decay. Defaults to now (UTC).
    """
    if not articles:
        return 0.0

    if as_of is None:
        as_of = datetime.now(timezone.utc)

    weighted_sum = 0.0
    weight_total = 0.0

    for article in articles:
        credibility = CREDIBILITY.get(article.source_category, CREDIBILITY["unknown"])
        lam = DECAY_LAMBDA.get(article.source_category, DECAY_LAMBDA["unknown"])

        pub = article.published_at
        if pub.tzinfo is None:
            pub = pub.replace(tzinfo=timezone.utc)
        delta_hours = max(0.0, (as_of - pub).total_seconds() / 3600)

        w = credibility * math.exp(-lam * delta_hours)
        weighted_sum += w * article.score
        weight_total += w

    return weighted_sum / weight_total if weight_total > 0 else 0.0
