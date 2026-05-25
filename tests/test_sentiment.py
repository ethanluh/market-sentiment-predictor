"""Tests for sentiment aggregation (inference tests excluded — require model download)."""

from datetime import datetime, timedelta, timezone

import pytest

from src.sentiment.aggregation import ScoredArticle, aggregate
from src.graph.categories import NODE_FIN_PRESS, NODE_UNINFORMED_RETAIL


def _article(score: float, source: str, hours_ago: float, ref: datetime) -> ScoredArticle:
    return ScoredArticle(
        score=score,
        source_category=source,
        published_at=ref - timedelta(hours=hours_ago),
    )


class TestAggregation:
    def setup_method(self):
        self.now = datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc)

    def test_empty_returns_zero(self):
        assert aggregate([], as_of=self.now) == 0.0

    def test_single_article(self):
        article = _article(0.8, NODE_FIN_PRESS, hours_ago=1.0, ref=self.now)
        score = aggregate([article], as_of=self.now)
        assert -1.0 <= score <= 1.0
        assert score > 0.0  # should be positive

    def test_credibility_weighting(self):
        # High-credibility positive vs low-credibility negative
        articles = [
            _article(+0.9, NODE_FIN_PRESS, hours_ago=1.0, ref=self.now),
            _article(-0.9, NODE_UNINFORMED_RETAIL, hours_ago=1.0, ref=self.now),
        ]
        score = aggregate(articles, as_of=self.now)
        # fin_press credibility (0.8) > uninformed_retail credibility (0.2)
        assert score > 0.0

    def test_recency_decay(self):
        # Recent article should outweigh older article of equal credibility and opposite sign
        articles = [
            _article(+0.9, NODE_FIN_PRESS, hours_ago=1.0, ref=self.now),
            _article(-0.9, NODE_FIN_PRESS, hours_ago=48.0, ref=self.now),
        ]
        score = aggregate(articles, as_of=self.now)
        assert score > 0.0

    def test_score_range(self):
        articles = [
            _article(s, NODE_FIN_PRESS, hours_ago=i + 1, ref=self.now)
            for i, s in enumerate([0.5, -0.3, 0.8, -0.1])
        ]
        score = aggregate(articles, as_of=self.now)
        assert -1.0 <= score <= 1.0
