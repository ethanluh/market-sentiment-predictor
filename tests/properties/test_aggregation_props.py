from datetime import datetime, timedelta, timezone

import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.sentiment.aggregation import CREDIBILITY, ScoredArticle, aggregate

AS_OF = datetime(2024, 6, 1, tzinfo=timezone.utc)

articles = st.lists(
    st.builds(
        ScoredArticle,
        score=st.floats(-1.0, 1.0),
        source_category=st.sampled_from(sorted(CREDIBILITY)),
        published_at=st.integers(0, 24 * 30).map(lambda h: AS_OF - timedelta(hours=h)),
    ),
    min_size=1,
    max_size=30,
)


@given(articles)
def test_weighted_mean_stays_within_score_range(arts: list[ScoredArticle]) -> None:
    lo, hi = min(a.score for a in arts), max(a.score for a in arts)
    assert lo - 1e-9 <= aggregate(arts, as_of=AS_OF) <= hi + 1e-9


@given(articles, st.randoms(use_true_random=False))
def test_order_of_articles_does_not_matter(arts: list[ScoredArticle], rnd) -> None:  # type: ignore[no-untyped-def]
    shuffled = list(arts)
    rnd.shuffle(shuffled)
    assert aggregate(shuffled, as_of=AS_OF) == pytest.approx(aggregate(arts, as_of=AS_OF))


@given(articles, st.floats(-1.0, 1.0))
def test_identical_scores_aggregate_to_that_score(arts: list[ScoredArticle], score: float) -> None:
    same = [ScoredArticle(score, a.source_category, a.published_at) for a in arts]
    assert aggregate(same, as_of=AS_OF) == pytest.approx(score)


@given(articles)
def test_flipping_every_sign_flips_the_aggregate(arts: list[ScoredArticle]) -> None:
    flipped = [ScoredArticle(-a.score, a.source_category, a.published_at) for a in arts]
    assert aggregate(flipped, as_of=AS_OF) == pytest.approx(-aggregate(arts, as_of=AS_OF))
