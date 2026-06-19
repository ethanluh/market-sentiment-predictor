"""Tests for pipeline.run — stages are stubbed so no network / FinBERT runs."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

import src.pipeline.run as run_mod
from src.pipeline.run import PredictionResult, _top_articles, run, step
from src.prediction.model import QuantilePrediction
from src.sentiment.aggregation import ScoredArticle


def _prices(n: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="D", tz="UTC")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    df = pd.DataFrame({"close": close, "volume": rng.integers(1e6, 5e6, n)}, index=idx)
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    return df


def test_step_decorator_preserves_return():
    @step("noop")
    def f(x: int) -> int:
        return x + 1

    assert f(1) == 2


def test_top_articles_sorted_by_abs_score():
    now = datetime.now(timezone.utc)
    arts = [
        ScoredArticle(0.1, "reuters", now),
        ScoredArticle(-0.9, "wsj", now),
        ScoredArticle(0.5, "benzinga", now),
    ]
    top = _top_articles(arts, limit=2)
    assert [a["score"] for a in top] == [-0.9, 0.5]


def test_run_returns_schema(monkeypatch):
    as_of = datetime(2023, 5, 1, tzinfo=timezone.utc)
    bundle = run_mod.IngestBundle(
        ticker="AAPL",
        as_of=as_of,
        articles_text=["good earnings"],
        article_meta=[ScoredArticle(0.0, "reuters", as_of - timedelta(hours=1))],
        prices=_prices(),
        sector_returns=pd.DataFrame(),
    )

    monkeypatch.setattr(run_mod, "_ingest", lambda ticker, as_of_, **k: bundle)
    monkeypatch.setattr(
        run_mod,
        "_score",
        lambda b: [ScoredArticle(0.7, "reuters", as_of - timedelta(hours=1))],
    )

    class _Model:
        def predict_one(self, fv):
            return {"1d": QuantilePrediction(-0.01, 0.002, 0.015)}

    monkeypatch.setattr(run_mod, "_load_model", lambda path: _Model())

    result = run("AAPL", horizon="1d", as_of=as_of)
    assert isinstance(result, PredictionResult)
    assert isinstance(result.estimated_lag_hours, float)
    assert set(result.predictions["1d"]) == {"p10", "p50", "p90"}
    assert result.predictions["1d"]["p10"] <= result.predictions["1d"]["p90"]
    assert result.top_articles


class TestMultiSourceIngest:
    """_ingest gathers news + social and uses filings as the sec_corp origin."""

    def _patch_prices(self, monkeypatch):
        import src.ingestion.price as price_mod

        monkeypatch.setattr(price_mod, "fetch_prices", lambda *a, **k: _prices())
        monkeypatch.setattr(
            price_mod,
            "fetch_returns_matrix",
            lambda *a, **k: _prices()[["log_return"]].rename(columns={"log_return": "AAPL"}),
        )

    def test_news_and_social_combined(self, monkeypatch):
        from src.graph.categories import NODE_SEC_CORP
        from src.ingestion.filings import Filing
        from src.ingestion.news import NewsArticle
        from src.ingestion.social import SocialPost

        self._patch_prices(monkeypatch)
        as_of = datetime(2023, 3, 1, tzinfo=timezone.utc)

        import src.ingestion.filings as filings_mod
        import src.ingestion.news as news_mod
        import src.ingestion.social as social_mod

        monkeypatch.setattr(
            news_mod,
            "fetch_news",
            lambda ticker, since=None, **k: [
                NewsArticle("Earnings beat", "strong", "Reuters", "u", as_of - timedelta(hours=2))
            ],
        )
        monkeypatch.setattr(
            social_mod,
            "fetch_reddit",
            lambda ticker, subreddits=None, **k: [
                SocialPost(
                    "DD",
                    "buy",
                    "wallstreetbets",
                    10,
                    as_of - timedelta(hours=1),
                    source_category="uninformed_retail",
                )
            ],
        )
        monkeypatch.setattr(
            filings_mod,
            "fetch_filings",
            lambda ticker, **k: [Filing("AAPL", "8-K", as_of - timedelta(hours=3), "p")],
        )

        bundle = run_mod._ingest("AAPL", as_of)
        # Both a news doc and a social doc are present for scoring.
        assert len(bundle.articles_text) == 2
        cats = {m.source_category for m in bundle.article_meta}
        assert "reuters" in cats and "uninformed_retail" in cats
        # A recent 8-K forces the sec_corp diffusion origin.
        assert bundle.origin_hint == NODE_SEC_CORP

    def test_old_filing_no_origin_hint(self, monkeypatch):
        from src.ingestion.filings import Filing

        self._patch_prices(monkeypatch)
        as_of = datetime(2023, 3, 1, tzinfo=timezone.utc)

        import src.ingestion.filings as filings_mod
        import src.ingestion.news as news_mod

        monkeypatch.setattr(news_mod, "fetch_news", lambda *a, **k: [])
        # Filing well outside the lookback window.
        monkeypatch.setattr(
            filings_mod,
            "fetch_filings",
            lambda ticker, **k: [Filing("AAPL", "8-K", as_of - timedelta(days=90), "p")],
        )
        bundle = run_mod._ingest("AAPL", as_of)
        assert bundle.origin_hint is None

    def test_source_failure_degrades(self, monkeypatch):
        from src.ingestion.news import NewsArticle

        self._patch_prices(monkeypatch)
        as_of = datetime(2023, 3, 1, tzinfo=timezone.utc)

        import src.ingestion.news as news_mod
        import src.ingestion.social as social_mod

        monkeypatch.setattr(
            news_mod,
            "fetch_news",
            lambda *a, **k: [NewsArticle("h", "b", "Reuters", "u", as_of - timedelta(hours=1))],
        )

        def _boom(*a, **k):
            raise RuntimeError("no reddit creds")

        monkeypatch.setattr(social_mod, "fetch_reddit", _boom)
        # Social raising must not abort ingest; news still present.
        bundle = run_mod._ingest("AAPL", as_of)
        assert len(bundle.articles_text) == 1
