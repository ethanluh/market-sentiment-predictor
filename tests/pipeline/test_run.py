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
