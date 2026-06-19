"""
End-to-end pipeline orchestration (lightweight, no Prefect).

Wires the four stages sequentially: ingest -> score sentiment -> assemble
features -> predict quantile returns. Each stage is a small typed step function
decorated with :func:`step` for timing/observability. Heavy and network-bound
dependencies (ingestion clients, FinBERT, a trained model) are imported lazily
inside the steps so importing this module stays cheap and unit tests can stub
each stage.

    python -m src.pipeline.run --ticker AAPL --horizon 1d
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from functools import wraps
from typing import Callable, TypeVar

from src.sentiment.aggregation import ScoredArticle

logger = logging.getLogger("pipeline.run")

F = TypeVar("F", bound=Callable[..., object])


def step(name: str) -> Callable[[F], F]:
    """Decorator that logs a stage's start/end and elapsed time."""

    def decorator(fn: F) -> F:
        @wraps(fn)
        def wrapper(*args: object, **kwargs: object) -> object:
            logger.info("step %s: start", name)
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                logger.info("step %s: done in %.3fs", name, time.perf_counter() - t0)

        return wrapper  # type: ignore[return-value]

    return decorator


@dataclass
class IngestBundle:
    ticker: str
    as_of: datetime
    articles_text: list[str]
    article_meta: list[ScoredArticle]  # everything except the score (score=0 placeholder)
    prices: object  # pd.DataFrame
    sector_returns: object  # pd.DataFrame


@dataclass
class PredictionResult:
    ticker: str
    estimated_lag_hours: float
    predictions: dict[str, dict[str, float]]
    top_articles: list[dict] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)


@step("ingest")
def _ingest(ticker: str, as_of: datetime, lookback_days: int = 7) -> IngestBundle:
    from datetime import timedelta

    from src.ingestion.news import fetch_news
    from src.ingestion.price import fetch_prices, fetch_returns_matrix

    since = as_of - timedelta(days=lookback_days)
    articles = fetch_news(ticker, since=since)
    prices = fetch_prices(ticker, start=(as_of - timedelta(days=180)).date().isoformat())
    sector_returns = fetch_returns_matrix(
        [ticker], start=(as_of - timedelta(days=180)).date().isoformat()
    )

    from src.ingestion.news import to_scored_article

    meta = [to_scored_article(a, 0.0) for a in articles]
    return IngestBundle(
        ticker=ticker,
        as_of=as_of,
        articles_text=[a.text for a in articles],
        article_meta=meta,
        prices=prices,
        sector_returns=sector_returns,
    )


@step("score_sentiment")
def _score(bundle: IngestBundle) -> list[ScoredArticle]:
    if not bundle.articles_text:
        return []
    from src.sentiment.inference import score_batch  # lazy: FinBERT

    scores = score_batch(bundle.articles_text)
    return [
        ScoredArticle(score=s, source_category=m.source_category, published_at=m.published_at)
        for s, m in zip(scores, bundle.article_meta)
    ]


@step("features")
def _features(bundle: IngestBundle, articles: list[ScoredArticle]):  # type: ignore[no-untyped-def]
    from src.prediction.features import build_feature_vector

    return build_feature_vector(
        bundle.ticker,
        bundle.as_of,
        articles,
        bundle.prices,
        bundle.sector_returns,
    )


@step("predict")
def _predict(model, feature_vector, horizons: list[str]):  # type: ignore[no-untyped-def]
    preds = model.predict_one(feature_vector)
    return {h: preds[h].as_dict() for h in horizons if h in preds}


def _load_model(model_path: str | None):  # type: ignore[no-untyped-def]
    """Load a trained model, or fall back to a baseline-backed shim."""
    from src.prediction.model import QuantileReturnModel

    if model_path:
        return QuantileReturnModel.load(model_path)
    # No trained model: return an unfitted model; _predict will yield {} which
    # the caller surfaces as an empty predictions map.
    return QuantileReturnModel(backend="linear")


def _top_articles(articles: list[ScoredArticle], limit: int = 5) -> list[dict]:
    ranked = sorted(articles, key=lambda a: abs(a.score), reverse=True)[:limit]
    return [
        {
            "source_category": a.source_category,
            "score": a.score,
            "published_at": a.published_at.isoformat(),
        }
        for a in ranked
    ]


def run(
    ticker: str,
    horizon: str = "1d",
    horizons: list[str] | None = None,
    model_path: str | None = None,
    as_of: datetime | None = None,
) -> PredictionResult:
    """Run the full pipeline for ``ticker`` and return a prediction result."""
    horizons = horizons or [horizon]
    as_of = as_of or datetime.now(timezone.utc)

    bundle = _ingest(ticker, as_of)
    scored = _score(bundle)
    feature_vector = _features(bundle, scored)
    model = _load_model(model_path)
    predictions = _predict(model, feature_vector, horizons)

    return PredictionResult(
        ticker=ticker,
        estimated_lag_hours=feature_vector.estimated_lag_hours,
        predictions=predictions,
        top_articles=_top_articles(scored),
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description="Run the prediction pipeline")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--horizon", default="1d")
    parser.add_argument("--model-path", default=None)
    args = parser.parse_args(argv)

    result = run(args.ticker, horizon=args.horizon, model_path=args.model_path)
    print(result.to_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
