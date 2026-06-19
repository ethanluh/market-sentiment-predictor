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
from typing import TYPE_CHECKING, Callable, TypeVar

from src.sentiment.aggregation import ScoredArticle

if TYPE_CHECKING:  # for type-only annotations of the lazy source fetchers
    from src.ingestion.filings import Filing
    from src.ingestion.social import SocialPost

logger = logging.getLogger("pipeline.run")

F = TypeVar("F", bound=Callable[..., object])
T = TypeVar("T")


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
    origin_hint: str | None = None  # forced diffusion origin (e.g. sec_corp on a filing)


@dataclass
class PredictionResult:
    ticker: str
    estimated_lag_hours: float
    predictions: dict[str, dict[str, float]]
    top_articles: list[dict] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)


def _safe(source: str, fn: Callable[[], T]) -> T | None:
    """Run a supplementary source fetch, degrading to ``None`` on any failure.

    Network errors / missing credentials for one source (e.g. no Reddit creds)
    should not abort the whole ingest — only prices are essential.
    """
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 - intentional broad degrade
        logger.warning("ingest: source %s unavailable (%s); skipping", source, exc)
        return None


@step("ingest")
def _ingest(
    ticker: str,
    as_of: datetime,
    interval: str = "1d",
    lookback_days: int = 7,
    subreddits: list[str] | None = None,
) -> IngestBundle:
    from datetime import timedelta

    from src.graph.categories import NODE_SEC_CORP
    from src.ingestion.news import fetch_news, to_scored_article
    from src.ingestion.price import fetch_prices, fetch_returns_matrix

    since = as_of - timedelta(days=lookback_days)
    # Intraday history is short-lived on Yahoo Finance; use a tighter price window
    # for intraday intervals and a long window for daily bars.
    price_days = 60 if interval not in ("1d", "1wk", "1mo") else 180
    price_start = (as_of - timedelta(days=price_days)).date().isoformat()

    # Prices are essential — let a failure propagate.
    prices = fetch_prices(ticker, start=price_start, interval=interval)
    sector_returns = fetch_returns_matrix([ticker], start=price_start, interval=interval)

    texts: list[str] = []
    meta: list[ScoredArticle] = []

    # News (financial press) — text-bearing sentiment.
    for article in _safe("news", lambda: fetch_news(ticker, since=since)) or []:
        texts.append(article.text)
        meta.append(to_scored_article(article, 0.0))

    # Social (retail) — text-bearing sentiment.
    def _fetch_social() -> "list[SocialPost]":
        from src.ingestion.social import fetch_reddit

        return fetch_reddit(ticker, subreddits=subreddits)

    for post in _safe("social", _fetch_social) or []:
        texts.append(post.text)
        meta.append(
            ScoredArticle(
                score=0.0, source_category=post.source_category, published_at=post.created_at
            )
        )

    # Filings (sec_corp) — origin/timing signal. A recent filing forces the
    # diffusion origin to sec_corp (the fastest edge), shaping estimated_lag.
    def _fetch_filings() -> "list[Filing]":
        from src.ingestion.filings import fetch_filings

        return fetch_filings(ticker)

    filings = _safe("filings", _fetch_filings) or []
    origin_hint = NODE_SEC_CORP if any(f.filed_at >= since for f in filings) else None

    return IngestBundle(
        ticker=ticker,
        as_of=as_of,
        articles_text=texts,
        article_meta=meta,
        prices=prices,
        sector_returns=sector_returns,
        origin_hint=origin_hint,
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
        origin_node=bundle.origin_hint,
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
    # No trained model: return an unfitted model with the default backend
    # (consistent with training/backtest); _predict yields {} which the caller
    # surfaces as an empty predictions map.
    return QuantileReturnModel()


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
    from src.prediction.baselines import horizon_to_interval

    horizons = horizons or [horizon]
    as_of = as_of or datetime.now(timezone.utc)
    model = _load_model(model_path)

    # Group horizons by the price-bar interval they are measured on, so "1h"
    # is predicted from hourly bars and "1d"/"5d" from daily bars.
    by_interval: dict[str, list[str]] = {}
    for h in horizons:
        by_interval.setdefault(horizon_to_interval(h), []).append(h)

    predictions: dict[str, dict[str, float]] = {}
    estimated_lag = 0.0
    scored: list[ScoredArticle] = []
    for interval, interval_horizons in by_interval.items():
        bundle = _ingest(ticker, as_of, interval=interval)
        scored = _score(bundle)
        feature_vector = _features(bundle, scored)
        estimated_lag = feature_vector.estimated_lag_hours
        predictions.update(_predict(model, feature_vector, interval_horizons))

    return PredictionResult(
        ticker=ticker,
        estimated_lag_hours=estimated_lag,
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
