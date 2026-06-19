"""
Train and persist a quantile return model from real price history.

Closes the gap where ``python -m src.pipeline.run`` with no ``--model-path``
falls back to an unfitted shim that yields empty predictions. This builds a
point-in-time feature matrix and fits a
:class:`~src.prediction.model.QuantileReturnModel`, saving an artifact the
pipeline / API can load.

Usage:
    python scripts/train_model.py --ticker AAPL --start 2022-01-01 \
        --horizons 1d 5d --interval 1d --out models/aapl.joblib

Notes:
- The model is trained on a single ``--interval`` frame (daily by default);
  targets are forward returns per horizon. Use a matching interval for the
  horizons you train (daily ``1d``/``5d`` on daily bars).
- Without a point-in-time news archive the sentiment / lag features are inert
  (see ``scripts/build_news_archive.py``); pass ``--peers`` to activate
  ``sector_proximity``. The script warns about which features are neutral so the
  trained model isn't over-interpreted.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running directly (python scripts/train_model.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logger = logging.getLogger("scripts.train_model")


def train(
    ticker: str,
    start: str,
    end: str | None = None,
    horizons: tuple[str, ...] = ("1d", "5d"),
    interval: str = "1d",
    out: str = "models/model.joblib",
    peers: list[str] | None = None,
    backend: str = "gbr",
):  # type: ignore[no-untyped-def]
    """Fetch prices, build features+targets, fit a model, and save it.

    ``interval`` is the bar size to fetch and train on. Forward-return targets
    are built per horizon via ``horizon_to_steps`` (which raises on an unknown
    horizon). All requested horizons are trained on the same ``interval`` frame.
    """
    from src.ingestion.price import fetch_prices, fetch_returns_matrix
    from src.prediction.baselines import horizon_to_steps
    from src.prediction.features import build_feature_matrix
    from src.prediction.model import QuantileReturnModel

    # Validate horizons up front (raises ValueError on an unknown label).
    for h in horizons:
        horizon_to_steps(h)

    prices = fetch_prices(ticker, start=start, end=end, interval=interval)
    if peers:
        sector_returns = fetch_returns_matrix(
            [ticker, *peers], start=start, end=end, interval=interval
        )
    else:
        logger.warning(
            "train_model: no --peers and no news archive, so sentiment_agg, "
            "estimated_lag_hours, sector_proximity and regime_label are neutral; "
            "the model learns from technical features only."
        )
        sector_returns = prices[["log_return"]].rename(columns={"log_return": ticker})

    features = build_feature_matrix(
        ticker,
        prices.index,
        articles_by_time=lambda _as_of: [],
        prices=prices,
        sector_returns=sector_returns,
    ).dropna()

    targets = {
        h: prices["log_return"].shift(-horizon_to_steps(h)).reindex(features.index)
        for h in horizons
    }

    model = QuantileReturnModel(horizons=tuple(horizons), backend=backend)
    model.fit(features, targets)

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out_path))
    logger.info("Saved model (%d rows, horizons=%s) -> %s", len(features), list(horizons), out_path)
    return model


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description="Train a quantile return model")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", default=None)
    parser.add_argument("--horizons", nargs="+", default=["1d", "5d"])
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--peers", nargs="*", default=None)
    parser.add_argument("--backend", default="gbr", choices=["linear", "gbr"])
    parser.add_argument("--out", default="models/model.joblib")
    args = parser.parse_args(argv)

    train(
        args.ticker,
        args.start,
        args.end,
        horizons=tuple(args.horizons),
        interval=args.interval,
        out=args.out,
        peers=args.peers,
        backend=args.backend,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
