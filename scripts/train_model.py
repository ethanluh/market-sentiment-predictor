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
    use_news_archive: bool = False,
    lookback_days: int = 7,
    use_trends: bool = False,
    use_insider_flow: bool = False,
    use_vix: bool = False,
):  # type: ignore[no-untyped-def]
    """Fetch prices, build features+targets, fit a model, and save it.

    ``interval`` is the bar size to fetch and train on. Forward-return targets
    are built per horizon via ``horizon_to_steps`` (which raises on an unknown
    horizon). All requested horizons are trained on the same ``interval`` frame.

    The opt-in flags activate enrichment features over the training window,
    each degrading gracefully to a neutral feature on failure:
    ``use_news_archive`` (sentiment_agg / estimated_lag_hours), ``peers``
    (sector_proximity), ``use_vix`` (regime_label via ^VIX), ``use_trends``
    (search_interest_zscore), ``use_insider_flow`` (insider_flow_npr).
    """
    from src.eval.targets import forward_log_return
    from src.prediction.baselines import horizon_to_interval, horizon_to_steps
    from src.prediction.dataset import DatasetConfig, build_dataset, regime_path
    from src.prediction.model import QuantileReturnModel

    # Validate horizons up front (raises ValueError on an unknown label), and that each is measured
    # on the bar interval being trained on: "1h" on daily bars would silently train a 1-day target.
    for h in horizons:
        horizon_to_steps(h)
        if horizon_to_interval(h) != interval:
            raise ValueError(
                f"horizon {h!r} is measured on {horizon_to_interval(h)!r} bars, not {interval!r}"
            )

    # One feature path for training, evaluation and forecasting (see prediction/dataset.py): enabled
    # sources are strict (a failure raises), and the regime classifier is fit on the whole training
    # window and saved beside the model so serving can compute the same ``regime_label``.
    ds = build_dataset(
        DatasetConfig(
            ticker=ticker,
            start=start,
            end=end,
            interval=interval,
            peers=tuple(peers or ()),
            use_vix=use_vix,
            use_trends=use_trends,
            use_insider_flow=use_insider_flow,
            use_news_archive=use_news_archive,
            lookback_days=lookback_days,
        ),
        regime="fit_all" if use_vix else "none",
    )
    features = ds.features

    # Same target as the evaluation harness: the cumulative log-return over the whole horizon.
    targets = {
        h: forward_log_return(ds.log_returns, horizon_to_steps(h)).reindex(features.index)
        for h in horizons
    }

    model = QuantileReturnModel(horizons=tuple(horizons), backend=backend)
    model.fit(features, targets)

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out_path))
    if ds.regime_classifier is not None:
        ds.regime_classifier.save(str(regime_path(out_path)))
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
    parser.add_argument("--use-news-archive", action="store_true", help="enable point-in-time news")
    parser.add_argument("--lookback-days", type=int, default=7, help="news-archive lookback window")
    parser.add_argument("--use-trends", action="store_true", help="fetch Google Trends interest")
    parser.add_argument(
        "--use-insider-flow", action="store_true", help="fetch Form 4 insider trades"
    )
    parser.add_argument("--use-vix", action="store_true", help="fetch ^VIX for the regime label")
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
        use_news_archive=args.use_news_archive,
        lookback_days=args.lookback_days,
        use_trends=args.use_trends,
        use_insider_flow=args.use_insider_flow,
        use_vix=args.use_vix,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
