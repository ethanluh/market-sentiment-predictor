"""
Lightweight walk-forward backtester (no vectorbt).

Evaluates a :class:`~src.prediction.model.QuantileReturnModel` out-of-sample
with a rolling train/test split and reports:

  - pinball (quantile) loss per horizon/quantile
  - empirical coverage of the [P10, P90] interval
  - directional hit-rate of the P50 sign
  - a simple PnL curve from taking ``sign(P50)`` positions, plus its Sharpe

Metric functions are pure (synthetic-frame testable). ``run_backtest`` /
``main`` wire in real ingestion behind a lazy import and an argparse CLI:

    python -m src.prediction.backtest --ticker AAPL --start 2023-01-01
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.prediction.baselines import (
    ARIMABaseline,
    BaselinePredictor,
    GARCHBaseline,
    MomentumBaseline,
)
from src.prediction.features import FEATURE_NAMES
from src.prediction.model import QUANTILES, QuantilePrediction, QuantileReturnModel

logger = logging.getLogger("prediction.backtest")


def _default_baselines() -> dict[str, BaselinePredictor]:
    """The standard benchmark set compared against the quantile model."""
    return {
        "momentum": MomentumBaseline(),
        "arima": ARIMABaseline(),
        "garch": GARCHBaseline(),
    }


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    """Mean pinball (quantile) loss for a single quantile level."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    diff = y_true - y_pred
    return float(np.mean(np.maximum(quantile * diff, (quantile - 1) * diff)))


def interval_coverage(y_true: np.ndarray, p_low: np.ndarray, p_high: np.ndarray) -> float:
    """Fraction of actuals falling within the [p_low, p_high] interval."""
    y_true = np.asarray(y_true, dtype=float)
    if y_true.size == 0:
        return float("nan")
    inside = (y_true >= np.asarray(p_low)) & (y_true <= np.asarray(p_high))
    return float(np.mean(inside))


@dataclass
class BacktestResult:
    pinball_loss: dict[str, float] = field(default_factory=dict)
    coverage: dict[str, float] = field(default_factory=dict)
    directional_hit_rate: dict[str, float] = field(default_factory=dict)
    pnl_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    sharpe: float = 0.0
    n_predictions: int = 0
    # name -> {p10,p50,p90} mean pinball loss for each baseline (empty if not run).
    baseline_pinball: dict[str, dict[str, float]] = field(default_factory=dict)


def _sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
    r = returns.dropna()
    if len(r) < 2 or r.std(ddof=0) == 0:
        return 0.0
    return float(r.mean() / r.std(ddof=0) * np.sqrt(periods_per_year))


def walk_forward(
    model: QuantileReturnModel,
    features: pd.DataFrame,
    targets: dict[str, pd.Series],
    train_window: int = 252,
    test_window: int = 21,
    step: int = 21,
    refit: bool = True,
    horizon: str = "1d",
    returns: pd.Series | None = None,
    baselines: dict[str, BaselinePredictor] | None = None,
) -> BacktestResult:
    """
    Rolling walk-forward backtest for a single ``horizon``.

    For each fold the model is fit on a ``train_window`` slice and evaluated on
    the following ``test_window`` slice. Features at time *t* use only data
    <= *t* (guaranteed upstream), so there is no look-ahead.

    When ``returns`` (the log-return series aligned to ``features.index``) is
    supplied, the baselines in ``baselines`` (default: momentum / ARIMA / GARCH)
    are also fit per fold on returns up to the train-window end and scored, so
    ``result.baseline_pinball`` gives a model-vs-baseline comparison.
    """
    features = features[FEATURE_NAMES]
    y = targets[horizon]
    n = len(features)
    if n < train_window + test_window:
        raise ValueError(
            f"Not enough rows ({n}) for train_window={train_window} + " f"test_window={test_window}"
        )

    if returns is not None and baselines is None:
        baselines = _default_baselines()
    use_baselines = returns is not None and bool(baselines)
    # name -> quantile -> per-prediction baseline forecasts (aligned to `actuals`).
    baseline_preds: dict[str, dict[float, list[float]]] = (
        {name: {q: [] for q in QUANTILES} for name in baselines}
        if (use_baselines and baselines is not None)
        else {}
    )

    preds_by_q: dict[float, list[float]] = {q: [] for q in QUANTILES}
    actuals: list[float] = []
    pnl: list[float] = []
    pnl_index: list[pd.Timestamp] = []

    start = 0
    while start + train_window + test_window <= n:
        tr = slice(start, start + train_window)
        te = slice(start + train_window, start + train_window + test_window)
        X_tr, X_te = features.iloc[tr], features.iloc[te]
        y_tr = y.iloc[tr]

        if refit or not model._available:  # noqa: SLF001 - internal flag check
            model.fit(X_tr, {horizon: y_tr})
        if horizon not in model._available:
            start += step
            continue

        # One baseline forecast per fold (fit on returns up to the train end).
        fold_baseline: dict[str, QuantilePrediction] = {}
        if use_baselines:
            assert returns is not None and baselines is not None  # narrowed by use_baselines
            train_returns = returns.loc[: X_tr.index[-1]]
            for name, baseline in baselines.items():
                try:
                    fold_baseline[name] = baseline.fit(train_returns).predict(horizon)
                except Exception:  # baselines degrade internally; guard alignment anyway
                    logger.warning("baseline %s failed this fold; using zero forecast", name)
                    fold_baseline[name] = QuantilePrediction(0.0, 0.0, 0.0)

        fold = model.predict(X_te).get(horizon, [])
        for i, qp in enumerate(fold):
            actual = float(y.iloc[start + train_window + i])
            if np.isnan(actual):
                continue
            preds_by_q[0.10].append(qp.p10)
            preds_by_q[0.50].append(qp.p50)
            preds_by_q[0.90].append(qp.p90)
            actuals.append(actual)
            pnl.append(np.sign(qp.p50) * actual)
            pnl_index.append(X_te.index[i])
            for name, bqp in fold_baseline.items():
                baseline_preds[name][0.10].append(bqp.p10)
                baseline_preds[name][0.50].append(bqp.p50)
                baseline_preds[name][0.90].append(bqp.p90)
        start += step

    result = BacktestResult()
    result.n_predictions = len(actuals)
    if not actuals:
        return result

    actual_arr = np.array(actuals)
    result.pinball_loss = {
        f"{horizon}_p{int(q * 100)}": pinball_loss(actual_arr, np.array(preds_by_q[q]), q)
        for q in QUANTILES
    }
    result.coverage = {
        horizon: interval_coverage(
            actual_arr, np.array(preds_by_q[0.10]), np.array(preds_by_q[0.90])
        )
    }
    result.directional_hit_rate = {
        horizon: float(np.mean(np.sign(np.array(preds_by_q[0.50])) == np.sign(actual_arr)))
    }
    pnl_series = pd.Series(pnl, index=pd.DatetimeIndex(pnl_index))
    result.pnl_curve = pnl_series.cumsum()
    result.sharpe = _sharpe(pnl_series)
    result.baseline_pinball = {
        name: {
            f"p{int(q * 100)}": pinball_loss(actual_arr, np.array(qpreds[q]), q) for q in QUANTILES
        }
        for name, qpreds in baseline_preds.items()
    }
    return result


def run_backtest(
    ticker: str,
    start: str,
    end: str | None = None,
    horizon: str = "1d",
    backend: str = "gbr",
    train_window: int = 252,
    test_window: int = 21,
    *,
    peers: list[str] | None = None,
    vix: "pd.Series | None" = None,
    trends: "pd.Series | None" = None,
    insider_flow: "pd.Series | None" = None,
    use_news_archive: bool = False,
    lookback_days: int = 7,
) -> BacktestResult:
    """
    End-to-end backtest on real price history (ingestion lazy-imported).

    The price frame is fetched at the *interval* the horizon is measured on
    (``"1h"`` uses hourly bars, ``"1d"``/``"5d"`` use daily), so ``"1h"`` is a
    genuine intraday horizon rather than a duplicate of ``"1d"``.

    Feature activation depends on the optional inputs:

      - ``use_news_archive=True`` wires a point-in-time news reader
        (``ingestion.news_archive``) into ``articles_by_time`` so ``sentiment_agg``
        and ``estimated_lag_hours`` are live; otherwise they are neutral.
      - ``peers`` builds a multi-ticker sector graph so ``sector_proximity`` is
        live; otherwise it is 0.0.
      - ``vix`` fits a :class:`RegimeClassifier` so ``regime_label`` is live;
        otherwise it is -1.
      - ``trends`` (a Google search-interest series) makes
        ``search_interest_zscore`` live; otherwise it is 0.0.
      - ``insider_flow`` (a signed-insider-shares series from Form 4) makes
        ``insider_flow_npr`` live; otherwise it is 0.0.
    """
    from src.ingestion.price import fetch_prices  # lazy: network
    from src.prediction.baselines import horizon_to_interval, horizon_to_steps
    from src.prediction.features import build_feature_matrix

    interval = horizon_to_interval(horizon)
    prices = fetch_prices(ticker, start=start, end=end, interval=interval)

    # Sector returns: peers (live proximity) or just the target column (inert).
    if peers:
        from src.ingestion.price import fetch_returns_matrix

        sector_returns = fetch_returns_matrix(
            [ticker, *peers], start=start, end=end, interval=interval
        )
    else:
        sector_returns = prices[["log_return"]].rename(columns={"log_return": ticker})

    # Regime classifier from VIX (live regime label) when supplied.
    regime_classifier = None
    if vix is not None:
        from src.prediction.regime import RegimeClassifier

        regime_classifier = RegimeClassifier().fit(vix.to_numpy())

    # Point-in-time news reader (live sentiment) when requested.
    if use_news_archive:
        from src.ingestion.news_archive import make_archive_reader

        articles_by_time = make_archive_reader(ticker, lookback_days=lookback_days)
    else:
        articles_by_time = lambda _as_of: []  # noqa: E731 - tiny inline provider

    inert = [
        name
        for name, active in [
            ("sentiment_agg/estimated_lag_hours", use_news_archive),
            ("sector_proximity", bool(peers)),
            ("regime_label", vix is not None),
            ("search_interest_zscore", trends is not None),
            ("insider_flow_npr", insider_flow is not None),
        ]
        if not active
    ]
    if inert:
        logger.warning(
            "run_backtest: the following features are neutralized in this call: %s. "
            "Enable use_news_archive / peers / vix / trends / insider_flow to activate them.",
            ", ".join(inert),
        )

    features = build_feature_matrix(
        ticker,
        prices.index,
        articles_by_time=articles_by_time,
        prices=prices,
        sector_returns=sector_returns,
        vix=vix,
        regime_classifier=regime_classifier,
        trends=trends,
        insider_flow=insider_flow,
    ).dropna()

    steps = horizon_to_steps(horizon)
    target = prices["log_return"].shift(-steps).reindex(features.index)

    model = QuantileReturnModel(horizons=(horizon,), backend=backend)
    return walk_forward(
        model,
        features,
        {horizon: target},
        train_window=train_window,
        test_window=test_window,
        horizon=horizon,
        returns=prices["log_return"].reindex(features.index),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Walk-forward quantile backtest")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", default=None)
    parser.add_argument("--horizon", default="1d")
    parser.add_argument("--backend", default="gbr", choices=["linear", "gbr"])
    args = parser.parse_args(argv)

    result = run_backtest(
        args.ticker, args.start, args.end, horizon=args.horizon, backend=args.backend
    )
    print(f"Backtest {args.ticker} [{args.horizon}] — n={result.n_predictions}")
    print(f"  pinball_loss: {result.pinball_loss}")
    print(f"  coverage:     {result.coverage}")
    print(f"  hit_rate:     {result.directional_hit_rate}")
    print(f"  sharpe:       {result.sharpe:.3f}")
    if result.baseline_pinball:
        model_p50 = result.pinball_loss.get(f"{args.horizon}_p50", float("nan"))
        print("  model vs baselines (p50 pinball):")
        print(f"    model:    {model_p50:.6f}")
        for name, losses in result.baseline_pinball.items():
            print(f"    {name:<9} {losses['p50']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
