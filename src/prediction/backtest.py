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

from src.prediction.features import FEATURE_NAMES
from src.prediction.model import QUANTILES, QuantileReturnModel

logger = logging.getLogger("prediction.backtest")


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
) -> BacktestResult:
    """
    Rolling walk-forward backtest for a single ``horizon``.

    For each fold the model is fit on a ``train_window`` slice and evaluated on
    the following ``test_window`` slice. Features at time *t* use only data
    <= *t* (guaranteed upstream), so there is no look-ahead.
    """
    features = features[FEATURE_NAMES]
    y = targets[horizon]
    n = len(features)
    if n < train_window + test_window:
        raise ValueError(
            f"Not enough rows ({n}) for train_window={train_window} + " f"test_window={test_window}"
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
    return result


def run_backtest(
    ticker: str,
    start: str,
    end: str | None = None,
    horizon: str = "1d",
    backend: str = "gbr",
    train_window: int = 252,
    test_window: int = 21,
) -> BacktestResult:
    """
    End-to-end backtest on real price history (ingestion lazy-imported).

    A point-in-time news archive is not available from the live APIs, so this
    harness builds the price/graph feature matrix with neutral sentiment for the
    historical window (``articles_by_time`` returns no articles). To backtest
    the full sentiment-driven feature set, pass a precomputed feature matrix
    directly to :func:`walk_forward`.

    NOTE: in this path three of the nine features are inert (``sentiment_agg``
    and ``estimated_lag_hours`` are constant with no articles, ``sector_proximity``
    is 0.0 with a single-ticker frame, and ``regime_label`` is -1 with no VIX
    classifier), so the result benchmarks the technical features only.
    """
    from src.ingestion.price import fetch_prices  # lazy: network
    from src.prediction.baselines import horizon_to_steps
    from src.prediction.features import build_feature_matrix

    logger.warning(
        "run_backtest: sentiment_agg, estimated_lag_hours, sector_proximity and "
        "regime_label are neutralized in this real-data path; results reflect the "
        "technical features only. Use walk_forward with a precomputed feature "
        "matrix to evaluate the full pipeline."
    )

    prices = fetch_prices(ticker, start=start, end=end)
    sector_returns = prices[["log_return"]].rename(columns={"log_return": ticker})

    features = build_feature_matrix(
        ticker,
        prices.index,
        articles_by_time=lambda _as_of: [],
        prices=prices,
        sector_returns=sector_returns,
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
