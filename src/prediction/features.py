"""
Feature assembly for quantile return prediction.

Fans in the upstream signals into a single numeric feature vector per
``(ticker, as_of)``:

  - aggregated, credibility/recency-weighted sentiment   (sentiment.aggregation)
  - information-diffusion lag to the institutional node   (graph.diffusion)
  - sector proximity from the correlation graph           (graph.sector)
  - price/technical indicators computed from log-returns
  - a volatility-regime label                             (prediction.regime)

This layer is pure-numeric and network-free: it consumes already-scored
articles and an already-built price frame, so it is fully unit-testable on the
scientific stack alone. All technical features use only data at or before
``as_of`` (no look-ahead).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Callable

import numpy as np
import pandas as pd

from src.graph.categories import (
    ALL_NODES,
    NODE_FIN_PRESS,
    NODE_SEC_CORP,
    NODE_UNINFORMED_RETAIL,
)
from src.graph.diffusion import estimate_lag
from src.graph.sector import build_sector_graph, sector_proximity
from src.sentiment.aggregation import CREDIBILITY, ScoredArticle, aggregate

if TYPE_CHECKING:  # avoid import cost / cycles at runtime
    from src.prediction.regime import RegimeClassifier

# Canonical numeric-feature order. Single source of truth shared with
# model.py; excludes identifiers (ticker, as_of) and the non-numeric origin.
FEATURE_NAMES: list[str] = [
    "sentiment_agg",
    "estimated_lag_hours",
    "sector_proximity",
    "momentum_5d",
    "momentum_20d",
    "realized_vol_20d",
    "volume_zscore",
    "return_lag_1",
    "regime_label",
]


def _coerce_utc(dt: datetime) -> datetime:
    """Coerce a (possibly naive) datetime to tz-aware UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


@dataclass(frozen=True)
class FeatureVector:
    ticker: str
    as_of: datetime

    # sentiment / graph
    sentiment_agg: float
    sentiment_origin: str
    estimated_lag_hours: float
    sector_proximity: float

    # price / technical
    momentum_5d: float
    momentum_20d: float
    realized_vol_20d: float
    volume_zscore: float
    return_lag_1: float

    # regime
    regime_label: int

    def to_array(self) -> np.ndarray:
        """Numeric vector ordered by ``FEATURE_NAMES`` (for sklearn)."""
        values = {
            "sentiment_agg": self.sentiment_agg,
            "estimated_lag_hours": self.estimated_lag_hours,
            "sector_proximity": self.sector_proximity,
            "momentum_5d": self.momentum_5d,
            "momentum_20d": self.momentum_20d,
            "realized_vol_20d": self.realized_vol_20d,
            "volume_zscore": self.volume_zscore,
            "return_lag_1": self.return_lag_1,
            "regime_label": float(self.regime_label),
        }
        return np.array([values[name] for name in FEATURE_NAMES], dtype=float)

    def to_series(self) -> pd.Series:
        """Named numeric series indexed by ``FEATURE_NAMES``."""
        return pd.Series(self.to_array(), index=FEATURE_NAMES, name=self.as_of)


# Map specific outlet credibility keys to their diffusion graph node. Node
# categories (already in ALL_NODES) map to themselves.
_SOURCE_NODE_MAP: dict[str, str] = {
    "reuters": NODE_FIN_PRESS,
    "bloomberg": NODE_FIN_PRESS,
    "wsj": NODE_FIN_PRESS,
    "ft": NODE_FIN_PRESS,
    "benzinga": NODE_FIN_PRESS,
    "seeking_alpha": NODE_FIN_PRESS,
    "unknown": NODE_UNINFORMED_RETAIL,
}


def source_to_node(source_category: str) -> str:
    """Map an article source category to a diffusion graph node in ALL_NODES."""
    if source_category in ALL_NODES:
        return source_category
    return _SOURCE_NODE_MAP.get(source_category, NODE_FIN_PRESS)


def select_origin_node(articles: list[ScoredArticle]) -> str:
    """
    Pick the signal-origin node category as the highest-credibility source
    present among ``articles``. Falls back to ``uninformed_retail`` when empty.
    """
    if not articles:
        return NODE_UNINFORMED_RETAIL
    best = max(
        articles,
        key=lambda a: CREDIBILITY.get(a.source_category, CREDIBILITY["unknown"]),
    )
    return source_to_node(best.source_category)


def compute_technical_features(
    prices: pd.DataFrame,
    as_of: datetime,
    mom_windows: tuple[int, int] = (5, 20),
    vol_window: int = 20,
) -> dict[str, float]:
    """
    Compute momentum, realized vol, volume z-score and last log-return as of
    ``as_of`` using only rows with index <= ``as_of`` (no look-ahead).

    Returns ``np.nan`` for any feature whose lookback window is not satisfied
    rather than raising.
    """
    as_of = _coerce_utc(as_of)
    short_w, long_w = mom_windows

    window = prices[prices.index <= as_of]
    rets = window["log_return"].dropna()
    vols = window["volume"].dropna() if "volume" in window else pd.Series(dtype=float)

    def _sum_tail(series: pd.Series, n: int) -> float:
        return float(series.tail(n).sum()) if len(series) >= n else float("nan")

    momentum_5d = _sum_tail(rets, short_w)
    momentum_20d = _sum_tail(rets, long_w)
    realized_vol = (
        float(rets.tail(vol_window).std(ddof=0)) if len(rets) >= vol_window else float("nan")
    )
    return_lag_1 = float(rets.iloc[-1]) if len(rets) >= 1 else float("nan")

    if len(vols) >= vol_window:
        tail = vols.tail(vol_window)
        std = tail.std(ddof=0)
        volume_z = float((tail.iloc[-1] - tail.mean()) / std) if std > 0 else 0.0
    else:
        volume_z = float("nan")

    return {
        "momentum_5d": momentum_5d,
        "momentum_20d": momentum_20d,
        "realized_vol_20d": realized_vol,
        "volume_zscore": volume_z,
        "return_lag_1": return_lag_1,
    }


def _regime_label(classifier: "RegimeClassifier | None", vix_value: float | None) -> int:
    """Resolve the regime label, returning -1 (unknown) when unavailable."""
    if classifier is None or vix_value is None:
        return -1
    return int(classifier.predict(np.array([vix_value]))[0])


def build_feature_vector(
    ticker: str,
    as_of: datetime,
    articles: list[ScoredArticle],
    prices: pd.DataFrame,
    sector_returns: pd.DataFrame,
    *,
    origin_node: str | None = None,
    regime_classifier: "RegimeClassifier | None" = None,
    vix_value: float | None = None,
    alpha: float = 0.5,
    sector_window: int = 60,
    min_corr: float = 0.5,
) -> FeatureVector:
    """
    Assemble a single :class:`FeatureVector` for ``(ticker, as_of)``.

    Reuses :func:`aggregation.aggregate`, :func:`diffusion.estimate_lag`, and
    the sector graph helpers. ``origin_node`` defaults to the highest-credibility
    source present in ``articles``.
    """
    as_of = _coerce_utc(as_of)
    origin = origin_node or select_origin_node(articles)

    sentiment = aggregate(articles, as_of=as_of)
    lag = estimate_lag(origin, alpha=alpha)

    if sector_returns is not None and not sector_returns.empty:
        graph = build_sector_graph(sector_returns, window=sector_window, min_corr=min_corr)
        proximity = sector_proximity(graph, ticker)
    else:
        proximity = 0.0

    tech = compute_technical_features(prices, as_of)
    regime = _regime_label(regime_classifier, vix_value)

    return FeatureVector(
        ticker=ticker,
        as_of=as_of,
        sentiment_agg=sentiment,
        sentiment_origin=origin,
        estimated_lag_hours=lag,
        sector_proximity=proximity,
        momentum_5d=tech["momentum_5d"],
        momentum_20d=tech["momentum_20d"],
        realized_vol_20d=tech["realized_vol_20d"],
        volume_zscore=tech["volume_zscore"],
        return_lag_1=tech["return_lag_1"],
        regime_label=regime,
    )


def build_feature_matrix(
    ticker: str,
    as_of_index: pd.DatetimeIndex,
    articles_by_time: Callable[[datetime], list[ScoredArticle]],
    prices: pd.DataFrame,
    sector_returns: pd.DataFrame,
    *,
    vix: pd.Series | None = None,
    regime_classifier: "RegimeClassifier | None" = None,
    **kwargs: object,
) -> pd.DataFrame:
    """
    Build a feature matrix (one row per ``as_of`` timestamp, columns =
    ``FEATURE_NAMES``) for training / backtesting.

    ``articles_by_time(as_of)`` returns the articles known at that timestamp.
    Rows with insufficient history will contain NaNs; callers are expected to
    drop or impute them.
    """
    rows: list[pd.Series] = []
    for as_of in as_of_index:
        as_of_dt = as_of.to_pydatetime()
        vix_value = float(vix.loc[as_of]) if vix is not None and as_of in vix.index else None
        fv = build_feature_vector(
            ticker,
            as_of_dt,
            articles_by_time(as_of_dt),
            prices,
            sector_returns,
            regime_classifier=regime_classifier,
            vix_value=vix_value,
            **kwargs,  # type: ignore[arg-type]
        )
        rows.append(fv.to_series())

    if not rows:
        return pd.DataFrame(columns=FEATURE_NAMES)
    matrix = pd.DataFrame(rows)
    matrix.index = pd.DatetimeIndex(as_of_index)
    return matrix
