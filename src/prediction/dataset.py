"""
The single feature path: training, evaluation and live forecasting all build features here.

``build_dataset`` assembles prices, peers, VIX, trends, insider flow and the point-in-time news
reader into one feature matrix with ``features.build_feature_matrix``. Having one path is what
stops train/serve skew: a model trained on these features sees the same columns, computed the same
way, when it forecasts.

Design rules:

- **Strict.** A source that was asked for and fails, comes back empty, does not cover the window,
  or is stale raises. There is no ``safe_fetch`` fallback to a neutral value, so no model is ever
  trained, scored or traded on silently degraded features.
- **One information time per bar.** Daily bars are stamped 00:00 UTC but represent the close, so
  every lookup for a bar (prices, VIX, trends, news and the recency decay of that news) is made at
  ``bar_close``: 20:00 UTC on the bar's date. That is exact in US summer and an hour conservative in
  winter. Same-day pre-close news is included, post-close news never is, and an article's age is
  measured from the close, as it will be when forecasting live. The next bar's stamp (the next
  midnight) is after the close, so price lookups never see it.
- **Daily bars only.** Hourly or weekly bars need their own, separately leakage-tested alignment
  (yfinance stamps hourly bars at their start), so they raise instead of being guessed at.
- **Regime.** ``regime="none"`` leaves ``regime_label`` at -1 and returns the VIX so a harness can
  refit the classifier per fold without leakage. ``regime="fit_all"`` fits it on the whole window
  (for a final model) and returns it so it can be saved with the model (``regime_path``).
- **Injectable sources.** Tests pass fakes; the defaults import the real fetchers at call time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import pandas as pd

from src.prediction.features import build_feature_matrix
from src.prediction.regime import RegimeClassifier
from src.sentiment.aggregation import ScoredArticle

logger = logging.getLogger("prediction.dataset")

NEWS_CUTOFF_UTC = time(20, 0)  # about the US equity close; see module docstring
# Google Trends stamps weekly values at the week start, so a value is only fully known about a week
# later; daily values are known after midnight. A uniform 7-day availability lag is safe for both
# (and conservative for daily data).
TRENDS_AVAILABILITY_LAG = pd.Timedelta(days=7)
# The insider net-purchase ratio looks back 90 days, so fetch 120 days before the window starts or
# the first rows would see a truncated window.
INSIDER_LOOKBACK_MARGIN = pd.Timedelta(days=120)
MAX_TRENDS_STALENESS = pd.Timedelta(days=14)
MAX_VIX_STALENESS = pd.Timedelta(days=5)

NewsReader = Callable[[datetime], list[ScoredArticle]]


@dataclass(frozen=True)
class DatasetConfig:
    ticker: str
    start: str | None = None
    end: str | None = None
    interval: str = "1d"
    peers: tuple[str, ...] = ()
    use_vix: bool = False
    use_trends: bool = False
    use_insider_flow: bool = False
    use_news_archive: bool = False
    lookback_days: int = 7

    def __post_init__(self) -> None:
        if self.interval != "1d":
            raise NotImplementedError(
                f"interval {self.interval!r} is not supported: only daily bars have a leakage-tested "
                "alignment (hourly bars are stamped at their start, weekly bars are a week stale)"
            )


@dataclass
class Sources:
    """Where the data comes from. Every field is a plain callable so tests can fake it."""

    prices: Callable[[str, str | None, str | None, str], pd.DataFrame]
    returns_matrix: Callable[[list[str], str | None, str | None, str], pd.DataFrame]
    trends: Callable[[str, str | None, str | None], pd.Series]
    insider_flow: Callable[[str, pd.Timestamp], pd.Series]
    news_reader: Callable[[str, int], NewsReader]


def default_sources() -> Sources:
    """The real fetchers, looked up when called so tests can monkeypatch the modules."""

    def prices(ticker: str, start: str | None, end: str | None, interval: str) -> pd.DataFrame:
        from src.ingestion import price

        return price.fetch_prices(ticker, start=start, end=end, interval=interval)

    def returns_matrix(
        tickers: list[str], start: str | None, end: str | None, interval: str
    ) -> pd.DataFrame:
        from src.ingestion import price

        return price.fetch_returns_matrix(tickers, start=start, end=end, interval=interval)

    def trends(ticker: str, start: str | None, end: str | None) -> pd.Series:
        from src.ingestion import trends as trends_mod

        return trends_mod.fetch_trends(ticker, start=start, end=end)

    def insider_flow(ticker: str, after: pd.Timestamp) -> pd.Series:
        from src.ingestion import form4

        return form4.fetch_form4(ticker, after=after, limit=1000)

    def news_reader(ticker: str, lookback_days: int) -> NewsReader:
        from src.ingestion.news_archive import make_archive_reader

        return make_archive_reader(ticker, lookback_days=lookback_days)

    return Sources(prices, returns_matrix, trends, insider_flow, news_reader)


@dataclass
class Dataset:
    config: DatasetConfig
    features: (
        pd.DataFrame
    )  # NaN-free, columns FEATURE_NAMES, consecutive bars ending at the last bar
    log_returns: pd.Series  # the full price series' log-returns (not trimmed to ``features``)
    prices: pd.DataFrame
    vix: pd.Series | None = None  # the close of ^VIX over the same window, when requested
    regime_classifier: RegimeClassifier | None = None
    dropped_rows: int = 0
    news_coverage: float | None = None  # share of bars with at least one article in the lookback
    notes: list[str] = field(default_factory=list)


def regime_path(model_path: str | Path) -> Path:
    """Where a model's regime classifier is saved: the model file name plus ``.regime.joblib``."""
    p = Path(model_path)
    return p.with_name(p.name + ".regime.joblib")


def bar_close(as_of: datetime, interval: str = "1d") -> datetime:
    """When the information in the daily bar stamped ``as_of`` was complete (20:00 UTC that date)."""
    if interval != "1d":
        raise NotImplementedError(
            f"interval {interval!r} is not supported; see the module docstring"
        )
    return datetime.combine(as_of.date(), NEWS_CUTOFF_UTC, tzinfo=timezone.utc)


def _require_series(value: object, what: str) -> pd.Series:
    if not isinstance(value, pd.Series) or value.empty:
        raise RuntimeError(f"{what} was requested but returned no data")
    if not isinstance(value.index, pd.DatetimeIndex) or value.index.tz is None:
        raise RuntimeError(f"{what} must be indexed by a timezone-aware DatetimeIndex")
    return value.sort_index()


def _require_fresh(
    series: pd.Series, last_bar: pd.Timestamp, max_age: pd.Timedelta, what: str
) -> None:
    """A source that ends long before the last bar would be silently forward-filled as stale data."""
    if series.index.max() < last_bar - max_age:
        raise RuntimeError(
            f"{what} ends {series.index.max():%Y-%m-%d}, more than {max_age.days} days before the "
            f"last bar ({last_bar:%Y-%m-%d}); it is stale"
        )


def build_dataset(
    cfg: DatasetConfig,
    sources: Sources | None = None,
    regime: Literal["none", "fit_all"] = "none",
) -> Dataset:
    """Build the feature matrix for ``cfg``; raises if any requested source is unusable."""
    src = sources or default_sources()
    if regime not in ("none", "fit_all"):
        raise ValueError(f"regime must be 'none' or 'fit_all', got {regime!r}")
    if regime == "fit_all" and not cfg.use_vix:
        raise ValueError("regime='fit_all' needs use_vix=True")

    prices = src.prices(cfg.ticker, cfg.start, cfg.end, cfg.interval)
    if prices is None or prices.empty:
        raise RuntimeError(f"no price data for {cfg.ticker}")
    if not isinstance(prices.index, pd.DatetimeIndex) or prices.index.tz is None:
        raise RuntimeError(f"prices for {cfg.ticker} must have a timezone-aware DatetimeIndex")
    missing_cols = {"close", "volume", "log_return"} - set(prices.columns)
    if missing_cols:
        raise RuntimeError(f"prices for {cfg.ticker} lack columns {sorted(missing_cols)}")
    # The feature code skips NaN returns and volumes, so a hole in the price data would be silently
    # bridged (a "5-day" momentum spanning a gap). A hole is a data problem; refuse it here.
    holes = prices[["close", "volume"]].isna().any(axis=1) | prices["log_return"].iloc[
        1:
    ].isna().reindex(prices.index, fill_value=False)
    if holes.any():
        raise RuntimeError(
            f"price data for {cfg.ticker} has {int(holes.sum())} bar(s) with missing close, volume or "
            f"return (first: {holes.idxmax():%Y-%m-%d})"
        )
    last_bar = prices.index.max()

    if cfg.peers:
        sector_returns = src.returns_matrix(
            [cfg.ticker, *cfg.peers], cfg.start, cfg.end, cfg.interval
        )
        absent = [t for t in (cfg.ticker, *cfg.peers) if t not in sector_returns.columns]
        if absent:
            raise RuntimeError(f"returns matrix is missing {absent}")
    else:
        sector_returns = prices[["log_return"]].rename(columns={"log_return": cfg.ticker})

    vix = None
    if cfg.use_vix:
        vix_frame = src.prices("^VIX", cfg.start, cfg.end, cfg.interval)
        vix = _require_series(vix_frame["close"] if vix_frame is not None else None, "^VIX")
        _require_fresh(vix, last_bar, MAX_VIX_STALENESS, "^VIX")
    trends = None
    if cfg.use_trends:
        trends = _require_series(src.trends(cfg.ticker, cfg.start, cfg.end), "Google Trends")
        _require_fresh(trends, last_bar, MAX_TRENDS_STALENESS, "Google Trends")
        trends = trends.copy()
        trends.index = (
            trends.index + TRENDS_AVAILABILITY_LAG
        )  # a value is usable only once complete
    insider_flow = None
    if cfg.use_insider_flow:
        start_ts = pd.Timestamp(cfg.start) if cfg.start else prices.index[0]
        start_ts = (
            start_ts.tz_localize("UTC") if start_ts.tzinfo is None else start_ts.tz_convert("UTC")
        )
        insider_flow = _require_series(
            src.insider_flow(cfg.ticker, start_ts - INSIDER_LOOKBACK_MARGIN), "Form 4 insider flow"
        )

    regime_classifier = None
    if regime == "fit_all":
        assert vix is not None
        regime_classifier = RegimeClassifier().fit(vix.to_numpy())

    news = {"bars": 0, "with_articles": 0}
    if cfg.use_news_archive:
        reader = src.news_reader(cfg.ticker, cfg.lookback_days)

        def articles_by_time(as_of: datetime) -> list[ScoredArticle]:
            articles = reader(as_of)  # as_of is already the bar close (see below)
            news["bars"] += 1
            news["with_articles"] += bool(articles)
            return articles

    else:

        def articles_by_time(as_of: datetime) -> list[ScoredArticle]:
            return []

    inert = [
        name
        for name, active in [
            ("sentiment_agg/estimated_lag_hours", cfg.use_news_archive),
            ("sector_proximity", bool(cfg.peers)),
            ("regime_label", cfg.use_vix),
            ("search_interest_zscore", cfg.use_trends),
            ("insider_flow_npr", cfg.use_insider_flow),
        ]
        if not active
    ]
    notes = [f"not enabled, so neutral: {', '.join(inert)}"] if inert else []
    if inert:
        logger.warning("dataset %s: features neutral because not enabled: %s", cfg.ticker, inert)

    # Every lookup (prices, VIX, trends, news, and the news recency decay) happens at the bar close.
    close_index = pd.DatetimeIndex(
        [bar_close(ts.to_pydatetime(), cfg.interval) for ts in prices.index], tz="UTC"
    )
    raw = build_feature_matrix(
        cfg.ticker,
        close_index,
        articles_by_time=articles_by_time,
        prices=prices,
        sector_returns=sector_returns,
        vix=vix,
        regime_classifier=regime_classifier,
        trends=trends,
        insider_flow=insider_flow,
    )
    raw.index = prices.index  # rows are keyed by the bar stamp, as targets and returns are

    if cfg.use_news_archive and news["with_articles"] == 0:
        raise RuntimeError(
            f"{cfg.ticker}: the news archive is enabled but no bar had an article in its "
            f"{cfg.lookback_days}-day lookback (missing archive, or articles outside the window)"
        )

    features = raw.dropna()
    if features.empty:
        raise RuntimeError(f"{cfg.ticker}: no complete feature rows ({len(raw)} bars is too few)")
    # Only the warm-up rows at the start may be missing. An interior gap (a NaN volume day, a hole in
    # a series) or a missing latest bar would silently shift or shorten what the model sees.
    pos = prices.index.get_indexer(features.index)
    if (np.diff(pos) != 1).any():
        raise RuntimeError(
            f"{cfg.ticker}: feature rows are not consecutive bars; an interior NaN (gaps in "
            "volume, trends or insider data?) would shift the rows"
        )
    if pos[-1] != len(prices) - 1:
        raise RuntimeError(f"{cfg.ticker}: the latest bar has no complete features")
    if vix is not None:
        aligned = vix.reindex(features.index, method="ffill")
        if aligned.isna().any():
            raise RuntimeError(
                f"^VIX has no value at or before {int(aligned.isna().sum())} feature rows; "
                "it does not cover the price window"
            )
    return Dataset(
        config=cfg,
        features=features,
        log_returns=prices["log_return"],
        prices=prices,
        vix=vix,
        regime_classifier=regime_classifier,
        dropped_rows=len(raw) - len(features),
        news_coverage=(news["with_articles"] / news["bars"]) if news["bars"] else None,
        notes=notes,
    )
