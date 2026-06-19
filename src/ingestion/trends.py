"""
Google Trends search-interest ingestion via ``pytrends``.

Returns a timezone-aware (UTC) ``pd.Series`` of relative search interest
(0–100) for a ticker's query, named ``"search_interest"``. This is a *retail
attention* proxy — distinct from sentiment — consumed downstream as the
``search_interest_zscore`` feature (see ``prediction.features``).

Google Trends is unofficial: long windows return weekly (not daily) granularity
and the endpoint may rate-limit. Callers that need graceful degradation (the
pipeline) wrap this in a try/except; the feature falls back to a neutral value
when no series is available.
"""

from __future__ import annotations

import pandas as pd


def _ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce a DataFrame's index to a timezone-aware UTC ``DatetimeIndex``."""
    idx = pd.DatetimeIndex(df.index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    df = df.copy()
    df.index = idx
    return df


def _resolve_timeframe(start: str | None, end: str | None, timeframe: str | None) -> str:
    """Pick the pytrends ``timeframe`` string from the supplied arguments."""
    if timeframe is not None:
        return timeframe
    if start is not None and end is not None:
        return f"{start} {end}"
    if start is not None:
        return f"{start} {pd.Timestamp.now('UTC'):%Y-%m-%d}"
    return "today 12-m"


def fetch_trends(
    ticker: str,
    start: str | None = None,
    end: str | None = None,
    *,
    geo: str = "",
    timeframe: str | None = None,
    keyword: str | None = None,
) -> pd.Series:
    """
    Fetch Google search-interest (0–100) for ``ticker`` and return a UTC-indexed
    float ``Series`` named ``"search_interest"``.

    Parameters
    ----------
    ticker : str
        Stock symbol, e.g. ``"AAPL"``. Used as the search query unless
        ``keyword`` overrides it.
    start, end : str | None
        ISO date strings (``YYYY-MM-DD``). When both are given they form the
        pytrends ``timeframe``; otherwise ``timeframe`` (or its default) is used.
    geo : str
        Two-letter geo code (``""`` = worldwide).
    timeframe : str | None
        Explicit pytrends timeframe (e.g. ``"today 12-m"``). Overrides
        ``start``/``end`` when provided.
    keyword : str | None
        Search term; defaults to ``ticker``.

    Notes
    -----
    No API key is required. Long windows return weekly granularity, so the
    resulting index is not guaranteed to align to a daily trading grid.
    """
    term = keyword or ticker
    tf = _resolve_timeframe(start, end, timeframe)

    try:
        from pytrends.request import TrendReq  # lazy: optional dep + network
    except ImportError as exc:
        raise RuntimeError(
            "pytrends is required for fetch_trends; install it with "
            "`pip install pytrends` (see requirements.txt)"
        ) from exc

    client = TrendReq()
    client.build_payload([term], timeframe=tf, geo=geo)
    try:
        raw = client.interest_over_time()
    except Exception as exc:  # pytrends raises various errors on rate-limit/empty
        raise ValueError(f"No Google Trends data returned for {ticker!r}") from exc

    if raw is None or raw.empty or term not in raw.columns:
        raise ValueError(f"No Google Trends data returned for {ticker!r}")

    raw = _ensure_utc_index(raw)
    return raw[term].astype(float).rename("search_interest")
