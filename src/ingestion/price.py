"""
Price / volume ingestion via Yahoo Finance (``yfinance``).

Returns a timezone-aware (UTC) ``pd.DatetimeIndex`` frame with columns
``close``, ``volume`` and ``log_return``. Log-returns are computed here so that
everything downstream of ingestion operates on log-returns, never raw prices
(per the project conventions in CLAUDE.md).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def to_log_returns(close: pd.Series) -> pd.Series:
    """Return the log-returns of a close-price series: ``ln(p_t / p_{t-1})``."""
    return np.log(close / close.shift(1))


def _ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce a DataFrame's index to a timezone-aware UTC ``DatetimeIndex``."""
    idx = pd.DatetimeIndex(df.index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    df = df.copy()
    df.index = idx
    return df


def fetch_prices(
    ticker: str,
    start: str | None = None,
    end: str | None = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Fetch OHLCV from Yahoo Finance and return a frame with ``close``,
    ``volume`` and ``log_return`` columns indexed by a UTC ``DatetimeIndex``.

    Parameters
    ----------
    ticker : str
        Stock symbol, e.g. ``"AAPL"``.
    start, end : str | None
        ISO date strings (``YYYY-MM-DD``). ``None`` uses yfinance defaults.
    interval : str
        Bar size (``"1d"``, ``"1h"``, ...). Intraday is best-effort and subject
        to Yahoo Finance's lookback limits.
    """
    import yfinance as yf  # lazy: heavy + network

    raw = yf.download(
        ticker,
        start=start,
        end=end,
        interval=interval,
        progress=False,
        auto_adjust=True,
    )
    if raw is None or raw.empty:
        raise ValueError(f"No price data returned for {ticker!r}")

    # yfinance may return MultiIndex columns when multiple tickers are passed;
    # flatten to the single-ticker case.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw = _ensure_utc_index(raw)
    out = pd.DataFrame(index=raw.index)
    out["close"] = raw["Close"].astype(float)
    out["volume"] = raw["Volume"].astype(float)
    out["log_return"] = to_log_returns(out["close"])
    return out


def fetch_returns_matrix(
    tickers: list[str],
    start: str | None = None,
    end: str | None = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Fetch log-returns for several tickers and return a wide DataFrame whose
    columns are tickers and index is a shared UTC ``DatetimeIndex``. Used to
    build the sector correlation graph.
    """
    cols: dict[str, pd.Series] = {}
    for tkr in tickers:
        cols[tkr] = fetch_prices(tkr, start, end, interval)["log_return"]
    return pd.DataFrame(cols).dropna(how="all")
