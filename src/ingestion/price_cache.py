"""
Price snapshots, so an evaluation can be reproduced exactly.

Yahoo's adjusted history is revised over time (splits, dividends, corrections), so fetching the same
window on two days can give different returns. ``cached_sources`` wraps the price fetchers: with no
snapshot it fetches fresh data and saves it under today's date (reusing it for the rest of the day);
with an explicit snapshot date it reads exactly that saved data and raises if it is missing, so a
"reproduction" can never silently re-fetch.
"""

from __future__ import annotations

import os
import re
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Callable

import pandas as pd

from src.prediction.dataset import Sources

DEFAULT_ROOT = Path("data/cache/prices")


def _key(ticker: str, start: str | None, end: str | None, interval: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", ticker)
    return f"{safe}_{interval}_{start or 'min'}_{end or 'max'}"


def cached_sources(
    base: Sources,
    root: Path = DEFAULT_ROOT,
    snapshot: date | None = None,
    today: Callable[[], date] = date.today,
) -> tuple[Sources, Callable[[], date]]:
    """
    Return ``base`` with price fetching cached, and a function reporting which snapshot date the
    data came from (known after the first fetch; it is the explicit one, or today's).
    """
    used: set[date] = set()

    def prices(ticker: str, start: str | None, end: str | None, interval: str) -> pd.DataFrame:
        folder = root / _key(ticker, start, end, interval)
        day = snapshot or today()
        path = folder / f"{day.isoformat()}.csv"
        if path.exists():
            frame = pd.read_csv(path, index_col=0)
            frame.index = pd.to_datetime(frame.index, utc=True)
            used.add(day)
            return frame
        if snapshot is not None:
            raise FileNotFoundError(
                f"no price snapshot for {ticker} on {snapshot} at {path}; fetch without a snapshot first"
            )
        fresh = base.prices(ticker, start, end, interval)
        if end is None:
            # "Up to now" includes today's still-forming bar when run during the session. Dropping it
            # keeps a snapshot from freezing a partial close and volume that would then "reproduce".
            fresh = fresh[fresh.index.date < day]
        folder.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(
            f".csv.{os.getpid()}.tmp"
        )  # per-process, so concurrent runs cannot interleave
        fresh.to_csv(tmp)
        tmp.replace(path)
        used.add(day)
        return fresh

    def returns_matrix(
        tickers: list[str], start: str | None, end: str | None, interval: str
    ) -> pd.DataFrame:
        # Same semantics as price.fetch_returns_matrix, but through the cached fetcher.
        cols = {t: prices(t, start, end, interval)["log_return"] for t in tickers}
        return pd.DataFrame(cols).dropna(how="all")

    def snapshot_used() -> date:
        if not used:
            raise RuntimeError("no prices have been fetched yet")
        if len(used) > 1:
            raise RuntimeError(
                f"prices came from several snapshot dates {sorted(used)} (the run crossed midnight); "
                "rerun, or replay a single date with --snapshot"
            )
        return next(iter(used))

    return replace(base, prices=prices, returns_matrix=returns_matrix), snapshot_used
