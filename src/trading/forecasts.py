"""Forecast sources for the trading CLI."""

from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path

from src.trading.signals import Forecast


def read_universe(path: Path) -> tuple[str, ...]:
    """Tickers one per line; blank lines and ``#`` comments are ignored."""
    tickers = [
        line.strip().upper()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not tickers:
        raise ValueError(f"no tickers in {path}")
    return tuple(tickers)


# A 5-day log-return forecast beyond +/-100% is a data error, not a signal.
MAX_ABS_FORECAST = 1.0
_TOP_LEVEL_KEYS = {"as_of", "horizon", "forecasts"}


def _reject_constant(name: str) -> float:
    raise ValueError(f"forecast file contains the non-JSON constant {name}")


def _number(value: object, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where} must be a JSON number, got {value!r}")
    x = float(value)
    if not math.isfinite(x) or abs(x) > MAX_ABS_FORECAST:
        raise ValueError(f"{where} must be finite and within +/-{MAX_ABS_FORECAST}, got {x}")
    return x


def read_forecast_file(
    path: Path, universe: tuple[str, ...], market_date: date, horizon: str
) -> list[Forecast]:
    """
    Forecasts from a JSON file::

        {"as_of": "2026-01-09", "horizon": "5d",
         "forecasts": {"AAPL": {"p10": -0.02, "p50": 0.01, "p90": 0.04}}}

    ``as_of`` must be the broker's market date (stale forecasts are refused), ``horizon`` must
    equal the configured horizon, unknown top-level keys are refused, every value must be a finite
    JSON number, and every universe ticker must have a forecast: a missing one would read as
    "no signal" and sell a held name.
    """
    doc = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    unknown = sorted(set(doc) - _TOP_LEVEL_KEYS)
    if unknown or not _TOP_LEVEL_KEYS <= set(doc):
        raise ValueError(
            f"forecast file keys must be {sorted(_TOP_LEVEL_KEYS)}; unexpected={unknown}"
        )
    if doc["horizon"] != horizon:
        raise ValueError(f"forecast file is for horizon {doc['horizon']!r}, expected {horizon!r}")
    as_of = date.fromisoformat(doc["as_of"])
    if as_of != market_date:
        raise ValueError(f"forecast file is for {as_of}, but the market date is {market_date}")
    raw = doc["forecasts"]
    missing = sorted(set(universe) - set(raw))
    extra = sorted(set(raw) - set(universe))
    if missing or extra:
        raise ValueError(f"forecast file mismatch: missing={missing}, outside universe={extra}")
    return [
        Forecast(
            t,
            _number(raw[t]["p10"], f"{t}.p10"),
            _number(raw[t]["p50"], f"{t}.p50"),
            _number(raw[t]["p90"], f"{t}.p90"),
        )
        for t in universe
    ]
