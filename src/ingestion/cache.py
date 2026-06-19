"""
On-disk JSON cache for external API responses.

Keyed by (source, ticker, date) per docs/architecture.md. Files live under
``data/cache/`` which is gitignored. The cache is best-effort: a corrupt or
unreadable entry is treated as a miss rather than raising.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

# Cache root; resolved relative to the repository root (four parents up from
# this file: ingestion -> src -> repo).
CACHE_ROOT = Path(__file__).resolve().parents[2] / "data" / "cache"


def _date_key(when: date | datetime | str) -> str:
    """Normalize a date-like value to an ISO ``YYYY-MM-DD`` string."""
    if isinstance(when, datetime):
        return when.date().isoformat()
    if isinstance(when, date):
        return when.isoformat()
    return str(when)


def cache_path(source: str, ticker: str, when: date | datetime | str) -> Path:
    """Return the cache file path for a ``(source, ticker, date)`` triple."""
    key = _date_key(when)
    return CACHE_ROOT / source / f"{ticker.upper()}_{key}.json"


def read_cache(source: str, ticker: str, when: date | datetime | str) -> Any | None:
    """Return cached JSON payload, or ``None`` on miss / unreadable entry."""
    path = cache_path(source, ticker, when)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def write_cache(source: str, ticker: str, when: date | datetime | str, payload: Any) -> Path:
    """Write ``payload`` as JSON to the cache and return the path written."""
    path = cache_path(source, ticker, when)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    return path
