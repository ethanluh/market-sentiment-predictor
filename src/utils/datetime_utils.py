"""
Shared datetime helpers.

The pipeline standardizes on timezone-aware UTC timestamps everywhere downstream
of ingestion, so these helpers centralize the small coercion/parsing snippets
that were previously duplicated across modules (``features._coerce_utc``,
``filings._utc``, ``cache._date_key``, ``news._parse_published``).
"""

from __future__ import annotations

from datetime import date, datetime, timezone


def to_utc(dt: datetime) -> datetime:
    """Coerce a (possibly naive) datetime to tz-aware UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def parse_iso_utc(value: str) -> datetime:
    """Parse an ISO-8601 timestamp (accepting a trailing ``Z``) as tz-aware UTC."""
    return to_utc(datetime.fromisoformat(value.replace("Z", "+00:00")))


def to_date_key(when: date | datetime | str) -> str:
    """Normalize a date-like value to an ISO ``YYYY-MM-DD`` string."""
    # datetime is a subclass of date, so check it first.
    if isinstance(when, datetime):
        return when.date().isoformat()
    if isinstance(when, date):
        return when.isoformat()
    return str(when)
