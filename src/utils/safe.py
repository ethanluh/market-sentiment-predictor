"""Graceful-degrade helper for optional data-source fetches."""

from __future__ import annotations

import logging
from typing import Callable, TypeVar

logger = logging.getLogger("utils.safe")

T = TypeVar("T")


def safe_fetch(source: str, fn: Callable[[], T]) -> T | None:
    """Run a supplementary source fetch, degrading to ``None`` on any failure.

    Network errors / missing credentials / rate limits for one optional source
    (e.g. no Reddit creds, a Google Trends 429) should neutralize only that
    feature rather than abort the whole run. Returns the fetched value on
    success, or ``None`` (with a logged warning) on any exception.
    """
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 - intentional broad degrade
        logger.warning("source %s unavailable (%s); skipping", source, exc)
        return None
