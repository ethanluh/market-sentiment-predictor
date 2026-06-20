"""Bounded retry-with-backoff helper for transient (network) failures.

Ingestion clients hit external APIs that occasionally blip (timeouts, 5xx, rate
limits). ``retry_call`` retries a thunk a few times with exponential backoff so a
transient error neutralizes a feature only after genuine repeated failure rather
than on the first hiccup. The final exception is re-raised so callers (e.g.
:func:`src.utils.safe.safe_fetch`) can still decide how to degrade.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, TypeVar

logger = logging.getLogger("utils.retry")

T = TypeVar("T")


def retry_call(
    fn: Callable[[], T],
    *,
    attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call ``fn`` up to ``attempts`` times, backing off exponentially between tries.

    Returns the first successful result. Re-raises the last exception if every
    attempt fails. ``sleep`` is injectable so tests can run without real delays.
    """
    if attempts < 1:
        raise ValueError("attempts must be >= 1")

    last_exc: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except exceptions as exc:
            last_exc = exc
            if attempt == attempts:
                break
            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            logger.warning(
                "attempt %d/%d failed (%s); retrying in %.2fs", attempt, attempts, exc, delay
            )
            sleep(delay)

    assert last_exc is not None  # loop ran at least once (attempts >= 1)
    raise last_exc
