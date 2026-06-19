"""
SEC filings ingestion via ``sec-edgar-downloader``.

Filings (8-K material events, 10-Q/10-K periodics) are the primary signal
origin and are categorised as ``sec_corp`` nodes in the diffusion graph.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.graph.categories import NODE_SEC_CORP

# Where sec-edgar-downloader writes its filing tree.
EDGAR_DOWNLOAD_ROOT = Path(__file__).resolve().parents[2] / "data" / "raw" / "edgar"

# SEC requires a descriptive User-Agent (company / email).
_DEFAULT_UA_NAME = "market-sentiment-predictor"


@dataclass
class Filing:
    ticker: str
    form: str
    filed_at: datetime  # tz-aware UTC
    path: str
    source_category: str = NODE_SEC_CORP


def _utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def fetch_filings(
    ticker: str,
    form: str = "8-K",
    limit: int = 10,
    after: datetime | None = None,
) -> list[Filing]:
    """
    Download recent ``form`` filings for ``ticker`` and return metadata.

    The downloader email is read from ``SEC_EDGAR_EMAIL`` (falls back to the
    value in ``.env.example`` style usage). Returns up to ``limit`` filings.
    """
    from sec_edgar_downloader import Downloader  # lazy: network

    email = os.environ.get("SEC_EDGAR_EMAIL", "research@example.com")
    EDGAR_DOWNLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    dl = Downloader(_DEFAULT_UA_NAME, email, str(EDGAR_DOWNLOAD_ROOT))

    after_str = _utc(after).date().isoformat() if after is not None else None
    dl.get(form, ticker, limit=limit, after=after_str, download_details=True)

    # The downloader writes to <root>/sec-edgar-filings/<ticker>/<form>/<id>/.
    base = EDGAR_DOWNLOAD_ROOT / "sec-edgar-filings" / ticker.upper() / form
    filings: list[Filing] = []
    if base.exists():
        for entry in sorted(base.iterdir(), reverse=True)[:limit]:
            if not entry.is_dir():
                continue
            filed_at = datetime.fromtimestamp(entry.stat().st_mtime, tz=timezone.utc)
            filings.append(
                Filing(
                    ticker=ticker.upper(),
                    form=form,
                    filed_at=filed_at,
                    path=str(entry),
                )
            )
    return filings
