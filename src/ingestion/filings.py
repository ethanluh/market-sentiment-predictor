"""
SEC filings ingestion via ``sec-edgar-downloader``.

Filings (8-K material events, 10-Q/10-K periodics) are the primary signal
origin and are categorised as ``sec_corp`` nodes in the diffusion graph.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.graph.categories import NODE_SEC_CORP

# Where sec-edgar-downloader writes its filing tree.
EDGAR_DOWNLOAD_ROOT = Path(__file__).resolve().parents[2] / "data" / "raw" / "edgar"

# SEC requires a descriptive User-Agent (company / email).
_DEFAULT_UA_NAME = "market-sentiment-predictor"

# EDGAR SGML header markers carrying the true filing timestamp.
_ACCEPTANCE_RE = re.compile(r"<ACCEPTANCE-DATETIME>(\d{14})")
_FILED_DATE_RE = re.compile(r"FILED AS OF DATE:\s*(\d{8})")


@dataclass
class Filing:
    ticker: str
    form: str
    filed_at: datetime  # tz-aware UTC
    path: str
    source_category: str = NODE_SEC_CORP


def _utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def _parse_filing_date(entry: Path) -> datetime | None:
    """
    Parse the true filing timestamp from an accession folder's EDGAR submission
    header (``<ACCEPTANCE-DATETIME>`` then ``FILED AS OF DATE:``). Returns
    ``None`` if no submission file/header is found.
    """
    candidates = ["full-submission.txt", "primary-document.html", "filing-details.html"]
    for name in candidates:
        path = entry / name
        if not path.exists():
            continue
        try:
            header = path.read_text(encoding="utf-8", errors="ignore")[:8192]
        except OSError:
            continue
        m = _ACCEPTANCE_RE.search(header)
        if m:
            return datetime.strptime(m.group(1), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        m = _FILED_DATE_RE.search(header)
        if m:
            return datetime.strptime(m.group(1), "%Y%m%d").replace(tzinfo=timezone.utc)
    return None


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
        for entry in base.iterdir():
            if not entry.is_dir():
                continue
            # Prefer the real EDGAR filing date; fall back to the download mtime
            # only when the submission header is unavailable.
            filed_at = _parse_filing_date(entry) or datetime.fromtimestamp(
                entry.stat().st_mtime, tz=timezone.utc
            )
            filings.append(
                Filing(
                    ticker=ticker.upper(),
                    form=form,
                    filed_at=filed_at,
                    path=str(entry),
                )
            )
    # Sort by actual filing date (newest first) rather than accession-number string.
    filings.sort(key=lambda f: f.filed_at, reverse=True)
    return filings[:limit]
