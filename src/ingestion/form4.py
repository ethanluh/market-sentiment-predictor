"""
SEC Form 4 (insider-trading) ingestion.

Returns a timezone-aware (UTC) ``pd.Series`` of *signed shares* for the
discretionary, open-market insider transactions in a ticker's recent Form 4
filings — positive for purchases, negative for sales — indexed by each filing's
acceptance datetime. This feeds the ``insider_flow_npr`` feature
(``prediction.features``), a Net Purchase Ratio of insider buying.

Design notes:

- Only ``P`` (open-market purchase) and ``S`` (open-market sale) non-derivative
  transactions are kept. Grants/awards (A), option exercises (M), tax
  withholding (F), gifts (G) and the whole derivative table are routine /
  non-discretionary and carry no directional signal, so they are excluded.
- Point-in-time gating uses the **filing/acceptance datetime**, not the
  transaction date: an insider may file up to two business days after trading,
  so the data only becomes "known" when EDGAR accepts the filing. The filing
  download + acceptance-date parsing is reused from ``ingestion.filings``.
- Parsing is defensive: a filing whose ownership XML can't be parsed contributes
  nothing rather than raising, consistent with the project's no-guess rule.
"""

from __future__ import annotations

import logging
from pathlib import Path
from xml.etree import ElementTree as ET

import pandas as pd

logger = logging.getLogger("ingestion.form4")

# Discretionary open-market transaction codes carrying directional signal.
_INFORMATIVE_CODES = {"P", "S"}


def _text(node: ET.Element | None) -> str | None:
    """Return stripped text of a node (or its ``<value>`` child), else None."""
    if node is None:
        return None
    value = node.find("value")
    target = value if value is not None else node
    return target.text.strip() if target.text else None


def _extract_ownership_xml(folder: Path) -> str | None:
    """
    Return the raw ``<ownershipDocument>`` XML for a Form 4 accession folder.

    Prefers the clean ``primary-document.xml`` written with
    ``download_details=True``; falls back to slicing the ``<TYPE>4`` document's
    ``<TEXT>`` payload out of the SGML ``full-submission.txt`` wrapper.
    """
    primary = folder / "primary-document.xml"
    if primary.exists():
        try:
            return primary.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return None

    full = folder / "full-submission.txt"
    if not full.exists():
        return None
    try:
        raw = full.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    start = raw.find("<ownershipDocument")
    end = raw.find("</ownershipDocument>")
    if start == -1 or end == -1:
        return None
    return raw[start : end + len("</ownershipDocument>")]


def _parse_form4_transactions(folder: Path) -> list[float]:
    """
    Return signed shares for each P/S non-derivative transaction in a Form 4.

    Positive = acquired (purchase), negative = disposed (sale). Returns ``[]``
    on any parse failure or when no informative transactions are present.
    """
    xml = _extract_ownership_xml(folder)
    if xml is None:
        return []
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        logger.warning("form4: unparseable ownership XML, skipping %s", folder)
        return []

    signed: list[float] = []
    for txn in root.iterfind(".//nonDerivativeTransaction"):
        coding = txn.find("transactionCoding")
        # transactionCode is a *direct* child of transactionCoding (not <value>).
        code = _text(coding.find("transactionCode")) if coding is not None else None
        if code not in _INFORMATIVE_CODES:
            continue
        amounts = txn.find("transactionAmounts")
        if amounts is None:
            continue
        shares_str = _text(amounts.find("transactionShares"))
        if shares_str is None:
            continue
        try:
            shares = float(shares_str)
        except ValueError:
            continue
        ad = _text(amounts.find("transactionAcquiredDisposedCode"))
        sign = -1.0 if ad == "D" else 1.0  # A = acquired/buy, D = disposed/sell
        signed.append(sign * shares)
    return signed


def fetch_form4(
    ticker: str,
    limit: int = 50,
    after: "pd.Timestamp | None" = None,
) -> pd.Series:
    """
    Fetch recent Form 4 filings for ``ticker`` and return a UTC-indexed Series
    of signed insider shares (one row per P/S transaction), named
    ``"insider_net_shares"``. Positive = purchase, negative = sale.

    Reuses :func:`ingestion.filings.fetch_filings` for the download and
    acceptance-datetime parsing, so each transaction is stamped with the filing
    date (look-ahead safe). Raises ``ValueError`` when no informative insider
    transactions are found.
    """
    from src.ingestion.filings import fetch_filings  # reuse download + date parse

    after_dt = after.to_pydatetime() if isinstance(after, pd.Timestamp) else after
    filings = fetch_filings(ticker, form="4", limit=limit, after=after_dt)

    timestamps: list[pd.Timestamp] = []
    shares: list[float] = []
    for filing in filings:
        for signed in _parse_form4_transactions(Path(filing.path)):
            timestamps.append(pd.Timestamp(filing.filed_at))
            shares.append(signed)

    if not shares:
        raise ValueError(f"No Form 4 insider transactions found for {ticker!r}")

    series = pd.Series(shares, index=pd.DatetimeIndex(timestamps), name="insider_net_shares")
    return series.sort_index()
