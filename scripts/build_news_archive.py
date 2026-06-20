"""
Populate the point-in-time news archive for a ticker.

Fetches news from a chosen source and writes per-day files under
``data/raw/news/<TICKER>/<YYYY-MM-DD>.json`` (append-only). The archive is then
consumable by ``src.ingestion.news_archive.make_archive_reader`` for
sentiment-driven backtests.

Sources (``--source``):
  - ``newsapi``       NewsAPI (default). Free tier serves only ~30 days of
                      history, so building a long archive requires repeated runs
                      over time (or a paid plan). Needs ``NEWS_API_KEY``.
  - ``alphavantage``  Alpha Vantage NEWS_SENTIMENT. Free tier reaches back to
                      ~2022 and is ticker-native, so ``--start`` / ``--end`` can
                      backfill real history in one run. Needs
                      ``ALPHAVANTAGE_API_KEY``.

Usage:
    python scripts/build_news_archive.py --ticker AAPL --start 2024-01-01
    python scripts/build_news_archive.py --ticker AAPL --source alphavantage \
        --start 2022-01-01 --end 2024-01-01
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# Allow running directly (python scripts/build_news_archive.py) by putting the
# repo root on the path before importing the src package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.news import NewsArticle  # noqa: E402
from src.ingestion.news_archive import write_day  # noqa: E402


def _fetch_articles(
    source: str, ticker: str, start: datetime | None, end: datetime | None
) -> list[NewsArticle]:
    """Dispatch to the chosen ingestion source (imported lazily per source)."""
    if source == "alphavantage":
        from src.ingestion.alpha_vantage_news import fetch_alpha_vantage_news

        return fetch_alpha_vantage_news(ticker, time_from=start, time_to=end)
    if source == "newsapi":
        from src.ingestion.news import fetch_news

        return fetch_news(ticker, since=start)
    raise ValueError(f"Unknown news source: {source!r}")


def build(
    ticker: str,
    source: str = "newsapi",
    start: datetime | None = None,
    end: datetime | None = None,
) -> int:
    articles = _fetch_articles(source, ticker, start, end)
    by_day: dict[datetime, list[NewsArticle]] = defaultdict(list)
    for a in articles:
        day = a.published_at.astimezone(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        by_day[day].append(a)

    written = 0
    for day, day_articles in sorted(by_day.items()):
        path = write_day(ticker, day, day_articles)
        if path is not None:
            written += 1
            print(f"wrote {len(day_articles):3d} articles -> {path}")
        else:
            print(f"skip (exists) {day.date().isoformat()}")
    return written


def _parse_date(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc) if value else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the point-in-time news archive")
    parser.add_argument("--ticker", required=True)
    parser.add_argument(
        "--source",
        default="newsapi",
        choices=["newsapi", "alphavantage"],
        help="ingestion source (default: newsapi)",
    )
    parser.add_argument("--start", default=None, help="ISO date; earliest publish date to keep")
    parser.add_argument(
        "--end", default=None, help="ISO date; latest publish date (alphavantage only)"
    )
    args = parser.parse_args(argv)

    written = build(
        args.ticker,
        source=args.source,
        start=_parse_date(args.start),
        end=_parse_date(args.end),
    )
    print(f"\nDone: {written} new day file(s) written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
