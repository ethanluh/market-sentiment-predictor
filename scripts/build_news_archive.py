"""
Populate the point-in-time news archive for a ticker.

Fetches news via the NewsAPI client and writes per-day files under
``data/raw/news/<TICKER>/<YYYY-MM-DD>.json`` (append-only). The archive is then
consumable by ``src.ingestion.news_archive.make_archive_reader`` for
sentiment-driven backtests.

Usage:
    python scripts/build_news_archive.py --ticker AAPL --start 2024-01-01

Note: the NewsAPI free tier only serves roughly the last 30 days of history, so
building a long archive requires repeated runs over time (or a paid plan).
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

from src.ingestion.news import NewsArticle, fetch_news  # noqa: E402
from src.ingestion.news_archive import write_day  # noqa: E402


def build(ticker: str, since: datetime | None) -> int:
    articles = fetch_news(ticker, since=since)
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the point-in-time news archive")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--start", default=None, help="ISO date; earliest publish date to keep")
    args = parser.parse_args(argv)

    since = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc) if args.start else None
    written = build(args.ticker, since)
    print(f"\nDone: {written} new day file(s) written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
