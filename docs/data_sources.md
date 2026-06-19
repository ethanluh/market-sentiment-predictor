# Data Sources

**Status legend:** ✅ implemented · 🔜 planned / not yet wired.

## Price / Volume

| Source | Library | Granularity | Notes |
|---|---|---|---|
| Yahoo Finance ✅ | `yfinance` | 1m – 1d | Default; daily + intraday |
| Polygon.io 🔜 | REST API | 1s – 1d | Production; cleaner rate limits |

## News

Fetched via NewsAPI (`src/ingestion/news.py`); outlet → category mapping drives
credibility weighting in `src/sentiment/aggregation.py`.

| Source | Credibility Weight | Notes |
|---|---|---|
| Reuters / Bloomberg ✅ | 1.0 | Highest signal quality |
| WSJ / FT ✅ | 0.9 | |
| Benzinga ✅ | 0.7 | Mapped via NewsAPI |
| Seeking Alpha ✅ | 0.5 | Mapped via NewsAPI |
| Financial blogs ✅ | informed-retail | Apply skepticism |

Recency decay: `w(t) = credibility * exp(-lambda * delta_t_hours)`
Default lambda per category in `src/sentiment/aggregation.py`.

## SEC Filings

Fetched via `sec-edgar-downloader` (`src/ingestion/filings.py`); used as the
`sec_corp` diffusion origin.

| Filing | Signal | Status |
|---|---|---|
| 8-K | Material events (earnings, M&A, leadership) | ✅ |
| 10-Q / 10-K | Periodic financials | ✅ |
| Form 4 | Insider trading disclosures | 🔜 |

## Social

| Source | Library | Status |
|---|---|---|
| Reddit (WSB, investing, stocks) | `praw` | ✅ retail sentiment |
| Google Trends | `pytrends` | 🔜 retail attention proxy |

## Earnings Transcripts 🔜

Planned: Finnhub API (`/stock/earnings`) or Motley Fool, parsing the Q&A section
separately — management tone in Q&A diverges from prepared remarks. Not yet wired.
