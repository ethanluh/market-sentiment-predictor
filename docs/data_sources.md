# Data Sources

## Price / Volume

| Source | Library | Granularity | Notes |
|---|---|---|---|
| Yahoo Finance | `yfinance` | 1m – 1d | Free; use for prototyping |
| Polygon.io | REST API | 1s – 1d | Production; cleaner rate limits |

Pull: OHLCV, VWAP, options chain snapshot (for flow analysis).

## News

| Source | Library | Credibility Weight | Notes |
|---|---|---|---|
| Reuters / Bloomberg | NewsAPI | 1.0 | Highest signal quality |
| WSJ / FT | NewsAPI | 0.9 | |
| Benzinga | Benzinga API | 0.7 | Good for earnings coverage |
| Seeking Alpha | RSS | 0.5 | Variable quality |
| Financial blogs | NewsAPI | 0.3 | Apply skepticism |

Recency decay: `w(t) = credibility * exp(-lambda * delta_t_hours)`
Default lambda per category in `src/sentiment/aggregation.py`.

## SEC Filings

| Filing | Signal | Fetch |
|---|---|---|
| 8-K | Material events (earnings, M&A, leadership) | `sec-edgar-downloader` |
| 10-Q / 10-K | Periodic financials | `sec-edgar-downloader` |
| Form 4 | Insider trading disclosures | SEC EDGAR full-text search |

## Social

| Source | Library | Notes |
|---|---|---|
| Reddit (WSB, investing, stocks) | `praw` | Noisy; useful for retail sentiment |
| Google Trends | `pytrends` | Retail attention proxy; lagged signal |

## Earnings Transcripts

Finnhub API (`/stock/earnings`) or Motley Fool. Parse Q&A section separately — management tone in Q&A diverges from prepared remarks and carries additional signal.
