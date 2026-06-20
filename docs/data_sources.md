# Data Sources

**Status legend:** ✅ implemented · 🔜 planned / not yet wired.

## Price / Volume

| Source | Library | Granularity | Notes |
|---|---|---|---|
| Yahoo Finance ✅ | `yfinance` | 1m – 1d | Default; daily + intraday |
| Polygon.io 🔜 | REST API | 1s – 1d | Production; cleaner rate limits |

## News

Two interchangeable providers, both emitting the same `NewsArticle` type; the
outlet → category mapping (`src/ingestion/news.py`) drives credibility weighting
in `src/sentiment/aggregation.py`. Provider sentiment scores (when present) are
ignored — every article is (re)scored by FinBERT for consistency.

| Provider | Module | Free history | Notes |
|---|---|---|---|
| NewsAPI ✅ | `src/ingestion/news.py` | ~30 days | Default; `NEWS_API_KEY`. Long archives require accumulating over time. |
| Alpha Vantage ✅ | `src/ingestion/alpha_vantage_news.py` | ~2022 → now | `NEWS_SENTIMENT`; ticker-native; `ALPHAVANTAGE_API_KEY`; can backfill history in one run. Free tier 25 req/day. |

Select the provider when building the archive:
`python scripts/build_news_archive.py --ticker AAPL --source alphavantage --start 2022-01-01`.

| Outlet | Credibility Weight | Notes |
|---|---|---|
| Reuters / Bloomberg ✅ | 1.0 | Highest signal quality |
| WSJ / FT ✅ | 0.9 | |
| Benzinga ✅ | 0.7 | Mapped from source name |
| Seeking Alpha ✅ | 0.5 | Mapped from source name |
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
| Form 4 | Insider trading disclosures | ✅ |

Form 4 feeds the `insider_flow_npr` feature (`src/ingestion/form4.py`): a Net
Purchase Ratio `(buys − sells) / (buys + sells)` over a trailing 90-day window,
restricted to discretionary open-market codes (P/S) and gated by the SEC
acceptance datetime (no look-ahead). The pipeline degrades gracefully to a
neutral value when no filings are returned.

## Social

| Source | Library | Status |
|---|---|---|
| Reddit (WSB, investing, stocks) | `praw` | ✅ retail sentiment |
| Google Trends | `pytrends` | ✅ retail attention proxy (`search_interest_zscore` feature) |

Google Trends feeds a point-in-time `search_interest_zscore` feature (a z-scored
*attention* signal, distinct from sentiment) via `src/ingestion/trends.py`. The
endpoint is unofficial: windows longer than ~9 months return weekly (not daily)
granularity, and it may rate-limit — the pipeline degrades gracefully to a
neutral value when no data is returned.

## Earnings Transcripts 🔜

Planned: Finnhub API (`/stock/earnings`) or Motley Fool, parsing the Q&A section
separately — management tone in Q&A diverges from prepared remarks. Not yet wired.
