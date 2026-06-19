# Testing the News-Driven Thesis

This project's hypothesis is that **FinBERT news sentiment + graph-diffusion lag
estimation adds predictive edge over price-only baselines**. Validating it
requires the news-sentiment path to be *live* during a walk-forward backtest, not
neutralized. This guide is the runbook for that.

## What "live" vs "neutralized" means

`src.prediction.backtest` runs end-to-end on real price history with no news
setup, but it logs a warning listing the features it has **neutralized** (held at
neutral values). A price-only run neutralizes:

- `sentiment_agg`, `estimated_lag_hours` — the core news/graph signal
- `sector_proximity`, `regime_label`, `search_interest_zscore`, `insider_flow_npr`

Each is activated by an opt-in flag (see below). The thesis is only under test
once `sentiment_agg` / `estimated_lag_hours` drop off the neutralized list.

## Prerequisites

### 1. A news API key

The archive builder supports two providers (`--source`); pick by how much history
you need. See `docs/data_sources.md` for the full comparison.

- **Alpha Vantage** (recommended for a real test) — free key at
  **https://www.alphavantage.co/support/#api-key**. Its `NEWS_SENTIMENT` feed is
  ticker-native and reaches back to **~2022**, so you can backfill real history in
  one run. Set it as `ALPHAVANTAGE_API_KEY` (free tier: 25 requests/day).
- **NewsAPI** (simplest) — free key at **https://newsapi.org/register** (shown on
  your dashboard at **https://newsapi.org/account**). Set it as `NEWS_API_KEY`.

> ⚠️ **The binding constraint is news history.** The NewsAPI free ("Developer")
> plan only serves roughly the **last 30 days** and is non-commercial, capping a
> free backtest to ~20 trading days (low statistical power). For a multi-year,
> statistically meaningful test, either use **Alpha Vantage** to backfill history
> directly, pay for a NewsAPI archival plan, or run
> `scripts/build_news_archive.py` on a schedule (see `scripts/cron_news_archive.sh`)
> to **accumulate** news going forward — the archive under `data/raw/news/` is
> append-only by design.

### 2. The full runtime stack (FinBERT / torch, ~2 GB)

```bash
python3 -m venv .venv && source .venv/bin/activate   # Python 3.11+ required
pip install -r requirements.txt                      # NOT requirements-dev.txt
```

`requirements-dev.txt` deliberately omits `torch`/`transformers`, so it cannot
run the sentiment path — the full `requirements.txt` is required here.

## Runbook

### Step 1 — configure keys

```bash
cp .env.example .env
# edit .env:  ALPHAVANTAGE_API_KEY=<key> (or NEWS_API_KEY=<key>)  SEC_EDGAR_EMAIL=<email>
export $(grep -v '^#' .env | xargs)
```

### Step 2 — populate the point-in-time news archive

Writes per-day files to `data/raw/news/<TICKER>/<YYYY-MM-DD>.json` (append-only).

```bash
# Alpha Vantage: backfill real history in one run (recommended)
python scripts/build_news_archive.py --ticker AAPL --source alphavantage \
    --start 2022-01-01 --end 2024-01-01

# NewsAPI: free tier only reaches back ~30 days, so pick a recent --start
python scripts/build_news_archive.py --ticker AAPL --start 2024-05-20
```

With NewsAPI, re-run periodically (or via `scripts/cron_news_archive.sh`) to grow
the archive over time.

### Step 3 — backtest with the full thesis active

```bash
python -m src.prediction.backtest --ticker AAPL \
    --start 2024-05-20 --horizon 1d \
    --use-news-archive --lookback-days 7 \
    --use-vix --peers MSFT GOOG --use-trends --use-insider-flow
```

FinBERT scores are cached to `data/processed/news_scores/<TICKER>.json`, so
walk-forward folds do not re-run the model on every fold.

## Reading the result

1. **Confirm the signal is live:** `sentiment_agg` / `estimated_lag_hours` should
   no longer appear in the "neutralized" warning line.
2. **Compare against baselines:** the run prints `model` vs `momentum` / `arima` /
   `garch` P50 pinball loss (lower is better). The thesis is supported if the
   model now **beats** the baselines it loses to in a price-only run.
3. **Sanity checks:** interval coverage of `[P10, P90]` should sit near ~0.80;
   directional hit-rate should be above 0.50; the PnL Sharpe should be positive.

## Note

The pipeline machinery is fully built and wired; the remaining variable for a
rigorous verdict is **news-history depth**. Alpha Vantage's free tier reaches back
to ~2022 (rate-limited to 25 requests/day, and a single call returns up to 1000
articles, so very wide windows may need to be fetched in chunks); NewsAPI's free
tier is limited to ~30 days. A free key supports a genuine test today; the widest,
densest coverage still favors a paid archival plan.
