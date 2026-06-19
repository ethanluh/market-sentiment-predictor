# Market Sentiment Predictor — Architecture Overview

## Pipeline Overview

1. **Symbol Selection**
  - User specifies a stock ticker (e.g., AAPL).

2. **Data Ingestion**
  - Collects information related to the symbol:
    - News articles
    - Social media posts
    - Financial filings
    - General company information

3. **Indicator Extraction**
  - Extracts key indicators for sentiment analysis, such as:
    - Company performance (earnings, revenue, growth)
    - Management actions (leadership changes, buybacks)
    - Ethical concerns (ESG, controversies)
    - Market/sector trends
    - Product launches or failures
    - Regulatory/legal news

4. **Sentiment Analysis**
  - Applies NLP models (e.g., FinBERT) to each indicator.
  - Labels each indicator as:
    - Very Good
    - Good
    - Neutral
    - Bad
    - Very Bad

5. **Graph Diffusion Modeling**
  - Constructs a sector correlation graph.
  - Propagates sentiment through the graph to model indirect effects (e.g., sector-wide impact, peer influence).

6. **Prediction**
  - Uses the graph-informed sentiment to predict quantile returns for the selected symbol over a chosen horizon.

## Key Design Choices

- **Indicators**: Chosen for their relevance to price movement and news sensitivity. The set can be expanded as needed.
- **Sentiment Categories**: Five-point scale provides granularity for downstream modeling.
- **Graph Theory**: Captures both direct and indirect effects of sentiment, leveraging sector and peer relationships.

## Example Flow

1. User selects TSLA.
2. System ingests news, filings, and social posts.
3. Extracts indicators: e.g., “Q1 earnings,” “autopilot investigation,” “battery tech.”
4. Sentiment model labels “Q1 earnings” as Good, “autopilot investigation” as Bad, etc.
5. Graph model diffuses these sentiments across the auto sector.
6. Quantile regression predicts TSLA’s return distribution for the next day.

# Architecture

## Data Flow

```
External APIs
  └── src/ingestion/
        ├── price.py        → OHLCV → UTC log-returns (yfinance)
        ├── news.py         → articles + source→category mapping (NewsAPI)
        ├── filings.py      → SEC 8-K, 10-Q + real filing dates (sec-edgar-downloader)
        ├── social.py       → Reddit posts (praw)
        ├── news_archive.py → point-in-time per-day news store + leakage-safe reader
        └── cache.py        → JSON API cache

src/sentiment/
  ├── inference.py         → FinBERT forward pass, returns float score in [-1,1]
  └── aggregation.py       → credibility-weighted, recency-decayed score per (ticker, window)

src/graph/
  ├── categories.py        → canonical node category definitions
  ├── diffusion.py         → build G, compute Laplacian, heat kernel diffusion
  └── sector.py            → rolling Pearson correlation graph over sector tickers

src/prediction/
  ├── features.py          → assemble feature vector per (ticker, as_of); no look-ahead
  ├── regime.py            → VIX k-means regime classifier (vol-monotonic labels)
  ├── model.py             → quantile regression per (horizon, quantile); no crossing
  ├── baselines.py         → momentum, ARIMA, EWMA-GARCH + horizon/interval specs
  └── backtest.py          → custom walk-forward harness; model-vs-baseline pinball

src/pipeline/
  ├── run.py               → in-process sequential flow wiring all stages (+ CLI)
  └── api.py               → FastAPI POST /predict, GET /health

scripts/
  ├── train_model.py       → fit + persist a QuantileReturnModel artifact
  ├── build_news_archive.py→ populate the point-in-time news archive
  └── calibrate_graph.py   → estimate edge prob/lag → calibrated_edges.json
```

## Storage

| Layer                            | Store                | Notes                                       |
|---                               |---                   |---                                          |
| Raw price/news                   | TimescaleDB          | Hypertable on timestamp                     |
| Embeddings                       | Postgres (pgvector)  | Optional; for similarity clustering         |
| Graph metadata                   | SQLite               | Node weights, edge lag distributions        |
| Experiments                      | MLflow               | All model runs tracked                      |
| API cache                        | Local JSON files     | Gitignored; keyed by (source, ticker, date) |

## Serving

FastAPI app at `src/pipeline/api.py`. Single endpoint:

```
POST /predict
{
  "ticker": "AAPL",
  "horizons": ["1h", "1d", "5d"]
}
→ {
  "estimated_lag_hours": 2.4,
  "predictions": {
    "1d": {"p10": -0.012, "p50": 0.004, "p90": 0.021}
  },
  "top_articles": [...]
}
```
