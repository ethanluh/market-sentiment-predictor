# market-sentiment-predictor

A news-driven stock price prediction pipeline combining financial NLP, graph-theoretic information diffusion modeling, and quantile regression over returns.

## Pipeline Overview

```
News / Filings / Social
        │
        ▼
  [1] Ingestion
  yfinance · Polygon · EDGAR · NewsAPI · praw
        │
        ▼
  [2] Sentiment Classification
  FinBERT → continuous score · source-credibility weighting · recency decay
        │
        ▼
  [3] Graph Diffusion (impact timing)
  Information diffusion graph (heat kernel on Laplacian)
  Sector correlation graph (rolling Pearson on log-returns)
  → estimated_lag · sector_proximity features
        │
        ▼
  [4] Quantile Return Prediction
  Predict P10 / P50 / P90 of log-return at 1h, 1d, 5d horizons
  Conditioned on VIX regime
        │
        ▼
  [5] Backtesting & Serving
  vectorbt backtest · FastAPI endpoint
```

## Quickstart

```bash
git clone https://github.com/<you>/market-sentiment-predictor
cd market-sentiment-predictor

pip install -r requirements.txt

cp .env.example .env
# Fill in API keys

# Run pipeline for a single ticker
python -m src.pipeline.run --ticker AAPL --horizon 1d

# Backtest
python -m src.prediction.backtest --ticker AAPL --start 2023-01-01 --end 2024-01-01

# Run tests
pytest tests/ -v
```

## Project Structure

```
src/
  ingestion/      # price/volume, news, SEC filings, social signals
  sentiment/      # FinBERT inference, aggregation
  graph/          # diffusion model, sector graph, node categories
  prediction/     # quantile regression, baselines, backtesting
  pipeline/       # Prefect orchestration flows
docs/             # architecture, graph model details, data sources
tests/            # mirrors src/
scripts/          # utilities, one-off data pulls
data/
  raw/            # append-only raw fetches
  processed/      # derived features and embeddings
  cache/          # API cache (gitignored)
```

## Key Design Decisions

**Continuous sentiment scores** — FinBERT logits are kept as floats in [-1, 1] throughout; discretization only happens at the final output layer if needed.

**Graph-based impact timing** — A directed weighted graph over actor categories (SEC/Corp → Institutional → Financial Press → Informed Retail → Uninformed Retail) models information diffusion via a heat kernel on the graph Laplacian. The time at which the institutional node reaches 50% peak concentration predicts the lag before price impact.

**Quantile regression output** — Predicts the P10/P50/P90 of log-returns at each horizon rather than binary direction. This supports position sizing and threshold-based entry.

**Regime conditioning** — VIX is discretized into low/medium/high regimes (k-means, k=3) and used as a conditioning variable, since sentiment predictiveness varies significantly with volatility.

## Dependencies

See `requirements.txt`. Core:
- `transformers`, `torch` — FinBERT inference
- `yfinance`, `sec-edgar-downloader`, `newsapi-python`, `praw` — data ingestion
- `networkx`, `scipy` — graph construction and Laplacian diffusion
- `scikit-learn`, `statsmodels` — quantile regression, ARIMA/GARCH baselines
- `vectorbt` — backtesting
- `prefect` — pipeline orchestration
- `mlflow` — experiment tracking
- `fastapi`, `uvicorn` — serving

## References

- FinBERT: [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert)
- Information diffusion on graphs: Kempe et al. (2003), "Maximizing the Spread of Influence through a Social Network"
- Heat kernel on graphs: Kondor & Lafferty (2002)
