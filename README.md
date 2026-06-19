# market-sentiment-predictor

[![CI](https://github.com/ethanluh/market-sentiment-predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/ethanluh/market-sentiment-predictor/actions/workflows/ci.yml)

A news-driven stock price prediction pipeline combining financial NLP, graph-theoretic information diffusion modeling, and quantile regression over returns.

## Pipeline Overview

```
News / Filings / Social
        │
        ▼
  [1] Ingestion (multi-source)
  prices (yfinance) · news (NewsAPI) · filings (EDGAR) · social (Reddit/praw)
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
  custom walk-forward backtest · FastAPI endpoint
```

## Quickstart

```bash
git clone https://github.com/ethanluh/market-sentiment-predictor
cd market-sentiment-predictor

pip install -r requirements.txt        # full runtime (incl. FinBERT)
# or, for tests/CI only (no torch/network clients):
pip install -r requirements-dev.txt

cp .env.example .env
# Fill in API keys (NEWS_API_KEY, REDDIT_*, SEC_EDGAR_EMAIL, ...)

# Train a model artifact (price/technical features; --peers activates sector_proximity)
python scripts/train_model.py --ticker AAPL --start 2022-01-01 --out models/aapl.joblib

# Run the pipeline for a single ticker (multi-source ingest -> quantile prediction)
python -m src.pipeline.run --ticker AAPL --horizon 1d --model-path models/aapl.joblib

# Backtest (prints model-vs-baseline pinball losses)
python -m src.prediction.backtest --ticker AAPL --start 2023-01-01 --end 2024-01-01

# Serve the prediction API
python -m src.pipeline.api          # POST /predict, GET /health on :8000

# Run tests
pytest tests/ -v
```

### Optional workflows

```bash
# Build a point-in-time news archive (for sentiment-driven backtests)
python scripts/build_news_archive.py --ticker AAPL --start 2024-01-01

# Calibrate diffusion edge probabilities/lags from historical events,
# then feed them back via categories.load_calibrated_edges(...)
python scripts/calibrate_graph.py --events-file data/raw/events.csv
```

## Project Structure

```
src/
  ingestion/      # price/volume, news, SEC filings, social, point-in-time news archive
  sentiment/      # FinBERT inference, credibility/recency aggregation
  graph/          # diffusion model, sector graph, node categories + calibration loader
  prediction/     # features, regime, quantile model, baselines, walk-forward backtest
  pipeline/       # in-process orchestration flow (run) + FastAPI app (api)
  utils/          # shared datetime helpers
docs/             # architecture, graph model, data sources, deploy
tests/            # mirrors src/
scripts/          # train_model, build_news_archive, calibrate_graph
data/
  raw/            # append-only raw fetches (incl. news/<TICKER>/<date>.json archive)
  processed/      # derived features, cached news sentiment scores
  cache/          # API cache (gitignored)
models/           # trained model artifacts (gitignored)
```

## Key Design Decisions

**Continuous sentiment scores** — FinBERT logits are kept as floats in [-1, 1] throughout; discretization only happens at the final output layer if needed.

**Graph-based impact timing** — A directed weighted graph over actor categories (SEC/Corp → Institutional → Financial Press → Informed Retail → Uninformed Retail) models information diffusion via a heat kernel on the graph Laplacian. The time at which the institutional node reaches 50% peak concentration predicts the lag before price impact.

**Quantile regression output** — Predicts the P10/P50/P90 of log-returns at each horizon rather than binary direction. This supports position sizing and threshold-based entry.

**Regime conditioning** — VIX is discretized into low/medium/high regimes (k-means, k=3) and used as a conditioning variable, since sentiment predictiveness varies significantly with volatility.

**Multi-source ingest** — The pipeline fans in news (financial press), Reddit (retail), and SEC filings. News + social are scored by FinBERT; a recent filing forces the diffusion origin to `sec_corp` (the fastest edge). Each supplementary source degrades gracefully if unavailable.

**Baseline benchmarking** — The walk-forward backtest scores the quantile model against momentum / ARIMA / EWMA-GARCH baselines (pinball loss), so reported performance is always relative.

**Calibrated diffusion** — `scripts/calibrate_graph.py` estimates per-edge transmission probabilities and lags from historical events; `categories.load_calibrated_edges` feeds them back into the graph used by `estimate_lag`.

**Lightweight stack** — Orchestration is an in-process sequential flow (`src/pipeline/run.py`, no Prefect) and backtesting is a custom walk-forward harness (`src/prediction/backtest.py`, no vectorbt), keeping the dependency footprint to the scientific stack + FastAPI + FinBERT.

## Development

CI (GitHub Actions, `.github/workflows/ci.yml`) runs `black --check`, `isort --check-only`, `mypy src`, and `pytest` on every push/PR using the lightweight `requirements-dev.txt` (heavy/network deps are mocked in tests).

```bash
black src tests scripts && isort src tests scripts
mypy src
pytest tests -q
```

## Dependencies

See `requirements.txt`. Core:
- `transformers`, `torch` — FinBERT inference
- `yfinance`, `sec-edgar-downloader`, `newsapi-python`, `praw` — data ingestion
- `networkx`, `scipy` — graph construction and Laplacian diffusion
- `scikit-learn`, `statsmodels` — quantile regression, ARIMA/GARCH-like baselines
- `fastapi`, `uvicorn` — serving

## References

- FinBERT: [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert)
- Information diffusion on graphs: Kempe et al. (2003), "Maximizing the Spread of Influence through a Social Network"
- Heat kernel on graphs: Kondor & Lafferty (2002)
