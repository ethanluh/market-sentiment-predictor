# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

News-driven stock price prediction pipeline: multi-source ingestion → FinBERT sentiment → graph diffusion (impact timing) → quantile return prediction → backtest/serve. Python 3.11+, `pip` (no conda). Type hints required on all public functions (`mypy` runs with `disallow_untyped_defs = true`).

## Commands

```bash
# Install: full runtime (incl. torch/FinBERT + network clients) for live runs
pip install -r requirements.txt
# ...or the lightweight set for tests/lint/CI (no torch, no network clients)
pip install -r requirements-dev.txt

# Tests (config in pyproject.toml: testpaths=tests, -v --tb=short)
pytest tests/ -v
pytest tests/prediction/test_model.py -v            # single file
pytest tests/prediction/test_model.py::test_name    # single test

# Lint / format (line-length 100, isort profile=black) + type check
black src tests scripts && isort src tests scripts
mypy src

# Train a model artifact (enrichment features are opt-in flags; each degrades
# gracefully if its source is unavailable — see README for the full list):
python scripts/train_model.py --ticker AAPL --start 2022-01-01 --out models/aapl.joblib \
    --peers MSFT GOOG --use-trends --use-insider-flow --use-vix

# Run the pipeline for one ticker; backtest vs baselines; serve the API
python -m src.pipeline.run --ticker AAPL --horizon 1d --model-path models/aapl.joblib
python -m src.prediction.backtest --ticker AAPL --start 2023-01-01 --end 2024-01-01
python -m src.pipeline.api          # POST /predict, GET /health on :8000
```

## Architecture

The pipeline is five sequential stages; `src/pipeline/run.py` orchestrates them **in-process** (no Prefect), and `src/prediction/backtest.py` is a **custom walk-forward harness** (no vectorbt). Keeping the stack to the scientific libraries + FastAPI + FinBERT is a deliberate choice.

1. **Ingestion** (`src/ingestion/`) — prices (yfinance), news (NewsAPI), SEC filings (EDGAR), Form 4 insider flow, social (Reddit/praw), Google Trends, and a point-in-time news archive (`news_archive.py`). `cache.py` wraps API responses.
2. **Sentiment** (`src/sentiment/`) — `inference.py` runs FinBERT; `aggregation.py` combines sources with credibility weighting + recency decay. Scores stay continuous in [-1, 1] — never discretize before the final prediction step.
3. **Graph** (`src/graph/`) — `diffusion.py` runs a heat kernel on the graph Laplacian over actor categories (SEC/Corp → Institutional → Press → Informed/Uninformed Retail) to estimate the lag before price impact; `sector.py` builds a rolling-Pearson correlation graph for `sector_proximity`. **Node categories live only in `src/graph/categories.py` — do not hardcode category strings elsewhere.** `calibrate_graph.py` + `categories.load_calibrated_edges` feed historically-estimated edge probabilities/lags back in.
4. **Prediction** (`src/prediction/`) — `features.py` assembles features, `regime.py` discretizes VIX into low/med/high (k-means, k=3) as a conditioning variable, `model.py` predicts P10/P50/P90 of log-return at 1h/1d/5d, and `baselines.py` provides momentum/ARIMA/EWMA-GARCH benchmarks scored by pinball loss.
5. **Pipeline** (`src/pipeline/`) — `run.py` (in-process flow) and `api.py` (FastAPI app; its `__main__` runs uvicorn on 0.0.0.0:8000, also the Docker `CMD`).

Shared helpers in `src/utils/` (`datetime_utils.py`, `safe.py`). `tests/` mirrors `src/`.

## Conventions

- All time series indexed with timezone-aware (UTC) `pd.DatetimeIndex`.
- Log-returns, not raw prices, everywhere downstream of ingestion.
- **Lazy heavy imports:** prediction/pipeline modules must not import `transformers`, `torch`, or `uvicorn` at module load — tests assert this via the `assert_no_heavy_imports` fixture (`tests/conftest.py`), so CI can run on `requirements-dev.txt` with those deps mocked.
- `data/raw/` is append-only; never overwrite existing files there. `data/cache/` and `models/` are gitignored.
- API keys via environment variables only (see `.env.example`); never hardcode.
- Branch names follow `<type>/<short-description>`, where `<type>` is one of `docs`, `feature`, `bug`, `fix`, etc. (e.g. `feature/new-sign-in`).

## CI & Deploy

- CI (`.github/workflows/ci.yml`) runs `black --check`, `isort --check-only`, `mypy src`, and `pytest` on the lightweight `requirements-dev.txt`; network clients and FinBERT are mocked in tests. Match this locally before pushing.
- `Dockerfile` builds the serving image from the **full** `requirements.txt` (FinBERT/torch needed for live sentiment); see `docs/deploy.md`.

## Docs References

- Architecture: `@docs/architecture.md`
- Graph model: `@docs/graph_model.md`
- Data sources + credibility weights: `@docs/data_sources.md`
- Deployment: `@docs/deploy.md`
