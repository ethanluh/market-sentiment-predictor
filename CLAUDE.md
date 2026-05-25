# market-sentiment-predictor

News-driven stock price prediction pipeline: ingestion → sentiment (FinBERT) → graph diffusion → quantile return prediction.

## Stack

- Python 3.11+, C++ extensions where perf-critical
- Package manager: `pip` with `requirements.txt` (no conda)
- Testing: `pytest`
- Formatting: `black`, `isort`
- Type hints required on all public functions

## Project Layout

```
src/
  ingestion/    # data fetching: price, news, filings, social
  sentiment/    # FinBERT inference + source aggregation
  graph/        # diffusion model, sector correlation graph
  prediction/   # quantile regression, baselines, backtesting
  pipeline/     # orchestration (Prefect flows)
docs/           # architecture notes, implementation plans
tests/          # mirrors src/ structure
scripts/        # one-off utilities, data pulls
data/
  raw/          # never modified after write
  processed/    # derived features, embeddings
  cache/        # API response cache (gitignored)
```

## Commands

```bash
# Install
pip install -r requirements.txt --break-system-packages

# Run tests
pytest tests/ -v

# Lint + format
black src/ tests/ && isort src/ tests/

# Type check
mypy src/

# Run full pipeline (single ticker)
python -m src.pipeline.run --ticker AAPL --horizon 1d

# Backtest
python -m src.prediction.backtest --ticker AAPL --start 2023-01-01
```

## Key Conventions

- All time series indexed with `pd.DatetimeIndex`, timezone-aware (UTC)
- Log-returns, not raw prices, everywhere downstream of ingestion
- Graph node categories are defined in `src/graph/categories.py` — do not hardcode strings elsewhere
- Sentiment scores are continuous floats in [-1, 1]; never discretize before the final prediction step
- `data/raw/` is append-only; never overwrite existing files there
- API keys via environment variables only (see `.env.example`); never hardcode

## Docs References

- Architecture overview: `@docs/architecture.md`
- Graph model details: `@docs/graph_model.md`
- Data sources + credibility weights: `@docs/data_sources.md`
