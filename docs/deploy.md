# Deployment

The prediction API (`src/pipeline/api.py`, FastAPI) can be served in a container.

## Build

```bash
docker build -t market-sentiment-predictor .
```

The image installs the full runtime from `requirements.txt`, including
`torch` / `transformers` for FinBERT, so it is **large** (multiple GB) and the
first build downloads sizeable wheels. CI/test runs use the much smaller
`requirements-dev.txt` instead.

## Run

```bash
docker run --rm -p 8000:8000 --env-file .env market-sentiment-predictor
```

Then:

```bash
curl localhost:8000/health
curl -X POST localhost:8000/predict \
  -H 'content-type: application/json' \
  -d '{"ticker": "AAPL", "horizons": ["1d", "5d"]}'
```

## Environment

Provide credentials via `--env-file .env` (see `.env.example`):

- `NEWS_API_KEY` — news ingestion
- `REDDIT_CLIENT_ID` / `REDDIT_CLIENT_SECRET` / `REDDIT_USER_AGENT` — social
- `SEC_EDGAR_EMAIL` — SEC EDGAR User-Agent
- `MODEL_PATH` — **required**: path to a trained model artifact for `/predict` (see below)
- `LOG_LEVEL` — serving log level (default `INFO`)

Missing optional source credentials degrade gracefully (that source is skipped);
price data is required.

### Serving a trained model

The serving API **requires** `MODEL_PATH`: it loads and validates the artifact
once at startup and **refuses to start** (fails fast) if `MODEL_PATH` is unset or
unloadable, instead of silently returning empty predictions on the first request.
Train an artifact and point the container at it:

```bash
python scripts/train_model.py --ticker AAPL --start 2022-01-01 --out models/aapl.joblib

docker run --rm -p 8000:8000 --env-file .env \
  -e MODEL_PATH=/models/aapl.joblib \
  -v "$PWD/models:/models:ro" \
  market-sentiment-predictor
```

## Notes

- **First call latency**: FinBERT weights are **pre-downloaded at build time**
  (baked into the image), so the first `/predict` no longer pays the ~30-60s
  Hugging Face download. (The `src.pipeline.run` CLI, run outside the image,
  still downloads on first use.)
- **Trained model**: the API requires `MODEL_PATH` and fails fast at startup if
  it is missing/unloadable (see *Serving a trained model* above).
- **Health**: `GET /health` returns `{"status": "ok", "model_loaded": <bool>}`;
  the container runs as a non-root user and ships a Docker healthcheck.
