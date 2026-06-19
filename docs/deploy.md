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
- `MODEL_PATH` — path to a trained model artifact for `/predict` (see below)

Missing optional source credentials degrade gracefully (that source is skipped);
price data is required.

### Serving a trained model

`/predict` reads `MODEL_PATH`; if unset it falls back to the unfitted default and
returns no predictions. Train an artifact and point the container at it:

```bash
python scripts/train_model.py --ticker AAPL --start 2022-01-01 --out models/aapl.joblib

docker run --rm -p 8000:8000 --env-file .env \
  -e MODEL_PATH=/models/aapl.joblib \
  -v "$PWD/models:/models:ro" \
  market-sentiment-predictor
```

## Notes

- **First call latency**: FinBERT weights download from Hugging Face on the
  first request. To bake them in, extend the Dockerfile to pre-download the
  model during build, or mount a Hugging Face cache volume.
- **Trained model**: without `MODEL_PATH` the API serves the unfitted default
  (empty predictions). Set `MODEL_PATH` to a trained artifact (see *Serving a
  trained model* above) to get real forecasts.
- The container runs as a non-root user and exposes a `/health` healthcheck.
