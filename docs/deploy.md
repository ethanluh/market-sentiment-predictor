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

Missing optional source credentials degrade gracefully (that source is skipped);
price data is required.

## Notes

- **First call latency**: FinBERT weights download from Hugging Face on the
  first request. To bake them in, extend the Dockerfile to pre-download the
  model during build, or mount a Hugging Face cache volume.
- **Trained model**: the API currently serves the pipeline's default
  (unfitted) model, which yields empty predictions. To serve real forecasts,
  train an artifact (`scripts/train_model.py`), make it available in the
  container (COPY or a mounted volume under `models/`), and wire its path
  through `pipeline.run.run(model_path=...)` / the API handler.
- The container runs as a non-root user and exposes a `/health` healthcheck.
