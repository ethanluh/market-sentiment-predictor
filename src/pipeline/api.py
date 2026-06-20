"""
FastAPI serving layer.

Exposes a single ``POST /predict`` endpoint matching the schema in
``docs/architecture.md``, plus a ``GET /health`` probe. ``uvicorn`` is imported
lazily in :func:`main` so it is only required when actually serving.

Serving **requires** a trained :class:`~src.prediction.model.QuantileReturnModel`
artifact: set the ``MODEL_PATH`` environment variable (see
``scripts/train_model.py``). The app validates and loads it once at startup and
refuses to start if it is missing or unloadable — a misconfigured deploy fails
fast instead of silently returning empty predictions on the first request. (The
``src.pipeline.run`` CLI and the backtest still run without a model.)
"""

from __future__ import annotations

import logging
import os
import re
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, field_validator

from src.pipeline.run import run

logger = logging.getLogger("pipeline.api")

# Tickers: 1-6 chars, uppercase letters plus '.'/'-' (e.g. BRK.B, RDS-A).
_TICKER_RE = re.compile(r"^[A-Z][A-Z.\-]{0,5}$")


def _configure_logging() -> None:
    """Honour the ``LOG_LEVEL`` env var for the serving process (default INFO)."""
    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=getattr(logging, level, logging.INFO))


def _load_serving_model(model_path: str | None):  # type: ignore[no-untyped-def]
    """Load the model artifact used to serve predictions (reuses run's loader)."""
    from src.pipeline.run import _load_model

    return _load_model(model_path)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Validate + load the model artifact once, failing fast on misconfiguration."""
    _configure_logging()
    model_path = os.environ.get("MODEL_PATH")
    if not model_path:
        raise RuntimeError(
            "MODEL_PATH is not set; the serving API requires a trained model "
            "artifact (see scripts/train_model.py and docs/deploy.md)."
        )
    try:
        model = _load_serving_model(model_path)
    except Exception as exc:  # surface a clear, actionable boot failure
        raise RuntimeError(f"failed to load MODEL_PATH={model_path!r}: {exc}") from exc

    app.state.model = model
    app.state.model_path = model_path
    logger.info("loaded serving model from %s", model_path)
    yield


app = FastAPI(title="market-sentiment-predictor", lifespan=lifespan)


@app.middleware("http")
async def _log_requests(request: Request, call_next):  # type: ignore[no-untyped-def]
    t0 = time.perf_counter()
    response = await call_next(request)
    logger.info(
        "%s %s -> %d (%.3fs)",
        request.method,
        request.url.path,
        response.status_code,
        time.perf_counter() - t0,
    )
    return response


class PredictRequest(BaseModel):
    ticker: str
    horizons: list[str] = ["1h", "1d", "5d"]

    @field_validator("ticker")
    @classmethod
    def _valid_ticker(cls, v: str) -> str:
        v = v.strip().upper()
        if not _TICKER_RE.match(v):
            raise ValueError("ticker must be 1-6 chars: letters, optionally '.' or '-'")
        return v

    @field_validator("horizons")
    @classmethod
    def _valid_horizons(cls, v: list[str]) -> list[str]:
        from src.prediction.baselines import HORIZON_SPECS

        if not v:
            raise ValueError("horizons must be non-empty")
        unknown = [h for h in v if h not in HORIZON_SPECS]
        if unknown:
            raise ValueError(f"unknown horizons {unknown}; valid: {sorted(HORIZON_SPECS)}")
        return v


class QuantileOut(BaseModel):
    p10: float
    p50: float
    p90: float


class PredictResponse(BaseModel):
    estimated_lag_hours: float
    predictions: dict[str, QuantileOut]
    top_articles: list[dict]


@app.get("/health")
def health(request: Request) -> dict[str, object]:
    loaded = getattr(request.app.state, "model", None) is not None
    return {"status": "ok", "model_loaded": loaded}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, request: Request) -> PredictResponse:
    try:
        result = run(
            req.ticker, horizons=req.horizons, model=getattr(request.app.state, "model", None)
        )
    except Exception as exc:  # ingestion / model failures → 502 (no internals leaked)
        logger.exception("prediction failed for ticker %s", req.ticker)
        raise HTTPException(status_code=502, detail="prediction failed; see server logs") from exc

    return PredictResponse(
        estimated_lag_hours=result.estimated_lag_hours,
        predictions={h: QuantileOut(**v) for h, v in result.predictions.items()},
        top_articles=result.top_articles,
    )


def main() -> None:
    import uvicorn  # lazy: only needed to serve

    _configure_logging()
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
