"""
FastAPI serving layer.

Exposes a single ``POST /predict`` endpoint matching the schema in
``docs/architecture.md``, plus a ``GET /health`` probe. ``uvicorn`` is imported
lazily in :func:`main` so it is only required when actually serving.

Set the ``MODEL_PATH`` environment variable to a trained
:class:`~src.prediction.model.QuantileReturnModel` artifact (see
``scripts/train_model.py``) so ``/predict`` serves real forecasts; without it
the pipeline falls back to the unfitted default and returns no predictions.
"""

from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.pipeline.run import run

app = FastAPI(title="market-sentiment-predictor")


class PredictRequest(BaseModel):
    ticker: str
    horizons: list[str] = ["1h", "1d", "5d"]


class QuantileOut(BaseModel):
    p10: float
    p50: float
    p90: float


class PredictResponse(BaseModel):
    estimated_lag_hours: float
    predictions: dict[str, QuantileOut]
    top_articles: list[dict]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    try:
        result = run(req.ticker, horizons=req.horizons, model_path=os.environ.get("MODEL_PATH"))
    except Exception as exc:  # ingestion / model failures → 502
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return PredictResponse(
        estimated_lag_hours=result.estimated_lag_hours,
        predictions={h: QuantileOut(**v) for h, v in result.predictions.items()},
        top_articles=result.top_articles,
    )


def main() -> None:
    import uvicorn  # lazy: only needed to serve

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
