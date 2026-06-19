"""Tests for pipeline.api using FastAPI's TestClient (no uvicorn)."""

from __future__ import annotations

import sys

from fastapi.testclient import TestClient

import src.pipeline.api as api_mod
from src.pipeline.api import app
from src.pipeline.run import PredictionResult

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_predict_schema(monkeypatch):
    def _fake_run(ticker, horizons=None, **kwargs):
        return PredictionResult(
            ticker=ticker,
            estimated_lag_hours=2.4,
            predictions={"1d": {"p10": -0.012, "p50": 0.004, "p90": 0.021}},
            top_articles=[{"source_category": "reuters", "score": 0.5}],
        )

    monkeypatch.setattr(api_mod, "run", _fake_run)
    resp = client.post("/predict", json={"ticker": "AAPL", "horizons": ["1d"]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["estimated_lag_hours"] == 2.4
    assert body["predictions"]["1d"] == {"p10": -0.012, "p50": 0.004, "p90": 0.021}


def test_predict_error_maps_to_502(monkeypatch):
    def _boom(ticker, horizons=None, **kwargs):
        raise RuntimeError("ingestion failed")

    monkeypatch.setattr(api_mod, "run", _boom)
    resp = client.post("/predict", json={"ticker": "AAPL"})
    assert resp.status_code == 502


def test_model_path_env_threaded_to_run(monkeypatch):
    captured = {}

    def _fake_run(ticker, horizons=None, **kwargs):
        captured["model_path"] = kwargs.get("model_path")
        return PredictionResult(
            ticker=ticker, estimated_lag_hours=1.0, predictions={}, top_articles=[]
        )

    monkeypatch.setenv("MODEL_PATH", "models/aapl.joblib")
    monkeypatch.setattr(api_mod, "run", _fake_run)
    resp = client.post("/predict", json={"ticker": "AAPL"})
    assert resp.status_code == 200
    assert captured["model_path"] == "models/aapl.joblib"


def test_no_model_path_passes_none(monkeypatch):
    captured = {}

    def _fake_run(ticker, horizons=None, **kwargs):
        captured["model_path"] = kwargs.get("model_path", "MISSING")
        return PredictionResult(
            ticker=ticker, estimated_lag_hours=1.0, predictions={}, top_articles=[]
        )

    monkeypatch.delenv("MODEL_PATH", raising=False)
    monkeypatch.setattr(api_mod, "run", _fake_run)
    resp = client.post("/predict", json={"ticker": "AAPL"})
    assert resp.status_code == 200
    assert captured["model_path"] is None


def test_no_uvicorn_imported():
    assert "uvicorn" not in sys.modules
