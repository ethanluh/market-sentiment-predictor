"""Tests for pipeline.api using FastAPI's TestClient (no uvicorn).

A bare ``TestClient(app)`` does not trigger lifespan startup; tests that need the
startup model-load (fail-fast / health) use ``with TestClient(app)`` so the
lifespan runs.
"""

from __future__ import annotations

import sys

import pytest
from fastapi.testclient import TestClient

import src.pipeline.api as api_mod
from src.pipeline.api import app
from src.pipeline.run import PredictionResult

client = TestClient(app)


@pytest.fixture(autouse=True)
def _reset_state():
    """Clear any model loaded into app.state by a previous lifespan run."""
    for attr in ("model", "model_path"):
        if hasattr(app.state, attr):
            delattr(app.state, attr)
    yield


def test_health_without_model():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "model_loaded": False}


def test_health_reports_model_loaded(monkeypatch):
    sentinel = object()
    monkeypatch.setenv("MODEL_PATH", "models/fake.joblib")
    monkeypatch.setattr(api_mod, "_load_serving_model", lambda p: sentinel)
    with TestClient(app) as c:
        resp = c.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "model_loaded": True}


def test_missing_model_path_fails_startup(monkeypatch):
    monkeypatch.delenv("MODEL_PATH", raising=False)
    with pytest.raises(RuntimeError):
        with TestClient(app):
            pass


def test_unloadable_model_fails_startup(monkeypatch):
    monkeypatch.setenv("MODEL_PATH", "models/missing.joblib")

    def _boom(path):
        raise FileNotFoundError(path)

    monkeypatch.setattr(api_mod, "_load_serving_model", _boom)
    with pytest.raises(RuntimeError):
        with TestClient(app):
            pass


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


def test_predict_threads_loaded_model(monkeypatch):
    sentinel = object()
    captured = {}

    def _fake_run(ticker, horizons=None, **kwargs):
        captured["model"] = kwargs.get("model")
        return PredictionResult(
            ticker=ticker, estimated_lag_hours=1.0, predictions={}, top_articles=[]
        )

    monkeypatch.setenv("MODEL_PATH", "models/fake.joblib")
    monkeypatch.setattr(api_mod, "_load_serving_model", lambda p: sentinel)
    monkeypatch.setattr(api_mod, "run", _fake_run)
    with TestClient(app) as c:
        resp = c.post("/predict", json={"ticker": "AAPL", "horizons": ["1d"]})
    assert resp.status_code == 200
    assert captured["model"] is sentinel


def test_invalid_ticker_returns_422():
    resp = client.post("/predict", json={"ticker": "not a ticker!"})
    assert resp.status_code == 422


def test_unknown_horizon_returns_422():
    resp = client.post("/predict", json={"ticker": "AAPL", "horizons": ["99y"]})
    assert resp.status_code == 422


def test_empty_horizons_returns_422():
    resp = client.post("/predict", json={"ticker": "AAPL", "horizons": []})
    assert resp.status_code == 422


def test_ticker_is_normalized(monkeypatch):
    captured = {}

    def _fake_run(ticker, horizons=None, **kwargs):
        captured["ticker"] = ticker
        return PredictionResult(
            ticker=ticker, estimated_lag_hours=1.0, predictions={}, top_articles=[]
        )

    monkeypatch.setattr(api_mod, "run", _fake_run)
    resp = client.post("/predict", json={"ticker": "  aapl ", "horizons": ["1d"]})
    assert resp.status_code == 200
    assert captured["ticker"] == "AAPL"


def test_predict_error_maps_to_502_without_leaking_internals(monkeypatch):
    def _boom(ticker, horizons=None, **kwargs):
        raise RuntimeError("ingestion failed: SECRET_KEY=topsecret")

    monkeypatch.setattr(api_mod, "run", _boom)
    resp = client.post("/predict", json={"ticker": "AAPL"})
    assert resp.status_code == 502
    assert "SECRET" not in resp.text
    assert "topsecret" not in resp.text


def test_no_uvicorn_imported():
    assert "uvicorn" not in sys.modules
