"""Tests for scripts/train_model.py (ingestion mocked, no network)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO = Path(__file__).resolve().parents[2]


def _load_train_model():
    spec = importlib.util.spec_from_file_location(
        "train_model", _REPO / "scripts" / "train_model.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _synthetic_prices(n: int = 320, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="D", tz="UTC")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    df = pd.DataFrame({"close": close, "volume": rng.integers(1e6, 5e6, n)}, index=idx)
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    return df


def test_train_saves_loadable_model(tmp_path, monkeypatch):
    import src.ingestion.price as price_mod

    monkeypatch.setattr(price_mod, "fetch_prices", lambda *a, **k: _synthetic_prices())

    tm = _load_train_model()
    out = tmp_path / "model.joblib"
    tm.train("AAPL", start="2022-01-01", horizons=("1d", "5d"), out=str(out), backend="linear")

    assert out.exists()
    from src.prediction.model import QuantileReturnModel

    loaded = QuantileReturnModel.load(str(out))
    preds = loaded.predict(_features_frame())
    assert set(preds) <= {"1d", "5d"}
    for horizon_preds in preds.values():
        for qp in horizon_preds:
            assert qp.p10 <= qp.p50 <= qp.p90


def _features_frame() -> pd.DataFrame:
    from src.prediction.features import FEATURE_NAMES

    rng = np.random.default_rng(1)
    return pd.DataFrame(rng.normal(0, 1, (3, len(FEATURE_NAMES))), columns=FEATURE_NAMES)


def test_train_with_enrichment_sources(tmp_path, monkeypatch):
    """--use-trends / --use-insider-flow / --use-vix fetch and feed the model."""
    import src.ingestion.form4 as form4_mod
    import src.ingestion.price as price_mod
    import src.ingestion.trends as trends_mod

    prices = _synthetic_prices()
    monkeypatch.setattr(price_mod, "fetch_prices", lambda *a, **k: prices)  # also serves ^VIX

    calls = {"trends": 0, "form4": 0}

    def _fake_trends(ticker, start=None, end=None, **k):
        calls["trends"] += 1
        return pd.Series(
            np.linspace(10, 90, len(prices)), index=prices.index, name="search_interest"
        )

    def _fake_form4(ticker, after=None, limit=50):
        calls["form4"] += 1
        rng = np.random.default_rng(5)
        return pd.Series(
            rng.integers(-5000, 5000, len(prices)).astype(float),
            index=prices.index,
            name="insider_net_shares",
        )

    monkeypatch.setattr(trends_mod, "fetch_trends", _fake_trends)
    monkeypatch.setattr(form4_mod, "fetch_form4", _fake_form4)

    tm = _load_train_model()
    out = tmp_path / "model.joblib"
    tm.train(
        "AAPL",
        start="2022-01-01",
        horizons=("1d",),
        out=str(out),
        backend="linear",
        use_trends=True,
        use_insider_flow=True,
        use_vix=True,
    )
    assert calls == {"trends": 1, "form4": 1}
    assert out.exists()

    from src.prediction.model import QuantileReturnModel

    loaded = QuantileReturnModel.load(str(out))
    preds = loaded.predict(_features_frame())
    assert "1d" in preds


def test_unknown_horizon_rejected(monkeypatch):
    import src.ingestion.price as price_mod

    # Should fail fast on horizon validation, before any fetch.
    monkeypatch.setattr(price_mod, "fetch_prices", lambda *a, **k: _synthetic_prices())
    tm = _load_train_model()
    with pytest.raises(ValueError):
        tm.train("AAPL", start="2022-01-01", horizons=("3d",), out="unused.joblib")
