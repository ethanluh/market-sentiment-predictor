"""Tests for prediction.model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.prediction.features import FEATURE_NAMES
from src.prediction.model import QuantilePrediction, QuantileReturnModel


def _make_xy(n: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(0, 1, (n, len(FEATURE_NAMES))), columns=FEATURE_NAMES)
    # Target loosely depends on sentiment + momentum + noise.
    y = 0.01 * X["sentiment_agg"] + 0.005 * X["momentum_5d"] + rng.normal(0, 0.01, n)
    return X, pd.Series(y)


class TestQuantilePrediction:
    def test_from_sorted_enforces_order(self):
        qp = QuantilePrediction.from_sorted([0.9, -0.1, 0.3])
        assert qp.p10 <= qp.p50 <= qp.p90


class TestQuantileReturnModel:
    def test_fit_predict_shapes(self):
        X, y = _make_xy()
        model = QuantileReturnModel(horizons=("1d",), backend="linear")
        model.fit(X, {"1d": y})
        preds = model.predict(X)
        assert "1d" in preds
        assert len(preds["1d"]) == len(X)

    def test_no_quantile_crossing(self):
        X, y = _make_xy()
        model = QuantileReturnModel(horizons=("1d",), backend="linear").fit(X, {"1d": y})
        for qp in model.predict(X)["1d"]:
            assert qp.p10 <= qp.p50 <= qp.p90

    def test_missing_feature_raises(self):
        X, y = _make_xy()
        model = QuantileReturnModel(horizons=("1d",), backend="linear").fit(X, {"1d": y})
        with pytest.raises(ValueError):
            model.predict(X.drop(columns=["sentiment_agg"]))

    def test_all_nan_horizon_skipped(self):
        X, y = _make_xy()
        nan_y = pd.Series(np.nan, index=X.index)
        with pytest.warns(UserWarning):
            model = QuantileReturnModel(horizons=("1d",), backend="linear").fit(X, {"1d": nan_y})
        assert "1d" not in model._available
        assert model.predict(X) == {}

    def test_save_load_roundtrip(self, tmp_path):
        X, y = _make_xy()
        model = QuantileReturnModel(horizons=("1d",), backend="linear").fit(X, {"1d": y})
        path = str(tmp_path / "model.joblib")
        model.save(path)
        loaded = QuantileReturnModel.load(path)
        a = model.predict(X.head(3))["1d"]
        b = loaded.predict(X.head(3))["1d"]
        assert [p.p50 for p in a] == [p.p50 for p in b]
