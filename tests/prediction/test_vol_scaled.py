import numpy as np
import pandas as pd
import pytest

from src.eval.synthetic import make_synthetic
from src.prediction.vol_scaled import (
    CANDIDATES,
    COMPLEXITY_ORDER,
    CONTROL,
    FEATURE_SETS,
    PooledQuantileModel,
    normalised_features,
)
from src.prediction.volatility import ewma_vol


def _features_and_sigma() -> tuple[pd.DataFrame, pd.Series]:
    d = make_synthetic(400, signal=0.5, seed=1)
    return d.features, ewma_vol(d.log_returns).reindex(d.features.index)


def test_the_protocol_candidates_are_exactly_the_eight_plus_the_control() -> None:
    assert [c.id for c in COMPLEXITY_ORDER] == [
        "lq-a0.01-price",
        "lq-a0.01-news",
        "lq-a0.001-price",
        "lq-a0.001-news",
        "gb-50-price",
        "gb-50-news",
        "gb-100-price",
        "gb-100-news",
    ]
    assert set(CANDIDATES) == {c.id for c in COMPLEXITY_ORDER} | {CONTROL.id}
    assert CONTROL.feature_set == "none"


def test_normalised_features_match_the_protocol_definitions() -> None:
    f, sigma = _features_and_sigma()
    x = normalised_features(f, sigma, "news")
    assert list(x.columns) == list(FEATURE_SETS["news"])
    t = f.index[100]
    s = sigma[t]
    assert x.loc[t, "mom5"] == pytest.approx(f.loc[t, "momentum_5d"] / (s * np.sqrt(5)))
    assert x.loc[t, "mom20"] == pytest.approx(f.loc[t, "momentum_20d"] / (s * np.sqrt(20)))
    assert x.loc[t, "ret1"] == pytest.approx(f.loc[t, "return_lag_1"] / s)
    assert x.loc[t, "volratio"] == pytest.approx(f.loc[t, "realized_vol_20d"] / s)
    assert x.loc[t, "sent"] == f.loc[t, "sentiment_agg"]
    assert list(normalised_features(f, sigma, "price").columns) == list(FEATURE_SETS["price"])
    assert normalised_features(f, sigma, "none").shape == (len(f), 0)


def test_normalised_features_reject_bad_input() -> None:
    f, sigma = _features_and_sigma()
    with pytest.raises(ValueError, match="unknown feature set"):
        normalised_features(f, sigma, "everything")
    with pytest.raises(ValueError, match="missing columns"):
        normalised_features(f.drop(columns=["sentiment_agg"]), sigma, "news")
    for bad in (np.nan, 0.0, -1.0):
        s = sigma.copy()
        s.iloc[5] = bad
        with pytest.raises(ValueError, match="finite and positive"):
            normalised_features(f, s, "price")


def test_control_predicts_the_training_quantiles_for_every_row() -> None:
    rng = np.random.default_rng(0)
    y = rng.normal(size=500)
    x = pd.DataFrame(index=range(500))
    p = PooledQuantileModel(CONTROL).fit(x, y).predict(pd.DataFrame(index=range(7)))
    assert p.shape == (7, 3)
    assert (p == p[0]).all()
    assert p[0] == pytest.approx(np.quantile(y, [0.1, 0.5, 0.9]))


@pytest.mark.parametrize("cid", ["lq-a0.01-news", "gb-50-price"])
def test_model_is_deterministic_and_outputs_are_sorted(cid: str) -> None:
    f, sigma = _features_and_sigma()
    cand = CANDIDATES[cid]
    x = normalised_features(f, sigma, cand.feature_set)
    y = np.random.default_rng(2).normal(size=len(x))
    a = PooledQuantileModel(cand).fit(x, y).predict(x)
    b = PooledQuantileModel(cand).fit(x, y).predict(x)
    assert np.array_equal(a, b)
    assert (np.diff(a, axis=1) >= 0).all()


def test_the_scaler_is_fit_on_training_rows_only() -> None:
    f, sigma = _features_and_sigma()
    cand = CANDIDATES["lq-a0.01-price"]
    x = normalised_features(f, sigma, cand.feature_set)
    y = np.random.default_rng(3).normal(size=len(x))
    model = PooledQuantileModel(cand).fit(x.iloc[:200], y[:200])
    before = model.predict(x.iloc[200:210])
    model.predict(x.iloc[210:] * 100.0)  # predicting must not refit or alter the scaler
    assert np.array_equal(model.predict(x.iloc[200:210]), before)
    assert model._scaler.mean_ == pytest.approx(x.iloc[:200].mean().to_numpy())  # type: ignore[attr-defined]


def test_unfitted_and_malformed_inputs_raise() -> None:
    f, sigma = _features_and_sigma()
    cand = CANDIDATES["lq-a0.01-price"]
    x = normalised_features(f, sigma, cand.feature_set)
    with pytest.raises(RuntimeError, match="not fitted"):
        PooledQuantileModel(cand).predict(x)
    with pytest.raises(RuntimeError, match="not fitted"):
        PooledQuantileModel(CONTROL).predict(x)
    with pytest.raises(ValueError, match="matching"):
        PooledQuantileModel(cand).fit(x, np.zeros(3))
    with pytest.raises(ValueError, match="do not match"):
        PooledQuantileModel(cand).fit(x.iloc[:, :2], np.zeros(len(x)))
    bad = x.copy()
    bad.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        PooledQuantileModel(cand).fit(bad, np.zeros(len(x)))
    fitted = PooledQuantileModel(cand).fit(x, np.random.default_rng(0).normal(size=len(x)))
    with pytest.raises(ValueError, match="NaN"):
        fitted.predict(bad)
