import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from src.prediction.backtest import interval_coverage, pinball_loss
from src.prediction.features import FEATURE_NAMES
from src.prediction.model import QuantileReturnModel

seeds = st.integers(0, 2**32 - 1)
finite = st.floats(-1.0, 1.0)
arrays = st.lists(finite, min_size=1, max_size=50).map(np.array)


@settings(max_examples=15)
@given(seeds)
def test_predict_returns_the_sorted_raw_estimator_outputs(seed: int) -> None:
    """``predict`` must sort each row's three quantiles, whatever the raw estimators say."""
    rng = np.random.default_rng(seed)
    x = pd.DataFrame(rng.normal(size=(80, len(FEATURE_NAMES))), columns=FEATURE_NAMES)
    y = pd.Series(rng.normal(0, 0.01, 80) + 0.005 * x.iloc[:, 0])
    model = QuantileReturnModel(horizons=("1d",), backend="linear").fit(
        x.iloc[:60], {"1d": y.iloc[:60]}
    )
    test_x = x.iloc[60:]
    raw = np.column_stack(
        [model._models[("1d", q)].predict(test_x.to_numpy()) for q in model.quantiles]
    )
    got = np.array([[p.p10, p.p50, p.p90] for p in model.predict(test_x)["1d"]])
    assert np.array_equal(got, np.sort(raw, axis=1))
    assert (np.diff(got, axis=1) >= 0).all()


def test_predict_repairs_crossed_raw_estimators() -> None:
    """Force crossing by training the P10 estimator on higher targets than the P90 one."""
    rng = np.random.default_rng(0)
    x = pd.DataFrame(rng.normal(size=(60, len(FEATURE_NAMES))), columns=FEATURE_NAMES)
    model = QuantileReturnModel(horizons=("1d",), backend="linear").fit(
        x, {"1d": pd.Series(rng.normal(size=60))}
    )
    model._models[("1d", 0.10)], model._models[("1d", 0.90)] = (
        model._models[("1d", 0.90)],
        model._models[("1d", 0.10)],
    )
    raw_p10 = model._models[("1d", 0.10)].predict(x.to_numpy())
    raw_p90 = model._models[("1d", 0.90)].predict(x.to_numpy())
    assert (raw_p10 > raw_p90).all()  # the raw estimators really are crossed
    for p in model.predict(x)["1d"]:
        assert p.p10 <= p.p50 <= p.p90


@given(arrays, st.sampled_from([0.1, 0.5, 0.9]))
def test_pinball_is_non_negative_and_zero_for_perfect_forecasts(y: np.ndarray, q: float) -> None:
    assert pinball_loss(y, y, q) == 0.0
    assert pinball_loss(y, y + 0.3, q) >= 0.0


@given(arrays, st.floats(0.0, 1.0), st.floats(0.0, 1.0))
def test_wider_intervals_never_cover_less(y: np.ndarray, narrow: float, extra: float) -> None:
    centre = np.zeros_like(y)
    assert interval_coverage(
        y, centre - narrow - extra, centre + narrow + extra
    ) >= interval_coverage(y, centre - narrow, centre + narrow)
    assert 0.0 <= interval_coverage(y, centre - narrow, centre + narrow) <= 1.0
