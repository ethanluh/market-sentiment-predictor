import numpy as np
import pandas as pd
import pytest

from src.eval.baselines import MIN_HISTORY, make_baselines
from src.eval.targets import forward_log_return


def _returns(n: int = 300, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n, tz="UTC")
    return pd.Series(rng.normal(0, 0.01, n), index=idx)


def _make(r: pd.Series, steps: int = 1, **kw: bool) -> dict:
    horizon = {1: "1d", 5: "5d"}[steps]
    return make_baselines(r, forward_log_return(r, steps), horizon, steps, window=100, **kw)


def test_zero_baseline_forecasts_zero() -> None:
    assert _make(_returns())["zero"](150) == (0.0, 0.0, 0.0)


@pytest.mark.parametrize("steps", [1, 5])
def test_baselines_ignore_data_after_the_forecast_row(steps: int) -> None:
    r = _returns()
    pos = 200
    before = {name: f(pos) for name, f in _make(r, steps).items()}
    altered = r.copy()
    altered.iloc[pos + 1 :] = 5.0  # wildly different future
    after = {name: f(pos) for name, f in _make(altered, steps).items()}
    assert before == after


def test_empirical_quantiles_use_exactly_the_realised_targets() -> None:
    r = _returns()
    steps, window, pos = 5, 100, 200
    target = forward_log_return(r, steps)
    # At row 200 the 5-bar target of row i is known only if i + 5 <= 200.
    realised = target.iloc[pos - steps + 1 - window : pos - steps + 1].dropna()
    assert realised.index[-1] == r.index[pos - steps]
    expected = tuple(np.quantile(realised, [0.10, 0.50, 0.90]))
    assert _make(r, steps=steps)["empirical_quantile"](pos) == pytest.approx(expected)
    assert expected[0] <= expected[1] <= expected[2]


def test_short_history_raises() -> None:
    r = _returns()
    with pytest.raises(ValueError, match=str(MIN_HISTORY)):
        _make(r)["momentum"](MIN_HISTORY - 5)
    with pytest.raises(ValueError):
        _make(r)["empirical_quantile"](MIN_HISTORY - 5)


def test_arima_fallback_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    import statsmodels.tsa.arima.model as arima_model

    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("convergence failure")

    monkeypatch.setattr(arima_model, "ARIMA", boom)
    with pytest.raises(RuntimeError, match="fell back to momentum"):
        _make(_returns(), include_arima=True)["arima"](150)


def test_arima_is_opt_in() -> None:
    assert "arima" not in _make(_returns())
    assert "arima" in _make(_returns(), include_arima=True)


def test_empirical_baseline_never_wraps_to_the_end_of_the_array() -> None:
    r = _returns()
    with pytest.raises(ValueError, match="no realised targets"):
        _make(r, steps=5)["empirical_quantile"](2)
