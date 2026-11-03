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


# ---- filtered historical simulation (fhs) --------------------------------------------------------


def _rel_gap(r: pd.Series, steps: int, window: int = 252) -> float:
    """(pinball of fhs - pinball of garch) / pinball of garch: same volatility, different quantile shape."""
    from src.eval.baselines import make_baselines
    from src.eval.metrics import pinball_per_row

    horizon = {1: "1d", 5: "5d"}[steps]
    target = forward_log_return(r, steps)
    base = make_baselines(r, target, horizon, steps, window)
    rows = [i for i in range(300, len(r) - steps) if not np.isnan(target.iloc[i])]
    y = target.iloc[rows].to_numpy()

    def score(name: str) -> float:
        p = np.array([base[name](i) for i in rows])
        return float(
            np.mean([pinball_per_row(y, p[:, k], q).mean() for k, q in enumerate((0.1, 0.5, 0.9))])
        )

    return (score("fhs") - score("garch")) / score("garch")


def _iid_returns(draw, seed: int, n: int = 1500) -> pd.Series:  # type: ignore[no-untyped-def]
    idx = pd.bdate_range("2018-01-01", periods=n, tz="UTC")
    r = pd.Series(draw(np.random.default_rng(seed), n) * 0.01, index=idx)
    r.iloc[0] = np.nan
    return r


def test_fhs_is_in_the_standard_baseline_set() -> None:
    assert {"zero", "empirical_quantile", "momentum", "garch", "fhs"} <= set(_make(_returns()))


def test_fhs_matches_a_hand_computation() -> None:
    from src.prediction.volatility import ewma_vol

    r = _returns(n=400)
    steps, window, pos = 5, 100, 250
    target = forward_log_return(r, steps)
    vol = ewma_vol(r)
    end = pos - steps + 1
    idx = np.arange(max(0, end - window), end)
    z = (target.iloc[idx] / (vol.iloc[idx] * np.sqrt(steps))).dropna().to_numpy()
    expected = np.quantile(z, [0.1, 0.5, 0.9]) * vol.iloc[pos] * np.sqrt(steps)
    assert _make(r, steps=steps)["fhs"](pos) == pytest.approx(tuple(expected))


@pytest.mark.parametrize("steps", [1, 5])
def test_fhs_uses_only_what_is_known_at_the_row(steps: int) -> None:
    r = _returns()
    pos = 220
    before = _make(r, steps)["fhs"](pos)
    altered = r.copy()
    altered.iloc[pos + 1 :] = 7.0  # a wild future must change nothing
    assert _make(altered, steps)["fhs"](pos) == before


def test_fhs_quantiles_are_ordered_and_scale_with_current_volatility() -> None:
    rng = np.random.default_rng(3)
    calm_then_wild = pd.Series(
        np.r_[rng.normal(0, 0.005, 300), rng.normal(0, 0.02, 100)],
        index=pd.bdate_range("2020-01-01", periods=400, tz="UTC"),
    )
    f = _make(calm_then_wild, steps=1)["fhs"]
    q_calm, q_wild = f(299), f(399)
    assert q_calm[0] <= q_calm[1] <= q_calm[2] and q_wild[0] <= q_wild[1] <= q_wild[2]
    assert (q_wild[2] - q_wild[0]) > 2.0 * (q_calm[2] - q_calm[0])


def test_fhs_needs_enough_realised_history() -> None:
    r = _returns()
    with pytest.raises(ValueError, match="realised targets"):
        _make(r, steps=5)["fhs"](2)
    with pytest.raises(ValueError, match="standardised targets"):
        _make(r, steps=1)["fhs"](25)


def test_fhs_beats_normal_quantiles_by_a_wide_margin_on_skewed_returns() -> None:
    """Same volatility forecast, only the quantile shape differs; skew is what a normal cannot fit.

    Measured over 16 seeds: FHS wins on every one, by about 5%. The margin asserted is far below that
    but far above the noise, so a regression that left FHS no better than normal quantiles fails.
    """
    gaps = [
        _rel_gap(_iid_returns(lambda g, n: g.exponential(1.0, n) - 1.0, 200 + k), steps=1)
        for k in range(8)
    ]
    assert all(g < -0.02 for g in gaps) and np.mean(gaps) < -0.035


def test_fhs_costs_only_a_little_estimation_noise_when_returns_are_normal() -> None:
    gaps = [
        _rel_gap(_iid_returns(lambda g, n: g.normal(0, 1, n), 300 + k), steps=1) for k in range(8)
    ]
    assert 0.0 < np.mean(gaps) < 0.02  # about +1%: the price of estimating the shape


def test_at_five_steps_summed_returns_are_near_normal_so_the_two_baselines_are_close() -> None:
    """The central limit theorem at work: skew largely washes out over five bars, so 'beat both
    baselines' mostly means 'beat the normal-quantile one'. Documented in docs/protocol_h2.md."""
    gaps = [
        _rel_gap(_iid_returns(lambda g, n: g.exponential(1.0, n) - 1.0, 400 + k), steps=5)
        for k in range(8)
    ]
    assert abs(np.mean(gaps)) < 0.02


def test_fhs_has_an_exact_closed_form_on_a_deterministic_series() -> None:
    """Returns alternate +0.01/-0.01, so the EWMA volatility is exactly 0.01 and every standardised
    1-step target is +1 or -1: the forecast must be (-0.01, 0.0, +0.01)."""
    idx = pd.bdate_range("2020-01-01", periods=400, tz="UTC")
    r = pd.Series(np.where(np.arange(400) % 2 == 0, 0.01, -0.01), index=idx)
    r.iloc[0] = np.nan
    got = _make(r, steps=1)["fhs"](300)
    assert got == pytest.approx((-0.01, 0.0, 0.01))


def test_fhs_refuses_a_row_whose_volatility_is_not_available() -> None:
    r = _returns().copy()
    r.iloc[220] = np.nan  # long realised history exists, but no volatility at this very bar
    with pytest.raises(ValueError, match="volatility is not available"):
        _make(r, steps=1)["fhs"](220)
