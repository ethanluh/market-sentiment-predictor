import numpy as np
import pandas as pd
import pytest

from src.eval import voltarget as vt
from src.prediction.volatility import ewma_vol

START = "2020-01-02"


def panel(
    persist: float, n: int = 1300, k: int = 25, seed: int = 0, drift: float = 0.0
) -> pd.DataFrame:
    """Daily log returns of k tickers sharing one volatility factor (AR(1) log-vol, ``persist``)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-09-02", periods=n, tz="UTC")
    log_vol = np.zeros(n)
    shocks = rng.normal(size=n)
    for t in range(1, n):
        log_vol[t] = persist * log_vol[t - 1] + 0.15 * shocks[t]
    z = rng.standard_t(5, size=(n, k)) / np.sqrt(5 / 3)
    beta = rng.uniform(0.8, 1.2, k)
    data = 0.01 * np.exp(log_vol)[:, None] * beta[None, :] * z + drift
    return pd.DataFrame(data, index=idx, columns=[f"T{i}" for i in range(k)])


def run(lr: pd.DataFrame, name: str = "ewma94", cost_bps: float = vt.COST_BPS) -> vt.Backtest:
    sigma = vt.annualised_forecasts(lr, name)
    target = vt.calibrate_target(sigma, START, "2022-12-31")
    return vt.backtest(lr, vt.target_weights(sigma, target), cost_bps).period(START)


def test_forecasters_match_their_definitions() -> None:
    lr = panel(0.9, n=200, k=1)["T0"]
    pd.testing.assert_series_equal(vt.forecast_vol(lr, "ewma94"), ewma_vol(lr, lam=0.94))
    pd.testing.assert_series_equal(vt.forecast_vol(lr, "ewma97"), ewma_vol(lr, lam=0.97))
    manual = float(np.sqrt((lr.iloc[100:163] ** 2).mean()))  # uncentred, last 63 bars to t=162
    assert vt.forecast_vol(lr, "rolling63").iloc[162] == pytest.approx(manual)
    assert vt.forecast_vol(lr, "rolling63").iloc[:62].isna().all()
    with pytest.raises(ValueError, match="unknown forecaster"):
        vt.forecast_vol(lr, "garch")


def test_weights_are_capped_at_one_and_scale_inversely() -> None:
    sigma = pd.DataFrame({"a": [0.1, 0.2, 0.4, np.nan]})
    w = vt.target_weights(sigma, 0.2)
    assert w["a"].iloc[:3].tolist() == [1.0, 1.0, 0.5]
    assert np.isnan(w["a"].iloc[3])
    with pytest.raises(ValueError, match="positive"):
        vt.target_weights(sigma, 0.0)


def test_the_weight_set_at_t_earns_the_return_of_t_plus_1_and_entry_is_charged() -> None:
    idx = pd.bdate_range("2020-01-01", periods=4, tz="UTC")
    lr = pd.DataFrame(
        {"a": np.log1p([0.01, 0.02, -0.01, 0.03]), "b": np.log1p([0.0, 0.01, 0.02, 0.0])}, index=idx
    )
    w = pd.DataFrame({"a": [1.0, 0.5, 0.5, 1.0], "b": [1.0, 1.0, 0.0, 1.0]}, index=idx)
    bt = vt.backtest(lr, w, cost_bps=100.0, min_tickers=2)
    # bar 1 earns weights of bar 0 (entry from 0 -> 1 charged), bar 2 earns bar 1's, and so on
    assert bt.net.index[0] == idx[1]
    e1 = (1.0 * 0.02 - 0.01 + 1.0 * 0.01 - 0.01) / 2
    assert bt.net.iloc[0] == pytest.approx(e1)
    assert bt.net.iloc[1] == pytest.approx((0.5 * -0.01 - 0.005 + 1.0 * 0.02) / 2)
    assert bt.buyhold.iloc[0] == pytest.approx((0.02 + 0.01) / 2)
    assert bt.turnover.iloc[0] == pytest.approx(1.0)  # both entered from zero


def test_zero_cost_net_equals_the_weighted_gross_return() -> None:
    lr = panel(0.97, n=400)
    sigma = vt.annualised_forecasts(lr, "ewma94")
    w = vt.target_weights(sigma, vt.calibrate_target(sigma, START, "2020-12-31"))
    bt = vt.backtest(lr, w, cost_bps=0.0)
    gross = (w.shift(1) * lr.transform(np.expm1)).where(w.shift(1).notna()).mean(axis=1).dropna()
    pd.testing.assert_series_equal(bt.net, gross.loc[bt.net.index], check_names=False)
    costly = vt.backtest(lr, w, cost_bps=10.0)
    assert (costly.net <= bt.net + 1e-15).all() and costly.net.sum() < bt.net.sum()


def test_too_few_available_tickers_raises_instead_of_filling() -> None:
    lr = panel(0.97, n=300, k=25)
    lr.iloc[150:, :10] = np.nan  # ten tickers vanish: 15 < 20
    sigma = vt.annualised_forecasts(lr, "ewma94")
    with pytest.raises(ValueError, match="fewer than 20 available"):
        vt.backtest(lr, vt.target_weights(sigma, 0.2))


def test_a_ticker_with_a_short_history_is_excluded_until_it_has_a_forecast() -> None:
    lr = panel(0.97, n=300, k=25)
    lr.iloc[:100, 0] = np.nan  # ticker 0 lists late
    sigma = vt.annualised_forecasts(lr, "ewma94")
    bt = vt.backtest(lr, vt.target_weights(sigma, 0.2))
    assert bt.net.index[0] == lr.index[20]  # the others' forecasts start first; 24 >= 20 tickers


def test_there_is_no_look_ahead_and_a_leaking_rule_is_caught() -> None:
    lr = panel(0.97, n=700)
    cut = 500
    sigma = vt.annualised_forecasts(lr, "ewma94")
    target = vt.calibrate_target(sigma, START, "2021-06-30")
    full = vt.backtest(lr, vt.target_weights(sigma, target))
    trunc_lr = lr.iloc[: cut + 1]
    trunc = vt.backtest(
        trunc_lr, vt.target_weights(vt.annualised_forecasts(trunc_lr, "ewma94"), target)
    )
    last = lr.index[cut]
    pd.testing.assert_series_equal(full.net.loc[:last], trunc.net.loc[:last])

    # negative control: sizing on the volatility of the NEXT 8 bars must change the prefix when the
    # data end at the cut (a forward window truncated by the cut gives different values)
    def leaky(frame: pd.DataFrame) -> pd.Series:
        fwd = frame.pow(2)[::-1].rolling(8, min_periods=1).mean()[::-1].pow(0.5) * np.sqrt(252.0)
        return vt.backtest(frame, vt.target_weights(fwd, target)).net

    assert not np.allclose(leaky(lr).loc[:last], leaky(trunc_lr).loc[:last])


def test_clustered_volatility_lowers_relative_dispersion_and_iid_does_not() -> None:
    clustered = vt.evaluate(run(panel(0.98)), n_boot=200)
    assert clustered["vv_ratio"] < 0.80 and clustered["h3a"]
    assert (
        clustered["max_drawdown"] > clustered["max_drawdown_matched"] - 0.05
    )  # reported, not gamed
    null = vt.evaluate(run(panel(0.0)), n_boot=200)
    assert 0.95 < null["vv_ratio"] < 1.08 and not null["h3a"]


def test_the_exposure_matched_control_is_scaled_buy_and_hold() -> None:
    ev = vt.evaluate(run(panel(0.98)), n_boot=100)
    assert 0.0 < ev["mean_exposure"] < 1.0
    assert ev["max_drawdown_matched"] > ev["max_drawdown_buyhold"]  # less exposure, shallower


def test_evaluate_is_deterministic_for_a_seed_and_needs_enough_bars() -> None:
    bt = run(panel(0.98, n=700))
    a, b = vt.evaluate(bt, n_boot=100, seed=3), vt.evaluate(bt, n_boot=100, seed=3)
    assert a == b
    with pytest.raises(ValueError, match="usable RV windows"):
        vt.evaluate(bt.period("2021-01-01", "2021-01-29"), n_boot=10)


def test_dispersion_and_drawdown_helpers() -> None:
    assert vt.dispersion(np.array([1.0, 1.0, 1.0])) == 0.0
    with pytest.raises(ValueError, match="at least two"):
        vt.dispersion(np.array([1.0]))
    r = pd.Series([0.1, -0.5, 0.1, 0.1])
    assert vt.max_drawdown(r) == pytest.approx(-0.5)
    assert vt.worst_window(pd.Series([0.1] * 5 + [-0.5] + [0.1] * 5), window=2) == pytest.approx(
        1.1 * 0.5 - 1.0
    )


def test_a_zero_variance_series_makes_the_sharpe_test_fail_loudly() -> None:
    flat = np.zeros(50)
    noisy = np.random.default_rng(0).normal(size=50)
    with pytest.raises(ValueError, match="zero-variance"):
        vt._sharpe_diff(flat, noisy)


def _ev(ratio: float, h3a: bool = True, h3b: bool = True) -> dict[str, object]:
    return {"vv_ratio": ratio, "h3a": h3a, "h3b": h3b}


def test_selection_takes_the_lowest_ratio_then_the_first_registered_within_tolerance() -> None:
    dev = {"ewma94": _ev(0.70), "ewma97": _ev(0.69), "rolling63": _ev(0.60)}
    assert vt.select(dev).winner == "rolling63"  # 0.60 is more than 0.02 below the others
    dev = {"ewma94": _ev(0.70), "ewma97": _ev(0.69), "rolling63": _ev(0.685)}
    assert vt.select(dev).winner == "ewma94"  # all within 0.02 of 0.685; registered order breaks it


def test_selection_closes_the_holdout_when_the_winner_fails_a_rule() -> None:
    dev = {"ewma94": _ev(0.9, h3a=False), "ewma97": _ev(0.95, h3a=False), "rolling63": _ev(0.99)}
    sel = vt.select(dev)
    assert not sel.proceed and sel.reasons == ["ewma94 fails h3a"]
    with pytest.raises(ValueError, match="exactly"):
        vt.select({"ewma94": _ev(0.7)})


def test_the_target_is_the_median_forecast_over_the_window_only() -> None:
    idx = pd.bdate_range("2020-01-01", periods=6, tz="UTC")
    sigma = pd.DataFrame({"a": [1.0, 2.0, 3.0, 100.0, 100.0, 100.0]}, index=idx)
    assert vt.calibrate_target(sigma, "2020-01-01", str(idx[2].date())) == 2.0
    with pytest.raises(ValueError, match="no finite forecasts"):
        vt.calibrate_target(sigma, "2030-01-01", "2030-02-01")


def test_a_scored_bar_with_no_available_ticker_raises_but_warm_up_may_be_empty() -> None:
    lr = panel(0.97, n=300, k=25)
    sigma = vt.annualised_forecasts(lr, "ewma94")
    w = vt.target_weights(sigma, 0.2)
    assert (
        vt.backtest(lr, w).net.index[0] == lr.index[20]
    )  # warm-up bars (< 2020-01-02) not checked
    lr2 = lr.copy()
    lr2.iloc[200:203] = np.nan  # a whole-market outage after the scored start
    with pytest.raises(ValueError, match="scored bars have fewer"):
        vt.backtest(lr2, vt.target_weights(vt.annualised_forecasts(lr2, "ewma94"), 0.2))


def test_the_availability_rule_ignores_a_thin_warm_up() -> None:
    lr = panel(0.97, n=300, k=25)
    lr.iloc[:60, :10] = np.nan  # 15 tickers before bar 60: below 20, but before the scored start
    lr.index = pd.bdate_range("2019-09-02", periods=300, tz="UTC")
    sigma = vt.annualised_forecasts(lr, "ewma94")
    vt.backtest(lr, vt.target_weights(sigma, 0.2), scored_from=str(lr.index[80].date()))
    with pytest.raises(ValueError, match="scored bars have fewer"):
        vt.backtest(lr, vt.target_weights(sigma, 0.2), scored_from=str(lr.index[30].date()))


def test_a_ticker_that_leaves_and_returns_is_flat_in_between_and_recharged_on_re_entry() -> None:
    idx = pd.bdate_range("2020-01-01", periods=6, tz="UTC")
    lr = pd.DataFrame({"a": np.log1p([0.01] * 6), "b": np.log1p([0.01] * 6)}, index=idx)
    lr.loc[idx[3], "a"] = np.nan  # a has no return on bar 3
    w = pd.DataFrame(1.0, index=idx, columns=["a", "b"])
    bt = vt.backtest(lr, w, cost_bps=100.0, min_tickers=1, scored_from=str(idx[1].date()))
    # bar 1: both enter (turnover 1); bar 2: nothing; bar 3: only b; bar 4: a re-enters (turnover 0.5)
    assert bt.turnover.loc[idx[1]] == pytest.approx(1.0)
    assert bt.turnover.loc[idx[2]] == 0.0
    assert bt.turnover.loc[idx[3]] == 0.0
    assert bt.turnover.loc[idx[4]] == pytest.approx(0.5)


def test_period_bounds_are_inclusive_utc_dates_and_the_split_weight_is_carried() -> None:
    lr = panel(0.97, n=1300)
    full = run(lr)
    dev, hold = full.period(START, vt.DEV_END), full.period(vt.HOLDOUT_START)
    assert dev.net.index[-1] <= pd.Timestamp(vt.DEV_END, tz="UTC") < hold.net.index[0]
    assert len(dev.net) + len(hold.net) == len(full.net)
    # RV windows come from the slice: the first holdout RV21 needs 21 holdout bars
    first = vt.rv21(hold.net).dropna().index[0]
    assert first == hold.net.index[vt.RV_WINDOW - 1]
    # the weight carried across the split: bar 1 of the holdout has no entry charge
    assert hold.turnover.iloc[0] == full.turnover.loc[hold.net.index[0]] < 0.5  # not a fresh entry
    late = pd.Timestamp("2023-12-31 20:00", tz="UTC")
    odd = vt.Backtest(
        pd.Series([1.0, 2.0], index=[late - pd.Timedelta(days=1), late]),
        pd.Series([1.0, 2.0], index=[late - pd.Timedelta(days=1), late]),
        pd.Series([1.0, 2.0], index=[late - pd.Timedelta(days=1), late]),
        pd.Series([1.0, 2.0], index=[late - pd.Timedelta(days=1), late]),
    )
    assert (
        len(odd.period("2020-01-01", "2023-12-31").net) == 2
    )  # a late-evening bar is still that date


def test_the_bootstrap_uses_a_90_percent_interval_and_the_thresholds_bind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[float] = []
    lo_hi = [(-0.05, 0.9)]

    def fake(stat_fn, *arrays, block, n_boot, seed, level):  # type: ignore[no-untyped-def]
        seen.append(level)
        assert (block, seed) == (vt.BLOCK, vt.SEED)
        return lo_hi[0]

    monkeypatch.setattr(vt, "block_bootstrap_ci", fake)
    bt = run(panel(0.98))
    ev = vt.evaluate(bt, n_boot=10)
    assert seen == [0.90, 0.90] and ev["h3a"] and ev["h3b"]
    lo_hi[0] = (-0.1, 1.0)  # 95th percentile of 1.0 is not below 1.0; 5th pct -0.10 is allowed
    ev = vt.evaluate(bt, n_boot=10)
    assert not ev["h3a"] and ev["h3b"]
    lo_hi[0] = (-0.1001, 0.99)
    ev = vt.evaluate(bt, n_boot=10)
    assert ev["h3a"] and not ev["h3b"]


def test_dispersion_uses_ddof_one_and_the_tie_boundary_is_inclusive() -> None:
    assert vt.dispersion(np.array([1.0, 2.0, 3.0])) == pytest.approx(
        0.5
    )  # std 1.0 (ddof 1) / mean 2
    dev = {"ewma94": _ev(0.72), "ewma97": _ev(0.70), "rolling63": _ev(0.80)}
    assert vt.select(dev).winner == "ewma94"  # exactly 0.02 above the best still ties
