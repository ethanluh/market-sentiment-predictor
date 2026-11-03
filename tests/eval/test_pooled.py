import numpy as np
import pandas as pd
import pytest

from src.eval.live import with_baseline
from src.eval.panel import panel_test
from src.eval.pooled import fold_schedule, pooled_walk_forward
from src.eval.synthetic import make_synthetic

KW = dict(train_dates=252, n_boot=20)
TICKERS = 5


def _panel(signal: float, days: int = 1300) -> tuple[dict, dict, pd.Series]:
    data = [make_synthetic(days, signal, seed=s) for s in range(TICKERS)]
    return (
        {f"T{i}": d.features for i, d in enumerate(data)},
        {f"T{i}": d.log_returns for i, d in enumerate(data)},
        data[0].vix,
    )


@pytest.fixture(scope="module")
def planted() -> tuple[dict, dict, pd.Series]:
    return _panel(3.0)


@pytest.fixture(scope="module")
def null() -> tuple[dict, dict, pd.Series]:
    return _panel(0.0)


def _gap(panel: tuple, candidate: str) -> float:
    """Panel mean relative loss of the candidate against the featureless pooled control."""
    f, lr, vix = panel
    cand = pooled_walk_forward(f, lr, vix, candidate, **KW)
    control = pooled_walk_forward(f, lr, vix, "fhs-pooled", **KW)
    paired = {t: with_baseline(cand[t], "control", control[t]) for t in cand}
    return panel_test(paired, "control").mean_relative_diff


def test_a_planted_news_effect_is_recovered_and_price_features_do_not_find_it(
    planted: tuple,
) -> None:
    news, price = _gap(planted, "lq-a0.01-news"), _gap(planted, "lq-a0.01-price")
    assert news < -0.08  # measured about -0.14
    assert price > news / 2  # the effect lives in the news features, not the price ones


def test_with_no_effect_the_model_gains_nothing_over_the_control(null: tuple) -> None:
    for cid in ("lq-a0.01-news", "gb-50-news"):
        assert abs(_gap(null, cid)) < 0.01  # measured within 0.2%


def test_results_are_in_return_units_with_baselines_on_identical_rows(planted: tuple) -> None:
    f, lr, vix = planted
    res = pooled_walk_forward(f, lr, vix, "lq-a0.01-price", **KW)
    other = pooled_walk_forward(f, lr, vix, "fhs-pooled", **KW)
    assert set(res) == set(f)
    for t, r in res.items():
        assert r.index.equals(other[t].index) and np.array_equal(r.y, other[t].y)
        assert {"model", "fhs", "garch", "empirical_quantile", "momentum", "zero"} <= set(r.preds)
        assert r.preds["model"].shape == (len(r.y), 3)
        assert (np.diff(r.preds["model"], axis=1) >= 0).all()
        # return units: the model's interval is on the scale of the garch interval
        ratio = np.median(r.preds["model"][:, 2] - r.preds["model"][:, 0]) / np.median(
            r.preds["garch"][:, 2] - r.preds["garch"][:, 0]
        )
        assert 0.5 < ratio < 2.0


def test_pooled_control_uses_one_standardised_distribution_for_all_tickers(
    planted: tuple,
) -> None:
    f, lr, vix = planted
    res = pooled_walk_forward(f, lr, vix, "fhs-pooled", **KW)
    from src.prediction.volatility import ewma_vol

    steps = 5
    ratios = []
    for t, r in res.items():
        sig = ewma_vol(lr[t]).reindex(r.index).to_numpy()
        ratios.append(r.preds["model"] / (sig * np.sqrt(steps))[:, None])
    # on the same date every ticker gets the same standardised quantiles
    for other in ratios[1:]:
        assert np.allclose(other, ratios[0], atol=1e-9)


def test_scoring_window_is_decided_by_dates_not_by_candidate(planted: tuple) -> None:
    f, lr, vix = planted
    cap, first = "2022-06-30", "2021-06-01"
    a = pooled_walk_forward(
        f, lr, vix, "lq-a0.01-price", score_start=first, label_end_max=cap, **KW
    )
    b = pooled_walk_forward(f, lr, vix, "gb-50-news", score_start=first, label_end_max=cap, **KW)
    for t in a:
        assert a[t].index.equals(b[t].index)
        assert a[t].index[0] >= pd.Timestamp(first, tz="UTC")
        last_bar = lr[t].index[lr[t].index.get_loc(a[t].index[-1]) + 5]
        assert last_bar <= pd.Timestamp(cap, tz="UTC")  # the last label ends on or before the cap
        # and the next row's label would not: the cap is tight
        nxt = lr[t].index[lr[t].index.get_loc(a[t].index[-1]) + 1 + 5]
        assert nxt > pd.Timestamp(cap, tz="UTC")


def test_forecasts_ignore_data_after_the_scored_rows_and_react_to_data_before_them(
    planted: tuple,
) -> None:
    f, lr, vix = planted
    cut = 1100  # positions on the synthetic calendar
    end = lr["T0"].index[cut]
    f_cut = {t: x.loc[x.index <= end] for t, x in f.items()}
    lr_cut = {t: x.loc[x.index <= end] for t, x in lr.items()}
    full = pooled_walk_forward(f, lr, vix, "lq-a0.01-news", **KW)
    short = pooled_walk_forward(f_cut, lr_cut, vix.loc[vix.index <= end], "lq-a0.01-news", **KW)
    for t in short:
        common = short[t].index.intersection(full[t].index)
        assert len(common) > 100
        a = short[t].preds["model"][short[t].index.get_indexer(common)]
        b = full[t].preds["model"][full[t].index.get_indexer(common)]
        assert np.allclose(a, b, atol=1e-9)

    # negative control: perturbing a training-window return does change later forecasts
    changed = {t: x.copy() for t, x in lr_cut.items()}
    for x in changed.values():
        x.iloc[600:620] *= 3.0
    moved = pooled_walk_forward(f_cut, changed, vix.loc[vix.index <= end], "lq-a0.01-news", **KW)
    assert not np.allclose(moved["T0"].preds["model"], short["T0"].preds["model"])


def test_invalid_inputs_raise(planted: tuple) -> None:
    f, lr, vix = planted
    with pytest.raises(ValueError, match="unknown candidate"):
        pooled_walk_forward(f, lr, vix, "nope", **KW)
    with pytest.raises(ValueError, match="same tickers"):
        pooled_walk_forward(f, {"T0": lr["T0"]}, vix, "fhs-pooled", **KW)
    with pytest.raises(ValueError, match="vix is required"):
        pooled_walk_forward(f, lr, None, "lq-a0.01-news", **KW)
    with pytest.raises(ValueError, match="no scored rows"):
        pooled_walk_forward(f, lr, vix, "fhs-pooled", score_start="2030-01-01", **KW)
    gapped = {**f, "T0": f["T0"].drop(f["T0"].index[700])}
    with pytest.raises(ValueError, match="consecutive"):
        pooled_walk_forward(gapped, lr, vix, "fhs-pooled", **KW)
    nan = {**f, "T0": f["T0"].copy()}
    nan["T0"].iloc[800, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        pooled_walk_forward(nan, lr, vix, "fhs-pooled", **KW)


def test_fold_schedule_has_the_protocol_window_refit_and_embargo() -> None:
    n, train, refit, steps = 1100, 504, 21, 5
    folds = fold_schedule(n, train, refit, steps)
    for f in folds:
        assert len(f.train) == train
        # the last training row's label (steps bars ahead) ends before the first test row
        assert f.train[-1] + steps < f.test[0]
        assert f.test[0] - f.train[-1] - 1 == steps
        assert (np.diff(f.test) == 1).all() and len(f.test) <= refit
    assert all(b.test[0] == a.test[-1] + 1 for a, b in zip(folds, folds[1:]))  # no gaps, no overlap
    assert folds[0].test[0] == train + steps
    assert folds[-1].test[-1] == n - 1  # the newest dates are scored, even in a partial block
    assert np.array_equal(folds[1].train, folds[0].train + refit)


def test_the_newest_dates_are_scored(planted: tuple) -> None:
    f, lr, vix = planted
    res = pooled_walk_forward(f, lr, vix, "fhs-pooled", **KW)
    # the last row with a realised 5-bar label is the 6th from the end
    assert res["T0"].index[-1] == f["T0"].index[-6]


def test_tickers_with_different_start_dates_pool_correctly(planted: tuple) -> None:
    f, lr, vix = planted
    late = {**f, "T0": f["T0"].iloc[-200:]}
    res = pooled_walk_forward(late, lr, vix, "lq-a0.01-price", **KW)
    assert res["T0"].index[0] > res["T1"].index[0]
    assert len(res["T0"].y) > 100


def test_a_ticker_missing_a_calendar_date_is_rejected(planted: tuple) -> None:
    f, lr, vix = planted
    day = f["T0"].index[700]
    holed_f = {**f, "T0": f["T0"].drop(day)}
    holed_lr = {**lr, "T0": lr["T0"].drop(day)}
    with pytest.raises(ValueError, match="contiguous"):
        pooled_walk_forward(holed_f, holed_lr, vix, "fhs-pooled", **KW)


def test_a_stale_vix_series_is_rejected(planted: tuple) -> None:
    f, lr, vix = planted
    with pytest.raises(ValueError, match="stale"):
        pooled_walk_forward(f, lr, vix.iloc[:700], "lq-a0.01-news", **KW)


def test_baseline_cache_gives_identical_results(planted: tuple) -> None:
    f, lr, vix = planted
    cache: dict = {}
    a = pooled_walk_forward(f, lr, vix, "fhs-pooled", baseline_cache=cache, **KW)
    n = len(cache)
    b = pooled_walk_forward(f, lr, vix, "lq-a0.01-price", baseline_cache=cache, **KW)
    assert n == len(f) == len(cache)
    for t in a:
        assert all(np.array_equal(a[t].preds[k], b[t].preds[k]) for k in a[t].preds if k != "model")
