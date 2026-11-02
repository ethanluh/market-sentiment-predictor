import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.eval import runner
from src.eval.runner import evaluate, render_markdown, save_summary
from src.eval.synthetic import SyntheticData, make_synthetic

KW = dict(train_window=150, test_window=20, backend="linear", n_boot=100)


@pytest.fixture(scope="module")
def planted() -> SyntheticData:
    return make_synthetic(n_days=900, signal=2.0, seed=1)


@pytest.fixture(scope="module")
def planted_result(planted: SyntheticData) -> runner.EvalResult:
    return evaluate(planted.features, planted.log_returns, "1d", vix=planted.vix, **KW)


def test_planted_signal_beats_every_baseline_significantly(
    planted_result: runner.EvalResult,
) -> None:
    s = planted_result.summary
    assert s["skill_vs_empirical_quantile"] > 0.02
    for name, v in s["model_vs"].items():
        assert v["loss_diff"] < 0, name
        assert v["dm_p"] < 0.05, name
        assert v["diff_ci95"][1] < 0, name


def test_no_signal_control_shows_no_edge() -> None:
    d = make_synthetic(n_days=900, signal=0.0, seed=1)
    s = evaluate(d.features, d.log_returns, "1d", vix=d.vix, **KW).summary
    v = s["model_vs"]["empirical_quantile"]
    assert not (v["loss_diff"] < 0 and v["dm_p"] < 0.05)
    assert s["skill_vs_empirical_quantile"] < 0.01


def test_predictions_start_after_train_window_plus_embargo(
    planted: SyntheticData, planted_result: runner.EvalResult
) -> None:
    assert planted_result.index[0] >= planted.features.index[KW["train_window"] + 1]
    assert planted_result.index.is_monotonic_increasing
    n = len(planted_result.y)
    assert all(p.shape == (n, 3) for p in planted_result.preds.values())
    assert not np.isnan(planted_result.y).any()


def test_five_day_horizon_uses_cumulative_target_and_larger_embargo(planted: SyntheticData) -> None:
    res = evaluate(planted.features, planted.log_returns, "5d", vix=planted.vix, **KW)
    assert res.steps == 5
    # First test row sits embargo=5 rows after the first fold's training window.
    assert res.index[0] == planted.features.index[KW["train_window"] + 5]
    assert res.summary["trading"]["n_periods"] == pytest.approx(len(res.y) / 5, abs=2)


def test_regime_classifier_is_fit_on_each_folds_training_rows_only(
    planted: SyntheticData, monkeypatch: pytest.MonkeyPatch
) -> None:
    fits: list[np.ndarray] = []
    real = runner.RegimeClassifier

    class Spy(real):  # type: ignore[valid-type, misc]
        def fit(self, vix):  # type: ignore[no-untyped-def]
            fits.append(np.asarray(vix).copy())
            return super().fit(vix)

    monkeypatch.setattr(runner, "RegimeClassifier", Spy)
    res = evaluate(planted.features, planted.log_returns, "1d", vix=planted.vix, **KW)
    assert len(fits) == res.n_folds
    assert all(len(f) == KW["train_window"] for f in fits)
    expected_first = planted.vix.reindex(planted.features.index).to_numpy()[: KW["train_window"]]
    assert np.array_equal(fits[0], expected_first)


def test_nan_features_are_rejected(planted: SyntheticData) -> None:
    bad = planted.features.copy()
    bad.iloc[10, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        evaluate(bad, planted.log_returns, "1d", **KW)


def test_non_consecutive_rows_are_rejected(planted: SyntheticData) -> None:
    gappy = planted.features.drop(planted.features.index[100])
    with pytest.raises(ValueError, match="consecutive"):
        evaluate(gappy, planted.log_returns, "1d", **KW)


def test_vix_missing_for_early_rows_is_rejected(planted: SyntheticData) -> None:
    late = planted.vix[planted.vix.index > planted.features.index[50]]
    with pytest.raises(ValueError, match="vix"):
        evaluate(planted.features, planted.log_returns, "1d", vix=late, **KW)


def test_summary_json_round_trip_maps_nan_to_null(
    planted_result: runner.EvalResult, tmp_path: Path
) -> None:
    summary = json.loads(json.dumps(planted_result.summary))
    summary["trading"]["sharpe_net"] = float("nan")
    out = tmp_path / "s.json"
    save_summary(summary, out)
    loaded = json.loads(out.read_text())
    assert loaded["trading"]["sharpe_net"] is None
    assert loaded["horizon"] == "1d"


def test_markdown_lists_every_forecaster(planted_result: runner.EvalResult) -> None:
    md = render_markdown(planted_result.summary)
    for name in planted_result.preds:
        assert f"| {name} |" in md


def test_evaluation_is_deterministic(planted: SyntheticData) -> None:
    a = evaluate(planted.features, planted.log_returns, "1d", vix=planted.vix, **KW)
    b = evaluate(planted.features, planted.log_returns, "1d", vix=planted.vix, **KW)
    assert a.summary == b.summary


def test_synthetic_features_are_causal_and_signal_points_forward() -> None:
    d = make_synthetic(n_days=600, signal=1.5, seed=3)
    r = d.log_returns
    for t in d.features.index[::97]:
        past = r.loc[:t]
        row = d.features.loc[t]
        assert row["momentum_5d"] == pytest.approx(past.tail(5).sum())
        assert row["realized_vol_20d"] == pytest.approx(past.tail(20).std(ddof=0))
        assert row["return_lag_1"] == pytest.approx(past.iloc[-1])
    sentiment = d.features["sentiment_agg"]
    next_ret = r.shift(-1).reindex(sentiment.index)
    same_ret = r.reindex(sentiment.index)
    # Sentiment is AR(1), so it also correlates with today's return via yesterday's value;
    # the forward link must still be clearly stronger.
    assert sentiment.corr(next_ret) > 0.15
    assert sentiment.corr(next_ret) > 1.5 * sentiment.corr(same_ret)


def test_step_cannot_be_set_so_test_rows_never_repeat(planted: SyntheticData) -> None:
    with pytest.raises(TypeError):
        evaluate(planted.features, planted.log_returns, "1d", step=10, **KW)  # type: ignore[call-arg]
    res = evaluate(planted.features, planted.log_returns, "1d", **KW)
    assert res.index.is_unique


def test_regime_neutralisation_survives_evaluation(planted: SyntheticData) -> None:
    from src.eval.ablation import neutralize

    price_only = evaluate(neutralize(planted.features, ["regime"]), planted.log_returns, "1d", **KW)
    with_regime = evaluate(planted.features, planted.log_returns, "1d", vix=planted.vix, **KW)
    assert not np.array_equal(price_only.preds["model"], with_regime.preds["model"])
