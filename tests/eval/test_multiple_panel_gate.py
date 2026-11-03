import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.multitest import multipletests

from src.eval.gate import GateRule, apply_gate
from src.eval.multiple import benjamini_hochberg, one_sided_p
from src.eval.panel import PanelTest, panel_test, relative_loss_differential
from src.eval.runner import EvalResult

# ---- multiple testing ----------------------------------------------------------------------------


def test_bh_matches_statsmodels_on_random_inputs() -> None:
    rng = np.random.default_rng(0)
    for _ in range(50):
        p = rng.uniform(0, 1, rng.integers(1, 40))
        np.testing.assert_allclose(benjamini_hochberg(p), multipletests(p, method="fdr_bh")[1])


def test_bh_hand_computed_case() -> None:
    # sorted p: .005 .01 .03 .04 (m=4) -> raw q: .02 .02 .04 .04, already monotone
    q = benjamini_hochberg([0.01, 0.04, 0.03, 0.005])
    np.testing.assert_allclose(q, [0.02, 0.04, 0.04, 0.02])


def test_bh_enforces_monotonicity_and_caps_at_one() -> None:
    q = benjamini_hochberg([0.01, 0.9, 0.04, 0.03])
    assert q.max() <= 1.0 and (q >= np.array([0.01, 0.9, 0.04, 0.03])).all()
    ordered = q[np.argsort([0.01, 0.9, 0.04, 0.03])]
    assert (np.diff(ordered) >= 0).all()


def test_bh_controls_false_discoveries_under_the_global_null() -> None:
    rng = np.random.default_rng(1)
    naive = adjusted = 0
    for _ in range(400):
        p = rng.uniform(0, 1, 23)  # 23 tickers, no real edge anywhere
        naive += int((p < 0.05).any())
        adjusted += int((benjamini_hochberg(p) <= 0.10).any())
    assert naive / 400 > 0.6  # testing each ticker at 5% finds a "winner" most of the time
    assert adjusted / 400 < 0.16  # BH keeps the chance of any false discovery near 10%


def test_bh_validates_input() -> None:
    for bad in ([], [0.5, float("nan")], [-0.1], [1.2]):
        with pytest.raises(ValueError):
            benjamini_hochberg(bad)


def test_one_sided_p() -> None:
    assert one_sided_p(-2.0, 0.04) == pytest.approx(0.02)  # the model is better
    assert one_sided_p(2.0, 0.04) == pytest.approx(0.98)  # the model is worse
    with pytest.raises(ValueError):
        one_sided_p(1.0, 1.5)


# ---- panel ---------------------------------------------------------------------------------------

IDX = pd.bdate_range("2023-01-02", periods=400, tz="UTC")


def result(model_scale: float, seed: int, index=IDX, steps: int = 1) -> EvalResult:  # type: ignore[no-untyped-def]
    """A forecaster whose quantile band is ``model_scale`` times as wide as the baseline's."""
    rng = np.random.default_rng(seed)
    n = len(index)
    y = rng.normal(0, 0.01, n)
    z = np.array([-1.2816, 0.0, 1.2816]) * 0.01
    base = np.tile(z, (n, 1))
    return EvalResult(
        "1d", steps, 10, index, y, {"model": base * model_scale, "empirical_quantile": base}
    )


def test_a_better_model_gives_a_negative_differential_and_a_small_p() -> None:
    # A band matched to the true volatility beats one that is 3x too wide.
    results = {f"T{i}": result(1.0, i) for i in range(5)}
    for r in results.values():
        r.preds["empirical_quantile"] = r.preds["empirical_quantile"] * 3.0
    pt = panel_test(results, "empirical_quantile")
    assert pt.mean_relative_diff < 0 and pt.dm_stat < 0 and pt.dm_p_two_sided < 0.001
    assert pt.dm_p_two_sided == max(pt.dm_p_hac, pt.p_bootstrap)  # the more conservative of the two
    assert (pt.n_dates, pt.n_tickers) == (400, 5)


def test_identical_forecasters_have_zero_differential() -> None:
    # model == baseline: zero variance, which the DM test refuses rather than reporting a p-value
    results = {"A": result(1.0, 1)}
    with pytest.raises(ValueError, match="variance"):
        panel_test(results, "empirical_quantile")
    assert relative_loss_differential(results["A"], "empirical_quantile").abs().max() == 0.0


def test_two_equally_good_forecasters_show_no_edge_at_the_nominal_rate() -> None:
    """Null: each forecaster estimates the true band with independent error, so neither is better.
    The panel test should reject at about its nominal 5% rate, not much more."""
    z = np.array([-1.2816, 0.0, 1.2816])

    def null_panel(seed: int) -> float:
        rng = np.random.default_rng(seed)
        results = {}
        for i in range(6):
            y = rng.normal(0, 0.01, len(IDX))
            model = z[None, :] * 0.01 * (1 + 0.15 * rng.normal(size=(len(IDX), 1)))
            base = z[None, :] * 0.01 * (1 + 0.15 * rng.normal(size=(len(IDX), 1)))
            results[f"T{i}"] = EvalResult(
                "1d", 1, 10, IDX, y, {"model": model, "empirical_quantile": base}
            )
        return panel_test(results, "empirical_quantile").dm_p_two_sided

    ps = np.array([null_panel(seed) for seed in range(60)])
    assert (ps < 0.05).mean() < 0.15  # nominal 5%; allow sampling noise
    assert ps.mean() > 0.3  # roughly uniform, not piled up near zero


def test_a_systematically_worse_model_is_flagged_as_worse() -> None:
    rng = np.random.default_rng(3)
    results = {}
    for i in range(6):
        r = result(1.0, i)
        r.preds["model"] = r.preds["model"] * (
            1.0 + 0.05 * rng.normal(size=(len(IDX), 1))
        )  # off-optimum
        results[f"T{i}"] = r
    pt = panel_test(results, "empirical_quantile")
    assert pt.mean_relative_diff > 0 and pt.dm_stat > 0  # lower loss is better; this model is worse


def test_a_single_ticker_panel_equals_its_own_series() -> None:
    r = result(0.8, 4)
    r.preds["empirical_quantile"] = r.preds["empirical_quantile"] * 1.2
    pt = panel_test({"A": r}, "empirical_quantile")
    assert pt.mean_relative_diff == pytest.approx(
        relative_loss_differential(r, "empirical_quantile").mean()
    )


def test_tickers_are_averaged_per_date_even_with_different_date_sets() -> None:
    a, b = result(0.8, 1), result(0.8, 2, index=IDX[100:300])
    for r in (a, b):
        r.preds["empirical_quantile"] = r.preds["empirical_quantile"] * 1.5
    pt = panel_test({"A": a, "B": b}, "empirical_quantile")
    assert pt.n_dates == 400 and pt.n_tickers == 2
    both_only = panel_test({"A": a, "B": b}, "empirical_quantile", min_tickers_per_date=2)
    assert both_only.n_dates == 200


def test_panel_inputs_are_validated() -> None:
    with pytest.raises(ValueError, match="no results"):
        panel_test({}, "empirical_quantile")
    with pytest.raises(ValueError, match="no baseline"):
        panel_test({"A": result(1.0, 1)}, "nope")
    with pytest.raises(ValueError, match="step counts"):
        panel_test(
            {"A": result(1.0, 1, steps=1), "B": result(1.0, 2, steps=5)}, "empirical_quantile"
        )


# ---- the gate ------------------------------------------------------------------------------------


def summary(skill: float, dm_stat: float, dm_p: float, cov: float = 0.8, n: int = 1000) -> dict:
    return {
        "n_predictions": n,
        "skill_vs_empirical_quantile": skill,
        "model_vs": {"empirical_quantile": {"dm_stat": dm_stat, "dm_p": dm_p}},
        "forecasters": {"model": {"coverage_80": cov}},
    }


GOOD_PANEL = PanelTest(-0.02, -3.0, 0.003, 400, 6)


def six(**per) -> dict:  # type: ignore[no-untyped-def]
    base = {f"T{i}": summary(0.03, -3.0, 0.003) for i in range(6)}
    base.update(per)
    return base


def test_a_clear_edge_passes() -> None:
    res = apply_gate(six(), GOOD_PANEL)
    assert res.passed and res.reasons == [] and len(res.details["passing_tickers"]) == 6


def test_a_panel_that_is_not_significant_fails() -> None:
    res = apply_gate(six(), PanelTest(-0.01, -1.0, 0.30, 400, 6))
    assert not res.passed and any(r.startswith("panel:") for r in res.reasons)


def test_a_panel_where_the_model_is_worse_fails_even_if_the_stat_is_large() -> None:
    res = apply_gate(six(), PanelTest(+0.02, 3.0, 0.003, 400, 6))
    assert not res.passed and any(r.startswith("panel:") for r in res.reasons)


def test_too_few_tickers_with_a_real_edge_fails() -> None:
    weak = {f"T{i}": summary(0.0005, -0.5, 0.6) for i in range(6)}
    weak["T0"] = summary(0.03, -3.0, 0.003)
    res = apply_gate(weak, GOOD_PANEL)
    assert not res.passed and any("positive edge at FDR" in r for r in res.reasons)


def test_fdr_control_rejects_one_lucky_ticker_among_many() -> None:
    lucky = {f"T{i}": summary(-0.01, 0.5, 0.6) for i in range(23)}
    lucky["T0"] = summary(0.02, -2.0, 0.046)  # p = 0.023 one-sided: significant alone, not after BH
    res = apply_gate(lucky, GOOD_PANEL, GateRule(min_tickers_passing=1))
    assert not res.passed and res.details["passing_tickers"] == []


def test_miscalibrated_forecasts_fail_even_with_an_edge() -> None:
    res = apply_gate({f"T{i}": summary(0.03, -3.0, 0.003, cov=0.55) for i in range(6)}, GOOD_PANEL)
    assert not res.passed and any("coverage" in r for r in res.reasons)


def test_tickers_with_too_little_data_are_excluded_and_reported() -> None:
    s = six()
    for t in ("T0", "T1", "T2", "T3"):
        s[t] = summary(0.03, -3.0, 0.003, n=100)
    res = apply_gate(s, GOOD_PANEL)
    assert not res.passed and any("scored rows" in r for r in res.reasons)
    assert res.details["excluded_tickers"] == ["T0", "T1", "T2", "T3"]


def test_every_failing_criterion_is_reported() -> None:
    bad = {f"T{i}": summary(-0.02, 2.0, 0.05, cov=0.5, n=50) for i in range(4)}
    res = apply_gate(bad, PanelTest(0.01, 1.0, 0.3, 400, 4))
    assert len(res.reasons) >= 3


def test_gate_rule_validates_itself() -> None:
    for kwargs in (
        {"max_panel_p": 0.0},
        {"max_fdr": 1.0},
        {"min_tickers_passing": 0},
        {"coverage_band": (0.9, 0.7)},
    ):
        with pytest.raises(ValueError):
            GateRule(**kwargs)  # type: ignore[arg-type]


def test_the_panel_uses_only_the_longest_consecutive_stretch_of_covered_dates() -> None:
    a = result(0.8, 1)  # all 400 dates
    b = result(0.8, 2, index=IDX.delete(range(150, 200)))  # a hole in the middle
    for r in (a, b):
        r.preds["empirical_quantile"] = r.preds["empirical_quantile"] * 1.5
    pt = panel_test({"A": a, "B": b}, "empirical_quantile", min_tickers_per_date=2)
    # both tickers cover 0-149 and 200-399; the longer block (200 dates) is used, with no gap inside
    assert pt.n_dates == 200


def test_by_default_a_date_needs_half_of_the_tickers() -> None:
    a = result(0.8, 1)
    shorts = {f"S{i}": result(0.8, 10 + i, index=IDX[:50]) for i in range(3)}  # only 50 dates each
    for r in [a, *shorts.values()]:
        r.preds["empirical_quantile"] = r.preds["empirical_quantile"] * 1.5
    pt = panel_test({"A": a, **shorts}, "empirical_quantile")  # need ceil(4/2) = 2 tickers per date
    assert pt.n_dates == 50 and pt.n_tickers == 4


def test_a_ticker_outside_the_per_ticker_coverage_band_fails_even_if_the_average_is_fine() -> None:
    s = {f"T{i}": summary(0.03, -3.0, 0.003, cov=0.80) for i in range(6)}
    s["T0"] = summary(0.03, -3.0, 0.003, cov=0.55)
    s["T1"] = summary(0.03, -3.0, 0.003, cov=0.97)  # the mean is still about 0.80
    res = apply_gate(s, GOOD_PANEL)
    assert 0.70 <= res.details["mean_coverage"] <= 0.90
    assert not res.passed and any("ticker P10-P90 coverage outside" in r for r in res.reasons)
    assert "T0" in "".join(res.reasons) and "T1" in "".join(res.reasons)


def test_the_ticker_coverage_band_is_validated() -> None:
    with pytest.raises(ValueError):
        GateRule(ticker_coverage_band=(0.9, 0.6))
