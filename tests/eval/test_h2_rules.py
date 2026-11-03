import numpy as np
import pandas as pd
import pytest

from src.eval.h2_gate import H2GateRule, apply_h2_gate, compare_pipelines
from src.eval.h2_select import MATERIALITY, TIE_TOLERANCE, select
from src.eval.runner import EvalResult, summarize
from src.prediction.vol_scaled import CANDIDATES, COMPLEXITY_ORDER, CONTROL

Z = 1.2815515655446004
N = 600


def _result(
    seed: int, model_scale: float, garch_scale: float = 1.6, fhs_scale: float = 1.6, n: int = N
) -> EvalResult:
    """True return scale is ``s``; each forecaster's interval half-width is ``Z * scale * s``."""
    rng = np.random.default_rng(seed)
    ar = np.zeros(n)
    for t in range(1, n):
        ar[t] = 0.9 * ar[t - 1] + 0.3 * rng.normal()
    s = 0.01 * np.exp(ar)
    y = s * rng.normal(size=n)

    def q(scale: float) -> np.ndarray:
        h = Z * scale * s
        return np.column_stack([-h, np.zeros(n), h])

    idx = pd.bdate_range("2024-01-02", periods=n, tz="UTC")
    zero = np.zeros((n, 3))
    preds = {
        "model": q(model_scale),
        "garch": q(garch_scale),
        "fhs": q(fhs_scale),
        "empirical_quantile": q(fhs_scale),
        "momentum": zero,
        "zero": zero,
    }
    res = EvalResult("5d", 5, 1, idx, y, preds)
    res.summary = summarize(res, cost_bps=1.0, n_boot=20, seed=0)
    return res


def _panel(model_scale: float, tickers: int = 8, **kw: float) -> dict[str, EvalResult]:
    return {f"T{i}": _result(i, model_scale, **kw) for i in range(tickers)}


def test_gate_passes_when_the_model_clearly_beats_both_references() -> None:
    g = apply_h2_gate(_panel(1.0))
    assert g.passed, g.reasons
    assert len(g.details["passing_tickers"]) >= 3  # type: ignore[arg-type]


def test_gate_fails_when_the_model_is_no_better_than_the_references() -> None:
    g = apply_h2_gate(_panel(1.62))
    assert not g.passed
    assert any("panel vs garch" in r for r in g.reasons)


def test_beating_only_one_reference_is_not_enough() -> None:
    g = apply_h2_gate(_panel(1.0, fhs_scale=1.002))  # fhs is as good as the model
    assert not g.passed
    assert any("panel vs fhs" in r for r in g.reasons)
    assert not any("panel vs garch" in r for r in g.reasons)


def test_gate_rejects_miscalibrated_intervals_even_with_an_edge() -> None:
    g = apply_h2_gate(_panel(0.5, garch_scale=2.5, fhs_scale=2.5))  # too narrow
    assert not g.passed
    assert any("coverage" in r for r in g.reasons)


def test_too_many_excluded_tickers_fail_before_anything_else() -> None:
    results = {f"T{i}": _result(i, 1.0, n=300 if i < 3 else N) for i in range(8)}
    g = apply_h2_gate(results)  # 3 of 8 below 500 rows: 5 usable < ceil(0.75 * 8) = 6
    assert not g.passed
    assert "only 5 of 8" in g.reasons[0]
    assert g.details["excluded_tickers"] == ["T0", "T1", "T2"]


def test_a_short_ticker_within_the_allowance_is_dropped_from_every_criterion() -> None:
    results = {f"T{i}": _result(i, 1.0, n=300 if i == 0 else N) for i in range(8)}
    g = apply_h2_gate(results)
    assert g.passed, g.reasons
    assert g.details["excluded_tickers"] == ["T0"]
    assert "T0" not in g.details["q_values_vs_garch"]  # type: ignore[operator]


def test_gate_rule_validation() -> None:
    with pytest.raises(ValueError):
        H2GateRule(references=())
    with pytest.raises(ValueError):
        H2GateRule(max_panel_p=0.0)
    with pytest.raises(ValueError):
        apply_h2_gate({})


def test_h2b_needs_the_news_pipeline_to_beat_the_price_pipeline() -> None:
    assert compare_pipelines(_panel(1.0), _panel(1.4)).passed
    worse = compare_pipelines(_panel(1.4), _panel(1.0))
    assert not worse.passed and "news vs price" in worse.reasons[0]
    with pytest.raises(ValueError, match="same tickers"):
        compare_pipelines(_panel(1.0, tickers=8), _panel(1.0, tickers=7))


# ---- selection ----


def _dev(
    scale_of: dict[str, float], control_scale: float = 1.6
) -> dict[str, dict[str, EvalResult]]:
    out = {CONTROL.id: _panel(control_scale, tickers=4, garch_scale=1.61, fhs_scale=1.61)}
    for c in COMPLEXITY_ORDER:
        out[c.id] = _panel(scale_of.get(c.id, 1.62), tickers=4, garch_scale=1.61, fhs_scale=1.61)
    return out


def test_selection_picks_the_best_of_each_feature_set_and_the_overall_winner() -> None:
    sel = select(
        _dev({"lq-a0.01-price": 1.3, "gb-50-price": 1.2, "gb-50-news": 1.0, "gb-100-news": 1.1})
    )
    assert sel.best_price == "gb-50-price"
    assert sel.best_news == "gb-50-news"
    assert sel.overall == "gb-50-news"
    assert sel.proceed, sel.reasons


def test_ties_go_to_the_simpler_candidate() -> None:
    sel = select(_dev({"lq-a0.001-news": 1.0, "gb-100-news": 1.0, "gb-50-price": 1.3}))
    assert sel.best_news == "lq-a0.001-news"  # identical scores; earlier in the complexity order


def test_a_candidate_just_inside_the_tolerance_loses_to_a_simpler_one_but_not_beyond_it() -> None:
    near = _dev({"lq-a0.01-news": 1.0, "gb-100-news": 1.0005})
    scores = select(near).scores
    assert abs(scores["lq-a0.01-news"]["fhs"] - scores["gb-100-news"]["fhs"]) < TIE_TOLERANCE
    clear = select(_dev({"lq-a0.01-news": 1.3, "gb-100-news": 1.0}))
    assert clear.best_news == "gb-100-news"  # much better beats simpler


def test_the_holdout_stays_closed_when_nothing_clears_the_materiality_margin() -> None:
    sel = select(_dev({}))
    assert not sel.proceed
    assert len(sel.reasons) == 3 and str(f"{MATERIALITY:+.2%}") in sel.reasons[0]


def test_beating_the_baselines_but_not_the_pooled_control_does_not_proceed() -> None:
    sel = select(_dev({"gb-50-news": 1.0}, control_scale=1.002))  # the control is as good
    assert not sel.proceed
    assert any("fhs-pooled" in r for r in sel.reasons)


def test_selection_requires_every_candidate() -> None:
    dev = _dev({})
    del dev["gb-100-news"]
    with pytest.raises(ValueError, match="exactly"):
        select(dev)
    assert set(_dev({})) == set(CANDIDATES)


def test_the_overall_winner_is_the_lowest_s_over_all_eight_not_a_second_tolerance_pass() -> None:
    from src.eval.h2_select import _pick

    scores = {c.id: {"fhs": 0.0} for c in COMPLEXITY_ORDER}
    scores["lq-a0.01-price"]["fhs"] = -0.0086
    scores["gb-50-price"]["fhs"] = -0.0095
    scores["lq-a0.01-news"]["fhs"] = -0.0091
    scores["gb-100-news"]["fhs"] = -0.0100
    ids = [c.id for c in COMPLEXITY_ORDER]
    assert (
        _pick(ids, scores) == "lq-a0.01-news"
    )  # within 0.001 of -0.0100 and simpler than gb-100-news
    price = [c.id for c in COMPLEXITY_ORDER if c.feature_set == "price"]
    news = [c.id for c in COMPLEXITY_ORDER if c.feature_set == "news"]
    assert _pick(price, scores) == "lq-a0.01-price"
    assert _pick(news, scores) == "lq-a0.01-news"
