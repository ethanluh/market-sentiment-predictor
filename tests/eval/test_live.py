import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.eval.gate import GateRule
from src.eval.live import (
    LiveEvalConfig,
    render_report_markdown,
    run_live_eval,
    save_report,
    with_baseline,
)
from src.eval.run import main
from src.eval.runner import EvalResult
from src.prediction.dataset import Sources

N = 740
IDX = pd.bdate_range("2021-01-04", periods=N, tz="UTC")
TICKERS = ("AAA", "BBB", "CCC")


def make_prices(seed: int) -> dict[str, pd.DataFrame]:
    """Correlated random walks with no predictable structure: a genuine null."""
    rng = np.random.default_rng(seed)
    common = rng.normal(0, 0.01, N)
    out = {}
    for i, t in enumerate(TICKERS):
        r = 0.7 * common + rng.normal(0, 0.008, N)
        close = 100 * np.exp(np.cumsum(r))
        df = pd.DataFrame(
            {"close": close, "volume": rng.integers(1e6, 5e6, N).astype(float)}, index=IDX
        )
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        out[t] = df
    return out


def fake_sources(seed: int = 0, spy: list | None = None) -> Sources:
    prices = make_prices(seed)
    vix = pd.DataFrame(
        {"close": 15 + 8 * np.abs(np.sin(np.arange(N) / 23.0)), "volume": 1.0, "log_return": 0.0},
        index=IDX,
    )
    boom = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("source not expected")
    )  # noqa: E731

    def returns_matrix(tickers, start, end, interval):  # type: ignore[no-untyped-def]
        if spy is not None:
            spy.append(tuple(tickers))
        return pd.DataFrame({t: prices[t]["log_return"] for t in tickers})

    return Sources(
        prices=lambda t, s, e, i: vix if t == "^VIX" else prices[t],
        returns_matrix=returns_matrix,
        trends=boom,
        insider_flow=boom,
        news_reader=boom,
    )


def cfg(**kw) -> LiveEvalConfig:  # type: ignore[no-untyped-def]
    base = dict(
        universe=TICKERS,
        start="2021-01-01",
        horizon="5d",
        backend="linear",
        use_news_archive=False,
        train_window=150,
        test_window=20,
        n_boot=40,
    )
    base.update(kw)
    return LiveEvalConfig(**base)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def report():  # type: ignore[no-untyped-def]
    return run_live_eval(cfg(), fake_sources(), progress=lambda m: None)


def test_the_report_covers_every_ticker_and_config(report) -> None:  # type: ignore[no-untyped-def]
    assert set(report.results) == {"price_only", "full"}
    for rs in report.results.values():
        assert set(rs) == set(TICKERS)
        assert all(r.horizon == "5d" and r.steps == 5 and r.n_folds > 3 for r in rs.values())
    assert report.data["AAA"]["rows"] > 600 and report.data["AAA"]["news_coverage"] is None
    # The gate only judges tickers with at least 500 scored rows; this fixture must clear that bar
    # or the gate tests below would pass for the wrong reason (every ticker silently excluded).
    assert all(r.summary["n_predictions"] >= 500 for r in report.results["full"].values())
    assert set(report.panels) == {"full_vs_empirical_quantile"}


def test_the_two_configs_are_scored_on_exactly_the_same_rows(report) -> None:  # type: ignore[no-untyped-def]
    for t in TICKERS:
        a, b = report.results["full"][t], report.results["price_only"][t]
        assert a.index.equals(b.index) and np.array_equal(a.y, b.y)


def test_the_thesis_comparison_is_reported_with_multiple_testing_control(report) -> None:  # type: ignore[no-untyped-def]
    t = report.thesis
    assert set(t["per_ticker"]) == set(TICKERS)
    for stats in t["per_ticker"].values():
        assert 0.0 <= stats["q_value"] <= 1.0 and 0.0 <= stats["p_one_sided"] <= 1.0
    assert {"mean_relative_diff", "dm_p_two_sided", "n_dates", "n_tickers"} <= set(t["panel"])


def test_null_data_fails_the_gate_for_the_right_reasons(report) -> None:  # type: ignore[no-untyped-def]
    g = report.gate
    assert not g.passed
    assert set(g.details["q_values"]) == set(
        TICKERS
    )  # nobody was excluded; every ticker was judged
    assert g.details["excluded_tickers"] == [] and np.isfinite(g.details["mean_coverage"])
    assert g.details["passing_tickers"] == []
    assert any(
        "positive edge at FDR" in r for r in g.reasons
    )  # the per-ticker criterion really fired


def test_every_ticker_gets_all_others_as_peers() -> None:
    spy: list[tuple] = []
    run_live_eval(cfg(configs=("full",)), fake_sources(spy=spy), progress=lambda m: None)
    assert sorted(spy) == sorted(
        [("AAA", "BBB", "CCC"), ("BBB", "AAA", "CCC"), ("CCC", "AAA", "BBB")]
    )


def test_a_run_where_no_ticker_has_enough_rows_fails_loudly_instead_of_gating_nothing() -> None:
    with pytest.raises(RuntimeError, match="no ticker has >= 1000000000 scored rows"):
        run_live_eval(
            cfg(configs=("full",)),
            fake_sources(),
            rule=GateRule(min_predictions=10**9),
            progress=lambda m: None,
        )


def test_the_panel_uses_only_the_tickers_the_gate_keeps(monkeypatch: pytest.MonkeyPatch) -> None:
    # A rule that keeps tickers with >= 500 rows; shorten one ticker's history so it is excluded.
    sources = fake_sources()
    real = sources.prices
    sources.prices = lambda t, s, e, i: (
        real(t, s, e, i).iloc[300:] if t == "CCC" else real(t, s, e, i)
    )
    rep = run_live_eval(cfg(configs=("full",)), sources, progress=lambda m: None)
    assert rep.gate.details["excluded_tickers"] == ["CCC"]
    assert rep.panels["full_vs_empirical_quantile"].n_tickers == 2  # CCC is in neither place


def test_json_is_saved_with_undefined_numbers_as_null(report, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "out" / "r.json"
    save_report(report, path, "2026-01-05")
    doc = json.loads(path.read_text())
    assert doc["price_snapshot"] == "2026-01-05" and doc["gate"]["passed"] is False
    assert set(doc["summaries"]["full"]) == set(TICKERS) and "thesis_full_vs_price_only" in doc
    assert "NaN" not in path.read_text() and "Infinity" not in path.read_text()


def test_the_markdown_lists_every_ticker_the_gate_and_the_thesis_test(report) -> None:  # type: ignore[no-untyped-def]
    md = render_report_markdown(report)
    assert "Gate: **FAIL**" in md and all(f"| {t} |" in md for t in TICKERS)
    assert "Panel (average across tickers per date)" in md and "full vs price-only" in md


def test_comparing_models_on_different_rows_is_refused() -> None:
    idx = IDX[:50]
    a = EvalResult("5d", 5, 1, idx, np.zeros(50), {"model": np.zeros((50, 3))})
    b = EvalResult("5d", 5, 1, idx[1:], np.zeros(49), {"model": np.zeros((49, 3))})
    with pytest.raises(ValueError, match="different rows"):
        with_baseline(a, "other", b)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"universe": ("AAA",)}, "at least two tickers"),
        ({"configs": ("full", "bogus")}, "unknown ablation configs"),
        ({"configs": ("price_only",)}, "must include 'full'"),
    ],
)
def test_config_is_validated(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        cfg(**kwargs)


# ---- CLI -----------------------------------------------------------------------------------------


@pytest.fixture
def live_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    monkeypatch.chdir(tmp_path)  # the price cache lives under ./data/cache/prices
    monkeypatch.setattr("src.prediction.dataset.default_sources", lambda: fake_sources())
    uni = tmp_path / "uni.txt"
    uni.write_text("# test universe\n" + "\n".join(TICKERS) + "\n")
    base = [
        "--live",
        "--universe",
        str(uni),
        "--start",
        "2021-01-01",
        "--horizon",
        "5d",
        "--backend",
        "linear",
        "--no-news",
        "--train-window",
        "150",
        "--test-window",
        "20",
        "--n-boot",
        "40",
    ]
    return base, tmp_path


def test_the_live_cli_writes_a_report_and_a_failed_gate_still_exits_zero(live_cli, capsys) -> None:  # type: ignore[no-untyped-def]
    base, tmp = live_cli
    out = tmp / "r.json"
    assert main([*base, "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "Gate: **FAIL**" in text and "price snapshot" in text
    assert "positive edge at FDR" in text  # a real reason is printed, not just a verdict
    assert json.loads(out.read_text())["gate"]["passed"] is False


def test_require_pass_turns_a_failed_gate_into_exit_one(live_cli) -> None:  # type: ignore[no-untyped-def]
    base, tmp = live_cli
    assert main([*base, "--out", str(tmp / "r.json"), "--require-pass"]) == 1


def test_a_saved_snapshot_reproduces_the_run_exactly(live_cli, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
    base, tmp = live_cli
    first = tmp / "a.json"
    main([*base, "--out", str(first)])
    day = json.loads(first.read_text())["price_snapshot"]
    # The live source now returns different data; replaying the snapshot must ignore it.
    monkeypatch.setattr("src.prediction.dataset.default_sources", lambda: fake_sources(seed=99))
    second = tmp / "b.json"
    main([*base, "--out", str(second), "--snapshot", day])
    a, b = json.loads(first.read_text()), json.loads(second.read_text())
    assert a["summaries"] == b["summaries"] and a["gate"] == b["gate"]


def test_a_missing_snapshot_fails_loudly(live_cli) -> None:  # type: ignore[no-untyped-def]
    base, tmp = live_cli
    with pytest.raises(FileNotFoundError, match="no price snapshot"):
        main([*base, "--out", str(tmp / "r.json"), "--snapshot", date(2020, 1, 1).isoformat()])
