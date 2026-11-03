import numpy as np
import pandas as pd
import pytest

from src.eval import h2, protocol
from src.eval.h2 import Panel, run_dev, run_holdout, truncate
from src.eval.synthetic import make_synthetic

TICKERS = 5


@pytest.fixture(scope="module")
def panel() -> Panel:
    data = [make_synthetic(1300, 3.0, seed=s) for s in range(TICKERS)]
    return Panel(
        {f"T{i}": d.features for i, d in enumerate(data)},
        {f"T{i}": d.log_returns for i, d in enumerate(data)},
        data[0].vix,
    )


def test_truncate_removes_everything_after_the_cut(panel: Panel) -> None:
    cut = pd.Timestamp("2021-06-30", tz="UTC")
    p = truncate(panel, "2021-06-30")
    assert all(f.index.max() <= cut and len(f) for f in p.features.values())
    assert all(r.index.max() <= cut for r in p.log_returns.values())
    assert p.vix.index.max() <= cut
    assert panel.features["T0"].index.max() > cut  # the input is not modified


def test_dev_never_scores_or_trains_on_data_after_the_dev_end(
    panel: Panel, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(protocol, "DEV_END", "2022-06-30")
    seen: list[pd.Timestamp] = []
    real = h2.pooled_walk_forward

    def spy(features: dict, log_returns: dict, *a: object, **kw: object) -> dict:
        seen.extend(f.index.max() for f in features.values())
        seen.extend(r.index.max() for r in log_returns.values())
        return real(features, log_returns, *a, **kw)  # type: ignore[arg-type]

    monkeypatch.setattr(h2, "pooled_walk_forward", spy)
    res = run_dev(panel, ("fhs-pooled",), n_boot=10)
    assert max(seen) <= pd.Timestamp("2022-06-30", tz="UTC")
    for r in res["fhs-pooled"].values():
        assert r.index.max() <= pd.Timestamp("2022-06-30", tz="UTC")


def test_holdout_scores_only_rows_from_the_holdout_start_and_reports_both_hypotheses(
    panel: Panel,
) -> None:
    frozen = {
        "overall": "lq-a0.01-news",
        "best_price": "lq-a0.01-price",
        "best_news": "lq-a0.01-news",
    }
    start = "2022-01-03"
    rep = run_holdout(panel, frozen, holdout_start=start, n_boot=10)
    assert set(rep["summaries"]) == {"lq-a0.01-news", "lq-a0.01-price"}  # type: ignore[arg-type]
    for ticker_summaries in rep["summaries"].values():  # type: ignore[attr-defined]
        for s in ticker_summaries.values():
            assert s["n_predictions"] > 0
    assert set(rep) >= {"h2a", "h2b", "frozen", "summaries"}
    # too few rows per ticker for the real 500-row rule: the gate must say so, not crash or pass
    assert rep["h2a"]["passed"] is False  # type: ignore[index]
    assert "only" in rep["h2a"]["reasons"][0]  # type: ignore[index]


def test_cli_holdout_claims_the_shot_before_evaluating_and_cannot_run_twice(
    tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch, panel: Panel
) -> None:
    out = tmp_path / "h2"  # type: ignore[operator]
    frozen = {"overall": "a", "best_price": "a", "best_news": "a"}
    monkeypatch.setattr(
        h2.protocol,
        "verify_freeze",
        lambda path, repo: {
            "git_commit": "abc",
            "frozen": frozen,
            "context": {"universe": ["A", "B"]},
        },
    )
    monkeypatch.setattr(h2, "_panel_from_args", lambda args: (panel, "2026-11-03"))
    monkeypatch.setattr(h2.protocol, "require_unclaimed", lambda marker, repo: None)
    monkeypatch.setattr(h2, "_universe", lambda path: ("A", "B"))

    def fake(p: Panel, f: dict) -> dict:
        assert (out / "holdout_run.json").exists()  # claimed before any evaluation
        return {"h2a": {"passed": False}, "h2b": {"passed": False}}

    monkeypatch.setattr(h2, "run_holdout", fake)
    args = ["holdout", "--snapshot", "2026-11-03", "--out-dir", str(out)]
    assert h2.main(args) == 0
    assert (out / "holdout.json").exists()
    with pytest.raises(protocol.ProtocolError, match="already run or started"):
        h2.main(args)


def test_cli_holdout_load_failure_does_not_spend_the_shot(
    tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "h2"  # type: ignore[operator]
    monkeypatch.setattr(
        h2.protocol,
        "verify_freeze",
        lambda path, repo: {"git_commit": "abc", "frozen": {}, "context": {"universe": ["A"]}},
    )
    monkeypatch.setattr(h2.protocol, "require_unclaimed", lambda marker, repo: None)
    monkeypatch.setattr(h2, "_universe", lambda path: ("A",))

    def boom(args: object) -> None:
        raise RuntimeError("no network")

    monkeypatch.setattr(h2, "_panel_from_args", boom)
    with pytest.raises(RuntimeError, match="no network"):
        h2.main(["holdout", "--snapshot", "2026-11-03", "--out-dir", str(out)])
    assert not (out / "holdout_run.json").exists()


def test_cli_holdout_needs_an_explicit_snapshot_and_the_frozen_universe(
    tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "h2"  # type: ignore[operator]
    monkeypatch.setattr(h2.protocol, "require_unclaimed", lambda marker, repo: None)
    monkeypatch.setattr(
        h2.protocol, "verify_freeze", lambda path, repo: {"context": {"universe": ["A", "B"]}}
    )
    with pytest.raises(SystemExit):  # argparse error: no --snapshot
        h2.main(["holdout", "--out-dir", str(out)])
    monkeypatch.setattr(h2, "_universe", lambda path: ("A", "C"))
    with pytest.raises(protocol.ProtocolError, match="universe differs"):
        h2.main(["holdout", "--snapshot", "2026-11-03", "--out-dir", str(out)])
    assert not (out / "holdout_run.json").exists()


def test_cli_dev_saves_the_selection_and_freeze_reads_it_back(
    tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch, panel: Panel
) -> None:
    from src.eval.h2_select import Selection

    out = tmp_path / "h2"  # type: ignore[operator]
    sel = Selection({"c": {"fhs": -0.02}}, "p", "n", "n", True, [])
    monkeypatch.setattr(h2, "_universe", lambda path: ("A", "B"))
    monkeypatch.setattr(h2.protocol, "code_hash", lambda repo: "h")
    monkeypatch.setattr(h2, "_panel_from_args", lambda args: (panel, "2026-11-03"))
    monkeypatch.setattr(h2, "run_dev", lambda p: {})
    monkeypatch.setattr(h2, "select", lambda r: sel)
    assert h2.main(["dev", "--out-dir", str(out)]) == 0
    seen: dict = {}

    def fake_freeze(s: Selection, repo: object, path: object, context: dict) -> dict:
        seen["context"] = context
        seen["sel"] = s
        return {"frozen": "x", "git_commit": "abc"}

    monkeypatch.setattr(h2.protocol, "freeze", fake_freeze)
    assert h2.main(["freeze", "--out-dir", str(out)]) == 0
    assert seen["sel"] == sel
    assert seen["context"] == {"universe": ["A", "B"], "dev_snapshot": "2026-11-03"}

    monkeypatch.setattr(h2.protocol, "code_hash", lambda repo: "changed")
    with pytest.raises(protocol.ProtocolError, match="rerun dev"):
        h2.main(["freeze", "--out-dir", str(out)])
