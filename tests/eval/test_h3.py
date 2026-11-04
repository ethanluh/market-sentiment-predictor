import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.eval import h3, protocol, voltarget
from tests.eval.test_voltarget import panel as make_panel

DRIFT = 0.0004  # about 10% a year: without drift the Sharpe difference is too noisy for H3b
N_BARS = 1500  # 2019-09-02 to mid 2025: a dev period and a holdout


@pytest.fixture(scope="module")
def clustered() -> pd.DataFrame:
    return make_panel(0.98, n=N_BARS, drift=DRIFT)


@pytest.fixture(scope="module")
def dev_results(clustered: pd.DataFrame) -> dict[str, dict[str, object]]:
    return h3.run_dev(clustered)


def test_truncate_removes_later_bars_by_utc_date(clustered: pd.DataFrame) -> None:
    cut = h3.truncate(clustered, "2021-06-30")
    assert cut.index.max() <= pd.Timestamp("2021-06-30", tz="UTC")
    assert clustered.index.max() > pd.Timestamp("2021-06-30", tz="UTC")


def test_dev_results_do_not_depend_on_anything_after_the_dev_end(
    clustered: pd.DataFrame, dev_results: dict[str, dict[str, object]]
) -> None:
    changed = clustered.copy()
    after = changed.index > pd.Timestamp(voltarget.DEV_END, tz="UTC")
    changed.loc[after] = changed.loc[after] * 50.0  # wild later data
    again = h3.run_dev(changed)
    for name in voltarget.FORECASTERS:
        for key in ("vv_ratio", "sharpe_diff", "target", "ratio_hi", "n_bars"):
            assert again[name][key] == dev_results[name][key], (name, key)


def test_dev_reports_every_forecaster_with_costs_and_per_ticker_detail(
    dev_results: dict[str, dict[str, object]], clustered: pd.DataFrame
) -> None:
    assert set(dev_results) == set(voltarget.FORECASTERS)
    r = dev_results["ewma94"]
    assert r["target"] > 0 and set(r["cost_sensitivity"]) == {"0.0", "20.0"}  # type: ignore[arg-type]
    assert set(r["per_ticker"]) == set(clustered.columns)  # type: ignore[arg-type]
    cs = r["cost_sensitivity"]
    assert cs["0.0"]["sharpe"] >= cs["20.0"]["sharpe"]  # type: ignore[index]
    sel = voltarget.select(dev_results)
    assert sel.winner in voltarget.FORECASTERS


def test_a_planted_volatility_effect_proceeds_and_an_iid_panel_does_not(
    dev_results: dict[str, dict[str, object]],
) -> None:
    assert voltarget.select(dev_results).proceed
    null = h3.run_dev(make_panel(0.0, n=N_BARS, drift=DRIFT))
    assert not voltarget.select(null).proceed


def test_holdout_scores_only_bars_from_the_holdout_start(
    clustered: pd.DataFrame, dev_results: dict[str, dict[str, object]]
) -> None:
    sel = voltarget.select(dev_results)
    rep = h3.run_holdout(clustered, h3.frozen_from(sel, dev_results))
    n_hold = int((clustered.index >= pd.Timestamp(voltarget.HOLDOUT_START, tz="UTC")).sum())
    assert 0 < rep["result"]["n_bars"] <= n_hold  # type: ignore[index]
    assert rep["forecaster"] == sel.winner and isinstance(rep["passed"], bool)
    if sel.winner != "rolling63":
        assert rep["h3c"]["applicable"] is True  # type: ignore[index]


def test_h3c_is_not_applicable_when_rolling63_is_the_frozen_forecaster(
    clustered: pd.DataFrame, dev_results: dict[str, dict[str, object]]
) -> None:
    frozen = {
        "forecaster": "rolling63",
        "targets": {n: dev_results[n]["target"] for n in dev_results},
    }
    assert h3.run_holdout(clustered, frozen)["h3c"] == {"applicable": False}


def test_the_holdout_uses_the_frozen_target_not_one_recalibrated_on_the_holdout(
    clustered: pd.DataFrame, dev_results: dict[str, dict[str, object]]
) -> None:
    frozen = h3.frozen_from(voltarget.select(dev_results), dev_results)
    rep = h3.run_holdout(clustered, frozen)
    assert rep["target"] == frozen["targets"][frozen["forecaster"]]  # type: ignore[index]
    assert rep["result"]["target"] == rep["target"]  # type: ignore[index]


def git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), "-c", "core.hooksPath=/dev/null", "-c", "user.name=t",
         "-c", "user.email=t@t", *args],
        check=True, capture_output=True,
    )  # fmt: skip


def test_freeze_refuses_a_closed_selection_and_writes_the_h3_document(
    tmp_path: Path, dev_results: dict[str, dict[str, object]]
) -> None:
    git(tmp_path, "init", "-q", "-b", "main")
    for name in h3.H3.extra_frozen:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text("x\n")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 1\n")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-q", "-m", "init")
    closed = voltarget.Selection("ewma94", False, ["ewma94 fails h3a"])
    with pytest.raises(protocol.ProtocolError, match="stays closed"):
        h3.freeze(closed, dev_results, tmp_path, tmp_path / "out" / "freeze.json")
    sel = voltarget.Selection("ewma97", True, [])
    doc = h3.freeze(
        sel, dev_results, tmp_path, tmp_path / "out" / "freeze.json", {"universe": ["A"]}
    )
    assert doc["hypothesis"] == "H3" and doc["frozen"]["forecaster"] == "ewma97"  # type: ignore[index]
    assert set(doc["frozen"]["targets"]) == set(voltarget.FORECASTERS)  # type: ignore[index,arg-type]
    assert set(doc["dev_scores"]) == set(voltarget.FORECASTERS)  # type: ignore[arg-type]
    json.dumps(doc["dev_scores"])  # the scores are plain JSON values


def test_the_fingerprint_names_h3_and_pins_the_protocol_constants() -> None:
    fp = h3.fingerprint()
    assert fp["hypothesis"] == h3.H3.hypothesis == "H3"
    assert fp["block"] == 21 and fp["n_boot"] == 2000 and fp["max_vv_ratio"] == 0.80


def test_cli_holdout_claims_the_shot_before_evaluating_and_cannot_run_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clustered: pd.DataFrame
) -> None:
    out = tmp_path / "h3"
    frozen = {"forecaster": "ewma94", "targets": {"ewma94": 0.2}}
    doc = {
        "git_commit": "abc",
        "frozen": frozen,
        "context": {"universe": ["A", "B"], "dev_snapshot": "2026-11-02"},
    }
    monkeypatch.setattr(h3.protocol, "verify_freeze", lambda path, repo, proto: doc)
    monkeypatch.setattr(h3, "_returns_from_args", lambda args: (clustered, "2026-11-02"))
    monkeypatch.setattr(h3, "_universe", lambda path: ("A", "B"))

    def fake(returns: pd.DataFrame, f: dict) -> dict:
        assert (out / "holdout_run.json").exists()  # claimed before any evaluation
        return {"passed": False, "h3c": {"applicable": False}}

    monkeypatch.setattr(h3, "run_holdout", fake)
    args = ["holdout", "--snapshot", "2026-11-02", "--out-dir", str(out)]
    monkeypatch.setattr(h3.protocol, "require_unclaimed", lambda marker, repo, proto: None)
    assert h3.main(args) == 0
    assert json.loads((out / "holdout_run.json").read_text())["hypothesis"] == "H3"
    assert (out / "holdout.json").exists()
    with pytest.raises(protocol.ProtocolError, match="already run or started"):
        h3.main(args)


def test_cli_holdout_load_failure_does_not_spend_the_shot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "h3"
    doc = {
        "git_commit": "abc",
        "frozen": {},
        "context": {"universe": ["A"], "dev_snapshot": "2026-11-02"},
    }
    monkeypatch.setattr(h3.protocol, "verify_freeze", lambda path, repo, proto: doc)
    monkeypatch.setattr(h3.protocol, "require_unclaimed", lambda marker, repo, proto: None)
    monkeypatch.setattr(h3, "_universe", lambda path: ("A",))

    def boom(args: object) -> None:
        raise RuntimeError("no network")

    monkeypatch.setattr(h3, "_returns_from_args", boom)
    with pytest.raises(RuntimeError, match="no network"):
        h3.main(["holdout", "--snapshot", "2026-11-02", "--out-dir", str(out)])
    assert not (out / "holdout_run.json").exists()


def test_cli_holdout_needs_an_explicit_snapshot_and_the_frozen_universe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "h3"
    monkeypatch.setattr(h3.protocol, "require_unclaimed", lambda marker, repo, proto: None)
    monkeypatch.setattr(
        h3.protocol,
        "verify_freeze",
        lambda path, repo, proto: {
            "context": {"universe": ["A", "B"], "dev_snapshot": "2026-11-02"}
        },
    )
    with pytest.raises(SystemExit):  # argparse error: no --snapshot
        h3.main(["holdout", "--out-dir", str(out)])
    monkeypatch.setattr(h3, "_universe", lambda path: ("A", "C"))
    with pytest.raises(protocol.ProtocolError, match="universe differs"):
        h3.main(["holdout", "--snapshot", "2026-11-02", "--out-dir", str(out)])
    assert not (out / "holdout_run.json").exists()


def test_cli_holdout_refuses_an_h2_freeze(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "h3"
    out.mkdir()
    (out / "freeze.json").write_text(json.dumps({"hypothesis": "H2", "fingerprint": {}}))
    monkeypatch.setattr(h3.protocol, "require_unclaimed", lambda marker, repo, proto: None)
    with pytest.raises(protocol.ProtocolError, match="freeze for H2, not H3"):
        h3.main(["holdout", "--snapshot", "2026-11-02", "--out-dir", str(out)])


def test_cli_dev_saves_the_selection_and_freeze_reads_it_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clustered: pd.DataFrame
) -> None:
    out = tmp_path / "h3"
    sel = voltarget.Selection("ewma94", True, [])
    results = {n: {"target": 0.2} for n in voltarget.FORECASTERS}
    monkeypatch.setattr(h3, "_universe", lambda path: ("A", "B"))
    monkeypatch.setattr(h3.protocol, "code_hash", lambda repo, proto: "h")
    monkeypatch.setattr(h3, "_returns_from_args", lambda args: (clustered, "2026-11-02"))
    monkeypatch.setattr(h3, "run_dev", lambda r: results)
    monkeypatch.setattr(h3.voltarget, "select", lambda r: sel)
    assert h3.main(["dev", "--out-dir", str(out)]) == 0
    seen: dict = {}

    def fake_freeze(
        s: voltarget.Selection, res: dict, repo: object, path: object, context: dict
    ) -> dict:
        seen.update(sel=s, results=res, context=context)
        return {"frozen": "x", "git_commit": "abc"}

    monkeypatch.setattr(h3, "freeze", fake_freeze)
    assert h3.main(["freeze", "--out-dir", str(out)]) == 0
    assert seen["sel"] == sel and seen["results"] == results
    assert seen["context"] == {"universe": ["A", "B"], "dev_snapshot": "2026-11-02"}
    monkeypatch.setattr(h3.protocol, "code_hash", lambda repo, proto: "changed")
    with pytest.raises(protocol.ProtocolError, match="rerun dev"):
        h3.main(["freeze", "--out-dir", str(out)])


def test_load_returns_builds_a_frame_with_one_column_per_ticker() -> None:
    idx = pd.bdate_range("2020-01-01", periods=5, tz="UTC")

    class Src:
        def prices(self, ticker: str, start: str, end: object, interval: str) -> pd.DataFrame:
            assert (start, end, interval) == (h3.START, None, "1d")
            return pd.DataFrame({"log_return": np.arange(5.0) * (1 + len(ticker))}, index=idx)

    frame = h3.load_returns(("AA", "BBB"), Src())
    assert list(frame.columns) == ["AA", "BBB"] and len(frame) == 5


def test_cli_holdout_refuses_a_different_snapshot_than_development(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "h3"
    doc = {"context": {"universe": ["A"], "dev_snapshot": "2026-11-02"}}
    monkeypatch.setattr(h3.protocol, "require_unclaimed", lambda marker, repo, proto: None)
    monkeypatch.setattr(h3.protocol, "verify_freeze", lambda path, repo, proto: doc)
    monkeypatch.setattr(h3, "_universe", lambda path: ("A",))
    with pytest.raises(protocol.ProtocolError, match="differs from the development snapshot"):
        h3.main(["holdout", "--snapshot", "2026-12-01", "--out-dir", str(out)])
    assert not (out / "holdout_run.json").exists()


def test_a_gappy_late_or_delisted_name_is_reported_skipped_and_never_stops_the_headline(
    clustered: pd.DataFrame, dev_results: dict[str, dict[str, object]]
) -> None:
    lr = clustered.copy()
    lr.iloc[1400:, 0] = np.nan  # T0 stops trading inside the holdout
    lr.iloc[1450, 1] = np.nan  # T1 misses one bar
    lr.iloc[:1450, 2] = np.nan  # T2 lists inside the holdout
    frozen = h3.frozen_from(voltarget.select(dev_results), dev_results)
    rep = h3.run_holdout(lr, frozen)
    per = rep["result"]["per_ticker"]  # type: ignore[index]
    assert "skipped" in per["T0"] and "missing bars" in per["T1"]["skipped"]
    assert "skipped" in per["T2"] and "vv_ratio" in per["T3"]
    assert isinstance(rep["passed"], bool)  # the headline survived


def test_the_holdout_uses_each_forecasters_own_frozen_target(
    clustered: pd.DataFrame, dev_results: dict[str, dict[str, object]]
) -> None:
    frozen = {"forecaster": "ewma94", "targets": {n: dev_results[n]["target"] for n in dev_results}}
    base = h3.run_holdout(clustered, frozen)
    own = {**frozen, "targets": {**frozen["targets"], "ewma94": frozen["targets"]["ewma94"] * 0.5}}  # type: ignore[dict-item]
    assert h3.run_holdout(clustered, own)["result"]["mean_exposure"] < base["result"]["mean_exposure"]  # type: ignore[index]
    other = {**frozen, "targets": {**frozen["targets"], "rolling63": frozen["targets"]["rolling63"] * 0.5}}  # type: ignore[dict-item]
    rep = h3.run_holdout(clustered, other)
    assert (
        rep["result"] == base["result"]
    )  # the frozen forecaster is untouched by rolling63's target
    assert rep["h3c"]["vv_ratio_vs_rolling63"] != base["h3c"]["vv_ratio_vs_rolling63"]  # type: ignore[index]
    assert rep["h3c"]["lower"] == (rep["h3c"]["vv_ratio_vs_rolling63"] < 1.0)  # type: ignore[index]
