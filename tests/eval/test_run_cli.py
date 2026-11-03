import json
from pathlib import Path

import pytest

from src.eval.run import main

FAST = ["--n-days", "400", "--train-window", "100", "--test-window", "20"]
FAST += ["--backend", "linear", "--n-boot", "50", "--signal", "1.5"]


def test_synthetic_run_writes_summary(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "res.json"
    assert main(["--synthetic", *FAST, "--out", str(out)]) == 0
    assert "| model |" in capsys.readouterr().out
    assert set(json.loads(out.read_text())) == {"full"}


def test_ablation_runs_every_config(tmp_path: Path) -> None:
    out = tmp_path / "ab.json"
    main(["--synthetic", *FAST, "--ablation", "--out", str(out)])
    assert {"price_only", "full", "price+sentiment"} <= set(json.loads(out.read_text()))


def test_a_mode_is_required_and_the_modes_are_exclusive() -> None:
    for argv in ([], ["--synthetic", "--live"]):
        with pytest.raises(SystemExit) as e:
            main(argv)
        assert e.value.code == 2


def test_regime_ablation_actually_changes_the_model(tmp_path: Path) -> None:
    out = tmp_path / "ab.json"
    main(["--synthetic", *FAST, "--ablation", "--out", str(out)])
    res = json.loads(out.read_text())

    def loss(cfg: str) -> float:
        return res[cfg]["forecasters"]["model"]["mean_pinball"]

    # price+regime keeps only regime; it must differ from price_only (which drops it).
    assert loss("price+regime") != loss("price_only")
    # full minus nothing differs from price+sentiment, i.e. each group is really toggled.
    assert loss("full") != loss("price+sentiment")
