import json
from datetime import date
from pathlib import Path

import pytest

from src.trading.broker import FakeBroker
from src.trading.forecasts import read_forecast_file, read_universe
from src.trading.risk import PriceBar
from src.trading.run import main

TODAY = date(2026, 1, 9)
UNIVERSE = ("AAA", "BBB")


@pytest.fixture
def files(tmp_path: Path) -> dict[str, Path]:
    uni = tmp_path / "uni.txt"
    uni.write_text("# comment\nAAA\n\n bbb \n")
    fc = tmp_path / "fc.json"
    fc.write_text(
        json.dumps(
            {
                "as_of": "2026-01-09",
                "horizon": "5d",
                "forecasts": {
                    "AAA": {"p10": -0.01, "p50": 0.02, "p90": 0.05},
                    "BBB": {"p10": -0.01, "p50": 0.0, "p90": 0.02},
                },
            }
        )
    )
    return {"uni": uni, "fc": fc, "ledger": tmp_path / "ledger"}


def broker(market_open: bool = False) -> FakeBroker:
    bars = {t: PriceBar(100.0, TODAY) for t in UNIVERSE}
    return FakeBroker(cash=100_000.0, bars=bars, today=TODAY, market_open=market_open)


def run(files, b, *extra):  # type: ignore[no-untyped-def]
    argv = [
        "--forecasts",
        str(files["fc"]),
        "--universe",
        str(files["uni"]),
        "--ledger-dir",
        str(files["ledger"]),
        *extra,
    ]
    return main(argv, broker_factory=lambda: b)


def test_default_is_a_dry_run(files, capsys) -> None:  # type: ignore[no-untyped-def]
    b = broker()
    assert run(files, b) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["dry_run"] is True and out["submitted"] == []
    assert [o["ticker"] for o in out["orders"]] == ["AAA"] and b.submitted == []


def test_submit_places_the_orders(files, capsys) -> None:  # type: ignore[no-untyped-def]
    b = broker()
    assert run(files, b, "--submit") == 0
    assert [o.ticker for o in b.submitted] == ["AAA"]
    assert json.loads(capsys.readouterr().out)["submitted"] == ["20260109-AAA-buy"]


def test_refuses_while_the_market_is_open(files, capsys) -> None:  # type: ignore[no-untyped-def]
    b = broker(market_open=True)
    assert run(files, b, "--submit") == 2
    assert "market is open" in capsys.readouterr().err and b.submitted == []


def test_a_risk_halt_exits_nonzero_without_orders(files, capsys, tmp_path) -> None:  # type: ignore[no-untyped-def]
    b = broker()
    kill = tmp_path / "KILL"
    kill.write_text("stop")
    assert run(files, b, "--submit", "--kill-switch", str(kill)) == 2
    assert "HALT" in capsys.readouterr().err and b.submitted == []


def test_a_mid_cycle_failure_reports_what_already_went_out(files, capsys) -> None:  # type: ignore[no-untyped-def]
    doc = json.loads(files["fc"].read_text())
    doc["forecasts"]["BBB"] = {"p10": -0.01, "p50": 0.02, "p90": 0.05}  # both names selected
    files["fc"].write_text(json.dumps(doc))

    class Flaky(FakeBroker):
        def submit_order(self, order):  # type: ignore[no-untyped-def]
            if order.ticker == "BBB":
                raise ConnectionError("reset by peer")
            return super().submit_order(order)

    bars = {t: PriceBar(100.0, TODAY) for t in UNIVERSE}
    with pytest.raises(ConnectionError):
        run(files, Flaky(cash=100_000.0, bars=bars, today=TODAY), "--submit")
    assert "20260109-AAA-buy" in capsys.readouterr().err


def test_stale_or_incomplete_forecast_files_are_refused(files) -> None:  # type: ignore[no-untyped-def]
    uni = read_universe(files["uni"])
    assert uni == UNIVERSE
    with pytest.raises(ValueError, match="market date"):
        read_forecast_file(files["fc"], uni, date(2026, 1, 12), "5d")
    with pytest.raises(ValueError, match="missing=\\['CCC'\\]"):
        read_forecast_file(files["fc"], (*uni, "CCC"), TODAY, "5d")
    with pytest.raises(ValueError, match="outside universe=\\['BBB'\\]"):
        read_forecast_file(files["fc"], ("AAA",), TODAY, "5d")


def test_empty_universe_file_raises(tmp_path: Path) -> None:
    f = tmp_path / "u.txt"
    f.write_text("# nothing\n")
    with pytest.raises(ValueError, match="no tickers"):
        read_universe(f)


def test_the_shipped_starter_universe_is_valid() -> None:
    uni = read_universe(Path("config/universe_starter.txt"))
    assert len(uni) == len(set(uni)) == 23 and "SPY" not in uni


def write(tmp_path: Path, mutate) -> Path:  # type: ignore[no-untyped-def]
    doc = {
        "as_of": "2026-01-09",
        "horizon": "5d",
        "forecasts": {"AAA": {"p10": -0.01, "p50": 0.02, "p90": 0.05}},
    }
    mutate(doc)
    f = tmp_path / "f.json"
    f.write_text(json.dumps(doc).replace('"__INF__"', "Infinity").replace('"__NAN__"', "NaN"))
    return f


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda d: d.update(horizon="1h"), "horizon"),
        (lambda d: d.pop("horizon"), "keys must be"),
        (lambda d: d.update(extra=1), "unexpected"),
        (lambda d: d["forecasts"]["AAA"].update(p50="__INF__", p90="__INF__"), "non-JSON constant"),
        (lambda d: d["forecasts"]["AAA"].update(p50="__NAN__"), "non-JSON constant"),
        (lambda d: d["forecasts"]["AAA"].update(p50=True), "JSON number"),
        (lambda d: d["forecasts"]["AAA"].update(p50="0.02"), "JSON number"),
        (lambda d: d["forecasts"]["AAA"].update(p50=1e9, p90=1e9), "within"),
    ],
)
def test_malformed_forecast_files_are_refused(tmp_path: Path, mutate, match) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError, match=match):
        read_forecast_file(write(tmp_path, mutate), ("AAA",), TODAY, "5d")


def test_a_well_formed_forecast_file_is_accepted(tmp_path: Path) -> None:
    got = read_forecast_file(write(tmp_path, lambda d: None), ("AAA",), TODAY, "5d")
    assert (got[0].ticker, got[0].p50) == ("AAA", 0.02)


def test_a_dry_run_ledger_records_are_all_flagged(files) -> None:  # type: ignore[no-untyped-def]
    from src.trading.ledger import Ledger

    run(files, broker())
    records = Ledger(files["ledger"], "paper").records()
    assert records and all(r["payload"]["dry_run"] is True for r in records)
    assert Ledger(files["ledger"], "paper").high_water_mark() is None
