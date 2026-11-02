import math
from datetime import date
from pathlib import Path

import pytest

from src.trading.broker import DuplicateOrderError, FakeBroker, OrderRejected
from src.trading.ledger import Ledger
from src.trading.rebalance import Order
from src.trading.risk import AccountSnapshot, PriceBar

D = date(2026, 1, 9)


def test_ledger_round_trip_and_queries(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "t", "paper")
    assert ledger.records() == [] and ledger.high_water_mark() is None
    for equity in (100.0, 120.0, 90.0):
        ledger.append("equity", {"as_of": D, "equity": equity})
    ledger.append("order_intent", {"client_order_id": "x", "as_of": D})
    assert ledger.high_water_mark() == 120.0
    assert ledger.attempted_client_ids() == {"x"}
    assert ledger.orders_attempted_on(D) == 1 and ledger.orders_attempted_on(date(2026, 1, 10)) == 0
    assert [r["kind"] for r in ledger.records("equity")] == ["equity"] * 3


def test_dry_run_equity_never_feeds_the_high_water_mark(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path, "paper")
    ledger.append("equity", {"equity": 100.0, "dry_run": False})
    ledger.append("equity", {"equity": 9_999.0, "dry_run": True})
    assert ledger.high_water_mark() == 100.0


def test_each_mode_has_its_own_file(tmp_path: Path) -> None:
    Ledger(tmp_path, "paper").append("equity", {"equity": 500.0})
    assert Ledger(tmp_path, "live").high_water_mark() is None
    assert Ledger(tmp_path, "paper").path.name == "ledger-paper.jsonl"


def test_ledger_is_append_only_across_instances(tmp_path: Path) -> None:
    Ledger(tmp_path, "paper").append("a", {})
    Ledger(tmp_path, "paper").append("b", {})
    assert [r["kind"] for r in Ledger(tmp_path, "paper").records()] == ["a", "b"]


def test_corrupt_ledger_line_raises(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path, "paper")
    ledger.append("a", {})
    with ledger.path.open("a") as fh:
        fh.write("{not json\n")
    with pytest.raises(ValueError, match="line 2"):
        ledger.records()


def test_torn_last_line_blocks_further_appends(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path, "paper")
    ledger.append("a", {})
    with ledger.path.open("a") as fh:
        fh.write('{"half": ')  # a crash mid-write: no trailing newline
    with pytest.raises(ValueError, match="torn line"):
        ledger.append("b", {})


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_values_are_refused(tmp_path: Path, bad: float) -> None:
    ledger = Ledger(tmp_path, "paper")
    with pytest.raises(ValueError):
        ledger.append("equity", {"equity": bad})
    assert ledger.records() == []


@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_non_finite_account_and_price_values_are_refused(bad: float) -> None:
    with pytest.raises(ValueError, match="not finite"):
        AccountSnapshot(equity=bad, cash=1.0, last_equity=1.0)
    with pytest.raises(ValueError, match="not finite"):
        AccountSnapshot(equity=1.0, cash=1.0, last_equity=bad)
    with pytest.raises(ValueError, match="finite and positive"):
        PriceBar(bad, D)


def test_zero_and_negative_prices_are_refused() -> None:
    for bad in (0.0, -3.0):
        with pytest.raises(ValueError):
            PriceBar(bad, D)


def fake() -> FakeBroker:
    return FakeBroker(
        cash=10_000.0, bars={"AAA": PriceBar(100.0, D)}, today=D, positions={"AAA": 10}
    )


def test_fake_broker_fills_and_tracks_cash() -> None:
    b = fake()
    b.submit_order(Order("AAA", "buy", 5, "i1"))
    b.submit_order(Order("AAA", "sell", 15, "i2"))
    assert b.get_positions() == {} and b.cash == 10_000.0 + 1_000.0
    assert b.get_account().equity == b.cash and b.today() == D


def test_fake_broker_rejects_bad_orders() -> None:
    b = fake()
    b.submit_order(Order("AAA", "buy", 1, "i1"))
    with pytest.raises(DuplicateOrderError):
        b.submit_order(Order("AAA", "buy", 1, "i1"))
    with pytest.raises(OrderRejected, match="cannot sell"):
        b.submit_order(Order("AAA", "sell", 99, "i2"))
    with pytest.raises(OrderRejected, match="insufficient cash"):
        b.submit_order(Order("AAA", "buy", 1000, "i3"))
