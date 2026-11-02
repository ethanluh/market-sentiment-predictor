from pathlib import Path

import pytest

from src.trading.broker import FakeBroker
from src.trading.config import TradingConfig
from src.trading.cycle import run_cycle
from src.trading.ledger import Ledger
from src.trading.rebalance import Order
from src.trading.risk import RiskLimitBreach
from src.trading.signals import Forecast
from tests.trading.conftest import AS_OF

GOOD = [Forecast("AAA", -0.01, 0.01, 0.05), Forecast("BBB", -0.01, 0.01, 0.05)]


def setup(tmp_path: Path, bars, **kw):  # type: ignore[no-untyped-def]
    return FakeBroker(cash=100_000.0, bars=bars, today=AS_OF, **kw), Ledger(
        tmp_path / "ledger", "paper"
    )


def test_a_weekly_cycle_buys_the_selected_names(cfg: TradingConfig, bars, tmp_path) -> None:  # type: ignore[no-untyped-def]
    broker, ledger = setup(tmp_path, bars)
    res = run_cycle(broker, GOOD, ledger, cfg, AS_OF)
    assert res.targets == {"AAA": 0.05, "BBB": 0.05}
    assert broker.get_positions() == {"AAA": 50, "BBB": 100}
    assert res.submitted == ["20260109-AAA-buy", "20260109-BBB-buy"]
    kinds = [r["kind"] for r in ledger.records()]
    assert kinds == [
        "forecasts",
        "equity",
        "plan",
        "order_intent",
        "order_accepted",
        "order_intent",
        "order_accepted",
    ]


def test_rerunning_the_same_day_submits_nothing(cfg: TradingConfig, bars, tmp_path) -> None:  # type: ignore[no-untyped-def]
    broker, ledger = setup(tmp_path, bars)
    run_cycle(broker, GOOD, ledger, cfg, AS_OF)
    again = run_cycle(broker, GOOD, ledger, cfg, AS_OF)
    assert again.orders == [] and again.submitted == []
    assert len(broker.submitted) == 2


def test_orders_attempted_earlier_are_never_resent_silently(cfg: TradingConfig, bars, tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """A stale position read hides the earlier fills: the cycle must stop, not drop or resend."""
    broker, ledger = setup(tmp_path, bars)
    run_cycle(broker, GOOD, ledger, cfg, AS_OF)
    monkeypatch.setattr(broker, "get_positions", lambda: {})
    with pytest.raises(RiskLimitBreach, match="attempted earlier"):
        run_cycle(broker, GOOD, ledger, cfg, AS_OF)
    assert len(broker.submitted) == 2


def test_open_orders_block_planning(cfg: TradingConfig, bars, tmp_path) -> None:  # type: ignore[no-untyped-def]
    broker, ledger = setup(tmp_path, bars)
    broker.pending.append(Order("AAA", "buy", 1, "pending"))
    with pytest.raises(RiskLimitBreach, match="open orders"):
        run_cycle(broker, GOOD, ledger, cfg, AS_OF)
    assert broker.submitted == []


def test_dry_run_plans_and_logs_but_sends_nothing(cfg: TradingConfig, bars, tmp_path) -> None:  # type: ignore[no-untyped-def]
    broker, ledger = setup(tmp_path, bars)
    res = run_cycle(broker, GOOD, ledger, cfg, AS_OF, dry_run=True)
    assert len(res.orders) == 2 and res.submitted == [] and broker.submitted == []
    assert [r["kind"] for r in ledger.records()] == ["forecasts", "equity", "plan"]
    assert ledger.records("plan")[0]["payload"]["dry_run"] is True


def test_a_breach_aborts_before_any_order(cfg: TradingConfig, bars, tmp_path) -> None:  # type: ignore[no-untyped-def]
    broker, ledger = setup(tmp_path, bars)
    cfg.kill_switch_path.write_text("stop")
    with pytest.raises(RiskLimitBreach):
        run_cycle(broker, GOOD, ledger, cfg, AS_OF)
    assert broker.submitted == [] and ledger.records("order_intent") == []


def test_drawdown_from_ledger_high_water_mark_halts(cfg: TradingConfig, bars, tmp_path) -> None:  # type: ignore[no-untyped-def]
    broker, ledger = setup(tmp_path, bars)
    ledger.append("equity", {"as_of": AS_OF, "equity": 120_000.0})  # past peak; now 100k = -16.7%
    with pytest.raises(RiskLimitBreach, match="drawdown"):
        run_cycle(broker, GOOD, ledger, cfg, AS_OF)


def test_dropped_names_are_sold_and_cash_is_kept(cfg: TradingConfig, bars, tmp_path) -> None:  # type: ignore[no-untyped-def]
    broker, ledger = setup(tmp_path, bars, positions={"CCC": 100})
    broker.cash = 98_000.0
    run_cycle(broker, [Forecast("CCC", -0.05, -0.02, 0.01)], ledger, cfg, AS_OF)
    assert broker.get_positions() == {}
    assert broker.cash == pytest.approx(100_000.0)


def test_as_of_must_be_the_brokers_market_date(cfg: TradingConfig, bars, tmp_path) -> None:  # type: ignore[no-untyped-def]
    broker, ledger = setup(tmp_path, bars)
    with pytest.raises(RiskLimitBreach, match="market date"):
        run_cycle(broker, GOOD, ledger, cfg, AS_OF.replace(day=10))
    assert ledger.records() == [] and broker.submitted == []


def test_dry_run_equity_cannot_cause_a_false_drawdown_halt(cfg: TradingConfig, bars, tmp_path) -> None:  # type: ignore[no-untyped-def]
    ledger = Ledger(tmp_path / "ledger", "paper")
    rich = FakeBroker(cash=1_000_000.0, bars=bars, today=AS_OF)
    run_cycle(rich, GOOD, ledger, cfg, AS_OF, dry_run=True)
    real = FakeBroker(cash=100_000.0, bars=bars, today=AS_OF)
    assert len(run_cycle(real, GOOD, ledger, cfg, AS_OF).submitted) == 2


def test_a_failed_submission_is_recorded_and_blocks_a_blind_rerun(cfg: TradingConfig, bars, tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    broker, ledger = setup(tmp_path, bars)
    real_submit = broker.submit_order
    calls = []

    def flaky(order):  # type: ignore[no-untyped-def]
        calls.append(order)
        if len(calls) == 2:
            raise ConnectionError("boom")
        return real_submit(order)

    monkeypatch.setattr(broker, "submit_order", flaky)
    with pytest.raises(ConnectionError):
        run_cycle(broker, GOOD, ledger, cfg, AS_OF)
    kinds = [r["kind"] for r in ledger.records()]
    assert kinds[-3:] == ["order_accepted", "order_intent", "order_failed"]
    assert "boom" in ledger.records("order_failed")[0]["payload"]["error"]
    monkeypatch.setattr(broker, "submit_order", real_submit)
    with pytest.raises(RiskLimitBreach, match="attempted earlier"):
        run_cycle(broker, GOOD, ledger, cfg, AS_OF)


def test_stale_bar_for_a_held_but_untraded_position_halts(cfg: TradingConfig, bars, tmp_path) -> None:  # type: ignore[no-untyped-def]
    from src.trading.risk import PriceBar

    bars["CCC"] = PriceBar(20.0, AS_OF.replace(day=2))  # 7 days old, position will not be traded
    broker, ledger = setup(tmp_path, bars, positions={"CCC": 100})
    broker.cash = 98_000.0
    with pytest.raises(RiskLimitBreach, match="days old"):
        run_cycle(broker, [Forecast("CCC", 0.0, 0.01, 0.03)], ledger, cfg, AS_OF)


def test_prices_that_disagree_with_broker_equity_halt(cfg: TradingConfig, bars, tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from src.trading.risk import AccountSnapshot

    broker, ledger = setup(tmp_path, bars, positions={"AAA": 500})  # 500 * 100 = 50k at the bar
    broker.cash = 50_000.0
    monkeypatch.setattr(
        broker,
        "get_account",
        lambda: AccountSnapshot(equity=125_000.0, cash=50_000.0, last_equity=125_000.0),
    )  # the broker marks AAA near 150, the bar still says 100
    with pytest.raises(RiskLimitBreach, match="stale or positions are misread"):
        run_cycle(broker, [Forecast("AAA", 0.0, 0.01, 0.03)], ledger, cfg, AS_OF)
    assert broker.submitted == []


def test_non_finite_high_water_mark_cannot_disable_the_drawdown_check(cfg, bars) -> None:  # type: ignore[no-untyped-def]
    from src.trading.risk import AccountSnapshot, check_plan

    acct = AccountSnapshot(equity=40_000.0, cash=40_000.0, last_equity=40_000.0)
    with pytest.raises(RiskLimitBreach, match="not finite"):
        check_plan([], acct, {}, bars, float("nan"), 0, AS_OF, cfg)


def test_a_glitched_equity_reading_never_becomes_the_high_water_mark(cfg: TradingConfig, bars, tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from src.trading.risk import AccountSnapshot

    broker, ledger = setup(tmp_path, bars)
    good = broker.get_account
    monkeypatch.setattr(
        broker, "get_account", lambda: AccountSnapshot(equity=5e6, cash=100_000.0, last_equity=5e6)
    )
    with pytest.raises(RiskLimitBreach, match="stale or positions are misread"):
        run_cycle(broker, GOOD, ledger, cfg, AS_OF)
    assert ledger.high_water_mark() is None
    assert [r["kind"] for r in ledger.records()] == ["forecasts", "breach"]
    monkeypatch.setattr(broker, "get_account", good)
    assert len(run_cycle(broker, GOOD, ledger, cfg, AS_OF).submitted) == 2  # no false halt later


def test_a_full_gross_plan_passes_the_cash_check(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from src.trading.risk import PriceBar

    names = tuple(f"T{i:02d}" for i in range(20))
    c = TradingConfig(universe=names, kill_switch_path=tmp_path / "K")
    broker = FakeBroker(
        cash=100_000.0, bars={n: PriceBar(100.0, AS_OF) for n in names}, today=AS_OF
    )
    ledger = Ledger(tmp_path / "ledger", "paper")
    res = run_cycle(broker, [Forecast(n, -0.01, 0.01, 0.05) for n in names], ledger, c, AS_OF)
    assert len(res.submitted) == 20
    assert broker.cash >= 0.0


def test_ledger_mode_must_match_config_mode(cfg: TradingConfig, bars, tmp_path) -> None:  # type: ignore[no-untyped-def]
    broker = FakeBroker(cash=100_000.0, bars=bars, today=AS_OF)
    with pytest.raises(RiskLimitBreach, match="ledger mode"):
        run_cycle(broker, GOOD, Ledger(tmp_path / "l", "live"), cfg, AS_OF)


def test_a_breach_is_recorded_in_the_ledger(cfg: TradingConfig, bars, tmp_path) -> None:  # type: ignore[no-untyped-def]
    broker, ledger = setup(tmp_path, bars)
    cfg.kill_switch_path.write_text("stop")
    with pytest.raises(RiskLimitBreach):
        run_cycle(broker, GOOD, ledger, cfg, AS_OF)
    assert "kill switch" in ledger.records("breach")[0]["payload"]["error"]


def test_stale_bar_for_a_held_name_inside_the_rebalance_band_halts(cfg: TradingConfig, bars, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """No order touches CCC, so only the held-position freshness check can catch this."""
    from src.trading.risk import PriceBar

    bars["CCC"] = PriceBar(20.0, AS_OF.replace(day=1))  # 8 days old
    broker, ledger = setup(
        tmp_path, bars, positions={"CCC": 245}, last_equity=100_000.0
    )  # target 250
    broker.cash = 100_000.0 - 245 * 20.0  # equity 100k, 245 vs 250 shares is inside the band
    with pytest.raises(RiskLimitBreach, match="CCC price bar is 8 days old"):
        run_cycle(broker, [Forecast("CCC", 0.0, 0.01, 0.03)], ledger, cfg, AS_OF)
    assert broker.submitted == []
