from dataclasses import replace
from pathlib import Path

import pytest

from src.trading.config import TradingConfig
from src.trading.rebalance import Order
from src.trading.risk import AccountSnapshot, PriceBar, RiskLimitBreach, check_plan
from tests.trading.conftest import AS_OF

ACCT = AccountSnapshot(equity=100_000.0, cash=100_000.0, last_equity=100_000.0)


def buy(t: str, qty: int) -> Order:
    return Order(t, "buy", qty, f"id-{t}-buy")


def sell(t: str, qty: int) -> Order:
    return Order(t, "sell", qty, f"id-{t}-sell")


def check(cfg, bars, orders, acct=ACCT, positions=None, hwm=100_000.0, today=0, as_of=AS_OF):  # type: ignore[no-untyped-def]
    check_plan(orders, acct, positions or {}, bars, hwm, today, as_of, cfg)


def test_a_safe_plan_passes(cfg, bars) -> None:  # type: ignore[no-untyped-def]
    check(cfg, bars, [buy("AAA", 50), buy("BBB", 100)])


def test_kill_switch_halts_everything(cfg: TradingConfig, bars) -> None:  # type: ignore[no-untyped-def]
    cfg.kill_switch_path.write_text("stop")
    with pytest.raises(RiskLimitBreach, match="kill switch"):
        check(cfg, bars, [])


def test_daily_loss_limit(cfg, bars) -> None:  # type: ignore[no-untyped-def]
    hurt = AccountSnapshot(equity=97_000.0, cash=97_000.0, last_equity=100_000.0)
    with pytest.raises(RiskLimitBreach, match="daily loss"):
        check(cfg, bars, [], acct=hurt, hwm=100_000.0)


def test_drawdown_limit(cfg, bars) -> None:  # type: ignore[no-untyped-def]
    low = AccountSnapshot(equity=89_000.0, cash=89_000.0, last_equity=89_000.0)
    with pytest.raises(RiskLimitBreach, match="drawdown"):
        check(cfg, bars, [], acct=low, hwm=100_000.0)


def test_order_count_limit_counts_orders_already_sent_today(cfg, bars) -> None:  # type: ignore[no-untyped-def]
    small = replace(cfg, max_orders_per_day=3)
    check(small, bars, [buy("AAA", 1)], today=2)
    with pytest.raises(RiskLimitBreach, match="exceeds"):
        check(small, bars, [buy("AAA", 1), buy("BBB", 1)], today=2)


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda b: b.update(AAA=PriceBar(4.0, AS_OF)), "below"),
        (lambda b: b.update(AAA=PriceBar(100.0, AS_OF.replace(day=1))), "days old"),
        (lambda b: b.update(AAA=PriceBar(100.0, AS_OF.replace(day=20))), "days old"),
        (lambda b: b.pop("AAA"), "no price bar"),
    ],
)
def test_bad_price_bars_stop_the_plan(cfg, bars, mutate, match) -> None:  # type: ignore[no-untyped-def]
    mutate(bars)
    with pytest.raises(RiskLimitBreach, match=match):
        check(cfg, bars, [buy("AAA", 10)])


def test_oversell_is_a_breach(cfg, bars) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(RiskLimitBreach, match="exceeds the holding"):
        check(cfg, bars, [sell("AAA", 10)], positions={"AAA": 5})


def test_margin_is_a_breach(cfg, bars) -> None:  # type: ignore[no-untyped-def]
    poor = AccountSnapshot(equity=100_000.0, cash=1_000.0, last_equity=100_000.0)
    with pytest.raises(RiskLimitBreach, match="margin"):
        check(cfg, bars, [buy("AAA", 50)], acct=poor, positions={"BBB": 1980})  # 99k at $50


def test_slippage_buffer_counts_against_cash(cfg, bars) -> None:  # type: ignore[no-untyped-def]
    snug = AccountSnapshot(equity=100_000.0, cash=5_000.0, last_equity=100_000.0)
    with pytest.raises(RiskLimitBreach, match="margin"):
        check(
            cfg, bars, [buy("AAA", 50)], acct=snug, positions={"BBB": 1900}
        )  # +1% buffer needs more


def test_position_cap(cfg, bars) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(RiskLimitBreach, match="above"):
        check(cfg, bars, [buy("AAA", 56)])  # 56 * 100 = 5.6% > 5.5% hard cap
    check(cfg, bars, [buy("AAA", 55)])


def test_gross_exposure_cap(tmp_path: Path, bars) -> None:  # type: ignore[no-untyped-def]
    c = TradingConfig(
        universe=("AAA", "BBB", "CCC", "DDD"),
        max_position_weight=0.5,
        max_gross_exposure=0.5,
        rebalance_band=0.0,
        kill_switch_path=tmp_path / "K",
    )
    with pytest.raises(RiskLimitBreach, match="gross"):
        check(c, bars, [buy("AAA", 400), buy("BBB", 400)])  # 40% + 20%, each under its cap


def test_unknown_ticker_and_bad_quantity(cfg, bars) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(RiskLimitBreach, match="outside the universe"):
        check(cfg, bars, [buy("ZZZ", 1)])
    with pytest.raises(ValueError, match="positive int"):
        buy("AAA", 0)


def test_nonpositive_equity_is_a_breach(cfg, bars) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(RiskLimitBreach, match="non-positive"):
        check(cfg, bars, [], acct=AccountSnapshot(0.0, 0.0, 100_000.0))


@pytest.mark.parametrize("qty", [0, -3, 1.5, float("nan"), True, None])
def test_orders_reject_non_positive_or_non_integer_quantities(qty: object) -> None:
    with pytest.raises(ValueError, match="positive int"):
        Order("AAA", "buy", qty, "id")  # type: ignore[arg-type]


def test_orders_reject_unknown_sides_and_empty_ids() -> None:
    with pytest.raises(ValueError, match="side"):
        Order("AAA", "hold", 1, "id")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty"):
        Order("AAA", "buy", 1, "")


@pytest.mark.parametrize(
    "field",
    [
        "round_trip_cost_bps",
        "edge_buffer_bps",
        "buy_slippage_buffer",
        "min_price",
        "min_p10",
        "max_position_weight",
        "max_daily_loss",
        "max_drawdown",
        "rebalance_band",
        "max_equity_mismatch",
    ],
)
@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_config_values_are_rejected(field: str, bad: float, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="finite"):
        TradingConfig(universe=("A",), kill_switch_path=tmp_path / "K", **{field: bad})  # type: ignore[arg-type]


def test_long_weekend_bars_are_not_a_false_halt(cfg, bars) -> None:  # type: ignore[no-untyped-def]
    bars["AAA"] = PriceBar(100.0, AS_OF.replace(day=5))  # 4 days before as_of: a holiday weekend
    check(cfg, bars, [buy("AAA", 10)])
