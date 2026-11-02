import numpy as np
import pytest

from src.trading.config import TradingConfig
from src.trading.rebalance import plan_orders
from tests.trading.conftest import AS_OF, UNIVERSE

PRICES = {"AAA": 100.0, "BBB": 50.0, "CCC": 20.0, "DDD": 10.0}


def test_buys_whole_shares_toward_target(cfg: TradingConfig) -> None:
    orders = plan_orders({"AAA": 0.05, "BBB": 0.05}, {}, 100_000.0, PRICES, AS_OF, cfg)
    assert [(o.ticker, o.side, o.qty) for o in orders] == [("AAA", "buy", 50), ("BBB", "buy", 100)]


def test_dropped_names_exit_fully_and_sells_come_first(cfg: TradingConfig) -> None:
    orders = plan_orders({"AAA": 0.05}, {"BBB": 7, "AAA": 0}, 100_000.0, PRICES, AS_OF, cfg)
    assert [(o.ticker, o.side, o.qty) for o in orders] == [("BBB", "sell", 7), ("AAA", "buy", 50)]


def test_small_drift_inside_the_band_is_left_alone(cfg: TradingConfig) -> None:
    # Target 50 shares ($5,000). Holding 49 differs by $100 < 0.5% of equity ($500).
    assert plan_orders({"AAA": 0.05}, {"AAA": 49}, 100_000.0, PRICES, AS_OF, cfg) == []


def test_oversized_positions_are_sold_down(cfg: TradingConfig) -> None:
    orders = plan_orders({"AAA": 0.05}, {"AAA": 80}, 100_000.0, PRICES, AS_OF, cfg)
    assert [(o.side, o.qty) for o in orders] == [("sell", 30)]


def test_client_order_ids_are_deterministic(cfg: TradingConfig) -> None:
    a = plan_orders({"AAA": 0.05}, {}, 100_000.0, PRICES, AS_OF, cfg)
    b = plan_orders({"AAA": 0.05}, {}, 100_000.0, PRICES, AS_OF, cfg)
    assert a == b and a[0].client_order_id == "20260109-AAA-buy"


def test_invalid_inputs_raise(cfg: TradingConfig) -> None:
    with pytest.raises(ValueError, match="long-only"):
        plan_orders({}, {"AAA": -5}, 100_000.0, PRICES, AS_OF, cfg)
    with pytest.raises(ValueError, match="outside the universe"):
        plan_orders({}, {"ZZZ": 5}, 100_000.0, {**PRICES, "ZZZ": 1.0}, AS_OF, cfg)
    with pytest.raises(ValueError, match="no price"):
        plan_orders({"AAA": 0.05}, {}, 100_000.0, {}, AS_OF, cfg)
    with pytest.raises(ValueError, match="equity"):
        plan_orders({}, {}, 0.0, PRICES, AS_OF, cfg)
    with pytest.raises(ValueError, match="non-positive price"):
        plan_orders({"AAA": 0.05}, {}, 100_000.0, {"AAA": 0.0}, AS_OF, cfg)


def test_random_portfolios_never_short_or_oversell(cfg: TradingConfig) -> None:
    rng = np.random.default_rng(0)
    for _ in range(300):
        held = {t: int(rng.integers(0, 400)) for t in UNIVERSE}
        targets = {t: 0.05 for t in UNIVERSE if rng.random() < 0.5}
        orders = plan_orders(targets, held, 100_000.0, PRICES, AS_OF, cfg)
        after = dict(held)
        for o in orders:
            assert o.qty > 0
            after[o.ticker] += o.qty if o.side == "buy" else -o.qty
        assert min(after.values()) >= 0
        for t in {o.ticker for o in orders}:
            if t in targets:  # rebalanced names land at or below target, never above it
                assert after[t] * PRICES[t] <= 0.05 * 100_000.0 + 1e-9
            else:  # dropped names exit fully
                assert after[t] == 0
