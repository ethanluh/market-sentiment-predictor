from pathlib import Path

import pytest

from src.trading.config import TradingConfig


def test_defaults_match_the_agreed_design(tmp_path: Path) -> None:
    c = TradingConfig(universe=("A",), kill_switch_path=tmp_path / "K")
    assert c.horizon == "5d" and c.mode == "paper"
    assert c.max_position_weight == 0.05 and c.max_gross_exposure == 1.0
    assert c.edge_threshold == pytest.approx(0.0025)
    assert c.hard_position_cap == pytest.approx(0.055)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"universe": ()},
        {"universe": ("A", "A")},
        {"universe": ("A",), "mode": "yolo"},
        {"universe": ("A",), "max_position_weight": 0.0},
        {"universe": ("A",), "max_position_weight": 1.5},
        {"universe": ("A",), "max_gross_exposure": 0.0},
        {"universe": ("A",), "max_daily_loss": 0.0},
        {"universe": ("A",), "max_drawdown": 1.0},
        {"universe": ("A",), "rebalance_band": 0.06},
        {"universe": ("A",), "round_trip_cost_bps": -1.0},
        {"universe": ("A",), "min_p10": 0.01},
        {"universe": ("A",), "max_orders_per_day": 0},
        {"universe": ("A",), "min_price": 0.0},
    ],
)
def test_invalid_values_raise(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        TradingConfig(**kwargs)  # type: ignore[arg-type]
