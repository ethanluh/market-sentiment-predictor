from datetime import date
from pathlib import Path

import pytest

from src.trading.config import TradingConfig
from src.trading.risk import PriceBar

AS_OF = date(2026, 1, 9)  # a Friday
UNIVERSE = ("AAA", "BBB", "CCC", "DDD")


@pytest.fixture
def cfg(tmp_path: Path) -> TradingConfig:
    return TradingConfig(universe=UNIVERSE, kill_switch_path=tmp_path / "KILL")


@pytest.fixture
def bars() -> dict[str, PriceBar]:
    return {t: PriceBar(price=p, as_of=AS_OF) for t, p in zip(UNIVERSE, (100.0, 50.0, 20.0, 10.0))}
