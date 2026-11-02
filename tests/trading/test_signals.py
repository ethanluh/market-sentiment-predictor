import pytest

from src.trading.config import TradingConfig
from src.trading.signals import Forecast, decide_targets


def f(t: str, p50: float, p10: float = -0.01, p90: float = 0.05) -> Forecast:
    return Forecast(t, p10, p50, p90)


def test_selects_only_names_clearing_the_edge_threshold(cfg: TradingConfig) -> None:
    out = decide_targets([f("AAA", 0.01), f("BBB", 0.001), f("CCC", -0.02, p10=-0.03)], cfg)
    assert set(out) == {"AAA"}  # 0.001 < 25 bps threshold


def test_equal_weight_is_capped_per_position(cfg: TradingConfig) -> None:
    out = decide_targets([f("AAA", 0.01), f("BBB", 0.01)], cfg)
    assert out == {"AAA": 0.05, "BBB": 0.05}  # 1/2 would be 50%, capped at 5%


def test_gross_exposure_never_exceeds_the_limit(tmp_path) -> None:  # type: ignore[no-untyped-def]
    names = tuple(f"T{i}" for i in range(40))
    c = TradingConfig(universe=names, kill_switch_path=tmp_path / "K")
    out = decide_targets([f(n, 0.01) for n in names], c)
    # Gross leaves 1% headroom for buy slippage, so a full plan still passes the cash check.
    assert sum(out.values()) == pytest.approx(1.0 / 1.01) and max(out.values()) == pytest.approx(
        1.0 / 1.01 / 40
    )


def test_p10_floor_excludes_fat_left_tails(cfg: TradingConfig) -> None:
    assert decide_targets([f("AAA", 0.02, p10=-0.09)], cfg) == {}


def test_nothing_selected_means_all_cash(cfg: TradingConfig) -> None:
    assert decide_targets([f("AAA", -0.01, p10=-0.02)], cfg) == {}
    assert decide_targets([], cfg) == {}


def test_bad_forecasts_raise(cfg: TradingConfig) -> None:
    with pytest.raises(ValueError, match="outside the universe"):
        decide_targets([f("ZZZ", 0.01)], cfg)
    with pytest.raises(ValueError, match="duplicate"):
        decide_targets([f("AAA", 0.01), f("AAA", 0.02)], cfg)
    with pytest.raises(ValueError, match="not ordered"):
        decide_targets([Forecast("AAA", 0.02, 0.01, 0.03)], cfg)
