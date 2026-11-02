import numpy as np
import pytest

from src.eval.trading import ANNUALISATION, annualized_sharpe, buy_and_hold_pnl, trade_pnl


def test_pnl_follows_sign_of_p50() -> None:
    pnl = trade_pnl(np.array([1.0, 1.0, -1.0]), np.array([0.01, 0.02, 0.03]), 1, 0.0)
    assert pnl.tolist() == pytest.approx([0.01, 0.02, -0.03])


def test_costs_charge_each_position_change() -> None:
    # 0 -> +1 (1 unit), +1 -> +1 (0), +1 -> -1 (2 units).
    pnl = trade_pnl(np.array([1.0, 1.0, -1.0]), np.zeros(3), 1, 10.0)
    assert pnl.tolist() == pytest.approx([-0.001, 0.0, -0.002])


def test_overlapping_horizon_trades_every_steps_rows() -> None:
    pnl = trade_pnl(np.ones(10), np.arange(10, dtype=float), steps=5, cost_bps=0.0)
    assert pnl.tolist() == [0.0, 5.0]
    assert buy_and_hold_pnl(np.arange(10, dtype=float), 5).tolist() == [0.0, 5.0]


def test_long_only_floors_shorts() -> None:
    pnl = trade_pnl(np.array([-1.0, 1.0]), np.array([0.5, 0.5]), 1, 0.0, long_only=True)
    assert pnl.tolist() == [0.0, 0.5]


def test_sharpe_annualisation() -> None:
    x = np.array([0.01, -0.005, 0.02, 0.0, 0.015])
    expected = x.mean() / x.std(ddof=1) * np.sqrt(252)
    assert annualized_sharpe(x, 252.0) == pytest.approx(expected)


def test_sharpe_is_nan_without_variance_and_raises_when_too_short() -> None:
    assert np.isnan(annualized_sharpe(np.zeros(10), 252.0))
    with pytest.raises(ValueError):
        annualized_sharpe(np.array([0.1]), 252.0)


def test_five_day_horizon_annualises_non_overlapping_periods() -> None:
    assert ANNUALISATION["5d"] == pytest.approx(252 / 5)


def test_invalid_steps_raises() -> None:
    with pytest.raises(ValueError):
        trade_pnl(np.ones(3), np.ones(3), 0, 0.0)
