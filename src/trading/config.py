"""Trading configuration. Invalid values raise; nothing is clamped or defaulted silently."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Literal

Mode = Literal["paper", "live"]


@dataclass(frozen=True)
class TradingConfig:
    universe: tuple[str, ...]
    horizon: str = "5d"
    mode: Mode = "paper"

    # Signal: a name is held only if its median forecast clears round-trip cost plus a buffer,
    # and its P10 does not forecast a loss worse than ``min_p10``. Returns are log-returns.
    round_trip_cost_bps: float = 10.0
    edge_buffer_bps: float = 15.0
    min_p10: float = -0.08

    # Sizing: long-only, no leverage. ``rebalance_band`` is the largest drift (as a fraction of
    # equity) left untraded; the hard risk cap is ``max_position_weight + rebalance_band``.
    max_position_weight: float = 0.05
    max_gross_exposure: float = 1.0
    rebalance_band: float = 0.005

    # Risk limits.
    max_orders_per_day: int = 60
    max_daily_loss: float = 0.02
    max_drawdown: float = 0.10
    min_price: float = 5.0
    max_price_age_days: int = 5
    max_equity_mismatch: float = 0.02
    buy_slippage_buffer: float = 0.01

    kill_switch_path: Path = Path("data/trading/KILL")

    def __post_init__(self) -> None:
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"{f.name} must be finite, got {value}")
        if not self.universe:
            raise ValueError("universe must not be empty")
        if len(set(self.universe)) != len(self.universe):
            raise ValueError("universe contains duplicate tickers")
        if self.mode not in ("paper", "live"):
            raise ValueError(f"mode must be 'paper' or 'live', got {self.mode!r}")
        for name in ("max_position_weight", "max_gross_exposure"):
            if not 0.0 < getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in (0, 1], got {getattr(self, name)}")
        for name in ("max_daily_loss", "max_drawdown"):
            if not 0.0 < getattr(self, name) < 1.0:
                raise ValueError(f"{name} must be in (0, 1), got {getattr(self, name)}")
        if not 0.0 <= self.rebalance_band < self.max_position_weight:
            raise ValueError("rebalance_band must be in [0, max_position_weight)")
        if not 0.0 < self.max_equity_mismatch < 1.0:
            raise ValueError(
                f"max_equity_mismatch must be in (0, 1), got {self.max_equity_mismatch}"
            )
        for name in ("round_trip_cost_bps", "edge_buffer_bps", "buy_slippage_buffer"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)}")
        if self.min_p10 >= 0.0:
            raise ValueError(f"min_p10 must be negative, got {self.min_p10}")
        if self.max_orders_per_day < 1 or self.max_price_age_days < 0 or self.min_price <= 0.0:
            raise ValueError("max_orders_per_day >= 1, max_price_age_days >= 0, min_price > 0")

    @property
    def edge_threshold(self) -> float:
        """Minimum median log-return forecast needed to hold a name."""
        return (self.round_trip_cost_bps + self.edge_buffer_bps) / 1e4

    @property
    def hard_position_cap(self) -> float:
        return self.max_position_weight + self.rebalance_band
