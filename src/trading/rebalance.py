"""Turn target weights into a whole-share, long-only order list. Pure; no I/O."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Literal

from src.trading.config import TradingConfig

Side = Literal["buy", "sell"]


@dataclass(frozen=True)
class Order:
    ticker: str
    side: Side
    qty: int
    client_order_id: str

    def __post_init__(self) -> None:
        if self.side not in ("buy", "sell"):
            raise ValueError(f"side must be 'buy' or 'sell', got {self.side!r}")
        if isinstance(self.qty, bool) or not isinstance(self.qty, int) or self.qty <= 0:
            raise ValueError(f"qty must be a positive int, got {self.qty!r}")
        if not self.ticker or not self.client_order_id:
            raise ValueError("ticker and client_order_id must be non-empty")


def client_order_id(as_of: date, ticker: str, side: Side) -> str:
    """Deterministic id, so re-running a cycle cannot create a second order for the same day."""
    return f"{as_of:%Y%m%d}-{ticker}-{side}"


def plan_orders(
    targets: dict[str, float],
    positions: dict[str, int],
    equity: float,
    prices: dict[str, float],
    as_of: date,
    cfg: TradingConfig,
) -> list[Order]:
    """
    Orders that move ``positions`` toward ``targets``; sells come first.

    Dropped names exit fully. For held names, deltas smaller than ``rebalance_band`` of
    equity are left alone to avoid churn. A sell never exceeds the held quantity.
    """
    if equity <= 0.0:
        raise ValueError(f"equity must be positive, got {equity}")
    outside = sorted((set(targets) | set(positions)) - set(cfg.universe))
    if outside:
        raise ValueError(f"tickers outside the universe: {outside}")
    for ticker, qty in positions.items():
        if qty < 0:
            raise ValueError(f"short position in {ticker} ({qty}); this system is long-only")

    orders: list[Order] = []
    for ticker in sorted(set(targets) | {t for t, q in positions.items() if q > 0}):
        held = positions.get(ticker, 0)
        weight = targets.get(ticker, 0.0)
        if ticker not in prices:
            raise ValueError(f"no price for {ticker}")
        price = prices[ticker]
        if price <= 0.0:
            raise ValueError(f"non-positive price for {ticker}: {price}")

        target_qty = math.floor(weight * equity / price)
        delta = target_qty - held
        if weight > 0.0 and abs(delta) * price < cfg.rebalance_band * equity:
            continue
        if delta < 0:
            orders.append(Order(ticker, "sell", -delta, client_order_id(as_of, ticker, "sell")))
        elif delta > 0:
            orders.append(Order(ticker, "buy", delta, client_order_id(as_of, ticker, "buy")))

    return sorted(orders, key=lambda o: (o.side != "sell", o.ticker))
