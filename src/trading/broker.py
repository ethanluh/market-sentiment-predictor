"""Broker interface and an in-memory fake for tests and dry runs."""

from __future__ import annotations

from datetime import date
from typing import Protocol

from src.trading.rebalance import Order
from src.trading.risk import AccountSnapshot, PriceBar


class OrderRejected(Exception):
    """The broker refused an order."""


class DuplicateOrderError(OrderRejected):
    """An order with this client_order_id was already submitted."""


class Broker(Protocol):
    def today(self) -> date:
        """The broker's current market date (never the caller's clock)."""
        ...

    def get_account(self) -> AccountSnapshot: ...

    def get_positions(self) -> dict[str, int]: ...

    def get_bars(self, tickers: list[str]) -> dict[str, PriceBar]: ...

    def get_open_orders(self) -> list[Order]: ...

    def submit_order(self, order: Order) -> str:
        """Submit a market order; returns the broker's order id. Raises ``OrderRejected``."""
        ...


class FakeBroker:
    """Fills market orders immediately at the current bar price. No margin, no shorting."""

    def __init__(
        self,
        cash: float,
        bars: dict[str, PriceBar],
        today: date,
        positions: dict[str, int] | None = None,
        last_equity: float | None = None,
    ) -> None:
        self._today = today
        self.cash = cash
        self.bars = dict(bars)
        self.positions = dict(positions or {})
        self.pending: list[Order] = []
        self.submitted: list[Order] = []
        self._ids: set[str] = set()
        self._last_equity = last_equity if last_equity is not None else self._equity()

    def _equity(self) -> float:
        return self.cash + sum(q * self.bars[t].price for t, q in self.positions.items())

    def today(self) -> date:
        return self._today

    def get_account(self) -> AccountSnapshot:
        return AccountSnapshot(equity=self._equity(), cash=self.cash, last_equity=self._last_equity)

    def get_positions(self) -> dict[str, int]:
        return {t: q for t, q in self.positions.items() if q != 0}

    def get_bars(self, tickers: list[str]) -> dict[str, PriceBar]:
        missing = [t for t in tickers if t not in self.bars]
        if missing:
            raise KeyError(f"no bars for {missing}")
        return {t: self.bars[t] for t in tickers}

    def get_open_orders(self) -> list[Order]:
        return list(self.pending)

    def submit_order(self, order: Order) -> str:
        if order.client_order_id in self._ids:
            raise DuplicateOrderError(order.client_order_id)
        price = self.bars[order.ticker].price
        held = self.positions.get(order.ticker, 0)
        if order.side == "sell":
            if order.qty > held:
                raise OrderRejected(f"cannot sell {order.qty} {order.ticker}; holding {held}")
            self.positions[order.ticker] = held - order.qty
            self.cash += order.qty * price
        else:
            cost = order.qty * price
            if cost > self.cash:
                raise OrderRejected(f"insufficient cash for {order}: need {cost}, have {self.cash}")
            self.positions[order.ticker] = held + order.qty
            self.cash -= cost
        self._ids.add(order.client_order_id)
        self.submitted.append(order)
        return f"fake-{len(self.submitted)}"
