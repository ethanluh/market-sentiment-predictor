"""
Alpaca adapter for the ``Broker`` protocol. Paper trading only.

Only ``paper_from_env`` constructs a real client, and it always passes ``paper=True`` and then
checks the client's base URL. Live trading is not implemented: it needs its own reviewed change.
Only plain DAY market orders are ever sent (queued for the next open when placed after the
close), with whole-share quantities and the system's deterministic client order ids.
"""

from __future__ import annotations

import os
from datetime import date, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from alpaca.common.enums import BaseURL
from alpaca.common.exceptions import APIError
from alpaca.data.enums import DataFeed
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.enums import (
    AccountStatus,
    OrderSide,
    OrderStatus,
    PositionSide,
    QueryOrderStatus,
    TimeInForce,
)
from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest

from src.trading.broker import DuplicateOrderError, OrderRejected
from src.trading.rebalance import Order
from src.trading.risk import AccountSnapshot, PriceBar

NY = ZoneInfo("America/New_York")
# Daily bars for the session just closed are not trusted before this New York time.
BAR_FINAL_AFTER = time(16, 30)


class LiveTradingNotImplemented(Exception):
    """Live trading is deliberately unavailable."""


def _error_text(exc: APIError) -> str:
    """The broker's message, or the raw body when it is not JSON (a 5xx gateway page, say)."""
    try:
        return str(exc.message)
    except Exception:  # APIError.message json-decodes the body and raises on non-JSON
        return str(exc)


def _float(value: object, name: str) -> float:
    if value is None:
        raise ValueError(f"Alpaca account field {name} is missing")
    return float(str(value))


def _whole(value: object, what: str) -> int:
    number = float(str(value))
    if number != int(number):
        raise ValueError(f"{what} is fractional ({value}); this system trades whole shares only")
    return int(number)


class AlpacaBroker:
    def __init__(self, trading: Any, data: Any) -> None:
        self._trading = trading
        self._data = data

    @classmethod
    def paper_from_env(cls, mode: str = "paper") -> "AlpacaBroker":
        if mode != "paper":
            raise LiveTradingNotImplemented(
                "live trading is not implemented; it needs a separate, reviewed change"
            )
        key, secret = os.environ.get("ALPACA_PAPER_KEY_ID"), os.environ.get(
            "ALPACA_PAPER_SECRET_KEY"
        )
        if not key or not secret:
            raise RuntimeError("ALPACA_PAPER_KEY_ID / ALPACA_PAPER_SECRET_KEY are not set")
        from alpaca.data.historical.stock import StockHistoricalDataClient
        from alpaca.trading.client import TradingClient

        trading = TradingClient(key, secret, paper=True)
        base_url = getattr(trading, "_base_url", None)
        if base_url != BaseURL.TRADING_PAPER:
            raise RuntimeError(f"trading client is not on the paper endpoint: {base_url!r}")
        return cls(trading, StockHistoricalDataClient(key, secret))

    def today(self) -> date:
        return self._trading.get_clock().timestamp.astimezone(NY).date()  # type: ignore[no-any-return]

    def is_market_open(self) -> bool:
        return bool(self._trading.get_clock().is_open)

    def get_account(self) -> AccountSnapshot:
        a = self._trading.get_account()
        if a.status != AccountStatus.ACTIVE:
            raise RuntimeError(f"Alpaca account status is {a.status}, not ACTIVE")
        if a.trading_blocked or a.account_blocked or a.trade_suspended_by_user:
            raise RuntimeError("Alpaca account is blocked or suspended")
        return AccountSnapshot(
            equity=_float(a.equity, "equity"),
            cash=_float(a.cash, "cash"),
            last_equity=_float(a.last_equity, "last_equity"),
        )

    def get_positions(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for p in self._trading.get_all_positions():
            qty = _whole(p.qty, f"position {p.symbol}")
            # A short shows as a negative quantity so the planner rejects it as non-long-only.
            out[p.symbol] = -abs(qty) if p.side == PositionSide.SHORT else qty
        return out

    def get_bars(self, tickers: list[str]) -> dict[str, PriceBar]:
        """Latest completed daily bar per ticker (IEX feed). Refuses a partial bar mid-session."""
        if not tickers:
            return {}
        clock = self._trading.get_clock()
        request = StockBarsRequest(
            symbol_or_symbols=tickers,
            timeframe=TimeFrame.Day,
            start=clock.timestamp - timedelta(days=14),
            feed=DataFeed.IEX,
        )
        series = self._data.get_stock_bars(request).data
        today = clock.timestamp.astimezone(NY).date()
        out: dict[str, PriceBar] = {}
        for ticker in tickers:
            bars = series.get(ticker)
            if not bars:
                raise KeyError(f"Alpaca returned no daily bars for {ticker}")
            last = bars[-1]
            bar_date = last.timestamp.astimezone(NY).date()
            if bar_date == today:
                if clock.is_open:
                    raise RuntimeError(
                        f"{ticker}: the market is open, so today's daily bar is partial"
                    )
                if clock.timestamp.astimezone(NY).time() < BAR_FINAL_AFTER:
                    raise RuntimeError(
                        f"{ticker}: today's daily bar may not be final before {BAR_FINAL_AFTER} New York time"
                    )
            out[ticker] = PriceBar(price=float(last.close), as_of=bar_date)
        return out

    def get_open_orders(self) -> list[Order]:
        request = GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500)
        out: list[Order] = []
        for o in self._trading.get_orders(request):
            if o.qty is None:
                raise ValueError(f"open order {o.client_order_id} has no share quantity")
            out.append(
                Order(
                    ticker=o.symbol,
                    side="buy" if o.side == OrderSide.BUY else "sell",
                    qty=_whole(o.qty, f"open order {o.client_order_id}"),
                    client_order_id=o.client_order_id,
                )
            )
        return out

    def submit_order(self, order: Order) -> str:
        request = MarketOrderRequest(
            symbol=order.ticker,
            qty=order.qty,
            side=OrderSide.BUY if order.side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
            client_order_id=order.client_order_id,
        )
        try:
            response = self._trading.submit_order(request)
        except APIError as exc:
            text = _error_text(exc)
            if "client_order_id" in text.lower() and "unique" in text.lower():
                raise DuplicateOrderError(order.client_order_id) from exc
            raise OrderRejected(f"{order.client_order_id}: {text}") from exc
        if response.status in (OrderStatus.REJECTED, OrderStatus.CANCELED, OrderStatus.EXPIRED):
            raise OrderRejected(f"{order.client_order_id} came back {response.status}")
        return str(response.id)
