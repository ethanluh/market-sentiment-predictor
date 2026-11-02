from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS

import pytest
from alpaca.common.enums import BaseURL
from alpaca.common.exceptions import APIError
from alpaca.data.enums import DataFeed
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.enums import (
    AccountStatus,
    OrderSide,
    OrderStatus,
    PositionSide,
    QueryOrderStatus,
    TimeInForce,
)

from src.trading import alpaca_broker
from src.trading.alpaca_broker import AlpacaBroker, LiveTradingNotImplemented
from src.trading.broker import DuplicateOrderError, OrderRejected
from src.trading.rebalance import Order

UTC = timezone.utc
# 2026-01-10 03:00 UTC is still Friday 2026-01-09 evening in New York.
AFTER_CLOSE = datetime(2026, 1, 10, 3, 0, tzinfo=UTC)
MID_SESSION = datetime(2026, 1, 9, 17, 0, tzinfo=UTC)


class FakeTrading:
    def __init__(self, now: datetime = AFTER_CLOSE, is_open: bool = False) -> None:
        self.now, self.is_open = now, is_open
        self.account = NS(
            status=AccountStatus.ACTIVE,
            trading_blocked=False,
            account_blocked=False,
            trade_suspended_by_user=False,
            equity="100000.5",
            cash="90000",
            last_equity="99000",
        )
        self.positions: list = []
        self.orders: list = []
        self.submitted: list = []
        self.order_filters: list = []
        self.submit_result = NS(id="abc-123", status=OrderStatus.ACCEPTED)
        self.submit_error: Exception | None = None

    def get_clock(self):  # type: ignore[no-untyped-def]
        return NS(timestamp=self.now, is_open=self.is_open)

    def get_account(self):  # type: ignore[no-untyped-def]
        return self.account

    def get_all_positions(self):  # type: ignore[no-untyped-def]
        return self.positions

    def get_orders(self, request):  # type: ignore[no-untyped-def]
        self.order_filters.append(request)
        return self.orders

    def submit_order(self, request):  # type: ignore[no-untyped-def]
        self.submitted.append(request)
        if self.submit_error:
            raise self.submit_error
        return self.submit_result


class FakeData:
    def __init__(self, series: dict) -> None:
        self.series, self.requests = series, []

    def get_stock_bars(self, request):  # type: ignore[no-untyped-def]
        self.requests.append(request)
        return NS(data=self.series)


def bar(close: float, ts: datetime) -> NS:
    return NS(close=close, timestamp=ts)


def make(trading=None, series=None):  # type: ignore[no-untyped-def]
    t = trading or FakeTrading()
    d = FakeData(series if series is not None else {})
    return AlpacaBroker(t, d), t, d


def test_today_is_the_new_york_market_date() -> None:
    assert make()[0].today().isoformat() == "2026-01-09"  # not Saturday UTC


def test_market_open_flag() -> None:
    assert make(FakeTrading(MID_SESSION, True))[0].is_market_open() is True
    assert make()[0].is_market_open() is False


def test_account_snapshot_parses_strings() -> None:
    acct = make()[0].get_account()
    assert (acct.equity, acct.cash, acct.last_equity) == (100000.5, 90000.0, 99000.0)


@pytest.mark.parametrize(
    "change, match",
    [
        ({"status": AccountStatus.ACCOUNT_CLOSED}, "not ACTIVE"),
        ({"trading_blocked": True}, "blocked"),
        ({"account_blocked": True}, "blocked"),
        ({"trade_suspended_by_user": True}, "blocked"),
        ({"last_equity": None}, "missing"),
    ],
)
def test_unusable_accounts_raise(change: dict, match: str) -> None:
    t = FakeTrading()
    for k, v in change.items():
        setattr(t.account, k, v)
    with pytest.raises((RuntimeError, ValueError), match=match):
        make(t)[0].get_account()


def test_positions_long_short_and_fractional() -> None:
    t = FakeTrading()
    t.positions = [
        NS(symbol="AAA", qty="10", side=PositionSide.LONG),
        NS(symbol="BBB", qty="3", side=PositionSide.SHORT),
    ]
    assert make(t)[0].get_positions() == {"AAA": 10, "BBB": -3}
    t.positions = [NS(symbol="CCC", qty="2.5", side=PositionSide.LONG)]
    with pytest.raises(ValueError, match="fractional"):
        make(t)[0].get_positions()


def test_bars_use_the_last_daily_bar_and_the_documented_request() -> None:
    fri = datetime(2026, 1, 9, 5, 0, tzinfo=UTC)  # daily bars are stamped at midnight New York
    series = {"AAA": [bar(99.0, fri - timedelta(days=1)), bar(101.5, fri)]}
    broker, trading, data = make(series=series)
    got = broker.get_bars(["AAA"])
    assert got["AAA"].price == 101.5 and got["AAA"].as_of.isoformat() == "2026-01-09"
    req = data.requests[0]
    assert req.timeframe.value == TimeFrame.Day.value == "1Day" and req.feed == DataFeed.IEX
    # The SDK stores request times as naive UTC; the instant is what matters.
    assert req.start == (AFTER_CLOSE - timedelta(days=14)).replace(tzinfo=None)


def test_a_partial_bar_mid_session_is_refused() -> None:
    today_bar = bar(100.0, datetime(2026, 1, 9, 5, 0, tzinfo=UTC))
    broker, *_ = make(FakeTrading(MID_SESSION, True), {"AAA": [today_bar]})
    with pytest.raises(RuntimeError, match="partial"):
        broker.get_bars(["AAA"])


def test_missing_bars_raise_and_empty_request_is_empty() -> None:
    broker, *_ = make(series={})
    with pytest.raises(KeyError, match="AAA"):
        broker.get_bars(["AAA"])
    assert broker.get_bars([]) == {}


def test_open_orders_map_to_system_orders() -> None:
    t = FakeTrading()
    t.orders = [NS(symbol="AAA", side=OrderSide.BUY, qty="5", client_order_id="x-1")]
    broker = make(t)[0]
    assert broker.get_open_orders() == [Order("AAA", "buy", 5, "x-1")]
    assert t.order_filters[0].status == QueryOrderStatus.OPEN
    t.orders = [NS(symbol="AAA", side=OrderSide.BUY, qty=None, client_order_id="n")]
    with pytest.raises(ValueError, match="no share quantity"):
        broker.get_open_orders()


def test_submit_sends_only_a_plain_day_market_order() -> None:
    broker, trading, _ = make()
    assert broker.submit_order(Order("AAA", "sell", 7, "20260109-AAA-sell")) == "abc-123"
    req = trading.submitted[0]
    assert (req.symbol, req.qty, req.side, req.time_in_force, req.client_order_id) == (
        "AAA",
        7,
        OrderSide.SELL,
        TimeInForce.DAY,
        "20260109-AAA-sell",
    )
    assert req.extended_hours is None and req.take_profit is None and req.stop_loss is None


def test_duplicate_and_rejected_orders_map_to_system_errors() -> None:
    broker, trading, _ = make()
    trading.submit_error = APIError(
        '{"code": 40010001, "message": "client_order_id must be unique"}'
    )
    with pytest.raises(DuplicateOrderError):
        broker.submit_order(Order("AAA", "buy", 1, "dup"))
    trading.submit_error = APIError('{"code": 40310000, "message": "insufficient buying power"}')
    with pytest.raises(OrderRejected, match="insufficient buying power"):
        broker.submit_order(Order("AAA", "buy", 1, "x"))
    trading.submit_error = None
    trading.submit_result = NS(id="r", status=OrderStatus.REJECTED)
    with pytest.raises(OrderRejected, match="REJECTED|rejected"):
        broker.submit_order(Order("AAA", "buy", 1, "y"))


def test_network_errors_propagate_unchanged() -> None:
    broker, trading, _ = make()
    trading.submit_error = ConnectionError("reset")
    with pytest.raises(ConnectionError):
        broker.submit_order(Order("AAA", "buy", 1, "z"))


def test_live_mode_is_not_implemented() -> None:
    with pytest.raises(LiveTradingNotImplemented):
        AlpacaBroker.paper_from_env("live")


def test_missing_credentials_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ALPACA_PAPER_KEY_ID", raising=False)
    monkeypatch.delenv("ALPACA_PAPER_SECRET_KEY", raising=False)
    with pytest.raises(RuntimeError, match="not set"):
        AlpacaBroker.paper_from_env()


def test_paper_from_env_builds_a_real_paper_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Uses the real SDK client (dummy keys; constructing it makes no network call)."""
    monkeypatch.setenv("ALPACA_PAPER_KEY_ID", "k")
    monkeypatch.setenv("ALPACA_PAPER_SECRET_KEY", "s")
    broker = AlpacaBroker.paper_from_env()
    assert broker._trading._base_url == BaseURL.TRADING_PAPER
    assert broker._trading._base_url.value == "https://paper-api.alpaca.markets"


def test_paper_from_env_passes_paper_true(monkeypatch: pytest.MonkeyPatch) -> None:
    from alpaca.trading.client import TradingClient

    seen = {}

    def spy(key, secret, paper):  # type: ignore[no-untyped-def]
        seen["paper"] = paper
        return TradingClient(key, secret, paper=paper)

    monkeypatch.setenv("ALPACA_PAPER_KEY_ID", "k")
    monkeypatch.setenv("ALPACA_PAPER_SECRET_KEY", "s")
    monkeypatch.setattr("alpaca.trading.client.TradingClient", spy)
    AlpacaBroker.paper_from_env()
    assert seen == {"paper": True}


def test_a_real_live_client_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    from alpaca.trading.client import TradingClient

    monkeypatch.setenv("ALPACA_PAPER_KEY_ID", "k")
    monkeypatch.setenv("ALPACA_PAPER_SECRET_KEY", "s")
    monkeypatch.setattr(
        "alpaca.trading.client.TradingClient",
        lambda key, secret, paper: TradingClient(
            key, secret, paper=False
        ),  # a live-endpoint client
    )
    with pytest.raises(RuntimeError, match="paper endpoint"):
        AlpacaBroker.paper_from_env()


def test_non_json_error_bodies_keep_the_raw_text() -> None:
    broker, trading, _ = make()
    trading.submit_error = APIError("<html>502 Bad Gateway</html>")
    with pytest.raises(OrderRejected, match="502 Bad Gateway"):
        broker.submit_order(Order("AAA", "buy", 1, "x"))
    trading.submit_error = APIError("client_order_id must be unique")  # plain-text duplicate
    with pytest.raises(DuplicateOrderError):
        broker.submit_order(Order("AAA", "buy", 1, "y"))


def test_todays_bar_is_not_trusted_just_after_the_close() -> None:
    today_bar = bar(100.0, datetime(2026, 1, 9, 5, 0, tzinfo=UTC))
    just_after = datetime(2026, 1, 9, 21, 5, tzinfo=UTC)  # 16:05 New York, market closed
    broker, *_ = make(FakeTrading(just_after, False), {"AAA": [today_bar]})
    with pytest.raises(RuntimeError, match="may not be final"):
        broker.get_bars(["AAA"])
    later = datetime(2026, 1, 9, 21, 45, tzinfo=UTC)  # 16:45 New York
    broker, *_ = make(FakeTrading(later, False), {"AAA": [today_bar]})
    assert broker.get_bars(["AAA"])["AAA"].price == 100.0
