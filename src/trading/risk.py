"""
Pre-trade risk checks. Every limit raises ``RiskLimitBreach``; nothing is clamped or skipped.

The caller decides what a breach means (the cycle aborts without submitting any order).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from src.trading.config import TradingConfig
from src.trading.rebalance import Order


class RiskLimitBreach(Exception):
    """A risk limit stopped the cycle."""


@dataclass(frozen=True)
class AccountSnapshot:
    equity: float
    cash: float
    last_equity: float  # equity at the previous close

    def __post_init__(self) -> None:
        for name in ("equity", "cash", "last_equity"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"AccountSnapshot.{name} is not finite: {getattr(self, name)}")


@dataclass(frozen=True)
class PriceBar:
    price: float
    as_of: date

    def __post_init__(self) -> None:
        if not math.isfinite(self.price) or self.price <= 0.0:
            raise ValueError(f"PriceBar.price must be finite and positive, got {self.price}")


def check_plan(
    orders: list[Order],
    account: AccountSnapshot,
    positions: dict[str, int],
    bars: dict[str, PriceBar],
    high_water_mark: float,
    orders_today: int,
    as_of: date,
    cfg: TradingConfig,
) -> None:
    """Raise ``RiskLimitBreach`` unless the plan is safe to submit."""
    if cfg.kill_switch_path.exists():
        raise RiskLimitBreach(f"kill switch present at {cfg.kill_switch_path}")
    if not math.isfinite(high_water_mark):
        raise RiskLimitBreach(f"high-water mark is not finite: {high_water_mark}")
    if account.equity <= 0.0 or account.last_equity <= 0.0 or high_water_mark <= 0.0:
        raise RiskLimitBreach(f"non-positive equity figures in {account}, hwm={high_water_mark}")

    daily_loss = (account.last_equity - account.equity) / account.last_equity
    if daily_loss > cfg.max_daily_loss:
        raise RiskLimitBreach(f"daily loss {daily_loss:.2%} exceeds {cfg.max_daily_loss:.2%}")
    drawdown = (high_water_mark - account.equity) / high_water_mark
    if drawdown > cfg.max_drawdown:
        raise RiskLimitBreach(f"drawdown {drawdown:.2%} exceeds {cfg.max_drawdown:.2%}")

    if orders_today + len(orders) > cfg.max_orders_per_day:
        raise RiskLimitBreach(
            f"{orders_today} orders already sent + {len(orders)} planned exceeds "
            f"{cfg.max_orders_per_day}/day"
        )

    # Every ticker we hold or trade must have a fresh bar, and the bars must reconcile with the
    # broker's own equity: stale prices would otherwise size positions against the wrong value.
    for ticker in sorted({o.ticker for o in orders} | {t for t, q in positions.items() if q}):
        if ticker not in cfg.universe:
            raise RiskLimitBreach(f"{ticker} is outside the universe")
        bar = bars.get(ticker)
        if bar is None:
            raise RiskLimitBreach(f"no price bar for {ticker}")
        age = (as_of - bar.as_of).days
        if age > cfg.max_price_age_days or age < 0:
            raise RiskLimitBreach(f"{ticker} price bar is {age} days old (as_of {as_of})")
    marked = account.cash + sum(q * bars[t].price for t, q in positions.items() if q)
    mismatch = abs(marked - account.equity) / account.equity
    if mismatch > cfg.max_equity_mismatch:
        raise RiskLimitBreach(
            f"bars value the account at {marked:.2f} but the broker reports {account.equity:.2f} "
            f"({mismatch:.2%} apart); prices are stale or positions are misread"
        )

    qty_after = dict(positions)
    cash_after = account.cash
    for order in orders:
        if order.ticker not in cfg.universe:
            raise RiskLimitBreach(f"{order.ticker} is outside the universe")
        if order.qty <= 0:
            raise RiskLimitBreach(f"non-positive quantity in {order}")
        bar = bars.get(order.ticker)
        if bar is None:
            raise RiskLimitBreach(f"no price bar for {order.ticker}")
        if bar.price < cfg.min_price:
            raise RiskLimitBreach(f"{order.ticker} price {bar.price} below {cfg.min_price}")

        if order.side == "sell":
            if order.qty > qty_after.get(order.ticker, 0):
                raise RiskLimitBreach(f"sell of {order.qty} {order.ticker} exceeds the holding")
            qty_after[order.ticker] = qty_after.get(order.ticker, 0) - order.qty
            cash_after += order.qty * bar.price
        else:
            qty_after[order.ticker] = qty_after.get(order.ticker, 0) + order.qty
            cash_after -= order.qty * bar.price * (1.0 + cfg.buy_slippage_buffer)

    if cash_after < 0.0:
        raise RiskLimitBreach(f"plan needs margin: cash after trades would be {cash_after:.2f}")

    gross = 0.0
    for ticker, qty in qty_after.items():
        if qty == 0:
            continue
        bar = bars.get(ticker)
        if bar is None:
            raise RiskLimitBreach(f"no price bar for held {ticker}")
        weight = qty * bar.price / account.equity
        if weight > cfg.hard_position_cap + 1e-12:
            raise RiskLimitBreach(
                f"{ticker} would be {weight:.2%} of equity, above {cfg.hard_position_cap:.2%}"
            )
        gross += weight
    if gross > cfg.max_gross_exposure + cfg.rebalance_band + 1e-12:
        raise RiskLimitBreach(f"gross exposure {gross:.2%} exceeds {cfg.max_gross_exposure:.2%}")
