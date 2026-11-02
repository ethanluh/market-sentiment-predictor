"""One rebalance cycle: forecasts -> targets -> orders -> risk check -> submit -> ledger."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date

from src.trading.broker import Broker
from src.trading.config import TradingConfig
from src.trading.ledger import Ledger
from src.trading.rebalance import Order, plan_orders
from src.trading.risk import RiskLimitBreach, check_plan
from src.trading.signals import Forecast, decide_targets


@dataclass(frozen=True)
class CycleResult:
    targets: dict[str, float]
    orders: list[Order]
    submitted: list[str]  # client_order_ids accepted by the broker
    dry_run: bool


def run_cycle(
    broker: Broker,
    forecasts: list[Forecast],
    ledger: Ledger,
    cfg: TradingConfig,
    as_of: date,
    dry_run: bool = False,
) -> CycleResult:
    """
    Plan and (unless ``dry_run``) submit one rebalance.

    Refuses to run when ``as_of`` is not the broker's current market date, while orders are open
    at the broker, or when any planned order was already attempted (its state is unknown and
    must be reconciled by hand). Any risk breach raises before a single order is sent.

    A rerun after a partial fill or a rejected order also stops here: order ids repeat within a
    day, so the remainder of that day is reconciled with the broker by hand, never retried blindly.
    """
    if ledger.mode != cfg.mode:
        raise RiskLimitBreach(
            f"ledger mode {ledger.mode!r} does not match config mode {cfg.mode!r}"
        )
    if as_of != broker.today():
        raise RiskLimitBreach(f"as_of {as_of} is not the broker's market date {broker.today()}")
    open_orders = broker.get_open_orders()
    if open_orders:
        raise RiskLimitBreach(f"{len(open_orders)} open orders pending; refusing to plan")

    account = broker.get_account()
    positions = broker.get_positions()
    targets = decide_targets(forecasts, cfg)
    bars = broker.get_bars(sorted(set(targets) | set(positions)))
    prices = {t: b.price for t, b in bars.items()}

    ledger.append("forecasts", {"as_of": as_of, "forecasts": [asdict(f) for f in forecasts]})

    orders = plan_orders(targets, positions, account.equity, prices, as_of, cfg)
    attempted = ledger.attempted_client_ids()
    repeats = sorted(o.client_order_id for o in orders if o.client_order_id in attempted)
    if repeats:
        raise RiskLimitBreach(
            f"orders {repeats} were attempted earlier but positions do not reflect them; "
            "reconcile with the broker by hand before rerunning"
        )
    high_water_mark = max(ledger.high_water_mark() or 0.0, account.equity)
    try:
        check_plan(
            orders,
            account,
            positions,
            bars,
            high_water_mark,
            ledger.orders_attempted_on(as_of),
            as_of,
            cfg,
        )
    except RiskLimitBreach as exc:
        ledger.append("breach", {"as_of": as_of, "error": str(exc), "dry_run": dry_run})
        raise
    # Recorded only after validation, so a glitched reading can never become the high-water mark.
    ledger.append(
        "equity",
        {"as_of": as_of, "equity": account.equity, "cash": account.cash, "dry_run": dry_run},
    )
    ledger.append(
        "plan",
        {
            "as_of": as_of,
            "targets": targets,
            "orders": [asdict(o) for o in orders],
            "dry_run": dry_run,
        },
    )

    submitted: list[str] = []
    if not dry_run:
        for order in orders:
            ledger.append("order_intent", {**asdict(order), "as_of": as_of})
            try:
                broker_id = broker.submit_order(order)
            except Exception as exc:
                ledger.append(
                    "order_failed", {"client_order_id": order.client_order_id, "error": repr(exc)}
                )
                raise
            ledger.append(
                "order_accepted",
                {"client_order_id": order.client_order_id, "broker_order_id": broker_id},
            )
            submitted.append(order.client_order_id)
    return CycleResult(targets=targets, orders=orders, submitted=submitted, dry_run=dry_run)
