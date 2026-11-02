"""
Trading CLI. Dry-run by default; ``--submit`` places PAPER orders. There is no live mode.

    python -m src.trading.run --forecasts data/trading/forecasts.json            # dry run
    python -m src.trading.run --forecasts data/trading/forecasts.json --submit   # paper orders

Run after the close. The cycle refuses to run while the market is open.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from src.trading.broker import Broker
from src.trading.config import TradingConfig
from src.trading.cycle import run_cycle
from src.trading.forecasts import read_forecast_file, read_universe
from src.trading.ledger import Ledger
from src.trading.risk import RiskLimitBreach


def _default_broker() -> Broker:
    from dotenv import load_dotenv

    from src.trading.alpaca_broker import AlpacaBroker

    load_dotenv()
    return AlpacaBroker.paper_from_env()


def main(
    argv: list[str] | None = None, broker_factory: Callable[[], Broker] = _default_broker
) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--forecasts", type=Path, required=True, help="forecast JSON (see forecasts.py)")
    p.add_argument("--universe", type=Path, default=Path("config/universe_starter.txt"))
    p.add_argument("--ledger-dir", type=Path, default=Path("data/trading"))
    p.add_argument("--kill-switch", type=Path, default=Path("data/trading/KILL"))
    p.add_argument("--submit", action="store_true", help="send PAPER orders (default: dry run)")
    args = p.parse_args(argv)

    cfg = TradingConfig(
        universe=read_universe(args.universe), mode="paper", kill_switch_path=args.kill_switch
    )
    broker = broker_factory()
    if broker.is_market_open():
        print("refusing to run: the market is open (run after the close)", file=sys.stderr)
        return 2
    as_of = broker.today()
    forecasts = read_forecast_file(args.forecasts, cfg.universe, as_of, cfg.horizon)
    ledger = Ledger(args.ledger_dir, cfg.mode)
    try:
        result = run_cycle(broker, forecasts, ledger, cfg, as_of, dry_run=not args.submit)
    except RiskLimitBreach as exc:
        print(f"HALT: {exc}", file=sys.stderr)
        return 2
    except Exception:
        # Partial submission is the likeliest real failure; say what already went out, then fail.
        accepted = [r["payload"]["client_order_id"] for r in ledger.records("order_accepted")]
        print(f"FAILED mid-cycle. Orders accepted so far (all dates): {accepted}", file=sys.stderr)
        raise
    print(
        json.dumps(
            {
                "as_of": as_of.isoformat(),
                "dry_run": result.dry_run,
                "targets": result.targets,
                "orders": [asdict(o) for o in result.orders],
                "submitted": result.submitted,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
