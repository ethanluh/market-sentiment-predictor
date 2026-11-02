"""
Append-only JSONL ledger: every forecast, plan, order intent and equity snapshot, for audit.

One file per mode (``ledger-paper.jsonl``), so a paper high-water mark can never govern a later
live run. Order intents are written before the broker call (write-ahead), so an order that may
have reached the broker is never forgotten.
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any


class Ledger:
    def __init__(self, directory: Path, mode: str) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.path = directory / f"ledger-{mode}.jsonl"

    def append(self, kind: str, payload: dict[str, object]) -> None:
        """Append one record. NaN/inf anywhere in the payload raises rather than being stored."""
        record = {"ts": datetime.now(timezone.utc).isoformat(), "kind": kind, "payload": payload}
        line = json.dumps(record, sort_keys=True, default=str, allow_nan=False)
        if self.path.exists() and self.path.stat().st_size > 0:
            with self.path.open("rb") as fh:
                fh.seek(-1, os.SEEK_END)
                if fh.read(1) != b"\n":
                    raise ValueError(f"ledger {self.path} ends in a torn line; repair it by hand")
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def records(self, kind: str | None = None) -> list[dict[str, Any]]:
        """All records, optionally of one kind. A corrupt line raises with its line number."""
        if not self.path.exists():
            return []
        out: list[dict[str, Any]] = []
        with self.path.open(encoding="utf-8") as fh:
            for n, line in enumerate(fh, start=1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"corrupt ledger line {n} in {self.path}") from exc
                if kind is None or record["kind"] == kind:
                    out.append(record)
        return out

    def attempted_client_ids(self) -> set[str]:
        """Orders whose submission was started, whether or not it is known to have succeeded."""
        return {str(r["payload"]["client_order_id"]) for r in self.records("order_intent")}

    def orders_attempted_on(self, as_of: date) -> int:
        return sum(
            1 for r in self.records("order_intent") if r["payload"]["as_of"] == as_of.isoformat()
        )

    def high_water_mark(self) -> float | None:
        """Highest equity seen in real (non-dry-run) cycles, or None before the first."""
        values = [
            float(r["payload"]["equity"])
            for r in self.records("equity")
            if not r["payload"].get("dry_run", False)
        ]
        return max(values) if values else None
