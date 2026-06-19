"""Tests for the graceful-degrade fetch helper."""

from __future__ import annotations

import logging

from src.utils.safe import safe_fetch


def test_returns_value_on_success():
    assert safe_fetch("demo", lambda: 42) == 42


def test_returns_none_and_warns_on_failure(caplog):
    def _boom():
        raise RuntimeError("nope")

    with caplog.at_level(logging.WARNING, logger="utils.safe"):
        result = safe_fetch("demo", _boom)

    assert result is None
    assert any("demo" in r.message for r in caplog.records)
