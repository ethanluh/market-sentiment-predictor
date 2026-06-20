"""Tests for the bounded retry-with-backoff helper."""

from __future__ import annotations

import pytest

from src.utils.retry import retry_call


def test_returns_first_success_without_sleeping():
    slept: list[float] = []
    assert retry_call(lambda: 7, sleep=slept.append) == 7
    assert slept == []


def test_retries_then_succeeds():
    calls = {"n": 0}
    slept: list[float] = []

    def _flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ValueError("transient")
        return "ok"

    assert retry_call(_flaky, attempts=3, base_delay=0.1, sleep=slept.append) == "ok"
    assert calls["n"] == 3
    assert slept == [0.1, 0.2]  # exponential backoff between the 3 attempts


def test_reraises_last_exception_after_exhausting_attempts():
    calls = {"n": 0}

    def _always():
        calls["n"] += 1
        raise RuntimeError("nope")

    with pytest.raises(RuntimeError, match="nope"):
        retry_call(_always, attempts=2, base_delay=0.0, sleep=lambda _: None)
    assert calls["n"] == 2


def test_only_retries_listed_exceptions():
    def _boom():
        raise KeyError("unlisted")

    with pytest.raises(KeyError):
        retry_call(_boom, attempts=3, exceptions=(ValueError,), sleep=lambda _: None)


def test_invalid_attempts_rejected():
    with pytest.raises(ValueError):
        retry_call(lambda: 1, attempts=0)
