"""Shared pytest fixtures and lazy-import guards."""

from __future__ import annotations

import sys

import pytest


@pytest.fixture
def assert_no_heavy_imports():
    """
    Returns a callable that asserts heavy/optional deps were not imported.

    Used to enforce the lazy-import constraint: importing the prediction /
    pipeline modules must not pull in transformers, torch, or uvicorn.
    """

    def _check() -> None:
        heavy = {"transformers", "torch", "uvicorn"}
        loaded = heavy & set(sys.modules)
        assert not loaded, f"heavy deps imported unexpectedly: {loaded}"

    return _check
