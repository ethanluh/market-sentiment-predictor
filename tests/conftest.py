"""Shared pytest fixtures and lazy-import guards."""

from __future__ import annotations

import sys

import pytest
from hypothesis import settings

# Timing is not under test, and the first sklearn/scipy call is slow, so a wall-clock
# deadline only produces flaky failures. Inputs stay random so new cases get explored;
# a failure prints a reproduce_failure blob.
settings.register_profile("project", deadline=None, print_blob=True)
settings.load_profile("project")


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
