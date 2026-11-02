"""Purged walk-forward splits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Fold:
    """Row positions of one fold; ``train`` and ``test`` are ascending."""

    train: np.ndarray
    test: np.ndarray


def purged_walk_forward(
    n: int, train_window: int, test_window: int, step: int, embargo: int
) -> list[Fold]:
    """
    Rolling-window folds with ``embargo`` unused rows between train and test.

    A training row *i* has a label built from bars ``i+1 .. i+steps``. Setting
    ``embargo >= steps`` guarantees every training label ends before the first test
    row, so no training label overlaps the test period.
    """
    for name, value in (("train_window", train_window), ("test_window", test_window)):
        if value < 1:
            raise ValueError(f"{name} must be >= 1, got {value}")
    if step < 1:
        raise ValueError(f"step must be >= 1, got {step}")
    if embargo < 0:
        raise ValueError(f"embargo must be >= 0, got {embargo}")

    folds: list[Fold] = []
    start = 0
    while start + train_window + embargo + test_window <= n:
        test_start = start + train_window + embargo
        folds.append(
            Fold(
                train=np.arange(start, start + train_window),
                test=np.arange(test_start, test_start + test_window),
            )
        )
        start += step
    if not folds:
        raise ValueError(
            f"{n} rows is too few for train_window={train_window} + embargo={embargo} "
            f"+ test_window={test_window}"
        )
    return folds
