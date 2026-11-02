import numpy as np
import pytest

from src.eval.splits import purged_walk_forward


@pytest.mark.parametrize("steps", [1, 5])
def test_no_training_label_reaches_the_test_period(steps: int) -> None:
    folds = purged_walk_forward(400, 100, 20, 20, embargo=steps)
    for f in folds:
        # Train row i's label spans bars i+1..i+steps; the test period starts after that.
        assert f.train.max() + steps < f.test.min()


def test_insufficient_embargo_would_leak() -> None:
    f = purged_walk_forward(400, 100, 20, 20, embargo=0)[0]
    assert f.train.max() + 5 >= f.test.min()


def test_fold_geometry() -> None:
    folds = purged_walk_forward(100, 30, 10, 10, embargo=2)
    assert len(folds) == 6
    assert np.array_equal(folds[0].train, np.arange(0, 30))
    assert np.array_equal(folds[0].test, np.arange(32, 42))
    assert np.array_equal(folds[1].train, np.arange(10, 40))
    assert folds[-1].test.max() < 100


def test_train_and_test_never_overlap() -> None:
    for f in purged_walk_forward(300, 80, 15, 15, embargo=5):
        assert not set(f.train) & set(f.test)


def test_too_few_rows_raises() -> None:
    with pytest.raises(ValueError, match="too few"):
        purged_walk_forward(50, 40, 10, 10, embargo=5)


@pytest.mark.parametrize(
    "kwargs", [{"train_window": 0}, {"test_window": 0}, {"step": 0}, {"embargo": -1}]
)
def test_invalid_arguments_raise(kwargs: dict[str, int]) -> None:
    args = {"n": 200, "train_window": 50, "test_window": 10, "step": 10, "embargo": 1, **kwargs}
    with pytest.raises(ValueError):
        purged_walk_forward(**args)
