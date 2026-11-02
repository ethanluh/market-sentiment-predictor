import numpy as np
import pandas as pd
import pytest

from src.eval.targets import forward_log_return


def test_forward_return_is_sum_over_next_steps() -> None:
    r = pd.Series([np.nan, 0.01, 0.02, -0.03, 0.04, 0.05, 0.06])
    out = forward_log_return(r, 3)
    assert out.iloc[0] == pytest.approx(0.01 + 0.02 - 0.03)
    assert out.iloc[1] == pytest.approx(0.02 - 0.03 + 0.04)
    assert out.iloc[3] == pytest.approx(0.04 + 0.05 + 0.06)


def test_last_steps_rows_are_nan() -> None:
    out = forward_log_return(pd.Series(np.arange(10, dtype=float)), 4)
    assert out.iloc[-4:].isna().all()
    assert out.iloc[:-4].notna().all()


def test_single_step_matches_shift() -> None:
    r = pd.Series(np.random.default_rng(0).normal(size=20))
    pd.testing.assert_series_equal(forward_log_return(r, 1), r.shift(-1))


def test_multi_step_differs_from_plain_shift() -> None:
    r = pd.Series([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    assert forward_log_return(r, 2).iloc[1] != r.shift(-2).iloc[1]


def test_rejects_nonpositive_steps() -> None:
    with pytest.raises(ValueError):
        forward_log_return(pd.Series([0.1, 0.2]), 0)
