from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.ingestion.price_cache import cached_sources
from src.prediction.dataset import Sources

IDX = pd.bdate_range("2024-01-02", periods=30, tz="UTC")


def frame(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(IDX))))
    df = pd.DataFrame(
        {"close": close, "volume": rng.integers(1_000, 9_000, len(IDX)).astype(float)}, index=IDX
    )
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    return df


class Base:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def sources(self) -> Sources:
        def prices(ticker, start, end, interval):  # type: ignore[no-untyped-def]
            self.calls.append((ticker, start, end, interval))
            return frame(abs(hash(ticker)) % 1000)

        boom = lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("must not be called")
        )  # noqa: E731
        return Sources(
            prices=prices, returns_matrix=boom, trends=boom, insider_flow=boom, news_reader=boom
        )


def test_the_first_fetch_is_saved_and_the_same_day_reuses_it(tmp_path: Path) -> None:
    base = Base()
    src, used = cached_sources(base.sources(), tmp_path, today=lambda: date(2026, 1, 5))
    first = src.prices("AAPL", "2024-01-01", None, "1d")
    second = src.prices("AAPL", "2024-01-01", None, "1d")
    assert len(base.calls) == 1
    pd.testing.assert_frame_equal(first, second, check_freq=False)
    assert used() == date(2026, 1, 5)
    assert (tmp_path / "AAPL_1d_2024-01-01_max" / "2026-01-05.csv").exists()


def test_the_round_trip_preserves_values_dtypes_and_timezone(tmp_path: Path) -> None:
    src, _ = cached_sources(Base().sources(), tmp_path, today=lambda: date(2026, 1, 5))
    original = src.prices("AAPL", None, None, "1d")
    cached = src.prices("AAPL", None, None, "1d")  # read back from disk
    pd.testing.assert_frame_equal(original, cached, check_freq=False)
    assert str(cached.index.tz) == "UTC" and cached.dtypes.eq(float).all()


def test_a_new_day_fetches_fresh_data_and_keeps_the_old_snapshot(tmp_path: Path) -> None:
    base = Base()
    day = {"d": date(2026, 1, 5)}
    src, _ = cached_sources(base.sources(), tmp_path, today=lambda: day["d"])
    src.prices("AAPL", None, None, "1d")
    day["d"] = date(2026, 1, 6)
    src.prices("AAPL", None, None, "1d")
    assert len(base.calls) == 2
    assert sorted(p.name for p in (tmp_path / "AAPL_1d_min_max").iterdir()) == [
        "2026-01-05.csv",
        "2026-01-06.csv",
    ]


def test_an_explicit_snapshot_reads_exactly_that_data_and_never_fetches(tmp_path: Path) -> None:
    base = Base()
    saver, _ = cached_sources(base.sources(), tmp_path, today=lambda: date(2026, 1, 5))
    saved = saver.prices("AAPL", None, None, "1d")
    replay_base = Base()
    replay, used = cached_sources(
        replay_base.sources(), tmp_path, snapshot=date(2026, 1, 5), today=lambda: date(2027, 1, 1)
    )
    pd.testing.assert_frame_equal(replay.prices("AAPL", None, None, "1d"), saved, check_freq=False)
    assert replay_base.calls == [] and used() == date(2026, 1, 5)


def test_a_missing_snapshot_raises_instead_of_refetching(tmp_path: Path) -> None:
    base = Base()
    src, _ = cached_sources(base.sources(), tmp_path, snapshot=date(2026, 1, 5))
    with pytest.raises(FileNotFoundError, match="no price snapshot for AAPL on 2026-01-05"):
        src.prices("AAPL", None, None, "1d")
    assert base.calls == []


def test_keys_separate_tickers_windows_and_unsafe_names(tmp_path: Path) -> None:
    base = Base()
    src, _ = cached_sources(base.sources(), tmp_path, today=lambda: date(2026, 1, 5))
    src.prices("AAPL", None, None, "1d")
    src.prices("AAPL", "2024-01-01", None, "1d")
    src.prices("^VIX", None, None, "1d")
    src.prices("BRK.B", None, None, "1d")
    names = sorted(p.name for p in tmp_path.iterdir())
    assert len(names) == 4 and all("^" not in n for n in names)
    assert "_VIX_1d_min_max" in names and "BRK.B_1d_min_max" in names


def test_a_failed_fetch_leaves_nothing_behind(tmp_path: Path) -> None:
    base = Base().sources()
    base.prices = lambda *a: (_ for _ in ()).throw(ConnectionError("down"))  # type: ignore[assignment,return-value]
    src, _ = cached_sources(base, tmp_path, today=lambda: date(2026, 1, 5))
    with pytest.raises(ConnectionError):
        src.prices("AAPL", None, None, "1d")
    assert list(tmp_path.rglob("*.csv*")) == []


def test_no_temp_file_is_left_after_a_save(tmp_path: Path) -> None:
    src, _ = cached_sources(Base().sources(), tmp_path, today=lambda: date(2026, 1, 5))
    src.prices("AAPL", None, None, "1d")
    assert [p.suffix for p in tmp_path.rglob("*") if p.is_file()] == [".csv"]


def test_the_returns_matrix_goes_through_the_cache_with_the_original_semantics(
    tmp_path: Path,
) -> None:
    base = Base()
    src, _ = cached_sources(base.sources(), tmp_path, today=lambda: date(2026, 1, 5))
    m = src.returns_matrix(["AAPL", "MSFT"], None, None, "1d")
    assert list(m.columns) == ["AAPL", "MSFT"] and len(base.calls) == 2
    assert not m.isna().all(axis=1).any()  # all-NaN rows are dropped, as in fetch_returns_matrix
    src.returns_matrix(["AAPL", "MSFT"], None, None, "1d")
    assert len(base.calls) == 2  # the second matrix came from the cache


def test_the_snapshot_date_is_unknown_until_something_is_fetched(tmp_path: Path) -> None:
    _, used = cached_sources(Base().sources(), tmp_path)
    with pytest.raises(RuntimeError, match="no prices have been fetched"):
        used()


def test_todays_still_forming_bar_is_dropped_when_the_window_is_open_ended(tmp_path: Path) -> None:
    base = Base()
    today = IDX[-1].date()
    src, _ = cached_sources(base.sources(), tmp_path, today=lambda: today)
    got = src.prices("AAPL", None, None, "1d")  # end=None: "up to now"
    assert got.index.max() < pd.Timestamp(today, tz="UTC")  # the bar dated today is gone
    assert len(got) == len(IDX) - 1
    saved = pd.read_csv(next(tmp_path.rglob("*.csv")), index_col=0)
    assert len(saved) == len(IDX) - 1  # and the snapshot never froze it


def test_an_explicit_end_keeps_every_bar(tmp_path: Path) -> None:
    src, _ = cached_sources(Base().sources(), tmp_path, today=lambda: IDX[-1].date())
    assert len(src.prices("AAPL", None, "2030-01-01", "1d")) == len(IDX)


def test_a_run_that_crosses_midnight_cannot_report_a_single_snapshot(tmp_path: Path) -> None:
    day = {"d": date(2026, 1, 5)}
    src, used = cached_sources(Base().sources(), tmp_path, today=lambda: day["d"])
    src.prices("AAPL", None, None, "1d")
    day["d"] = date(2026, 1, 6)
    src.prices("MSFT", None, None, "1d")
    with pytest.raises(RuntimeError, match="several snapshot dates"):
        used()
