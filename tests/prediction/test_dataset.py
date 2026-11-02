from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.ingestion import news_archive as na
from src.ingestion.news import NewsArticle
from src.prediction import features as features_mod
from src.prediction.dataset import (
    NEWS_CUTOFF_UTC,
    TRENDS_AVAILABILITY_LAG,
    DatasetConfig,
    Sources,
    bar_close,
    build_dataset,
    regime_path,
)
from src.prediction.features import FEATURE_NAMES
from src.sentiment.aggregation import aggregate

N = 400
IDX = pd.bdate_range("2022-01-03", periods=N, tz="UTC")
_rng = np.random.default_rng(0)
_COMMON = _rng.normal(0, 0.01, N)  # a shared factor, so the correlation graph has real edges


def _prices(seed: int) -> pd.DataFrame:
    r = 0.9 * _COMMON + np.random.default_rng(seed).normal(0, 0.005, N)
    close = 100 * np.exp(np.cumsum(r))
    vol = np.random.default_rng(seed + 10).integers(1_000_000, 5_000_000, N).astype(float)
    df = pd.DataFrame({"close": close, "volume": vol}, index=IDX)
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    return df


PRICES = {"AAA": _prices(1), "BBB": _prices(2), "CCC": _prices(3)}
RETURNS = pd.DataFrame({t: p["log_return"] for t, p in PRICES.items()})
VIX = pd.DataFrame(
    {"close": 15 + 10 * np.abs(np.sin(np.arange(N) / 17.0)), "volume": 1.0, "log_return": 0.0},
    index=IDX,
)
TRENDS = pd.Series(50 + 20 * np.sin(np.arange(N) / 9.0), index=IDX, name="search_interest")
INSIDER = pd.Series(np.random.default_rng(5).integers(-5000, 5000, N).astype(float), index=IDX)


def _cut(obj, start, end):  # type: ignore[no-untyped-def]
    out = obj
    if start:
        out = out[out.index >= pd.Timestamp(start, tz="UTC")]
    if end:
        out = out[out.index <= pd.Timestamp(end, tz="UTC")]
    return out


def fake_sources(upto: int | None = None, reader=None, peek_days: int = 0, spy=None, **overrides) -> Sources:  # type: ignore[no-untyped-def]
    """Fakes whose DATA ends at bar ``upto`` (not just a requested end date), so a look-ahead in the
    builder or the feature code shows up as a difference against the full-data run."""
    cut = IDX[upto] if upto is not None else None

    def visible(obj):  # type: ignore[no-untyped-def]
        return obj if cut is None else obj[obj.index <= cut]

    def prices(ticker, start, end, interval):  # type: ignore[no-untyped-def]
        return visible(_cut(VIX if ticker == "^VIX" else PRICES[ticker], start, end))

    def returns_matrix(tickers, start, end, interval):  # type: ignore[no-untyped-def]
        out = visible(_cut(RETURNS[tickers], start, end))
        if spy is not None:
            spy["returns"] = out  # exactly the data this run was given, for the look-ahead control
        return out

    def news_reader(ticker, lookback):  # type: ignore[no-untyped-def]
        base = reader or (lambda as_of: [])
        horizon = bar_close(cut.to_pydatetime()) if cut is not None else None

        def read(as_of):  # type: ignore[no-untyped-def]
            articles = base(as_of + timedelta(days=peek_days))
            return (
                articles if horizon is None else [a for a in articles if a.published_at <= horizon]
            )

        return read

    s = Sources(
        prices=prices,
        returns_matrix=returns_matrix,
        trends=lambda ticker, start, end: visible(_cut(TRENDS, start, end)),
        insider_flow=lambda ticker, after: visible(INSIDER[INSIDER.index >= after]),
        news_reader=news_reader,
    )
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


FULL = dict(peers=("BBB", "CCC"), use_vix=True, use_trends=True, use_insider_flow=True)


@pytest.fixture
def archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(na, "ARCHIVE_ROOT", tmp_path / "raw")
    monkeypatch.setattr(na, "SCORE_CACHE_ROOT", tmp_path / "scores")
    return tmp_path


@pytest.fixture
def busy_archive(archive: Path):  # type: ignore[no-untyped-def]
    """A few articles every bar, some after the 20:00 UTC cutoff, with cached scores."""
    rng = np.random.default_rng(7)
    scores = {}
    for i, day in enumerate(IDX):
        arts = []
        for k in range(4):
            url = f"u-{i}-{k}"
            hour = int(rng.integers(6, 24))
            arts.append(
                NewsArticle(f"h{i}{k}", "", "reuters", url, day.to_pydatetime().replace(hour=hour))
            )
            scores[url] = float(rng.uniform(-1, 1))
        na.write_day("AAA", day.to_pydatetime(), arts)
    na._save_score_cache("AAA", scores)
    return na.make_archive_reader("AAA", lookback_days=7)


# ---- the basics ----------------------------------------------------------------------------------


def test_builds_a_clean_feature_matrix() -> None:
    ds = build_dataset(DatasetConfig("AAA", start="2022-01-03", **FULL), fake_sources())
    assert list(ds.features.columns) == FEATURE_NAMES and not ds.features.isna().any().any()
    assert ds.dropped_rows > 0 and len(ds.features) + ds.dropped_rows == N
    assert ds.log_returns.index.equals(IDX) and ds.vix is not None
    assert (ds.features["regime_label"] == -1).all()  # left for the harness to refit per fold
    assert ds.notes == ["not enabled, so neutral: sentiment_agg/estimated_lag_hours"]
    assert ds.features.index[-1] == IDX[-1]  # the latest bar is always present


def test_every_enabled_source_actually_moves_its_feature() -> None:
    """Guards the leakage tests below: they mean nothing if a feature is inert in the fixture."""
    f = build_dataset(DatasetConfig("AAA", **FULL), fake_sources()).features
    assert (f["sector_proximity"] > 0).sum() > 100 and f["sector_proximity"].nunique() > 5
    assert f["search_interest_zscore"].nunique() > 50
    assert (f["insider_flow_npr"] != 0).sum() > 100 and f["insider_flow_npr"].nunique() > 50


def test_rows_are_consecutive_bars() -> None:
    ds = build_dataset(DatasetConfig("AAA", **FULL), fake_sources())
    assert (np.diff(IDX.get_indexer(ds.features.index)) == 1).all()


def test_regime_fit_all_labels_match_the_classifier_and_vary() -> None:
    ds = build_dataset(DatasetConfig("AAA", **FULL), fake_sources(), regime="fit_all")
    assert ds.regime_classifier is not None
    labels = ds.features["regime_label"]
    assert labels.nunique() >= 2 and set(labels.unique()) <= {0.0, 1.0, 2.0}
    expected = ds.regime_classifier.predict(
        ds.vix.reindex(ds.features.index, method="ffill").to_numpy()
    )
    assert (labels.to_numpy() == expected).all()


def test_regime_arguments_are_validated() -> None:
    with pytest.raises(ValueError, match="fit_all"):
        build_dataset(DatasetConfig("AAA"), fake_sources(), regime="fit_all")
    with pytest.raises(ValueError, match="regime must be"):
        build_dataset(DatasetConfig("AAA", use_vix=True), fake_sources(), regime="bogus")  # type: ignore[arg-type]


def test_the_regime_sidecar_path_never_collides_across_model_names() -> None:
    assert regime_path("models/m.joblib") == Path("models/m.joblib.regime.joblib")
    assert regime_path("models/aapl.v2") != regime_path("models/aapl.v3")
    assert regime_path("models/m.joblib") != regime_path("models/m.pkl")
    assert regime_path("models/aapl") == Path("models/aapl.regime.joblib")


def test_only_daily_bars_are_supported() -> None:
    for interval in ("1h", "1wk", "5m"):
        with pytest.raises(NotImplementedError, match="not supported"):
            DatasetConfig("AAA", interval=interval)
    with pytest.raises(NotImplementedError):
        bar_close(datetime(2024, 1, 5, tzinfo=timezone.utc), "1h")


# ---- strictness: nothing degrades silently -------------------------------------------------------


def _boom(*a, **k):  # type: ignore[no-untyped-def]
    raise ConnectionError("source down")


@pytest.mark.parametrize(
    "flag, source", [("use_trends", "trends"), ("use_insider_flow", "insider_flow")]
)
def test_a_failing_enabled_source_raises(flag: str, source: str) -> None:
    with pytest.raises(ConnectionError, match="source down"):
        build_dataset(DatasetConfig("AAA", **{flag: True}), fake_sources(**{source: _boom}))


def test_a_failing_vix_fetch_raises() -> None:
    def prices(ticker, start, end, interval):  # type: ignore[no-untyped-def]
        if ticker == "^VIX":
            raise ConnectionError("vix down")
        return PRICES[ticker]

    with pytest.raises(ConnectionError, match="vix down"):
        build_dataset(DatasetConfig("AAA", use_vix=True), fake_sources(prices=prices))


_EMPTY = pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"))
_EMPTY_FRAME = pd.DataFrame(
    columns=["close", "volume", "log_return"], index=pd.DatetimeIndex([], tz="UTC")
)


@pytest.mark.parametrize(
    "flag, override",
    [
        ("use_trends", {"trends": lambda *a: _EMPTY}),
        ("use_insider_flow", {"insider_flow": lambda *a: _EMPTY}),
        ("use_vix", {"prices": lambda t, s, e, i: _EMPTY_FRAME if t == "^VIX" else PRICES[t]}),
    ],
)
def test_each_enabled_source_that_returns_nothing_raises(flag: str, override: dict) -> None:
    with pytest.raises(RuntimeError, match="returned no data"):
        build_dataset(DatasetConfig("AAA", **{flag: True}), fake_sources(**override))


def test_vix_that_starts_late_raises_instead_of_labelling_early_rows_unknown() -> None:
    late = lambda t, s, e, i: VIX.iloc[100:] if t == "^VIX" else PRICES[t]  # noqa: E731
    with pytest.raises(RuntimeError, match="does not cover the price window"):
        build_dataset(
            DatasetConfig("AAA", use_vix=True), fake_sources(prices=late), regime="fit_all"
        )


@pytest.mark.parametrize(
    "name, kwargs", [("^VIX", {"use_vix": True}), ("Google Trends", {"use_trends": True})]
)
def test_a_source_that_ends_long_before_the_last_bar_is_stale(name: str, kwargs: dict) -> None:
    ends_early = fake_sources(
        prices=lambda t, s, e, i: VIX.iloc[:300] if t == "^VIX" else PRICES[t],
        trends=lambda *a: TRENDS.iloc[:300],
    )
    with pytest.raises(RuntimeError, match=f"{name.replace('^', '.')} ends .* it is stale"):
        build_dataset(DatasetConfig("AAA", **kwargs), ends_early)


def test_an_empty_news_archive_raises(archive: Path) -> None:
    from src.ingestion.news_archive import make_archive_reader

    empty = make_archive_reader("AAA", lookback_days=7)  # nothing archived for AAA
    with pytest.raises(RuntimeError, match="no bar had an article"):
        build_dataset(DatasetConfig("AAA", use_news_archive=True), fake_sources(reader=empty))
    with pytest.raises(RuntimeError, match="no bar had an article"):
        build_dataset(
            DatasetConfig("AAA", use_news_archive=True), fake_sources()
        )  # reader returns []


def test_sparse_news_is_allowed_and_its_coverage_is_reported(archive: Path) -> None:
    day = IDX[200]
    na.write_day(
        "AAA",
        day.to_pydatetime(),
        [NewsArticle("h", "", "reuters", "u", day.to_pydatetime().replace(hour=15))],
    )
    na._save_score_cache("AAA", {"u": 0.5})
    ds = build_dataset(
        DatasetConfig("AAA", use_news_archive=True),
        fake_sources(reader=na.make_archive_reader("AAA", lookback_days=7)),
    )
    assert 0 < ds.news_coverage < 0.1  # one article covers its own bar plus the next few days


def test_missing_prices_or_peers_raise() -> None:
    with pytest.raises(RuntimeError, match="no price data"):
        build_dataset(DatasetConfig("AAA"), fake_sources(prices=lambda *a: pd.DataFrame()))
    short = lambda t, s, e, i: pd.DataFrame({"AAA": PRICES["AAA"]["log_return"]})  # noqa: E731
    with pytest.raises(RuntimeError, match="missing \\['BBB'\\]"):
        build_dataset(DatasetConfig("AAA", peers=("BBB",)), fake_sources(returns_matrix=short))


def test_prices_must_be_timezone_aware_and_complete() -> None:
    naive = PRICES["AAA"].tz_localize(None)
    with pytest.raises(RuntimeError, match="timezone-aware"):
        build_dataset(DatasetConfig("AAA"), fake_sources(prices=lambda *a: naive))
    with pytest.raises(RuntimeError, match="lack columns"):
        build_dataset(
            DatasetConfig("AAA"), fake_sources(prices=lambda *a: PRICES["AAA"][["close"]])
        )


def test_too_few_bars_raise() -> None:
    with pytest.raises(RuntimeError, match="no complete feature rows"):
        build_dataset(DatasetConfig("AAA"), fake_sources(prices=lambda *a: PRICES["AAA"].iloc[:10]))


@pytest.mark.parametrize("column", ["close", "volume", "log_return"])
@pytest.mark.parametrize("row", [150, N - 3])
def test_holes_in_the_price_data_raise_instead_of_being_bridged(column: str, row: int) -> None:
    holey = PRICES["AAA"].copy()
    holey.iloc[row, holey.columns.get_loc(column)] = np.nan
    with pytest.raises(RuntimeError, match="missing close, volume or return"):
        build_dataset(DatasetConfig("AAA"), fake_sources(prices=lambda t, *a: holey))


def test_the_first_bars_missing_return_is_normal() -> None:
    assert np.isnan(PRICES["AAA"]["log_return"].iloc[0])  # a return needs a previous close
    build_dataset(DatasetConfig("AAA"), fake_sources())  # and must not trip the hole check


def test_non_consecutive_feature_rows_or_a_missing_latest_bar_are_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invariant guards on build_feature_matrix's output (unreachable with valid prices, but they
    protect every downstream index-aligned step, so they are tested directly)."""
    import src.prediction.dataset as dataset_mod

    real = dataset_mod.build_feature_matrix

    def with_interior_nan(*a, **k):  # type: ignore[no-untyped-def]
        m = real(*a, **k)
        m.iloc[200, 0] = np.nan
        return m

    monkeypatch.setattr(dataset_mod, "build_feature_matrix", with_interior_nan)
    with pytest.raises(RuntimeError, match="not consecutive bars"):
        build_dataset(DatasetConfig("AAA"), fake_sources())

    def with_nan_tail(*a, **k):  # type: ignore[no-untyped-def]
        m = real(*a, **k)
        m.iloc[-1, 0] = np.nan
        return m

    monkeypatch.setattr(dataset_mod, "build_feature_matrix", with_nan_tail)
    with pytest.raises(RuntimeError, match="latest bar has no complete features"):
        build_dataset(DatasetConfig("AAA"), fake_sources())


# ---- leakage: truncating the DATA never changes a row already built ------------------------------


def _same_after_truncation(cut: int, **kw) -> bool:  # type: ignore[no-untyped-def]
    cfg = DatasetConfig("AAA", use_news_archive=True, **FULL)
    short = build_dataset(cfg, fake_sources(upto=cut, **kw))
    full = build_dataset(cfg, fake_sources(**kw))
    assert len(short.features) < len(full.features)
    try:
        pd.testing.assert_frame_equal(short.features, full.features.loc[short.features.index])
        return True
    except AssertionError:
        return False


@pytest.mark.parametrize("cut", [150, 250, 330])
def test_no_enabled_source_lets_a_row_see_past_its_own_bar(cut: int, busy_archive) -> None:  # type: ignore[no-untyped-def]
    """Prices, peers, trends, insider flow and news all on; the data itself ends at ``cut``."""
    assert _same_after_truncation(cut, reader=busy_archive)


# Negative controls: the test above is only worth having if it CATCHES look-ahead. Each control
# injects a real look-ahead bug into the feature code and expects the comparison to fail.


def test_the_test_catches_insider_lookahead(monkeypatch: pytest.MonkeyPatch, busy_archive) -> None:  # type: ignore[no-untyped-def]
    orig = features_mod._insider_flow_npr
    monkeypatch.setattr(
        features_mod,
        "_insider_flow_npr",
        lambda s, as_of, *a, **k: orig(s, as_of + pd.Timedelta(days=30), *a, **k),
    )
    assert not _same_after_truncation(250, reader=busy_archive)


def test_the_test_catches_trends_lookahead(monkeypatch: pytest.MonkeyPatch, busy_archive) -> None:  # type: ignore[no-untyped-def]
    orig = features_mod._search_interest_zscore
    monkeypatch.setattr(
        features_mod,
        "_search_interest_zscore",
        lambda t, as_of, *a, **k: orig(t, as_of + pd.Timedelta(days=30), *a, **k),
    )
    assert not _same_after_truncation(250, reader=busy_archive)


def test_the_test_catches_sector_graph_lookahead(monkeypatch: pytest.MonkeyPatch, busy_archive) -> None:  # type: ignore[no-untyped-def]
    spy: dict = {}
    orig = features_mod.build_sector_graph
    # Peek 30 days past `as_of` into the returns this run was given (shorter in the truncated run).
    monkeypatch.setattr(
        features_mod,
        "build_sector_graph",
        lambda hist, **kw: orig(spy["returns"].loc[: hist.index[-1] + pd.Timedelta(days=30)], **kw),
    )
    assert not _same_after_truncation(250, reader=busy_archive, spy=spy)


def test_the_test_catches_news_lookahead(busy_archive) -> None:  # type: ignore[no-untyped-def]
    assert not _same_after_truncation(250, reader=busy_archive, peek_days=2)


# ---- news alignment ------------------------------------------------------------------------------


def test_daily_bars_query_news_at_the_close_not_midnight() -> None:
    seen: list[datetime] = []

    def reader(as_of: datetime):  # type: ignore[no-untyped-def]
        seen.append(as_of)
        return []

    with pytest.raises(RuntimeError, match="no bar had an article"):  # an empty reader is an error
        build_dataset(DatasetConfig("AAA", use_news_archive=True), fake_sources(reader=reader))
    assert seen and all(t.time() == NEWS_CUTOFF_UTC and t.tzinfo is not None for t in seen)
    assert bar_close(datetime(2024, 1, 5, tzinfo=timezone.utc)) == datetime(
        2024, 1, 5, 20, tzinfo=timezone.utc
    )


def test_same_day_pre_close_news_counts_and_post_close_news_waits_a_day(archive: Path) -> None:
    day = IDX[120]
    pre = NewsArticle("pre-close", "", "reuters", "u-pre", day.to_pydatetime().replace(hour=15))
    post = NewsArticle("post-close", "", "reuters", "u-post", day.to_pydatetime().replace(hour=22))
    na.write_day("AAA", day.to_pydatetime(), [pre, post])
    na._save_score_cache("AAA", {"u-pre": 0.8, "u-post": -0.9})

    ds = build_dataset(
        DatasetConfig("AAA", use_news_archive=True),
        fake_sources(reader=na.make_archive_reader("AAA", lookback_days=7)),
    )
    on_day, next_day = (
        ds.features.loc[day, "sentiment_agg"],
        ds.features.loc[IDX[121], "sentiment_agg"],
    )
    assert on_day == pytest.approx(0.8)  # only the pre-close article is visible on its own day
    assert next_day < on_day  # the post-close negative article arrives with the next bar
    assert ds.features.loc[IDX[119], "sentiment_agg"] == 0.0  # nothing earlier


def test_news_recency_decay_is_measured_from_the_close(archive: Path) -> None:
    """With midnight as the reference, a same-day article gets zero decay (age clamps to 0)."""
    day, prev = IDX[150], IDX[149]
    old = NewsArticle("old", "", "reuters", "u-old", prev.to_pydatetime().replace(hour=15))
    new = NewsArticle("new", "", "reuters", "u-new", day.to_pydatetime().replace(hour=15))
    na.write_day("AAA", prev.to_pydatetime(), [old])
    na.write_day("AAA", day.to_pydatetime(), [new])
    na._save_score_cache("AAA", {"u-old": -0.6, "u-new": 0.8})
    reader = na.make_archive_reader("AAA", lookback_days=7)
    close = bar_close(day.to_pydatetime())

    ds = build_dataset(DatasetConfig("AAA", use_news_archive=True), fake_sources(reader=reader))
    expected = aggregate(reader(close), as_of=close)
    assert ds.features.loc[day, "sentiment_agg"] == pytest.approx(expected)
    wrong = aggregate(reader(close), as_of=day.to_pydatetime())  # midnight reference
    assert (
        abs(expected - wrong) > 1e-6
    )  # the two references really differ, so the test is not vacuous


# ---- trends and insider availability -------------------------------------------------------------


def test_a_trends_value_is_not_used_before_it_is_complete() -> None:
    spike_day = IDX[200]
    trends = TRENDS.copy() * 0.0 + 50.0
    trends.loc[spike_day] = 5000.0
    sources = fake_sources(trends=lambda *a: trends)
    z = build_dataset(DatasetConfig("AAA", use_trends=True), sources).features[
        "search_interest_zscore"
    ]
    first_visible = z[z.abs() > 1].index.min()
    assert first_visible >= spike_day + TRENDS_AVAILABILITY_LAG - pd.Timedelta(days=1)


def test_the_insider_fetch_starts_early_enough_for_a_full_window() -> None:
    seen: list[pd.Timestamp] = []

    def insider(ticker, after):  # type: ignore[no-untyped-def]
        seen.append(after)
        return INSIDER

    build_dataset(
        DatasetConfig("AAA", start="2022-03-01", use_insider_flow=True),
        fake_sources(insider_flow=insider),
    )
    assert seen[0] == pd.Timestamp("2022-03-01", tz="UTC") - pd.Timedelta(days=120)


def test_importing_the_builder_pulls_in_no_heavy_dependencies(assert_no_heavy_imports) -> None:  # type: ignore[no-untyped-def]
    import src.prediction.dataset  # noqa: F401

    assert_no_heavy_imports()
