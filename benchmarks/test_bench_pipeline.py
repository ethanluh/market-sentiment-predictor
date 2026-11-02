"""
Performance benchmarks. Run with ``pytest benchmarks/ --benchmark-only``; they are not
part of the default ``pytest tests/`` run because timings are machine-specific.
"""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src.graph.diffusion import estimate_lag
from src.graph.sector import build_sector_graph, sector_proximity
from src.prediction.features import FEATURE_NAMES, build_feature_matrix
from src.prediction.model import QuantileReturnModel
from src.sentiment.aggregation import ScoredArticle, aggregate


def _market(n: int, n_peers: int, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n, tz="UTC")
    rets = pd.DataFrame(
        0.6 * rng.normal(size=(n, 1)) + rng.normal(size=(n, n_peers + 1)),
        index=idx,
        columns=["T0", *[f"P{i}" for i in range(n_peers)]],
    )
    prices = pd.DataFrame(
        {"log_return": rets["T0"], "volume": rng.lognormal(10, 0.3, n)}, index=idx
    )
    return prices, rets


@pytest.mark.parametrize("n_rows", [100, 250])
def test_build_feature_matrix(benchmark: pytest.fixture, n_rows: int) -> None:  # type: ignore[valid-type]
    prices, rets = _market(n_rows + 40, n_peers=5)
    index = prices.index[40:]
    benchmark.pedantic(
        lambda: build_feature_matrix("T0", index, lambda as_of: [], prices, rets),
        rounds=3,
        iterations=1,
    )


def test_estimate_lag_uncached(benchmark: pytest.fixture) -> None:  # type: ignore[valid-type]
    benchmark(lambda: estimate_lag("sec_corp", alpha=0.5))


def test_sector_graph_and_proximity_20_tickers(benchmark: pytest.fixture) -> None:  # type: ignore[valid-type]
    _, rets = _market(120, n_peers=19)
    benchmark(lambda: sector_proximity(build_sector_graph(rets, window=60, min_corr=0.3), "T0"))


@pytest.mark.parametrize("backend", ["gbr", "linear"])
def test_model_fit_and_predict(benchmark: pytest.fixture, backend: str) -> None:  # type: ignore[valid-type]
    rng = np.random.default_rng(0)
    x = pd.DataFrame(rng.normal(size=(500, len(FEATURE_NAMES))), columns=FEATURE_NAMES)
    y = pd.Series(rng.normal(0, 0.01, 500))

    def run() -> None:
        model = QuantileReturnModel(horizons=("1d",), backend=backend).fit(x.iloc[:400], {"1d": y})
        model.predict(x.iloc[400:])

    benchmark.pedantic(run, rounds=3, iterations=1)


def test_aggregate_1000_articles(benchmark: pytest.fixture) -> None:  # type: ignore[valid-type]
    as_of = datetime(2024, 6, 1, tzinfo=timezone.utc)
    arts = [ScoredArticle(0.1, "reuters", as_of - timedelta(hours=h % 200)) for h in range(1000)]
    benchmark(lambda: aggregate(arts, as_of=as_of))
