import networkx as nx
import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from src.graph.sector import build_sector_graph, sector_proximity
from src.prediction.regime import RegimeClassifier


def _returns(seed: int, n_tickers: int, n_rows: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    common = rng.normal(size=(n_rows, 1))
    data = 2.0 * common + rng.normal(size=(n_rows, n_tickers))  # pairwise corr about 0.8
    return pd.DataFrame(data, columns=[f"T{i}" for i in range(n_tickers)])


seeds = st.integers(0, 2**32 - 1)


@given(seeds, st.integers(2, 6), st.floats(0.1, 0.9))
def test_graph_edges_respect_the_threshold(seed: int, n: int, min_corr: float) -> None:
    g = build_sector_graph(_returns(seed, n), window=60, min_corr=min_corr)
    assume(g.number_of_edges() > 0)
    for _, _, d in g.edges(data=True):
        assert min_corr <= d["weight"] <= 1.0 + 1e-9


@given(seeds, st.integers(2, 6), st.floats(0.1, 0.9))
def test_proximity_is_finite_and_non_negative(seed: int, n: int, min_corr: float) -> None:
    g = build_sector_graph(_returns(seed, n), window=60, min_corr=min_corr)
    assume(g.degree("T0") > 0)
    value = sector_proximity(g, "T0")
    assert np.isfinite(value) and value > 0.0


@given(seeds, st.integers(2, 6))
def test_a_target_with_no_edges_has_zero_proximity(seed: int, n: int) -> None:
    g = build_sector_graph(_returns(seed, n), window=60, min_corr=2.0)  # no |r| reaches 2
    assert g.number_of_edges() == 0 and sector_proximity(g, "T0") == 0.0


@given(
    st.lists(st.floats(8.0, 80.0), min_size=30, max_size=120, unique=True),
)
def test_regime_labels_never_decrease_as_vix_rises(vix: list[float]) -> None:
    x = np.array(vix)
    assume(np.ptp(x) > 1.0)  # k-means needs genuinely separated levels
    clf = RegimeClassifier().fit(x)
    labels = clf.predict(np.sort(x))
    assert set(labels) <= {0, 1, 2}
    assert (np.diff(labels) >= 0).all()


def test_a_single_edge_gives_proximity_equal_to_its_weight() -> None:
    g = nx.Graph()
    g.add_edge("T0", "T1", weight=0.5)
    assert sector_proximity(g, "T0") == pytest.approx(0.5)


def test_proximity_is_the_inverse_of_mean_distance_to_peers() -> None:
    g = nx.Graph()
    g.add_edge("T0", "T1", weight=0.5)  # distance 2
    g.add_edge("T0", "T2", weight=0.25)  # distance 4
    assert sector_proximity(g, "T0") == pytest.approx(1.0 / np.mean([2.0, 4.0]))


@given(
    st.lists(st.floats(0.1, 1.0), min_size=3, max_size=3),
    st.floats(1.0, 3.0),
)
def test_stronger_correlations_never_reduce_proximity(weights: list[float], k: float) -> None:
    def proximity(ws: list[float]) -> float:
        g = nx.Graph()
        g.add_edge("T0", "T1", weight=ws[0])
        g.add_edge("T0", "T2", weight=ws[1])
        g.add_edge("T1", "T2", weight=ws[2])
        return sector_proximity(g, "T0")

    stronger = [min(1.0, w * k) for w in weights]
    assert proximity(stronger) >= proximity(weights) - 1e-9
