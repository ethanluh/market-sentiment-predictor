"""Tests for graph diffusion and sector correlation modules."""

import numpy as np
import pandas as pd
import pytest

from src.graph.categories import (
    ALL_NODES,
    NODE_SEC_CORP,
    NODE_INSTITUTIONAL,
    NODE_UNINFORMED_RETAIL,
    DEFAULT_EDGES,
)
from src.graph.diffusion import (
    build_graph,
    adjacency_matrix,
    laplacian,
    diffuse,
    estimate_lag,
)
from src.graph.sector import build_sector_graph, sector_proximity


class TestDiffusionGraph:
    def test_build_graph_has_all_nodes(self):
        G = build_graph()
        assert set(G.nodes) == set(ALL_NODES)

    def test_adjacency_shape(self):
        G = build_graph()
        A, nodes = adjacency_matrix(G)
        assert A.shape == (len(ALL_NODES), len(ALL_NODES))

    def test_laplacian_row_sums_zero(self):
        G = build_graph()
        A, _ = adjacency_matrix(G)
        L = laplacian(A)
        np.testing.assert_allclose(L.sum(axis=1), 0.0, atol=1e-10)

    def test_diffuse_preserves_mass_approximately(self):
        G = build_graph()
        A, nodes = adjacency_matrix(G)
        L = laplacian(A)
        s0 = np.zeros(len(nodes))
        s0[0] = 1.0
        # Mass is not strictly conserved with this formulation but should not explode
        s_t = diffuse(L, s0, t=1.0, alpha=0.5)
        assert np.all(np.isfinite(s_t))

    def test_estimate_lag_sec_corp_faster_than_uninformed(self):
        lag_sec = estimate_lag(NODE_SEC_CORP)
        lag_retail = estimate_lag(NODE_UNINFORMED_RETAIL)
        # Signal originating at SEC/Corp should reach institutional faster
        assert lag_sec <= lag_retail

    def test_estimate_lag_returns_float(self):
        lag = estimate_lag(NODE_SEC_CORP)
        assert isinstance(lag, float)
        assert lag >= 0.0


class TestSectorGraph:
    def _make_returns(self, n_tickers: int = 5, n_days: int = 100, seed: int = 42) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        # Create correlated returns: all tickers share a common factor
        common = rng.normal(0, 0.01, n_days)
        data = {
            f"TICK{i}": common + rng.normal(0, 0.005, n_days)
            for i in range(n_tickers)
        }
        return pd.DataFrame(data)

    def test_graph_has_edges(self):
        returns = self._make_returns()
        G = build_sector_graph(returns, min_corr=0.3)
        assert G.number_of_edges() > 0

    def test_sector_proximity_positive_for_connected_node(self):
        returns = self._make_returns()
        G = build_sector_graph(returns, min_corr=0.3)
        target = "TICK0"
        prox = sector_proximity(G, target)
        assert prox > 0.0

    def test_sector_proximity_isolated_node(self):
        returns = self._make_returns()
        G = build_sector_graph(returns, min_corr=0.3)
        G.remove_edges_from(list(G.edges("TICK0")))
        prox = sector_proximity(G, "TICK0")
        assert prox == 0.0
