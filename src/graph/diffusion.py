"""
Heat kernel diffusion on the information flow graph.

Signal propagation: ds/dt = -alpha * L * s(t)
Solution:          s(t)  = expm(-alpha * L * t) @ s(0)

estimate_lag() returns the time (hours) at which the institutional node
reaches 50% of its peak concentration given a signal origin.
"""

from __future__ import annotations

import numpy as np
import networkx as nx
from scipy.linalg import expm

from src.graph.categories import (
    ALL_NODES,
    NODE_INSTITUTIONAL,
    DEFAULT_EDGES,
    EdgeConfig,
)


def build_graph(edges: list[EdgeConfig] | None = None) -> nx.DiGraph:
    """Build the directed information flow graph."""
    if edges is None:
        edges = DEFAULT_EDGES
    G = nx.DiGraph()
    G.add_nodes_from(ALL_NODES)
    for e in edges:
        G.add_edge(e.source, e.target, prob=e.prob, lag_mean=e.lag_mean, lag_std=e.lag_std)
    return G


def adjacency_matrix(G: nx.DiGraph) -> tuple[np.ndarray, list[str]]:
    """Return (A, node_order) where A[i,j] = prob weight on edge i→j."""
    nodes = list(G.nodes)
    n = len(nodes)
    idx = {node: i for i, node in enumerate(nodes)}
    A = np.zeros((n, n))
    for u, v, data in G.edges(data=True):
        A[idx[u], idx[v]] = data.get("prob", 1.0)
    return A, nodes


def laplacian(A: np.ndarray) -> np.ndarray:
    """Out-degree Laplacian: L = D - A."""
    D = np.diag(A.sum(axis=1))
    return D - A


def diffuse(L: np.ndarray, s0: np.ndarray, t: float, alpha: float = 0.5) -> np.ndarray:
    """Return signal concentration vector at time t."""
    return expm(-alpha * L * t) @ s0


def estimate_lag(
    origin_node: str,
    alpha: float = 0.5,
    target_node: str = NODE_INSTITUTIONAL,
    t_max: float = 48.0,
    n_steps: int = 500,
    edges: list[EdgeConfig] | None = None,
) -> float:
    """
    Estimate hours until target_node reaches 50% of its peak concentration
    after a signal originates at origin_node.

    Returns float hours. Returns t_max if target never reaches threshold.
    """
    G = build_graph(edges)
    A, nodes = adjacency_matrix(G)
    L = laplacian(A)
    node_idx = {n: i for i, n in enumerate(nodes)}

    s0 = np.zeros(len(nodes))
    s0[node_idx[origin_node]] = 1.0

    times = np.linspace(0, t_max, n_steps)
    target_i = node_idx[target_node]

    concentrations = np.array([diffuse(L, s0, t, alpha)[target_i] for t in times])
    peak = concentrations.max()
    if peak == 0:
        return t_max

    threshold = 0.5 * peak
    crossings = np.where(concentrations >= threshold)[0]
    return float(times[crossings[0]]) if len(crossings) > 0 else t_max
