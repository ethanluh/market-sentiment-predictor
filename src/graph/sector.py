"""
Sector correlation graph.

Nodes: tickers in the same GICS sector as the target.
Edges: rolling Pearson correlation of log-returns, thresholded at |r| >= min_corr.
Feature: sector_proximity = 1 / mean_shortest_path_distance(peers → target).
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd


def build_sector_graph(
    returns: pd.DataFrame,
    window: int = 60,
    min_corr: float = 0.5,
) -> nx.Graph:
    """
    Build undirected correlation graph from a DataFrame of log-returns.

    Parameters
    ----------
    returns : pd.DataFrame
        Columns are ticker symbols, index is DatetimeIndex (UTC).
        Values should be log-returns (already computed).
    window : int
        Rolling window in trading days for correlation.
    min_corr : float
        Minimum absolute correlation to include an edge.
    """
    corr = returns.tail(window).corr()
    G = nx.Graph()
    tickers = corr.columns.tolist()
    G.add_nodes_from(tickers)
    for i, t1 in enumerate(tickers):
        for t2 in tickers[i + 1 :]:
            w = corr.loc[t1, t2]
            if not np.isnan(w) and abs(w) >= min_corr:
                G.add_edge(t1, t2, weight=abs(w))
    return G


def sector_proximity(G: nx.Graph, target: str) -> float:
    """
    Compute sector_proximity feature for target ticker.
    Returns 1 / mean shortest-path distance from all connected peers to target.
    Returns 0.0 if target is isolated.
    """
    if target not in G or G.degree(target) == 0:
        return 0.0

    # Use 1/weight as distance (higher correlation = shorter distance)
    for u, v, d in G.edges(data=True):
        G[u][v]["distance"] = 1.0 / (d["weight"] + 1e-9)

    try:
        # Undirected graph: shortest weighted distance from `target` to every
        # reachable peer (Dijkstra over the 1/correlation distances above).
        lengths = nx.single_source_dijkstra_path_length(G, target, weight="distance")
        distances = [d for node, d in lengths.items() if node != target and d > 0]
        if not distances:
            return 0.0
        return 1.0 / np.mean(distances)
    except nx.NetworkXError:
        return 0.0
