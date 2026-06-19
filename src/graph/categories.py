"""
Canonical node category definitions for the information diffusion graph.
All other modules import from here; never hardcode category strings elsewhere.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

NODE_SEC_CORP = "sec_corp"
NODE_INSTITUTIONAL = "institutional"
NODE_FIN_PRESS = "fin_press"
NODE_INFORMED_RETAIL = "informed_retail"
NODE_UNINFORMED_RETAIL = "uninformed_retail"

ALL_NODES = [
    NODE_SEC_CORP,
    NODE_INSTITUTIONAL,
    NODE_FIN_PRESS,
    NODE_INFORMED_RETAIL,
    NODE_UNINFORMED_RETAIL,
]


@dataclass
class EdgeConfig:
    source: str
    target: str
    prob: float  # transmission probability
    lag_mean: float  # mean lag (hours), log-normal
    lag_std: float  # std of lag (hours), log-normal


# Default edge priors — update via scripts/calibrate_graph.py
DEFAULT_EDGES: list[EdgeConfig] = [
    EdgeConfig(NODE_SEC_CORP, NODE_INSTITUTIONAL, prob=0.95, lag_mean=0.25, lag_std=0.10),
    EdgeConfig(NODE_SEC_CORP, NODE_FIN_PRESS, prob=0.80, lag_mean=3.00, lag_std=1.50),
    EdgeConfig(NODE_INSTITUTIONAL, NODE_FIN_PRESS, prob=0.70, lag_mean=4.00, lag_std=2.00),
    EdgeConfig(NODE_FIN_PRESS, NODE_INFORMED_RETAIL, prob=0.60, lag_mean=6.00, lag_std=3.00),
    EdgeConfig(
        NODE_INFORMED_RETAIL, NODE_UNINFORMED_RETAIL, prob=0.40, lag_mean=12.00, lag_std=6.00
    ),
]


def load_calibrated_edges(path: str | Path) -> list[EdgeConfig]:
    """
    Return the ``DEFAULT_EDGES`` topology with ``prob`` / ``lag_mean`` /
    ``lag_std`` overridden from a ``calibrated_edges.json`` (written by
    ``scripts/calibrate_graph.py``).

    The JSON is keyed ``"src__tgt"`` with optional ``prob`` / ``lag_mean`` /
    ``lag_std`` fields. Edges absent from the file keep their default priors;
    keys not in the default topology are ignored (the graph topology is fixed
    by ``DEFAULT_EDGES``). The result is suitable for
    ``diffusion.build_graph(edges=...)`` / ``estimate_lag(edges=...)``.
    """
    with open(path, "r", encoding="utf-8") as fh:
        calibrated = json.load(fh)

    edges: list[EdgeConfig] = []
    for e in DEFAULT_EDGES:
        entry = calibrated.get(f"{e.source}__{e.target}", {})
        edges.append(
            EdgeConfig(
                source=e.source,
                target=e.target,
                prob=float(entry.get("prob", e.prob)),
                lag_mean=float(entry.get("lag_mean", e.lag_mean)),
                lag_std=float(entry.get("lag_std", e.lag_std)),
            )
        )
    return edges
