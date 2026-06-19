"""
Canonical node category definitions for the information diffusion graph.
All other modules import from here; never hardcode category strings elsewhere.
"""

from dataclasses import dataclass, field

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
