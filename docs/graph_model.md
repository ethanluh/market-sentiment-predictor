# Graph Model

## Node Categories

Defined in `src/graph/categories.py`. Five categories:

| ID | Category | Description |
|---|---|---|
| `sec_corp` | SEC / Corporate | 8-K, press releases, earnings calls. Primary signal origin. |
| `institutional` | Institutional | Hedge funds, quant desks, market makers. React in minutes via EDGAR API. |
| `fin_press` | Financial Press | Reuters, Bloomberg, WSJ, analyst Substacks. 1–6h lag. |
| `informed_retail` | Informed Retail | Options flow traders, active finance Reddit. Hours to 1d lag. |
| `uninformed_retail` | Uninformed Retail | Mainstream social, Google Trends. Weak signal except meme stocks. |

## Directed Information Flow Graph

G = (V, E) where edges (u → v) encode empirically estimated:
- `prob`: transmission probability (fraction of historical events where u activation preceded v activation)
- `lag_mean`, `lag_std`: log-normal parameters for delay distribution (hours)

Default priors (update with historical calibration):

```python
EDGES = [
    ("sec_corp",       "institutional",    prob=0.95, lag_mean=0.25,  lag_std=0.1),
    ("sec_corp",       "fin_press",        prob=0.80, lag_mean=3.0,   lag_std=1.5),
    ("institutional",  "fin_press",        prob=0.70, lag_mean=4.0,   lag_std=2.0),
    ("fin_press",      "informed_retail",  prob=0.60, lag_mean=6.0,   lag_std=3.0),
    ("informed_retail","uninformed_retail",prob=0.40, lag_mean=12.0,  lag_std=6.0),
]
```

## Heat Kernel Diffusion

Signal propagation modeled as continuous-time diffusion on G:

    ds/dt = -α · L · s(t)
    s(t)  = exp(-α · L · t) · s(0)

where:
- L = D - A is the graph Laplacian (D = out-degree matrix, A = weighted adjacency)
- s(0) is a one-hot vector at the originating node category
- α ∈ (0, 1] is the diffusion rate (tunable hyperparameter)

**estimated_lag** for a given signal: smallest t such that s_institutional(t) ≥ 0.5 · max_t(s_institutional(t))

Computed in `src/graph/diffusion.py::estimate_lag(origin_node, alpha)`.

## Sector Correlation Graph

Secondary graph G_sector = (V_sector, E_sector):
- Nodes: all tickers in the same GICS sector as the target
- Edges: rolling 60-day Pearson correlation of log-returns, thresholded at |r| ≥ 0.5
- Edge weight = |r|

**sector_proximity** feature = 1 / mean_shortest_path_distance(peers → target), using edge weight as inverse distance.

Recomputed weekly. Stored in SQLite.

## Calibration

To update edge transmission probabilities and lag distributions from data:
1. Identify events with a known `sec_corp` origination (8-K filing timestamps)
2. Measure time-to-first-reaction for each downstream category (first article, options flow spike, Reddit mention)
3. Per edge, estimate `prob` (fraction of events where the downstream node reacted) and fit a log-normal MLE for the lag

Script: `scripts/calibrate_graph.py` → writes `data/processed/calibrated_edges.json`.

Feed it back into the model (the diffusion adjacency keys off `prob`):

```python
from src.graph.categories import load_calibrated_edges
from src.graph.diffusion import estimate_lag

edges = load_calibrated_edges("data/processed/calibrated_edges.json")
lag = estimate_lag("sec_corp", edges=edges)   # uses calibrated probabilities
```

`load_calibrated_edges` overrides the `DEFAULT_EDGES` priors where the file has
data and keeps defaults otherwise, so the topology is preserved.
