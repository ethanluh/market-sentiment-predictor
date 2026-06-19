"""
Calibrate edge transmission probabilities and lag distributions from historical
event data.

For each edge it estimates:
- ``prob``: fraction of events where the downstream node reacted (non-null
  reaction time). This is what the diffusion adjacency uses, so it feeds back
  into the model via ``categories.load_calibrated_edges``.
- ``lag_mean`` / ``lag_std``: log-normal fit of the reaction delay (hours).

Usage:
    python scripts/calibrate_graph.py --events-file data/raw/events.csv
    # then, in code:
    #   from src.graph.categories import load_calibrated_edges
    #   edges = load_calibrated_edges("data/processed/calibrated_edges.json")
    #   estimate_lag(origin, edges=edges)

Expected CSV columns:
    event_id, ticker, origin_node, institutional_reaction_hours,
    fin_press_reaction_hours, informed_retail_reaction_hours, uninformed_retail_reaction_hours
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import lognorm


def fit_lognormal(values: np.ndarray) -> tuple[float, float]:
    """Return (mean, std) of best-fit log-normal to observed lag values (hours)."""
    values = values[values > 0]
    if len(values) < 5:
        raise ValueError("Not enough data points to fit log-normal (need >= 5)")
    shape, loc, scale = lognorm.fit(values, floc=0)
    mean = np.exp(np.log(scale) + shape**2 / 2)
    std = mean * np.sqrt(np.exp(shape**2) - 1)
    return float(mean), float(std)


def calibrate(events_file: Path) -> dict:
    df = pd.read_csv(events_file)
    edges = {}

    edge_columns = {
        ("sec_corp", "institutional"): "institutional_reaction_hours",
        ("sec_corp", "fin_press"): "fin_press_reaction_hours",
        ("fin_press", "informed_retail"): "informed_retail_reaction_hours",
        ("informed_retail", "uninformed_retail"): "uninformed_retail_reaction_hours",
    }

    total = len(df)
    for (src, tgt), col in edge_columns.items():
        if col not in df.columns:
            print(f"Warning: column {col} not found, skipping edge {src} → {tgt}")
            continue
        reactions = df[col].dropna()
        # Transmission probability: fraction of events where the downstream node reacted.
        prob = float(len(reactions) / total) if total else 0.0
        entry: dict = {"prob": prob, "n": int(len(reactions))}
        try:
            mean, std = fit_lognormal(reactions.to_numpy())
            entry["lag_mean"] = mean
            entry["lag_std"] = std
            print(
                f"{src} → {tgt}: prob={prob:.2f}, mean={mean:.2f}h, std={std:.2f}h (n={len(reactions)})"
            )
        except ValueError as e:
            print(f"{src} → {tgt}: prob={prob:.2f} (lag not fit: {e})")
        edges[f"{src}__{tgt}"] = entry

    return edges


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--events-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/processed/calibrated_edges.json"))
    args = parser.parse_args()

    results = calibrate(args.events_file)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nCalibrated edges written to {args.output}")
