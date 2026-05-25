"""
Calibrate edge lag distributions from historical event data.

Usage:
    python scripts/calibrate_graph.py --events-file data/raw/events.csv

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

    for (src, tgt), col in edge_columns.items():
        if col not in df.columns:
            print(f"Warning: column {col} not found, skipping edge {src} → {tgt}")
            continue
        values = df[col].dropna().values
        try:
            mean, std = fit_lognormal(values)
            edges[f"{src}__{tgt}"] = {"lag_mean": mean, "lag_std": std, "n": len(values)}
            print(f"{src} → {tgt}: mean={mean:.2f}h, std={std:.2f}h (n={len(values)})")
        except ValueError as e:
            print(f"Skipping {src} → {tgt}: {e}")

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
