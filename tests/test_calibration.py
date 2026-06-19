"""Tests for the graph calibration loop (calibrate_graph -> load_calibrated_edges)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.graph.categories import DEFAULT_EDGES, load_calibrated_edges
from src.graph.diffusion import estimate_lag

_REPO = Path(__file__).resolve().parents[1]


def _load_calibrate_script():
    spec = importlib.util.spec_from_file_location(
        "calibrate_graph", _REPO / "scripts" / "calibrate_graph.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _events_csv(tmp_path, seed=0):
    rng = np.random.default_rng(seed)
    n = 40
    df = pd.DataFrame(
        {
            "event_id": range(n),
            "institutional_reaction_hours": rng.lognormal(0.0, 0.5, n),
            # fin_press reacts only ~half the time -> prob < 1
            "fin_press_reaction_hours": [
                v if i % 2 == 0 else np.nan for i, v in enumerate(rng.lognormal(1.0, 0.5, n))
            ],
        }
    )
    path = tmp_path / "events.csv"
    df.to_csv(path, index=False)
    return path


class TestCalibrateProb:
    def test_prob_reflects_reaction_fraction(self, tmp_path):
        cg = _load_calibrate_script()
        edges = cg.calibrate(_events_csv(tmp_path))
        # institutional reacted on every event; fin_press on ~half.
        assert edges["sec_corp__institutional"]["prob"] == 1.0
        assert 0.3 < edges["sec_corp__fin_press"]["prob"] < 0.7
        assert "lag_mean" in edges["sec_corp__institutional"]


class TestLoadCalibratedEdges:
    def test_overrides_prob_and_preserves_topology(self, tmp_path):
        payload = {"sec_corp__institutional": {"prob": 0.5, "lag_mean": 1.0, "lag_std": 0.2}}
        path = tmp_path / "calibrated_edges.json"
        path.write_text(json.dumps(payload))

        edges = load_calibrated_edges(path)
        # Same topology as defaults.
        assert [(e.source, e.target) for e in edges] == [
            (e.source, e.target) for e in DEFAULT_EDGES
        ]
        by_key = {(e.source, e.target): e for e in edges}
        # Overridden edge.
        assert by_key[("sec_corp", "institutional")].prob == 0.5
        # Edge absent from the file keeps its default prob.
        default_fp = next(
            e for e in DEFAULT_EDGES if (e.source, e.target) == ("sec_corp", "fin_press")
        )
        assert by_key[("sec_corp", "fin_press")].prob == default_fp.prob

    def test_estimate_lag_runs_with_calibrated_edges(self, tmp_path):
        payload = {"sec_corp__institutional": {"prob": 0.9}}
        path = tmp_path / "calibrated_edges.json"
        path.write_text(json.dumps(payload))
        lag = estimate_lag("sec_corp", edges=load_calibrated_edges(path))
        assert isinstance(lag, float)
        assert lag >= 0.0
