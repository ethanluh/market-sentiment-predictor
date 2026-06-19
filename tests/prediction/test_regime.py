"""Tests for prediction.regime."""

from __future__ import annotations

import numpy as np
import pytest

from src.prediction.regime import RegimeClassifier


def _three_cluster_vix(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    low = rng.normal(13, 0.5, 50)
    med = rng.normal(20, 0.5, 50)
    high = rng.normal(35, 1.0, 50)
    return np.concatenate([low, med, high])


class TestRegimeClassifier:
    def test_labels_monotonic_in_vol(self):
        clf = RegimeClassifier(k=3).fit(_three_cluster_vix())
        # Lowest VIX -> label 0, highest -> label 2.
        assert clf.predict(np.array([12.0]))[0] == 0
        assert clf.predict(np.array([40.0]))[0] == 2

    def test_label_names_length(self):
        clf = RegimeClassifier(k=3)
        assert len(clf.label_names()) == 3

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            RegimeClassifier().predict(np.array([20.0]))

    def test_too_few_unique_values_raises(self):
        with pytest.raises(ValueError):
            RegimeClassifier(k=3).fit(np.array([20.0, 20.0, 20.0, 20.0]))

    def test_save_load_roundtrip(self, tmp_path):
        clf = RegimeClassifier(k=3).fit(_three_cluster_vix())
        path = str(tmp_path / "regime.joblib")
        clf.save(path)
        loaded = RegimeClassifier.load(path)
        probe = np.array([12.0, 20.0, 40.0])
        assert np.array_equal(clf.predict(probe), loaded.predict(probe))
