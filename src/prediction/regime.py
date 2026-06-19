"""
Volatility-regime classification from VIX.

A k-means model (k=3 by default) partitions VIX levels into low / medium / high
volatility regimes. Cluster ids are remapped so that the emitted label is
monotonic in volatility (0 = lowest-vol regime, k-1 = highest), since sentiment
predictiveness varies with the regime and downstream models rely on the
ordering being stable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_DEFAULT_NAMES = {3: ["low_vol", "med_vol", "high_vol"]}


@dataclass
class RegimeClassifier:
    k: int = 3
    _km: object | None = field(default=None, repr=False)
    _order: np.ndarray | None = field(default=None, repr=False)

    @staticmethod
    def _as_2d(vix: np.ndarray | "object") -> np.ndarray:
        arr = np.asarray(vix, dtype=float).reshape(-1, 1)
        return arr

    def fit(self, vix: np.ndarray | "object") -> "RegimeClassifier":
        """Fit k-means on VIX values and build the vol-monotonic label map."""
        from sklearn.cluster import KMeans

        x = self._as_2d(vix)
        n_unique = len(np.unique(x))
        if n_unique < self.k:
            raise ValueError(f"Need at least k={self.k} distinct VIX values to fit; got {n_unique}")

        km = KMeans(n_clusters=self.k, n_init=10, random_state=0)
        km.fit(x)
        # rank[raw_cluster] = position of that cluster's centroid in ascending
        # order, so label 0 is the lowest-volatility cluster.
        centroids = km.cluster_centers_.ravel()
        ascending = np.argsort(centroids)
        order = np.empty(self.k, dtype=int)
        order[ascending] = np.arange(self.k)

        self._km = km
        self._order = order
        return self

    def predict(self, vix: np.ndarray | "object") -> np.ndarray:
        """Return vol-monotonic regime labels (0..k-1) for each VIX value."""
        if self._km is None or self._order is None:
            raise RuntimeError("RegimeClassifier is not fitted; call fit() first")
        raw = self._km.predict(self._as_2d(vix))  # type: ignore[attr-defined]
        return self._order[raw]

    def label_names(self) -> list[str]:
        """Human-readable names ordered low→high volatility."""
        if self.k in _DEFAULT_NAMES:
            return list(_DEFAULT_NAMES[self.k])
        return [f"regime_{i}" for i in range(self.k)]

    def save(self, path: str) -> None:
        import joblib

        joblib.dump({"k": self.k, "km": self._km, "order": self._order}, path)

    @classmethod
    def load(cls, path: str) -> "RegimeClassifier":
        import joblib

        state = joblib.load(path)
        obj = cls(k=state["k"])
        obj._km = state["km"]
        obj._order = state["order"]
        return obj
